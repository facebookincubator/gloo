/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gloo/transport/tcp/loop.h>

#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include <array>

#include <gloo/common/error.h>
#include <gloo/common/logging.h>

namespace gloo {
namespace transport {
namespace tcp {

Deferrables::Deferrables() {
  int fds[2];
  auto rv = pipe2(fds, O_NONBLOCK);
  GLOO_ENFORCE_NE(rv, -1, "pipe: ", strerror(errno));
  rfd_ = fds[0];
  wfd_ = fds[1];
}

Deferrables::~Deferrables() {
  close(rfd_);
  close(wfd_);
}

void Deferrables::defer(function_t fn) {
  std::lock_guard<std::mutex> guard(mutex_);
  functions_.push_back(std::move(fn));

  // Write byte to pipe to make epoll(2) wake up.
  if (!triggered_) {
    for (;;) {
      char byte = 0;
      auto rv = write(wfd_, &byte, sizeof(byte));
      if (rv == -1 && errno == EINTR) {
        continue;
      }
      GLOO_ENFORCE_NE(rv, -1, "write: ", strerror(errno));
      break;
    }
    triggered_ = true;
  }
}

void Deferrables::handleEvents(int events) {
  decltype(functions_) localFunctions;

  {
    std::lock_guard<std::mutex> guard(mutex_);
    std::swap(localFunctions, functions_);

    // Read byte from pipe to drain it.
    for (;;) {
      char byte = 0;
      auto rv = read(rfd_, &byte, sizeof(byte));
      if (rv == -1 && errno == EINTR) {
        continue;
      }
      GLOO_ENFORCE_NE(rv, -1, "read: ", strerror(errno));
      break;
    }
    triggered_ = false;
  }

  // Execute deferred functions.
  for (auto fn : localFunctions) {
    fn();
  }
}

Loop::Loop() : fd_(-1), done_(false) {
  fd_ = epoll_create(1);
  GLOO_ENFORCE_NE(fd_, -1, "epoll_create: ", strerror(errno));
  loop_.reset(new std::thread(&Loop::run, this));
  registerDescriptor(deferrables_.rfd_, EPOLLIN, &deferrables_);
}

Loop::~Loop() {
  if (loop_) {
    done_ = true;
    loop_->join();
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

void Loop::registerDescriptor(int fd, int events, Handler* h) {
  struct epoll_event ev;
  ev.events = events;
  ev.data.ptr = h;

  auto rv = epoll_ctl(fd_, EPOLL_CTL_ADD, fd, &ev);
  if (rv == -1 && errno == EEXIST) {
    rv = epoll_ctl(fd_, EPOLL_CTL_MOD, fd, &ev);
  }
  GLOO_ENFORCE_NE(rv, -1, "epoll_ctl: ", strerror(errno));
}

void Loop::unregisterDescriptor(int fd) {
  auto rv = epoll_ctl(fd_, EPOLL_CTL_DEL, fd, nullptr);
  GLOO_ENFORCE_NE(rv, -1, "epoll_ctl: ", strerror(errno));

  // Wait for loop to tick before returning, to make sure the handler
  // for this fd is not called once this function returns.
  if (std::this_thread::get_id() != loop_->get_id()) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock);
  }
}

void Loop::defer(std::function<void()> fn) {
  deferrables_.defer(std::move(fn));
}

void Loop::run() {
  std::array<struct epoll_event, capacity_> events;
  int nfds;

  while (!done_) {
    // Wakeup everyone waiting for a loop tick to finish.
    cv_.notify_all();

    // Wait for something to happen
    nfds = epoll_wait(fd_, events.data(), events.size(), 10);
    if (nfds == 0) {
      continue;
    }
    if (nfds == -1 && errno == EINTR) {
      continue;
    }

    GLOO_ENFORCE_NE(nfds, -1);

    for (int i = 0; i < nfds; i++) {
      Handler* h = reinterpret_cast<Handler*>(events[i].data.ptr);
      h->handleEvents(events[i].events);
    }
  }
}

} // namespace tcp
} // namespace transport
} // namespace gloo
