/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gloo/transport/tcp/loop.h>

#include <string.h>
#include <unistd.h>

#include <array>

#include <gloo/common/error.h>
#include <gloo/common/logging.h>

#if defined(__SANITIZE_THREAD__)
#define TSAN_ENABLED
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define TSAN_ENABLED
#endif
#endif

#ifdef TSAN_ENABLED
#define TSAN_ANNOTATE_HAPPENS_BEFORE(addr) \
    AnnotateHappensBefore(__FILE__, __LINE__, (void*)(addr))
#define TSAN_ANNOTATE_HAPPENS_AFTER(addr) \
    AnnotateHappensAfter(__FILE__, __LINE__, (void*)(addr))
extern "C" void AnnotateHappensBefore(const char* f, int l, void* addr);
extern "C" void AnnotateHappensAfter(const char* f, int l, void* addr);
#else
#define TSAN_ANNOTATE_HAPPENS_BEFORE(addr)
#define TSAN_ANNOTATE_HAPPENS_AFTER(addr)
#endif

namespace gloo {
namespace transport {
namespace tcp {

Loop::Loop() {
  fd_ = epoll_create(1);
  GLOO_ENFORCE_NE(fd_, -1, "epoll_create: ", strerror(errno));
  loop_.reset(new std::thread(&Loop::run, this));
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

void Loop::unregisterDescriptor(int fd, Handler* h) {
  auto rv = epoll_ctl(fd_, EPOLL_CTL_DEL, fd, nullptr);
  GLOO_ENFORCE_NE(rv, -1, "epoll_ctl: ", strerror(errno));

  // Wait for loop to tick before returning, to make sure the handler
  // for this fd is not called once this function returns.
  if (std::this_thread::get_id() != loop_->get_id()) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock);
    TSAN_ANNOTATE_HAPPENS_AFTER(h);
  }
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
      TSAN_ANNOTATE_HAPPENS_BEFORE(h);
    }
  }
}

} // namespace tcp
} // namespace transport
} // namespace gloo
