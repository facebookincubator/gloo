/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "floo/transport/tcp/pair.h"

#include <sstream>

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "floo/common/logging.h"
#include "floo/transport/tcp/buffer.h"

namespace floo {
namespace transport {
namespace tcp {

Pair::Pair(const std::shared_ptr<Device>& dev)
    : dev_(dev), fd_(-1), sendBufferSize_(0), state_(INITIALIZING) {
  memset(&rx_, 0, sizeof(rx_));
  memset(&tx_, 0, sizeof(tx_));
  listen();
}

Pair::~Pair() {
  // Needs lock so that this doesn't race with read/write of the
  // underlying file descriptor on the device thread.
  std::lock_guard<std::mutex> lock(m_);
  changeState(CLOSED);
}

const Address& Pair::address() const {
  return self_;
}

void Pair::connect(const std::vector<char>& bytes) {
  auto peer = Address(bytes);
  connect(peer);
}

void Pair::listen() {
  std::lock_guard<std::mutex> lock(m_);
  int rv;

  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = dev_->attr_.ai_family;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo* result;
  rv = getaddrinfo(dev_->attr_.hostname.data(), nullptr, &hints, &result);
  FLOO_ENFORCE_NE(rv, -1);
  for (auto rp = result; rp != nullptr; rp = rp->ai_next) {
    auto fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }

    rv = bind(fd, rp->ai_addr, rp->ai_addrlen);
    if (rv == -1) {
      close(fd);
      continue;
    }

    // bind(2) successful, keep copy of address
    self_ = Address::fromSockName(fd);

    // listen(2) on socket
    fd_ = fd;
    rv = ::listen(fd_, 1);
    FLOO_ENFORCE_NE(rv, -1, "listen");
    break;
  }

  // Expect listening file descriptor at this point.
  // If there is none, build error message that includes all
  // addresses that we attempted to bind to.
  if (fd_ == -1) {
    std::stringstream err;
    for (auto rp = result; rp != nullptr; rp = rp->ai_next) {
      err << Address(rp->ai_addr, rp->ai_addrlen).str();
      if (rp->ai_next != nullptr) {
        err << ", ";
      }
    }
    FLOO_ENFORCE_NE(fd_, -1, "Attempted to bind to: ", err);
  }

  freeaddrinfo(result);

  // Register with device so we're called when peer connects
  changeState(LISTENING);
  dev_->registerDescriptor(fd_, EPOLLIN, this);

  return;
}

void Pair::connect(const Address& peer) {
  std::unique_lock<std::mutex> lock(m_);
  int rv;
  socklen_t addrlen;

  peer_ = peer;

  // Addresses have to have same family
  FLOO_ENFORCE_EQ(
      self_.ss_.ss_family, peer_.ss_.ss_family, "address family mismatch");

  if (self_.ss_.ss_family == AF_INET) {
    struct sockaddr_in* sa = (struct sockaddr_in*)&self_.ss_;
    struct sockaddr_in* sb = (struct sockaddr_in*)&peer_.ss_;
    addrlen = sizeof(struct sockaddr_in);
    rv = memcmp(&sa->sin_addr, &sb->sin_addr, sizeof(struct in_addr));
    if (rv == 0) {
      rv = sa->sin_port - sb->sin_port;
    }
  } else if (peer_.ss_.ss_family == AF_INET6) {
    struct sockaddr_in6* sa = (struct sockaddr_in6*)&self_.ss_;
    struct sockaddr_in6* sb = (struct sockaddr_in6*)&peer_.ss_;
    addrlen = sizeof(struct sockaddr_in6);
    rv = memcmp(&sa->sin6_addr, &sb->sin6_addr, sizeof(struct in6_addr));
    if (rv == 0) {
      rv = sa->sin6_port - sb->sin6_port;
    }
  } else {
    FLOO_ENFORCE(false, "unknown sa_family");
  }

  FLOO_ENFORCE_NE(rv, 0, "cannot connect to self");

  // self_ < peer_; we are listening side.
  if (rv < 0) {
    while (state_ < CONNECTED) {
      cv_.wait(lock);
    }
    return;
  }

  // self_ > peer_; we are connecting side.
  // First destroy listening socket.
  dev_->unregisterDescriptor(fd_);
  close(fd_);

  // Create new socket to connect to peer.
  fd_ = socket(peer_.ss_.ss_family, SOCK_STREAM | SOCK_NONBLOCK, 0);
  FLOO_ENFORCE_NE(fd_, -1, "socket: ", strerror(errno));

  // Connect to peer
  rv = ::connect(fd_, (struct sockaddr*)&peer_.ss_, addrlen);
  if (rv == -1 && errno != EINPROGRESS) {
    FLOO_ENFORCE_NE(rv, -1, "connect: ", strerror(errno));
  }

  // Register with device so we're called when connection completes.
  changeState(CONNECTING);
  dev_->registerDescriptor(fd_, EPOLLIN | EPOLLOUT, this);

  // Wait for connection to complete
  while (state_ < CONNECTED) {
    cv_.wait(lock);
  }
}

// write is called from:
// 1) the device thread (the handleEvents function)
// 2) a user thread (the send function)
//
// In either case, the lock is held and the write function
// below inherits it.
//
bool Pair::write(Op& op) {
  std::array<struct iovec, 2> iov;
  int ioc = 0;
  int nbytes = 0;

  // Include preamble if necessary
  if (op.nwritten_ < sizeof(op.preamble_)) {
    iov[ioc].iov_base = ((char*)&op.preamble_) + op.nwritten_;
    iov[ioc].iov_len = sizeof(op.preamble_) - op.nwritten_;
    nbytes += iov[ioc].iov_len;
    ioc++;
  }

  // Include remaining piece of buffer
  int offset = op.preamble_.offset_;
  int length = op.preamble_.length_;
  if (op.nwritten_ > sizeof(op.preamble_)) {
    offset += op.nwritten_ - sizeof(op.preamble_);
    length -= op.nwritten_ - sizeof(op.preamble_);
  }
  iov[ioc].iov_base = ((char*)op.buf_->ptr_) + offset;
  iov[ioc].iov_len = length;
  nbytes += iov[ioc].iov_len;
  ioc++;

  int rv = writev(fd_, iov.data(), ioc);
  if (rv == -1 && errno == EAGAIN) {
    return false;
  }

  FLOO_ENFORCE_NE(rv, -1, "writev: ", strerror(errno));
  op.nwritten_ += rv;
  if (rv < nbytes) {
    return false;
  }

  FLOO_ENFORCE_EQ(rv, nbytes);
  return true;
}

// read is only called from the device thread (the handleEvents function).
// The lock is held from that function and is inherited below.
bool Pair::read(Op& op) {
  for (;;) {
    struct iovec iov;

    if (op.nread_ < sizeof(op.preamble_)) {
      // Read preamble
      iov.iov_base = ((char*)&op.preamble_) + op.nread_;
      iov.iov_len = sizeof(op.preamble_) - op.nread_;
    } else {
      // Read payload
      if (op.buf_ == nullptr) {
        op.buf_ = getBuffer(op.preamble_.slot_);
        // Buffer not (yet) registered, leave it for next loop iteration
        if (op.buf_ == nullptr) {
          return false;
        }
      }
      auto offset = op.nread_ - sizeof(op.preamble_);
      iov.iov_base = ((char*)op.buf_->ptr_) + offset;
      iov.iov_len = op.preamble_.length_ - offset;
    }

    int rv = readv(fd_, &iov, 1);
    if (rv == -1) {
      // EAGAIN happens when there are no more bytes left to read
      if (errno == EAGAIN) {
        return false;
      }

      // ECONNRESET happens when the remote peer unexpectedly terminates
      if (errno == ECONNRESET) {
        changeState(CLOSED);
        return false;
      }

      // Unexpected error
      FLOO_ENFORCE_EQ(
          errno, 0, "reading from ", peer_.str(), ": ", strerror(errno));
    }

    // Transition to CLOSED on EOF
    if (rv == 0) {
      changeState(CLOSED);
      return false;
    }

    op.nread_ += rv;

    // Verify the payload is non-empty after reading preamble
    if (op.nread_ == sizeof(op.preamble_)) {
      FLOO_ENFORCE_NE(op.preamble_.length_, 0);
    }

    // Return if op is complete
    if (op.nread_ == sizeof(op.preamble_) + op.preamble_.length_) {
      return true;
    }
  }
}

void Pair::handleEvents(int events) {
  // Try to acquire the pair's lock so the device thread (the thread
  // that ends up calling handleEvents) can mutate the tx and rx op
  // fields of this instance. If the lock cannot be acquired that
  // means some other thread is trying to mutate this pair's state,
  // which in turn might require calling into (and locking) the
  // underlying device (for example, when the pair transitions to the
  // CLOSED state). To avoid deadlocks, attempt to lock the pair and
  // skip handling the events until the next tick if the lock cannot
  // be acquired.
  std::unique_lock<std::mutex> lock(m_, std::try_to_lock);
  if (!lock) {
    return;
  }

  if (state_ == CONNECTED) {
    if (events & EPOLLOUT) {
      FLOO_ENFORCE(
          tx_.buf_ != nullptr,
          "tx_.buf_ cannot be NULL because EPOLLOUT happened");
      if (write(tx_)) {
        tx_.buf_->handleSendCompletion();
        memset(&tx_, 0, sizeof(tx_));
        dev_->registerDescriptor(fd_, EPOLLIN, this);
        cv_.notify_all();
      } else {
        // Write didn't complete, wait for epoll again
      }
    }
    if (events & EPOLLIN) {
      while (read(rx_)) {
        rx_.buf_->handleRecvCompletion();
        memset(&rx_, 0, sizeof(rx_));
      }
    }
    return;
  }

  if (state_ == LISTENING) {
    handleListening();
    return;
  }

  if (state_ == CONNECTING) {
    handleConnecting();
    return;
  }

  FLOO_ENFORCE(false, "Unexpected state: ", state_);
}

void Pair::handleListening() {
  struct sockaddr_storage addr;
  socklen_t addrlen = sizeof(addr);
  int rv;

  rv = accept(fd_, (struct sockaddr*)&addr, &addrlen);
  FLOO_ENFORCE_NE(rv, -1, "accept: ", strerror(errno));

  // Connected, replace file descriptor
  dev_->unregisterDescriptor(fd_);
  close(fd_);
  fd_ = rv;

  // Common connection-made code
  handleConnected();
}

void Pair::handleConnecting() {
  int optval;
  socklen_t optlen = sizeof(optval);
  int rv;

  // Verify that connecting was successful
  rv = getsockopt(fd_, SOL_SOCKET, SO_ERROR, &optval, &optlen);
  FLOO_ENFORCE_NE(rv, -1);
  FLOO_ENFORCE_EQ(optval, 0, "SO_ERROR: ", strerror(optval));

  // Common connection-made code
  handleConnected();
}

void Pair::handleConnected() {
  int rv;

  // Reset addresses
  self_ = Address::fromSockName(fd_);
  peer_ = Address::fromPeerName(fd_);

  // Make sure socket is non-blocking
  rv = fcntl(fd_, F_GETFL);
  FLOO_ENFORCE_NE(rv, -1);
  rv = fcntl(fd_, F_SETFL, rv | O_NONBLOCK);
  FLOO_ENFORCE_NE(rv, -1);

  int flag = 1;
  socklen_t optlen = sizeof(flag);
  rv = setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, optlen);
  FLOO_ENFORCE_NE(rv, -1);

  dev_->registerDescriptor(fd_, EPOLLIN, this);
  changeState(CONNECTED);
}

// getBuffer must only be called when holding lock.
Buffer* Pair::getBuffer(int slot) {
  for (;;) {
    auto it = buffers_.find(slot);
    if (it == buffers_.end()) {
      // The remote peer already sent some bytes destined for the
      // buffer at this slot, but this side of the pair hasn't
      // registed it yet.
      //
      // The current strategy is to return and let the the device loop
      // repeatedly call us again until the buffer has been
      // registered. This essentially means busy waiting while
      // yielding to other pairs. This is not a problem as this only
      // happens at initialization time.
      //
      return nullptr;
    }

    return it->second;
  }
}

void Pair::registerBuffer(Buffer* buf) {
  std::lock_guard<std::mutex> lock(m_);
  FLOO_ENFORCE(
      buffers_.find(buf->slot_) == buffers_.end(),
      "duplicate buffer for slot ",
      buf->slot_);
  buffers_[buf->slot_] = buf;
  cv_.notify_all();
}

void Pair::unregisterBuffer(Buffer* buf) {
  std::lock_guard<std::mutex> lock(m_);
  buffers_.erase(buf->slot_);
}

// changeState must only be called when holding lock.
void Pair::changeState(state state) {
  // Clean up file descriptor when transitioning to CLOSED
  if (state_ == CONNECTED && state == CLOSED) {
    dev_->unregisterDescriptor(fd_);
    close(fd_);
    fd_ = -1;
  }

  state_ = state;
  cv_.notify_all();
}

void Pair::send(Op& op) {
  std::unique_lock<std::mutex> lock(m_);

  // Wait for pair to be connected
  while (state_ < CONNECTED) {
    cv_.wait(lock);
  }

  FLOO_ENFORCE_EQ(
      CONNECTED,
      state_,
      "Pair is closed (",
      self_.str(),
      " <--> ",
      peer_.str(),
      ")");

  // Try to size the send buffer such that the write below completes
  // synchronously and we don't need to finish the write later.
  auto size = 2 * (sizeof(op.preamble_) + op.preamble_.length_);
  if (sendBufferSize_ < size) {
    int rv;
    int optval = size;
    socklen_t optlen = sizeof(optval);
    rv = setsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &optval, optlen);
    FLOO_ENFORCE_NE(rv, -1);
    rv = getsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &optval, &optlen);
    FLOO_ENFORCE_NE(rv, -1);
    sendBufferSize_ = optval;
  }

  // Wait until event loop has finished current write.
  while (tx_.buf_ != nullptr) {
    cv_.wait(lock);
  }

  // Write immediately without checking socket for writeability.
  if (write(op)) {
    op.buf_->handleSendCompletion();
    return;
  }

  // Write didn't complete; pass to event loop
  tx_ = op;
  dev_->registerDescriptor(fd_, EPOLLIN | EPOLLOUT, this);
}

std::unique_ptr<::floo::transport::Buffer>
Pair::createSendBuffer(int slot, void* ptr, size_t size) {
  auto buffer = new Buffer(this, slot, ptr, size);
  return std::unique_ptr<::floo::transport::Buffer>(buffer);
}

std::unique_ptr<::floo::transport::Buffer>
Pair::createRecvBuffer(int slot, void* ptr, size_t size) {
  auto buffer = new Buffer(this, slot, ptr, size);
  registerBuffer(buffer);
  return std::unique_ptr<::floo::transport::Buffer>(buffer);
}

} // namespace tcp
} // namespace transport
} // namespace floo
