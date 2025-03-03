/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>

#include <gloo/common/logging.h>
#include <gloo/transport/tcp/error.h>
#include <gloo/transport/tcp/loop.h>
#include <gloo/transport/tcp/socket.h>

namespace gloo {
namespace transport {
namespace tcp {

// ReadValueOperation asynchronously reads a value of type T from the
// socket specified at construction. Upon completion or error, the
// callback is called. Its lifetime is coupled with completion of the
// operation, so the called doesn't need to hold on to the instance.
// It does so by storing a shared_ptr to itself (effectively a leak)
// until the event loop calls back.
template <typename T>
class ReadValueOperation final
    : public Handler,
      public std::enable_shared_from_this<ReadValueOperation<T>> {
 public:
  using callback_t =
      std::function<void(std::shared_ptr<Socket>, const Error& error, T&& t)>;

  ReadValueOperation(
      std::shared_ptr<Loop> loop,
      std::shared_ptr<Socket> socket,
      callback_t fn)
      : loop_(std::move(loop)),
        socket_(std::move(socket)),
        fn_(std::move(fn)) {}

  void run() {
    // Cannot initialize leak until after the object has been
    // constructed, because the std::make_shared initialization
    // doesn't run after construction of the underlying object.
    leak_ = this->shared_from_this();
    // Register with loop only after we've leaked the shared_ptr,
    // because we unleak it when the event loop thread calls.
    loop_->registerDescriptor(socket_->fd(), EPOLLIN | EPOLLONESHOT, this);
  }

  void handleEvents(int events) override {
    // Move leaked shared_ptr to the stack so that this object
    // destroys itself once this function returns.
    auto self = std::move(this->leak_);

    // Read T.
    auto rv = socket_->read(&t_, sizeof(t_));
    if (rv == -1) {
      fn_(socket_,
          SystemError("read", errno, socket_->peerName()),
          std::move(t_));
      return;
    }

    // Check for short read (assume we can read in a single call).
    if (rv < sizeof(t_)) {
      fn_(socket_,
          ShortReadError(rv, sizeof(t_), socket_->peerName()),
          std::move(t_));
      return;
    }

    fn_(socket_, Error::kSuccess, std::move(t_));
  }

 private:
  std::shared_ptr<Loop> loop_;
  std::shared_ptr<Socket> socket_;
  callback_t fn_;
  std::shared_ptr<ReadValueOperation<T>> leak_;

  T t_;
};

template <typename T>
void read(
    std::shared_ptr<Loop> loop,
    std::shared_ptr<Socket> socket,
    typename ReadValueOperation<T>::callback_t fn) {
  auto x = std::make_shared<ReadValueOperation<T>>(
      std::move(loop), std::move(socket), std::move(fn));
  x->run();
}

// WriteValueOperation asynchronously writes a value of type T to the
// socket specified at construction. Upon completion or error, the
// callback is called. Its lifetime is coupled with completion of the
// operation, so the called doesn't need to hold on to the instance.
// It does so by storing a shared_ptr to itself (effectively a leak)
// until the event loop calls back.
template <typename T>
class WriteValueOperation final
    : public Handler,
      public std::enable_shared_from_this<WriteValueOperation<T>> {
 public:
  using callback_t =
      std::function<void(std::shared_ptr<Socket>, const Error& error)>;

  WriteValueOperation(
      std::shared_ptr<Loop> loop,
      std::shared_ptr<Socket> socket,
      T t,
      callback_t fn)
      : loop_(std::move(loop)),
        socket_(std::move(socket)),
        fn_(std::move(fn)),
        t_(std::move(t)) {}

  void run() {
    // Cannot initialize leak until after the object has been
    // constructed, because the std::make_shared initialization
    // doesn't run after construction of the underlying object.
    leak_ = this->shared_from_this();
    // Register with loop only after we've leaked the shared_ptr,
    // because we unleak it when the event loop thread calls.
    loop_->registerDescriptor(socket_->fd(), EPOLLOUT | EPOLLONESHOT, this);
  }

  void handleEvents(int events) override {
    // Move leaked shared_ptr to the stack so that this object
    // destroys itself once this function returns.
    auto leak = std::move(this->leak_);

    // Write T.
    auto rv = socket_->write(&t_, sizeof(t_));
    if (rv == -1) {
      fn_(socket_, SystemError("write", errno, socket_->peerName()));
      return;
    }

    // Check for short write (assume we can write in a single call).
    if (rv < sizeof(t_)) {
      fn_(socket_, ShortWriteError(rv, sizeof(t_), socket_->peerName()));
      return;
    }

    fn_(socket_, Error::kSuccess);
  }

 private:
  std::shared_ptr<Loop> loop_;
  std::shared_ptr<Socket> socket_;
  callback_t fn_;
  std::shared_ptr<WriteValueOperation<T>> leak_;

  T t_;
};

template <typename T>
void write(
    std::shared_ptr<Loop> loop,
    std::shared_ptr<Socket> socket,
    T t,
    typename WriteValueOperation<T>::callback_t fn) {
  auto x = std::make_shared<WriteValueOperation<T>>(
      std::move(loop), std::move(socket), std::move(t), std::move(fn));
  x->run();
}

class ConnectOperation final
    : public Handler,
      public std::enable_shared_from_this<ConnectOperation> {
 public:
  using callback_t =
      std::function<void(std::shared_ptr<Socket>, const Error& error)>;
  ConnectOperation(
      std::shared_ptr<Loop> loop,
      const Address& remote,
      std::chrono::milliseconds timeout,
      callback_t fn)
      : remote_(remote),
        deadline_(std::chrono::steady_clock::now() + timeout),
        loop_(std::move(loop)),
        fn_(std::move(fn)) {}

  void run() {
    // Cannot initialize leak until after the object has been
    // constructed, because the std::make_shared initialization
    // doesn't run after construction of the underlying object.
    leak_ = this->shared_from_this();

    const auto& sockaddr = remote_.getSockaddr();

    // Create new socket to connect to peer.
    socket_ = Socket::createForFamily(sockaddr.ss_family);
    socket_->reuseAddr(true);
    socket_->noDelay(true);
    socket_->connect(sockaddr);

    // Register with loop only after we've leaked the shared_ptr,
    // because we unleak it when the event loop thread calls.
    // Register for EPOLLOUT, because we want to be notified when
    // the connect completes. EPOLLERR is also necessary because
    // connect() can fail.
    if (auto loop = loop_.lock()) {
      loop->registerDescriptor(
          socket_->fd(), EPOLLOUT | EPOLLERR | EPOLLONESHOT, this);
    } else {
      fn_(socket_, LoopError("loop is gone"));
    }
  }

  void handleEvents(int events) override {
    // Move leaked shared_ptr to the stack so that this object
    // destroys itself once this function returns.
    auto leak = std::move(this->leak_);

    int result;
    socklen_t result_len = sizeof(result);
    if (getsockopt(socket_->fd(), SOL_SOCKET, SO_ERROR, &result, &result_len) <
        0) {
      fn_(socket_, SystemError("getsockopt", errno, remote_));
      return;
    }
    if (result != 0) {
      SystemError e("SO_ERROR", result, remote_);
      bool willRetry = std::chrono::steady_clock::now() < deadline_ &&
          retry_++ < maxRetries_;
      GLOO_ERROR(
          "failed to connect, willRetry=",
          willRetry,
          ", retry=",
          retry_,
          ", remote=",
          remote_.str(),
          ", error=",
          e.what());
      // check deadline
      if (willRetry) {
        run();
      } else {
        fn_(socket_, TimeoutError("timed out connecting: " + e.what()));
      }
      return;
    }

    fn_(socket_, Error::kSuccess);
  }

 private:
  const Address remote_;
  const std::chrono::time_point<std::chrono::steady_clock> deadline_;
  const int maxRetries_{3};

  int retry_{0};

  // We use a weak_ptr to the loop to avoid a reference cycle when an error
  // occurs.
  std::weak_ptr<Loop> loop_;
  std::shared_ptr<Socket> socket_;
  callback_t fn_;
  std::shared_ptr<ConnectOperation> leak_;
};

void connectLoop(
    std::shared_ptr<Loop> loop,
    const Address& remote,
    std::chrono::milliseconds timeout,
    typename ConnectOperation::callback_t fn);

} // namespace tcp
} // namespace transport
} // namespace gloo
