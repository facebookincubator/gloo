/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

#include <sys/epoll.h>

namespace gloo {
namespace transport {
namespace tcp {

// Handler abstract base class called by the epoll(2) event loop.
// Dispatch to multiple types is needed because we must deal with a
// single listening socket on the device instance and I/O for all pair
// instances. Before this approach, we'd exclusively deal with `Pair`
// instances and didn't need to dispatch events to different types.
class Handler {
 public:
  virtual ~Handler() = default;

  virtual void handleEvents(int events) = 0;
};

class Loop;

// Functions can be deferred to the epoll(2) thread through the this
// class. It uses readability of a pipe to wake up the event loop.
class Deferrables final : public Handler {
public:
  using function_t = std::function<void()>;

  Deferrables();

  ~Deferrables() override;

  void defer(function_t fn);

  void handleEvents(int events) override;

private:
  int rfd_;
  int wfd_;

  std::mutex mutex_;
  std::list<function_t> functions_;
  bool triggered_{false};

  friend class Loop;
};

class Loop final : public std::enable_shared_from_this<Loop> {
 public:
  explicit Loop();

  ~Loop();

  void registerDescriptor(int fd, int events, Handler* h);

  void unregisterDescriptor(int fd);

  void defer(std::function<void()> fn);

  void run();

 private:
  static constexpr auto capacity_ = 64;

  int fd_{-1};
  std::atomic<bool> done_{false};
  Deferrables deferrables_;
  std::unique_ptr<std::thread> loop_;

  std::mutex m_;
  std::condition_variable cv_;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
