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

class Loop final : public std::enable_shared_from_this<Loop> {
 public:
  explicit Loop();

  ~Loop();

  void registerDescriptor(int fd, int events, Handler* h);

  void unregisterDescriptor(int fd, Handler *h);

  void run();

 private:
  static constexpr auto capacity_ = 64;

  int fd_{-1};
  std::atomic<bool> done_{false};
  std::unique_ptr<std::thread> loop_;

  std::mutex m_;
  std::condition_variable cv_;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
