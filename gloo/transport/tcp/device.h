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

#include <sys/socket.h>

#include "gloo/transport/device.h"

namespace gloo {
namespace transport {
namespace tcp {

struct attr {
  attr() {}
  /* implicit */ attr(const char* ptr) : hostname(ptr) {}

  std::string hostname;

  std::string iface;

  // The address family defaults to AF_UNSPEC such that getaddrinfo(3)
  // will try to find either IPv4 or IPv6 addresses.
  int ai_family = AF_UNSPEC;
  int ai_socktype;
  int ai_protocol;
  struct sockaddr_storage ai_addr;
  int ai_addrlen;
};

std::shared_ptr<::gloo::transport::Device> CreateDevice(
    const struct attr&);

// Forward declarations
class Pair;
class Buffer;

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

class Device : public ::gloo::transport::Device,
               public std::enable_shared_from_this<Device> {
 public:
  explicit Device(const struct attr& attr);
  virtual ~Device();

  virtual std::string str() const override;

  virtual const std::string& getPCIBusID() const override;

  virtual int getInterfaceSpeed() const override;

  virtual std::shared_ptr<::gloo::transport::Context> createContext(
      int rank, int size) override;

 protected:
  void loop();

  void registerDescriptor(int fd, int events, Handler* h);
  void unregisterDescriptor(int fd);

  const struct attr attr_;
  std::atomic<bool> done_;
  std::unique_ptr<std::thread> loop_;

  friend class Pair;
  friend class Buffer;

 private:
  static constexpr auto capacity_ = 64;

  int fd_;
  std::string interfaceName_;
  int interfaceSpeedMbps_;
  std::string pciBusID_;

  std::mutex m_;
  std::condition_variable cv_;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
