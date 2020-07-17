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

#include <gloo/transport/device.h>
#include <gloo/transport/tcp/attr.h>
#include <gloo/transport/tcp/loop.h>

namespace gloo {
namespace transport {
namespace tcp {

struct attr CreateDeviceAttr(const struct attr& src);

std::shared_ptr<::gloo::transport::Device> CreateDevice(
    const struct attr&);

// Forward declarations
class Pair;
class Buffer;

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

  void registerDescriptor(int fd, int events, Handler* h);
  void unregisterDescriptor(int fd);

 protected:
  const struct attr attr_;

  friend class Pair;
  friend class Buffer;

 private:
  std::shared_ptr<Loop> loop_;

  std::string interfaceName_;
  int interfaceSpeedMbps_;
  std::string pciBusID_;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
