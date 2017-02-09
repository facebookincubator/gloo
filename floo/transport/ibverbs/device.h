/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <atomic>
#include <thread>

#include <infiniband/verbs.h>

#include "floo/transport/device.h"

namespace floo {
namespace transport {
namespace ibverbs {

struct attr {
  std::string name;
  int port;
  int index;
};

std::shared_ptr<::floo::transport::Device> CreateDevice(
    const struct attr&);

// Forward declarations
class Pair;
class Buffer;

// Pure virtual base class for Pair/Buffer.
// Used to dispatch completion handling from device loop.
class Handler {
 public:
  virtual ~Handler() {}
  virtual void handleCompletion(struct ibv_wc* wc) = 0;
};

class Device : public ::floo::transport::Device,
               public std::enable_shared_from_this<Device> {
  static const int capacity_ = 64;

 public:
  Device(const struct attr& attr, ibv_context* context);
  virtual ~Device();

  virtual std::unique_ptr<::floo::transport::Pair> createPair()
      override;

 protected:
  struct attr attr_;
  ibv_context* context_;
  ibv_pd* pd_;
  ibv_comp_channel* comp_channel_;
  ibv_cq* cq_;

  void loop();

  std::atomic<bool> done_;
  std::unique_ptr<std::thread> loop_;

  friend class Pair;
  friend class Buffer;
};

} // namespace ibverbs
} // namespace transport
} // namespace floo
