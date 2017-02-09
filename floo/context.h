/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <memory>
#include <vector>

#include "floo/rendezvous/store.h"
#include "floo/transport/device.h"
#include "floo/transport/pair.h"

namespace floo {

class Context {
 public:
  Context(int rank, int size);

  const int rank_;
  const int size_;

  void connectFullMesh(
      rendezvous::Store& store,
      std::shared_ptr<transport::Device>& dev);

  std::unique_ptr<transport::Pair>& getPair(int i) {
    return pairs_.at(i);
  }

 protected:
  std::vector<std::unique_ptr<transport::Pair>> pairs_;
};

} // namespace floo
