/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "floo/algorithm.h"
#include "floo/common/logging.h"

namespace floo {

class Barrier : public Algorithm {
 public:
  explicit Barrier(const std::shared_ptr<Context>& context)
      : Algorithm(context) {}

  virtual ~Barrier(){};
};

} // namespace floo
