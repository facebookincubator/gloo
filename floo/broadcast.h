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

template <typename T>
class Broadcast : public Algorithm {
 public:
  Broadcast(const std::shared_ptr<Context>& context, int rootRank)
      : Algorithm(context), rootRank_(rootRank) {
    FLOO_ENFORCE_GE(rootRank_, 0);
    FLOO_ENFORCE_LT(rootRank_, contextSize_);
  }

  virtual ~Broadcast(){};

 protected:
  const int rootRank_;
};

} // namespace floo
