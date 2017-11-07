/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <vector>

#include "gloo/algorithm.h"
#include "gloo/config.h"
#include "gloo/context.h"

#if GLOO_USE_CUDA
#include "gloo/cuda.h"
#endif

namespace gloo {

template <typename T>
class BroadcastBuilder {
 public:

  // Construct BroadcastBuilder with following defaults:
  BroadcastBuilder();

  BroadcastBuilder(const BroadcastBuilder<T>& rhs) = default;

  BroadcastBuilder<T>& setInputs(const std::vector<T*>& inputs);

  BroadcastBuilder<T>& setCount(int count);

  BroadcastBuilder<T>& setRootRank(int rootRank);

  BroadcastBuilder<T>& setRootPointerRank(int rootPointerRank);

#if GLOO_USE_CUDA
  BroadcastBuilder<T>& setStreams(const std::vector<cudaStream_t>& streams);

  BroadcastBuilder<T>& setGPUDirect(bool on);
#endif

  std::unique_ptr<Algorithm> getAlgorithm(std::shared_ptr<Context>& context);

 protected:
  std::vector<T*> inputs_;
  int count_;
  int rootRank_;
  int rootPointerRank_;

#if GLOO_USE_CUDA
  std::vector<cudaStream_t> streams_;
  bool gpuDirect_;
#endif
};

} // namespace
