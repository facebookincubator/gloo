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
class AllreduceBuilder {
 public:
  enum Implementation {
    Ring = 1,
    RingChunked = 2,
    HalvingDoubling = 3,
    HalvingDoublingPipelined = 4,
    Bcube = 5,
  };

  // Construct AllreduceBuilder with following defaults:
  // - ReductionType: SUM
  // - Implementation: HalvingDoubling
  AllreduceBuilder();

  AllreduceBuilder(const AllreduceBuilder<T>& rhs) = default;

  AllreduceBuilder<T>& setInputs(const std::vector<T*>& inputs);

  AllreduceBuilder<T>& setCount(int count);

  AllreduceBuilder<T>& setReductionType(ReductionType reductionType);

  AllreduceBuilder<T>& setImplementation(Implementation implementation);

#if GLOO_USE_CUDA
  AllreduceBuilder<T>& setStreams(const std::vector<cudaStream_t>& streams);

  AllreduceBuilder<T>& setGPUDirect(bool on);
#endif

  std::unique_ptr<Algorithm> getAlgorithm(std::shared_ptr<Context>& context);

 protected:
  std::vector<T*> inputs_;
  int count_;
  ReductionType reductionType_;
  Implementation implementation_;

#if GLOO_USE_CUDA
  std::vector<cudaStream_t> streams_;
  bool gpuDirect_;
#endif
};

} // namespace
