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
    HalvingDoubling,
    HalvingDoublingPipelined,
    Ring,
    RingChunked,
  };

  AllreduceBuilder();

  AllreduceBuilder(const AllreduceBuilder& rhs) = default;

  AllreduceBuilder<T> setInputs(const std::vector<T*>& inputs) const;

  AllreduceBuilder<T> setCount(int count) const;

  AllreduceBuilder<T> setReductionType(ReductionType reductionType) const;

  AllreduceBuilder<T> setImplementation(Implementation implementation) const;

#if GLOO_USE_CUDA
  AllreduceBuilder<T> setStreams(
      const std::vector<cudaStream_t>& streams) const;

  AllreduceBuilder<T> useGPUDirect() const;
#endif

  std::unique_ptr<Algorithm> getAlgorithm(std::shared_ptr<Context>& context);

 protected:
  std::vector<T*> inputs_;
  std::vector<T*> outputs_;
  int count_;
  ReductionType reductionType_;
  Implementation implementation_;

#if GLOO_USE_CUDA
  std::vector<cudaStream_t> streams_;
  bool gpuDirect_;
#endif
};

} // namespace
