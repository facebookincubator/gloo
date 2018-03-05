/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <algorithm>

#include "gloo/allreduce_bcube.h"
#include "gloo/allreduce_builder.h"
#include "gloo/allreduce_halving_doubling.h"
#include "gloo/allreduce_local.h"
#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/common/logging.h"

#if GLOO_USE_CUDA
#include "gloo/cuda_allreduce_halving_doubling.h"
#include "gloo/cuda_allreduce_halving_doubling_pipelined.h"
#include "gloo/cuda_allreduce_local.h"
#include "gloo/cuda_allreduce_ring.h"
#include "gloo/cuda_allreduce_ring_chunked.h"
#endif

namespace gloo {

template <typename T>
AllreduceBuilder<T>::AllreduceBuilder() :
    inputs_(0),
    count_(0),
    reductionType_(SUM),
    implementation_(HalvingDoubling) {
  // Expect downstream code to set all properties
#if GLOO_USE_CUDA
  streams_.clear();
  gpuDirect_ = false;
#endif
}

template <typename T>
AllreduceBuilder<T>& AllreduceBuilder<T>::setInputs(
    const std::vector<T*>& inputs) {
  inputs_ = inputs;
  return *this;
}

template <typename T>
AllreduceBuilder<T>& AllreduceBuilder<T>::setCount(int count) {
  count_ = count;
  return *this;
}

template <typename T>
AllreduceBuilder<T>& AllreduceBuilder<T>::setReductionType(
    ReductionType reductionType) {
  reductionType_ = reductionType;
  return *this;
}

template <typename T>
AllreduceBuilder<T>& AllreduceBuilder<T>::setImplementation(
    Implementation implementation) {
  implementation_ = implementation;
  return *this;
}

#if GLOO_USE_CUDA

template <typename T>
AllreduceBuilder<T>& AllreduceBuilder<T>::setStreams(
    const std::vector<cudaStream_t>& streams) {
  streams_ = streams;
  return *this;
}

template <typename T>
AllreduceBuilder<T>& AllreduceBuilder<T>::setGPUDirect(bool on) {
  gpuDirect_ = on;
  return *this;
}

namespace {

template <
  template <typename T, typename W> class A,
  typename T,
  typename ...Params>
std::unique_ptr<Algorithm> getAlgorithmCuda(
    bool gpuDirect,
    Params&&... params) {
  if (gpuDirect) {
    return std::unique_ptr<::gloo::Algorithm>(
      new A<T, CudaDeviceWorkspace<T>>(
        std::forward<Params>(params)...));
  } else {
    return std::unique_ptr<::gloo::Algorithm>(
      new A<T, CudaHostWorkspace<T>>(
        std::forward<Params>(params)...));
  }
}

} // namespace

#endif

template <typename T>
std::unique_ptr<Algorithm> AllreduceBuilder<T>::getAlgorithm(
    std::shared_ptr<Context>& context) {
  // Expect necessary properties to be set
  GLOO_ENFORCE_NE(reductionType_, 0);
  GLOO_ENFORCE_NE(implementation_, 0);

#if GLOO_USE_CUDA
  // Instantiate CUDA algorithm if all pointers are GPU pointers
  if (gloo::BuilderHelpers<T>::checkAllPointersGPU(inputs_)) {
    // TODO(pietern): Pass through the right reduction function to algorithm.
    // The CUDA algorithms are still hardcoded to use SUM.
    GLOO_ENFORCE_EQ(reductionType_, SUM);

    if (context->size == 1) {
      return std::unique_ptr<::gloo::Algorithm>(
        new CudaAllreduceLocal<T>(
          context,
          inputs_,
          count_,
          streams_));
    }

    switch (implementation_) {
      case HalvingDoubling:
        return getAlgorithmCuda<CudaAllreduceHalvingDoubling, T>(
          gpuDirect_, context, inputs_, count_, streams_);
        break;
      case HalvingDoublingPipelined:
        return getAlgorithmCuda<CudaAllreduceHalvingDoublingPipelined, T>(
          gpuDirect_, context, inputs_, count_, streams_);
        break;
      case Ring:
        return getAlgorithmCuda<CudaAllreduceRing, T>(
          gpuDirect_, context, inputs_, count_, streams_);
        break;
      case RingChunked:
        return getAlgorithmCuda<CudaAllreduceRingChunked, T>(
          gpuDirect_, context, inputs_, count_, streams_);
        break;
      default:
        GLOO_ENFORCE(false, "Unhandled implementation: ", implementation_);
        break;
    }
  }
#endif

  const ReductionFunction<T>* fn;
  switch (reductionType_) {
    case SUM:
      fn = ReductionFunction<T>::sum;
      break;
    case PRODUCT:
      fn = ReductionFunction<T>::product;
      break;
    case MIN:
      fn = ReductionFunction<T>::min;
      break;
    case MAX:
      fn = ReductionFunction<T>::max;
      break;
    default:
      GLOO_ENFORCE(false, "Unhandled reduction type: ", reductionType_);
      break;
  }

  if (context->size == 1) {
    return std::unique_ptr<::gloo::Algorithm>(
      new AllreduceLocal<T>(context, inputs_, count_, fn));
  }

  switch (implementation_) {
    case Bcube:
      return std::unique_ptr<::gloo::Algorithm>(
          new AllreduceBcube<T>(context, inputs_, count_, fn));
      break;
    case HalvingDoubling:
      return std::unique_ptr<::gloo::Algorithm>(
        new AllreduceHalvingDoubling<T>(context, inputs_, count_, fn));
      break;
    case HalvingDoublingPipelined:
      return std::unique_ptr<::gloo::Algorithm>(
        new AllreduceHalvingDoubling<T>(context, inputs_, count_, fn));
      break;
    case Ring:
      return std::unique_ptr<::gloo::Algorithm>(
        new AllreduceRing<T>(context, inputs_, count_, fn));
      break;
    case RingChunked:
      return std::unique_ptr<::gloo::Algorithm>(
        new AllreduceRingChunked<T>(context, inputs_, count_, fn));
      break;
    default:
      GLOO_ENFORCE(false, "Unhandled implementation: ", implementation_);
      break;
  }
}

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T) template class AllreduceBuilder<T>;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(uint8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
