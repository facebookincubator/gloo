/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <algorithm>

#include "gloo/broadcast_builder.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/common/logging.h"

#if GLOO_USE_CUDA
#include "gloo/cuda_broadcast_one_to_all.h"
#endif

namespace gloo {

template <typename T>
BroadcastBuilder<T>::BroadcastBuilder() :
    inputs_(0),
    count_(0),
    rootRank_(0),
    rootPointerRank_(0) {
  // Expect downstream code to set all properties
#if GLOO_USE_CUDA
  streams_.clear();
  gpuDirect_ = false;
#endif
}

template <typename T>
BroadcastBuilder<T>& BroadcastBuilder<T>::setInputs(
    const std::vector<T*>& inputs) {
  inputs_ = inputs;
  return *this;
}

template <typename T>
BroadcastBuilder<T>& BroadcastBuilder<T>::setCount(int count) {
  count_ = count;
  return *this;
}

template <typename T>
BroadcastBuilder<T>& BroadcastBuilder<T>::setRootRank(int rootRank) {
  rootRank_ = rootRank;
  return *this;
}

template <typename T>
BroadcastBuilder<T>& BroadcastBuilder<T>::setRootPointerRank(
    int rootPointerRank) {
  rootPointerRank_ = rootPointerRank;
  return *this;
}

#if GLOO_USE_CUDA

template <typename T>
BroadcastBuilder<T>& BroadcastBuilder<T>::setStreams(
    const std::vector<cudaStream_t>& streams) {
  streams_ = streams;
  return *this;
}

template <typename T>
BroadcastBuilder<T>& BroadcastBuilder<T>::setGPUDirect(bool on) {
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
std::unique_ptr<Algorithm> BroadcastBuilder<T>::getAlgorithm(
    std::shared_ptr<Context>& context) {

#if GLOO_USE_CUDA
  // Instantiate CUDA algorithm if all pointers are GPU pointers
  if (gloo::BuilderHelpers<T>::checkAllPointersGPU(inputs_)) {
    return getAlgorithmCuda<CudaBroadcastOneToAll, T>(
      gpuDirect_, context, inputs_, count_, rootRank_, rootPointerRank_, streams_);
  }
#endif

  return std::unique_ptr<::gloo::Algorithm>(
    new BroadcastOneToAll<T>(context, inputs_, count_,
      rootRank_, rootPointerRank_));
}

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T) template class BroadcastBuilder<T>;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(uint8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
