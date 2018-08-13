/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/hip_allreduce_local.h"

#include "gloo/common/logging.h"
#include "gloo/hip_collectives_device.h"
#include "gloo/hip_private.h"

namespace gloo {

template <typename T>
HipAllreduceLocal<T>::HipAllreduceLocal(
  const std::shared_ptr<Context>& context,
  const std::vector<T*>& ptrs,
  const int count,
  const std::vector<hipStream_t>& streams)
    : Algorithm(context),
      count_(count),
      bytes_(count_ * sizeof(T)),
      fn_(HipReductionFunction<T>::sum),
      synchronizeDeviceOutputs_(streams.size() == 0) {
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  for (auto i = 0; i < ptrs.size(); i++) {
    auto ptr = HipDevicePointer<T>::create(ptrs[i], count_);
    if (newStream) {
      streams_.push_back(HipStream(ptr.getDeviceID()));
    } else {
      streams_.push_back(HipStream(ptr.getDeviceID(), streams[i]));
    }
    devicePtrs_.push_back(std::move(ptr));
  }

  // Initialize local reduce / local broadcast
  // TODO(pietern): Optimize this to use real direct allreduce if possible
  if (devicePtrs_.size() > 1) {
    localReduceOp_ =
      hipDeviceReduce(streams_, devicePtrs_, devicePtrs_[0], fn_, 0, count_);
    localBroadcastOp_ =
      hipDeviceBroadcast(streams_, devicePtrs_, devicePtrs_[0], 0, count_);
  }
}

template <typename T>
void HipAllreduceLocal<T>::run() {
  HipDeviceGuard guard;

  if (devicePtrs_.size() > 1) {
    localReduceOp_->runAsync();
    localBroadcastOp_->runAsync();
    if (synchronizeDeviceOutputs_) {
      localBroadcastOp_->wait();
    }
  }
}

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T) template class HipAllreduceLocal<T>;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(uint8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
