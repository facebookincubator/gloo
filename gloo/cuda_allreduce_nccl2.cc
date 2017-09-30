/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_allreduce_nccl2.h"

#include "gloo/broadcast_one_to_all.h"
#include "gloo/cuda_private.h"


namespace gloo {

template <typename T, typename W>
CudaAllreduceNccl2<T, W>::CudaAllreduceNccl2(
  const std::shared_ptr<Context>& context,
  const std::vector<T*>& ptrs,
  const int count,
  const std::vector<cudaStream_t>& streams)
    : Algorithm(context),
      count_(count),
      bytes_(count_ * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0),
      fn_(CudaReductionFunction<T>::sum) {
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  for (auto i = 0; i < ptrs.size(); i++) {
    auto ptr = CudaDevicePointer<T>::create(ptrs[i], count_);
    if (newStream) {
      streams_.push_back(CudaStream(ptr.getDeviceID()));
    } else {
      streams_.push_back(CudaStream(ptr.getDeviceID(), streams[i]));
    }
    devicePtrs_.push_back(std::move(ptr));
  }

  // Generate unique ID on root node
  ncclUniqueId id;
  std::vector<int8_t*> ids;
  ids.push_back((int8_t*)id.internal);
  if (context->rank == 0) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    ncclGetUniqueId(&id);
  }

  // Broadcast ID to other nodes
  BroadcastOneToAll<int8_t>(context, ids, NCCL_UNIQUE_ID_BYTES).run();

  // send localDevices to all nodes
  const int localDevices = ptrs.size();
  std::vector<int> numDevices(context->size);
  std::vector<std::vector<int*>> numDevicesRefs(context->size);
  for (int i=0; i<context->size; i++) {
    numDevicesRefs[i].push_back(&numDevices[i]);
    numDevices[i] = (i == context->rank) ? localDevices : -1;
    BroadcastOneToAll<int>(context, numDevicesRefs[i], 1, i).run();
  }

  // Initialize nccl comms
  int ncclSize = 0;
  int ncclRank = 0;
  for (int i=0; i<context->size; i++) {
    ncclSize += numDevices[i];
    if (i < context->rank)
      ncclRank += numDevices[i];
  }
  comms_.resize(localDevices);
  {
    ncclGroupStart();
    for (int i=0; i<localDevices; i++) {
      CUDA_CHECK(cudaSetDevice(devicePtrs_[i].getDeviceID()));
      std::lock_guard<std::mutex> lock(CudaShared::getMutex());
      NCCL_CHECK(ncclCommInitRank(&comms_[i], ncclSize, id, ncclRank + i));
    }
    ncclGroupEnd();
  }
}

template <typename T, typename W>
void CudaAllreduceNccl2<T, W>::run() {
  {
    ncclGroupStart();
    for (int i=0; i<devicePtrs_.size(); i++) {
      std::lock_guard<std::mutex> lock(CudaShared::getMutex());
      NCCL_CHECK(ncclAllReduce(
            (const void*)(*devicePtrs_[i]), (void*)(*devicePtrs_[i]),
            count_, nccl::ncclTypeWrapper<T>::type, ncclSum, comms_[i], *streams_[i]));
    }
    ncclGroupEnd();
  }

  for (int i=0; i<devicePtrs_.size(); i++)
    CUDA_CHECK(cudaStreamSynchronize(*streams_[i]));
}

template <typename T, typename W>
CudaAllreduceNccl2<T, W>::~CudaAllreduceNccl2() {
  std::lock_guard<std::mutex> lock(CudaShared::getMutex());
  for (auto& comm : comms_)
    ncclCommDestroy(comm);
}

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T)                                         \
template class CudaAllreduceNccl2<T, CudaHostWorkspace<T> >;            \
template class CudaAllreduceNccl2<T, CudaDeviceWorkspace<T> >;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
