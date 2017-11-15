/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * Copyright (c) 2017-present, NVIDIA CORPORATION,
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_allreduce_nccl2.h"

#include "gloo/broadcast_one_to_all.h"
#include "gloo/cuda_private.h"

#include <unordered_map>

namespace gloo {

namespace {

// Creating NCCL communicators is expensive. So we cache and reuse them.
static std::shared_ptr<NCCLCommList> getCachedCommList(
    const std::shared_ptr<Context>& context,
    const std::vector<int> localDevices)
{
  // per-process cache of communicators
  static std::unordered_map<std::string, std::shared_ptr<NCCLCommList> >
    commLists;

  // generate key
  const int numDevices = localDevices.size();
  std::string key = std::to_string(context->size) + ' ' +
    std::to_string(context->rank);
  for (auto i = 0; i < numDevices; ++i) {
    key += ' ' + std::to_string(localDevices[i]);
  }

  // globally lock here for 2 reasons:
  // 1. Only one set of communicators should be created, make sure we
  //    don't get multiple threads trying to do it at the same time.
  // 2. Prevent concurrent NCCL initialisations - shouldn't be an issue,
  //    but might as well be cautious.
  {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());

    if (!commLists[key]) {
      commLists[key] = std::make_shared<NCCLCommList>(context, localDevices);
    }
  }

  const auto commList = commLists[key];
  GLOO_ENFORCE_NE(commList.get(), (void*)nullptr);
  return commList;
}


// Cache streams across algorithms unless user passes streams
//  - this enforces an ordering on NCCL calls from within gloo
//    and should reduce possbility of deadlocks
static std::shared_ptr<NCCLStreamList> getCachedStreamList(
    const std::shared_ptr<Context>& context,
    const std::vector<int> localDevices) {
  // per-process cache of streams
  static std::unordered_map<std::string, std::shared_ptr<NCCLStreamList> >
    streamLists;

  // generate key
  const int numDevices = localDevices.size();
  std::string key = std::to_string(context->size) + ' ' +
    std::to_string(context->rank);
  for (auto i = 0; i < numDevices; ++i) {
    key += ' ' + std::to_string(localDevices[i]);
  }

  // globally lock here
  // Only one set of streams should be created, make sure we
  // don't get multiple threads trying to do it at the same time.
  {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());

    if (!streamLists[key]) {
      streamLists[key] = std::make_shared<NCCLStreamList>(context, localDevices);
    }
  }

  const auto streamList = streamLists[key];
  GLOO_ENFORCE_NE(streamList.get(), (void*)nullptr);
  return streamList;
}

} // namespace

// Create a set of NCCL communicators for caching.
// NB. This constructor will only ever be called from within a mutex,
//     so no need to lock inside for any reason.
NCCLCommList::NCCLCommList(const std::shared_ptr<Context>& context,
    const std::vector<int> localDevices) {
  // generate unique ID on root node
  ncclUniqueId *id = new ncclUniqueId;
  std::vector<char*> ids;
  ids.push_back(id->internal);
  if (context->rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(id));
  }

  // broadcast ID to other nodes
  BroadcastOneToAll<char>(context, ids, NCCL_UNIQUE_ID_BYTES).run();

  // create comms
  // FIXME currently, we assume all ranks use the same number of devices
  const int numDevices = localDevices.size();
	// num_ranks * num_devices_per_rank
  const int ncclSize = context->size * numDevices;
  // rank_id * num_devices_per_rank
  const int ncclRankStart = context->rank * numDevices;

  comms.reserve(numDevices);
  {
    NCCL_CHECK(ncclGroupStart());
    for (auto i = 0; i < numDevices; ++i) {
      NCCL_CHECK(ncclCommInitRank(&comms[i], ncclSize, *id,
                 ncclRankStart + i));
    }
    NCCL_CHECK(ncclGroupEnd());
  }
}

NCCLCommList::~NCCLCommList() {
  for (auto i = 0; i < comms.size(); ++i) {
    // Shouldn't be necessary, perhaps overly cautious
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    ncclCommDestroy(comms[i]);
  }
}

// NB. Only called from within a mutexed region
NCCLStreamList::NCCLStreamList(const std::shared_ptr<Context>& context,
                               const std::vector<int> localDevices) {
  const int size = localDevices.size();

  streams.clear();

  for (auto i=0; i<size; ++i) {
    streams.push_back(CudaStream(localDevices[i]));
  }
}

template <typename T>
CudaAllreduceNccl2<T>::CudaAllreduceNccl2(
  const std::shared_ptr<Context>& context,
  const std::vector<T*>& ptrs,
  const int count,
  const std::vector<cudaStream_t>& streams)
    : Algorithm(context),
      count_(count),
      bytes_(count_ * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0),
      fn_(CudaReductionFunction<T>::sum) {

  // translate input pointers into gloo structures
  std::vector<int> localDevices(ptrs.size());
  for (auto i = 0; i < ptrs.size(); i++) {
    auto ptr = CudaDevicePointer<T>::create(ptrs[i], count_);
    devicePtrs_.push_back(std::move(ptr));
		localDevices[i] = devicePtrs_[i].getDeviceID();
  }

  // get / initialize cached communicators (thread-safe)
	commList_ = getCachedCommList(context, localDevices);

  // setup what streams we're going to use.
  // Honour what the user passes, or generate some streams that we'll create and cache;
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }
  streams_.reserve(ptrs.size());

  if (!newStream) {
    for (auto i=0; i< devicePtrs_.size(); ++i) {
      streams_.push_back(CudaStream(devicePtrs_[i].getDeviceID(), streams[i]));
    }
  } else {
    // get / initialize cached streams (thread-safe)
    streamList_ = getCachedStreamList(context, localDevices);

    // convert cached streams to local streams
    for (auto i=0; i < devicePtrs_.size(); ++i) {
      streams_.push_back(CudaStream(devicePtrs_[i].getDeviceID(), *streamList_->streams[i]));
    }
  }

  // TODO: necessary?
  barrier_.reset(new gloo::BarrierAllToAll(context));
}

template <typename T>
void CudaAllreduceNccl2<T>::run() {
    // abundance of caution again
    barrier_->run();
    {
      // Need to lock out both other NCCL calls and anything that could
      // cause deadlocks.
      std::lock_guard<std::mutex> lock(CudaShared::getMutex());
      NCCL_CHECK(ncclGroupStart());
      for (int i=0; i<devicePtrs_.size(); i++) {
        NCCL_CHECK(ncclAllReduce(
              (const void*)(*devicePtrs_[i]), (void*)(*devicePtrs_[i]),
              count_, nccl::ncclTypeWrapper<T>::type, ncclSum, commList_->comms[i],
              *streams_[i]));
      }
      NCCL_CHECK(ncclGroupEnd());
    }

  // Now that we've returned from NCCL, perform stream syncs
  for (int i=0; i<devicePtrs_.size(); i++) {
    CUDA_CHECK(cudaStreamSynchronize(*streams_[i]));
  }
}

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T)                                         \
template class CudaAllreduceNccl2<T>;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
