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

#pragma once

#include "gloo/algorithm.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/cuda.h"
#include "gloo/cuda_workspace.h"
#include "gloo/nccl/nccl.h"

#include <unordered_map>

namespace gloo {

class NCCLCommList {
 public:
  NCCLCommList(const std::shared_ptr<Context>& context,
      const std::vector<int> localDevices);
  ~NCCLCommList();
  std::vector<ncclComm_t> comms;
};

class NCCLStreamList {
 public:
  NCCLStreamList(const std::shared_ptr<Context>& context,
      const std::vector<int> localDevices);

  std::vector<CudaStream> streams;
};

template <typename T>
class CudaAllreduceNccl2 : public Algorithm {
 public:
  CudaAllreduceNccl2(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<cudaStream_t>& streams = std::vector<cudaStream_t>());

  virtual void run() override;

 protected:
  std::vector<CudaDevicePointer<T> > devicePtrs_;
  std::vector<CudaStream> streams_;
  CudaStream* scratchStream_;

  std::shared_ptr<NCCLCommList> commList_;
  std::shared_ptr<NCCLStreamList> streamList_;

  const int count_;
  const int bytes_;
  const bool synchronizeDeviceOutputs_;
  const CudaReductionFunction<T>* fn_;

  std::unique_ptr<LocalOp<T> > localReduceOp_;
  std::unique_ptr<LocalOp<T> > localBroadcastOp_;

  std::unique_ptr<transport::Buffer> sendDataBuf_;
  std::unique_ptr<transport::Buffer> recvDataBuf_;

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;

  int rank_;
  std::unique_ptr<gloo::BarrierAllToAll> barrier_;
};

} // namespace gloo
