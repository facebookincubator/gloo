/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/cuda_private.h"
#include "gloo/test/base_test.h"


namespace gloo {
namespace test {

void cudaSleep(cudaStream_t stream, size_t clocks);

int cudaNumDevices();

class CudaBaseTest : public BaseTest {};

template <typename T>
class CudaFixture : public Fixture<T> {
 public:
  CudaFixture(const std::shared_ptr<Context> context, int ptrs, int count)
      : Fixture<T>(context, ptrs, count) {
    for (int i = 0; i < ptrs; i++) {
      CudaDeviceScope scope(i % cudaNumDevices());
      cudaSrcs.push_back(CudaMemory<T>(count));
      cudaPtrs.push_back(
        CudaDevicePointer<T>::create(*cudaSrcs.back(), count));
      cudaStreams.push_back(CudaStream(i));
    }
  }

  // Upon returning, the memory allocated by this fixture is set. The
  // fixture data is visible to any kernel launched on any stream.
  void assignValues() {
    Fixture<T>::assignValues();
    for (auto i = 0; i < cudaSrcs.size(); i++) {
      CudaDeviceScope scope(cudaStreams[i].getDeviceID());
      CUDA_CHECK(cudaMemcpyAsync(
        *cudaSrcs[i],
        this->srcs[i].get(),
        cudaSrcs[i].bytes,
        cudaMemcpyHostToDevice,
        *cudaStreams[i]));
    }
    // Synchronize every stream to ensure the memory copies have completed.
    for (auto i = 0; i < cudaSrcs.size(); i++) {
      CudaDeviceScope scope(cudaStreams[i].getDeviceID());
      CUDA_CHECK(cudaStreamSynchronize(*cudaStreams[i]));
    }
  }

  // Upon returning, the memory allocated by this fixture is being
  // set. The fixture data will be visible to any kernel launched on
  // the same stream as the memory is being set on. Otherwise,
  // additional synchronization is required.
  void assignValuesAsync() {
    Fixture<T>::assignValues();
    for (auto i = 0; i < cudaSrcs.size(); i++) {
      CudaDeviceScope scope(cudaStreams[i].getDeviceID());
      // Insert sleep on stream to force to artificially delay the
      // kernel that actually populates the memory to surface
      // synchronization errors.
      cudaSleep(*cudaStreams[i], 100000);
      CUDA_CHECK(cudaMemcpyAsync(
        *cudaSrcs[i],
        this->srcs[i].get(),
        cudaSrcs[i].bytes,
        cudaMemcpyHostToDevice,
        *cudaStreams[i]));
    }
  }

  std::vector<T*> getCudaPointers() const {
    std::vector<T*> out;
    for (const auto& ptr : cudaPtrs) {
      out.push_back(*ptr);
    }
    return out;
  }

  std::vector<cudaStream_t> getCudaStreams() const {
    std::vector<cudaStream_t> out;
    for (const auto& stream : cudaStreams) {
      out.push_back(stream.getStream());
    }
    return out;
  }

  void copyToHost() {
    for (auto i = 0; i < cudaSrcs.size(); i++) {
      CUDA_CHECK(cudaMemcpyAsync(
        this->srcs[i].get(),
        *cudaSrcs[i],
        cudaSrcs[i].bytes,
        cudaMemcpyDeviceToHost,
        *cudaStreams[i]));
    }
    synchronizeCudaStreams();
  }

  void synchronizeCudaStreams() {
    for (const auto& stream : cudaStreams) {
      CudaDeviceScope scope(stream.getDeviceID());
      CUDA_CHECK(cudaStreamSynchronize(stream.getStream()));
    }
  }

  std::vector<CudaMemory<T>> cudaSrcs;
  std::vector<CudaDevicePointer<T>> cudaPtrs;
  std::vector<CudaStream> cudaStreams;
};

} // namespace test
} // namespace gloo
