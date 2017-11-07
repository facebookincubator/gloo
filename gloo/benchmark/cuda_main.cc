/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <map>
#include <memory>

#include "gloo/allreduce_builder.h"
#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/runner.h"
#include "gloo/broadcast_builder.h"
#include "gloo/common/logging.h"
#include "gloo/cuda_broadcast_one_to_all.h"
#include "gloo/cuda_private.h"

using namespace gloo;
using namespace gloo::benchmark;

namespace {

int cudaNumDevices() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  return n;
}

template <typename T>
class CudaBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;

 public:
  virtual ~CudaBenchmark() {}

 protected:
  virtual std::vector<T*> allocate(int inputs, int elements) override {
    GLOO_ENFORCE_LE(inputs, cudaNumDevices());
    std::vector<T*> ptrs;

    const auto stride = this->context_->size * inputs;
    for (auto i = 0; i < inputs; i++) {
      CudaDeviceScope scope(i);
      auto cudaMemory = CudaMemory<T>(elements);
      cudaMemory.set((this->context_->rank * inputs) + i, stride);
      ptrs.push_back(*cudaMemory);
      inputs_.push_back(std::move(cudaMemory));
    }
    return ptrs;
  }

  std::vector<CudaMemory<T>> inputs_;
};

template <typename T>
class CudaAllreduceBenchmark : public CudaBenchmark<T> {
 public:
  CudaAllreduceBenchmark(
    std::shared_ptr<::gloo::Context>& context,
    struct options& options,
    ::gloo::AllreduceBuilder<T> builder)
      : CudaBenchmark<T>(context, options),
        builder_(builder) {
  }

  virtual void initialize(int elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    this->algorithm_ = builder_.
      setInputs(ptrs).
      setCount(elements).
      getAlgorithm(this->context_);
  }

  virtual void verify() override {
    // Size is the total number of pointers across the context
    const auto size = this->context_->size * this->inputs_.size();
    // Expected is set to the expected value at ptr[0]
    const auto expected = (size * (size - 1)) / 2;
    // The stride between values at subsequent indices is equal to
    // "size", and we have "size" of them. Therefore, after
    // allreduce, the stride between expected values is "size^2".
    const auto stride = size * size;
    for (const auto& input : this->inputs_) {
      auto ptr = input.copyToHost();
      for (int i = 0; i < input.elements; i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(T(offset + expected), ptr[i], "Mismatch at index: ", i);
      }
    }
  }

 protected:
  ::gloo::AllreduceBuilder<T> builder_;
};

template <typename T>
class CudaBroadcastOneToAllBenchmark : public CudaBenchmark<T> {
  using CudaBenchmark<T>::CudaBenchmark;
 public:
   CudaBroadcastOneToAllBenchmark(
     std::shared_ptr<::gloo::Context>& context,
     struct options& options,
     ::gloo::BroadcastBuilder<T> builder)
       : CudaBenchmark<T>(context, options),
         builder_(builder) {
   }

 public:
  virtual void initialize(int elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    this->algorithm_ = builder_.
      setInputs(ptrs).
      setCount(elements).
      setRootRank(rootRank_).
      setRootPointerRank(rootPointerRank_).
      setStreams({}).
      getAlgorithm(this->context_);
    }

  virtual void verify() override {
    const auto rootOffset = rootRank_ * this->inputs_.size() + rootPointerRank_;
    const auto stride = this->context_->size * this->inputs_.size();
    for (const auto& input : this->inputs_) {
      auto ptr = input.copyToHost();
      for (int i = 0; i < input.elements; i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(
            T(rootOffset + offset), ptr[i], "Mismatch at index: ", i);
      }
    }
  }

 protected:
  const int rootRank_ = 0;
  const int rootPointerRank_ = 0;
  ::gloo::BroadcastBuilder<T> builder_;
};

} // namespace

template <class TC>
bool beginsWith(const TC& input, const TC& match) {
  return input.size() >= match.size()
    && equal(match.begin(), match.end(), input.begin());
}

template <typename T>
void runBenchmark(options& x) {
  Runner::BenchmarkFn<T> fn;

  if (x.benchmark == "cuda_broadcast_one_to_all") {
    auto builder = gloo::BroadcastBuilder<T>();
    if (x.gpuDirect) {
      builder.setGPUDirect(true);
    }

    fn = [&, builder](std::shared_ptr<Context>& context) {
      return std::unique_ptr<Benchmark<T>>(
        new CudaBroadcastOneToAllBenchmark<T>(context, x, builder));
    };
  } else if (beginsWith(x.benchmark, std::string("cuda_allreduce_"))) {
    auto builder = gloo::AllreduceBuilder<T>();
    if (x.gpuDirect) {
      builder.setGPUDirect(true);
    }
    if (x.benchmark == "cuda_allreduce_halving_doubling") {
      builder.setImplementation(
        gloo::AllreduceBuilder<T>::HalvingDoubling);
    } else if (x.benchmark == "cuda_allreduce_halving_doubling_pipelined") {
      builder.setImplementation(
        gloo::AllreduceBuilder<T>::HalvingDoublingPipelined);
    } else if (x.benchmark == "cuda_allreduce_ring") {
      builder.setImplementation(
        gloo::AllreduceBuilder<T>::Ring);
    } else if (x.benchmark == "cuda_allreduce_ring_chunked") {
      builder.setImplementation(
        gloo::AllreduceBuilder<T>::RingChunked);
    } else {
      GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);
    }
    fn = [&, builder](std::shared_ptr<Context>& context) {
      return std::unique_ptr<Benchmark<T>>(
        new CudaAllreduceBenchmark<T>(context, x, builder));
    };
  }

  if (!fn) {
    GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);
  }

  Runner r(x);
  r.run(fn);
}

int main(int argc, char** argv) {
  auto x = benchmark::parseOptions(argc, argv);
  if (x.halfPrecision) {
    runBenchmark<float16>(x);
  } else {
    runBenchmark<float>(x);
  }
  return 0;
}
