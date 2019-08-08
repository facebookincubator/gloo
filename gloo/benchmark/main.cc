/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>

#include "gloo/allgather_ring.h"
#include "gloo/allreduce.h"
#include "gloo/allreduce_halving_doubling.h"
#include "gloo/allreduce_bcube.h"
#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/barrier_all_to_one.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/pairwise_exchange.h"
#include "gloo/reduce_scatter.h"
#include "gloo/common/aligned_allocator.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/context.h"
#include "gloo/types.h"

#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/runner.h"

using namespace gloo;
using namespace gloo::benchmark;

namespace {

template <typename T>
class AllgatherBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    auto inPtrs = this->allocate(this->options_.inputs, elements);
    GLOO_ENFORCE_EQ(inPtrs.size(), this->options_.inputs);
    outputs_.resize(this->options_.inputs * this->context_->size * elements);
    this->algorithm_.reset(new AllgatherRing<T>(
        this->context_, this->getInputs(), outputs_.data(), elements));
  }

  void verify() override {
    const auto stride = this->context_->size * this->inputs_.size();
    const auto elements = this->inputs_[0].size();
    for (int rank = 0; rank < this->context_->size; rank++) {
      auto val = rank * this->inputs_.size();
      for (int elem = 0; elem < elements; elem++) {
        T exp(elem * stride + val);
        for (int input = 0; input < this->inputs_.size(); input++) {
          const auto rankOffset = rank * elements * this->inputs_.size();
          const auto inputOffset = input * elements;
          GLOO_ENFORCE_EQ(
            outputs_[rankOffset + inputOffset + elem], exp + T(input),
            "Mismatch at index: [", rank, ", ", input, ", ", elem, "]");
        }
      }
    }
  }

 protected:
  std::vector<T> outputs_;
};

template <class A, typename T>
class AllreduceBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    this->algorithm_.reset(new A(this->context_, ptrs, elements));
  }

  void verify() override {
    // Size is the total number of pointers across the context
    const auto size = this->context_->size * this->inputs_.size();
    // Expected is set to the expected value at ptr[0]
    const auto expected = (size * (size - 1)) / 2;
    // The stride between values at subsequent indices is equal to
    // "size", and we have "size" of them. Therefore, after
    // allreduce, the stride between expected values is "size^2".
    const auto stride = size * size;
    for (const auto& input : this->inputs_) {
      for (int i = 0; i < input.size(); i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(
            T(offset + expected), input[i], "Mismatch at index: ", i);
      }
    }
  }
};

template <typename T>
class BarrierAllToAllBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t /* unused */) override {
    this->algorithm_.reset(new BarrierAllToAll(this->context_));
  }
};

template <typename T>
class BarrierAllToOneBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t /* unused */) override {
    // This tool measures at rank=0, so use root=1 for the all to one
    // barrier to measure the end-to-end latency (otherwise we might
    // not account for the send-to-root part of the algorithm).
    this->algorithm_.reset(new BarrierAllToOne(this->context_, 1));
  }
};

template <typename T>
class BroadcastOneToAllBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    this->algorithm_.reset(
        new BroadcastOneToAll<T>(this->context_, ptrs, elements, rootRank_));
  }

  void verify() override {
    const auto stride = this->context_->size * this->inputs_.size();
    for (const auto& input : this->inputs_) {
      for (int i = 0; i < input.size(); i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(
            T(offset + rootRank_), input[i], "Mismatch at index: ", i);
      }
    }
  }

 protected:
  const int rootRank_ = 0;
};

template <typename T>
class PairwiseExchangeBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    this->algorithm_.reset(new PairwiseExchange(
        this->context_, elements, this->getOptions().destinations));
  }
};

template <typename T>
class ReduceScatterBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    auto rem = elements;
    auto chunkSize =
        (elements + this->context_->size - 1) / this->context_->size;
    for (int i = 0; i < this->context_->size; ++i) {
      recvCounts_.push_back(std::min(chunkSize, rem));
      rem = rem > chunkSize ? rem - chunkSize : 0;
    }
    this->algorithm_.reset(
        new ReduceScatterHalvingDoubling<T>(
            this->context_, ptrs, elements, recvCounts_));
  }

  void verify() override {
    // Size is the total number of pointers across the context
    const auto size = this->context_->size * this->inputs_.size();
    // Expected is set to the expected value at ptr[0]
    const auto expected = (size * (size - 1)) / 2;
    // The stride between values at subsequent indices is equal to
    // "size", and we have "size" of them. Therefore, after
    // reduce-scatter, the stride between expected values is "size^2".
    const auto stride = size * size;
    for (const auto& input : this->inputs_) {
      int numElemsSoFar = 0;
      for (int i = 0; i < this->context_->rank; ++i) {
          numElemsSoFar += recvCounts_[i];
      }
      for (int i = 0; i < recvCounts_[this->context_->rank]; ++i) {
        auto offset = (numElemsSoFar + i) * stride;
        GLOO_ENFORCE_EQ(
            T(offset + expected), input[i], "Mismatch at index: ", i);
      }
    }
  }

 protected:
   std::vector<int> recvCounts_;
};

} // namespace

// Namespace for the new style algorithm benchmarks.
namespace {

template <typename T>
class NewAllreduceBenchmark : public Benchmark<T> {
  using allocation = std::vector<std::vector<T, aligned_allocator<T, kBufferAlignment>>>;

 public:
  NewAllreduceBenchmark(
    std::shared_ptr<::gloo::Context>& context,
    struct options& options)
      : Benchmark<T>(context, options),
        opts_(context) {}

  allocation newAllocation(int inputs, size_t elements) {
    allocation out;
    out.reserve(inputs);
    for (size_t i = 0; i < inputs; i++) {
      out.emplace_back(elements);
    }
    return out;
  }

  void initialize(size_t elements) override {
    inputAllocation_ = newAllocation(this->options_.inputs, elements);
    outputAllocation_ = newAllocation(this->options_.inputs, elements);

    // Stride between successive values in any input.
    const auto stride = this->context_->size * this->options_.inputs;
    for (size_t i = 0; i < this->options_.inputs; i++) {
      // Different for every input at every node. This means all
      // values across all inputs and all nodes are different and we
      // can accurately detect correctness errors.
      const auto value = (this->context_->rank * this->options_.inputs) + i;
      for (size_t j = 0; j < elements; j++) {
        inputAllocation_[i][j] = (j * stride) + value;
      }
    }

    // Generate vectors with pointers to populate the options struct.
    std::vector<T*> inputPointers;
    std::vector<T*> outputPointers;
    for (size_t i = 0; i < this->options_.inputs; i++) {
      inputPointers.push_back(inputAllocation_[i].data());
      outputPointers.push_back(outputAllocation_[i].data());
    }

    // Configure AllreduceOptions struct
    opts_.setInputs(inputPointers, elements);
    opts_.setOutputs(outputPointers, elements);
    opts_.setAlgorithm(AllreduceOptions::Algorithm::RING);
    void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;
    opts_.setReduceFunction(fn);
  }

  void run() override {
    allreduce(opts_);
  }

 private:
  AllreduceOptions opts_;

  allocation inputAllocation_;
  allocation outputAllocation_;
};

}

#define RUN_BENCHMARK(T)                                                   \
  Runner::BenchmarkFn<T> fn;                                               \
  if (x.benchmark == "allgather_ring") {                                   \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<AllgatherBenchmark<T>>(context, x);         \
    };                                                                     \
  } else if (x.benchmark == "allreduce_ring") {                            \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<AllreduceBenchmark<AllreduceRing<T>, T>>(   \
          context, x);                                                     \
    };                                                                     \
  } else if (x.benchmark == "allreduce_ring_chunked") {                    \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<                                            \
          AllreduceBenchmark<AllreduceRingChunked<T>, T>>(context, x);     \
    };                                                                     \
  } else if (x.benchmark == "allreduce_halving_doubling") {                \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<                                            \
          AllreduceBenchmark<AllreduceHalvingDoubling<T>, T>>(context, x); \
    };                                                                     \
  } else if (x.benchmark == "allreduce_bcube") {                           \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<                                            \
          AllreduceBenchmark<AllreduceBcube<T>, T>>(context, x);           \
    };                                                                     \
  } else if (x.benchmark == "barrier_all_to_all") {                        \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<BarrierAllToAllBenchmark<T>>(context, x);   \
    };                                                                     \
  } else if (x.benchmark == "barrier_all_to_one") {                        \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<BarrierAllToOneBenchmark<T>>(context, x);   \
    };                                                                     \
  } else if (x.benchmark == "broadcast_one_to_all") {                      \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<BroadcastOneToAllBenchmark<T>>(context, x); \
    };                                                                     \
  } else if (x.benchmark == "pairwise_exchange") {                         \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<PairwiseExchangeBenchmark<T>>(context, x);  \
    };                                                                     \
  } else if (x.benchmark == "reduce_scatter") {                            \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<ReduceScatterBenchmark<T>>(context, x);  \
    };                                                                     \
  }                                                                        \
  if (!fn) {                                                               \
    GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);               \
  }                                                                        \
  Runner r(x);                                                             \
  r.run(fn);

template <typename T>
void runNewBenchmark(options& options) {
  Runner::BenchmarkFn<T> fn;

  const auto name = options.benchmark.substr(4);
  if (name == "allreduce_ring") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<NewAllreduceBenchmark<T>>(context, options);
    };
  } else {
    GLOO_ENFORCE(false, "Invalid benchmark name: ", options.benchmark);
  }

  Runner runner(options);
  runner.run(fn);
}

int main(int argc, char** argv) {
  auto x = benchmark::parseOptions(argc, argv);

  // Run new style benchmarks if the benchmark name starts with "new_".
  // Eventually we'd like to deprecate all the old style ones...
  if (x.benchmark.substr(0, 4) == "new_") {
    runNewBenchmark<float>(x);
    return 0;
  }

  if (x.benchmark == "pairwise_exchange") {
    RUN_BENCHMARK(char);
  } else if (x.halfPrecision) {
    RUN_BENCHMARK(float16);
  } else {
    RUN_BENCHMARK(float);
  }
  return 0;
}
