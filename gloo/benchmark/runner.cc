/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "runner.h"

#include <iomanip>
#include <iostream>

#include "gloo/barrier_all_to_one.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/transport/device.h"

#if GLOO_USE_REDIS
#include "gloo/rendezvous/redis_store.h"
#endif

#if GLOO_USE_MPI
#include "gloo/mpi/context.h"
#endif

#if BENCHMARK_TCP
#include "gloo/transport/tcp/device.h"
#endif

#if BENCHMARK_IBVERBS
#include "gloo/transport/ibverbs/device.h"
#endif

namespace gloo {
namespace benchmark {

Runner::Runner(const options& options) : options_(options) {
#ifdef BENCHMARK_TCP
  if (options_.transport == "tcp") {
    if (options_.tcpDevice.empty()) {
      transport::tcp::attr attr;
      transportDevices_.push_back(transport::tcp::CreateDevice(attr));
    } else {
      for (const auto& name : options_.tcpDevice) {
        transport::tcp::attr attr;
        attr.iface = name;
        transportDevices_.push_back(transport::tcp::CreateDevice(attr));
      }
    }
  }
#endif
#ifdef BENCHMARK_IBVERBS
  if (options_.transport == "ibverbs") {
    if (options_.ibverbsDevice.empty()) {
      transport::ibverbs::attr attr;
      attr.port = options_.ibverbsPort;
      attr.index = options_.ibverbsIndex;
      transportDevices_.push_back(transport::ibverbs::CreateDevice(attr));
    } else {
      for (const auto& name : options_.ibverbsDevice) {
        transport::ibverbs::attr attr;
        attr.name = name;
        attr.port = options_.ibverbsPort;
        attr.index = options_.ibverbsIndex;
        transportDevices_.push_back(transport::ibverbs::CreateDevice(attr));
      }
    }
  }
#endif

  GLOO_ENFORCE(
      !transportDevices_.empty(),
      "Unknown transport: ",
      options_.transport);

  // Spawn threads that run the actual benchmark loop
  for (auto i = 0; i < options_.threads; i++) {
    threads_.push_back(make_unique<RunnerThread>());
  }

#if GLOO_USE_REDIS
  if (!contextFactory_) {
    rendezvousRedis();
  }
#endif

#if GLOO_USE_MPI
  if (!contextFactory_) {
    rendezvousMPI();
  }
#endif

  if (!contextFactory_) {
    rendezvousFileSystem();
  }

  GLOO_ENFORCE(contextFactory_, "No means for rendezvous");

  // Create broadcast algorithm to synchronize between participants
  broadcast_.reset(
    new BroadcastOneToAll<long>(newContext(), {&broadcastValue_}, 1));

  // Create barrier for run-to-run synchronization
  barrier_.reset(new BarrierAllToOne(newContext()));
}

Runner::~Runner() {
  // Reset algorithms and context factory such that all
  // shared_ptr's to contexts are destructed.
  // This is necessary so that all MPI common worlds are
  // destroyed before MPI_Finalize is called.
  barrier_.reset();
  broadcast_.reset();
  contextFactory_.reset();

#if GLOO_USE_MPI
  if (options_.mpi) {
    MPI_Finalize();
  }
#endif
}

#if GLOO_USE_REDIS
void Runner::rendezvousRedis() {
  // Don't rendezvous through Redis if the host is not set
  if (options_.redisHost.empty()) {
    return;
  }

  rendezvous::RedisStore redisStore(options_.redisHost, options_.redisPort);
  rendezvous::PrefixStore prefixStore(options_.prefix, redisStore);
  auto backingContext = std::make_shared<rendezvous::Context>(
      options_.contextRank, options_.contextSize);
  backingContext->connectFullMesh(prefixStore, transportDevices_.front());
  contextFactory_ = std::make_shared<rendezvous::ContextFactory>(
      backingContext);
}
#endif

#if GLOO_USE_MPI
void Runner::rendezvousMPI() {
  // Don't rendezvous using MPI if not started through mpirun
  if (!options_.mpi) {
    return;
  }

  auto rv = MPI_Init(nullptr, nullptr);
  GLOO_ENFORCE_EQ(rv, MPI_SUCCESS);
  MPI_Comm_rank(MPI_COMM_WORLD, &options_.contextRank);
  MPI_Comm_size(MPI_COMM_WORLD, &options_.contextSize);
  auto backingContext = std::make_shared<::gloo::mpi::Context>(MPI_COMM_WORLD);
  backingContext->connectFullMesh(transportDevices_.front());
  contextFactory_ = std::make_shared<rendezvous::ContextFactory>(
      backingContext);
}
#endif

void Runner::rendezvousFileSystem() {
  // Don't rendezvous using the file system if the shared path is not set
  if (options_.sharedPath.empty()) {
    return;
  }

  rendezvous::FileStore fileStore(options_.sharedPath);
  rendezvous::PrefixStore prefixStore(options_.prefix, fileStore);
  auto backingContext = std::make_shared<rendezvous::Context>(
    options_.contextRank, options_.contextSize);
  backingContext->connectFullMesh(prefixStore, transportDevices_.front());
  contextFactory_ = std::make_shared<rendezvous::ContextFactory>(
    backingContext);
}

long Runner::broadcast(long value) {
  // Set value to broadcast only on root.
  // Otherwise it can race with the actual broadcast
  // operation writing to the same memory location.
  if (options_.contextRank == 0) {
    broadcastValue_ = value;
  }
  broadcast_->run();
  return broadcastValue_;
}

std::shared_ptr<Context> Runner::newContext() {
  auto context = contextFactory_->makeContext(transportDevices_.front());
  return context;
}

template <typename T>
void Runner::run(BenchmarkFn<T>& fn) {
  printHeader();

  if (options_.elements > 0) {
    run(fn, options_.elements);
    return;
  }

  // Run sweep over number of elements
  for (int i = 100; i <= 1000000; i *= 10) {
    std::vector<int> js = {i * 1, i * 2, i * 5};
    for (auto& j : js) {
      run(fn, j);
    }
  }
}

template <typename T>
void Runner::run(BenchmarkFn<T>& fn, int n) {
  std::vector<std::unique_ptr<Benchmark<T>>> benchmarks;

  // Initialize one set of objects for every thread
  for (auto i = 0; i < options_.threads; i++) {
    auto context = contextFactory_->makeContext(
        transportDevices_[i % transportDevices_.size()]);
    context->base = options_.base;
    auto benchmark = fn(context);
    benchmark->initialize(n);

    // Switch pairs to sync mode if configured to do so
    if (options_.sync) {
      for (int j = 0; j < context->size; j++) {
        auto& pair = context->getPair(j);
        if (pair) {
          pair->setSync(true, options_.busyPoll);
        }
      }
    }

    // Verify correctness of initial run
    if (options_.verify) {
      benchmark->run();
      benchmark->verify();
      barrier_->run();
    }

    benchmarks.push_back(std::move(benchmark));
  }

  // Switch mode based on iteration count or time spent
  auto iterations = options_.iterationCount;
  if (iterations <= 0) {
    GLOO_ENFORCE_GT(options_.iterationTimeNanos, 0);

    // Create warmup jobs for every thread
    std::vector<std::unique_ptr<RunnerJob>> jobs;
    for (auto i = 0; i < options_.threads; i++) {
      auto fn = [&benchmark = benchmarks[i]] { benchmark->run(); };
      auto job = make_unique<RunnerJob>(fn, options_.warmupIterationCount);
      jobs.push_back(std::move(job));
    }

    // Start jobs on every thread (synchronized across processes)
    barrier_->run();
    for (auto i = 0; i < options_.threads; i++) {
      threads_[i]->run(jobs[i].get());
    }

    // Wait for completion and merge latency distributions
    Samples samples;
    for (auto i = 0; i < options_.threads; i++) {
      jobs[i]->wait();
      samples.merge(jobs[i]->getSamples());
    }

    // Broadcast duration of median iteration during warmup,
    // so all nodes agree on the number of iterations to run for.
    Distribution warmup(samples);
    auto nanos = broadcast(warmup.percentile(0.5));
    iterations = std::max(1L, options_.iterationTimeNanos / nanos);
  }

  // Create jobs for every thread
  std::vector<std::unique_ptr<RunnerJob>> jobs;
  for (auto i = 0; i < options_.threads; i++) {
    auto fn = [&benchmark = benchmarks[i]] { benchmark->run(); };
    auto job = make_unique<RunnerJob>(fn, iterations);
    jobs.push_back(std::move(job));
  }

  // Start jobs on every thread (synchronized across processes)
  barrier_->run();
  for (auto i = 0; i < options_.threads; i++) {
    threads_[i]->run(jobs[i].get());
  }

  // Wait for completion
  for (auto i = 0; i < options_.threads; i++) {
    jobs[i]->wait();
  }

  // Merge results
  Samples samples;
  for (auto i = 0; i < options_.threads; i++) {
    samples.merge(jobs[i]->getSamples());
  }

  // Print results
  Distribution latency(samples);
  printDistribution(n, sizeof(T), latency);
}

void Runner::printHeader() {
  if (options_.contextRank != 0) {
    return;
  }

  if (transportDevices_.size() == 1) {
    std::cout << std::left << std::setw(13) << "Device:";
    std::cout << transportDevices_.front()->str() << std::endl;
  } else {
    std::cout << std::left << std::setw(13) << "Devices:" << std::endl;
    for (const auto& device : transportDevices_) {
      std::cout << "  - " << device->str() << std::endl;
    }
  }
  std::cout << std::left << std::setw(13) << "Algorithm:";
  std::cout << options_.benchmark << std::endl;

  std::cout << std::left << std::setw(13) << "Options:";
  std::cout << "processes=" << options_.contextSize;
  std::cout << ", inputs=" << options_.inputs;
  std::cout << ", threads=" << options_.threads;
  if (options_.benchmark == "allreduce_bcube") {
    std::cout << ", base=" << options_.base;
  }
  if (options_.benchmark.compare(0, 5, "cuda_") == 0) {
    std::cout << ", gpudirect=";
    if (options_.transport == "ibverbs" && options_.gpuDirect) {
      std::cout << "yes";
    } else {
      std::cout << "no";
    }
  }
  std::cout << std::endl << std::endl;

  std::string suffix = "(us)";
  if (options_.showNanos) {
    suffix = "(ns)";
  }
  std::string bwSuffix = "(GB/s)";

  std::cout << std::right;
  std::cout << std::setw(11) << "elements";
  std::cout << std::setw(11) << ("min " + suffix);
  std::cout << std::setw(11) << ("p50 " + suffix);
  std::cout << std::setw(11) << ("p99 " + suffix);
  std::cout << std::setw(11) << ("max " + suffix);
  std::cout << std::setw(13) << ("avg " + bwSuffix);
  std::cout << std::setw(11) << "samples";
  std::cout << std::endl;
}

void Runner::printDistribution(
    int elements,
    int elementSize,
    const Distribution& latency) {
  if (options_.contextRank != 0) {
    return;
  }

  auto div = 1000;
  if (options_.showNanos) {
    div = 1;
  }

  GLOO_ENFORCE_GE(latency.size(), 1, "No latency samples found");

  auto bytes = elements * elementSize;
  auto totalBytes = bytes * latency.size();
  auto totalNanos = latency.sum() / options_.threads;
  auto totalBytesPerSec = (totalBytes * 1e9f) / totalNanos;
  auto totalGigaBytesPerSec = totalBytesPerSec / (1024 * 1024 * 1024);

  std::cout << std::setw(11) << elements;
  std::cout << std::setw(11) << (latency.min() / div);
  std::cout << std::setw(11) << (latency.percentile(0.50) / div);
  std::cout << std::setw(11) << (latency.percentile(0.99) / div);
  std::cout << std::setw(11) << (latency.max() / div);
  std::cout << std::fixed << std::setprecision(3);
  std::cout << std::setw(13) << totalGigaBytesPerSec;
  std::cout << std::setw(11) << latency.size();
  std::cout << std::endl;
}

template void Runner::run(BenchmarkFn<char>& fn);
template void Runner::run(BenchmarkFn<char>& fn, int n);
template void Runner::run(BenchmarkFn<float>& fn);
template void Runner::run(BenchmarkFn<float>& fn, int n);
template void Runner::run(BenchmarkFn<float16>& fn);
template void Runner::run(BenchmarkFn<float16>& fn, int n);

RunnerThread::RunnerThread() : stop_(false), job_(nullptr) {
  thread_ = std::thread(&RunnerThread::spawn, this);
}

RunnerThread::~RunnerThread() {
  mutex_.lock();
  stop_ = true;
  mutex_.unlock();
  cond_.notify_one();
  thread_.join();
}

void RunnerThread::run(RunnerJob* job) {
  std::unique_lock<std::mutex> lock(mutex_);
  job_ = job;
  cond_.notify_one();
}

void RunnerThread::spawn() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stop_) {
    while (job_ == nullptr) {
      cond_.wait(lock);
      if (stop_) {
        return;
      }
    }

    for (auto i = 0; i < job_->iterations_; i++) {
      Timer dt;
      job_->fn_();
      job_->samples_.add(dt);
    }

    job_->done();
    job_ = nullptr;
  }
}

} // namespace benchmark
} // namespace gloo
