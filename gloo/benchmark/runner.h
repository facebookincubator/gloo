/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "gloo/algorithm.h"
#include "gloo/barrier.h"
#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/options.h"
#include "gloo/benchmark/timer.h"
#include "gloo/config.h"
#include "gloo/rendezvous/context.h"
#include "gloo/transport/device.h"

namespace gloo {
namespace benchmark {

// Forward declaration
class RunnerThread;

// RunnerJob holds the state associated with repetetive calls of an arbitrary
// function (which is typically equal to the benchmark function).
class RunnerJob {
 public:
  explicit RunnerJob(std::function<void()> fn, int i) :
    done_(false), fn_(fn), iterations_(i) {}

  const Samples& getSamples() const {
    return samples_;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!done_) {
      cond_.wait(lock);
    }
  }

 protected:
  void done() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_ = true;
    cond_.notify_all();
  }

  bool done_;
  std::mutex mutex_;
  std::condition_variable cond_;

  std::function<void()> fn_;
  int iterations_;
  Samples samples_;

  friend class RunnerThread;
};

// RunnerThread takes a RunnerJob and runs the function a number of times,
// keeping track of its runtime in the job's member variable `samples_`. Upon
// completion, it signals the job using the `done()` function.
class RunnerThread {
 public:
  RunnerThread();
  ~RunnerThread();

  void run(RunnerJob* job);

 protected:
  void spawn();

  bool stop_;
  RunnerJob* job_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::thread thread_;
};

class Runner {
 public:
  template <typename T>
  using BenchmarkFn = std::function<std::unique_ptr<Benchmark<T>>(
      std::shared_ptr<::gloo::Context>&)>;

  explicit Runner(const options& options);
  ~Runner();

  template <typename T>
  void run(BenchmarkFn<T>& fn);

  template <typename T>
  void run(BenchmarkFn<T>& fn, int n);

 protected:
#if GLOO_USE_REDIS
  void rendezvousRedis();
#endif

#if GLOO_USE_MPI
  void rendezvousMPI();
#endif

  void rendezvousFileSystem();

  long broadcast(long value);

  std::shared_ptr<Context> newContext();

  void printHeader();
  void printDistribution(
      int elements,
      int elementSize,
      const Distribution& samples);

  options options_;
  std::vector<std::shared_ptr<transport::Device>> transportDevices_;
  std::shared_ptr<rendezvous::ContextFactory> contextFactory_;
  std::vector<std::unique_ptr<RunnerThread>> threads_;

  long broadcastValue_;
  std::unique_ptr<Algorithm> broadcast_;
  std::unique_ptr<Barrier> barrier_;
};

} // namespace benchmark
} // namespace gloo
