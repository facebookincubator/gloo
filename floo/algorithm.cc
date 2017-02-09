/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "floo/algorithm.h"

#include "floo/common/logging.h"

namespace floo {

Algorithm::Algorithm(const std::shared_ptr<Context>& context)
    : context_(context),
      contextRank_(context_->rank_),
      contextSize_(context_->size_) {}

// Have to provide implementation for pure virtual destructor.
Algorithm::~Algorithm() {}

std::unique_ptr<transport::Pair>& Algorithm::getPair(int i) {
  return context_->getPair(i);
}

// Helper for ring algorithms
std::unique_ptr<transport::Pair>& Algorithm::getLeftPair() {
  auto rank = (context_->size_ + context_->rank_ - 1) % context_->size_;
  FLOO_ENFORCE(context_->getPair(rank), "pair missing (index ", rank, ")");
  return context_->getPair(rank);
}

// Helper for ring algorithms
std::unique_ptr<transport::Pair>& Algorithm::getRightPair() {
  auto rank = (context_->rank_ + 1) % context_->size_;
  FLOO_ENFORCE(context_->getPair(rank), "pair missing (index ", rank, ")");
  return context_->getPair(rank);
}

} // namespace floo
