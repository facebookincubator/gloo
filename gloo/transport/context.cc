/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/transport/context.h"

namespace gloo {
namespace transport {

Context::Context(int rank, int size) : rank(rank), size(size) {
  pairs_.resize(size);
}

// Have to provide implementation for pure virtual destructor.
Context::~Context() {}

std::unique_ptr<transport::Pair>& Context::getPair(int rank) {
  return pairs_.at(rank);
}

Context::LazyTally::LazyTally(std::vector<Tally>& vec, slot_t slot)
    : vec_(vec), slot_(slot), initialized_(false) {}

Context::LazyTally::~LazyTally() {
  // Remove empty tally from vector.
  if (initialized_ && it_ != vec_.end() && it_->empty()) {
    vec_.erase(it_);
  }
}

bool Context::LazyTally::exists() {
  initialize_iterator();
  return it_ != vec_.end();
}

Context::Tally& Context::LazyTally::get() {
  initialize_iterator();
  if (it_ == vec_.end()) {
    vec_.emplace_back(slot_);
    it_ = (vec_.end() - 1);
  }
  return *it_;
}

void Context::LazyTally::initialize_iterator() {
  if (initialized_) {
    return;
  }

  it_ =
      std::find_if(vec_.begin(), vec_.end(), [this](const Context::Tally& op) {
        return op.slot == slot_;
      });
  initialized_ = true;
}

Context::Mutator::Mutator(Context& context, slot_t slot, rank_t rank)
    : lock_(context.mutex_),
      context_(context),
      slot_(slot),
      rank_(rank),
      pendingOperations_(context_.pendingOperations_, slot_),
      expectedNotifications_(context_.expectedNotifications_, slot_) {}

void Context::Mutator::pushRemotePendingRecv() {
  pendingOperations_.get().pushRecv(rank_);
}

void Context::Mutator::pushRemotePendingSend() {
  pendingOperations_.get().pushSend(rank_);
}

bool Context::Mutator::shiftRemotePendingRecv() {
  if (!pendingOperations_.exists()) {
    return false;
  }
  return pendingOperations_.get().shiftRecv(rank_);
}

bool Context::Mutator::shiftRemotePendingSend() {
  if (!pendingOperations_.exists()) {
    return false;
  }
  return pendingOperations_.get().shiftSend(rank_);
}

void Context::Mutator::pushExpectedSendNotification() {
  expectedNotifications_.get().pushSend(rank_);
}

bool Context::Mutator::shiftExpectedSendNotification() {
  if (!expectedNotifications_.exists()) {
    return false;
  }
  return expectedNotifications_.get().shiftSend(rank_);
}

std::vector<Context::Tally>::iterator Context::findPendingOperations(
    slot_t slot) {
  return std::find_if(
      pendingOperations_.begin(),
      pendingOperations_.end(),
      [slot](const Tally& op) { return op.slot == slot; });
}

} // namespace transport
} // namespace gloo
