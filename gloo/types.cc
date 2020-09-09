/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/types.h"

#include <stdexcept>
#include <string>

namespace gloo {

Slot Slot::build(uint8_t prefix, uint32_t tag) {
  uint64_t u64prefix = ((uint64_t)prefix) << 56;
  uint64_t u64tag = (((uint64_t)tag) & 0xffffffff) << 24;
  return Slot(u64prefix | u64tag, 0);
}

Slot Slot::operator+(uint8_t i) const {
  // Maximum of 8 bits for use in a single collective operation.
  // To avoid conflicts between them, raise if it overflows.
  auto delta = delta_ + i;
  if (delta > 0xff) {
    throw std::runtime_error(
        "Slot overflow: delta " + std::to_string(delta) + " > 0xff");
  }

  return Slot(base_, delta);
}

} // namespace gloo
