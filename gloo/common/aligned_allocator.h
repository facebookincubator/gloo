/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>
#include <memory>

namespace gloo {

// Align buffers to 32 bytes to support vectorized code
const size_t kBufferAlignment = 32;

template <typename T, int ALIGNMENT = kBufferAlignment>
class aligned_allocator {
  static_assert(
      !(ALIGNMENT & (ALIGNMENT - 1)),
      "alignment must be a power of 2");

 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind {
    using other = aligned_allocator<U, ALIGNMENT>;
  };

  inline explicit aligned_allocator() = default;
  inline ~aligned_allocator() = default;
  inline explicit aligned_allocator(const aligned_allocator& a) = default;

  inline pointer address(reference r) {
    return &r;
  }

  inline const_pointer address(const_reference r) {
    return &r;
  }

  inline pointer allocate(size_type sz) {
    pointer p;
    if (posix_memalign(
            reinterpret_cast<void**>(&p), ALIGNMENT, sizeof(T) * sz)) {
      abort();
    }
    return p;
  }

  void deallocate(pointer p, size_type /*sz*/) {
    free(p);
  }
};

} // namespace gloo
