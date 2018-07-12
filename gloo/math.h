/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/types.h"

namespace gloo {

template <typename T>
void sum(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = x[i] + y[i];
  }
}

template <typename T>
void product(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = x[i] * y[i];
  }
}

template <typename T>
void max(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = std::max(x[i], y[i]);
  }
}

template <typename T>
void min(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = std::min(x[i], y[i]);
  }
}

#if GLOO_USE_AVX

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void sum<float16>(float16* x, const float16* y, size_t n);
extern template
void sum<float16>(float16* x, const float16* y, size_t n);

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void product<float16>(float16* x, const float16* y, size_t n);
extern template
void product<float16>(float16* x, const float16* y, size_t n);

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void max<float16>(float16* x, const float16* y, size_t n);
extern template
void max<float16>(float16* x, const float16* y, size_t n);

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void min<float16>(float16* x, const float16* y, size_t n);
extern template
void min<float16>(float16* x, const float16* y, size_t n);

#endif

} // namespace gloo
