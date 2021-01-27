/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

#include "gloo/math.h"

#include <algorithm>

#if GLOO_USE_AVX
#include <immintrin.h>
#endif

#include "gloo/types.h"

namespace gloo {

#if GLOO_USE_AVX

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void sum<float16>(void* c_, const void* a_, const void* b_, size_t n) {
  float16* c = static_cast<float16*>(c_);
  const float16* a = static_cast<const float16*>(a_);
  const float16* b = static_cast<const float16*>(b_);
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&a[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&b[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_add_ps(va32, vb32), 0);
    _mm_storeu_si128((__m128i*)(&c[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void product<float16>(void* c_, const void* a_, const void* b_, size_t n) {
  float16* c = static_cast<float16*>(c_);
  const float16* a = static_cast<const float16*>(a_);
  const float16* b = static_cast<const float16*>(b_);
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&a[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&b[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_mul_ps(va32, vb32), 0);
    _mm_storeu_si128((__m128i*)(&c[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    c[i] = a[i] * b[i];
  }
}

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void max<float16>(void* c_, const void* a_, const void* b_, size_t n) {
  float16* c = static_cast<float16*>(c_);
  const float16* a = static_cast<const float16*>(a_);
  const float16* b = static_cast<const float16*>(b_);
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&a[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&b[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_max_ps(va32, vb32), 0);
    _mm_storeu_si128((__m128i*)(&c[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    c[i] = std::max(a[i], b[i]);
  }
}

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void min<float16>(void* c_, const void* a_, const void* b_, size_t n) {
  float16* c = static_cast<float16*>(c_);
  const float16* a = static_cast<const float16*>(a_);
  const float16* b = static_cast<const float16*>(b_);
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&a[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&b[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_min_ps(va32, vb32), 0);
    _mm_storeu_si128((__m128i*)(&c[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    c[i] = std::min(a[i], b[i]);
  }
}

#endif

} // namespace gloo
