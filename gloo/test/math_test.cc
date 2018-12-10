/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/test/base_test.h"

#include <chrono>
#include <iostream>

#include "gloo/math.h"

namespace gloo {
namespace test {
namespace {

template <typename T>
class MathTest : public BaseTest {};

using MathTestTypes = ::testing::Types<
    int8_t,
    uint8_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t,
    float,
    double,
    float16>;

TYPED_TEST_CASE(MathTest, MathTestTypes);

constexpr auto num = 50;

using Func =
    std::function<void(void* c, const void* a, const void* b, size_t n)>;

// Returns two versions of the specified function such that
// we can easily test that it is commutative.
std::vector<Func> reorderFunctionArguments(
    void (*fn)(void* c, const void* a, const void* b, size_t n)) {
  std::vector<Func> out;
  out.push_back([fn](void* c, const void* a, const void* b, size_t n) {
    fn(c, a, b, n);
  });
  out.push_back([fn](void* c, const void* a, const void* b, size_t n) {
    fn(c, b, a, n);
  });
  return out;
}

TYPED_TEST(MathTest, Sum) {
  std::array<TypeParam, num> a, b, c;
  for (auto& fn : reorderFunctionArguments(&gloo::sum<TypeParam>)) {
    for (auto i = 0; i < num; i++) {
      for (auto j = 0; j < num; j++) {
        a[j] = 1;
        b[j] = 1;
        c[j] = 0;
      }
      a[i] = 2;
      fn(&c[0], &a[0], &b[0], num);
      for (auto j = 0; j < num; j++) {
        if (j == i) {
          ASSERT_EQ(c[j], 3);
        } else {
          ASSERT_EQ(c[j], 2);
        }
      }
    }
  }
};

TYPED_TEST(MathTest, Product) {
  std::array<TypeParam, num> a, b, c;
  for (auto& fn : reorderFunctionArguments(&gloo::product<TypeParam>)) {
    for (auto i = 0; i < num; i++) {
      for (auto j = 0; j < num; j++) {
        a[j] = 2;
        b[j] = 2;
        c[j] = 0;
      }
      a[i] = 4;
      fn(&c[0], &a[0], &b[0], num);
      for (auto j = 0; j < num; j++) {
        if (j == i) {
          ASSERT_EQ(c[j], 8);
        } else {
          ASSERT_EQ(c[j], 4);
        }
      }
    }
  }
};

TYPED_TEST(MathTest, Min) {
  std::array<TypeParam, num> a, b, c;
  for (auto& fn : reorderFunctionArguments(&gloo::min<TypeParam>)) {
    for (auto i = 0; i < num; i++) {
      for (auto j = 0; j < num; j++) {
        a[j] = 2;
        b[j] = 3;
        c[j] = 0;
      }
      a[i] = 1;
      fn(&c[0], &a[0], &b[0], num);
      for (auto j = 0; j < num; j++) {
        if (j == i) {
          ASSERT_EQ(c[j], 1);
        } else {
          ASSERT_EQ(c[j], 2);
        }
      }
    }
  }
};

TYPED_TEST(MathTest, Max) {
  std::array<TypeParam, num> a, b, c;
  for (auto& fn : reorderFunctionArguments(&gloo::max<TypeParam>)) {
    for (auto i = 0; i < num; i++) {
      for (auto j = 0; j < num; j++) {
        a[j] = 2;
        b[j] = 3;
        c[j] = 0;
      }
      a[i] = 4;
      fn(&c[0], &a[0], &b[0], num);
      for (auto j = 0; j < num; j++) {
        if (j == i) {
          ASSERT_EQ(c[j], 4);
        } else {
          ASSERT_EQ(c[j], 3);
        }
      }
    }
  }
};

template <typename TypeParam>
void perf(void (*fn)(void* c, const void* a, const void* b, size_t n)) {
  std::array<TypeParam, 1000> a, b, c;
  size_t n = 0;
  auto start = std::chrono::system_clock::now();
  std::chrono::nanoseconds duration;
  for (;;) {
    fn(&c[0], &a[0], &b[0], c.size());
    n += c.size();
    duration = std::chrono::system_clock::now() - start;
    if (duration.count() >= 1000000) {
      break;
    }
  }
  std::cout << n / ((float)duration.count());
  std::cout << " Gop/s" << std::endl;
}

TYPED_TEST(MathTest, SumPerf) {
  perf<TypeParam>(&gloo::sum<TypeParam>);
}

TYPED_TEST(MathTest, ProductPerf) {
  perf<TypeParam>(&gloo::product<TypeParam>);
}

TYPED_TEST(MathTest, MinPerf) {
  perf<TypeParam>(&gloo::min<TypeParam>);
}

TYPED_TEST(MathTest, MaxPerf) {
  perf<TypeParam>(&gloo::max<TypeParam>);
}

} // namespace
} // namespace test
} // namespace gloo
