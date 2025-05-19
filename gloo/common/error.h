/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <exception>
#include <condition_variable>

#include "gloo/common/string.h"

#define GLOO_ERROR_MSG(...) \
  ::gloo::MakeString("[", __FILE__, ":", __LINE__, "] ", __VA_ARGS__)

namespace gloo {

const std::chrono::milliseconds kNoTimeout = std::chrono::milliseconds::zero();

bool _is_aborted();
void abort();
void _register_cv(std::condition_variable *cv);
void _deregister_cv(std::condition_variable *cv);

// A base class for all gloo runtime errors
struct Exception : public std::runtime_error {
  Exception() = delete;
  explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
};

#define GLOO_THROW(...) throw ::gloo::Exception(GLOO_ERROR_MSG(__VA_ARGS__))

// Thrown for invalid operations on gloo APIs
struct InvalidOperationException : public ::gloo::Exception {
  InvalidOperationException() = delete;
  explicit InvalidOperationException(const std::string& msg)
      : ::gloo::Exception(msg) {}
};

#define GLOO_THROW_INVALID_OPERATION_EXCEPTION(...) \
  throw ::gloo::InvalidOperationException(GLOO_ERROR_MSG(__VA_ARGS__))

// Thrown for unrecoverable IO errors
struct IoException : public ::gloo::Exception {
  IoException() = delete;
  explicit IoException(const std::string& msg) : ::gloo::Exception(msg) {}
};

#define GLOO_THROW_IO_EXCEPTION(...) \
  throw ::gloo::IoException(GLOO_ERROR_MSG(__VA_ARGS__))

} // namespace gloo
