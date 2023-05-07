/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace gloo {
namespace transport {
namespace tcp {

class Error {
 public:
  // Constant instance that indicates success.
  static const Error kSuccess;

  /* implicit */ Error() : valid_(false) {}

  virtual ~Error() = default;

  // Converting to boolean means checking if there is an error. This
  // means we don't need to use an `std::optional` and allows for a
  // snippet like the following:
  //
  //   if (error) {
  //     // Deal with it.
  //   }
  //
  operator bool() const {
    return valid_;
  }

  // Returns an explanatory string.
  // Like `std::exception` but returns a `std::string`.
  virtual std::string what() const;

 protected:
  explicit Error(bool valid) : valid_(valid) {}

 private:
  const bool valid_;
};

class SystemError : public Error {
 public:
  explicit SystemError(const char* syscall, int error)
      : Error(true), syscall_(syscall), error_(error) {}

  std::string what() const override;

 private:
  const char* syscall_;
  const int error_;
};

class ShortReadError : public Error {
 public:
  ShortReadError(ssize_t expected, ssize_t actual)
      : Error(true), expected_(expected), actual_(actual) {}

  std::string what() const override;

 private:
  const ssize_t expected_;
  const ssize_t actual_;
};

class ShortWriteError : public Error {
 public:
  ShortWriteError(ssize_t expected, ssize_t actual)
      : Error(true), expected_(expected), actual_(actual) {}

  std::string what() const override;

 private:
  const ssize_t expected_;
  const ssize_t actual_;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
