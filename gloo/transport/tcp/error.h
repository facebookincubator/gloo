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
  Error() : valid_(false) {}

  static const Error OK;

  operator bool() const {
    return valid_;
  }

  virtual std::string what() const;

 protected:
  Error(bool valid) : valid_(valid){};

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
  int error_;
};

class ShortReadError : public Error {
 public:
  ShortReadError(ssize_t expected, ssize_t actual)
      : Error(true), expected_(expected), actual_(actual) {}

  std::string what() const override;

 private:
  ssize_t expected_;
  ssize_t actual_;
};

class ShortWriteError : public Error {
 public:
  ShortWriteError(ssize_t expected, ssize_t actual)
      : Error(true), expected_(expected), actual_(actual) {}

  std::string what() const override;

 private:
  ssize_t expected_;
  ssize_t actual_;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
