/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <string>

#include <sys/socket.h>

#include "floo/transport/address.h"

namespace floo {
namespace transport {
namespace tcp {

// Forward declaration
class Pair;

class Address : public ::floo::transport::Address {
 public:
  Address() {}
  explicit Address(const struct sockaddr_storage&);
  explicit Address(const struct sockaddr* addr, size_t addrlen);
  explicit Address(const std::vector<char>&);
  virtual ~Address() {}

  virtual std::vector<char> bytes() const override;
  virtual std::string str() const override;

  static Address fromSockName(int fd);
  static Address fromPeerName(int fd);

 protected:
  struct sockaddr_storage ss_;

  // Pair can access ss_ directly
  friend class Pair;
};

} // namespace tcp
} // namespace transport
} // namespace floo
