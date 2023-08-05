/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <system_error>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <unistd.h>
#endif

#include "gloo/common/utils.h"

namespace gloo {

constexpr int HOSTNAME_MAX_SIZE = 192;

std::string getHostname() {
  // Get Hostname using syscall
  char hostname[HOSTNAME_MAX_SIZE]; // NOLINT
  int rv = gethostname(hostname, HOSTNAME_MAX_SIZE);
  if (rv != 0) {
    throw std::system_error(errno, std::system_category());
  }
  return std::string(hostname);
}

} // namespace gloo
