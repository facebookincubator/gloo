/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <winsock2.h>

#pragma comment(lib, "Ws2_32.lib")

namespace gloo {

void init_winsock();

} // namespace gloo
