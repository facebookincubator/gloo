/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

#pragma once

#include "linux.h"

namespace gloo {

// We defined these structs in a separate header and dont include
// it from and .cu files because of an NVCC 7.5 bug

// Matches 03 (Display controller), 02 (3D controller)
const auto kPCIClass3D = PCIClassMatch{0x030200, 0xffff00};

// Matches 02 (Network controller), both Ethernet (00) and InfiniBand (07)
const auto kPCIClassNetwork = PCIClassMatch{0x020000, 0xff0000};

} // namespace gloo
