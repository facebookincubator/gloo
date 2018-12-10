/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <set>
#include <string>
#include <vector>

namespace gloo {

const std::set<std::string>& kernelModules();

struct PCIClassMatch {
  int value;
  int mask;
};

std::vector<std::string> pciDevices(PCIClassMatch);

int pciDistance(const std::string& a, const std::string& b);

const std::string& interfaceToBusID(const std::string& name);

int getInterfaceSpeedByName(const std::string& ifname);

const std::string& infinibandToBusID(const std::string& name);

} // namespace gloo
