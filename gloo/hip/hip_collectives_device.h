/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <algorithm>
#include <cmath>

#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/config.h"
#include "gloo/hip.h"
#include "gloo/hip_private.h"

#if GLOO_USE_RCCL
#include "gloo/hip_collectives_rccl.h"
#else
#include "gloo/hip_collectives_native.h"
#endif

namespace gloo {

template <typename T, typename Dst>
std::unique_ptr<LocalOp<T> > hipDeviceReduce(
    std::vector<HipStream>& streams,
    std::vector<HipDevicePointer<T> >& devicePtrs,
    Dst& targetPtr,
    const HipReductionFunction<T>* fn,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());

  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      HipLocalMemcpy<T, HipDevicePointer<T>, Dst> >(
          streams[0],
          devicePtrs[0],
          targetPtr,
          offset,
          count);
  }

#if GLOO_USE_RCCL
  return make_unique<HipLocalRCCLReduce<T, Dst> >(
      streams, devicePtrs, targetPtr, fn, offset, count);
#else
  return make_unique<HipLocalNativeReduce<T, Dst> >(
      streams, devicePtrs, targetPtr, fn, offset, count);
#endif
}

template <typename T, typename Src>
std::unique_ptr<LocalOp<T> > hipDeviceBroadcast(
    std::vector<HipStream>& streams,
    std::vector<HipDevicePointer<T> >& devicePtrs,
    Src& sourcePtr,
    size_t offset,
    size_t count) {
  GLOO_ENFORCE_EQ(streams.size(), devicePtrs.size());

  // Simple copy operation if there is only a single device pointer.
  if (devicePtrs.size() == 1) {
    return make_unique<
      HipLocalMemcpy<T, Src, HipDevicePointer<T> > >(
          streams[0],
          sourcePtr,
          devicePtrs[0],
          offset,
          count);
  }

#if GLOO_USE_RCCL
  return make_unique<HipLocalRCCLBroadcast<T, Src> >(
      streams, devicePtrs, sourcePtr, offset, count);
#else
  return make_unique<HipLocalNativeBroadcast<T, Src> >(
      streams, devicePtrs, sourcePtr, offset, count);
#endif
}

} // namespace gloo
