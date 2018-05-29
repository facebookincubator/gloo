/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/pair.h"

namespace gloo {
namespace transport {

// Have to provide implementation for pure virtual destructor.


#if 0
//#define SLOT (this->context_->nextSlot())
#define SLOT (96369)

int Pair::send(void* buf, size_t size){
  auto slot = SLOT;
  if (!hasSync){
    srcBuf = this->createSendBuffer(slot, buf, size);
    hasSync = size;

    dataBuf->waitSend();
    dataBuf->waitRecv();
  }
  return 1;
}

int Pair::recv(void* buf, size_t size){ 
  auto slot = SLOT;
  if (!hasSync){
    dstBuf = this->createRecvBuffer(slot, buf, size);
    hasSync = size;
  }
}
#endif


Pair::~Pair() {}

} // namespace transport
} // namespace gloo
