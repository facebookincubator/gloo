/**
 * Copyright (c) 2020-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/test/base_test.h"
#include "gloo/test/openssl_utils.h"

namespace gloo {
namespace test {

const char *kDefaultDevice = "localhost";

std::shared_ptr<::gloo::transport::Device> createDevice(Transport transport) {
#if GLOO_HAVE_TRANSPORT_TCP
  if (transport == Transport::TCP) {
    return ::gloo::transport::tcp::CreateDevice(kDefaultDevice);
  }
#endif
#if GLOO_HAVE_TRANSPORT_TCP_TLS
  if (transport == Transport::TCP_TLS) {
    return ::gloo::transport::tcp::tls::CreateDevice(
        kDefaultDevice, pkey_file, cert_file, ca_cert_file, "");
  }
#endif
#if GLOO_HAVE_TRANSPORT_UV
  if (transport == Transport::UV) {
#ifdef _WIN32
    gloo::transport::uv::attr attr;
    attr.ai_family = AF_UNSPEC;
    return ::gloo::transport::uv::CreateDevice(attr);
#else
    return ::gloo::transport::uv::CreateDevice(kDefaultDevice);
#endif
  }
#endif
  return nullptr;
}

} // namespace test
} // namespace gloo
