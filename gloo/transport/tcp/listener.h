/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <unordered_map>

#include <gloo/transport/tcp/address.h>
#include <gloo/transport/tcp/attr.h>
#include <gloo/transport/tcp/error.h>
#include <gloo/transport/tcp/loop.h>
#include <gloo/transport/tcp/socket.h>

namespace gloo {
namespace transport {
namespace tcp {

class Listener final : public Handler {
 public:
  using connect_callback_t =
      std::function<void(std::shared_ptr<Socket> socket, Error error)>;

  static constexpr auto kBacklog = 512;

  explicit Listener(std::shared_ptr<Loop> loop, const attr& attr);

  ~Listener();

  void handleEvents(int events) override;

  Address nextAddress();

  void waitForConnection(
      const Address& addr,
      std::chrono::milliseconds timeout,
      connect_callback_t fn);

  void haveConnection(std::shared_ptr<Socket> socket, sequence_number_t seq);

 private:
  std::mutex mutex_;
  std::shared_ptr<Loop> loop_;
  std::shared_ptr<Socket> listener_;

  // Address of this listener and the sequence number for the next
  // connection. Sequence numbers are written by a peer right after
  // establishing a new connection and used locally to match a new
  // connection to a pair instance.
  Address addr_;
  sequence_number_t seq_{0};

  std::unordered_map<sequence_number_t, connect_callback_t> seqToCallback_;
  std::unordered_map<sequence_number_t, std::shared_ptr<Socket>> seqToSocket_;
};

} // namespace tcp
} // namespace transport
} // namespace gloo
