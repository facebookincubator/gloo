/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/socket.h>
#include <sys/uio.h>

#include "gloo/transport/pair.h"
#include "gloo/transport/tcp/address.h"
#include "gloo/transport/tcp/device.h"

namespace gloo {
namespace transport {
namespace tcp {

// Forward declaration
class Buffer;

// Forward declaration
class UnboundBuffer;

struct Op {
  // Default constructor initializes everything to 0.
  explicit Op();

  enum Opcode {
    SEND_BUFFER = 0,
    SEND_UNBOUND_BUFFER = 1,
    NOTIFY_SEND_READY = 2,
    NOTIFY_RECV_READY = 3,
  };

  inline enum Opcode getOpcode() {
    return static_cast<Opcode>(preamble.opcode);
  }

  struct {
    size_t nbytes;
    size_t opcode;
    size_t slot;
    size_t offset;
    size_t length;
    size_t roffset;
  } preamble;

  // Used internally
  Buffer* buf;
  UnboundBuffer* ubuf;
  size_t nread;
  size_t nwritten;
};

class Pair : public ::gloo::transport::Pair {
  enum state {
    INITIALIZING = 1,
    LISTENING = 2,
    CONNECTING = 3,
    CONNECTED = 4,
    CLOSED = 5,
  };

 public:
  explicit Pair(
      const std::shared_ptr<Device>& dev,
      std::chrono::milliseconds timeout,
      std::function<tcp::UnboundBuffer*(uint64_t slot)> fn);

  virtual ~Pair();

  Pair(const Pair& that) = delete;

  Pair& operator=(const Pair& that) = delete;

  virtual const Address& address() const override;

  virtual void connect(const std::vector<char>& bytes) override;

  virtual void setSync(bool sync, bool busyPoll) override;

  virtual std::unique_ptr<::gloo::transport::Buffer>
  createSendBuffer(int slot, void* ptr, size_t size) override;

  virtual std::unique_ptr<::gloo::transport::Buffer>
  createRecvBuffer(int slot, void* ptr, size_t size) override;

  // Send from the specified buffer to remote side of pair.
  virtual void send(transport::UnboundBuffer* tbuf, uint64_t tag) override;

  // Receive into the specified buffer from the remote side of pair.
  virtual void recv(transport::UnboundBuffer* tbuf, uint64_t tag) override;

  // Attempt to receive into the specified buffer from the remote side
  // of pair. Returns true if there was a remote pending send and the
  // recv is in progress, false otherwise.
  bool tryRecv(transport::UnboundBuffer* tbuf, uint64_t tag);

  void handleEvents(int events);

  void close() override;

 protected:
  std::shared_ptr<Device> dev_;
  state state_;
  std::atomic<bool> sync_;
  const std::chrono::milliseconds timeout_;
  // When set, instructs pair to use busy-polling on receive.
  // Can only be used with sync receive mode.
  bool busyPoll_;
  int fd_;
  size_t sendBufferSize_;

  Address self_;
  Address peer_;

  std::mutex m_;
  std::condition_variable cv_;
  std::map<int, Buffer*> buffers_;

  std::unordered_map<uint64_t, std::deque<tcp::UnboundBuffer*>> localPendingSend_;
  std::unordered_map<uint64_t, std::deque<tcp::UnboundBuffer*>> localPendingRecv_;
  std::unordered_map<uint64_t, int> remotePendingSend_;
  std::unordered_map<uint64_t, int> remotePendingRecv_;

  void sendUnboundBuffer(tcp::UnboundBuffer* buf, uint64_t slot);
  void sendNotifyRecvReady(const tcp::UnboundBuffer* buf, uint64_t slot);
  void sendNotifySendReady(const tcp::UnboundBuffer* buf, uint64_t slot);

  // Callback to issue when the remote side of the pair has
  // notified us that a send operation is ready to go. This is used to
  // implement recv-from-any on unbound buffers. The callback returns
  // an unbound buffer if there is a pending recv-from-any that
  // matches the rank of the remote side of this pair.
  std::function<tcp::UnboundBuffer*(uint64_t slot)> recvFromAnyCallback_;

  void listen();
  void connect(const Address& peer);

  Buffer* getBuffer(int slot);
  void registerBuffer(Buffer* buf);
  void unregisterBuffer(Buffer* buf);

  void sendSyncMode(Op& op);
  void sendAsyncMode(Op& op);
  void send(Op& op);
  void recv();

  const Address& peer() const {
    return peer_;
  }

  bool isSync() const {
    return sync_;
  }

  std::chrono::milliseconds getTimeout() const {
    return timeout_;
  }

  void signalIoFailureExternal(const std::string& msg);

  friend class Buffer;

 private:
  // Maintain state of a single operation for receiving operations
  // from the remote side of the pair.
  Op rx_;

  // Maintain state of multiple operations for transmitting operations
  // to the remote side. To support send/recv of unbound buffers,
  // transmission of notifications may be triggered from the event
  // loop, where it is not possible to block and wait on other I/O
  // operations to complete. Instead, if transmission cannot complete
  // in place, it must be queued and executed later.
  std::deque<Op> tx_;

  std::exception_ptr ex_;

  ssize_t writeBuildIov(Op& op, struct iovec* iov, int& ioc);
  bool write(Op& op);
  bool readBuildIov(Op& op, struct iovec& iov);
  bool read();

  void handleRemotePendingSend(const Op& op);
  void handleRemotePendingRecv(const Op& op);

  void handleListening();
  void handleConnecting();
  void handleConnected();

  void changeState(state nextState);
  void waitUntilConnected(std::unique_lock<std::mutex>& lock, bool useTimeout);
  void verifyConnected();

  // Used to signal IO exceptions from one thread and propagate onto others.
  void signalIoFailure(const std::string& msg);
  void checkErrorState();
};

} // namespace tcp
} // namespace transport
} // namespace gloo
