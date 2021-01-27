/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

#include <gloo/transport/uv/libuv.h>

#include <cassert>

namespace gloo {
namespace transport {
namespace uv {
namespace libuv {

void TCP::uv__connection_cb(uv_stream_t* server, int status) {
  TCP& ref = *(static_cast<TCP*>(server->data));
  if (status) {
    ref.publish(ErrorEvent{status});
  } else {
    ref.publish(ListenEvent{});
  }
}

void TCP::uv__alloc_cb(
    uv_handle_t* handle,
    size_t suggested_size,
    uv_buf_t* buf) {
  TCP& ref = *(static_cast<TCP*>(handle->data));
  if (!ref.reads_.empty()) {
    ref.reads_.front().alloc(buf);
  } else {
    abort();
  }
}

void TCP::uv__read_cb(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf) {
  TCP& ref = *(static_cast<TCP*>(stream->data));
  if (nread > 0) {
    if (ref.reads_.empty()) {
      abort();
    }
    auto& segment = ref.reads_.front();
    segment.read(nread);
    if (segment.done()) {
      ref.publish(segment.event());
      ref.reads_.pop_front();
      if (ref.reads_.empty()) {
        auto rv = uv_read_stop(ref.template get<uv_stream_t>());
        UV_ASSERT(rv, "uv_read_stop");
      }
    }
  } else if (nread == UV_EOF) {
    ref.publish(EndEvent{});
  } else if (nread < 0) {
    ref.publish(ErrorEvent(nread));
  }
}

} // namespace libuv
} // namespace uv
} // namespace transport
} // namespace gloo
