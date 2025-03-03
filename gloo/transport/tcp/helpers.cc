#include <gloo/transport/tcp/helpers.h>

namespace gloo {
namespace transport {
namespace tcp {

void connectLoop(
    std::shared_ptr<Loop> loop,
    const Address& remote,
    std::chrono::milliseconds timeout,
    typename ConnectOperation::callback_t fn) {
  auto x = std::make_shared<ConnectOperation>(
      std::move(loop), remote, timeout, std::move(fn));
  x->run();
}

} // namespace tcp
} // namespace transport
} // namespace gloo
