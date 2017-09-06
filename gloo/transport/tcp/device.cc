/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/tcp/device.h"

#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>

#include "gloo/common/linux.h"
#include "gloo/common/logging.h"
#include "gloo/common/error.h"
#include "gloo/transport/tcp/pair.h"

namespace gloo {
namespace transport {
namespace tcp {

static void lookupAddrForIface(struct attr& attr) {
  struct ifaddrs* ifap;
  auto rv = getifaddrs(&ifap);
  GLOO_ENFORCE_NE(rv, -1, strerror(errno));
  struct ifaddrs *ifa;
  for (ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
    // Skip entry if ifa_addr is NULL (see getifaddrs(3))
    if (ifa->ifa_addr == nullptr) {
      continue;
    }
    // Skip entry if the name doesn't match
    if (strcmp(attr.iface.c_str(), ifa->ifa_name) != 0) {
      continue;
    }
    // Match on address family
    switch (attr.ai_family) {
      case AF_INET:
        if (ifa->ifa_addr->sa_family != AF_INET) {
          continue;
        }
        attr.ai_addrlen = sizeof(struct sockaddr_in);
        memcpy(&attr.ai_addr, ifa->ifa_addr, attr.ai_addrlen);
        break;
      case AF_INET6:
        if (ifa->ifa_addr->sa_family != AF_INET6) {
          continue;
        }
        attr.ai_addrlen = sizeof(struct sockaddr_in6);
        memcpy(&attr.ai_addr, ifa->ifa_addr, attr.ai_addrlen);
        break;
      case AF_UNSPEC:
        switch (ifa->ifa_addr->sa_family) {
          case AF_INET:
            attr.ai_family = AF_INET;
            attr.ai_addrlen = sizeof(struct sockaddr_in);
            break;
          case AF_INET6:
            attr.ai_family = AF_INET6;
            attr.ai_addrlen = sizeof(struct sockaddr_in6);
            break;
          default:
            continue;
        }
        memcpy(&attr.ai_addr, ifa->ifa_addr, attr.ai_addrlen);
        break;
      default:
        GLOO_ENFORCE(false, "Unknown ai_family: ", attr.ai_family);
        break;
    }

    attr.ai_socktype = SOCK_STREAM;
    attr.ai_protocol = 0;
    break;
  }
  GLOO_ENFORCE(
    ifa != nullptr,
    "Unable to find address for: ",
    attr.iface);
  freeifaddrs(ifap);
  return;
}

static void lookupAddrForHostname(struct attr& attr) {
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = attr.ai_family;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* result;
  auto rv = getaddrinfo(attr.hostname.data(), nullptr, &hints, &result);
  GLOO_ENFORCE_NE(rv, -1);
  struct addrinfo* rp;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    auto fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }

    rv = bind(fd, rp->ai_addr, rp->ai_addrlen);
    if (rv == -1) {
      close(fd);
      continue;
    }

    attr.ai_family = rp->ai_family;
    attr.ai_socktype = rp->ai_socktype;
    attr.ai_protocol = rp->ai_protocol;
    memcpy(&attr.ai_addr, rp->ai_addr, rp->ai_addrlen);
    attr.ai_addrlen = rp->ai_addrlen;
    close(fd);
    break;
  }

  // Check that we found an address we were able to bind to
  GLOO_ENFORCE(
    rp != nullptr,
    "Unable to find address for: ",
    attr.hostname);
  freeaddrinfo(result);
  return;
}

std::shared_ptr<transport::Device> CreateDevice(const struct attr& src) {
  struct attr attr = src;

  if (attr.iface.size() > 0) {
    // Initialize attributes using network interface name
    lookupAddrForIface(attr);
  } else {
    // Initialize attributes using hostname/IP address
    // If not already specified, use this machine's hostname
    if (attr.hostname.size() == 0) {
      std::array<char, HOST_NAME_MAX> hostname;
      auto rv = gethostname(hostname.data(), hostname.size());
      GLOO_ENFORCE_EQ(rv, 0);
      attr.hostname = hostname.data();
    }
    lookupAddrForHostname(attr);
  }

  auto device = std::make_shared<Device>(attr);
  return std::shared_ptr<transport::Device>(device);
}

bool isLocalhostAddr(const struct sockaddr* addr) {
  if (addr->sa_family == AF_INET) {
    // Check if the address is in the range '127.x.x.x'
    auto in = (struct sockaddr_in*)addr;
    auto mask = htonl(IN_CLASSA_NET);
    auto subnet = htonl(INADDR_LOOPBACK) & mask;
    return (in->sin_addr.s_addr & mask) == subnet;
  }
  return false;
}

const std::string sockaddrToInterfaceName(const struct attr& attr) {
  struct ifaddrs* ifap;
  std::string iface;
  auto rv = getifaddrs(&ifap);
  GLOO_ENFORCE_NE(rv, -1, strerror(errno));
  auto addrIsLocalhost = isLocalhostAddr((struct sockaddr*)&attr.ai_addr);
  struct ifaddrs *ifa;
  for (ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
    // Skip entry if ifa_addr is NULL (see getifaddrs(3))
    if (ifa->ifa_addr == nullptr) {
      continue;
    }
    if (ifa->ifa_addr->sa_family == AF_INET) {
      auto sz = sizeof(struct sockaddr_in);
      // Check if this interface address matches the provided address, or if
      // this is the localhost interface and the provided address is in the
      // localhost subnet.
      if ((memcmp(&attr.ai_addr, ifa->ifa_addr, sz) == 0) ||
          (addrIsLocalhost && isLocalhostAddr(ifa->ifa_addr))) {
        iface = ifa->ifa_name;
        break;
      }
    } else if (ifa->ifa_addr->sa_family == AF_INET6) {
      auto sz = sizeof(struct sockaddr_in6);
      if (memcmp(&attr.ai_addr, ifa->ifa_addr, sz) == 0) {
        iface = ifa->ifa_name;
        break;
      }
    }
  }
  GLOO_ENFORCE(
    ifa != nullptr,
    "Unable to find interface for: ",
    Address(attr.ai_addr).str());
  freeifaddrs(ifap);
  return iface;
}

Device::Device(const struct attr& attr)
    : attr_(attr),
      interfaceName_(sockaddrToInterfaceName(attr_)),
      interfaceSpeedMbps_(getInterfaceSpeedByName(interfaceName_)),
      pciBusID_(interfaceToBusID(interfaceName_)) {
  fd_ = epoll_create(1);
  GLOO_ENFORCE_NE(fd_, -1, "epoll_create: ", strerror(errno));

  done_ = false;
  loop_.reset(new std::thread(&Device::loop, this));
}

Device::~Device() {
  done_ = true;
  loop_->join();

  close(fd_);
}

std::string Device::str() const {
  std::stringstream ss;
  ss << "tcp";
  ss << ", pci=" << pciBusID_;
  ss << ", iface=" << interfaceName_;
  ss << ", speed=" << interfaceSpeedMbps_;
  ss << ", addr=" << Address(attr_.ai_addr).str();
  return ss.str();
}

const std::string& Device::getPCIBusID() const {
  return pciBusID_;
}

int Device::getInterfaceSpeed() const {
  return interfaceSpeedMbps_;
}

std::unique_ptr<transport::Pair> Device::createPair(
    std::chrono::milliseconds timeout) {
  if (timeout < std::chrono::milliseconds::zero()) {
    GLOO_THROW_INVALID_OPERATION_EXCEPTION("Invalid timeout", timeout.count());
  }
  auto pair = new Pair(shared_from_this(), timeout);
  return std::unique_ptr<transport::Pair>(pair);
}

void Device::registerDescriptor(int fd, int events, Pair* p) {
  struct epoll_event ev;
  int rv;

  ev.events = events;
  ev.data.ptr = p;

  rv = epoll_ctl(fd_, EPOLL_CTL_ADD, fd, &ev);
  if (rv == -1 && errno == EEXIST) {
    rv = epoll_ctl(fd_, EPOLL_CTL_MOD, fd, &ev);
  }
  GLOO_ENFORCE_NE(rv, -1, "epoll_ctl: ", strerror(errno));

  return;
}

void Device::unregisterDescriptor(int fd) {
  int rv;

  rv = epoll_ctl(fd_, EPOLL_CTL_DEL, fd, nullptr);
  GLOO_ENFORCE_NE(rv, -1, "epoll_ctl: ", strerror(errno));

  // Wait for loop to tick before returning, to make sure the handler
  // for this fd is not called once this function returns.
  if (std::this_thread::get_id() != loop_->get_id()) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock);
  }

  return;
}

void Device::loop() {
  std::array<struct epoll_event, capacity_> events;
  int nfds;

  while (!done_) {
    // Wakeup everyone waiting for a loop tick to finish.
    cv_.notify_all();

    // Wait for something to happen
    nfds = epoll_wait(fd_, events.data(), events.size(), 10);
    if (nfds == 0) {
      continue;
    }
    if (nfds == -1 && errno == EINTR) {
      continue;
    }

    GLOO_ENFORCE_NE(nfds, -1);

    for (int i = 0; i < nfds; i++) {
      Pair* p = reinterpret_cast<Pair*>(events[i].data.ptr);
      p->handleEvents(events[i].events);
    }
  }
}

} // namespace tcp
} // namespace transport
} // namespace gloo
