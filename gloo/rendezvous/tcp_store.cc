/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tcp_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <iostream>
#include <stdexcept>

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#endif

#include "gloo/common/error.h"
#include "gloo/common/logging.h"

#define BUFFER_SIZE 1024
const std::string POST_ACTION_SET = "set";
const std::string POST_ACTION_GET = "get";
const std::string NOT_FOUND = "NOT_FOUND";

namespace gloo
{
  namespace rendezvous
  {
    TCPStore::~TCPStore()
    {
      close(server_fd);
    }

    TCPStore::TCPStore(const std::string &hostname, int port, int world_size, bool is_master, int timeout)
        : hostname_(hostname),
          host_ip_(host_to_ip(hostname)),
          port_(port),
          world_size_(world_size),
          is_master_(is_master),
          timeout_(timeout),
          data_({})
    {
      uint16_t PORT = static_cast<uint16_t>(port);
      std::cout << "hostname: " << hostname_ << ", " << host_ip_ << ", port: " << port << ", world_size: " << world_size
                << ", is_master: " << is_master << std::endl;
      if (is_master)
      {
        // 创建 socket
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == -1)
        {
          auto err = std::string("Socket creation failed: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // 设置服务器地址信息
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY; // 监听所有的网络接口
        address.sin_port = htons(PORT);

        // 绑定 socket 到地址
        if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
        {
          auto err = std::string("Socket bind failed: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // 开始监听
        if (listen(server_fd, 3) < 0)
        {
          auto err = std::string("Socket listen failed: ") + strerror(errno);
          GLOO_THROW(err);
        }

        std::thread(&TCPStore::accept_func, this).detach();
      }
      else
      {
        // 创建 socket
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == -1)
        {
          auto err = std::string("Socket creation failed: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // 设置服务器地址信息
        server_address.sin_family = AF_INET;
        server_address.sin_port = htons(PORT);

        // 将 IP 地址从文本转换为二进制形式
        if (inet_pton(AF_INET, host_ip_.c_str(), &server_address.sin_addr) <= 0)
        {
          auto err = std::string("Invalid address: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // 连接服务器
        if (connect(server_fd, (struct sockaddr *)&server_address, sizeof(server_address)) < 0)
        {
          auto err = std::string("Connection to server failed: ") + strerror(errno);
          GLOO_THROW(err);
        }
      }
    }

    void TCPStore::accept_func()
    {
      // 接受客户端连接
      int new_socket;
      struct sockaddr_in client_address;
      socklen_t addr_len = sizeof(client_address);
      new_socket = accept(server_fd, (struct sockaddr *)&client_address, &addr_len);
      if (new_socket < 0)
      {
        auto err = std::string("Accept client connection failed: ") + strerror(errno);
        GLOO_THROW(err);
      }

      std::cout << "Connection established with client." << std::endl;

      // 服务器进入循环，持续接受客户端连接
      while (true)
      {
        // 读取客户端消息
        char buffer[BUFFER_SIZE] = {0};
        int valread = read(new_socket, buffer, BUFFER_SIZE);
        if (valread > 0)
        {
          std::string buffer_str = std::string(buffer);
          std::vector<std::string> buffer_split = str_split(buffer_str, ':');
          if (buffer_split.size() < 2)
          {
            GLOO_THROW("Invalid message format, must be formated as [action]:[key]:[value] or [action]:[key]!");
          }

          std::string action = buffer_split[0];
          if (action == POST_ACTION_SET)
          {
            std::string key = buffer_split[1];
            std::string value = buffer_split[2];
            std::vector<char> value_vec(value.begin(), value.end());
            mtx.lock();
            data_[key] = value_vec;
            mtx.unlock();

            // 向客户端发送响应
            const char *response = "OK";
            send(new_socket, response, strlen(response), 0);
            // std::cout << "Response sent to client." << std::endl;
          }
          else if (action == POST_ACTION_GET)
          {
            std::string key = buffer_split[1];
            bool found = false;
            std::vector<char> value = {};

            mtx.lock();
            if (data_.find(key) != data_.end())
            {
              found = true;
              value = data_[key];
            }
            mtx.unlock();

            std::string value_str(value.begin(), value.end());
            value_str = found ? value_str : NOT_FOUND;
            const char *response = value_str.c_str();
            send(new_socket, response, strlen(response), 0);
          }
          else
          {
            // 向客户端发送响应
            const char *response = "OK";
            send(new_socket, response, strlen(response), 0);
            // std::cout << "Response sent to client." << std::endl;
          }
        }
      }
      close(new_socket);
    }

    void TCPStore::set(const std::string &key, const std::vector<char> &data)
    {
      if (is_master_)
      {
        mtx.lock();
        data_[key] = data;
        mtx.unlock();
      }
      else
      {
        // 向服务器发送消息
        std::string key_with_data = POST_ACTION_SET + ":" + key + ":" + std::string(data.begin(), data.end());
        const char *message = key_with_data.c_str();
        send(server_fd, message, strlen(message), 0);
        // std::cout << "Message sent to server." << std::endl;

        // 读取服务器响应
        char buffer[BUFFER_SIZE] = {0};
        int valread = read(server_fd, buffer, BUFFER_SIZE);
        // std::cout << "Server response: " << buffer << std::endl;
      }
    }

    std::vector<char> TCPStore::get(const std::string &key)
    {
      if (is_master_)
      {
        bool found = false;
        std::vector<char> value = {};

        mtx.lock();
        if (data_.find(key) != data_.end())
        {
          found = true;
          value = data_[key];
        }
        mtx.unlock();

        std::string value_str(value.begin(), value.end());
        value_str = found ? value_str : NOT_FOUND;
        return std::vector<char>(value_str.begin(), value_str.end());
      }
      else
      {
        // 向服务器发送消息
        std::string key_with_data = POST_ACTION_GET + ":" + key;
        const char *message = key_with_data.c_str();
        send(server_fd, message, strlen(message), 0);
        // std::cout << "Message sent to server." << std::endl;

        // 读取服务器响应
        char buffer[BUFFER_SIZE] = {0};
        int valread = read(server_fd, buffer, BUFFER_SIZE);
        if (valread > 0)
        {
          std::string buffer_str = std::string(buffer);
          // std::cout << "Server response: " << buffer_str << std::endl;

          return std::vector<char>(buffer_str.begin(), buffer_str.end());
        }
        else
        {
          GLOO_THROW("Server response failed!");
        }
      }
    }

    void TCPStore::wait(
        const std::vector<std::string> &keys,
        const std::chrono::milliseconds &timeout)
    {
      const auto start = std::chrono::steady_clock::now();
      auto check = [&](const std::vector<std::string> &keys) -> bool
      {
        for (const auto &key : keys)
        {
          auto data = get(key);
          std::string buffer_str(data.begin(), data.end());
          // std::cout << "key: " << key << ", data: <" << buffer_str << ">" << std::endl;
          if (buffer_str == NOT_FOUND)
          {
            return false;
          }
        }
        return true;
      };

      while (!check(keys))
      {
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start);
        if (timeout != kNoTimeout && elapsed > timeout)
        {
          GLOO_THROW_IO_EXCEPTION(GLOO_ERROR_MSG(
              "Wait timeout for key(s): ", ::gloo::MakeString(keys)));
        }
        /* sleep override */
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }

  } // namespace rendezvous
} // namespace gloo
