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
#define ACTION_SIZE 3
#define SIZE_OF_SIZE 16
#define RESPONSE_SIZE 2

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
          port_(static_cast<uint16_t>(port)),
          world_size_(world_size),
          is_master_(is_master),
          timeout_(timeout),
          data_({})
    {
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
        address.sin_port = htons(port_);

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
    }

    void TCPStore::accept_func()
    {

      // 服务器进入循环，持续接受客户端连接
      while (true)
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

        // 读取客户端消息
        char act_buffer[ACTION_SIZE + 1] = {0};
        int valread = read(new_socket, act_buffer, ACTION_SIZE);
        std::string action = std::string(act_buffer);
        if (action == POST_ACTION_SET)
        {
          std::cout << "Set request received." << std::endl;

          // read key size
          char key_size_buffer[SIZE_OF_SIZE + 1] = {0};
          int valread = read(new_socket, key_size_buffer, SIZE_OF_SIZE);
          int key_size = atoi(key_size_buffer);
          std::cout << "key size: " << key_size << std::endl;

          // read key
          char key_buffer[key_size + 1] = {0};
          valread = read(new_socket, key_buffer, key_size);
          std::string key = std::string(key_buffer);
          std::cout << "key: " << key << std::endl;

          // read data size
          char data_size_buffer[SIZE_OF_SIZE + 1] = {0};
          valread = read(new_socket, data_size_buffer, SIZE_OF_SIZE);
          int data_size = atoi(data_size_buffer);
          std::cout << "data size: " << data_size << std::endl;

          // read data
          char data_buffer[data_size + 1] = {0};
          valread = read(new_socket, data_buffer, data_size);
          std::string value = std::string(data_buffer);
          std::vector<char> value_vec(data_buffer, data_buffer + data_size);
          std::cout << "data_buffer: <" << data_buffer << ">" << std::endl;
          std::cout << "value read: " << valread << "value: <" << value << ">" << std::endl;

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
          std::cout << "Get request received." << std::endl;
          // read key size
          char key_size_buffer[SIZE_OF_SIZE + 1] = {0};
          int valread = read(new_socket, key_size_buffer, SIZE_OF_SIZE);
          int key_size = atoi(key_size_buffer);

          // read key
          char key_buffer[key_size + 1] = {0};
          valread = read(new_socket, key_buffer, key_size);
          std::string key = std::string(key_buffer);
          std::cout << "get key: " << key << std::endl;

          bool found = false;
          std::vector<char> value = {};

          mtx.lock();
          if (data_.find(key) != data_.end())
          {
            found = true;
            value = data_[key];
          }
          mtx.unlock();

          if (found)
          {
            send(new_socket, value.data(), value.size(), 0);
          }
          else
          {
            const char *response = NOT_FOUND.c_str();
            send(new_socket, response, strlen(response), 0);
          }
        }
        else
        {
          // 向客户端发送响应
          const char *response = "OK";
          send(new_socket, response, strlen(response), 0);
          std::cout << "Response sent to client." << std::endl;
        }

        close(new_socket);
      }
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
        // 创建 socket
        int new_server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (new_server_fd == -1)
        {
          auto err = std::string("Socket creation failed: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // 设置服务器地址信息
        server_address.sin_family = AF_INET;
        server_address.sin_port = htons(port_);

        // 将 IP 地址从文本转换为二进制形式
        if (inet_pton(AF_INET, host_ip_.c_str(), &server_address.sin_addr) <= 0)
        {
          auto err = std::string("Invalid address: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // 连接服务器
        if (connect(new_server_fd, (struct sockaddr *)&server_address, sizeof(server_address)) < 0)
        {
          auto err = std::string("Connection to server failed: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // send action
        std::string act_data = POST_ACTION_SET;
        const char *message = act_data.c_str();
        send(new_server_fd, message, strlen(message), 0);

        // send key size
        size_t len = key.length();
        std::string len_str = std::to_string(len);
        len_str = std::string(SIZE_OF_SIZE - len_str.length(), '0') + len_str;
        message = len_str.c_str();
        send(new_server_fd, message, strlen(message), 0);
        std::cout << "key size: " << len_str << std::endl;

        // send key
        message = key.c_str();
        send(new_server_fd, message, strlen(message), 0);
        std::cout << "key: " << key << std::endl;

        // send data size
        len = data.size();
        len_str = std::to_string(len);
        len_str = std::string(SIZE_OF_SIZE - len_str.length(), '0') + len_str;
        message = len_str.c_str();
        send(new_server_fd, message, strlen(message), 0);
        std::cout << "data size: " << len_str << std::endl;

        // send data
        void *data_ptr = static_cast<void *>(const_cast<char *>(data.data()));
        send(new_server_fd, data_ptr, len, 0);

        // 读取服务器响应
        char buffer[RESPONSE_SIZE] = {0};
        int valread = read(new_server_fd, buffer, RESPONSE_SIZE);
        std::cout << key << " set request, server response: " << buffer << std::endl;

        close(new_server_fd);
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
        // 创建 socket
        int new_server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (new_server_fd == -1)
        {
          auto err = std::string("Socket creation failed: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // 设置服务器地址信息
        server_address.sin_family = AF_INET;
        server_address.sin_port = htons(port_);

        // 将 IP 地址从文本转换为二进制形式
        if (inet_pton(AF_INET, host_ip_.c_str(), &server_address.sin_addr) <= 0)
        {
          auto err = std::string("Invalid address: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // 连接服务器
        if (connect(new_server_fd, (struct sockaddr *)&server_address, sizeof(server_address)) < 0)
        {
          auto err = std::string("Connection to server failed: ") + strerror(errno);
          GLOO_THROW(err);
        }

        // send action
        std::string act_data = POST_ACTION_GET;
        const char *message = act_data.c_str();
        send(new_server_fd, message, strlen(message), 0);
        // std::cout << "Message sent to server." << std::endl;

        // send key size
        size_t len = key.length();
        std::string len_str = std::to_string(len);
        len_str = std::string(SIZE_OF_SIZE - len_str.length(), '0') + len_str;
        message = len_str.c_str();
        send(new_server_fd, message, strlen(message), 0);

        // send key
        message = key.c_str();
        send(new_server_fd, message, strlen(message), 0);

        // 读取服务器响应
        char buffer[BUFFER_SIZE] = {0};
        int valread = read(new_server_fd, buffer, BUFFER_SIZE);
        if (valread > 0)
        {
          std::string buffer_str = std::string(buffer);
          std::cout << key << " get request, server response: " << buffer_str << std::endl;

          return std::vector<char>(buffer, buffer + valread);
        }
        else
        {
          GLOO_THROW("Server response failed!");
        }

        close(new_server_fd);
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
          std::cout << "key: " << key << ", data: <" << buffer_str << ">" << std::endl;
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
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
    }

  } // namespace rendezvous
} // namespace gloo
