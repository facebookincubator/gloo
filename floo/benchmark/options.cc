/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "options.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace floo {
namespace benchmark {

static void usage(int status, const char* argv0) {
  if (status != EXIT_SUCCESS) {
    fprintf(stderr, "Try `%s --help' for more information.\n", argv0);
    exit(status);
  }

  fprintf(stderr, "Usage: %s [OPTIONS] BENCHMARK\n", argv0);

#define X(x) fputs(x "\n", stderr);
  X("");
  X("Participation:");
  X("  -s, --size=SIZE        Number of processes");
  X("  -r, --rank=RANK        Rank of this process");
  X("");
  X("Rendezvous:");
  X("  -h, --redis-host=HOST  Host name of Redis server");
  X("  -p, --redis-port=PORT  Port number of Redis server");
  X("  -x, --prefix=PREFIX    Rendezvous prefix (unique for this run)");
  X("");
  X("Transport:");
  X("  -t, --transport=TRANSPORT Transport to use (tcp, ibverbs, ...)");
  X("");
  X("Benchmark parameters:");
  X("      --verify           Verify result first iteration (if applicable)");
  X("      --elements         Number of floats to use");
  X("      --iteration-count  Number of iterations");
  X("      --iteration-time   Number of seconds to run benchmark");
  X("");
  X("BENCHMARK is one of:");
  X("  allreduce_ring");
  X("  allreduce_ring_chunked");
  X("  barrier_all_to_all");
  X("  broadcast_one_to_all");
  X("");

  exit(status);
}

struct options parseOptions(int argc, char** argv) {
  struct options result;

  static struct option long_options[] = {
      {"rank", required_argument, nullptr, 'r'},
      {"size", required_argument, nullptr, 's'},
      {"redis-host", required_argument, nullptr, 'h'},
      {"redis-port", required_argument, nullptr, 'p'},
      {"prefix", required_argument, nullptr, 'x'},
      {"transport", required_argument, nullptr, 't'},
      {"verify", no_argument, nullptr, 0x1001},
      {"elements", required_argument, nullptr, 0x1002},
      {"iteration-count", required_argument, nullptr, 0x1003},
      {"iteration-time", required_argument, nullptr, 0x1004},
      {"help", no_argument, nullptr, 0xffff},
      {nullptr, 0, nullptr, 0}};

  int opt;
  while (1) {
    int option_index = 0;
    opt = getopt_long(argc, argv, "r:s:h:p:x:t:", long_options, &option_index);
    if (opt == -1) {
      break;
    }

    switch (opt) {
      case 'r': {
        result.contextRank = atoi(optarg);
        break;
      }
      case 's': {
        result.contextSize = atoi(optarg);
        break;
      }
      case 'h': {
        result.redisHost = std::string(optarg, strlen(optarg));
        break;
      }
      case 'p': {
        result.redisPort = atoi(optarg);
        break;
      }
      case 'x': {
        result.prefix = std::string(optarg, strlen(optarg));
        break;
      }
      case 't': {
        result.transport = std::string(optarg, strlen(optarg));
        break;
      }
      case 0x1001: // --verify
      {
        result.verify = true;
        break;
      }
      case 0x1002: // --elements
      {
        result.elements = atoi(optarg);
        break;
      }
      case 0x1003: // --iteration-count
      {
        result.iterationCount = atoi(optarg);
        break;
      }
      case 0x1004: // --iteration-time
      {
        long sec = atoi(optarg);
        result.iterationTimeNanos = sec * 1000 * 1000 * 1000;
        break;
      }
      case 0xffff: // --help
      {
        usage(EXIT_SUCCESS, argv[0]);
        break;
      }
      default: {
        usage(EXIT_FAILURE, argv[0]);
        break;
      }
    }
  }

  if (optind != (argc - 1)) {
    fprintf(stderr, "%s: missing benchmark specifier\n", argv[0]);
    usage(EXIT_FAILURE, argv[0]);
  }

  result.benchmark = argv[optind];
  return result;
}

} // namespace benchmark
} // namespace floo
