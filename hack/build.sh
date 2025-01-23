#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $ROOT_DIR/..

mkdir -p build
cd build
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DUSE_IBVERBS=1 -DBUILD_BENCHMARK=1 -DUSE_REDIS=1 \
    -DBUILD_SHARED_LIBS=1 \
    ../ 
make
make install