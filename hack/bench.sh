ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $ROOT_DIR/..

cd build/gloo/benchmark

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-6379}

./benchmark \
  --size ${NNODES} \
  --rank ${NODE_RANK} \
  --redis-host ${REDIS_HOST} \
  --redis-port ${REDIS_PORT} \
  --prefix test-for-benchmark \
  --transport ibverbs \
  --ib-device mlx5_10 \
  --ib-port 1 \
  --elements $(( 1024 * 1024 )) \
  --inputs 4 \
  --iteration-time 2s \
  allreduce_ring
  # allreduce_ring_chunked
