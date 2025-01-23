rm -rf /mnt/public/liqingping/opensource/gloo/tmp/file_store/*

# in rank 0
IB_DEVICE=mlx5_10 RANK=0 WORLD_SIZE=2 ./build/send_test

# in rank 1
IB_DEVICE=mlx5_10 RANK=1 WORLD_SIZE=2 ./build/send_test