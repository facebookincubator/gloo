#include <iostream>
#include <stdlib.h>

#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/transport/tcp/device.h>
#include <gloo/transport/ibverbs/device.h>
#include <gloo/allreduce.h>
#include <gloo/algorithm.h>
#include <gloo/allreduce_ring.h>
#include <gloo/common/aligned_allocator.h>

using namespace gloo;

// Function to instantiate and run algorithm.
using Func = void(
    std::shared_ptr<::gloo::Context>,
    std::vector<float *> dataPtrs,
    int dataSize);

// RAII handle for aligned buffer
template <typename T>
#ifdef _WIN32
std::vector<T> newBuffer(int size)
{
    return std::vector<T>(size);
#else
std::vector<T, aligned_allocator<T, kBufferAlignment>> newBuffer(int size)
{
    return std::vector<T, aligned_allocator<T, kBufferAlignment>>(size);
#endif
}

int main()
{
    // Initialize context
    auto rank = getenv("RANK");
    if (!rank)
    {
        rank = "0";
    }
    auto world_size = getenv("WORLD_SIZE");
    if (!world_size)
    {
        world_size = "1";
    }
    auto ib_device = getenv("IB_DEVICE");
    if (!ib_device)
    {
        ib_device = "mlx5_0";
    }
    auto myRank = atoi(rank);
    auto contextSize = atoi(world_size);
    gloo::rendezvous::Context context(myRank, contextSize);

    // Perform rendezvous for TCP pairs
    // gloo::transport::tcp::attr attr("localhost");
    // auto dev = gloo::transport::tcp::CreateDevice(attr);
    gloo::transport::ibverbs::attr attr = {
        ib_device, 1, 1};
    auto dev = gloo::transport::ibverbs::CreateDevice(attr);
    gloo::rendezvous::FileStore store("/mnt/public/liqingping/opensource/gloo/tmp/file_store");
    context.connectFullMesh(store, dev);

    std::cout << "rank = " << context.rank << ", size = " << context.size << std::endl;

    size_t data_size = 3;
    static std::function<Func> allreduceRing =
        [](std::shared_ptr<::gloo::Context> context,
           std::vector<float *> dataPtrs,
           int dataSize)
    {
        ::gloo::AllreduceRing<float> algorithm(context, dataPtrs, dataSize);
        algorithm.run();
    };

    std::shared_ptr<gloo::rendezvous::Context> rzv_context = std::make_shared<gloo::rendezvous::Context>(context);

    // std::vector<float *> ptr{new float[data_size * 2]};
    // for (auto i = 0; i < data_size; i++)
    // {
    //     ptr[0][i] = i + 1;
    //     ptr[0][i + data_size] = i + 1;
    // }
    // for (auto i = 0; i < data_size; i++)
    // {
    //     std::cout << "ptr[0][" << i << "] = " << ptr[0][i] << std::endl;
    //     std::cout << "ptr[0][" << i + data_size << "] = " << ptr[0][i + data_size] << std::endl;
    // }

    // allreduceRing(rzv_context, ptr, data_size * 2);

    const auto contextRank = rzv_context->rank;
    auto buffer = newBuffer<float>(data_size * 2);
    auto *ptr = buffer.data();

    for (int i = 0; i < data_size; i++)
    {
        ptr[i] = i + 1;
        ptr[i + data_size] = i + 1;
    }

    allreduceRing(rzv_context, std::vector<float *>{ptr}, data_size * 2);

    for (auto i = 0; i < data_size * 2; i++)
    {
        std::cout << "ptr[" << i << "] = " << ptr[i] << std::endl;
    }
    return 0;
}
