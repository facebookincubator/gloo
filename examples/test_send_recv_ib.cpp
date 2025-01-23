#include <iostream>
#include <stdlib.h>
#include <cstring>

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

    // // Perform rendezvous for TCP pairs
    // gloo::transport::tcp::attr attr("localhost");
    // auto dev = gloo::transport::tcp::CreateDevice(attr);
    gloo::transport::ibverbs::attr attr = {
        ib_device, 1, 1};
    auto dev = gloo::transport::ibverbs::CreateDevice(attr);
    gloo::rendezvous::FileStore store("/mnt/public/liqingping/opensource/gloo/tmp/file_store");
    context.connectFullMesh(store, dev);

    std::cout << "rank = " << context.rank << ", size = " << context.size << std::endl;

    std::shared_ptr<gloo::rendezvous::Context> rzv_context = std::make_shared<gloo::rendezvous::Context>(context);
    size_t data_size = 3;
    float sends[data_size] = {1 + float(myRank), 2 + float(myRank), 3 + float(myRank)};
    float recvs[data_size] = {0, 0, 0};
    for (auto i = 0; i < data_size; i++)
    {
        std::cout << "sends[" << i << "] = " << sends[i] << std::endl;
    }

    auto slot = context.nextSlot();
    int peer;
    if (context.rank == 0)
        peer = 1;
    else
        peer = 0;

    std::cout << "peer = " << peer << std::endl;
    int bytes_ = sizeof(float) * data_size;
    auto inbox_ = static_cast<float *>(malloc(bytes_));
    auto outbox_ = static_cast<float *>(malloc(bytes_));
    auto &pair = context.getPair(peer);
    std::unique_ptr<::gloo::transport::Buffer> sendBuf = pair->createSendBuffer(slot, outbox_, bytes_);
    std::unique_ptr<::gloo::transport::Buffer> recvBuf = pair->createRecvBuffer(slot, inbox_, bytes_);

    std::memcpy(outbox_, sends, bytes_);

    sendBuf->send();
    recvBuf->waitRecv();
    sendBuf->waitSend();

    std::memcpy(recvs, inbox_, bytes_);

    for (auto i = 0; i < data_size; i++)
    {
        std::cout << "recvs[" << i << "] = " << recvs[i] << std::endl;
    }

    free(inbox_);
    free(outbox_);
    return 0;
}
