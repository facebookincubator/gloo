#include <iostream>
#include <stdlib.h>

#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/transport/tcp/device.h>
#include <gloo/transport/ibverbs/device.h>
#include <gloo/allreduce.h>
#include <gloo/algorithm.h>

using namespace gloo;

int main()
{
    // Initialize context
    auto rank = getenv("RANK");
    auto world_size = getenv("WORLD_SIZE");
    auto myRank = atoi(rank);
    auto contextSize = atoi(world_size);
    gloo::rendezvous::Context context(myRank, contextSize);

    // Perform rendezvous for TCP pairs
    gloo::transport::tcp::attr attr("localhost");
    auto dev = gloo::transport::tcp::CreateDevice(attr);
    // gloo::transport::ibverbs::attr attr = {
    //     "mlx5_10", 1, 1};
    // auto dev = gloo::transport::ibverbs::CreateDevice(attr);
    gloo::rendezvous::FileStore store("/mnt/public/liqingping/opensource/gloo/tmp/file_store");
    context.connectFullMesh(store, dev);

    std::cout << "rank = " << context.rank << ", size = " << context.size << std::endl;

    size_t data_size = 3;
    std::vector<float *> inputs{new float[data_size * 2]};
    for (auto i = 0; i < data_size; i++)
    {
        inputs[0][i] = i + 1;
        inputs[0][i + data_size] = i + 1;
    }
    std::vector<float *> outputs{new float[data_size * 2]};
    for (auto i = 0; i < data_size; i++)
    {
        outputs[0][i] = 0;
        outputs[0][i + data_size] = 0;
    }

    for (auto i = 0; i < data_size; i++)
    {
        std::cout << "inputs[0][" << i << "] = " << inputs[0][i] << std::endl;
        std::cout << "inputs[1][" << i << "] = " << inputs[0][i + data_size] << std::endl;
    }

    std::shared_ptr<gloo::rendezvous::Context> rzv_context = std::make_shared<gloo::rendezvous::Context>(context);
    AllreduceOptions opts(rzv_context);
    auto algorithm = gloo::AllreduceOptions::Algorithm::RING;
    opts.setAlgorithm(algorithm);
    opts.setOutputs(outputs, data_size * 2);
    std::cout << "##### before setInputs #####" << std::endl;
    opts.setInputs(inputs, data_size * 2);
    outputs.clear();

    // gloo::AllreduceOptions::Func fn = [](void *a, const void *b, const void *c, size_t n)
    // {
    //     return gloo::sum<float>(a, b, c, n);
    // };

    // opts.setReduceFunction(fn);
    opts.setReduceFunction([](void *a, const void *b, const void *c, size_t n)
                           {
        std::cout << "a = " << a << ", b = " << b << ", c = " << c << ", n = " << n << std::endl;
      auto ua = static_cast<float*>(a);
      const auto ub = static_cast<const float*>(b);
      const auto uc = static_cast<const float*>(c);
      for (size_t i = 0; i < n; i++) {
        ua[i] = ub[i] + uc[i];
        std::cout << "ua[" << i << "] = " << ua[i] << " = " << ub[i] << " + " << uc[i] << std::endl;
      } });

    // A small maximum segment size triggers code paths where we'll
    // have a number of segments larger than the lower bound of
    // twice the context size.
    opts.setMaxSegmentSize(128);

    gloo::allreduce(opts);

    for (auto i = 0; i < data_size; i++)
    {
        std::cout << "outputs[0][" << i << "] = " << outputs[0][i] << std::endl;
        std::cout << "outputs[0][" << i << "] = " << outputs[0][i + data_size] << std::endl;
    }
    return 0;
}
