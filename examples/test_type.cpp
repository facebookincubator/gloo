#include <iostream>
#include <vector>
#include <cstring> // For std::memcpy
#include <stdlib.h>

#include <gloo/common/aligned_allocator.h>

using namespace gloo;

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
    // Example size and alignment
    constexpr std::size_t kBufferAlignment = 64;
    constexpr std::size_t size = 10;

    // Create the aligned vector
    auto a = std::vector<int, aligned_allocator<int, kBufferAlignment>>(size);

    // Simulate intptr_t pointing to external data
    int external_data[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    intptr_t b = reinterpret_cast<intptr_t>(external_data);

    // Write the data from b into a
    std::memcpy(a.data(), reinterpret_cast<void *>(b), size * sizeof(int));

    // Print the result
    for (auto val : a)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}