set(GLOO_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/allgather_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/allgatherv_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/barrier_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/broadcast_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/gather_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/linux_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/multiproc_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/reduce_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/send_recv_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/transport_test.cc"
  )

add_executable(gloo_test ${GLOO_TEST_SRCS})
target_link_libraries(gloo_test gloo gtest rt)

if(USE_CUDA)
  set(GLOO_TEST_CUDA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_allreduce_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_base_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_broadcast_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
    )

  cuda_add_executable(gloo_test_cuda ${GLOO_TEST_CUDA_SRCS})
  target_link_libraries(gloo_test_cuda gloo_cuda gtest)
endif()

if(USE_ROCM)
  set(GLOO_TEST_HIP_SRCS
    "${HIPIFY_OUTPUT_DIR}/test/hip_allreduce_test.cc"
    "${HIPIFY_OUTPUT_DIR}/test/hip_base_test.hip"
    "${HIPIFY_OUTPUT_DIR}/test/hip_broadcast_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
    )

  gloo_hip_add_executable(gloo_test_hip ${GLOO_TEST_HIP_SRCS})
  target_link_libraries(gloo_test_hip gloo_hip gtest)
endif()
