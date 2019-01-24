set(GLOO_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/allgather_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_builder_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/barrier_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/broadcast_builder_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/broadcast_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/gather_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/linux_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/reduce_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/send_recv_test.cc"
  )

add_executable(gloo_test ${GLOO_TEST_SRCS})
target_link_libraries(gloo_test gloo gloo_builder gtest)

if(USE_CUDA)
  set(GLOO_TEST_CUDA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_allreduce_builder_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_allreduce_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_base_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_broadcast_builder_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_broadcast_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
    )

  cuda_add_executable(gloo_test_cuda ${GLOO_TEST_CUDA_SRCS})
  target_link_libraries(gloo_test_cuda gloo_cuda gloo_builder gtest)
endif()

if(USE_HIP)
    set(GLOO_TEST_HIP_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/hip_allreduce_builder_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip_allreduce_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip_base_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip_broadcast_builder_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip_broadcast_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
    )

  hip_add_executable(gloo_test_hip ${GLOO_TEST_HIP_SRCS})
  set_source_files_properties(${GLOO_TEST_HIP_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  set_target_properties(gloo_test_hip PROPERTIES COMPILE_FLAGS ${HIP_HIPCC_FLAGS})
  target_link_libraries(gloo_test_hip gloo_hip gloo_builder gtest)
  set_target_properties(gloo_test_hip PROPERTIES LINKER_LANGUAGE HIP)

endif()
