set(GLOO_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/test_allreduce.cpp")
add_executable(gloo_test ${GLOO_TEST_SRCS})
target_link_libraries(gloo_test gloo ${GLOO_TEST_LIBRARIES})

set(IB_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/test_allreduce_ib.cpp")
add_executable(ib_test ${IB_TEST_SRCS})
target_link_libraries(ib_test gloo ${IB_TEST_LIBRARIES})

set(SEND_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/test_send_recv_ib.cpp")
add_executable(send_test ${SEND_TEST_SRCS})
target_link_libraries(send_test gloo ${IB_TEST_LIBRARIES})


add_executable(type_test "${CMAKE_CURRENT_SOURCE_DIR}/test_type.cpp")