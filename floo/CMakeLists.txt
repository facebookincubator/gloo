# FLOO_SRCS is the list of source files that we need to build libfloo.
set(FLOO_SRCS)

# FLOO_TEST_SRCS is the list of test sources.
set(FLOO_TEST_SRCS)

# FLOO_BENCHMARK_SRCS is the list of benchmark sources.
set(FLOO_TEST_SRCS)

# Compiled sources in root directory
list(APPEND FLOO_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
)

add_subdirectory(common)
add_subdirectory(rendezvous)
add_subdirectory(transport)

# Enable the following to get a list of source files
if(FALSE)
  message(STATUS "Sources: ")
  foreach(tmp ${FLOO_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

add_library(floo ${FLOO_SRCS})
target_link_libraries(floo ${floo_DEPENDENCY_LIBS})
install(TARGETS floo DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if(BUILD_TEST)
add_subdirectory(test)
add_executable(floo_test ${FLOO_TEST_SRCS})
target_link_libraries(floo_test floo gtest)
endif()

if(BUILD_BENCHMARK)
add_subdirectory(benchmark)
add_executable(floo_benchmark ${FLOO_BENCHMARK_SRCS})
target_link_libraries(floo_benchmark floo)
endif()
