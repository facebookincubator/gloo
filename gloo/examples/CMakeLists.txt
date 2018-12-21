add_executable(example1 example1.cc)
target_link_libraries(example1 gloo)
add_executable(looks_like_mpi looks_like_mpi.cc)
target_link_libraries(looks_like_mpi gloo)
