add_executable(example1 example1.cc)
set_property(TARGET example1 PROPERTY CXX_STANDARD 11)
target_link_libraries(example1 gloo)
