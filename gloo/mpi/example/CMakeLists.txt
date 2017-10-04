add_executable(main_managed main_managed.cc)
target_link_libraries(main_managed gloo)
add_executable(main_unmanaged main_unmanaged.cc)
target_link_libraries(main_unmanaged gloo)
