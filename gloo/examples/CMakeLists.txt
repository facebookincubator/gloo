if(MSVC)
    add_executable(example_reduce example_reduce.cc)
    target_link_libraries(example_reduce gloo)
    add_executable(example_allreduce example_allreduce.cc)
    target_link_libraries(example_allreduce gloo)

    if(USE_LIBUV)
        add_custom_command(TARGET example_reduce POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${libuv_DLL_PATH}"
                $<TARGET_FILE_DIR:example_reduce>)
    endif()
else()
    add_executable(example1 example1.cc)
    target_link_libraries(example1 gloo)
    add_executable(looks_like_mpi looks_like_mpi.cc)
    target_link_libraries(looks_like_mpi gloo)
endif()
