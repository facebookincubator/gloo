set(FLOO_TRANSPORT_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/address.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/buffer.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/device.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/pair.cc"
  )

add_subdirectory(tcp)
if(USE_IBVERBS)
    add_subdirectory(ibverbs)
endif()

list(APPEND FLOO_SRCS ${FLOO_TRANSPORT_SRCS})
set(FLOO_SRCS ${FLOO_SRCS} PARENT_SCOPE)
