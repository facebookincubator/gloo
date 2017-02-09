if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(WARNING "Floo doesn't support OSX (uses epoll)")
else()
    list(APPEND FLOO_TRANSPORT_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/address.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/buffer.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/device.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/pair.cc"
      )
endif()

set(FLOO_TRANSPORT_SRCS ${FLOO_TRANSPORT_SRCS} PARENT_SCOPE)
