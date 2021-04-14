if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  message(WARNING "Gloo TCP-TLS transport is supported only on Linux(uses epoll API)")
else ()
  list(APPEND GLOO_TRANSPORT_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/device.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/pair.cc"
    )
  list(APPEND GLOO_TRANSPORT_HDRS
    "${CMAKE_CURRENT_SOURCE_DIR}/context.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/device.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/pair.h"
    )
  add_subdirectory(openssl)
endif ()

set(GLOO_TRANSPORT_SRCS ${GLOO_TRANSPORT_SRCS} PARENT_SCOPE)
set(GLOO_TRANSPORT_HDRS ${GLOO_TRANSPORT_HDRS} PARENT_SCOPE)
