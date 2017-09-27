# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS
  ${NCCL_INCLUDE_DIR}
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/include)

find_library(NCCL_LIBRARIES
  NAMES nccl
  HINTS
  ${NCCL_LIB_DIR}
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/lib
  ${NCCL_ROOT_DIR}/lib/x86_64-linux-gnu
  ${NCCL_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nccl DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
mark_as_advanced(NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
