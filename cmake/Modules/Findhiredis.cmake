# Find the hiredis libraries
#
# The following variables are optionally searched for defaults
#  HIREDIS_ROOT_DIR: Base directory where all hiredis components are found
#  HIREDIS_INCLUDE_DIR: Directory where hiredis headers are found
#  HIREDIS_LIB_DIR: Directory where hiredis library is found
#
# The following are set after configuration is done:
#  HIREDIS_FOUND
#  HIREDIS_INCLUDE_DIRS
#  HIREDIS_LIBRARIES

find_path(HIREDIS_INCLUDE_DIRS
  NAMES hiredis.h
  HINTS
  ${HIREDIS_INCLUDE_DIR}
  ${HIREDIS_ROOT_DIR}
  ${HIREDIS_ROOT_DIR}/include
  PATH_SUFFIXES hiredis)

find_library(HIREDIS_LIBRARIES
  NAMES hiredis
  HINTS
  ${HIREDIS_LIB_DIR}
  ${HIREDIS_ROOT_DIR}
  ${HIREDIS_ROOT_DIR}/lib
  PATH_SUFFIXES hiredis)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hiredis DEFAULT_MSG HIREDIS_INCLUDE_DIRS HIREDIS_LIBRARIES)
mark_as_advanced(HIREDIS_INCLUDE_DIRS HIREDIS_LIBRARIES)
