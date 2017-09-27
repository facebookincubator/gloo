# Find the eigen library
#
# The following variables are optionally searched for defaults
#  EIGEN_ROOT_DIR: Base directory where all eigen components are found
#  EIGEN_INCLUDE_DIR: Directory where eigen headers are found
#
# The following are set after configuration is done:
#  EIGEN_FOUND
#  EIGEN_INCLUDE_DIRS

find_path(EIGEN_INCLUDE_DIRS
  NAMES Eigen/Core
  HINTS
  ${EIGEN_INCLUDE_DIR}
  ${EIGEN_ROOT_DIR}
  ${EIGEN_ROOT_DIR}/include
  PATH_SUFFIXES eigen3)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(eigen DEFAULT_MSG EIGEN_INCLUDE_DIRS)
mark_as_advanced(EIGEN_INCLUDE_DIRS)
