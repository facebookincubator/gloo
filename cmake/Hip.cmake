set(HAVE_HIP FALSE)

IF(NOT DEFINED ENV{ROCM_PATH})
  SET(ROCM_PATH /opt/rocm)
ELSE()
  SET(ROCM_PATH $ENV{ROCM_PATH})
ENDIF()

# HIP_PATH
IF(NOT DEFINED ENV{HIP_PATH})
  SET(HIP_PATH ${ROCM_PATH}/hip)
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()

IF(NOT EXISTS ${HIP_PATH})
  return()
ENDIF()

# HCC_PATH
IF(NOT DEFINED ENV{HCC_PATH})
  SET(HCC_PATH ${ROCM_PATH}/hcc)
ELSE()
  SET(HCC_PATH $ENV{HCC_PATH})
ENDIF()

# HSA_PATH
IF(NOT DEFINED ENV{HSA_PATH})
  SET(HSA_PATH ${ROCM_PATH}/hsa)
ELSE()
  SET(HSA_PATH $ENV{HSA_PATH})
ENDIF()

# ROCBLAS_PATH
IF(NOT DEFINED ENV{ROCBLAS_PATH})
  SET(ROCBLAS_PATH ${ROCM_PATH}/rocblas)
ELSE()
  SET(ROCBLAS_PATH $ENV{ROCBLAS_PATH})
ENDIF()

# ROCSPARSE_PATH
IF(NOT DEFINED ENV{ROCSPARSE_PATH})
  SET(ROCSPARSE_PATH ${ROCM_PATH}/rocsparse)
ELSE()
  SET(ROCSPARSE_PATH $ENV{ROCSPARSE_PATH})
ENDIF()

# ROCFFT_PATH
IF(NOT DEFINED ENV{ROCFFT_PATH})
  SET(ROCFFT_PATH ${ROCM_PATH}/rocfft)
ELSE()
  SET(ROCFFT_PATH $ENV{ROCFFT_PATH})
ENDIF()

# HIPSPARSE_PATH
IF(NOT DEFINED ENV{HIPSPARSE_PATH})
  SET(HIPSPARSE_PATH ${ROCM_PATH}/hipsparse)
ELSE()
  SET(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH})
ENDIF()

# THRUST_PATH
IF(DEFINED ENV{THRUST_PATH})
  SET(THRUST_PATH $ENV{THRUST_PATH})
ELSE()
  SET(THRUST_PATH ${ROCM_PATH}/include)
ENDIF()

# HIPRAND_PATH
IF(NOT DEFINED ENV{HIPRAND_PATH})
  SET(HIPRAND_PATH ${ROCM_PATH}/hiprand)
ELSE()
  SET(HIPRAND_PATH $ENV{HIPRAND_PATH})
ENDIF()

# ROCRAND_PATH
IF(NOT DEFINED ENV{ROCRAND_PATH})
  SET(ROCRAND_PATH ${ROCM_PATH}/rocrand)
ELSE()
  SET(ROCRAND_PATH $ENV{ROCRAND_PATH})
ENDIF()

# MIOPENGEMM
IF(NOT DEFINED ENV{MIOPENGEMM_PATH})
  SET(MIOPENGEMM_PATH ${ROCM_PATH}/miopengemm)
ELSE()
  SET(MIOPENGEMM_PATH $ENV{MIOPENGEMM_PATH})
ENDIF()

# MIOPEN_PATH
IF(NOT DEFINED ENV{MIOPEN_PATH})
  SET(MIOPEN_PATH ${ROCM_PATH}/miopen)
ELSE()
  SET(MIOPEN_PATH $ENV{MIOPEN_PATH})
ENDIF()

IF(NOT DEFINED ENV{GLOO_ROCM_ARCH})
  SET(GLOO_ROCM_ARCH gfx900;gfx906)
ELSE()
  SET(GLOO_ROCM_ARCH $ENV{GLOO_ROCM_ARCH})
ENDIF()

# Add HIP to the CMAKE Module Path
set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})

# Disable Asserts In Code (Can't use asserts on HIP stack.)
ADD_DEFINITIONS(-DNDEBUG)

# Find the HIP Package
find_package(HIP 1.0)

IF(HIP_FOUND)
  set(HAVE_HIP TRUE)

  set(CMAKE_HCC_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  set(CMAKE_HCC_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  FIND_LIBRARY(GLOO_HIP_HCC_LIBRARIES hip_hcc HINTS ${HIP_PATH}/lib)
  # Necessary includes for building Gloo since we include HIP headers that depend on hcc/hsa headers.
  set(hcc_INCLUDE_DIRS ${HCC_PATH}/include)
  set(hsa_INCLUDE_DIRS ${HSA_PATH}/include)

ENDIF()

################################################################################
function(gloo_hip_add_library target)
  set(sources ${ARGN})
  set_source_files_properties(${sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_add_library(${target} ${sources} ${GLOO_STATIC_OR_SHARED})
  target_include_directories(${target} PUBLIC ${GLOO_HIP_INCLUDE})
  target_compile_options(${target} PUBLIC ${HIP_CXX_FLAGS})
  target_link_libraries(${target} ${gloo_hip_DEPENDENCY_LIBS})
endfunction()

function(gloo_hip_add_executable target)
  set(sources ${ARGN})
  set_source_files_properties(${sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_add_executable(${target} ${sources})
  target_include_directories(${target} PUBLIC ${GLOO_HIP_INCLUDE})
  target_compile_options(${target} PUBLIC ${HIP_CXX_FLAGS})
  target_link_libraries(${target} ${gloo_hip_DEPENDENCY_LIBS})
endfunction()
