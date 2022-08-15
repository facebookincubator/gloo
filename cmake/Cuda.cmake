# Known NVIDIA GPU achitectures Gloo can be compiled for.
# This list will be used for CUDA_ARCH_NAME = All option
set(gloo_known_gpu_archs "")

################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   gloo_select_nvcc_arch_flags(out_variable)
function(gloo_select_nvcc_arch_flags out_variable)
  # List of arch names
  set(__archs_names "Kepler" "Maxwell" "Pascal" "Volta" "All")
  set(__archs_name_default "All")

  # Set CUDA_ARCH_NAME strings (so it will be seen as dropbox in the CMake GUI)
  set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target NVIDIA GPU architecture")
  set_property(CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names})
  mark_as_advanced(CUDA_ARCH_NAME)

  # Verify CUDA_ARCH_NAME value
  if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " __archs_names "${__archs_names}")
    message(FATAL_ERROR "Invalid CUDA_ARCH_NAME, supported values: ${__archs_names}")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(__cuda_arch_bin "30 35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    set(__cuda_arch_bin "50")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    set(__cuda_arch_bin "60 61")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
    set(__cuda_arch_bin "70")
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(__cuda_arch_bin ${gloo_known_gpu_archs})
  else()
    message(FATAL_ERROR "Invalid CUDA_ARCH_NAME")
  endif()

  # Remove dots and convert to lists
  string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" __cuda_arch_ptx "${CUDA_ARCH_PTX}")
  string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   __cuda_arch_ptx "${__cuda_arch_ptx}")
  list(REMOVE_DUPLICATES __cuda_arch_bin)
  list(REMOVE_DUPLICATES __cuda_arch_ptx)

  set(__nvcc_flags "")
  set(__nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(__arch ${__cuda_arch_bin})
    if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND __nvcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=sm_${__arch})
      list(APPEND __nvcc_archs_readable sm_${__arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(__arch ${__cuda_arch_ptx})
    list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=compute_${__arch})
    list(APPEND __nvcc_archs_readable compute_${__arch})
  endforeach()

  string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
  set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction()

################################################################################
# Function to append to list if specified sequence does not yet exist in list.
# Usage:
#   gloo_list_append_if_unique(list_variable arg1 arg2 ...)
function(gloo_list_append_if_unique list)
  list(LENGTH ARGN __match_length)
  set(__match_index 0)
  set(__match OFF)
  foreach(__elem ${${list}})
    list(GET ARGN ${__match_index} __match_elem)
    if("${__elem}" STREQUAL "${__match_elem}")
      MATH(EXPR __match_index "${__match_index}+1")
      if(${__match_index} EQUAL ${__match_length})
        set(__match ON)
        break()
      endif()
    else()
      # Mismatch; start from scratch.
      # This doesn't do backtracking but shouldn't be needed either.
      set(__match_index 0)
    endif()
  endforeach()

  # Only append arguments if we didn't find a match.
  if(NOT __match)
    list(APPEND ${list} ${ARGN})
    set(${list} ${${list}} PARENT_SCOPE)
  endif()
endfunction()

################################################################################
###  Non macro section
################################################################################

if(GLOO_USE_CUDA_TOOLKIT)
  find_package(CUDAToolkit 7.0 REQUIRED)
  set(GLOO_CUDA_VERSION ${CUDAToolkit_VERSION})

  # Convert -O2 -Xcompiler="-O2 -Wall" to "-O2;-Xcompiler=-O2,-Wall"
  separate_arguments(GLOO_NVCC_FLAGS UNIX_COMMAND "${CMAKE_CUDA_FLAGS}")
  string(REPLACE " " "," GLOO_NVCC_FLAGS "${GLOO_NVCC_FLAGS}")

  if(CUDA_USE_STATIC_CUDA_RUNTIME)
    set(GLOO_CUDA_LIBRARIES CUDA::cudart_static)
  else()
    set(GLOO_CUDA_LIBRARIES CUDA::cudart)
  endif()
else()
  find_package(CUDA 7.0)
  if(NOT CUDA_FOUND)
    return()
  endif()
  set(GLOO_CUDA_VERSION ${CUDA_VERSION})
  set(GLOO_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  set(GLOO_CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})

  # If the project including us doesn't set any -std=xxx directly, we set it to C++11 here.
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  if((NOT "${GLOO_NVCC_FLAGS}" MATCHES "-std=c\\+\\+") AND (NOT "${GLOO_NVCC_FLAGS}" MATCHES "-std=gnu\\+\\+"))
    if(NOT MSVC)
      gloo_list_append_if_unique(GLOO_NVCC_FLAGS "-std=c++11")
    endif()
  endif()

  mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
  mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)
endif()

set(HAVE_CUDA TRUE)
message(STATUS "CUDA detected: " ${GLOO_CUDA_VERSION})
if (${GLOO_CUDA_VERSION} LESS 9.0)
  list(APPEND GLOO_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND GLOO_NVCC_FLAGS "-D__STRICT_ANSI__")
else()
  # nvcc may complain that sm_xx is no longer supported. Suppress the warning for now.
  list(APPEND GLOO_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
endif()

if(GLOO_CUDA_VERSION VERSION_LESS 8.0)
  set(gloo_known_gpu_archs "30 35 50 52")
elseif(GLOO_CUDA_VERSION VERSION_LESS 9.0)
  set(gloo_known_gpu_archs "30 35 50 52 60 61")
elseif(GLOO_CUDA_VERSION VERSION_LESS 10.0)
  set(gloo_known_gpu_archs "30 35 50 52 60 61 70")
elseif(GLOO_CUDA_VERSION VERSION_LESS 11.0)
  set(gloo_known_gpu_archs "35 50 52 60 61 70 75")
elseif(GLOO_CUDA_VERSION VERSION_LESS 12.0)
  set(gloo_known_gpu_archs "35 50 52 60 61 70 75 80 86")
endif()

list(APPEND gloo_cuda_DEPENDENCY_LIBS ${GLOO_CUDA_LIBRARIES})

# Setting nvcc arch flags (or inherit if already set)
if (NOT ";${GLOO_NVCC_FLAGS};" MATCHES ";-gencode;")
  gloo_select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
  list(APPEND GLOO_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
  message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")
endif()

# Disable some nvcc diagnostic that apears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration set_but_not_used)
  gloo_list_append_if_unique(GLOO_NVCC_FLAGS -Xcudafe --diag_suppress=${diag})
endforeach()

if(NOT MSVC)
  gloo_list_append_if_unique(GLOO_NVCC_FLAGS "-Xcompiler" "-fPIC")
endif()

if(GLOO_USE_CUDA_TOOLKIT)
  # Convert list to space-separated string
  string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${GLOO_NVCC_FLAGS}")
else()
  set(CUDA_NVCC_FLAGS "${GLOO_NVCC_FLAGS}")
endif()
