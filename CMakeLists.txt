cmake_minimum_required(VERSION 3.15)
project(CusADi LANGUAGES CXX CUDA)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 --use_fast_math)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ---- Existing PyTorch-path kernel libraries (untouched) ----
set(LP_SOURCE codegen/level_parallel_kernel.cu)
add_library(level_parallel_kernel SHARED ${LP_SOURCE})
target_link_libraries(level_parallel_kernel)
set(LP_TILED_SOURCE codegen/level_parallel_tiled_kernel.cu)
add_library(level_parallel_tiled_kernel SHARED ${LP_TILED_SOURCE})

# ---- JAX FFI nanobind extension (new, feature/jax-ffi-backend) ----
# Skipped unless both nanobind and JAX FFI headers are discoverable.
find_package(Python COMPONENTS Interpreter Development)
if(Python_FOUND)
    # Locate nanobind via Python
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c
            "import nanobind, os; print(os.path.dirname(nanobind.__file__))"
        OUTPUT_VARIABLE NANOBIND_DIR
        RESULT_VARIABLE NANOBIND_RC
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Locate XLA FFI headers via JAX
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c
            "from jax import ffi; print(ffi.include_dir())"
        OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
        RESULT_VARIABLE XLA_FFI_RC
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NANOBIND_RC EQUAL 0 AND XLA_FFI_RC EQUAL 0)
        message(STATUS "nanobind dir:   ${NANOBIND_DIR}")
        message(STATUS "XLA FFI dir:    ${XLA_FFI_INCLUDE_DIR}")

        # Bring in nanobind's cmake config from its installation
        list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_DIR}/cmake")
        find_package(nanobind CONFIG)

        if(nanobind_FOUND)
            # Build the FFI extension + kernel .cu into a single shared object
            # so the nanobind module carries its own evaluate_lp_stream symbol.
            nanobind_add_module(_cusadi_jax_ffi NOMINSIZE
                src/jax_ffi/cusadi_jax_ffi.cpp
                codegen/level_parallel_kernel.cu
            )
            target_include_directories(_cusadi_jax_ffi PRIVATE
                "${XLA_FFI_INCLUDE_DIR}"
            )
            # CUDA runtime link (cudaMalloc etc.) via CMake's CUDA toolkit find
            find_package(CUDAToolkit)
            if(CUDAToolkit_FOUND)
                target_link_libraries(_cusadi_jax_ffi PRIVATE CUDA::cudart)
            endif()
            # Match ABI of the mixed CXX+CUDA TU
            set_target_properties(_cusadi_jax_ffi PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
            )
            message(STATUS "Added target _cusadi_jax_ffi")
        else()
            message(STATUS "nanobind CMake config not found; skipping _cusadi_jax_ffi")
        endif()
    else()
        message(STATUS "nanobind or jax.ffi not importable; skipping _cusadi_jax_ffi")
    endif()
endif()
