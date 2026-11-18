cmake_minimum_required(VERSION 3.24...3.31)
project(cusadi2 LANGUAGES CXX)

# cusadi2 is pure Python except for the JAX FFI extension (cusadi2/jax/_ffi):
# the function-independent CUDA kernel plus the XLA handler, compiled once with
# the nvcc / cudart / XLA headers shipped in pip packages (jax[cuda13] and
# friends). If that toolchain is absent in the build environment the extension
# is skipped and cusadi2.jax.jax_function is unavailable; everything else works.

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cusadi2/jax/_build_paths.py
    OUTPUT_VARIABLE _paths ERROR_VARIABLE _paths_err RESULT_VARIABLE _paths_rc
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(_paths_rc)
    message(WARNING "cusadi2.jax._ffi not built (GPU toolchain missing in the build "
                    "environment): ${_paths_err}")
    return()
endif()
foreach(_key NVCC CUDA_INCLUDE CUDART XLA_INCLUDE NANOBIND_DIR)
    string(REGEX MATCH "${_key}=([^\n]*)" _m "${_paths}")
    set(${_key} "${CMAKE_MATCH_1}")
endforeach()
message(STATUS "cusadi2.jax: nvcc ${NVCC}")
message(STATUS "cusadi2.jax: cudart ${CUDART}")
message(STATUS "cusadi2.jax: XLA headers ${XLA_INCLUDE}")

if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${NVCC}")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

find_package(nanobind CONFIG REQUIRED HINTS ${NANOBIND_DIR})
get_filename_component(_cudart_dir "${CUDART}" DIRECTORY)

# One module per scalar type: the kernel and the runtime are fixed to a scalar
# at compile time. The kernel source is generated from the op tables at build time.
foreach(_pair "double;_ffi_f64" "float;_ffi_f32")
    list(GET _pair 0 _dtype)
    list(GET _pair 1 _mod)
    set(_kernel ${CMAKE_CURRENT_BINARY_DIR}/cusadi2_kernel_${_dtype}.cu)
    add_custom_command(
        OUTPUT ${_kernel}
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cusadi2/jax/_gen_kernel.py ${_kernel} ${_dtype}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cusadi2/kernel_template.py
                ${CMAKE_CURRENT_SOURCE_DIR}/cusadi2/ops.py
                ${CMAKE_CURRENT_SOURCE_DIR}/cusadi2/jax/_gen_kernel.py
        COMMENT "Generating cusadi2 kernel source (${_dtype})")
    nanobind_add_module(${_mod} NOMINSIZE
        cusadi2/jax/ffi.cu
        ${_kernel})
    target_compile_definitions(${_mod} PRIVATE
        CUSADI2_SCALAR=${_dtype}
        CUSADI2_FFI_MODULE=${_mod})
    target_include_directories(${_mod} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cusadi2
        ${XLA_INCLUDE}
        ${CUDA_INCLUDE})
    target_link_libraries(${_mod} PRIVATE ${CUDART})
    set_target_properties(${_mod} PROPERTIES
        BUILD_RPATH "${_cudart_dir}"
        INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib"
        INSTALL_RPATH_USE_LINK_PATH FALSE)
    install(TARGETS ${_mod} LIBRARY DESTINATION cusadi2/jax)
endforeach()

# libcusadi2_f64.so / libcusadi2_f32.so: the same kernel as a plain shared library for C++ / CUDA
# consumers (cuipm), so that they include cusadi2.h and link it instead of compiling the kernel.
# Fixed architectures (not `native`): sm_80 / 89 / 90 / 100 / 120 machine code plus compute_120 PTX for
# newer GPUs. The CUDA runtime is the shared one of the nvidia-cuda-runtime wheel (the one JAX and
# cuipm load too, so a process holds one runtime), found next to this package. Installed
# with the header and a CMake package config under the installed package (cusadi2.get_include(),
# get_library(), cmake_dir(); find_package(cusadi2) gives cusadi2::cusadi2_f64 / _f32).
set(CUSADI2_LIB_ARCHITECTURES "80-real;89-real;90-real;100-real;120" CACHE STRING "Architectures of libcusadi2")
foreach(_pair "double;f64" "float;f32")
    list(GET _pair 0 _dtype)
    list(GET _pair 1 _sfx)
    set(_lib cusadi2_${_sfx})
    add_library(${_lib} SHARED ${CMAKE_CURRENT_BINARY_DIR}/cusadi2_kernel_${_dtype}.cu)
    target_include_directories(${_lib} PRIVATE ${CUDA_INCLUDE})
    target_link_libraries(${_lib} PRIVATE ${CUDART})
    set_target_properties(${_lib} PROPERTIES
        CUDA_ARCHITECTURES "${CUSADI2_LIB_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY None
        POSITION_INDEPENDENT_CODE ON
        BUILD_RPATH "${_cudart_dir}"
        INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib"
        INSTALL_RPATH_USE_LINK_PATH FALSE)
    install(TARGETS ${_lib} LIBRARY DESTINATION cusadi2/lib)
endforeach()
install(FILES cusadi2/cusadi2.h DESTINATION cusadi2/include)
install(FILES cmake/cusadi2Config.cmake DESTINATION cusadi2/cmake)
# Located through the installed package (an editable install keeps the Python sources in the
# checkout and the built files in site-packages; this module is among the latter).
install(FILES cmake/_native.py.in DESTINATION cusadi2 RENAME _native.py)
