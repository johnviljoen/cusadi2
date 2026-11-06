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
