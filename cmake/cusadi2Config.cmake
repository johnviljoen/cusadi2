# find_package(cusadi2 CONFIG HINTS <python -c "import cusadi2; print(cusadi2.cmake_dir())">)
#
# cusadi2::cusadi2_f64 / cusadi2::cusadi2_f32: the generic level-parallel kernel (libcusadi2_*.so,
# CUDA runtime linked statically) with cusadi2.h on the include path and CUSADI2_SCALAR set.
get_filename_component(_cusadi2_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
foreach(_pair "f64;double" "f32;float")
    list(GET _pair 0 _sfx)
    list(GET _pair 1 _dtype)
    if(NOT TARGET cusadi2::cusadi2_${_sfx})
        if(NOT EXISTS "${_cusadi2_root}/lib/libcusadi2_${_sfx}.so")
            set(cusadi2_FOUND FALSE)
            set(cusadi2_NOT_FOUND_MESSAGE "${_cusadi2_root}/lib/libcusadi2_${_sfx}.so missing (cusadi2 built without the GPU toolchain?)")
            return()
        endif()
        add_library(cusadi2::cusadi2_${_sfx} SHARED IMPORTED)
        set_target_properties(cusadi2::cusadi2_${_sfx} PROPERTIES
            IMPORTED_LOCATION "${_cusadi2_root}/lib/libcusadi2_${_sfx}.so"
            INTERFACE_INCLUDE_DIRECTORIES "${_cusadi2_root}/include"
            INTERFACE_COMPILE_DEFINITIONS "CUSADI2_SCALAR=${_dtype}")
    endif()
endforeach()
set(cusadi2_FOUND TRUE)
