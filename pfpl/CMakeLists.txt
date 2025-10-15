# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)

# Project configuration
project(PFPL VERSION 0.0.0 DESCRIPTION "Guaranteed-error bounded GPU lossy compression library")

# Enable languages
enable_language(CXX CUDA)

# Find required CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75 80 86 90 100)
set(CMAKE_CUDA_ARCHITECTURES 80 86 90 100)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE Release)
endif()

# Disable specific warnings (adjust as needed)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mno-fma -ffp-contract=off -Wno-unused-result")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -fmad=false -Xcompiler \"-O3 -march=native -mno-fma -ffp-contract=off\" -Wno-unused-result")

# List of source files
set(pfpl_SOURCES
    app/f32_abs_comp_gpu.cu
)

# Create static and shared libraries
add_library(pfpl_static STATIC ${pfpl_SOURCES})
add_library(pfpl_shared SHARED ${pfpl_SOURCES})

# Include directories for both libraries
target_include_directories(pfpl_static
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/PFPL/src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/app>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(pfpl_shared
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/PFPL/src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/app>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link CUDA runtime
target_link_libraries(pfpl_static PRIVATE CUDA::cudart)
target_link_libraries(pfpl_shared PRIVATE CUDA::cudart)

# Set output name for libraries
set_target_properties(pfpl_static PROPERTIES OUTPUT_NAME pfpl)
set_target_properties(pfpl_shared PROPERTIES OUTPUT_NAME pfpl)

# Headers for installation
set(public_headers
    app/f32_abs_comp_gpu.hh
)

## Optionally build examples
#option(pfpl_BUILD_EXAMPLES "Option to enable building example programs" ON)
#if (pfpl_BUILD_EXAMPLES)
#    add_subdirectory(examples)
#endif ()

# Include the Installing.cmake script
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Installing)

## Enable CTest
#enable_testing()
#
## Add test for installed pfpl_test binary
#add_test(NAME pfplTest_f32 COMMAND ${CMAKE_INSTALL_PREFIX}/bin/pfpl_test_f32)
#add_test(NAME pfplTest_f64 COMMAND ${CMAKE_INSTALL_PREFIX}/bin/pfpl_test_f64)
#
## Optional: Set working directory if required
#set_tests_properties(pfplTest_f32 PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
#set_tests_properties(pfplTest_f64 PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
