# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)

# Project configuration
project(EIP VERSION 1.0.0 DESCRIPTION "pSZ/cuSZ Encode-In-Place variant with Fully-Fused GPU kernel")

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
set(CMAKE_CUDA_ARCHITECTURES 86 90)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE Release)
endif()

# Disable specific warnings (adjust as needed)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mno-fma -ffp-contract=off -Wno-unused-result")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false -Xcompiler \"-O3 -march=native -mno-fma -ffp-contract=off\" -Wno-unused-result")

# List of source files
set(eip_SOURCES
    app/Compressor.cu
)

# Create static and shared libraries
add_library(eip_static STATIC ${eip_SOURCES})
add_library(eip_shared SHARED ${eip_SOURCES})

# Include directories for both libraries
target_include_directories(eip_static
    PRIVATE
        ${CMAKE_INSTALL_PREFIX}/EIP/include
        ${CMAKE_INSTALL_PREFIX}/EIP/include/cusz/include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:EIP/include>
        $<INSTALL_INTERFACE:EIP/include/cusz/include>
)

target_include_directories(eip_shared
    PRIVATE
        ${CMAKE_INSTALL_PREFIX}/EIP/include
        ${CMAKE_INSTALL_PREFIX}/EIP/include/cusz/include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:EIP/include>
        $<INSTALL_INTERFACE:EIP/include/cusz/include>
)

# Link CUDA runtime
target_link_libraries(eip_static PRIVATE CUDA::cudart)
target_link_libraries(eip_shared PRIVATE CUDA::cudart)

# Set output name for libraries
set_target_properties(eip_static PROPERTIES
    PUBLIC_HEADER app/Compressor.hh
    OUTPUT_NAME eip
)
set_target_properties(eip_shared PROPERTIES
    PUBLIC_HEADER app/Compressor.hh
    OUTPUT_NAME eip
)

# Include the Installing.cmake script
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Installing)
