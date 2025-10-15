# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)

# Project configuration
project(LC VERSION 0.0.0 DESCRIPTION "Lossless and guaranteed-error-bounded lossy data-compression algorithms")

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
set(LC_SOURCES
    app/lc-compressor-QUANT_ABS_0_f32-BIT_4-RZE_1.cu
)

# Create static and shared libraries
add_library(lc_static STATIC ${LC_SOURCES})
add_library(lc_shared SHARED ${LC_SOURCES})

# Include directories for both libraries
target_include_directories(lc_static
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LC-framework
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/app>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(lc_shared
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LC-framework
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/app>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link CUDA runtime
target_link_libraries(lc_static PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(lc_shared PRIVATE CUDA::cudart CUDA::cuda_driver)

# Set output name for libraries
set_target_properties(lc_static PROPERTIES OUTPUT_NAME lc)
set_target_properties(lc_shared PROPERTIES OUTPUT_NAME lc)

# Headers for installation
set(public_headers
    app/lc-compressor-QUANT_ABS_0_f32-BIT_4-RZE_1.hh
)

## Optionally build examples
#option(LC_BUILD_EXAMPLES "Option to enable building example programs" ON)
#if (LC_BUILD_EXAMPLES)
#    add_subdirectory(examples)
#endif ()

# Include the Installing.cmake script
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Installing)

## Enable CTest
#enable_testing()
#
## Add test for installed lc_test binary
#add_test(NAME lcTest_f32 COMMAND ${CMAKE_INSTALL_PREFIX}/bin/lc_test_f32)
#add_test(NAME lcTest_f64 COMMAND ${CMAKE_INSTALL_PREFIX}/bin/lc_test_f64)
#
## Optional: Set working directory if required
#set_tests_properties(lcTest_f32 PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
#set_tests_properties(lcTest_f64 PROPERTIES WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
