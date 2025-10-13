include(GNUInstallDirs)

# Set default install prefix
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "CMAKE_INSTALL_PREFIX is not set, defaulting to ${CMAKE_SOURCE_DIR}/install")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
else()
    message(STATUS "CMAKE_INSTALL_PREFIX set to ${CMAKE_INSTALL_PREFIX}")
endif()

# Install public headers in the include directory
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/app/lc-compressor-QUANT_ABS_0_f32-BIT_4-RZE_1.hh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# These headers are internal only so we don't install them
## **Gather all .h files from LC-framework/** and install them
#file(GLOB lc_inc_headers
#    ${CMAKE_CURRENT_SOURCE_DIR}/LC-framework/include/*.h
#)
#file(GLOB ld_comp_headers
#    ${CMAKE_CURRENT_SOURCE_DIR}/LC-framework/components/*.h
#)
#
## Install the lc headers into include/lc directory
#install(FILES ${lc_inc_headers}
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lc/include
#)
#install(FILES ${lc_comp_headers}
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lc/components
#)

# Install shared library and headers
install(TARGETS lc_shared
        EXPORT LCTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    # for executables
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}    # for shared libraries
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}  # for public headers
)

# Install static library without headers to avoid duplication
install(TARGETS lc_static
        EXPORT LCTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}    # for static libraries
)

# Generate and install export file only once
install(EXPORT LCTargets
        FILE LCTargets.cmake
        NAMESPACE LC::
        DESTINATION cmake
)

include(CMakePackageConfigHelpers)

# Generate and install version and config files
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/LCConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/LCConfig.cmake"
        INSTALL_DESTINATION cmake
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/LCConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/LCConfigVersion.cmake"
        DESTINATION cmake
)

# Export the build tree
export(EXPORT LCTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/LCTargets.cmake"
        NAMESPACE LC::
)
