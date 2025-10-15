include(GNUInstallDirs)

# Set default install prefix
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "CMAKE_INSTALL_PREFIX is not set, defaulting to ${CMAKE_SOURCE_DIR}/install")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
else()
    message(STATUS "CMAKE_INSTALL_PREFIX set to ${CMAKE_INSTALL_PREFIX}")
endif()

# Install public headers in the include directory
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/app/f32_abs_comp_gpu.hh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pfpl
)

# These headers are internal only so we don't install them
## **Gather all .h files from PFPL/** and install them
#file(GLOB pfpl_inc_headers
#    ${CMAKE_CURRENT_SOURCE_DIR}/PFPL/src/include/*.h
#)
#file(GLOB pfpl_comp_headers
#    ${CMAKE_CURRENT_SOURCE_DIR}/PFPL/src/components/*.h
#)
#
## Install the pfpl headers into include/pfpl directory
#install(FILES ${pfpl_inc_headers}
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pfpl/include
#)
#install(FILES ${pfpl_comp_headers}
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pfpl/components
#)

# Install shared library and headers
install(TARGETS pfpl_shared
        EXPORT PFPLTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    # for executables
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}    # for shared libraries
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}  # for public headers
)

# Install static library without headers to avoid duplication
install(TARGETS pfpl_static
        EXPORT PFPLTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}    # for static libraries
)

# Generate and install export file only once
install(EXPORT PFPLTargets
        FILE PFPLTargets.cmake
        NAMESPACE PFPL::
        DESTINATION cmake
)

include(CMakePackageConfigHelpers)

# Generate and install version and config files
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/PFPLConfig.cmake"
        INSTALL_DESTINATION cmake
)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/PFPLConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/PFPLConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/PFPLConfigVersion.cmake"
        DESTINATION cmake
)

# Export the build tree
export(EXPORT PFPLTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/PFPLTargets.cmake"
        NAMESPACE PFPL::
)
