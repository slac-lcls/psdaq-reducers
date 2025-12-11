include(GNUInstallDirs)

# Set default install prefix
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "CMAKE_INSTALL_PREFIX is not set, defaulting to ${CMAKE_SOURCE_DIR}/install")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
else()
    message(STATUS "CMAKE_INSTALL_PREFIX set to ${CMAKE_INSTALL_PREFIX}")
endif()

# Install libraries and headers
install(TARGETS lc_shared lc_static
        EXPORT LCTargets
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}         # for shared libraries
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}         # for static libraries
        RUNTIME       DESTINATION ${CMAKE_INSTALL_BINDIR}         # for executables
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lc  # for public headers
)

# Generate and install export file only once
install(EXPORT LCTargets
        NAMESPACE LC::
        DESTINATION cmake
)

include(CMakePackageConfigHelpers)

# Generate and install version and config files
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/LCConfig.cmake"
        INSTALL_DESTINATION cmake
)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/LCConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/LCConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/LCConfigVersion.cmake"
        DESTINATION cmake
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
