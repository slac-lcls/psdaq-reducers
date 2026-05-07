include(GNUInstallDirs)

# Set default install prefix
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "CMAKE_INSTALL_PREFIX is not set, defaulting to ${CMAKE_SOURCE_DIR}/install")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
else()
    message(STATUS "CMAKE_INSTALL_PREFIX set to ${CMAKE_INSTALL_PREFIX}")
endif()

# Install libraries and headers
install(TARGETS sleek_shared sleek_static
        EXPORT SLEEKTargets
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}           # for shared libraries
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}           # for static libraries
        RUNTIME       DESTINATION ${CMAKE_INSTALL_BINDIR}           # for executables
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sleek # for public headers
)

# Generate and install export file only once
install(EXPORT SLEEKTargets
        NAMESPACE SLEEK::
        DESTINATION cmake
)

include(CMakePackageConfigHelpers)

# Generate and install version and config files
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/SLEEKConfig.cmake"
        INSTALL_DESTINATION cmake
)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SLEEKConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/SLEEKConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SLEEKConfigVersion.cmake"
        DESTINATION cmake
)

# These headers are internal only so we don't install them
## **Gather all .h files from SLEEK/** and install them
#file(GLOB sleek_inc_headers
#    ${CMAKE_CURRENT_SOURCE_DIR}/SLEEK/src/include/*.h
#)
#file(GLOB sleek_comp_headers
#    ${CMAKE_CURRENT_SOURCE_DIR}/SLEEK/src/components/*.h
#)
#
## Install the sleek headers into include/sleek directory
#install(FILES ${sleek_inc_headers}
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sleek/include
#)
#install(FILES ${sleek_comp_headers}
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sleek/components
#)
