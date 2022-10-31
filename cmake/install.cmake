include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/mpi-cppConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

install(TARGETS mpi-cpp
    EXPORT mpi-cppTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
    BUNDLE DESTINATION bin)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/mpi-cppConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/mpi-cppConfig.cmake"
    INSTALL_DESTINATION lib/cmake/mpi-cpp
)

install(EXPORT mpi-cppTargets DESTINATION lib/cmake/mpi-cpp)
install(FILES
    "${PROJECT_BINARY_DIR}/mpi-cppConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/mpi-cppConfig.cmake"
    DESTINATION lib/cmake/mpi-cpp
    )
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION include)
