cmake_minimum_required(VERSION 3.10)
project(mpi-cpp VERSION 0.3.2 LANGUAGES CXX)

include(CMakePackageConfigHelpers)

option(
    MPI_CPP_ENABLE_TESTING
    "Should be disabled for scenarios where unit tests will not be run" ON
    )

# Dev Dependencies
find_package(Doxygen)

add_library(${PROJECT_NAME} INTERFACE)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )

target_link_libraries(
    ${PROJECT_NAME}
    INTERFACE MPI::MPI_C)

include(cmake/install.cmake)

if (MPI_CPP_ENABLE_TESTING)
    enable_testing()
    find_package(gtest-mpi 0.1 REQUIRED)
    add_subdirectory(test)
endif()

if (DOXYGEN_FOUND)
    set(DOXYGEN_SHOW_NAMESPACES YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_STRIP_FROM_PATH include)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
    doxygen_add_docs(
        mpi-cpp-docs
        README.md include
        COMMENT "Docs for MPI C++ library")

    set(DOXYGEN_GENERATE_HTML NO)
    set(DOXYGEN_GENERATE_XML YES)
    set(DOXYGEN_SHOW_NAMESPACES YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_STRIP_FROM_PATH include)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
    doxygen_add_docs(
        mpi-cpp-docs-xml
        README.md include
        COMMENT "Docs for MPI C++ library")
endif()