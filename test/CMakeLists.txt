cmake_minimum_required(VERSION 3.10.0)
project(mpi-cpp-test LANGUAGES CXX)

file(GLOB SOURCES src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} gtest gtest-mpi-main mpi-cpp)

add_test(
    NAME ${PROJECT_NAME}
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${PROJECT_NAME}> ${MPIEXEC_POSTFLAGS})
