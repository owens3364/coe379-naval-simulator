cmake_minimum_required(VERSION 3.10)
project(OceanSim CXX)

find_package(MPI REQUIRED)

add_executable(jonswap_mpi jonswap_mpi.cpp)

target_include_directories(jonswap_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(jonswap_mpi PRIVATE MPI::MPI_CXX m)

target_compile_options(jonswap_mpi PRIVATE -O2 -std=c++17)
