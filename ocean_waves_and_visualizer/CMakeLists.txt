cmake_minimum_required(VERSION 3.10)
project(OceanSim CXX)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(jonswap_mpi jonswap_mpi.cpp)
add_executable(gen_waves gen_waves.cpp)
add_executable(ship_motion ship_motion.cpp)

target_include_directories(jonswap_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gen_waves PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ship_motion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(jonswap_mpi PRIVATE MPI::MPI_CXX m)
target_link_libraries(gen_waves PRIVATE m)
target_link_libraries(ship_motion PRIVATE OpenMP::OpenMP_CXX m)

target_compile_options(jonswap_mpi PRIVATE -O2 -std=c++17)
target_compile_options(gen_waves PRIVATE -O2 -std=c++17)
target_compile_options(ship_motion PRIVATE -O2 -std=c++17)
