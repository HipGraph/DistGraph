cmake_minimum_required(VERSION 3.17)
project(DistBLAS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LocationByValue MPI
message("Finding MPI")
find_package(MPI REQUIRED)
message(STATUS "MPI include dir: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI libs: ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")


message("Finding OpenMP")
find_package(OpenMP REQUIRED)


message("Finding BLAS")
find_package(BLAS REQUIRED)

include_directories($ENV{COMBLAS_ROOT}/install/include)

link_directories($ENV{COMBLAS_ROOT}/install/lib)


message("CMAKE_BINARY_PATH ${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

#add_subdirectory(include)
#add_subdirectory(core)
#add_subdirectory(embedding)

add_library(disblas_lib SHARED
            core/common.hpp
            core/distributed_mat.hpp
            core/sparse_mat.cpp
            core/sparse_mat.hpp
            io/parrallel_IO.cpp
            io/parrallel_IO.hpp
            net/process_3D_grid.cpp
            net/process_3D_grid.hpp
            partition/partitioner.cpp
            partition/partitioner.hpp)

SET(BUILD_TESTS "" ON)

if (BUILD_TESTS)
    add_subdirectory(../tests)
    message(STATUS "Building tests ....")
endif ()

SET(SOURCES distblas.cpp)

add_executable(distblas ${SOURCES})


target_link_libraries(distblas disblas_lib)

target_link_libraries(distblas MPI::MPI_CXX)
target_link_libraries(distblas OpenMP::OpenMP_CXX)
target_link_libraries(distblas CombBLAS)
