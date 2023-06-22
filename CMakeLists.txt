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

#message("Finding MKL")
#find_package(MKL REQUIRED)


message("Finding OpenMP")
find_package(OpenMP REQUIRED)


message("Finding BLAS")
find_package(BLAS REQUIRED)

include_directories($ENV{COMBLAS_ROOT}/install/include $ENV{MKLROOT}/include)

link_directories($ENV{COMBLAS_ROOT}/install/lib $ENV{MKLROOT}/lib/intel64)




message("CMAKE_BINARY_PATH ${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

#add_subdirectory(include)
#add_subdirectory(core)
#add_subdirectory(embedding)

add_library(disblas_lib SHARED
        cpp/core/common.h
        cpp/core/common.cpp
        cpp/core/distributed_mat.hpp
        cpp/core/sparse_mat.hpp
        cpp/io/parrallel_IO.cpp
        cpp/io/parrallel_IO.hpp
        cpp/net/process_3D_grid.cpp
        cpp/net/process_3D_grid.hpp
        cpp/partition/partitioner.cpp
        cpp/partition/partitioner.hpp)

SET(BUILD_TESTS "" ON)

if (BUILD_TESTS)
    add_subdirectory(tests)
    message(STATUS "Building tests ....")
endif ()

SET(SOURCES cpp/distblas.cpp)

add_executable(distblas ${SOURCES})


target_compile_options(distblas PRIVATE  -DMKL_ILP64  -m64 -fopenmp -I$ENV{MKLROOT}/include)
target_link_options(distblas PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)


target_link_libraries(distblas disblas_lib)

target_link_libraries(distblas MPI::MPI_CXX)
target_link_libraries(distblas OpenMP::OpenMP_CXX)
target_link_libraries(distblas CombBLAS)
#target_link_libraries(distblas $ENV{MKLROOT})

