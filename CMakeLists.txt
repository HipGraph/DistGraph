cmake_minimum_required(VERSION 3.17)
project(DistEmbed)

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

add_library(disembed_lib SHARED
        cpp/core/common.h
        cpp/core/common.cpp
        cpp/core/distributed_mat.hpp
        cpp/core/sparse_mat.hpp
        cpp/core/mpi_type_creator.hpp
        cpp/io/parrallel_IO.cpp
        cpp/io/parrallel_IO.hpp
        cpp/net/process_3D_grid.cpp
        cpp/net/process_3D_grid.hpp
        cpp/net/data_comm.hpp
        cpp/algo/embedding/algo.hpp
        cpp/core/json.hpp
        cpp/partition/partitioner.cpp
        cpp/partition/partitioner.hpp
        cpp/algo/spmm/spmm.hpp
        cpp/algo/spgemm/spgemm.hpp
        cpp/algo/fusedMM/fusedmm.hpp
        cpp/core/sparse_mat_tile.hpp
        cpp/net/tile_based_data_comm.hpp
        cpp/algo/spgemm/spgemm_with_tiling.hpp
        cpp/algo/embedding/sparse_embedding.hpp
        cpp/algo/bfs/multi_source_bfs.hpp
        cpp/algo/baseline.hpp
        cpp/algo/spmm/baseline_spmm.hpp
        cpp/algo/fusedMM/baseline_fused_mm.hpp
        cpp/algo/gat/gat_layer.hpp
        cpp/algo/gat/gat.hpp
        cpp/algo/sddmm/sddmm.hpp
        )

SET(BUILD_TESTS "" ON)

#if (BUILD_TESTS)
#    add_subdirectory(tests)
#    message(STATUS "Building tests ....")
#endif ()

SET(SOURCES cpp/dist_embed.cpp cpp/algo/sddmm/sddmm.hpp)

add_executable(distembed ${SOURCES})


target_compile_options(distembed PRIVATE  -DMKL_ILP64  -m64 -fopenmp -I$ENV{MKLROOT}/include)
target_link_options(distembed PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)


target_link_libraries(distembed disembed_lib)

target_link_libraries(distembed MPI::MPI_CXX)
target_link_libraries(distembed OpenMP::OpenMP_CXX)
target_link_libraries(distembed CombBLAS)
#target_link_libraries(distblas $ENV{MKLROOT})

