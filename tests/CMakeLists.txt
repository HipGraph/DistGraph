project(DistBLAS)


### content to include google test
#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
#)
## For Windows: Prevent overriding the parent project's compiler/linker settings
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
##FetchContent_MakeAvailable(googletest)
#
#FetchContent_GetProperties(googletest)
#
#if (NOT googletest_POPULATED)
#    FetchContent_Populate(googletest)
#    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
#endif ()
#
#enable_testing()
#add_executable(distblas_test Process3DGridTests.cpp ../core/Process3DGrid.cpp)


#add_library(disblas_lib SHARED
#        ../cpp/core/common.hpp
#        ../cpp/core/distributed_mat.hpp
#        ../cpp/core/sparse_mat.cpp
#        ../cpp/core/sparse_mat.hpp
#        ../cpp/io/parrallel_IO.cpp
#        ../cpp/io/parrallel_IO.hpp
#        ../cpp/net/process_3D_grid.cpp
#        ../cpp/net/process_3D_grid.hpp
#        ../cpp/partition/partitioner.cpp
#        ../cpp/partition/partitioner.hpp)

SET(SOURCES data_structure_tests.cpp)

add_executable(distblas_test ${SOURCES})


#target_link_libraries(distblas_test disblas_lib)



#target_include_directories(distblas_test PRIVATE ../cpp/include)

message("comblas path: $ENV{COMBLAS_ROOT}")


#include_directories($ENV{COMBLAS_ROOT}/install/include $ENV{MKLROOT}/include)
#
#link_directories($ENV{COMBLAS_ROOT}/install/lib $ENV{MKLROOT}/lib/intel64)


#target_link_libraries(distblas_test disblas_lib)

target_compile_options(distblas_test PRIVATE  -DMKL_ILP64  -m64 -fopenmp -I$ENV{MKLROOT}/include)
target_link_options(distblas_test PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)

#target_link_libraries(distblas_test GTest::gtest_main)
#target_link_libraries(distblas_test MPI::MPI_CXX)
target_link_libraries(distblas_test OpenMP::OpenMP_CXX)
#target_link_libraries(distblas_test CombBLAS)
#target_link_libraries(distblas $ENV{MKLROOT})
#include(GoogleTest)
#gtest_discover_tests(distblas_test)
