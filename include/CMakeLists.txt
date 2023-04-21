project(DistBlas)
add_library(distblas_api  INTERFACE)
target_include_directories(distblas_api INTERFACE .)

target_link_libraries(distblas_api INTERFACE ${OpenBLAS_CXX_LIBRARIES})
target_link_libraries(distblas_api INTERFACE ${MPI_CXX_LIBRARIES})
