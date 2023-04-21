project(DistBlas)
add_library(distblas_api  INTERFACE)
target_include_directories(distblas_api INTERFACE .)

set_target_properties(distblas_api PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(distblas_api interface ${OpenBLAS_CXX_LIBRARIES})
target_link_libraries(distblas_api interface ${MPI_CXX_LIBRARIES})
