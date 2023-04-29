project(DistBlas)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)

FetchContent_GetProperties(googletest)

if (NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif ()

enable_testing()
#add_executable(distblas_test Process3DGridTests.cpp ../core/Process3DGrid.cpp)
add_executable(distblas_test IOTests.cpp ../io/CombBLASIOReader.cpp)
target_include_directories(distblas_test PRIVATE ../include)

message("comblas path: $ENV{COMBLAS_ROOT}")


include_directories($ENV{COMBLAS_ROOT}/install/include)

link_directories($ENV{COMBLAS_ROOT}/install/lib)


target_link_libraries(distblas_test GTest::gtest_main)
target_link_libraries(distblas_test MPI::MPI_CXX)
target_link_libraries(distblas_test OpenMP::OpenMP_CXX)
target_link_libraries(distblas_test CombBLAS)

include(GoogleTest)
#gtest_discover_tests(distblas_test)
