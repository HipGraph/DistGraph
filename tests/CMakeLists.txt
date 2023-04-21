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
endif()

enable_testing()
add_executable(distblas_test Process3DGridTests.cpp ../core/Process3DGrid.cpp)
target_include_directories(distblas_test PRIVATE ../include)
target_link_libraries(distblas_test GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(distblas_test)
