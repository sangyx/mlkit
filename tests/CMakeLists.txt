ENABLE_TESTING()

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../.travis/cmake)

# Find the required package.
FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(ArrayFire REQUIRED)

# Include the ArrayFire hreaders
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${GTest_INCLUDE_DIRS})

MACRO(PACKAGE_ADD_TEST TESTNAME)
    ADD_LIBRARY(TEST_CORE OBJECT ${ARGN})

    MESSAGE( STATUS "CPU = ${ArrayFire_CPU_FOUND}.")
    MESSAGE( STATUS "CUDA = ${ArrayFire_CUDA_FOUND}. BUILD_CUDA = ${BUILD_CUDA}.")

    IF(ArrayFire_CPU_FOUND)
        # MESSAGE(STATUS "ArrayFire CPU backend found. Enabling CPU benchmark")
        ADD_EXECUTABLE(${TESTNAME}_CPU $<TARGET_OBJECTS:TEST_CORE>)
        TARGET_LINK_LIBRARIES(${TESTNAME}_CPU ${ArrayFire_CPU_LIBRARIES}
           ${GTEST_LIBRARIES} -lpthread -lm)
        ADD_TEST(NAME ${TESTNAME}_CPU COMMAND ${TESTNAME}_CPU)

        IF(CMAKE_BUILD_TYPE STREQUAL "Coverage")
            SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../.travis/cmake)
            INCLUDE(CodeCoverage)
            SETUP_TARGET_FOR_COVERAGE(${TESTNAME}_COV ${TESTNAME}_CPU coverage)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
        ENDIF() #CMAKE_BUILD_TYPE STREQUAL "Coverage"
    ENDIF()

    # ArrayFire CUDA backend
    IF(ArrayFire_CUDA_FOUND AND BUILD_CUDA)
        ADD_EXECUTABLE(${TESTNAME}_CUDA  $<TARGET_OBJECTS:TEST_CORE>)
        TARGET_LINK_LIBRARIES(${TESTNAME}_CUDA ${ArrayFire_CUDA_LIBRARIES}
            ${GTEST_LIBRARIES} -lpthread -lm)
        ADD_TEST(NAME ${TESTNAME}_CUDA COMMAND ${TESTNAME}_CUDA)
    ENDIF()
ENDMACRO()

PACKAGE_ADD_TEST(
    MLKIT_TEST
    main.cpp
    preprocessing_test.cpp
    linear_model_test.cpp
    neighbors_test.cpp
)
