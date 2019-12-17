ENABLE_TESTING()

# Find the required package.
FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(ArrayFire REQUIRED)

# Include the ArrayFire hreaders
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${GTest_INCLUDE_DIRS})

MACRO(PACKAGE_ADD_TEST TESTNAME)
    ADD_LIBRARY(TEST_CORE OBJECT ${ARGN})

    IF(ArrayFire_CPU_FOUND)
    #IF(${ArrayFire_CPU_FOUND})
        # MESSAGE(STATUS "ArrayFire CPU backend found. Enabling CPU benchmark")
        ADD_EXECUTABLE(${TESTNAME}_CPU $<TARGET_OBJECTS:TEST_CORE>)
        TARGET_LINK_LIBRARIES(${TESTNAME}_CPU ${ArrayFire_CPU_LIBRARIES}
           ${GTEST_LIBRARIES} -lpthread -lm)
        ADD_TEST(NAME ${TESTNAME}_CPU COMMAND ${TESTNAME}_CPU)
    ENDIF()

    # ArrayFire CUDA backend
    # FIND_PACKAGE(CUDA)
    IF(ArrayFire_CUDA_FOUND)
    # IF(${ArrayFire_CUDA_FOUND} AND ${CUDA_FOUND})
        # We need to find CUDA and NVVM as transitive linking is disabled on some OSes
        # FIND_PACKAGE(CUDA REQUIRED)
        # FIND_PACKAGE(NVVM REQUIRED)
        # MESSAGE(STATUS ${CUDA_TOOLKIT_ROOT_DIR})
        # MESSAGE(STATUS "ArrayFire CUDA found. Enabling CUDA benchmark")
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
)