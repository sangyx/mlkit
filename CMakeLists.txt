CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(
	MLKIT
    VERSION 0.1
	DESCRIPTION "MLKIT: A CPP PACKAGE PROVIDES SKLEARN-LIKE API WITH GPU SUPPORT."
	LANGUAGES CXX
)

# Compile with C++11 support.
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++1z" COMPILER_SUPPORTS_CXX1Z)
IF(COMPILER_SUPPORTS_CXX17)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ELSEIF(COMPILER_SUPPORTS_CXX1Z)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
ELSE()
    MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
ENDIF()

# indicate we wish for the binaries to be written to the bin directory
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
# we have custom CMake configuration files, tell CMake to use them:
# LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

# Options. Turn on with 'cmake -Dmyvarname=ON'.
option(BUILD_TESTS "Build all tests." ON) # Makes boolean 'test' available.
option(BUILD_CUDA "Build CUDA tests." ON)

MESSAGE(STATUS "BUILD_TESTS = ${BUILD_TESTS}.")
IF(BUILD_TESTS)
    ADD_SUBDIRECTORY(tests)
ENDIF()