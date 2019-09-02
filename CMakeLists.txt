cmake_minimum_required(VERSION 3.1)
project(openvslam LANGUAGES CXX C)

if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 OLD)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

##################
# Set build type #
##################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

###########################
# Set application options #
###########################

set(USE_CCACHE ON CACHE BOOL "Use ccache to accelerate a build")

find_program(CCACHE_EXE ccache)
if(USE_CCACHE AND CCACHE_EXE)
    message(STATUS "ccache: Enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_EXE}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_EXE}")
else()
    message(STATUS "ccache: Disabled")
    set(USE_CCACHE OFF)
endif()

set(USE_SANITIZER OFF CACHE BOOL "Enable Address/Memory sanitizer (set env as ASAN_OPTIONS=detect_leaks=1)")

if(USE_SANITIZER)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    message(STATUS "Address/Memory sanitizer: ENABLED")
else()
    message(STATUS "Address/Memory sanitizer: DISABLED")
endif()

set(USE_STACK_TRACE_LOGGER OFF CACHE BOOL "Enable automatic stack trace logger of google-glog")

if(USE_STACK_TRACE_LOGGER)
    message(STATUS "Stack trace logger: ENABLED")
    # Glog
    find_package(Glog REQUIRED)
    include_directories(${GLOG_INCLUDE_DIR})
else()
    message(STATUS "Stack trace logger: DISABLED")
endif()

set(USE_GOOGLE_PERFTOOLS OFF CACHE BOOL "Enable profiler of google-perftools")

if(USE_GOOGLE_PERFTOOLS)
    message(STATUS "Google Perftools: ENABLED")
    # Gperftools
    find_package(Gperftools REQUIRED)
    include_directories(${GPERFTOOLS_INCLUDE_DIRS})
else()
    message(STATUS "Google Perftools: DISABLED")
endif()

########################
# Set compiler options #
########################

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math")

set(BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "Enable architecture-aware optimization (-march=native)")

set(BUILD_WITH_MARCH_NATIVE ON)
if(BUILD_WITH_MARCH_NATIVE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
    message(STATUS "Architecture-aware optimization (-march=native): ENABLED")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
    message(STATUS "Architecture-aware optimization (-march=native): DISABLED")
endif()

set(USE_SSE_ORB ON CACHE BOOL "Enable SSE3 instruction for ORB extraction (-msse3)")
set(USE_SSE_ORB ON)
if(USE_SSE_ORB)
    message(STATUS "SSE3 for ORB extraction (-msse3): ENABLED")
    add_definitions(-DUSE_SSE_ORB)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
else()
    message(STATUS "SSE3 for ORB extraction (-msse3): DISABLED")
    remove_definitions(-DUSE_SSE_ORB)
endif()

set(USE_SSE_FP_MATH ON CACHE BOOL "Enable SSE instruction for floating-point operation (-mfpmath=sse)")
set(USE_SSE_FP_MATH ON)

if(USE_SSE_FP_MATH)
    message(STATUS "SSE for floating-point operation (-mfpmath=sse): ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpmath=sse")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpmath=sse")
else()
    message(STATUS "SSE for floating-point operation (-mfpmath=sse): DISABLED")
endif()

#######################
# Check C++11 support #
#######################

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
check_cxx_compiler_flag("-std=c++14" COMPILER_SUPPORTS_CXX14)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    message(STATUS "C++11 support: OK (-std=c++17)")
elseif(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    message(STATUS "C++11 support: OK (-std=c++14)")
elseif(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    message(STATUS "C++11 support: OK (-std=c++11)")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    message(STATUS "C++11 support: OK (-std=c++0x)")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#################
# Find packages #
#################

# thread library
find_package(Threads REQUIRED)

# OpenCV
find_package (OpenCV 4.1.1 PATHS "/home/anson/work/lib/opencv411/lib/cmake/opencv4/" NO_DEFAULT_PATH REQUIRED)
# find_package(OpenCV 3.3.1 QUIET)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV == 4.1.1 not found")
endif()
message(STATUS "Use OpenCV ${OpenCV_VERSION} -- ${OpenCV_INCLUDE_DIRS} lib: ${OpenCV_LIBRARIES}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 3.2 REQUIRED)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found")
else()
    message(STATUS "Use Eigen3 ${Eigen3_VERSION} -- ${EIGEN3_INCLUDE_DIR}")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# yaml-cpp
find_package(YamlCpp REQUIRED)
include_directories(${YAMLCPP_INCLUDE_DIRS})

# spdlog
set(spdlog_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd/spdlog/include)
include_directories(${spdlog_INCLUDE_DIR})

# popl
set(popl_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd/popl/include)
include_directories(${popl_INCLUDE_DIR})

# json
set(json_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd/json/include)
include_directories(${json_INCLUDE_DIR})


#################
# BoW framework #
#################

set(BOW_FRAMEWORK "DBoW2" CACHE STRING "DBoW2 or FBoW")
set_property(CACHE BOW_FRAMEWORK PROPERTY STRINGS "DBoW2" "FBoW")

if(BOW_FRAMEWORK MATCHES "DBoW2")
    find_package(DBoW2 REQUIRED)
    set(BOW_INCLUDE_DIRS ${DBoW2_INCLUDE_DIRS})
    set(BOW_LIBS ${DBoW2_LIBS})
    add_definitions(-DUSE_DBOW2)
elseif(BOW_FRAMEWORK MATCHES "FBoW")
    find_package(fbow REQUIRED)
    set(BOW_INCLUDE_DIRS ${fbow_INCLUDE_DIRS})
    set(BOW_LIBS ${fbow_LIBS})
else()
    message(FATAL_ERROR "Invalid BoW framework: ${BOW_FRAMEWORK}")
endif()
include_directories(${BOW_INCLUDE_DIRS})
message(STATUS "BoW framework: ${BOW_FRAMEWORK} (found in ${BOW_INCLUDE_DIRS}), lib -- ${BOW_LIBS}")




include_directories(${PROJECT_SOURCE_DIR}/match)
#include_directories(${PROJECT_SOURCE_DIR}/initializer)

add_subdirectory(match)
#add_subdirectory(initializer)


