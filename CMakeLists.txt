cmake_minimum_required(VERSION 3.14)

project(cann C CXX)

# set variables for cann project
set(CANN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CANN_ROOT=${CANN_ROOT}")

# the install path for cann project
if(NOT BUILD_PATH)
  set(BUILD_PATH "${CMAKE_CURRENT_BINARY_DIR}")
endif()
set(INSTALL_PATH "${BUILD_PATH}/install")
message(STATUS "INSTALL_PATH=${INSTALL_PATH}")

# configurations
if(EXISTS "${BUILD_PATH}/config.cmake")
  # the current config
  include("${BUILD_PATH}/config.cmake")
else()
  # the template config
  include("${CANN_ROOT}/cmake/template/config.cmake")
endif()

if(GCC_PREFIX)
  set(CMAKE_C_COMPILER ${GCC_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${GCC_PREFIX}g++)
  set(CMAKE_LINKER ${GCC_PREFIX}ld)
  set(CMAKE_AR ${GCC_PREFIX}gcc-ar)
endif()

# compile and link options
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
  set(CMAKE_C_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -rdynamic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
  endif()
else()
  set(CMAKE_C_FLAGS "-O2 -Wall -Werror -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -Werror -fPIC ${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-all")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack")
set(CMAME_C_FLAGS "${CMAKE_C_FLAGS} -pipe")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-all")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack")
set(CMAME_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

# include useful funcs
include(cmake/util.cmake)
# build external prjects
set(THIRD_PARTY "${CMAKE_CURRENT_BINARY_DIR}/third_party/src")
message(STATUS "THIRD_PARTY=${THIRD_PARTY}")
include(ExternalProject)
include(cmake/external/secure_c.cmake)
include(cmake/external/nlohmann_json.cmake)
include(cmake/external/protobuf.cmake)

# find necessary packages
find_package(Python3 COMPONENTS Interpreter)
find_library(c_sec ${THIRD_PARTY})
find_library(protobuf ${THIRD_PARTY})

# sub projects
add_subdirectory(ops/built-in)
add_subdirectory(tbe)
