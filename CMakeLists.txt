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
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# include useful funcs
include(cmake/util.cmake)
# build external prjects
include(ExternalProject)
include(cmake/external/secure_c.cmake)
include(cmake/external/nlohmann_json.cmake)
include(cmake/external/protobuf.cmake)

# find necessary packages
find_package(Python3 3.7 COMPONENTS Interpreter)

# if D_LINK_PATH is set in environment variables, search libraries in given path
if(DEFINED ENV{D_LINK_PATH})
  # D_LINK_PATH is set
  set(CANN_LIB_PATH $ENV{D_LINK_PATH})
  message("Find env D_LINK_PATH, search libs under ${CANN_LIB_PATH}")
  find_library(slog libslog.so ${CANN_LIB_PATH})
  find_library(register libregister.so ${CANN_LIB_PATH})
  find_library(graph libgraph.so ${CANN_LIB_PATH})
  find_library(error_manager liberror_manager.so ${CANN_LIB_PATH})
  set(tbe_whl ${CANN_LIB_PATH}/te-0.4.0-py3-none-any.whl)
else()
  # Ascend mode
  if(DEFINED ENV{ASCEND_CUSTOM_PATH})
      set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
  else()
      set(ASCEND_DIR /usr/local/Ascend)
  endif()
  message("Search libs under install path ${ASCEND_DIR}")
  find_library(slog libslog.so ${ASCEND_DIR})
  find_library(register libregister.so ${ASCEND_DIR})
  find_library(graph libgraph.so ${ASCEND_DIR})
  find_library(error_manager liberror_manager.so ${ASCEND_DIR})
  find_file(tbe_whl te-0.4.0-py3-none-any.whl ${ASCEND_DIR})
endif()

# sub projects
add_subdirectory(ops/built-in)
add_subdirectory(auto_schedule)
add_subdirectory(tools)
