cmake_minimum_required(VERSION 3.14)

project(cann C CXX)

# set variables for cann project
set(CANN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CANN_ROOT=${CANN_ROOT}")

# configurations
if(EXISTS "${BUILD_PATH}/config.cmake")
  # the current config
  include("${BUILD_PATH}/config.cmake")
else()
  # the template config
  include("${CANN_ROOT}/cmake/template/config.cmake")
endif()

option(BUILD_OPEN_PROJECT "Build open canndev project." FALSE)

# the install path for cann project
if(NOT BUILD_PATH)
  set(BUILD_PATH "${CMAKE_BINARY_DIR}")
endif()
set(INSTALL_PATH "${BUILD_PATH}/install")
set(INSTALL_LIBRARY_DIR lib)
set(PROTO_BINARY_DIR "${BUILD_PATH}/proto")
set(ENABLE_TEST "")

# whether building open project
if(BUILD_OPEN_PROJECT)
  if(CANN_GCC_PREFIX)
    set(CMAKE_C_COMPILER ${CANN_GCC_PREFIX}gcc)
    set(CMAKE_CXX_COMPILER ${CANN_GCC_PREFIX}g++)
    set(CMAKE_LINKER ${CANN_GCC_PREFIX}ld)
    set(CMAKE_AR ${CANN_GCC_PREFIX}gcc-ar)
  endif()

  # tests flag
  if(UT_TEST AND ST_TEST)
    set(ENABLE_TEST ALL)
  elseif(UT_TEST)
    set(ENABLE_TEST UT)
  elseif(ST_TEST)
    set(ENABLE_TEST ST)
  endif()

  # build external prjects
  if(DEFINED ENV{D_PKG_SERVER})
    set(CANN_PKG_SERVER $ENV{D_PKG_SERVER})
    message("Download packages from PKG server")
  endif()

  if(WIN32)
    include(cmake/util/intf_pub_windows.cmake)
  elseif(UNIX)
    include(cmake/util/intf_pub_linux.cmake)
    if(ENABLE_TEST)
      include(cmake/util/intf_pub_llt_gccnative.cmake)
    endif()
  else()
    message(FATAL_ERROR "System not supported.")
  endif()

  # build external prjects
  include(ExternalProject)
  include(cmake/util/util.cmake)
  include(cmake/dependencies.cmake)
  include(cmake/external/secure_c.cmake)
  include(cmake/external/nlohmann_json.cmake)
  include(cmake/external/protobuf.cmake)
  include(cmake/external/gtest.cmake)
  include(cmake/external/eigen.cmake)

  # find necessary packages
  find_package(Python3 3.7 COMPONENTS Interpreter)
  set(HI_PYTHON ${Python3_EXECUTABLE})

  # includes path
  set(OPS_DIR ${CANN_ROOT}/ops)
  set(OPS_INCLUDE ${CANN_ROOT}/ops/common/inc)
  set(FWKACLLIB_INCLUDE ${CANN_ROOT}/third_party/fwkacllib/inc)
  set(METADEF_INCLUDE ${CANN_ROOT}/third_party/metadef/inc)
  set(METADEF_DIR ${CANN_ROOT}/third_party/metadef)
  set(GRAPHENGINE_INCLUDE ${CANN_ROOT}/third_party/metadef/third_party/graphengine/inc)

  add_subdirectory(ops)
  add_subdirectory(auto_schedule)
  add_subdirectory(tools)
  if(ENABLE_TEST)
    add_subdirectory(ops/tests)
  endif()

else()
  # includes path
  set(JSON_INCLUDE "${TOP_DIR}/third_party/json/include")
  # for compatibility in open project
  add_custom_target(json_headers ALL)
  set(Protobuf_INCLUDE "${TOP_DIR}/third_party/protobuf/include")
  set(OPS_DIR ${TOP_DIR}/cann/ops)
  set(OPS_INCLUDE ${TOP_DIR}/cann/ops/common/inc)
  set(FWKACLLIB_INCLUDE ${TOP_DIR}/inc)
  set(METADEF_INCLUDE ${TOP_DIR}/metadef/inc)
  set(METADEF_DIR ${TOP_DIR}/metadef)
  set(GRAPHENGINE_INCLUDE ${TOP_DIR}/graphengine/inc)

  add_subdirectory(ops)

endif()
