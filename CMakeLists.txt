cmake_minimum_required(VERSION 3.14)

project(cann C CXX)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_FOUND})
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_FOUND})
endif(CCACHE_FOUND)

#switch build mode
if(BUILD_MODE)
  message(${BUILD_MODE})
else()
  set(BUILD_MODE "FALSE")
endif()
if(${BUILD_MODE} STREQUAL "FALSE")
  set(CMAKE_BUILD_MODE "${BUILD_MODE}")
else()
  set(CMAKE_BUILD_MODE "-${BUILD_MODE}")
endif()

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
  if(UT_TEST_ALL AND ST_TEST)
    set(ENABLE_TEST ALL)
  elseif(UT_TEST_ALL OR  CPU_UT OR
         PASS_UT OR TILING_UT OR
         PROTO_UT OR PLUGIN_UT OR
         ONNX_PLUGIN_UT)
    set(ENABLE_TEST UT)
  elseif(ST_TEST)
    set(ENABLE_TEST ST)
  endif()

  if(NOT ENABLE_TEST)
      add_compile_options(-Werror=sign-compare -Werror=address -Werror=unused-variable
                          -Werror=narrowing -Werror=parentheses -Werror=unused-but-set-variable
                          -Werror=return-type)
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
  include(scripts/script.cmake)

  if(DEFINED ENV{TOOLCHAIN_DIR})
    set(TOOLCHAIN_DIR $ENV{TOOLCHAIN_DIR})
  else()
    if(MINRC)
      set(TOOLCHAIN_DIR /usr)
    else()
      set(TOOLCHAIN_DIR ${ASCEND_DIR}/toolkit/toolchain/hcc)
    endif()
  endif()

  if(NOT "x${TOOLCHAIN_DIR}" STREQUAL "x")
    if(NOT IS_DIRECTORY ${TOOLCHAIN_DIR})
      message(FATAL_ERROR "specify cross compile toolchain directroy(${TOOLCHAIN_DIR}) is not exist")
    endif()
  endif()
  message(STATUS "specify cross compile toolchain directroy=${TOOLCHAIN_DIR}")
  include(cmake/external/protobuf_static.cmake)

  if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(AICPU_OPP_ENV $ENV{ASCEND_CUSTOM_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel)
  else()
    set(AICPU_OPP_ENV /usr/local/Ascend/opp/op_impl/built-in/aicpu/aicpu_kernel)
  endif()

  # find necessary packages
  find_package(Python3 3.7 COMPONENTS Interpreter)
  set(HI_PYTHON ${Python3_EXECUTABLE})

  # includes path
  set(OPS_DIR ${CANN_ROOT}/ops)
  set(OPS_INCLUDE ${CANN_ROOT}/ops/common/inc)
  set(FWKACLLIB_INCLUDE ${CANN_ROOT}/third_party/fwkacllib/inc)
  set(ATC_INCLUDE ${CANN_ROOT}/third_party/atc/inc)
  set(METADEF_INCLUDE ${CANN_ROOT}/third_party/metadef/inc)
  set(METADEF_DIR ${CANN_ROOT}/third_party/metadef)
  set(GRAPHENGINE_INCLUDE ${CANN_ROOT}/third_party/metadef/third_party/graphengine/inc)
  set(FUSION_ENGINE_INCLUDE ${CANN_ROOT}/third_party/air/compiler/graphcompiler/engines/nneng/inc)

  if("x${PRODUCT_SIDE}" STREQUAL "xdevice")
    message("Add device project")
    if("x${ENABLE_TEST}" STREQUAL "x")
      add_subdirectory(ops/built-in/aicpu/context)
      add_subdirectory(ops/built-in/aicpu/impl)
    endif()
  else()
    message("Add host project")
    add_subdirectory(ops)
    if(("x${AICPU_ONLY}" STREQUAL "xFALSE") AND ("x${ENABLE_TEST}" STREQUAL "x"))
      add_subdirectory(auto_schedule)
      add_subdirectory(tools)
    endif()
  endif()
else()
  # includes path
  set(JSON_INCLUDE "${TOP_DIR}/third_party/json/include ${TOP_DIR}/open_source/json/include")
  set(Protobuf_INCLUDE "${TOP_DIR}/third_party/protobuf/include ${TOP_DIR}/open_source/protobuf/include")
  set(OPS_DIR ${TOP_DIR}/asl/ops/cann/ops)
  set(OPS_INCLUDE ${TOP_DIR}/asl/ops/cann/ops/common/inc)
  set(FWKACLLIB_INCLUDE ${TOP_DIR}/inc)
  set(ATC_INCLUDE ${TOP_DIR}/third_party/atc/inc)
  set(METADEF_INCLUDE ${TOP_DIR}/metadef/inc)
  set(METADEF_DIR ${TOP_DIR}/metadef)
  set(GRAPHENGINE_INCLUDE ${TOP_DIR}/graphengine/inc)
  set(FUSION_ENGINE_INCLUDE ${TOP_DIR}/air/compiler/graphcompiler/engines/nneng/inc)
  add_subdirectory(ops)

endif()
