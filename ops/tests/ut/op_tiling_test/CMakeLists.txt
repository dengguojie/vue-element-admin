# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if(BUILD_OPEN_PROJECT)
  file(GLOB OP_TILING_TC_FILES *.cpp)
  
  add_executable(ops_cpp_op_tiling_utest
    ${OP_TILING_TC_FILES}
    ${OP_TEST_PLUFIN_AND_PROTO_SOURCE}
  )
  
  target_include_directories(ops_cpp_op_tiling_utest PRIVATE
    ${GTEST_INCLUDE}
    ${OP_TEST_PLUFIN_AND_PROTO_INCLUDE}
  )
  
  target_compile_definitions(ops_cpp_op_tiling_utest PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
  )
  
  target_compile_options(ops_cpp_op_tiling_utest PUBLIC
    -O0
    -Dgoogle=ascend_private
  )
  
  target_link_libraries(ops_cpp_op_tiling_utest PRIVATE
    -Wl,--whole-archive
      optiling_llt
    -Wl,--no-whole-archive
  )
  
  add_custom_command(
    TARGET  ops_cpp_op_tiling_utest POST_BUILD
    COMMAND ops_cpp_op_tiling_utest
    COMMENT "Run ops op_tiling utest"
  )
else()
  set(CMAKE_VERBOSE_MAKEFILE on)
  include_directories(${OPS_SOURCE_CODE_ROOT}/../)
  include_directories(${OPS_SOURCE_CODE_ROOT}/../common/inc)
  include_directories(${OPS_SOURCE_CODE_ROOT}/op_tiling)
  
  include_directories(${REPO_ROOT}/libc_sec/include)
  include_directories(${REPO_ROOT}/third_party/json/include)
  include_directories(${REPO_ROOT}/third_party/protobuf/include)
  
  # fix inc
  include_directories(${INC_FIX_ROOT})
  
  
  aux_source_directory(${OPS_SOURCE_CODE_ROOT}/op_tiling OP_TILING_SOURCE)
  set(OP_TILING_SOURCE ${OP_TILING_SOURCE}
          ${OPS_SOURCE_CODE_ROOT}/fusion_pass/common/fp16_t.cpp
          ${OPS_SOURCE_CODE_ROOT}/op_proto/util/error_util.cpp)
  
  add_library(op_tiling_source SHARED ${OP_TILING_SOURCE})
  
  enable_testing()
  set(GTEST_ROOT ${TOP_DIR}/llt/third_party/gtest/googletest)
  
  include_directories(${GTEST_ROOT}/include)
  
  message(STATUS "OPS_SOURCE_CODE_ROOT======:${OPS_SOURCE_CODE_ROOT}")
  include_directories(${OPS_SOURCE_CODE_ROOT}/op_tiling)
  include_directories(${REPO_ROOT}/common)
  include_directories(${INC_FIX_ROOT}/external/graph)
  include_directories(${INC_FIX_ROOT}/external)
  include_directories(${INC_FIX_ROOT})
  # for om.pb.h
  include_directories(${OP_TEST_ROOT})
  
  message(STATUS "OP_TEST_ROOT========:${OP_TEST_ROOT}")
  file(GLOB PLUGIN_TILING_TC_FILES *.cpp)
  
  add_executable(ops_op_tiling_test ${PLUGIN_TILING_TC_FILES})
  
  target_link_libraries (ops_op_tiling_test gtest c_sec slog error_manager graph register op_tiling_source)
endif()
