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

file(GLOB OP_TILING_TC_FILES *.cpp)

if(BUILD_OPEN_PROJECT)
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
elseif(OPS_UT_YELLOW)
  file(GLOB_RECURSE op_tiling_srcs "${TOP_DIR}/cann/ops/built-in/op_tiling/*.cc")
  set(src_files
    ${TOP_DIR}/cann/ops/built-in/fusion_pass/common/fp16_t.cc
    ${TOP_DIR}/cann/ops/built-in/op_proto/util/error_util.cc
    ${op_tiling_srcs}
  )
  add_executable(ut_ops_cpp_op_tiling
    ${src_files}
    ${OP_TILING_TC_FILES}
  )
  target_include_directories(ut_ops_cpp_op_tiling PRIVATE
    ${TOP_DIR}/inc
    ${TOP_DIR}/common
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/metadef
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/metadef/inc/external
    ${TOP_DIR}/metadef/inc/external/graph
    ${TOP_DIR}/graphengine/inc
    ${TOP_DIR}/graphengine/inc/external
    ${TOP_DIR}/graphengine/inc/external/ge
    ${TOP_DIR}/graphengine/inc/framework
    ${TOP_DIR}/third_party/protobuf/include
    ${TOP_DIR}/lib_sec/include
    ${TOP_DIR}/cann/ops/common/inc
    ${TOP_DIR}/third_party/json/include
    ${TOP_DIR}/cann/ops/built-in
  )
  target_compile_options(ut_ops_cpp_op_tiling PRIVATE
    -g3
    -O0
  )
  target_compile_definitions(ut_ops_cpp_op_tiling PRIVATE
    CFG_BUILD_DEBUG
    google=ascend_private
  )
  target_link_libraries(ut_ops_cpp_op_tiling PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    -Wl,--no-as-needed
      c_sec
      ascend_protobuf
      graph
      error_manager
      register
      slog
    -Wl,--no-as-needed
  )
  run_llt_test(
    TARGET   ut_ops_cpp_op_tiling
    TASK_NUM 1
  )
endif()
