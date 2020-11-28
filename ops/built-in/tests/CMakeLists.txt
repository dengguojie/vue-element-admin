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
  set(OPS_BUILTIN_TEST_DIR ${OPS_DIR}/built-in/tests)
  set(OP_TEST_PLUFIN_AND_PROTO_SOURCE
    ${OPS_BUILTIN_TEST_DIR}/common/utils_plugin_and_op_proto/op_proto_test_util.cpp
  )
  set(OP_TEST_PLUFIN_AND_PROTO_INCLUDE
    ${OPS_BUILTIN_TEST_DIR}/common/utils_plugin_and_op_proto
  )
  set(OP_TEST_FUSION_PASS_SOURCE
    ${OPS_BUILTIN_TEST_DIR}/common/src/fusion_pass_utils/fusion_pass_test_utils.cpp
  )
  set(OP_TEST_FUSION_PASS_INCLUDE
    ${OPS_BUILTIN_TEST_DIR}/common/src/inc
  )
  if(ENABLE_TEST STREQUAL UT)
    add_subdirectory(ut)
    # /path/to/canndev/build/cann/ops/built-in
    get_filename_component(_ops_builtin_bin_path ${CMAKE_CURRENT_BINARY_DIR} DIRECTORY)
    set(_ops_cov_report ${BUILD_PATH}/test/ut/cov_report/ops)
    set(_ops_cov_html ${_ops_cov_report}/cpp_utest)
    set(_ops_cov_data ${_ops_cov_report}/cpp_utest/ops.info)
    # generate coverage
    add_custom_target(generate_ops_cpp_cov ALL
      DEPENDS ops_cpp_proto_utest
              ops_cpp_op_tiling_utest
              ops_cpp_fusion_pass_aicore_utest
              cpu_kernels_ut            
      COMMENT "Generate cpp coverage for ops utest"
    )
    add_custom_command(
      TARGET generate_ops_cpp_cov
      COMMAND ${GENERATE_CPP_COV} ${_ops_builtin_bin_path} ${_ops_cov_data} ${_ops_cov_html}
    )
  endif()
else()
  cmake_minimum_required(VERSION 3.5.1)
  project(ops)
  option(OPS_UT_YELLOW "Run ops utest in Yellow." TRUE)
  add_subdirectory(stub/slog)
  add_subdirectory(ut)
endif()
