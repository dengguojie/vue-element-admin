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

set(OP_TEST_PLUFIN_AND_PROTO_SOURCE
  ${OPS_DIR}/tests/common/utils_plugin_and_op_proto/op_proto_test_util.cpp
)

set(OP_TEST_PLUFIN_AND_PROTO_INCLUDE
  ${OPS_DIR}/tests/common/utils_plugin_and_op_proto
)

set(OP_TEST_FUSION_PASS_SOURCE
  ${OPS_DIR}/tests/common/src/fusion_pass_utils/fusion_pass_test_utils.cpp
)

set(OP_TEST_FUSION_PASS_INCLUDE
  ${OPS_DIR}/tests/common/src/inc
)

if(ENABLE_TEST STREQUAL UT)
  add_subdirectory(ut)

  # /path/to/canndev/build/cann/ops
  get_filename_component(_ops_bin_path ${CMAKE_CURRENT_BINARY_DIR} DIRECTORY)
  # /path/to/canndev/ops/tests
  set(_tests_source_path ${CMAKE_CURRENT_SOURCE_DIR})

  set(_ops_cov_report cov_report/ops)
  set(_ops_cov_html ${_tests_source_path}/${_ops_cov_report}/cpp_utest)
  set(_ops_cov_data ${_tests_source_path}/${_ops_cov_report}/cpp_utest/ops.info)

  # generate coverage
  add_custom_target(generate_ops_cpp_cov ALL
    DEPENDS ops_cpp_proto_utest
            ops_cpp_op_tiling_utest
            ops_cpp_fusion_pass_aicore_utest
    COMMENT "Generate cpp coverage for ops utest"
  )
  add_custom_command(
    TARGET generate_ops_cpp_cov
    COMMAND ${GENERATE_CPP_COV} ${_ops_bin_path}/built-in ${_ops_cov_data} ${_ops_cov_html}
  )
endif()
