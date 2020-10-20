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

set(_ut_proto_files
  ops_test/max_pool_v3/test_max_pool_v3_proto.cpp
)

add_executable(ops_cpp_proto_utest
  ${_ut_proto_files}
)

target_include_directories(ops_cpp_proto_utest PRIVATE
  ${CANN_ROOT}/ops/common/inc
  ${CANN_ROOT}/ops/built-in/op_proto
  ${CANN_ROOT}/inc
  ${CANN_ROOT}/inc/external
  ${CANN_ROOT}/third_party/fwkacllib/inc
  ${CANN_ROOT}/third_party/metadef/inc
  ${CANN_ROOT}/third_party/metadef/inc/external
  ${CANN_ROOT}/third_party/metadef/inc/external/graph
  ${GTEST_INCLUDE}
)

target_link_libraries(ops_cpp_proto_utest PRIVATE
  opsproto
  ${graph}
  ${regiser}
  ${slog}
  ${error_mangager}
  ${protobuf}
  c_sec
  gtest
  gtest_main
)

set(OP_PROTO_UTEST "${INSTALL_PATH}/tests/ops/ut/op_proto")
cann_install(
  TARGET      ops_cpp_proto_utest
  FILES       $<TARGET_FILE:ops_cpp_proto_utest>
  DESTINATION "${OP_PROTO_UTEST}"
)
