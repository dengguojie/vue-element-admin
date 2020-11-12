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

set(OP_TEST_COMMON_SOURCE
  ${OPS_DIR}/tests/common/utils_plugin_and_op_proto/op_proto_test_util.cpp
)

set(OP_TEST_COMMON_INCLUDE
  ${OPS_DIR}/tests/common/utils_plugin_and_op_proto
)

if(ENABLE_TEST STREQUAL UT)
  add_subdirectory(ut)
endif()
