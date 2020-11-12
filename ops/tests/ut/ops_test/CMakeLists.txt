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

file(GLOB PLUGIN_PROTO_TC_FILES **/*proto.cpp)

add_executable(ops_cpp_proto_utest
  ${PLUGIN_PROTO_TC_FILES}
  ${OP_TEST_COMMON_SOURCE}
)

target_include_directories(ops_cpp_proto_utest PRIVATE
  ${OP_TEST_COMMON_INCLUDE}
  ${METADEF_INCLUDE}/graph
)

target_compile_definitions(ops_cpp_proto_utest PRIVATE
  _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(ops_cpp_proto_utest PUBLIC
  -O0
  -Dgoogle=ascend_private
)

target_link_libraries(ops_cpp_proto_utest PRIVATE
  -Wl,--whole-archive
    opsproto_llt
  -Wl,--no-whole-archive
)

add_custom_command(
  TARGET  ops_cpp_proto_utest POST_BUILD
  COMMAND ops_cpp_proto_utest
  COMMENT "Run ops proto utest"
)
