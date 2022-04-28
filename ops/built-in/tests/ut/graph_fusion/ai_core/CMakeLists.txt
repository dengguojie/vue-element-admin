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

file(GLOB FUSION_PASS_TC_FILES *.cpp)

add_executable(ops_cpp_fusion_pass_aicore_utest
        ${FUSION_PASS_TC_FILES}
        ${OP_TEST_FUSION_PASS_SOURCE}
        )
target_include_directories(ops_cpp_fusion_pass_aicore_utest PRIVATE
        ${OPS_DIR}/built-in/op_proto/inc
        ${OPS_DIR}/built-in/fusion_pass
        ${OPS_DIR}/../third_party/air/compiler/graphcompiler/engines/nneng
        ${OPS_DIR}/../third_party/metadef/inc
        ${OP_TEST_FUSION_PASS_INCLUDE}
        ${GTEST_INCLUDE}
        ${OP_TEST_PLUFIN_AND_PROTO_INCLUDE}
        )
target_compile_definitions(ops_cpp_fusion_pass_aicore_utest PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        )
if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
   set(compile_opt_mode ${CMAKE_BUILD_MODE})
else()
   set(compile_opt_mode -g)
endif()
target_compile_options(ops_cpp_fusion_pass_aicore_utest PUBLIC
        ${compile_opt_mode}
        -Dgoogle=ascend_private
        )
target_link_libraries(ops_cpp_fusion_pass_aicore_utest PRIVATE
        -Wl,--whole-archive
        ops_fusion_pass_aicore_llt
        opsproto_llt
        -Wl,--no-as-needed
        runtime
        ascend_hal
        mmpa
        error_manager
        c_sec
        ascend_protobuf
        -Wl,--as-needed
        -Wl,--no-whole-archive
        -Wl,--allow-multiple-definition
        )
if(NOT UT_NO_EXEC)
        add_custom_command(
                TARGET ops_cpp_fusion_pass_aicore_utest POST_BUILD
                COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${ASCEND_RUNTIME_DIR} && ./ops_cpp_fusion_pass_aicore_utest
                COMMENT "Run ops fusion_pass utest"
        )
endif()
