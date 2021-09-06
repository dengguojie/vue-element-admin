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
file(GLOB PLUGIN_TC_FILES *plugin.cpp)

add_executable(ops_cpp_plugin_utest
        ${PLUGIN_TC_FILES}
        )

target_include_directories(ops_cpp_plugin_utest PRIVATE
	${GTEST_INCLUDE}
        ${METADEF_INCLUDE}/graph
        ${ATC_INCLUDE}/parser
        )

target_compile_definitions(ops_cpp_plugin_utest PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        )

if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
   set(compile_opt_mode ${CMAKE_BUILD_MODE})
else()
   set(compile_opt_mode -O0)
endif()

target_compile_options(ops_cpp_plugin_utest PUBLIC
        ${compile_opt_mode}
        -Dgoogle=ascend_private
        )

target_link_libraries(ops_cpp_plugin_utest PRIVATE
        -Wl,--whole-archive
        opsplugin_llt
        opsproto_llt
        -Wl,--no-whole-archive
        fmk_parser
        )

if(NOT UT_NO_EXEC)
        add_custom_command(
                TARGET ops_cpp_plugin_utest POST_BUILD
                COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${ASCEND_ATC_DIR} && ./ops_cpp_plugin_utest
                COMMENT "Run ops plugin utest"
        )
endif()
