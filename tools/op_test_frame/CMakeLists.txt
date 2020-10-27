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

aux_source_directory(src/model_run_tool ST_TOOL_SRC)

add_library(model_run_tool SHARED ${ST_TOOL_SRC})

target_compile_definitions(model_run_tool 
  PRIVATE
  -D_GLIBCXX_USE_CXX11_ABI=0
)

target_include_directories(model_run_tool
  PUBLIC
  src/model_run_tool
  ${CANN_ROOT}/third_party/fwkacllib/inc
)


set(op_test_frame_root ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(op_test_frame.whl ALL DEPENDS model_run_tool)

add_custom_command(TARGET op_test_frame.whl
  PRE_BUILD
  COMMAND echo ${op_test_frame_root}
  COMMAND rm -rf ${op_test_frame_root}/python/op_test_frame/libs
  COMMAND ${CMAKE_COMMAND} -E make_directory ${op_test_frame_root}/python/op_test_frame/libs
  COMMAND cp libmodel_run_tool.so ${op_test_frame_root}/python/op_test_frame/libs
  COMMAND ${op_test_frame_root}/python/build_whl.sh ${Python3_EXECUTABLE} ${op_test_frame_root}/python
  COMMAND cp ${op_test_frame_root}/python/dist/* ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND rm -rf ${op_test_frame_root}/python/build
  COMMAND rm -rf ${op_test_frame_root}/python/op_test_frame.egg-info
  COMMAND rm -rf ${op_test_frame_root}/python/dist
)

set(OP_TEST_FRAME ${INSTALL_PATH}/toolkit/tools)
cann_install(
  TARGET      op_test_frame.whl
  FILES       ${CMAKE_CURRENT_BINARY_DIR}/*.whl
  DESTINATION ${OP_TEST_FRAME}
)
