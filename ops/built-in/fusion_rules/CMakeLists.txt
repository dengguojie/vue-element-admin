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

set(FUSION_FULES_INSTALL_PATH "${INSTALL_PATH}/opp/fusion_fules/built-in")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ai_core")
  add_custom_target(copy_aicore_fusion_rules ALL)
  set(TBE_OP_PATH "${INSTALL_PATH}/opp/op_impl/built-in/ai_core/tbe")
  cann_install(
    TARGET      copy_aicore_fusion_rules
    DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}/ai_core"
    DESTINATION "${FUSION_FULES_INSTALL_PATH}/ai_core"
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vector_core")
  add_custom_target(copy_veccore_fusion_rules ALL)
  set(TBE_OP_PATH "${INSTALL_PATH}/opp/op_impl/built-in/ai_core/tbe")
  cann_install(
    TARGET      copy_veccore_fusion_rules
    DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}/vector_core"
    DESTINATION "${FUSION_FULES_INSTALL_PATH}/vector_core"
  )
endif()
