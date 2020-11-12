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

# convert ini to json
set(ALL_JSON_FILES "")
file(GLOB_RECURSE OP_FILES "op_info_cfg/*.ini")
foreach(_file ${OP_FILES})
  set(INI_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/parser/parser_ini.py")
  if(NOT EXISTS "${INI_PARSER}")
    message(FATAL_ERROR "Cannot find the the ini parser ${INI_PARSER}")
  endif()

  string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/ "" _rel_path "${_file}")
  get_filename_component(_ini_name "${_rel_path}" NAME)
  get_filename_component(_parent_dir "${_rel_path}" DIRECTORY)
  string(REPLACE ".ini" ".json" _json_name "${_ini_name}")
  set(_out_path "${CMAKE_CURRENT_BINARY_DIR}/config/${_parent_dir}")
  file(MAKE_DIRECTORY "${_out_path}")

  set(_out_json "${_out_path}/${_json_name}")
  add_custom_command(OUTPUT "${_out_json}"
    COMMAND "${HI_PYTHON}" "${INI_PARSER}" "${_file}" "${_out_json}" 1>/dev/null
    COMMENT "Parsing ini file ${_ini_name}"
  )
  list(APPEND ALL_JSON_FILES "${_out_json}")
endforeach()

set(TBE_OP_PATH "${INSTALL_PATH}/opp/op_impl/built-in")

add_custom_target(tbe_ops_json_info ALL
  DEPENDS ${ALL_JSON_FILES}
)
if(BUILD_OPEN_PROJECT)
  add_custom_target(tbe_aicore_ops_impl ALL)
  cann_install(
    TARGET      tbe_aicore_ops_impl
    DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/impl"
    DESTINATION "${TBE_OP_PATH}/ai_core/tbe"
  )
  cann_install(
    TARGET      tbe_ops_json_info
    COPY_CONTENTS
    DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/config/ai_core"
    DESTINATION "${TBE_OP_PATH}/ai_core/tbe/config"
  )
  cann_install(
    TARGET      tbe_ops_json_info
    FILES       "${CMAKE_CURRENT_BINARY_DIR}/config/vector_core/vector_core_tbe_ops_info.json"
    DESTINATION "${TBE_OP_PATH}/vector_core/tbe/config"
  )
else()
  install(
    FILES       ${ALL_JSON_FILES}
    DESTINATION ${INSTALL_LIBRARY_DIR}
    OPTIONAL
  )
endif()
