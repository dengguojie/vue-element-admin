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
find_package(Python3 COMPONENTS Interpreter)
foreach(_file ${OP_FILES})
  set(INI_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/parser/parser_ini.py")
  if(NOT EXISTS "${INI_PARSER}")
    message(FATAL_ERROR "Cannot find the the ini parser ${INI_PARSER}")
  endif()
  get_filename_component(_ini_name "${_file}" NAME)
  string(REPLACE ".ini" ".json" _json_name "${_ini_name}")
  set(_out_json "${CMAKE_CURRENT_BINARY_DIR}/${_json_name}")
  add_custom_command(OUTPUT "${_out_json}"
      COMMAND "${Python3_EXECUTABLE}" "${INI_PARSER}" "${_file}" "${_out_json}" 1>/dev/null
      COMMENT "Parsing ini file ${_ini_name}"
  )
  list(APPEND ALL_JSON_FILES "${_out_json}")
endforeach()

add_custom_target(parse_op_ini ALL
    DEPENDS ${ALL_JSON_FILES}
)
