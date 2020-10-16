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

function(cann_install)
  cmake_parse_arguments(
    MY_INSTALL
    "OPTIONAL"
    "TARGET;DESTINATION"
    "FILES;DIRECTORY"
    ${ARGN}
  )
  if(NOT MY_INSTALL_TARGET)
    message(FATAL_ERROR "Target required for installing files or dirs.")
  endif()

  if(MY_INSTALL_FILES)
    add_custom_command(
      TARGET "${MY_INSTALL_TARGET}" POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${MY_INSTALL_DESTINATION}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MY_INSTALL_FILES} ${MY_INSTALL_DESTINATION}
      COMMENT "Install files: ${MY_INSTALL_FILES} to ${MY_INSTALL_DESTINATION}"
    )
  elseif(MY_INSTALL_DIRECTORY)
    add_custom_command(
      TARGET "${MY_INSTALL_TARGET}" POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${MY_INSTALL_DIRECTORY} ${MY_INSTALL_DESTINATION}
      COMMENT "Install dirs: ${MY_INSTALL_DIRECTORY} to ${MY_INSTALL_DESTINATION}"
    )
  endif()
endfunction()

function(protobuf_generate_cc h_files cc_files output_path)
  if(NOT ARGN)
    message(SEND_ERROR "Error: protobuf_generate() called without any proto files")
    return()
  endif()

  set(${h_files})
  set(${cc_files})

  foreach(_file ${ARGN})
    if(NOT EXISTS ${_file})
      message(SEND_ERROR "Error: files ars not exist ${_file}")
      return()
    endif()
    get_filename_component(_abs_file ${_file} ABSOLUTE)
    get_filename_component(_proto_name ${_abs_file} NAME)
    get_filename_component(_proto_path ${_abs_file} DIRECTORY)
    get_filename_component(_proto_parent_dir ${_proto_path} NAME)

    if("${_proto_parent_dir}" STREQUAL "proto")
      set(_proto_outpath "${output_path}/proto")
    else()
      set(_proto_outpath "${output_path}/proto/${_proto_parent_dir}")
    endif()
    file(MAKE_DIRECTORY ${_proto_outpath})

    string(REPLACE ".proto" ".pb.h"  _proto_pb_h ${_proto_name})
    string(REPLACE ".proto" ".pb.cc" _proto_pb_cc ${_proto_name})
    set(_out_proto_h  "${_proto_outpath}/${_proto_pb_h}")
    set(_out_proto_cc "${_proto_outpath}/${_proto_pb_cc}")
    list(APPEND ${h_files}  ${_out_proto_h})
    list(APPEND ${cc_files} ${_out_proto_cc})

    add_custom_command(
      OUTPUT ${_out_proto_h} ${_out_proto_cc}
      COMMAND ${Protobuf_PROTOC_EXECUTABLE} -I${_proto_path} --cpp_out=${_proto_outpath} ${_file}
      DEPENDS protoc
      COMMENT "Running Protubuf to generate ${_proto_pb_cc} and ${_proto_pb_h}"
    )
  endforeach()

  set(${h_files}  ${${h_files}} PARENT_SCOPE)
  set(${cc_files} ${${cc_files}} PARENT_SCOPE)
  set_source_files_properties(${${h_files}} ${${cc_files}} PROPERTIES GENERATED TRUE)

endfunction()
