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
      COMMAND ${CMAKE_COMMAND} -E make_directory "${MY_INSTALL_DESTINATION}"
      COMMAND ${CMAKE_COMMAND} -E copy "${MY_INSTALL_FILES}" "${MY_INSTALL_DESTINATION}"
      COMMENT "Install files: ${MY_INSTALL_FILES} to ${MY_INSTALL_DESTINATION}"
    )
  elseif(MY_INSTALL_DIRECTORY)
    add_custom_command(
      TARGET "${MY_INSTALL_TARGET}" POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${MY_INSTALL_DIRECTORY}" "${MY_INSTALL_DESTINATION}"
      COMMENT "Install dirs: ${MY_INSTALL_DIRECTORY} to ${MY_INSTALL_DESTINATION}"
    )
  endif()
endfunction()
