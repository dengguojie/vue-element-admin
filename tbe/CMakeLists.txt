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

set(TBE_WHL "te-0.4.0-py3-none-any.whl")
set(TBE_WHL_FULL "${CMAKE_CURRENT_BINARY_DIR}/${TBE_WHL}")

if(EXISTS "${TBE_WHL_FULL}")
  # unzip first
  set(_tmp_path "${CMAKE_CURRENT_BINARY_DIR}/python")
  add_custom_target(unzip_tbe ALL)
  add_custom_command(TARGET unzip_tbe PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${_tmp_path}"
    COMMAND unzip -q -o "${TBE_WHL_FULL}" -d "${_tmp_path}"
    COMMENT "Unpacking ${TBE_WHL}"
  )

  # copy and replace
  set(_purelib_path "${_tmp_path}/te-0.4.0.data/purelib")
  set(_pkg_path "${_purelib_path}/te")
  add_custom_target(copy_tbe ALL DEPENDS unzip_tbe)
  add_custom_command(TARGET copy_tbe
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/python/auto_schedule" "${_pkg_path}"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py" "${_purelib_path}"
  )

  # repackaging
  add_custom_target(repack_tbe ALL DEPENDS copy_tbe)
  add_custom_command(TARGET repack_tbe
    COMMAND "${Python3}" "${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py" bdist_wheel
    COMMENT "Repacking ${TBE_WHL}"
  )

  # install
  set(TBE_ATC_PATH "${INSTALL_PATH}/atc/lib64")
  set(TBE_FWK_PATH "${INSTALL_PATH}/fwkacllib/lib64")
  cann_install(
    TARGET      repack_tbe
    FILES       "${CMAKE_CURRENT_BINARY_DIR}/dist/${TBE_WHL}"
    DESTINATION "${TBE_ATC_PATH}"
  )
  cann_install(
    TARGET      repack_tbe
    FILES       "${CMAKE_CURRENT_BINARY_DIR}/dist/${TBE_WHL}"
    DESTINATION "${TBE_FWK_PATH}"
  )
endif()
