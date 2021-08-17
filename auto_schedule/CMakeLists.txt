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

if(EXISTS "${tbe_whl}")
  get_filename_component(_tbe_whl_name "${tbe_whl}" NAME)
  set(AUTO_SCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

  # unzip first
  set(_tmp_path "${CMAKE_CURRENT_BINARY_DIR}/python")
  add_custom_target(unzip_tbe ALL)
  add_custom_command(TARGET unzip_tbe PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${_tmp_path}"
    COMMAND unzip -q -o "${tbe_whl}" -d "${_tmp_path}"
    COMMENT "Unpacking ${_tbe_whl_name}"
  )

  # copy and replace
  set(_purelib_path "${_tmp_path}/te-0.4.0.data/purelib")
  set(_pkg_path "${_purelib_path}/te")
  set(_pkg_path_tbe "${_purelib_path}/tbe")
  add_custom_target(copy_tbe ALL DEPENDS unzip_tbe)
  add_custom_command(TARGET copy_tbe
    COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/python/lang ${_pkg_path}
    COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/python/utils ${_pkg_path}
    COMMAND mkdir -p ${_pkg_path_tbe}
    COMMAND mkdir -p ${_pkg_path_tbe}/common
    COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/python/tbe/dsl ${_pkg_path_tbe}
    COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/python/tbe/common/utils ${_pkg_path_tbe}/common
    COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/python/tbe/common/context ${_pkg_path_tbe}/common
    COMMAND ${CMAKE_COMMAND} -E copy "${AUTO_SCH_PATH}/setup.py" "${AUTO_SCH_PATH}/LICENSE" "${_purelib_path}"
    COMMENT "Copying auto schedule"
  )

  # repackaging
  add_custom_target(repack_tbe ALL DEPENDS copy_tbe)
  add_custom_command(TARGET repack_tbe
    COMMAND ${CMAKE_COMMAND} -E remove_directory dist
    COMMAND ${CMAKE_COMMAND} -E chdir "${_purelib_path}" "${HI_PYTHON}" "setup.py" bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${_purelib_path}/dist" "${CMAKE_CURRENT_BINARY_DIR}/dist"
    COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/dist/*.whl ${CMAKE_CURRENT_BINARY_DIR}/dist/${_tbe_whl_name}
    COMMENT "Repacking ${_tbe_whl_name}"
  )

  # install
  if("$ENV{ALL_IN_ONE_ENABLE}" STREQUAL "1")
    set(TBE_COMPILER_PATH "${INSTALL_PATH}/compiler/lib64")
    cann_install(
      TARGET      repack_tbe
      FILES       "${CMAKE_CURRENT_BINARY_DIR}/dist/${_tbe_whl_name}"
      DESTINATION "${TBE_COMPILER_PATH}"
    )
  else()
    set(TBE_ATC_PATH "${INSTALL_PATH}/atc/lib64")
    set(TBE_FWK_PATH "${INSTALL_PATH}/fwkacllib/lib64")
    cann_install(
      TARGET      repack_tbe
      FILES       "${CMAKE_CURRENT_BINARY_DIR}/dist/${_tbe_whl_name}"
      DESTINATION "${TBE_ATC_PATH}"
    )
    cann_install(
      TARGET      repack_tbe
      FILES       "${CMAKE_CURRENT_BINARY_DIR}/dist/${_tbe_whl_name}"
      DESTINATION "${TBE_FWK_PATH}"
    )
  endif()

endif()
