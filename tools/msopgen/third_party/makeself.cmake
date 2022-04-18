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
include(ExternalProject)

set(MAKESLF_PATH ${CMAKE_CURRENT_SOURCE_DIR}/op_gen/template/op_project_tmpl/cmake/util/makeself)

ExternalProject_Add(makeself_third
  URL               ${_makeself_url}
                    https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/makeself/release-2.4.2.zip
  PREFIX            ${MAKESLF_PATH}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(makeself_third SOURCE_DIR)
ExternalProject_Get_Property(makeself_third BINARY_DIR)

add_custom_target(makeself_code ALL DEPENDS makeself_third
                 COMMAND cp ${MAKESLF_PATH}/src/makeself_third/COPYING ${MAKESLF_PATH}
                 COMMAND cp ${MAKESLF_PATH}/src/makeself_third/makeself.1 ${MAKESLF_PATH}
                 COMMAND cp ${MAKESLF_PATH}/src/makeself_third/makeself.lsm ${MAKESLF_PATH}
                 COMMAND cp ${MAKESLF_PATH}/src/makeself_third/*.sh ${MAKESLF_PATH}
                 COMMAND cp ${MAKESLF_PATH}/src/makeself_third/README.md ${MAKESLF_PATH}
                 COMMAND cp ${MAKESLF_PATH}/src/makeself_third/VERSION ${MAKESLF_PATH}
                 COMMAND rm -rf ${MAKESLF_PATH}/src
                 COMMAND rm -rf ${MAKESLF_PATH}/tmp)