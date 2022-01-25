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

set(_eigen_url "")
if(CANN_PKG_SERVER)
  set(_eigen_url "${CANN_PKG_SERVER}/libs/eigen3/eigen-3.3.9.tar.gz")
endif()

ExternalProject_Add(eigen
  URL               ${_eigen_url}
                    https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
  URL_MD5           609286804b0f79be622ccf7f9ff2b660
  DOWNLOAD_DIR      download/eigen
  PREFIX            third_party
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(eigen SOURCE_DIR)
ExternalProject_Get_Property(eigen BINARY_DIR)

set(EIGEN_INCLUDE ${SOURCE_DIR})

add_custom_target(eigen_headers ALL DEPENDS eigen)
