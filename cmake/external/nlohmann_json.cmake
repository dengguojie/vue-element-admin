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

ExternalProject_Add(nlohmann_json
  URL               https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip
  URL_MD5           0dc903888211db3a0f170304cd9f3a89
  DOWNLOAD_DIR      download/nlohmann_json
  PREFIX            third_party
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(nlohmann_json SOURCE_DIR)
ExternalProject_Get_Property(nlohmann_json BINARY_DIR)

set(JSON_INCLUDE ${SOURCE_DIR}/include)
