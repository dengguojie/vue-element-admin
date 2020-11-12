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

add_library(intf_pub INTERFACE)

target_compile_options(intf_pub INTERFACE
  -Wall
  -fPIC
  $<IF:$<STREQUAL:${OS_TYPE},centos>,-fstack-protector-all,-fstack-protector-strong>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
)

target_compile_definitions(intf_pub INTERFACE
  $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
  OS_TYPE=WIN64
  WIN64=1
  LINUX=0
  $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
  $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
)

target_link_options(intf_pub INTERFACE
  $<$<CONFIG:Release>:-Wl,--build-id=none>
)

target_link_directories(intf_pub INTERFACE)

target_link_libraries(intf_pub INTERFACE)
