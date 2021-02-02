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
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
)

target_compile_definitions(intf_pub INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
  $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
  WIN64=1
  LINUX=0
)

target_link_options(intf_pub INTERFACE
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
  $<$<CONFIG:Release>:-Wl,--build-id=none>
)

target_link_directories(intf_pub INTERFACE)

target_link_libraries(intf_pub INTERFACE
  -lpthread
)

#########intf_pub_aicpu#########
add_library(intf_pub_aicpu INTERFACE)

target_compile_options(intf_pub_aicpu INTERFACE
  -Wall
  -fPIC
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
)

target_compile_definitions(intf_pub_aicpu INTERFACE
  $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:_GLIBCXX_USE_CXX11_ABI=0>
  $<$<STREQUAL:${PRODUCT_SIDE},device>:_GLIBCXX_USE_CXX11_ABI=1>
  $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
  $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
  WIN64=1
  LINUX=0
)

target_link_options(intf_pub_aicpu INTERFACE
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
  $<$<CONFIG:Release>:-Wl,--build-id=none>
)

target_link_directories(intf_pub_aicpu INTERFACE)

target_link_libraries(intf_pub_aicpu INTERFACE
  -lpthread
)
