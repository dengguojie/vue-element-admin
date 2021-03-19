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

add_library(intf_llt_pub INTERFACE)

target_include_directories(intf_llt_pub INTERFACE
  ${GTEST_INCLUDE}
  ${GMOCK_INCLUDE}
)

target_compile_definitions(intf_llt_pub INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  CFG_BUILD_DEBUG
)

target_compile_options(intf_llt_pub INTERFACE
  -g
  --coverage
  -fprofile-arcs
  -ftest-coverage
  -w
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
  $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan>
  -fPIC
)

target_link_options(intf_llt_pub INTERFACE
  -fprofile-arcs -ftest-coverage
  $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -static-libasan -fsanitize=undefined  -static-libubsan>
)

target_link_libraries(intf_llt_pub INTERFACE
  gtest
  gtest_main
  gcov
  pthread
)
