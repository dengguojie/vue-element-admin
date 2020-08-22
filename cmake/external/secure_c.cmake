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

set(SEC_C "${CMAKE_CURRENT_BINARY_DIR}/third_party/src/secure_c")
set(SEC_C_SRCS "${SEC_C}/src/*.c")
set(SEC_C_INCS "${SEC_C}/include")
set(SEC_C_CFLAGS "-fstack-protector-strong -fPIC -Wall -D_FORTIFY_SOURCE=2 -O2")
set(SEC_C_BUILD 
  "gcc --shared -o libc_sec.so -I${SEC_C_INCS} ${SEC_C_CFLAGS} ${SEC_C_SRCS}"
)

ExternalProject_Add(secure_c
  URL               https://gitee.com/openeuler/bounds_checking_function/repository/archive/v1.1.10.tar.gz
  URL_MD5           0782dd2351fde6920d31a599b23d8c91
  DOWNLOAD_DIR      download/secure_c
  PREFIX            third_party
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     bash -c "${SEC_C_BUILD}"
  INSTALL_COMMAND   ""
)
