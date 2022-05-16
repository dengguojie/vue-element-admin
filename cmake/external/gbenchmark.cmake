# Copyright 2022 Huawei Technologies Co., Ltd
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
if (HAVE_BENCHMARK)
    return()
endif()

include(ExternalProject)

set(CMAKE_INSTALL_PREFIX ${CANN_ROOT}/output)
set (benchmark_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
ExternalProject_Add(benchmark_build
        URL https://gitee.com/mirrors/benchmark/repository/archive/v1.5.5.tar.gz
        URL_MD5 45c8bdef8e616b76c3afa213a474e5d0
        TLS_VERIFY OFF
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=${benchmark_CXXFLAGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/benchmark <SOURCE_DIR>
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_MACOSX_RPATH=TRUE
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        EXCLUDE_FROM_ALL TRUE
        )

set(BENCHMARK_PKG_DIR ${CMAKE_INSTALL_PREFIX}/benchmark)

file(MAKE_DIRECTORY ${BENCHMARK_PKG_DIR}/include)

add_library(benchmark SHARED IMPORTED)

target_include_directories(benchmark INTERFACE ${BENCHMARK_PKG_DIR}/include)

set_target_properties(benchmark PROPERTIES
        IMPORTED_LOCATION ${BENCHMARK_PKG_DIR}/lib/libbenchmark.so
        )

add_library(benchmark_main SHARED IMPORTED)

target_include_directories(benchmark_main INTERFACE ${BENCHMARK_PKG_DIR}/include)

set_target_properties(benchmark_main PROPERTIES
        IMPORTED_LOCATION ${BENCHMARK_PKG_DIR}/lib/libbenchmark_main.so
        )

add_dependencies(benchmark benchmark_build)

set(HAVE_BENCHMARK TRUE)
