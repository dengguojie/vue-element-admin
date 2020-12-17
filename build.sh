#!/bin/bash
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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

set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
RELEASE_PATH="${BASE_PATH}/output"
export BUILD_PATH="${BASE_PATH}/build"
INSTALL_PATH="${BUILD_PATH}/install"
CMAKE_HOST_PATH="${BUILD_PATH}/cann"
CMAKE_DEVICE_PATH="${BUILD_PATH}/cann_device"

source scripts/util.sh

# print usage message
usage() {
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-u] [-s] [-v] [-g]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -j[n] Set the number of threads used to build CANN, default is 8"
  echo "    -u Build UT"
  echo "    -s Build ST"
  echo "    -v Verbose"
  echo "    -g GCC compiler prefix, used to specify the compiler toolchain"
  echo "to be continued ..."
}

# parse and set optionss
checkopts() {
  VERBOSE=""
  THREAD_NUM=8
  GCC_PREFIX=""
  UT_TEST=FALSE
  ST_TEST=FALSE
  # Process the options
  while getopts 'hj:usvg:' opt
  do
    case "${opt}" in
      h) usage
         exit 0 ;;
      j) THREAD_NUM=$OPTARG ;;
      u) UT_TEST=TRUE ;;
      s) ST_TEST=TRUE ;;
      v) VERBOSE="VERBOSE=1" ;;
      g) GCC_PREFIX=$OPTARG ;;
      *) logging "Undefined option: ${opt}"
         usage
         exit 1 ;;
    esac
  done
}

# create build path
build_cann() {
  logging "Create build directory and build CANN"
  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE"
  if [[ "$GCC_PREFIX" != "" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGCC_PREFIX=$GCC_PREFIX"
  fi
  if [[ "$UT_TEST" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUT_TEST=TRUE"
  fi
  if [[ "$ST_TEST" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DST_TEST=TRUE"
  fi
  logging "Start build host target. CMake Args: ${CMAKE_ARGS}"

  mk_dir "${CMAKE_HOST_PATH}"
  cd "${CMAKE_HOST_PATH}" && cmake ${CMAKE_ARGS} ../..
  make ${VERBOSE} -j${THREAD_NUM}

  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE -DPRODUCT_SIDE=device"
  
  logging "Start build device target. CMake Args: ${CMAKE_ARGS}"
  mk_dir "${CMAKE_DEVICE_PATH}"
  cd "${CMAKE_DEVICE_PATH}" && cmake ${CMAKE_ARGS} ../..
  make ${VERBOSE} -j${THREAD_NUM}

  logging "CANN build success!"
}

release_cann() {
  logging "Create output directory"
  mk_dir "${RELEASE_PATH}"
  RELEASE_TARGET="cann.tar"
  cd ${INSTALL_PATH} && tar cfz "${RELEASE_TARGET}" * && mv "${RELEASE_TARGET}" "${RELEASE_PATH}"
}

main() {
  checkopts "$@"
  # CANN build start
  logging "---------------- CANN build start ----------------"
  ${GCC_PREFIX}g++ -v
  build_cann
  release_cann
  logging "---------------- CANN build finished ----------------"
}

main "$@"
