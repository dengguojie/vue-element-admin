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
CMAKE_PATH="${BUILD_PATH}/cann"

# print usage message
usage() {
  echo "Usage:"
  echo "sh build.sh [-j[n]] [-h] [-v] [-s] [-t] [-u] [-c]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -u Only compile ut, not execute"
  echo "    -s Build st"
  echo "    -j[n] Set the number of threads used to build CANN, default is 8"
  echo "    -t Build and execute ut"
  echo "    -c Build ut with coverage tag"
  echo "    -v Verbose"
  echo "to be continued ..."
}

logging() {
  echo "[INFO] $@"
}

# parse and set optionss
checkopts() {
  VERBOSE=""
  THREAD_NUM=8
  # ENABLE_CANN_UT_ONLY_COMPILE="off"
  ENABLE_CANN_UT="off"
  ENABLE_CANN_ST="off"
  ENABLE_CANN_COV="off"
  CANN_ONLY="on"
  # Process the options
  while getopts 'ustchj:v' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      u)
        # ENABLE_CANN_UT_ONLY_COMPILE="on"
        ENABLE_CANN_UT="on"
        CANN_ONLY="off"
        ;;
      s)
        ENABLE_CANN_ST="on"
        ;;
      t)
	      ENABLE_CANN_UT="on"
	      CANN_ONLY="off"
	      ;;
      c)
        ENABLE_CANN_COV="on"
        CANN_ONLY="off"
        ;;
      h)
        usage
        exit 0
        ;;
      j)
        THREAD_NUM=$OPTARG
        ;;
      v)
        VERBOSE="VERBOSE=1"
        ;;
      *)
        logging "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}

# mkdir directory
mk_dir() {
  local create_dir="$1"
  mkdir -pv "${create_dir}"
  logging "Created ${create_dir}"
}

# create build path
build_cann() {
  logging "Create build directory and build CANN"

  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH"
  if [[ "X$ENABLE_CANN_COV" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_CANN_COV=ON"
  fi

  if [[ "X$ENABLE_CANN_UT" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_CANN_UT=ON"
  fi
  if [[ "X$ENABLE_CANN_ST" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_CANN_ST=ON"
  fi

  logging "CMake Args: ${CMAKE_ARGS}"
  mk_dir "${CMAKE_PATH}"
  cd "${CMAKE_PATH}" && cmake ${CMAKE_ARGS} ../..
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
  g++ -v
  build_cann
  release_cann
  logging "---------------- CANN build finished ----------------"
}

main "$@"
