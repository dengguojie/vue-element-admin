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
export BASE_PATH=$(cd "$(dirname $0)"; pwd)
export BUILD_PATH="${BASE_PATH}/build"
RELEASE_PATH="${BASE_PATH}/output"
INSTALL_PATH="${BUILD_PATH}/install"
CMAKE_HOST_PATH="${BUILD_PATH}/cann"
CMAKE_DEVICE_PATH="${BUILD_PATH}/cann_device"

source scripts/util/util.sh

# print usage message
usage() {
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-u] [-s] [-v] [-g]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -j[n] Set the number of threads used to build CANN, default is 8"
  echo "    -u Build all UT"
  echo "    -s Build ST"
  echo "    -v Verbose"
  echo "    -g GCC compiler prefix, used to specify the compiler toolchain"
  echo "    -a|--aicpu only compile aicpu task"
  echo "    -m|--minirc aicpu only compile aicpu task"
  echo "    --cpu_kernels_ut Build aicpu ut"
  echo "    --pass_ut Build pass ut"
  echo "    --tiling_ut Build tiling ut"
  echo "    --proto_ut Build proto ut"
  echo "    --tf_plugin_ut Build tf plugin ut"
  echo "    --onnx_plugin_ut Build onnx plugin ut"
  echo "    --noexec Only compile ut"

  echo "to be continued ..."
}

# parse and set optionss
checkopts() {
  VERBOSE=""
  THREAD_NUM=8
  GCC_PREFIX=""
  UT_TEST_ALL=FALSE
  ST_TEST=FALSE
  AICPU_ONLY=FALSE
  MINIRC_AICPU_ONLY=FALSE
  CPU_UT=FALSE
  PASS_UT=FALSE
  TILING_UT=FALSE
  PROTO_UT=FALSE
  PLUGIN_UT=FALSE
  ONNX_PLUGIN_UT=FALSE
  UT_NO_EXEC=FALSE
  CHANGED_FILES=""
  # Process the options
  while getopts 'hj:usvg:a-:m-:f:' opt
  do
    case "${opt}" in
      h) usage
         exit 0 ;;
      j) THREAD_NUM=$OPTARG ;;
      u) UT_TEST_ALL=TRUE ;;
      s) ST_TEST=TRUE ;;
      v) VERBOSE="VERBOSE=1" ;;
      g) GCC_PREFIX=$OPTARG ;;
      a) AICPU_ONLY=TRUE ;;
      m) MINIRC_AICPU_ONLY=TRUE ;;
      f) CHANGED_FILES=$OPTARG ;;
      -) case $OPTARG in
           aicpu) AICPU_ONLY=TRUE ;;
           minirc) MINIRC_AICPU_ONLY=TRUE ;;
           cpu_kernels_ut) CPU_UT=TRUE ;;
           pass_ut) PASS_UT=TRUE ;;
           tiling_ut) TILING_UT=TRUE ;;
           proto_ut) PROTO_UT=TRUE ;;
           tf_plugin_ut) PLUGIN_UT=TRUE ;;
           onnx_plugin_ut) ONNX_PLUGIN_UT=TRUE ;;
           noexec) UT_NO_EXEC=TRUE ;;
           *) logging "Undefined option: $OPTARG"
              usage
              exit 1 ;;
         esac
         ;;
      *) logging "Undefined option: ${opt}"
         usage
         exit 1 ;;
    esac
  done
}

parse_changed_files() {
  CHANGED_FILES=$1

  if [[ "$CHANGED_FILES" != /* ]]; then
    CHANGED_FILES=$PWD/$CHANGED_FILES
  fi

  logging "changed files is "$CHANGED_FILES
  logging '-----------------------------------------------'
  logging "changed lines:"
  cat $CHANGED_FILES
  logging '-----------------------------------------------'

  related_ut=`python3.7 scripts/parse_changed_files.py $1`
  logging "related ut "$related_ut

  if [[ $related_ut =~ "CPU_UT" ]];then
    logging "CPU_UT is triggered!"
    CPU_UT=TRUE
  fi
  if [[ $related_ut =~ "PASS_UT" ]];then
    logging "PASS_UT is triggered!"
    PASS_UT=TRUE
  fi
  if [[ $related_ut =~ "TILING_UT" ]];then
    logging "TILING_UT is triggered!"
    TILING_UT=TRUE
  fi
  if [[ $related_ut =~ "PROTO_UT" ]];then
    logging "PROTO_UT is triggered!"
    PROTO_UT=TRUE
  fi
  if [[ $related_ut =~ "PLUGIN_UT" ]];then
    logging "PLUGIN_UT is triggered!"
    PLUGIN_UT=TRUE
  fi
  if [[ $related_ut =~ "ONNX_PLUGIN_UT" ]];then
    logging "ONNX_PLUGIN_UT is triggered!"
    ONNX_PLUGIN_UT=TRUE
  fi
  reg='^\{.*?\}$'
  if [[ ! "$related_ut" =~ $reg ]];then
    logging "no ut matched! no need to run!"
    logging "---------------- CANN build finished ----------------"
    #for ci,2 means no need to run c++ ut;then ci will skip check coverage
    exit 200
  fi
}


# create build path
build_cann() {
  logging "Create build directory and build CANN"
  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE"
  if [[ "$GCC_PREFIX" != "" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGCC_PREFIX=$GCC_PREFIX"
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUT_TEST_ALL=TRUE"
  else
    CMAKE_ARGS="$CMAKE_ARGS -DUT_TEST_ALL=FALSE"
  fi
  if [[ "$ST_TEST" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DST_TEST=TRUE"
  else
    CMAKE_ARGS="$CMAKE_ARGS -DST_TEST=FALSE"
  fi
  if [[ "$AICPU_ONLY" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DAICPU_ONLY=TRUE"
  else
    CMAKE_ARGS="$CMAKE_ARGS -DAICPU_ONLY=FALSE"
  fi

  CMAKE_ARGS="$CMAKE_ARGS -DUT_NO_EXEC=$UT_NO_EXEC  \
            -DCPU_UT=$CPU_UT -DPASS_UT=$PASS_UT \
            -DTILING_UT=$TILING_UT -DPROTO_UT=$PROTO_UT \
            -DPLUGIN_UT=$PLUGIN_UT -DONNX_PLUGIN_UT=$ONNX_PLUGIN_UT"

  logging "Start build host target. CMake Args: ${CMAKE_ARGS}"

  if [[ "$ST_TEST" == "FALSE" ]]; then
    mk_dir "${CMAKE_HOST_PATH}"
    cd "${CMAKE_HOST_PATH}" && cmake ${CMAKE_ARGS} ../..
    make ${VERBOSE} -j${THREAD_NUM}
  fi
  if [ "$UT_TEST_ALL" == "FALSE" -a "$CPU_UT" == "FALSE" \
        -a "$PASS_UT" == "FALSE" -a "$TILING_UT" == "FALSE" \
        -a "$PROTO_UT" == "FALSE" -a "$PLUGIN_UT" == "FALSE" \
        -a "$ONNX_PLUGIN_UT" == "FALSE" ]; then
    CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE -DPRODUCT_SIDE=device"

    logging "Start build device target. CMake Args: ${CMAKE_ARGS}"
    mk_dir "${CMAKE_DEVICE_PATH}"
    cd "${CMAKE_DEVICE_PATH}" && cmake ${CMAKE_ARGS} ../..
    make ${VERBOSE} -j${THREAD_NUM}
  fi
  logging "CANN build success!"
}

minirc(){
  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE -DPRODUCT_SIDE=device -DMINRC=TRUE"
  logging "Start build device target. CMake Args: ${CMAKE_ARGS}"
  mk_dir "${CMAKE_DEVICE_PATH}"
  cd "${CMAKE_DEVICE_PATH}" && cmake ${CMAKE_ARGS} ../..
  make ${VERBOSE} -j${THREAD_NUM}

}

release_cann() {
  logging "Create output directory"
  mk_dir "${RELEASE_PATH}"
  RELEASE_TARGET="cann.tar"
  if [ "$MINIRC_AICPU_ONLY" = "TRUE" ];then
     RELEASE_TARGET="aicpu_minirc.tar"
  fi
  cd ${INSTALL_PATH} && tar cfz "${RELEASE_TARGET}" * && mv "${RELEASE_TARGET}" "${RELEASE_PATH}"
}

main() {
  checkopts "$@"
  if [[ "$CHANGED_FILES" != "" ]]; then
    UT_TEST_ALL=FALSE
    parse_changed_files $CHANGED_FILES
  fi
  # CANN build start
  logging "---------------- CANN build start ----------------"
  if [ "$MINIRC_AICPU_ONLY" = "TRUE" ]; then
    ${GCC_PREFIX}g++ -v
    minirc 
  else
    ${GCC_PREFIX}g++ -v
    build_cann
  fi
  if [ "$CPU_UT" = "FALSE" -a "$PASS_UT" = "FALSE" \
    -a "$TILING_UT" = "FALSE" -a "$PROTO_UT" = "FALSE" \
    -a "$PLUGIN_UT" = "FALSE" -a "$ONNX_PLUGIN_UT" = "FALSE" \
    -a "$UT_TEST_ALL" = "FALSE" ]; then
    release_cann
  fi
  logging "---------------- CANN build finished ----------------"
}

set -o pipefail
main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
