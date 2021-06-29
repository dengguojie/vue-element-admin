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
core_nums=$(cat /proc/cpuinfo| grep "processor"| wc -l)
star_line="###############################################"
if [ $core_nums -ne 1 ];then
    core_nums=$((core_nums-1))
fi
create_lib(){
  git submodule init &&git submodule update
  down_third_libs
  if [ ! -d "${CMAKE_HOST_PATH}" ];then
    mkdir -p "${CMAKE_HOST_PATH}"
  fi
  echo ${CMAKE_ARGS}
  UT_FALSE="ops_all_caffe_plugin,ops_all_onnx_plugin,ops_all_plugin,ops_fusion_pass_vectorcore,ops_fusion_pass_aicore,\
           copy_veccore_fusion_rules,copy_aicore_fusion_rules,copy_op_proto_inc,opsproto,optiling,tbe_aicore_ops_impl,\
           tbe_ops_json_info,aicpu_ops_json_info,cpu_kernels_static,cpu_kernels_context_static,constant_folding_ops,\
           repack_tbe,copy_tbe,unzip_tbe,OpTestFrameFiles,MsopgenFiles"
  UT_TURE="protoc,secure_c,c_sec,eigen,protobuf_static_build,external_protobuf,nlohmann_json,\
          external_gtest,eigen_headers,ops_all_onnx_plugin_llt,opsplugin_llt,ops_fusion_pass_aicore_llt,\
          opsproto_llt,optiling_llt,generate_ops_cpp_cov,ops_cpp_proto_utest,ops_cpp_op_tiling_utest,\
          ops_cpp_fusion_pass_aicore_utest,cpu_kernels_ut,cpu_kernels_llt,ops_cpp_plugin_utest,ops_cpp_onnx_plugin_utest"
  
  if [[ "$UT_FALSE" =~ "$lib" ]];then
    CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE -DUT_TEST_ALL=FALSE -DST_TEST=FALSE -DAICPU_ONLY=FALSE -DUT_NO_EXEC=FALSE -DCPU_UT=FALSE -DPASS_UT=FALSE -DTILING_UT=FALSE -DPROTO_UT=FALSE -DPLUGIN_UT=FALSE -DONNX_PLUGIN_UT=FALSE" 
    cd "${CMAKE_HOST_PATH}" && cmake ${CMAKE_ARGS} ../..
    cmake --build . --target $lib -- -j $((core_nums-1))
    echo $star_line
    echo  "EXAMPLE cp ./build/cann/ops/built-in/op_tiling/liboptiling.so /usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/op_tiling/liboptiling.so"
    echo $star_line
  elif [[ "$UT_TURE" =~ "$lib" ]];then
    CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE -DUT_TEST_ALL=TURE -DST_TEST=FALSE -DAICPU_ONLY=FALSE -DUT_NO_EXEC=FALSE -DCPU_UT=FALSE -DPASS_UT=FALSE -DTILING_UT=FALSE -DPROTO_UT=FALSE -DPLUGIN_UT=FALSE -DONNX_PLUGIN_UT=FALSE"
    cd "${CMAKE_HOST_PATH}" && cmake ${CMAKE_ARGS} ../..
    cmake --build . --target $lib -- -j $((core_nums-1))
    echo $star_line
    echo  "EXAMPLE cp ./build/cann/ops/built-in/op_tiling/liboptiling.so /usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/op_tiling/liboptiling.so"
    echo $star_line
  fi
}

set_env(){
  cp ~/.bashrc ~/.bashrc.bak
  sed  -i '/#######auto create,link to ascend home ########/,/#######auto create,link to ascend home ########/d' ~/.bashrc
  echo  "#######auto create,link to ascend home ########" >> ~/.bashrc
  echo  "#This is the environment variable set for ASCEND" >> ~/.bashrc
  echo  " ">> ~/.bashrc

  ASCEND_CODE_HOME=$(cd "$(dirname $0)"; pwd)
  echo  export ASCEND_CODE_HOME=$ASCEND_CODE_HOME\$ASCEND_CODE_HOME >> ~/.bashrc
    
  ASCEND_HOME_FOR_ASCEND="/usr/local/Ascend"
  echo  export ASCEND_HOME=$ASCEND_HOME_FOR_ASCEND\$ASCEND_HOME >> ~/.bashrc
    
  OP_TEST_FRAME_INSTALL_HOME_FOR_ASCEND="${ASCEND_CODE_HOME}/tools/op_test_frame/python"
  echo  export OP_TEST_FRAME_INSTALL_HOME=\$ASCEND_CODE_HOME/tools/op_test_frame/python:\$OP_TEST_FRAME_INSTALL_HOME >> ~/.bashrc
    
  OPS_SOURCE_PATH_FOR_ASCEND="${ASCEND_CODE_HOME}/ops/built-in/tbe"
  echo  export OPS_SOURCE_PATH=\$ASCEND_CODE_HOME/ops/built-in/tbe:\$OPS_SOURCE_PATH >> ~/.bashrc
    
  ASCEND_OPP_PATH_FOR_ASCEND=$ASCEND_HOME_FOR_ASCEND/opp
  echo  export ASCEND_OPP_PATH=\$ASCEND_HOME/opp:\$ASCEND_OPP_PATH >> ~/.bashrc
    
  PYTHONPATH_FOR_ASCEND=$OPS_SOURCE_PATH_FOR_ASCEND:$OP_TEST_FRAME_INSTALL_HOME_FOR_ASCEND
  PYTHONPATH_FOR_USRLOCAL_ASCEND=$ASCEND_HOME_FOR_ASCEND/atc/python/site-packages:$ASCEND_HOME_FOR_ASCEND/toolkit/python/site-package
  if [ ! $usr_local ];then
    echo  export PYTHONPATH=\$OPS_SOURCE_PATH:\$OP_TEST_FRAME_INSTALL_HOME:\$ASCEND_HOME/atc/python/site-packages:\$ASCEND_HOME/toolkit/python/site-package:\$PYTHONPATH >> ~/.bashrc
  else
    echo  export PYTHONPATH=\$ASCEND_HOME/ops/op_impl/built-in/ai_core/tbe:\$ASCEND_HOME/atc/python/site-packages:\$ASCEND_HOME/toolkit/python/site-package:\$PYTHONPATH >> ~/.bashrc
  fi
  
  LD_LIBRARY_PATH_FOR_ASCEND=$ASCEND_HOME_FOR_ASCEND/atc/lib64:$ASCEND_CODE_HOME/lib
  echo  export LD_LIBRARY_PATH=\$ASCEND_HOME/atc/lib64:\$ASCEND_CODE_HOME/lib:\$LD_LIBRARY_PATH >> ~/.bashrc
    
  PATH_FOR_ASCEND=$ASCEND_HOME_FOR_ASCEND/atc/ccec_compiler/bin
  echo  export PATH=\$ASCEND_HOME/atc/ccec_compiler/bin:\$PATH >> ~/.bashrc

  echo  " ">> ~/.bashrc
  echo   $star_line >> ~/.bashrc
  echo   $star_line
  echo   ""
  echo   Please relogin shell to make sure environment variable correct.
  echo   ""
  echo   $star_line
}

query_env(){
  echo "ASCEND_HOME:$ASCEND_HOME"
  echo "OP_TEST_FRAME_INSTALL_HOME:$OP_TEST_FRAME_INSTALL_HOME"
  echo "OPS_SOURCE_PATH:$OPS_SOURCE_PATH"
  echo "ASCEND_OPP_PATH:$ASCEND_OPP_PATH"
  echo "PYTHONPATH:$PYTHONPATH"
  echo "LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
  echo "PATH:$PATH"
}

down_third_libs(){
  if [ ! -d "./build" ];then
    mkdir build
  fi
  if [ ! -d "./build/cann/download" ];then
    mkdir -p build/cann/download
  fi
  if [ ! -f "./build/cann/download/thirdlibs.zip" ];then
    wget -P build/cann/download https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/thirdlibs.zip
  fi
  if [ ! -d "./build/cann/download/ascned_thirdlibs" ];then
    unzip -d build/cann/download build/cann/download/thirdlibs.zip
  fi

  protobuf_md5=`echo -n ./build/cann/download/ascend_thirdlibs/download/protobuf/v3.13.0.tar.gz|md5sum|cut -d" " -f1`
  eigen_md5=`echo -n ./build/cann/download/ascend_thirdlibs/download/eigen/eigen-3.3.7.tar.gz|md5sum|cut -d" " -f1`
  gtest_md5=`echo -n ./build/cann/download/ascend_thirdlibs/download/gtest/release-1.8.0.tar.gz|md5sum|cut -d" " -f1`
  nlohmann_json_md5=`echo -n ./build/cann/download/ascend_thirdlibs/download/nlohmann_json/include.zip|md5sum|cut -d" " -f1`
  secure_c_md5=`echo -n ./build/cann/download/ascend_thirdlibs/download/secure_c/v1.1.10.tar.gz|md5sum|cut -d" " -f1`
  
  if [ ! $protobuf_md5="1a6274bc4a65b55a6fa70e264d796490" ];then
    echo "protobuf md5 not correct"
    exit -1
  fi
  if [ ! $eigen_md5="9e30f67e8531477de4117506fe44669b" ];then
    echo "eigen md5 not correct"
    exit -1
  fi
  if [ ! $gtest_md5="16877098823401d1bf2ed7891d7dce36" ];then
    echo "gtest md5 not correct"
    exit -1
  fi
  if [ ! $nlohmann_json_md5="0dc903888211db3a0f170304cd9f3a89" ];then
    echo "nlohmann_json md5 not correct"
    exit -1
  fi
  if [ ! $secure_c_md5="193f0ca5246c1dd84920db34d2d8249f" ];then
    echo "secure_c md5 not correct"
    exit -1
  fi
  
  if [ ! -f "./build/cann/download/eigen/eigen-3.3.7.tar.gz" ];then
      if [ ! -d "./build/cann/download/eigen" ];then
	    mkdir build/cann/download/eigen
      fi
        cp build/cann/download/ascned_thirdlibs/download/eigen/eigen-3.3.7.tar.gz build/cann/download/eigen/eigen-3.3.7.tar.gz
  fi
  if [ ! -f "./build/cann/download/gtest/release-1.8.0.tar.gz" ];then
      if [ ! -d "./build/cann/download/gtest" ];then
	    mkdir build/cann/download/gtest
      fi
        cp build/cann/download/ascned_thirdlibs/download/gtest/release-1.8.0.tar.gz build/cann/download/gtest/release-1.8.0.tar.gz      
  fi
  if [ ! -f "./build/cann/download/nlohmann_json/include.zip" ];then
      if [ ! -d "./build/cann/download/nlohmann_json" ];then
	    mkdir build/cann/download/nlohmann_json
      fi
        cp build/cann/download/ascned_thirdlibs/download/nlohmann_json/include.zip build/cann/download/nlohmann_json/include.zip      
  fi
  if [ ! -f "./build/cann/download/protobuf/v3.13.0.tar.gz" ];then
      if [ ! -d "./build/cann/download/protobuf" ];then
	    mkdir build/cann/download/protobuf
      fi
        cp build/cann/download/ascned_thirdlibs/download/protobuf/v3.13.0.tar.gz build/cann/download/protobuf/v3.13.0.tar.gz 
  fi
  if [ ! -f "./build/cann/download/secure_c/v1.1.10.tar.gz" ];then
      if [ ! -d "./build/cann/download/secure_c" ];then
	    mkdir build/cann/download/secure_c
      fi
        cp build/cann/download/ascned_thirdlibs/download/secure_c/v1.1.10.tar.gz build/cann/download/secure_c/v1.1.10.tar.gz      
  fi
}
check_third_libs(){
  if [ ! -f "./build/cann/download/eigen/eigen-3.3.7.tar.gz" ];then
	  echo "download from  https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/thirdlibs.zip failed"
  fi
  if [ ! -f "./build/cann/download/gtest/release-1.8.0.tar.gz" ];then
	  echo "download from  https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/thirdlibs.zip failed"
  fi
  if [ ! -f "./build/cann/download/nlohmann_json/include.zip" ];then
	  echo "download from  https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/thirdlibs.zip failed"
  fi
  if [ ! -f "./build/cann/download/protobuf/v3.13.0.tar.gz" ];then
	  echo "download from  https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/thirdlibs.zip failed"
  fi
  if [ ! -f "./build/cann/download/secure_c/v1.1.10.tar.gz" ];then
	  echo "download from  https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/thirdlibs.zip failed"
  fi
}
install_python_libs(){
  python_libs=$(pip3 list)
  if [[ ! "$python_libs" =~ "numpy" ]];then
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
  fi
  if [[ ! "$python_libs" =~ "decorator" ]];then
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple decorator
  fi
  if [[ ! "$python_libs" =~ "sympy" ]];then
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple sympy
  fi
}
source scripts/util/util.sh

# print usage message
usage() {
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-u] [-s] [-v] [-g]"
  echo ""
  echo "If you are using it for the first time, it needs to be executed ./build.sh --down_and_check_third_libs,"
  echo "./build.sh --set_env_gitee or ./build.sh --set_env_ascend"
  echo "./build.sh --install_python_libs"
  echo "Options:"
  echo "    -h Print usage"
  echo "    -x Download git submodule"
  echo "    -j[n] Set the number of threads used to build CANN, default is 8"
  echo "    -u Build all UT"
  echo "    -s Build ST"
  echo "    -v Verbose"
  echo "    -g GCC compiler prefix, used to specify the compiler toolchain"
  echo "    -a|--aicpu only compile aicpu task"
  echo "    -m|--minirc aicpu only compile aicpu task"
  echo "    --down_and_check_third_libs Download third party libs"
  echo "    --query_env query current env"
  echo "    --set_env_gitee set env for ascend,use gitee download python operator code"
  echo "    --set_env_ascend set env for ascend,use /usr/local/Ascend python operato code"
  echo "    --install_python_libs install necessary python libs"
  echo "    --cpu_kernels_ut Build aicpu ut"
  echo "    --pass_ut Build pass ut"
  echo "    --tiling_ut Build tiling ut"
  echo "    --proto_ut Build proto ut"
  echo "    --tf_plugin_ut Build tf plugin ut"
  echo "    --onnx_plugin_ut Build onnx plugin ut"
  echo "    --noexec Only compile ut"
  echo "    --sprotoc build sprotoc"
  echo "    --secure_c build secure_c"
  echo "    --c_sec build c_sec"
  echo "    --eigen build eigen"
  echo "    --protobuf_static_build build protobuf_static_build"
  echo "    --external_protobuf build external_protobuf"
  echo "    --nlohmann_json build nlohmann_json"
  echo "    --external_gtest build external_gtest"
  echo "    --eigen_headers build eigen_headers"
  echo "    --ops_all_onnx_plugin_llt build ops_all_onnx_plugin_llt"
  echo "    --opsplugin_llt build opsplugin_llt"
  echo "    --ops_fusion_pass_aicore_llt build ops_fusion_pass_aicore_llt"
  echo "    --opsproto_llt build opsproto_llt"
  echo "    --optiling_llt build optiling_llt"
  echo "    --generate_ops_cpp_cov build generate_ops_cpp_cov"
  echo "    --ops_cpp_proto_utest build ops_cpp_proto_utest"
  echo "    --ops_cpp_op_tiling_utest build ops_cpp_op_tiling_utest"
  echo "    --ops_cpp_fusion_pass_aicore_utest build ops_cpp_fusion_pass_aicore_utest"
  echo "    --cpu_kernels_ut build cpu_kernels_ut"
  echo "    --cpu_kernels_llt build cpu_kernels_llt"
  echo "    --ops_cpp_plugin_utest build ops_cpp_plugin_utest"
  echo "    --ops_cpp_onnx_plugin_utest build ops_cpp_onnx_plugin_utest"
  echo "    --ops_all_caffe_plugin build ops_all_caffe_plugin"
  echo "    --ops_all_onnx_plugin build ops_all_onnx_plugin"
  echo "    --ops_all_plugin build ops_all_plugin"
  echo "    --ops_fusion_pass_vectorcore build ops_fusion_pass_vectorcore"
  echo "    --ops_fusion_pass_aicore build ops_fusion_pass_aicore"
  echo "    --copy_veccore_fusion_rules build copy_veccore_fusion_rules"
  echo "    --copy_aicore_fusion_rules build copy_aicore_fusion_rules"
  echo "    --copy_op_proto_inc build copy_op_proto_inc"
  echo "    --opsproto build opsproto"
  echo "    --optiling build optiling"
  echo "    --tbe_aicore_ops_impl build tbe_aicore_ops_impl"
  echo "    --tbe_ops_json_info build tbe_ops_json_info"
  echo "    --aicpu_ops_json_info build aicpu_ops_json_info"
  echo "    --cpu_kernels_static build cpu_kernels_static"
  echo "    --cpu_kernels_context_static build cpu_kernels_context_static"
  echo "    --constant_folding_ops build constant_folding_ops"
  echo "    --repack_tbe build repack_tbe"
  echo "    --copy_tbe build copy_tbe"
  echo "    --unzip_tbe build unzip_tbe"
  echo "    --OpTestFrameFiles build OpTestFrameFiles"
  echo "    --MsopgenFiles build MsopgenFiles"
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
  while getopts 'xhj:usvg:a-:m-:f:' opt
  do
    case "${opt}" in
      x) git submodule init &&git submodule update
         exit 0;;
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
           down_and_check_third_libs) down_third_libs;check_third_libs
                                      exit ;;
           set_env_ascend) usr_local=TRUE
                          set_env
                          exit 0;;
           set_env_gitee) usr_local=""
                           set_env
                           exit 0;;          
                           
           query_env) query_env
                      exit 0;;
           install_python_libs) install_python_libs
                                exit 0;;
           sprotoc) lib="sprotoc"
                    create_lib
                    exit 0;;
           secure_c) lib="secure_c"
                     create_lib
                     exit 0;;
           c_sec) lib="c_sec"
                  create_lib
                  exit 0;;
           eigen) lib="eigen"
                  create_lib
                  exit 0;;
           protobuf_static_build) lib="protobuf_static_build"
                                  create_lib
                                  exit 0;;
           external_protobuf) lib="external_protobuf"
                              create_lib
                              exit 0;;
           nlohmann_json) lib="nlohmann_json"
                          create_lib
                          exit 0;;
           external_gtest) lib="external_gtest"
                           create_lib
                           exit 0;;
           eigen_headers) lib="eigen_headers"
                          create_lib
                          exit 0;;
           ops_all_onnx_plugin_llt) lib="ops_all_onnx_plugin_llt"
                                    create_lib
                                    exit 0;;
           opsplugin_llt) lib="opsplugin_llt"
                          create_lib
                          exit 0;;
           ops_fusion_pass_aicore_llt) lib="ops_fusion_pass_aicore_llt"
                                       create_lib
                                       exit 0;;
           opsproto_llt) lib="opsproto_llt"
                         create_lib
                         exit 0;;
           optiling_llt) lib="optiling_llt"
                         create_lib
                         exit 0;;
           generate_ops_cpp_cov) lib="generate_ops_cpp_cov"
                                 create_lib
                                 exit 0;;
           ops_cpp_proto_utest) lib="ops_cpp_proto_utest"
                                create_lib
                                exit 0;;
           ops_cpp_op_tiling_utest) lib="ops_cpp_op_tiling_utest"
                                    create_lib
                                    exit 0;;
           ops_cpp_fusion_pass_aicore_utest) lib="ops_cpp_fusion_pass_aicore_utest"
                                             create_lib
                                             exit 0;;
           cpu_kernels_ut) lib="cpu_kernels_ut"
                           create_lib
                           exit 0;;
           cpu_kernels_llt) lib="cpu_kernels_llt"
                            create_lib
                            exit 0;;
           ops_cpp_plugin_utest) lib="ops_cpp_plugin_utest"
                                 create_lib
                                 exit 0;;
           ops_cpp_onnx_plugin_utest) lib="ops_cpp_onnx_plugin_utest"
                                      create_lib
                                      exit 0;;
           ops_all_caffe_plugin) lib="ops_all_caffe_plugin"
                                 create_lib
                                 exit 0;;
           ops_all_onnx_plugin) lib="ops_all_onnx_plugin"
                                create_lib
                                exit 0;;
           ops_all_plugin) lib="ops_all_plugin"
                           create_lib
                           exit 0;;
           ops_fusion_pass_vectorcore) lib="ops_fusion_pass_vectorcore"
                                       create_lib
                                       exit 0;;
           ops_fusion_pass_aicore) lib="ops_fusion_pass_aicore"
                                   create_lib
                                   exit 0;;
           copy_veccore_fusion_rules) lib="copy_veccore_fusion_rules"
                                      create_lib
                                      exit 0;;
           copy_aicore_fusion_rules) lib="copy_aicore_fusion_rules"
                                     create_lib
                                     exit 0;;
           copy_op_proto_inc) lib="copy_op_proto_inc"
                              create_lib
                              exit 0;;
           opsproto) lib="opsproto"
                     create_lib
                     exit 0;;
           optiling) lib="optiling"
                     create_lib
                     exit 0;;
           tbe_aicore_ops_impl) lib="tbe_aicore_ops_impl"
                                create_lib
                                exit 0;;
           tbe_ops_json_info) lib="tbe_ops_json_info"
                              create_lib
                              exit 0;;
           aicpu_ops_json_info) lib="aicpu_ops_json_info"
                                create_lib
                                exit 0;;
           cpu_kernels_static) lib="cpu_kernels_static"
                               create_lib
                               exit 0;;
           cpu_kernels_context_static) lib="cpu_kernels_context_static"
                                       create_lib
                                       exit 0;;
           constant_folding_ops) lib="constant_folding_ops"
                                 create_lib
                                 exit 0;;
           repack_tbe) lib="repack_tbe"
                       create_lib
                       exit 0;;
           copy_tbe) lib="copy_tbe"
                     create_lib
                     exit 0;;
           unzip_tbe) lib="unzip_tbe"
                      create_lib
                      exit 0;;
           OpTestFrameFiles) lib="OpTestFrameFiles"
                             create_lib
                             exit 0;;
           MsopgenFiles) lib="MsopgenFiles"
                         create_lib
                         exit 0;;
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
