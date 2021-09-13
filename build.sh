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
dotted_line="----------------------------------------------------------------"
if [ $core_nums -ne 1 ];then
    core_nums=$((core_nums-1))
fi

get_target_lib(){
RELEASE_VERSION="ops_all_caffe_plugin,ops_all_onnx_plugin,ops_all_plugin,ops_fusion_pass_vectorcore,ops_fusion_pass_aicore,\
           copy_veccore_fusion_rules,copy_aicore_fusion_rules,copy_op_proto_inc,opsproto,optiling,tbe_aicore_ops_impl,\
           tbe_ops_json_info,aicpu_ops_json_info,cpu_kernels_static,cpu_kernels_context_static,constant_folding_ops,\
           repack_tbe,copy_tbe,unzip_tbe,OpTestFrameFiles,MsopgenFiles,aicpu_nodedef_builder"
UT_VERSION="protoc,secure_c,c_sec,eigen,protobuf_static_build,external_protobuf,nlohmann_json,\
          external_gtest,eigen_headers,ops_all_onnx_plugin_llt,opsplugin_llt,ops_fusion_pass_aicore_llt,\
          opsproto_llt,optiling_llt,generate_ops_cpp_cov,ops_cpp_proto_utest,ops_cpp_op_tiling_utest,\
          ops_cpp_fusion_pass_aicore_utest,cpu_kernels_ut,cpu_kernels_llt,ops_cpp_plugin_utest,ops_cpp_onnx_plugin_utest"
}

create_lib(){
  git submodule init &&git submodule update
  down_third_libs
  if [ ! -d "${CMAKE_HOST_PATH}" ];then
    mkdir -p "${CMAKE_HOST_PATH}"
  fi
  echo ${CMAKE_ARGS}
  cd "${CMAKE_HOST_PATH}" && cmake ${CMAKE_ARGS} ../..
  cmake --build . --target $lib -- -j $core_nums
  if [[ "$cov" =~ "TRUE" ]];then
    cmake --build . --target generate_ops_cpp_cov -- -j $core_nums
  fi
  echo $dotted_line
  echo "TIPS"
  echo "If you compile a shared or static lib, you can copy your lib from the subfolder of./build/cann to the corresponding folder of/usr/local/ascend"
  echo $dotted_line  
}

set_env(){
  cp ~/.bashrc ~/.bashrc.bak
  sed  -i '/#######auto create,link to ascend home ########/,/#######auto create,link to ascend home ########/d' ~/.bashrc
  echo  "#######auto create,link to ascend home ########" >> ~/.bashrc
  echo  "#This is the environment variable set for ASCEND" >> ~/.bashrc
  echo  " ">> ~/.bashrc
  
  echo  export ASCEND_CODE_HOME=$(cd "$(dirname $0)"; pwd) >> ~/.bashrc
  if [ $UID -ne 0 ];then
    echo  export ASCEND_HOME=~/Ascend >> ~/.bashrc
    echo  export ASCEND_CUSTOM_PATH=~/Ascend >> ~/.bashrc
  else
    echo  export ASCEND_HOME="/usr/local/Ascend" >> ~/.bashrc
    unset ASCEND_CUSTOM_PATH
  fi
  
  echo  export OP_TEST_FRAME_INSTALL_HOME=\$ASCEND_CODE_HOME/tools/op_test_frame/python >> ~/.bashrc
  echo  export OPS_SOURCE_PATH=\$ASCEND_CODE_HOME/ops/built-in/tbe>> ~/.bashrc
  echo  export ASCEND_OPP_PATH=\$ASCEND_HOME/opp >> ~/.bashrc
    
  if [ ! $usr_local ];then
    echo  export PYTHONPATH=\$OPS_SOURCE_PATH:\$OP_TEST_FRAME_INSTALL_HOME:\$ASCEND_HOME/atc/python/site-packages:\$ASCEND_HOME/toolkit/python/site-package:\$PYTHONPATH >> ~/.bashrc
  else
    echo  export PYTHONPATH=\$ASCEND_HOME/ops/op_impl/built-in/ai_core/tbe:\$ASCEND_HOME/atc/python/site-packages:\$ASCEND_HOME/toolkit/python/site-package:\$PYTHONPATH >> ~/.bashrc
  fi
  
  echo  export LD_LIBRARY_PATH=\$ASCEND_HOME/atc/lib64:\$ASCEND_CODE_HOME/lib:\$LD_LIBRARY_PATH >> ~/.bashrc
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
  echo $dotted_line
  echo "ASCEND_CODE_HOME:$ASCEND_CODE_HOME"
  echo "ASCEND_HOME:$ASCEND_HOME"
  echo "OP_TEST_FRAME_INSTALL_HOME:$OP_TEST_FRAME_INSTALL_HOME"
  echo "OPS_SOURCE_PATH:$OPS_SOURCE_PATH"
  echo "ASCEND_OPP_PATH:$ASCEND_OPP_PATH"
  echo "PYTHONPATH:$PYTHONPATH"
  echo "LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
  echo "PATH:$PATH"
}

get_libs_name(){
  libs=("secure_c" "protobuf" "eigen" "gtest" "nlohmann_json")
  secure_c_pack="v1.1.10.tar.gz"
  protobuf_pack="v3.13.0.tar.gz"
  eigen_pack="eigen-3.3.7.tar.gz"
  gtest_pack="release-1.8.0.tar.gz"
  nlohmann_json_pack="include.zip"
  eigen_link=https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
  gtest_link=https://github.com/google/googletest/archive/release-1.8.0.tar.gz
  nlohmann_json_link=https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip
  protobuf_link=https://github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz
  secure_c_link=https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz
}

down_third_libs(){
  set +e
  get_libs_name
  obs_addr="https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/thirdlibs.zip"
  if [ ! -d "./build/cann/download" ];then
    mkdir -p build/cann/download
  fi
  for mylib in ${libs[@]}
    do
      if [ ! -d "./build/cann/download/$mylib" ];then
        mkdir build/cann/download/$mylib
      fi
    done
  echo $dotted_line
  echo "begin to test  network..."
  wget --no-check-certificate https://www.gitee.com
  res_net=`echo $?`
  if [  $res_net -ne 0 ];then
    echo $dotted_line
    echo "The network doesn't work. please check..."
    echo "If you are in Huawei yellow area"
    echo "EXAMPLE"
    echo "export http_proxy=http//\$username:\$escape_pass@\${proxy:-proxy}.huawei.com:8080/"
    echo "NOTICE:password needs to be escaped"
    echo "export https_proxy=\$http_proxy"
    echo "If you are not in Huawei yellow area"
    echo "You need to configure a network proxy"
    exit -1
  else
    if [ -f "index.html" ];then
      rm index.html
      echo $dotted_line
      echo "Network testing completed"
    fi
  fi
  if [ ! -f "./build/cann/download/thirdlibs.zip" ];then 
    wget --no-check-certificate  --connect-timeout=5 -P build/cann/download $obs_addr
    res_down=`echo $?`
    if [  $res_down -eq 0 ];then
      echo "download from $obs_addr success"
    else
      echo $dotted_line
      echo "download from $obs_addr failed"
      echo "begin download from github"
      for mylib in ${libs[@]}
        do
          pack=${mylib}_pack
          link=${mylib}_link
          eval pack=$(echo \$$pack)
          eval link=$(echo \$$link)
          if [ ! -f "./build/cann/download/$mylib/$pack" ];then
            wget --no-check-certificate -P build/cann/download/$mylib --debug ${link}
          fi
        done
    fi
  else
    echo $dotted_line
    echo "./build/cann/download/thirdlibs.zip exist, do not need download"
  fi
  if [ ! -f "./build/cann/download/thirdlibs.zip" ];then 
    if [ $res_down -ne 0  ];then
      for mylib in ${libs[@]}
        do
          pack=${mylib}_pack
          link=${mylib}_link
          eval pack=$(echo \$$pack)
          eval link=$(echo \$$link)
          if [ -f "./build/cann/download/$mylib/$pack" ];then
            md5=`md5sum ./build/cann/download/$mylib/$pack | cut -d" " -f1`
            eval "${mylib}_md5=$md5"
          else
            echo "dwonload from $link failed"
            exit -1
          fi
        done
    fi
  fi
  if [ ! -d "./build/cann/download/ascend_thirdlibs" ];then
    if [ -f "./build/cann/download/thirdlibs.zip" ];then
      unzip -d build/cann/download build/cann/download/thirdlibs.zip
      mv ./build/cann/download/ascned_thirdlibs ./build/cann/download/ascend_thirdlibs
    fi
  fi
  if [ -d "./build/cann/download/ascend_thirdlibs" ];then
     for mylib in ${libs[@]}
        do
          pack=${mylib}_pack
          eval pack=$(echo \$$pack)
          md5=`md5sum ./build/cann/download/ascend_thirdlibs/download/$mylib/$pack | cut -d" " -f1`
          eval "${mylib}_md5=$md5"
        done
  fi
  for mylib in ${libs[@]}
    do
      pack=${mylib}_pack
      eval pack=$(echo \$$pack)
      if [ -f build/cann/download/ascend_thirdlibs/download/$mylib/$pack ];then
        if [ ! -f "./build/cann/download/$mylib/$pack" ];then
          cp build/cann/download/ascend_thirdlibs/download/$mylib/$pack build/cann/download/$mylib/$pack
        fi
      fi
    done

  expect_md5=("193f0ca5246c1dd84920db34d2d8249f" #is secure_c
              "1a6274bc4a65b55a6fa70e264d796490" #is protobuf
              "9e30f67e8531477de4117506fe44669b" #is eigen
              "16877098823401d1bf2ed7891d7dce36" #is gtest
              "0dc903888211db3a0f170304cd9f3a89" ) #is nlohmann_json
              
  echo "check md5 begin"
  for i in $(seq 0 4)
    do
      real_md5=${libs[i]}_md5
      eval real_md5=$(echo \$$real_md5)
      if [  $real_md5 != ${expect_md5[i]} ];then
        echo "${libs[i]} md5 not correct"
        exit -1
      fi
    done  
  echo "check md5 finish"    
  set -e
}

check_third_libs(){
  echo "check third libs begin"
  get_libs_name
  for mylib in ${libs[@]}
    do
      pack=${mylib}_pack
      link=${mylib}_link
      eval pack=$(echo \$$pack)
      eval link=$(echo \$$link)
      if [ ! -f "./build/cann/download/$mylib/$pack" ];then
        echo "dwonload from $link failed"
        exit -1
      fi      
    done
  echo "check third libs success"
}
install_python_libs(){
  python_libs=`pip3 list`
  for p in numpy decorator sympy wheel psutil attrs
    do
      if [[ "$python_libs" =~ "$p" ]];then
        echo $p
      else
         pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple $p
      fi
    done
}
source scripts/util/util.sh

make_clean(){
  if [ -d $CMAKE_HOST_PATH ];then
    cd $CMAKE_HOST_PATH
    make clean
  fi
}

make_clean_all(){
  if [ -d $CMAKE_HOST_PATH ];then
    cd ${CMAKE_HOST_PATH}
    rm -rf ./*
  fi
  rm -rf $RELEASE_PATH
  rm -rf $INSTALL_PATH
}

# print usage message
usage() {
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-u] [-s] [-v] [-g]"
  echo ""
  echo "If you are using it for the first time, it needs to be executed "
  echo "./build.sh --down_and_check_third_libs "
  echo "./build.sh --set_env_gitee or ./build.sh --set_env_ascend . These two commands can help you set environment variables automatically"
  echo "./build.sh --install_python_libs"
  echo "example, Build ops_cpp_proto_utest with O3 level compilation optimization and do not execute."
  echo "./build.sh --ops_cpp_proto_utest --build_mode_O3 --noexec"
  echo ""
  echo "Options:"
  echo      $dotted_line
  echo "    Download and install ascend "
  echo      $dotted_line  
  echo "    --install_daily  download and install Ascend, using package/daily/ the latest daily package"
  echo "    --install_etrans  download and install Ascend, using package/etrans/ the latest extrans package"
  echo "        *** You must use a single quotation mark for your username and password.***"
  echo "        *** example ./build.sh --install_daily 'username' 'password'   ***"
  echo "    --install_local  If you use the above two commands to download and install, the installation fails for some reasons. "
  echo "                     After solving the problem, you can use this command to install locally, so as to save download time."
  echo ""
  echo ""
  echo      $dotted_line
  echo "    Download third-party dependencies and python packages"
  echo      $dotted_line
  echo "    --down_and_check_third_libs Download third party libs"
  echo "    --query_env query current env"
  echo "    --install_python_libs install necessary python libs"
  echo ""
  echo ""
  echo      $dotted_line
  echo "    Setting environment variables"
  echo      $dotted_line
  echo "    --set_env_gitee set env for ascend,use gitee download python operator code"
  echo "    --set_env_ascend set env for ascend,use /usr/local/Ascend python operato code"  
  echo ""
  echo ""
  echo      $dotted_line
  echo "    Build parameters "
  echo      $dotted_line  
  echo "    -h Print usage"
  echo "    -x Download git submodule"
  echo "    -j[n] Set the number of threads used to build CANN, default is 8"
  echo "    -u Build all UT"
  echo "    -s Build ST"
  echo "    -v Verbose"
  echo "    -g GCC compiler prefix, used to specify the compiler toolchain"
  echo "    -a|--aicpu only compile aicpu task"
  echo "    -m|--minirc aicpu only compile aicpu task"
  echo "    --cov  When building uTest locally, adding --cov will count the coverage. For example ./build.sh --ops_cpp_op_tiling_utest --cov "
  echo "    --make_clean make clean"
  echo "    --make_clean_all make clean and delete related file"
  echo "    --noexec Only compile ut, do not execute the compiled executable file"
  echo "    --build_mode_xxx,the xxx can be in [O0 O1 O2 O3 g], for example build_mode_O2"
  echo ""
  echo ""
  echo      $dotted_line
  echo "    Next is the name that you can build directly"
  echo      $dotted_line
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
  echo "    --aicpu_nodedef_builder build aicpu_nodedef_builder"
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
  EXEC_UT=FALSE
  UT_MODE=FALSE
  CHANGED_FILES=""
  build_mode=FALSE
  cov=FALSE
  CI_MODE=FALSE
  # Process the options
  while getopts 'xhj:usvg:a-:m-:f:' opt
  do
    case "${opt}" in
      x) git submodule init &&git submodule update
         exit 0;;
      h) usage
         exit 0 ;;
      j) THREAD_NUM=$OPTARG ;;
      u) UT_TEST_ALL=TRUE 
	     UT_MODE=TRUE ;;
      s) ST_TEST=TRUE ;;
      v) VERBOSE="VERBOSE=1" ;;
      g) GCC_PREFIX=$OPTARG ;;
      a) AICPU_ONLY=TRUE ;;
      m) MINIRC_AICPU_ONLY=TRUE ;;
      f) CHANGED_FILES=$OPTARG ;;
      -) case $OPTARG in
           aicpu) AICPU_ONLY=TRUE ;;
           minirc) MINIRC_AICPU_ONLY=TRUE ;;
           noexec) UT_NO_EXEC=TRUE;;
           cov)cov=TRUE;;
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
           make_clean_all) make_clean_all
                           exit 0;;
           make_clean) make_clean
                       exit 0;;
           install_daily) username="$2"
                          pwsswd="$3"
                          echo $dotted_line
                          if [[ "`echo $#`" -eq 4 ]];then
                              day=$4
                          fi
                          echo $day
                          chmod 744 ./scripts/install_daily.sh 
                          ./scripts/install_daily.sh $username $pwsswd $day
                          exit 0;;
           install_etrans) username="$2"
                            pwsswd="$3"
                            if [[ "`echo $#`" -eq 4 ]];then
                              day=$4
                            fi
                            chmod 744 ./scripts/install_etrans.sh 
                            ./scripts/install_etrans.sh $username $pwsswd $day
                            exit 0;;
           install_local) rm -rf ./ascend_download/out
                          if [[ -f "./ascend_download/x86_ubuntu_os_devtoolset_package.zip" ]];then
                            chmod 744 ./scripts/install_etrans.sh
                            ./scripts/install_etrans.sh "install_local"
                          elif [[ -f "./ascend_download/arm_erler29_os_devtoolset_package.zip" ]];then
                            chmod 744 ./scripts/install_etrans.sh
                            ./scripts/install_etrans.sh "install_local"
                          else
                            chmod 744 ./scripts/install_daily.sh
                            ./scripts/install_daily.sh  "install_local"
                          fi 
                          exit 0;;
           *) for m in [ O0 O1 O2 O3 g ]
                do
                  if [[ "build_mode_$m" =~ "$OPTARG" ]];then
                    build_mode=$m
                  fi
                done
                
              get_target_lib
              if [[ "$RELEASE_VERSION" =~ "$OPTARG" ]];then
                UT_TEST_ALL=FALSE
                lib=$OPTARG
                create_lib_tag=TRUE
              elif [[ "$UT_VERSION" =~ "$OPTARG" ]];then
                no_all_ut=FALSE
                if [[ $OPTARG =~ "pass" ]];then
                  PASS_UT=TRUE
                  no_all_ut=TRUE
                fi
                if [[ $OPTARG =~ "tiling" ]];then
                  TILING_UT=TRUE
                  no_all_ut=TRUE
                fi
                if [[ $OPTARG =~ "proto" ]];then
                  PROTO_UT=TRUE
                  no_all_ut=TRUE
                fi
                if [[ $OPTARG =~ "plugin" ]];then
                  PLUGIN_UT=TRUE
                  no_all_ut=TRUE
                fi
                if [[ $OPTARG =~ "onnx" ]];then
                  ONNX_PLUGIN_UT=TRUE
                  no_all_ut=TRUE
                fi
                if [[ "$no_all_ut" =~ "FALSE" ]];then
                   UT_TEST_ALL=TRUE
                fi
                lib=$OPTARG
                create_lib_tag=TRUE
              else
                  if [[ "FALSE" =~ "$build_mode" ]];then
                    logging "Undefined option: $OPTARG"
                    usage
                    exit 1
                  fi
              fi                  
              esac;;
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
  
  if [[ "$UT_MODE" == "TRUE" ]]; then
    related_ut=`python3.7 scripts/parse_changed_files.py $1`
    logging "related ut "$related_ut
  else
    related=`python3.7 scripts/parse_compile_changed_files.py $1`
  fi 

  if [[ "$UT_MODE" == "TRUE" ]]; then 
      if [[ $related_ut =~ "CPU_UT" ]];then
        logging "CPU_UT is triggered!"
        CPU_UT=TRUE
        EXEC_UT=TRUE
      fi
      if [[ $related_ut =~ "PASS_UT" ]];then
        logging "PASS_UT is triggered!"
        PASS_UT=TRUE
        EXEC_UT=TRUE
      fi
      if [[ $related_ut =~ "TILING_UT" ]];then
        logging "TILING_UT is triggered!"
        TILING_UT=TRUE
        EXEC_UT=TRUE
      fi
      if [[ $related_ut =~ "PROTO_UT" ]];then
        logging "PROTO_UT is triggered!"
        PROTO_UT=TRUE
        EXEC_UT=TRUE
      fi
      if [[ $related_ut =~ "PLUGIN_UT" ]];then
        logging "PLUGIN_UT is triggered!"
        PLUGIN_UT=TRUE
        EXEC_UT=TRUE
      fi
      if [[ $related_ut =~ "ONNX_PLUGIN_UT" ]];then
        logging "ONNX_PLUGIN_UT is triggered!"
        ONNX_PLUGIN_UT=TRUE
        EXEC_UT=TRUE
      fi
  fi
  if [[ "$UT_MODE" == "TRUE" ]]; then 
    if [[ "$EXEC_UT" =~ "FALSE" ]];then
      logging "no ut matched! no need to run!"
      logging "---------------- CANN build finished ----------------"
      #for ci,2 means no need to run c++ ut;then ci will skip check coverage
      exit 200
    fi
  fi
}

compile_mod(){
      mk_dir "${CMAKE_HOST_PATH}"
      cd "${CMAKE_HOST_PATH}" && cmake ${CMAKE_ARGS} ../..
      compiled=FALSE
      rely_TF=(ops_all_plugin)
      rely_CAFFE=(ops_all_caffe_plugin)
      rely_ONNX=(ops_all_onnx_plugin)
      rely_PASS=(ops_fusion_pass_aicore ops_fusion_pass_vectorcore optiling)
      rely_TILING=(optiling)
      rely_PROTO=(opsproto ops_all_plugin ops_all_onnx_plugin optiling ops_fusion_pass_aicore ops_fusion_pass_vectorcore)
      mods=(TF CAFFE ONNX PASS TILING PROTO)
      for mod in "${mods[@]}"
        do
          if [[ "$related" =~ "$mod" ]]; then
          eval libs=('"${rely_'${mod}'[@]}"')
          for lib in  ${libs[@]}
            do
              lib_compiled=${lib}_compiled
              compiled=$(eval echo \$$lib_compiled)
              if [[ "$compiled" =~ "TRUE" ]];then
                echo $dotted_line
                echo $lib compiled
              else
                cmake --build . --target $lib -- -j ${THREAD_NUM}
                echo $dotted_line
                echo $lib
                eval "${lib}_compiled"=TRUE
              fi
            done
          fi
          done
      cmake --build . --target repack_tbe -- -j ${THREAD_NUM}
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
            -DPLUGIN_UT=$PLUGIN_UT -DONNX_PLUGIN_UT=$ONNX_PLUGIN_UT
            -DBUILD_MODE=$build_mode"

  logging "Start build host target. CMake Args: ${CMAKE_ARGS}"
  
  computer_arch=`uname -m`
  if [[ "$UT_MODE" == "FALSE"  ]] && [[ "$CI_MODE" == "TRUE"  ]] && [[ ! "$related" =~ "OTHER_FILE" ]] && [[ ! "$related" =~ "CPU" ]];then
      compile_mod
      CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE -DPRODUCT_SIDE=device -DBUILD_MODE=$build_mode"
      logging "Start build device target. CMake Args: ${CMAKE_ARGS}"
      mk_dir "${CMAKE_DEVICE_PATH}"
      cd "${CMAKE_DEVICE_PATH}" && cmake ${CMAKE_ARGS} ../..
      make ${VERBOSE} -j${THREAD_NUM}   
   else
      if [[ "$ST_TEST" == "FALSE" ]]; then
        mk_dir "${CMAKE_HOST_PATH}"
        cd "${CMAKE_HOST_PATH}" && cmake ${CMAKE_ARGS} ../..
        make ${VERBOSE} -j${THREAD_NUM}
      fi
      if [ "$UT_TEST_ALL" == "FALSE" -a "$CPU_UT" == "FALSE" \
            -a "$PASS_UT" == "FALSE" -a "$TILING_UT" == "FALSE" \
            -a "$PROTO_UT" == "FALSE" -a "$PLUGIN_UT" == "FALSE" \
            -a "$ONNX_PLUGIN_UT" == "FALSE" ]; then
        CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE -DPRODUCT_SIDE=device -DBUILD_MODE=$build_mode"

        logging "Start build device target. CMake Args: ${CMAKE_ARGS}"
        mk_dir "${CMAKE_DEVICE_PATH}"
        cd "${CMAKE_DEVICE_PATH}" && cmake ${CMAKE_ARGS} ../..
        make ${VERBOSE} -j${THREAD_NUM}
      fi  
  fi
  logging "CANN build success!"
}

minirc(){
  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE -DPRODUCT_SIDE=device -DMINRC=TRUE -DBUILD_MODE=$build_mode"
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
  if [ $create_lib_tag ];then
    args=`echo -n $@`
    CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DBUILD_OPEN_PROJECT=TRUE\
    -DUT_TEST_ALL=$UT_TEST_ALL -DST_TEST=$ST_TEST -DAICPU_ONLY=$AICPU_ONLY\
    -DCPU_UT=$CPU_UT -DPASS_UT=$PASS_UT -DTILING_UT=$TILING_UT\
    -DPROTO_UT=$PROTO_UT -DPLUGIN_UT=$PLUGIN_UT -DONNX_PLUGIN_UT=$ONNX_PLUGIN_UT\
    -DUT_NO_EXEC=$UT_NO_EXEC -DBUILD_MODE=$build_mode" 
    create_lib
    exit 0
  fi
  
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
if [[ "$@" == "-h" ]];then
  main "$@"
else
  main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
fi
