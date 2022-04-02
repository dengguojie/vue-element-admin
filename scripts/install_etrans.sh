#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
dotted_line="----------------------------------------------------------------"
input_parameter=`echo $0 $*`
install_local=FALSE
if [[ "$input_parameter" =~ "install_local" ]];then
  install_local=TRUE
fi

repo_host=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com

get_arch_and_filename(){
  echo $dotted_line
  computer_arch=`uname -m`
  if [[ "$computer_arch" =~ "x86" ]];then
    arch="x86_64"
    filename="ai_cann_x86"
  elif [[ "$computer_arch" =~ "aarch64" ]];then
    arch="aarch64"
    filename="ai_cann_arm"
  else
    echo "not support $arch"
    echo $dotted_line
    exit -1
  fi
  echo "Get computer architecture as: ${arch}"
}

download_run(){
  echo $dotted_line
  wget -qS --method=HEAD ${repo_host} --no-check-certificate 2>&1 | grep -q OBS
  ret=`echo $?`
  if [ ${ret} -ne 0 ]; then
    echo "Connect to ${repo_host} failed. please check..."
    echo "If you are in Huawei yellow zone, please:"
    echo "  export http_proxy=http//\$username:\$escape_pass@\${proxy:-proxy}.huawei.com:8080/"
    echo "  export https_proxy=\$http_proxy"
    echo "NOTICE: password needs to be escaped"
    echo "If you are not in Huawei yellow zone, please configure a network proxy."
    echo $dotted_line
    exit -1
  fi
  echo "${repo_host} is connected"

  start=`date +%Y%m%d`
  end=`date -d "-28 day ${start}" +%Y%m%d`
  day=""
  while [ ${start} -ge ${end} ]
  do
    set +e
    wget -q --method=HEAD "${repo_host}/CANN_daily_y2b/${start}_newest/${filename}.tar.gz" --no-check-certificate 2>/dev/null
    ret=$?
    set -e
    if [ ${ret} -eq 0 ]; then
      echo "The latest etrans package is archived on ${start}"
      day=${start}
      break
    fi
    start=`date -d "-1 day ${start}" +%Y%m%d`
    sleep 0.2
  done

  if [ x${day} == x ]; then
    echo "No valid package is updated in the last 28 days. Please check ..."
    echo $dotted_line
    exit -1
  fi

  echo $dotted_line
  echo "Going to download ${filename}.tar.gz ..."
  rm -rf ascend_download
  mkdir ascend_download
  wget -P ascend_download "${repo_host}/CANN_daily_y2b/${day}_newest/${filename}.tar.gz" --no-check-certificate
  ret=`echo $?`
  if [ ${ret} -ne 0 ]; then
    echo "Download package ${repo_host}/CANN_daily_y2b/${day}_newest/${filename}.tar.gz failed. please check..."
    echo $dotted_line
    exit -1
  fi
}

bak_ori_Ascend(){
  bak_time=$(date "+%Y%m%d%H%M%S")
  echo $dotted_line
  echo "Backup the old version of Ascend ..."
  if [ $UID -eq 0 ];then
    if [  -d "/usr/local/Ascend" ];then
      mv /usr/local/Ascend  /usr/local/Ascend_$bak_time
    fi
  else
    code_pwd=`pwd`
    cd ~
    if [ -d "Ascend" ];then
      mv Ascend  Ascend_$bak_time
    fi
    cd $code_pwd
  fi
}

extract_pack(){
  echo $dotted_line
  echo "Going to install new version of Ascend ..."
  cd ascend_download
  dir=`pwd`
  pwd
  rm -rf out
  mkdir out
  tar -zxvf  ./${filename}.tar.gz -C ./out
  chmod -R 744 out/
  cd out/${filename}
}

install_Ascend(){
  if [[ $UID -eq 0 ]];then
    set +e
    useradd HwHiAiUser
    set -e
  fi

  opp=`ls | grep opp | grep -v atlas`
  for pack in compiler toolkit runtime
  do
    if [ -f ./CANN-${pack}* ];then
      ./CANN-${pack}*  --full
    else
      echo "The ${pack} package does not exist, please check."
      exit -1
    fi
  done

  if [ -f ./$opp ];then
    ./$opp --full
  else
    echo "The $opp package does not exist, please check."
    exit -1
  fi

  if [ $UID -eq 0 ];then
    if [  -d "/usr/local/Ascend" ];then
      ln -s  /usr/local/Ascend/${arch}-linux/lib64/libruntime.so  /usr/local/Ascend/compiler/lib64/libruntime.so
    fi
  else
    ln -s  ~/Ascend/${arch}-linux/lib64/libruntime.so  ~/Ascend/compiler/lib64/libruntime.so
  fi
}

get_arch_and_filename
if [[ "$install_local" =~ "FALSE" ]];then
  download_run
fi
bak_ori_Ascend
extract_pack
install_Ascend

echo $dotted_line
echo "Successfully installed Ascend."

if [ $UID -eq 0 ];then
  echo "The Ascend install path is /usr/local/Ascend, the old version is /usr/local/Ascend_$bak_time"
else
  echo "The Ascend install path is ~/Ascend, the old version is ~/Ascend_$bak_time"
fi
