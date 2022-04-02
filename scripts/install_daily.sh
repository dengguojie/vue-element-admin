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

if [[ "`echo $#`" -ge 1 ]];then
  appoint_day=$1
fi

repo_host=https://container-obsfs-filesystem.obs.myhuaweicloud.com

get_arch(){
  echo $dotted_line
  computer_arch=`uname -m`
  if [[ "$computer_arch" =~ "x86" ]];then
    arch="x86"
    filename="Ascend-cann-toolkit_5.1.RC1.alpha005_linux-x86_64.run"
  elif [[ "$computer_arch" =~ "aarch64" ]];then
    arch="aarch64"
    filename="Ascend-cann-toolkit_5.1.RC1.alpha005_linux-aarch64.run"
  else
    echo "not support $arch"
    echo $dotted_line
    exit -1
  fi
  echo "Get computer architecture as: ${arch}"
}

print_network_result(){
  url=$1
  result=$2
  if [[ ${result} -ne 0 ]];then
    echo "Can not connect ${url}. please check..."
    echo $dotted_line
    exit -1
  else
    echo ${url} is connected
  fi
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

  day=""
  if [[ "$appoint_day" =~ ^20 ]];then
    echo "The package is specified as archived on ${start}"
    day="${appoint_day:0:6}/${appoint_day}0000"
  fi

  if [ x"${day}" == x ]; then
    start=`date +%Y%m%d`
    end=`date -d "-28 day ${start}" +%Y%m%d`
    while [ ${start} -ge ${end} ]
    do
      month=`date -d "${start}" +%Y%m`
      set +e
      wget -q --method=HEAD "${repo_host}/package/daily/${month}/${start}0000/${arch}/newest/${filename}" --no-check-certificate 2>/dev/null
      ret=$?
      set -e
      if [ ${ret} -eq 0 ]; then
        echo "The latest daily package is archived on ${start}"
        day="${month}/${start}0000"
        break
      fi
      start=`date -d "-1 day ${start}" +%Y%m%d`
      sleep 0.2
    done
  fi

  if [ x${day} == x ]; then
    echo "No valid package is updated in the last 28 days. Please check ..."
    echo $dotted_line
    exit -1
  fi

  echo $dotted_line
  echo "Going to download ${filename} ..."
  rm -rf ascend_download
  mkdir ascend_download
  wget -P ascend_download "${repo_host}/package/daily/${day}/${arch}/newest/${filename}" --no-check-certificate
  ret=`echo $?`
  if [ ${ret} -ne 0 ]; then
    echo "Download package ${repo_host}/package/daily/${day}/${arch}/newest/${filename} failed. please check..."
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
  chmod 744 $dir/Ascend-*.run
  pwd
  $dir/Ascend-*.run --noexec --extract=./out
  cd out/run_package
}

install_Ascend(){
  if [[ $UID -eq 0 ]];then
    set +e
    useradd HwHiAiUser
    set -e
  fi

  for pack in compiler opp toolkit runtime
  do
    if [ -f ./CANN-${pack}* ];then
      if [ ${pack} == "atc" ];then
        ./CANN-${pack}* --pylocal --full
      else
        ./CANN-${pack}*  --full
      fi
    else
      echo "The ${pack} package does not exist, please check."
      exit -1
    fi
  done

  if [[ $arch =~ "x86" ]];then
    arch="x86_64"
  fi
  if [ $UID -eq 0 ];then
    if [  -d "/usr/local/Ascend" ];then
      ln -s  /usr/local/Ascend/${arch}-linux/lib64/libruntime.so  /usr/local/Ascend/compiler/lib64/libruntime.so
    fi
  else
    ln -s  ~/Ascend/${arch}-linux/lib64/libruntime.so  ~/Ascend/compiler/lib64/libruntime.so
  fi
}

if [[ "$install_local" =~ "FALSE" ]];then
  get_arch
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



