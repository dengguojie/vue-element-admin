#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

repo_host=""
install_type=""
appoint_day=""
parse_args() {
  case $1 in
    "install_local")
      install_type="local"
    ;;
    "install_etrans")
      install_type="etrans"
      repo_host=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com
    ;;
    "install_daily")
      install_type="daily"
      repo_host=https://container-obsfs-filesystem.obs.myhuaweicloud.com
    ;;
    *)
      echo "invalid parameter. Should be:"
      echo "  install_cann.sh install_local|install_etrans|install_daily [specified_date]"
      echo "  specified_date is only valid for install_etrans & install_daily"
      exit 1
    ;;
  esac

  if [[ $# -ge 1 ]] && [[ x"${install_type}" != xlocal ]]; then
    appoint_day=$2
  fi
}

get_arch(){
  echo $dotted_line
  computer_arch=`uname -m`
  if [[ "$computer_arch" =~ "x86" ]];then
    arch="x86"
    if [[ x"${install_type}" == xdaily ]]; then
      file="Ascend-cann-toolkit_5.1.RC2.alpha002_linux-x86_64.run"
    else
      file="ai_cann_x86.tar.gz"
    fi
  elif [[ "$computer_arch" =~ "aarch64" ]];then
    arch="aarch64"
    if [[ x"${install_type}" == xdaily ]]; then
      file="Ascend-cann-toolkit_5.1.RC2.alpha002_linux-aarch64.run"
    else
      file="ai_cann_arm.tar.gz"
    fi
  else
    echo "not support $arch"
    echo $dotted_line
    exit 1
  fi
  echo "Get computer architecture as: ${arch}"
}

print_network_result(){
  url=$1
  result=$2
  if [[ ${result} -ne 0 ]];then
    echo "Can not connect ${url}. please check..."
    echo $dotted_line
    exit 1
  else
    echo ${url} is connected
  fi
}

test_network() {
  echo $dotted_line
  wget -qS --method=HEAD ${repo_host} --no-check-certificate 2>&1 | grep -q OBS
  ret=`echo $?`
  if [[ ${ret} -ne 0 ]]; then
    echo "Connect to ${repo_host} failed. please check..."
    echo "If you are in Huawei yellow zone, please:"
    echo "  export http_proxy=http//\$username:\$escape_pass@\${proxy:-proxy}.huawei.com:8080/"
    echo "  export https_proxy=\$http_proxy"
    echo "NOTICE: password needs to be escaped"
    echo "If you are not in Huawei yellow zone, please configure a network proxy."
    echo $dotted_line
    exit 1
  fi
  echo "${repo_host} is connected"
}

build_download_url() {
  local day=$1
  if [[ x"$install_type" == xdaily ]]; then
    local month=`date -d "${day}" +%Y%m`
    download_url="${repo_host}/package/daily/${month}/${day}0000/${arch}/newest/${file}"
  else
    download_url="${repo_host}/CANN_daily_y2b/${day}_newest/${file}"
  fi
}

download_run(){
  download_url=""
  if [[ "$appoint_day" =~ ^20 ]];then
    echo "The package is specified as archived on ${appoint_day}"
    build_download_url $appoint_day
  fi

  if [[ x"${download_url}" == x ]]; then
    start=`date +%Y%m%d`
    end=`date -d "-28 day ${start}" +%Y%m%d`
    while [ ${start} -ge ${end} ]
    do
      build_download_url ${start}
      set +e
      wget -q --method=HEAD "${download_url}" --no-check-certificate 2>/dev/null
      ret=$?
      set -e
      if [[ ${ret} -eq 0 ]]; then
        echo "The latest daily package is archived on ${start}"
        break
      fi
      start=`date -d "-1 day ${start}" +%Y%m%d`
      download_url=""
      sleep 0.2
    done
  fi

  if [[ x"${download_url}" == x ]]; then
    echo "No valid package is updated in the last 28 days. Please check ..."
    echo $dotted_line
    exit 1
  fi

  echo $dotted_line
  echo "Going to download ${file} ..."
  rm -rf ascend_download
  mkdir ascend_download
  set +e
  wget -P ascend_download "${download_url}" --no-check-certificate
  ret=`echo $?`
  set -e
  if [[ ${ret} -ne 0 ]]; then
    echo "Download package \"${download_url}\" failed. please check..."
    echo $dotted_line
    exit 1
  fi
}

bak_ori_Ascend(){
  bak_time=$(date "+%Y%m%d%H%M%S")
  echo $dotted_line
  echo "Backup the old version of Ascend ..."
  if [[ $UID -eq 0 ]];then
    if [  -d "/usr/local/Ascend" ];then
      mv /usr/local/Ascend  /usr/local/Ascend_$bak_time
    fi
  else
    if [[ -d ~/Ascend ]];then
      mv ~/Ascend  ~/Ascend_$bak_time
    fi
  fi
}

extract_pack(){
  echo $dotted_line
  echo "Going to install new version of Ascend ..."
  rm -rf ascend_download/out
  mkdir -p ascend_download/out
  local tmp_install_type=$install_type
  if [[ x"$install_type" == xlocal ]]; then
    ls -l ascend_download | grep -Eq "ai_cann_.*\.tar\.gz"
    local ret=$?
    if [ $ret -eq 0 ]; then
      install_type="etrans"
    else
      ls -l ascend_download 2>/dev/null | grep -Eq "Ascend-.*\.run"
      ret=$?
      if [ $ret -ne 0 ]; then
        echo "There is no package in ascend_download. Please check ..."
        echo $dotted_line
        exit 1
      fi
      install_type="daily"
    fi
  fi

  if [[ x"$install_type" == xdaily ]]; then
    chmod 744 ascend_download/Ascend-*.run
    ascend_download/Ascend-*.run --noexec --extract=ascend_download/out
  else
    tar -zxvf  ascend_download/${file} -C ascend_download/out
  fi
}

install_Ascend(){
  if [[ $UID -eq 0 ]];then
    set +e
    useradd HwHiAiUser
    set -e
  fi

  local folder=""
  if [[ x"$install_type" == xdaily ]]; then
    folder=ascend_download/out/run_package
  else
    folder=ascend_download/out/${file%%.*}
  fi

  find ${folder} -name "*.run" 2>/dev/null | xargs chmod 744 2>/dev/null

  opp=`ls ${folder} | grep "^CANN" | grep opp | grep -v atlas`
  if [[ x"$opp" == x ]]; then
    echo "The opp package does not exist, please check."
    exit 1
  else
    ${folder}/$opp --full
  fi

  for pack in compiler toolkit runtime
  do
    if [ -f ${folder}/CANN-${pack}* ];then
      ${folder}/CANN-${pack}*  --full
    else
      echo "The ${pack} package does not exist, please check."
      exit 1
    fi
  done

  if [[ $arch =~ "x86" ]];then
    arch="x86_64"
  fi
  if [[ $UID -eq 0 ]];then
    if [  -d "/usr/local/Ascend" ];then
      ln -s  /usr/local/Ascend/latest/${arch}-linux/lib64/libruntime.so  /usr/local/Ascend/latest/compiler/lib64/libruntime.so
    fi
  else
    chmod +w ~/Ascend/latest/compiler/lib64
    ln -s  ~/Ascend/latest/${arch}-linux/lib64/libruntime.so  ~/Ascend/latest/compiler/lib64/libruntime.so
    chmod -w ~/Ascend/latest/compiler/lib64
  fi
}

parse_args $@
get_arch
if [[ x${install_type} != xlocal ]];then
  test_network
  download_run
fi
bak_ori_Ascend
extract_pack
install_Ascend

echo $dotted_line
echo "Successfully installed Ascend."

if [[ $UID -eq 0 ]];then
  echo "The Ascend install path is /usr/local/Ascend, the old version is /usr/local/Ascend_$bak_time"
else
  echo "The Ascend install path is ~/Ascend, the old version is ~/Ascend_$bak_time"
fi
