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
username=$1
pwsswd=$2
net_addr=http://121.36.71.102/package/etrans/
rm -rf index.html*
rm -rf ascend_download

network_test(){
  if [[ $res_net -ne 0 ]];then
    echo $dotted_line
    echo "Can not connect $test_net_addr. please check..."
    exit -1
  else
    echo ${test_net_addr} is connected
  fi
}

get_arch_and_filename(){
    echo $dotted_line
    echo "Get computer architecture"
    computer_arch=`uname -m`
    if [[ "$computer_arch" =~ "x86" ]];then
      arch="x86"
      filename="x86_ubuntu_os_devtoolset_package"
    elif [[ "$computer_arch" =~ "aarch64" ]];then
      arch="aarch64"
      filename="arm_erler29_os_devtoolset_package"
    else
      echo "not support $arch"
    fi
}

download_run(){
    set +e
    wget -q --http-user=$username --http-passwd=$pwsswd $net_addr
    #http://121.36.71.102/package/etrans/
    #   20210713/
    #   20210728/ 
    set -e
    if [ ! -f "./index.html" ];then 
      echo "Your account name or password is incorrect"
      exit -1
    fi	
    res_net=`echo $?`
    test_net_addr=$net_addr
    network_test
    #<tr><td class="link"><a href="20210728/" title="20210728">20210728/</a></td><td class="size">-</td><td class="date">2021-Jul-29 10:59</td></tr>
    day=`grep 202[0-9][0,1] index.html |tail -n 1| awk '{print $4}' | awk -F ">" '{print $2}' | awk -F "/" '{print $1}' | grep 20`
    rm -rf index.html
    
    eval net=$(echo ${net_addr}${day}/${filename}.zip)
    echo $dotted_line
    echo "Starting download ${filename}.zip"
    mkdir ascend_download
    wget -P ascend_download --http-user=$username --http-passwd=$pwsswd $net
    res_net=`echo $?`
    test_net_addr=$net
    network_test
}

bak_ori_Ascend(){
    bak_time=$(date "+%Y%m%d%H%M%S")
    echo $dotted_line
    echo "Delete the original Ascend" 
    if [ $UID -eq 0 ];then
      if [  -d "/usr/local/Ascend" ];then
        mv /usr/local/Ascend  /usr/local/Ascend_$bak_time
      fi
    else
      if [ -d "~/Ascend" ];then
        mv ~/Ascend  ~/Ascend_$bak_time
      fi
    fi
}

extract_pack(){
    echo $dotted_line
    echo "start installation Ascend." 
    cd ascend_download
    dir=`pwd`
    #chmod 744 $dir/${filename}.zip
    pwd
    unzip -d ./out ./${filename}.zip
    chmod -R 744 out/${filename}
    cd out/${filename}
}

install_Ascend(){
    if [[ $UID -eq 0 ]];then
      set +e
      useradd HwHiAiUser
      set -e
    fi
    
    for pack in atc opp toolkit
      do 
        if [ -f ./Ascend-${pack}* ];then
          if [ ${pack} == "atc" ];then
            ./Ascend-${pack}* --pylocal --full 
          else 
            ./Ascend-${pack}*  --full
          fi
        else
          echo "The ${pack} package does not exist, please check "
          exit -1
        fi
      done
}

network_test
get_arch_and_filename
download_run
bak_ori_Ascend
extract_pack
install_Ascend


echo $dotted_line
echo "Successfully installed Ascend."
echo "Using $net" 
if [ $UID -eq 0 ];then
  echo "The Ascend install path is /usr/local/Ascend, the ori is /usr/local/Ascend_$bak_time"
else
  echo "The Ascend install path is ~/Ascend, the ori is ~/Ascend_$bak_time"
fi
