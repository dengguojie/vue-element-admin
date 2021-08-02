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
net_addr=http://121.36.71.102/package/daily/
rm -rf index.html*
rm -rf ascend_download

network_test(){
  if [  $res_net -ne 0 ];then
    echo $dotted_line
    echo "Can not connect $test_net_addr. please check..."
    exit -1
  else
    echo ${test_net_addr} is connected
fi
}

get_arch(){
    echo $dotted_line
    echo "Get computer architecture"
    computer_arch=`uname -m`
    if [[ "$computer_arch" =~ "x86" ]];then
      arch="x86"
    elif [[ "$computer_arch" =~ "aarch64" ]];then
      arch="aarch64"
    else
      echo "not support $arch"
    fi
}

download_run(){
    wget -q --http-user=$username --http-passwd=$pwsswd $net_addr
    #http://121.36.71.102/package/daily/202107/
    #   20210701/
    #   20210702/    
    res_net=`echo $?`
    test_net_addr=$net_addr
    network_test
    #<tr><td class="link"><a href="202107/" title="202107">202107/</a></td><td class="size">-</td><td class="date">2021-Jul-29 00:35</td></tr>
    month=`cat index.html  | grep title |tail -n 1| awk '{print $4}' | awk -F ">" '{print $2}' | awk -F "/" '{print $1}'`
    rm -rf index.html
    
    wget -q --http-user=$username --http-passwd=$pwsswd ${net_addr}${month}/
    #http://121.36.71.102/package/daily/202107/
    #    20210701/
    #    20210702/
    res_net=`echo $?`
    test_net_addr=${net_addr}${month}/
    network_test
    #</tbody></table></body></html><tr><td class="link"><a href="20210729/" title="20210729">20210729/</a></td><td class="size">-</td><td class="date">2021-Jul-29 00:35</td></tr>
    day=`cat index.html | grep title |tail -n 1| awk '{print $4}' | awk -F ">" '{print $2}' | awk -F "/" '{print $1}'`
    rm -rf index.html
    
    wget -q --http-user=$username --http-passwd=$pwsswd ${net_addr}${month}/${day}/${arch}/
    #http://121.36.71.102/package/daily/202107/20210727/x86/
    #    master_20210727002645_ae990a97a4571341a59efad0a8cf9a7d01e6ce71_newest/
    res_net=`echo $?`
    test_net_addr=${net_addr}${month}/${day}/${arch}/
    network_test
    folder_index_content=`cat index.html`
    folder_name_fragment=${folder_index_content##*href=\"}
    folder_name=${folder_name_fragment%%/*}
    echo $folder_name
    rm -rf index.html
    
    wget -q --http-user=$username --http-passwd=$pwsswd ${net_addr}${month}/${day}/${arch}/${folder_name}/
    #http://121.36.71.102/package/daily/202107/20210727/x86/master_20210727002645_ae990a97a4571341a59efad0a8cf9a7d01e6ce71_newest/
    #    master_20210727002645_ae990a97a4571341a59efad0a8cf9a7d01e6ce71_newest/Ascend-cann-toolkit_5.0.2.alpha005_linux-x86_64.run
    res_net=`echo $?`
    test_net_addr=${net_addr}${month}/${day}/${arch}/${folder_name}/
    network_test
    file_index_content=`cat index.html`
    file_name_fragment=${file_index_content#*Ascend-cann-toolkit_5}
    file_name_fragment1=${file_name_fragment#*=\"}
    file_name=${file_name_fragment1%%.deb*}
    rm -rf index.html

    echo $dotted_line
    echo "Starting download ${file_name}.run"
    mkdir ascend_download
    wget -P ascend_download --http-user=$username --http-passwd=$pwsswd ${net_addr}${month}/${day}/${arch}/${folder_name}/${file_name}.run
    res_net=`echo $?`
    test_net_addr=${net_addr}${month}/${day}/${arch}/${folder_name}/${file_name}.run
    network_test
}

delete_ori_Ascend(){
    echo $dotted_line
    echo "Delete the original Ascend" 
    if [ $UID -eq 0 ];then
      rm -rf /usr/local/Ascend
    else
      rm -rf ~/Ascend
    fi
}

install_Ascend(){
    echo $dotted_line
    echo "start installation Ascend." 
    cd ascend_download
    dir=`pwd`
    chmod 744 $dir/${file_name}.run
    pwd
    $dir/${file_name}.run --noexec --extract=./out
    cd out/run_package

    if [ $UID -eq 0 ];then
      useradd HwHiAiUser
    fi

    if [ -f ./Ascend-atc* ];then
      ./Ascend-atc* --pylocal --full
    else
      echo "The atc package does not exist, please check "
      exit -1
    fi

    if [ -f ./Ascend-opp* ];then
      ./Ascend-opp*  --full
    else
      echo "The opp package does not exist, please check "
      exit -1
    fi

    if [ -f ./Ascend-toolkit* ];then
      ./Ascend-atc* --pylocal --full
    else
      echo "The toolkit package does not exist, please check "
      exit -1
    fi
}

network_test
get_arch
download_run
delete_ori_Ascend
install_Ascend


echo $dotted_line
echo "Successfully installed Ascend."
echo "Using ${net_addr}${month}/${day}/${arch}/${folder_name}/${file_name}.run" 
if [ $UID -eq 0 ];then
  echo "The Ascend install path is /usr/local/Ascend"
else
  echo "The Ascend install path is ~/Ascend"
fi
