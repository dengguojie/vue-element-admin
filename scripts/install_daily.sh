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
username=$1
pwsswd=$2
if [[ "`echo $#`" -eq 3 ]];then
  appoint_day=$3
fi
net_addr=http://121.36.71.102/package/daily/
rm -rf index.html*
if [[ "$install_local" =~ "FALSE" ]];then
  rm -rf ascend_download
fi


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

network_test(){
  if [[ $1 -ne 0 ]];then
    echo $dotted_line
    echo "Can not connect $2. please check..."
    exit -1
  else
    echo $2 is connected
  fi
}

download_run(){
    set +e
    wget -q --http-user=$username --http-passwd=$pwsswd $net_addr
    #http://121.36.71.102/package/daily/202107/
    #   20210701/
    #   20210702/ 
    set -e
    if [ ! -f "./index.html" ];then 
      echo "Your account name or password is incorrect"
      exit -1
    fi
    res_net=`echo $?`
    test_net_addr=$net_addr
    network_test $res_net $test_net_addr
    #<tr><td class="link"><a href="202107/" title="202107">202107/</a></td><td class="size">-</td><td class="date">2021-Jul-29 00:35</td></tr>
    month=`cat index.html  | grep title |tail -n 1| awk '{print $4}' | awk -F ">" '{print $2}' | awk -F "/" '{print $1}'`
    if [[ "$appoint_day" =~ "20" ]];then
      month=${appoint_day:0:6}
    fi
    rm -rf index.html
    
    eval net=$(echo ${net_addr}${month}/)
    wget -q --http-user=$username --http-passwd=$pwsswd $net
    #http://121.36.71.102/package/daily/202107/
    #    20210701/
    #    20210702/
    res_net=`echo $?`
    test_net_addr=$net
    network_test $res_net $test_net_addr
    #</tbody></table></body></html><tr><td class="link"><a href="20210729/" title="20210729">20210729/</a></td><td class="size">-</td><td class="date">2021-Jul-29 00:35</td></tr>
    day=`cat index.html | grep title |tail -n 1| awk '{print $4}' | awk -F ">" '{print $2}' | awk -F "/" '{print $1}'`
    if [[ "$appoint_day" =~ "20" ]];then
      day=$appoint_day
    fi
    rm -rf index.html
    eval net=$(echo ${net}${day}/${arch}/)
    set +e
    wget -q --http-user=$username --http-passwd=$pwsswd $net
    #http://121.36.71.102/package/daily/202107/20210727/x86/
    #    master_20210727002645_ae990a97a4571341a59efad0a8cf9a7d01e6ce71_newest/
    
    if [[ ! -f ./index.html ]];then
      echo $dotted_line
      echo " "
      echo "Today's daily package does not exist, you have the following options"
      echo "(Recommended)       ./build.sh --install_etrans 'username' 'password' "
      echo "(Specify any date)  ./build.sh --install_etrans 'username' 'password' 20210101 "
      echo " "
      echo $dotted_line
      exit 1
    fi
    set -e
    res_net=`echo $?`
    test_net_addr=$net
    network_test $res_net $test_net_addr
    folder_index_content=`cat index.html`
    folder_name_fragment=${folder_index_content##*href=\"}
    folder_name=${folder_name_fragment%%/*}
    echo $folder_name
    rm -rf index.html
    
    eval net=$(echo ${net}${folder_name}/)
    wget -q --http-user=$username --http-passwd=$pwsswd $net
    #http://121.36.71.102/package/daily/202107/20210727/x86/master_20210727002645_ae990a97a4571341a59efad0a8cf9a7d01e6ce71_newest/
    #    master_20210727002645_ae990a97a4571341a59efad0a8cf9a7d01e6ce71_newest/Ascend-cann-toolkit_5.0.2.alpha005_linux-x86_64.run
    res_net=`echo $?`
    test_net_addr=$net

    network_test $res_net $test_net_addr
    file_index_content=`cat index.html | grep Ascend-cann-toolkit | grep run`
    file_name_fragment=${file_index_content%%.run*}
    file_name=${file_name_fragment#*href='"'}

    echo $file_name
    rm -rf index.html
    eval net=$(echo ${net}${file_name}.run)
    echo $dotted_line
    echo "Starting download ${file_name}.run"
    mkdir ascend_download
    wget -P ascend_download --http-user=$username --http-passwd=$pwsswd $net
    res_net=`echo $?`
    test_net_addr=$net
    network_test $res_net $test_net_addr
}

bak_ori_Ascend(){
    bak_time=$(date "+%Y%m%d%H%M%S")
    echo $dotted_line
    echo "Backup the original Ascend" 
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
    echo "start installation Ascend." 
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
          echo "The ${pack} package does not exist, please check "
          exit -1
        fi
      done
    if [ $UID -eq 0 ];then
      if [  -d "/usr/local/Ascend" ];then
        ln -s  /usr/local/Ascend/x86_64-linux/lib64/libruntime.so  /usr/local/Ascend/atc/lib64/libruntime.so
      fi
    else
      ln -s  ~/Ascend/x86_64-linux/lib64/libruntime.so  ~/Ascend/atc/lib64/libruntime.so
    fi
}
if [[ "$install_local" =~ "FALSE" ]];then
  network_test
  get_arch
  download_run
fi
bak_ori_Ascend
extract_pack
install_Ascend


echo $dotted_line
echo "Successfully installed Ascend."
if [[ "$install_local" =~ "FALSE" ]];then
  echo "Using ${net_addr}${month}/${day}/${arch}/${folder_name}/${file_name}.run" 
else
  echo "Using local run package" 
fi

if [ $UID -eq 0 ];then
  echo "The Ascend install path is /usr/local/Ascend, the ori is /usr/local/Ascend_$bak_time"
else
  echo "The Ascend install path is ~/Ascend, the ori is ~/Ascend_$bak_time"
fi


