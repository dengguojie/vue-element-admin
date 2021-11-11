set -e
install_python(){
  wget https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/Python-3.7.5.tgz
  tar -zxvf Python-3.7.5.tgz
  cd Python-3.7.5/
  ./configure --enable-shared --with-ssl
  make -j $core_nums
  make install
  ln -s /usr/local/lib/libpython3.7m.so /usr/lib/libpython3.7m.so.1.0
}

install_cmake(){
  sudo apt-get remove --purge cmake
  hash -r
  apt install -y build-essential libssl-dev
  wget https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cmake-3.18.4.tar.gz
  tar -zxvf cmake-3.18.4.tar.gz
  cd cmake-3.18.4
  ./bootstrap
  make -j $core_nums
  make install
}

main(){
  core_nums=$(cat /proc/cpuinfo| grep "processor"| wc -l)
  if [ $core_nums -ne 1 ];then
    core_nums=$((core_nums-1))
  fi
  echo "" > /etc/apt/sources.list
  echo deb https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse >> /etc/apt/sources.list
  echo deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse >> /etc/apt/sources.list
  echo deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse >> /etc/apt/sources.list
  echo deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse >> /etc/apt/sources.list
  echo deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse >> /etc/apt/sources.list
  echo deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse >> /etc/apt/sources.list
  echo deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse >> /etc/apt/sources.list
  echo deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse >> /etc/apt/sources.list
  echo deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse >> /etc/apt/sources.list
  echo deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse >> /etc/apt/sources.list

  apt-get update -y
  apt-get upgrade -y

  rm -rf ~/.pip
  mkdir ~/.pip
  touch ~/.pip/pip.conf
  echo [global] >> ~/.pip/pip.conf
  echo index-url = https://mirrors.aliyun.com/pypi/simple >> ~/.pip/pip.conf
  
  for lib in gcc gdb git autoconf automake make libtool curl g++ unzip zlib1g-dev libffi-dev lcov build-essential libssl-dev libbz2-dev libncurses5-dev libgdbm-dev liblzma-dev sqlite3 libsqlite3-dev openssl libssl-dev tcl8.6-dev tk8.6-dev libreadline-dev zlib1g-dev
  do
    apt-get install -y $lib
    res=`echo $?`
    if [  $res -ne 0 ];then
      echo "apt install $lib failed, please check"
      exit -1
    fi
  done
  
  cd /usr/local/src
  install_python
  res_install_python=`echo $?`
  if [  $res_install_python -ne 0 ];then
    echo "install python failed, please check"
    exit -1
  fi
  
  cd ..
  install_cmake
  res_install_cmake=`echo $?`
  if [  $res_install_cmake -ne 0 ];then
    echo "install cmake failed, please check"
    exit -1
  fi
}

main
