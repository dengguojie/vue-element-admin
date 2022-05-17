#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
# bash build_binary_single_op_exe_task.sh {opc_cmd_file} {out_cmd_file} {compile_thread_num}(optional, default=1)
set -e
SCRIPT_NAME_OF_GEN_OPCINFO="gen_opcinfo_for_socversion.sh"
COMMON_BINARY_CONCIG_NAME="binary_config.csv"
IMPL_FILE_NAME="tbe/impl_mode/all_ops_impl_mode.ini"

main() {
  echo "[INFO]excute file: $0"
  if [ $# -lt 1 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 {task_path} {compile_thread_num}(optional, default=1)"

    exit 1
  fi
  task_path=$1
  if [ $# -gt 1 ]; then
    compile_thread_num=$2
  else
    echo "[INFO]will set compile_thread_num = 1, use default value"
    compile_thread_num=1
  fi
  if [ ${compile_thread_num} -lt 1 ]; then
    echo "[INFO]will set compile_thread_num = 1, the compile_thread_num < 1"
    compile_thread_num=1
  fi

  source build_env.sh
  opc_cmd_file="./${task_path}/${opc_task_name}"
  out_cmd_file="./${task_path}/${out_task_name}"
  echo "[INFO]exe_task: opc_cmd_file = ${opc_cmd_file}"
  echo "[INFO]exe_task: out_cmd_file = ${out_cmd_file}"
  echo "[INFO]exe_task: compile_thread_num = ${compile_thread_num}"

  # step1: do compile kernel with compile_thread_num
  [ -e /tmp/binary_exe_task ] || mkfifo /tmp/binary_exe_task
  exec 8<>/tmp/binary_exe_task   # 创建文件描述符8
  rm -rf /tmp/binary_exe_task
  for ((i=1;i<=${compile_thread_num};i++))
  do
    echo >&8
  done

  opc_list_num=`cat ${opc_cmd_file} |wc -l`
  for ((i=1; i<=${opc_list_num}; i++))
  do
  read -u8
  {
    cmd_task=`sed -n ''${i}'p;' ${opc_cmd_file}`
    echo "[INFO]exe_task: begin to build kernel with cmd: ${cmd_task}."
    ${cmd_task}
    echo "[INFO]exe_task: end to build kernel with cmd: ${cmd_task}."
    echo >&8
  }&
  done
  wait  # 等待所有的算子编译结束

  # step2: gen output one by one
  cat ${out_cmd_file} | while read cmd_task
  do
    echo "[INFO]exe_task: begin to gen kernel list with cmd: ${cmd_task}."
    ${cmd_task}
  done
}
set -o pipefail
main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
