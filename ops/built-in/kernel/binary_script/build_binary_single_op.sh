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
# bash build_binary_single_op.sh {op_type} {soc_version} {opp_run_path} {output_path} {opc_info_csv}(optional)
SCRIPT_NAME_OF_GEN_OPCINFO="gen_opcinfo_for_socversion.sh"
COMMON_BINARY_CONCIG_NAME="binary_config.csv"
IMPL_FILE_NAME="tbe/impl_mode/all_ops_impl_mode.ini"

source get_threadnum_with_op.sh

main() {
  echo "[INFO]excute file: $0"
  if [ $# -lt 4 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 {op_type} {soc_version} {opp_run_path} {output_path} {opc_info_csv}(optional)"
    exit 1
  fi
  op_type=$1

  # step0: gen task
  source build_env.sh
  task_path=${op_type}
  [ -d "./${task_path}/" ] || mkdir -p ./${task_path}/
  rm -f ./${task_path}/*
  if [ $# -gt 4 ]; then
    bash build_binary_single_op_gen_task.sh $1 $2 $3 $4 ${task_path} $5
  else
    bash build_binary_single_op_gen_task.sh $1 $2 $3 $4 ${task_path}
  fi

  get_thread_num ${op_type}
  thread_num=$?
  # step1: exe task
  bash build_binary_single_op_exe_task.sh ${task_path} ${thread_num}
}
set -o pipefail
main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
