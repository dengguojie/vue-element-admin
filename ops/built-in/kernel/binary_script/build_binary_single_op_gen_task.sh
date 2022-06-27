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
source trans_soc_by_bash.sh

main() {
  echo "[INFO]excute file: $0"
  if [ $# -lt 5 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 {op_type} {soc_version} {opp_run_path} {output_path} {task_path} {opc_info_csv}(optional)"
    exit 1
  fi
  workdir=$(cd $(dirname $0); pwd)
  op_type=$1
  soc_version=$2
  soc_version_lower=${soc_version,,}
  opp_run_path=$3
  output_path=$4
  task_path=$5
  opc_info_csv=""
  is_need_gen_opc_info=TRUE
  python_arg=${HI_PYTHON}
  if [ "${python_arg}" = "" ]; then
    python_arg="python3.7"
  fi
  if [ $# -gt 5 ]; then
    echo "[INFO]will get opcinfo from input arg"
    opc_info_csv=$6
    is_need_gen_opc_info=FALSE
  fi
  if [ "${is_need_gen_opc_info}" = "TRUE" ];then
    echo "[INFO]begin to gen opcinfo use ${SCRIPT_NAME_OF_GEN_OPCINFO}"
    offset_name=`date +%Y%m%d%H%M%S`
    opc_info_csv="_${soc_version_lower}_tmp_${offset_name}.csv"
    gen_file="${workdir}/${SCRIPT_NAME_OF_GEN_OPCINFO}"
    cmd="bash -x ${gen_file} ${soc_version} ${opc_info_csv}"
    ${cmd}
    echo "[INFO]end to gen opcinfo use ${SCRIPT_NAME_OF_GEN_OPCINFO}"
  fi

  # step0: gen task
  source build_env.sh
  opc_task_cmd_file="./${task_path}/${opc_task_name}"
  out_task_cmd_file="./${task_path}/${out_task_name}"
  echo "[INFO]op:${op_type}: opc_task_cmd_file=${opc_task_cmd_file}"
  echo "[INFO]op:${op_type}: out_task_cmd_file=${out_task_cmd_file}"

  # step1: get op python file name from opc_info_csv
  cmd="sed -n "/^${op_type},/{p\;q\;}" ${opc_info_csv}"
  opc_info_list=$(${cmd})
  echo "[INFO]op:${op_type} get the opc info from ${opc_info_csv} is ${opc_info_list}"
  if [ "${opc_info_list}" == "" ];then
    echo "[WARNING]op:${op_type} do not get the opc info from ${opc_info_csv}, will ignore"
    exit 0
  fi
  op_python_file=`echo ${opc_info_list} | awk -F',' 'BEGIN{OFS=",";} { if (NR==1)print $2}' |tr -d '\\n' | tr -d '\\r'`
  # step2: get op function name from opc_info_csv
  op_func=`echo ${opc_info_list} | awk -F',' 'BEGIN{OFS=",";} { if (NR==1)print $3}' |tr -d '\\n' | tr -d '\\r'`
  # step3: get binary json name from config binary_config.csv or binary_config_{soc_version}.csv
  common_binary_config="${workdir}/../binary_config/${COMMON_BINARY_CONCIG_NAME}"
  cmd="sed -n "/^${op_type},/{p\;q\;}" ${common_binary_config}"
  opc_info_list=$(${cmd})
  binary_config_file=`echo ${opc_info_list} | awk -F',' 'BEGIN{OFS=",";} { if (NR==1)print $2}' |tr -d '\\n' | tr -d '\\r'`
  binary_config_file=`echo ${binary_config_file// /}`
  if [ "${binary_config_file}" == "" ];then
    echo "[WARNING]op:${op_type} do not get the binary config file, will ignore"
    exit 0
  fi
  echo "[INFO]op:${op_type} get the binary config file is ${binary_config_file}"

  # step 4: concat the full path and check
  op_python_full_path="${workdir}/../../tbe/impl/${op_python_file}"
  binary_config_full_path="${workdir}/../binary_config/${soc_version_lower}/${op_type}/${binary_config_file}"
  op_file_name=${op_python_full_path##*/}
  op_file_name_prefix=${op_file_name%.*}
  binary_compile_full_path="${output_path}/op_impl/built-in/ai_core/tbe/kernel/${soc_version_lower}/${op_file_name_prefix}/"
  binary_compile_json_full_path="${output_path}/op_impl/built-in/ai_core/tbe/kernel/config/${soc_version_lower}/${op_file_name_prefix}.json"
  binary_compile_json_path="${output_path}/op_impl/built-in/ai_core/tbe/kernel/config/${soc_version_lower}"
  echo "[INFO]op:${op_type} python full path is ${op_python_full_path}"
  echo "[INFO]op:${op_type} python func is ${op_func}"
  echo "[INFO]op:${op_type} binary config full path is ${binary_config_full_path}"
  echo "[INFO]op:${op_type} binary bin full path is ${binary_compile_full_path}"
  echo "[INFO]op:${op_type} binary json full path is ${binary_compile_json_full_path}"
  if [ ! -f ${binary_config_full_path} ];then
    echo "[WARNING]op:${op_type} do not find the binary config file for ${binary_config_full_path}, will ignore"
    exit 0
  fi
  if [ ! -d ${binary_compile_full_path} ];then
    echo "[INFO]op:${op_type} create binary bin path ${binary_compile_full_path}"
    mkdir -p ${binary_compile_full_path}
  fi
  if [ ! -d ${binary_compile_json_path} ];then
    echo "[INFO]op:${op_type} create binary json path ${binary_compile_json_path}"
    mkdir -p ${binary_compile_json_path}
  fi

  # step 5: get impl_mode from all_ops_impl_mode.ini
  impl_mode_full_path="${workdir}/../../${IMPL_FILE_NAME}"
  impl_list=`awk -F '=' '/^'${op_type}'=/{print $2;exit}' ${impl_mode_full_path}`
  if [ "${impl_list}" = "" ]; then
    # 默认高性能模式
    impl_list="high_performance"
  fi

  if [ -f ${binary_compile_json_full_path} ];then
    echo "[WARNING]op:${op_type} will clean ${binary_compile_json_full_path}"
    rm -f {binary_compile_json_full_path}
  fi
  # step 6: do opc compile all kernel
  impl_list_array=(${impl_list//,/ })

  # 获取opc 入参的SOC version
  opc_soc_version=$(trans_soc ${soc_version})

  # 根据配置文件 确认某个算子kernel编译时拆分个数
  # get_thread_num ${op_type}

  # 所有算子的kernel  onebyone 进行编译
  get_thread_num_with_json_config ${binary_config_full_path}
  thread_num=$?
  echo "[INFO]op:${op_type} thread_num = ${thread_num}"
  opc_json_list=()
  for impl_mode in ${impl_list_array[@]}
  do
    # gen new json file for impl_mode from ${binary_config_full_path}
    binary_config_new_path=${binary_config_full_path%/*}
    binary_config_name=${binary_config_full_path##*/}
    binary_config_name_prefix=${binary_config_name%.*}
    binary_config_new_full_path="${binary_config_new_path}/${binary_config_name_prefix}_${impl_mode}.json"
    ${python_arg} gen_opc_json_with_impl_mode.py ${binary_config_full_path} ${binary_config_new_full_path} ${impl_mode}
    ${python_arg} gen_opc_json_with_threadnum.py ${binary_config_new_full_path} ${thread_num}
    # get new file with thread

    for(( i=0;i<${thread_num};i=i+1)); do
    {
      new_file="${binary_config_new_full_path}_${i}"
      cmd="opc ${op_python_full_path} --main_func=${op_func} --input_param=${new_file} --soc_version=${opc_soc_version} --output=${binary_compile_full_path} --impl_mode=${impl_mode}"
      echo "[INFO]op:${op_type} do opc cmdis is ${cmd}"
      echo ${cmd} >> ${opc_task_cmd_file}
      cmd="${python_arg} gen_output_json.py ${new_file} ${binary_compile_full_path} ${binary_compile_json_full_path}"
      echo "[INFO]op:${op_type} gen compile json cmdis is ${cmd}"
      echo ${cmd} >> ${out_task_cmd_file}
    }
    done
  done
}
set -o pipefail
main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
