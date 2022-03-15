#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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
set -e
SCRIPT_NAME_OF_GEN_OPCINFO="gen_opcinfo_for_socversion.sh"
COMMON_BINARY_CONCIG_NAME="binary_config.csv"

main() {
  echo "[INFO]excute file: $0"
  if [ $# -lt 4 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 {op_type} {soc_version} {opp_run_path} {output_path} {opc_info_csv}(optional)"
    exit 1
  fi
  workdir=$(cd $(dirname $0); pwd)
  op_type=$1
  soc_version=$2
  soc_version_lower=${soc_version,,}
  opp_run_path=$3
  output_path=$4
  opc_info_csv=""
  is_need_gen_opc_info=TRUE
  if [ $# -gt 4 ]; then
    echo "[INFO]will get opcinfo from input arg"
    opc_info_csv=$5
    is_need_gen_opc_info=FALSE
  fi
  if [ "${is_need_gen_opc_info}" = "TRUE" ];then
    echo "[INFO]begin to gen opcinfo use ${SCRIPT_NAME_OF_GEN_OPCINFO}"
    offset_name=`date +%Y%m%d%H%M%S`
    opc_info_csv="_${soc_version_lower}_tmp_${offset_name}.csv"
    gen_file="${workdir}/${SCRIPT_NAME_OF_GEN_OPCINFO}"
    cmd="bash ${gen_file} ${soc_version} ${opc_info_csv}"
    ${cmd}
    echo "[INFO]end to gen opcinfo use ${SCRIPT_NAME_OF_GEN_OPCINFO}"
  fi

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

  # step 5: do opc compile
  opc_install_path="/usr/local/Ascend/latest/compiler/python/site-packages"
  cmd="python3.7 ${opc_install_path}/opc_tool/opc.py ${op_python_full_path} --main_func=${op_func} --input_param=${binary_config_full_path} --soc_version=${soc_version} --output=${binary_compile_full_path}"
  echo "[INFO]op:${op_type} do opc compile cmdis is ${cmd}"
  ${cmd}

  # step6: post process to gen op.json
  cmd="python3.7 gen_output_json.py ${binary_config_full_path} ${binary_compile_full_path} ${binary_compile_json_full_path}"
  echo "[INFO]op:${op_type} gen compile json cmdis is ${cmd}"
  ${cmd}
}
set -o pipefail
main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
