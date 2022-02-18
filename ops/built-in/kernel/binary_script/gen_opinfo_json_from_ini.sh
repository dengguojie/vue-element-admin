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

set -e

main() {
  echo "[INFO]excute file: $0"
  if [ $# != 2 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 soc_version output_json_file"
    exit 1
  fi
  soc_version=$1
  output_file=$2
  echo "[INFO]arg1: ${soc_version}"
  echo "[INFO]arg2: ${output_file}"
  # check
  output_file_extrnsion=${output_file##*.}
  if [ ${output_file_extrnsion} != "json" ]; then
    echo "[ERROR]the output must be .json, but is ${output_file_extrnsion}"
    exit 1
  fi
  soc_version_lower=${soc_version,,}
  workdir=$(cd $(dirname $0); pwd)
  ini_file="${workdir}/../../tbe/op_info_cfg/ai_core/${soc_version_lower}/aic-${soc_version_lower}-ops-info.ini"
  echo "[INFO]op ini path: ${ini_file}"
  if [ ! -f "${ini_file}" ]; then
    echo "[ERROR]the ops ini file in env is not exited, return fail"
    exit 1
  fi

  parer_python_file="${workdir}/../../tbe/op_info_cfg/parser/parser_ini.py"
  if [ ! -f "$parer_python_file" ]; then
    echo "[ERROR]the ops parser file in env is not exited, return fail"
    exit 1
  fi
  parer_cmd="python3.7 ${parer_python_file} ${ini_file} ${output_file}"
  echo "[INFO]parser cmd: ${parer_cmd}"
  ${parer_cmd}
}
set -o pipefail
main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
