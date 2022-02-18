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
SCRIPT_NAME_OF_GEN_OPINFO="gen_opinfo_json_from_ini.sh"
SCRIPT_NAME_OF_GEN_OPCINFO="gen_opcinfo_from_opinfo.py"

main() {
  echo "[INFO]excute file: $0"
  if [ $# -ne 2 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 soc_version out_opcinfo_csv_file"
    exit 1
  fi
  soc_version=$1
  out_opcinfo_csv_file=$2
  echo "[INFO]arg1: ${soc_version}"
  echo "[INFO]arg2: ${out_opcinfo_csv_file}"
  # check
  output_file_extrnsion=${out_opcinfo_csv_file##*.}
  if [ ${output_file_extrnsion} != "csv" ]; then
    echo "[ERROR]the output must be .csv, but is ${output_file_extrnsion}"
    exit 1
  fi
  soc_version_lower=${soc_version,,}
  workdir=$(cd $(dirname $0); pwd)
  
  script_file0="${workdir}/${SCRIPT_NAME_OF_GEN_OPINFO}"
  echo "[INFO]gen opinfo script path: ${script_file0}"
  if [ ! -f "${script_file0}" ]; then
    echo "[ERROR]gen opinfo script path is not exited, return fail"
    exit 1
  fi
  script_file1="${workdir}/${SCRIPT_NAME_OF_GEN_OPCINFO}"
  echo "[INFO]gen opcinfo script path: ${script_file1}"
  if [ ! -f "${script_file1}" ]; then
    echo "[ERROR]gen opcinfo script path is not exited, return fail"
    exit 1
  fi
  # step 1. gen opinfo json from ini opinfo with script gen_opinfo_json_from_ini.sh
  workspace_json_file="./_${soc_version_lower}.json"
  gen_cmd="bash ${script_file0} ${soc_version} ${workspace_json_file}"
  ${gen_cmd}
  if [ $? -ne 0 ]; then
    echo "[ERROR]exe failed"
    exit 1
  fi

  # step2. gen opcinfo csv with script gen_opcinfo_from_opinfo.py
  parer_cmd="python3.7 ${script_file1} ${workspace_json_file} ${out_opcinfo_csv_file}"
  echo "[INFO]parser cmd: ${parer_cmd}"
  ${parer_cmd}
}
set -o pipefail
main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
