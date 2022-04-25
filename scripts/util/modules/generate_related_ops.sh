#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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

get_related_ops() {
  if [[ $# -lt 2 ]]; then
    echo "[ERROR] invalid parameter for get_related_ops."
    echo "[ERROR] it should be: get_related_ops pr_file output_change_log"
    exit $STATUS_FAILED
  fi

  local pr_file="$1"
  local output_change_log="$2"

  change_log_dir=`dirname ${output_change_log}`
  mkdir -p ${change_log_dir}

  "$PYTHON" "${PY_GET_RELATED_OPS}" "${pr_file}" 2>&1 | tee "${output_change_log}"
  if [[ $? -ne 0 ]]; then
    echo "[ERROR] not found related ops stest"
    exit $STATUS_FAILED
  fi
}


# only for st now
install_related_ops() {
  if [[ $# -lt 3 ]]; then
    echo "[ERROR] invalid parameter for install_related_ops."
    echo "[ERROR] it should be: install_related_ops change_log ops_source_dir case_file_regexp"
    exit $STATUS_FAILED
  fi

  local change_log="$1"
  local ops_source_dir="$2"
  local case_file_regexp="$3"

  if [[ ! -f "${change_log}" ]]; then
    echo "[ERROR] no ops changed log found, run get_related_ops first"
    exit $STATUS_FAILED
  fi

  related_ops=$(cat ${change_log} | grep "^related_ops_dirs=" | awk -F\= '{print $2}')

  for op_name in $(echo $related_ops); do
    echo "[INFO] found related op [$op_name] to test"
    op_dir="${ops_source_dir}/${op_name}"
    if [[ ! -d "${op_dir}" ]]; then
      echo "[ERROR] no st directory found for ${op_dir}"
      # exit $STATUS_FAILED
    fi
    cases=$(find "${op_dir}" -name "${case_file_regexp}")
    if [[ -z $cases ]]; then
      echo "[ERROR] no testcases found in ${op_dir} for ${case_file_regexp}"
      # exit $STATUS_FAILED
    fi
    if [[ -z "${all_cases}" ]]; then
      all_cases="${op_dir}"
    else
      all_cases="${all_cases},${op_dir}"
    fi
  done
}
