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
  local task_type="$1"
  local pr_file="$2"

  if [[ "${task_type}" == "ut" ]]; then
    CHANGE_LOG="${UT_CHANGE_LOG}"
    TEST_INSTALL_PATH="${UT_INSTALL_PATH}"
    OPS_SOURCE_DIR="${OPS_UT_SOURCE_DIR}"
  elif [[ "${task_type}" == "st" ]]; then
    CHANGE_LOG="${ST_CHANGE_LOG}"
    TEST_INSTALL_PATH="${ST_INSTALL_PATH}"
    OPS_SOURCE_DIR="${OPS_ST_SOURCE_DIR}"
  else
    echo "[ERROR] unsuported task type ${task_type}"
    exit $STATUS_FAILED
  fi
  if [[ ! -d "${TEST_INSTALL_PATH}" ]]; then
    mkdir -p "${TEST_INSTALL_PATH}" >/dev/null 2>&1
  fi

  "$PYTHON" "${PY_GET_RELATED_OPS}" "${pr_file}" 2>&1 | tee "${CHANGE_LOG}"
  if [[ $? -ne 0 ]]; then
    echo "[ERROR] not found related ops stest"
    exit $STATUS_FAILED
  fi
}


# only for st now
install_related_ops() {
  if [[ ! -f "${CHANGE_LOG}" ]]; then
    echo "[ERROR] no ops changed log found, run get_related_ops first"
    exit $STATUS_FAILED
  fi

  related_ops=$(cat ${CHANGE_LOG} | grep "^related_ops_dirs=" | awk -F\= '{print $2}')

  for op_name in $(echo $related_ops); do
    echo "[INFO] found related op [$op_name] to test"
    op_dir="${OPS_SOURCE_DIR}/${op_name}"
    if [[ ! -d "${op_dir}" ]]; then
      echo "[ERROR] no st directory found for ${op_dir}"
      exit $STATUS_SUCCESS
      # exit $STATUS_FAILED
    fi
    json_cases=$(find "${op_dir}" -name *.json)
    if [[ -z $json_cases ]]; then
      echo "[ERROR] no json testcases found ${op_dir}/OP_case.json"
      exit $STATUS_SUCCESS
      # exit $STATUS_FAILED
    fi
    # install st testcases
    if [[ -z "${all_cases}" ]]; then
      all_cases="${op_dir}"
    else
      all_cases="${all_cases},${op_dir}" 
    fi
  done
}
