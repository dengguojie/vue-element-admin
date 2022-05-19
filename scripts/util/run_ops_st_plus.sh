#!/bin/bash
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
export CUR_DIR=$(cd $(dirname $0); pwd)

STATUS_SUCCESS=0
STATUS_FAILED=1

CASE_SOURCE="${CUR_DIR}/st_plus"
RESULT_SUMMARY="${CUR_DIR}/result.txt"

run() {
  local op_dir="${CASE_SOURCE}"

  echo "[INFO]===============now run st_plus ==================="

  local cases=$(find "${op_dir}" -name "*.csv" | head -1 2>/dev/null)
  if [[ -n "$cases" ]]; then
    echo "[INFO] run case file: $cases"
    # only dynamic
    "${CUR_DIR}/tbe_toolkits/run.sh" "$cases" "${CUR_DIR}/st_plus_result.csv --perf-compare=false"
    if [[ $? -ne 0 ]]; then
      echo "[ERROR] run ops stest plus failed, case file is: $cases."
      exit $STATUS_FAILED
    fi
    echo "[INFO]===============end of running st_plus ==================="
  else
    echo "[INFO]===============no testcase to run ==================="
    exit $STATUS_SUCCESS
  fi
}

tar_results () {
  cd "${CUR_DIR}"
  local files=""
  if [[ -f st_plus_result.csv ]]; then
    files=${files}" st_plus_result.csv"
    if [[ -f "tbe_toolkits/scripts/gen_result_html.py" ]]; then
      python3 tbe_toolkits/scripts/gen_result_html.py "st_plus_result.csv"
      [[ -f "click_me_for_result.html" ]] && files=${files}" click_me_for_result.html"
    fi
  fi

  local log_count=`ls ${CUR_DIR}/tbetoolkits-*.log 2>/dev/null | wc -l`
  if [[ $log_count -gt 0 ]]; then
    files=${files}" tbetoolkits-*.log"
  fi

  if [[ -d ~/ascend/log/plog ]]; then
    files=${files}" ~/ascend/log/plog"
  fi

  if [[ -n "$files" ]]; then
    tar czvf result_st_plus.tar.gz $files
  fi
}

get_results() {
  local result_file="${CUR_DIR}/st_plus_result.csv"
  if [[ ! -f "${result_file}" ]]; then
    echo "[ERROR] st_plus_result.csv is not found in ${CUR_DIR}"
    exit $STATUS_FAILED
  fi

  local lines=$(wc -l "${result_file}" 2>/dev/null| awk '{print $1}')
  if [[ $lines -le 1 ]]; then
    echo "[ERROR] st_plus_result.csv is empty or only titiles in it."
    exit $STATUS_FAILED
  fi
  local total_count=$(expr $lines - 1)

  # parse result.csv
  local fail_case=`grep -E "FAIL|CRASH|EXCEPTION" "${result_file}" | awk -F',' '{print $1}'`
  local arr=($fail_case)
  local fail_count=${#arr[@]}
  local succ_count=$(expr $total_count - $fail_count)

  echo "SUCCESS: $succ_count" > "${RESULT_SUMMARY}"
  echo "FAIL: $fail_count" >> "${RESULT_SUMMARY}"
  echo "FAIL_CASE: ${arr[@]}" >> "${RESULT_SUMMARY}"

  if [[ ${#arr[@]} -gt 0 ]]; then
    echo "[ERROR] Some TestCase(s) failed (${#arr[@]}): ${arr[@]}"
    exit $STATUS_FAILED
  fi

  echo "[INFO] ALL TestCases Pass ($succ_count) !"
}

main() {
  local start=$(date +%s)
  if [ ! -f "${CUR_DIR}/tbe_toolkits/run.sh" ]; then
    echo "[ERROR] tbe_toolkits is missing. please check."
    exit $STATUS_FAILED
  fi

  run
  local end=$(date +%s)
  echo "[INFO] Cost: `expr $end - $start`"

  tar_results
  get_results

  exit $STATUS_SUCCESS
}

main $@
