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
CUR_DIR=$(cd $(dirname $0); pwd)

STATUS_SUCCESS=0
STATUS_FAILED=1

CASE_SOURCE="${CUR_DIR}/st_plus"
TOOLKIT_RUN_FILE="${CUR_DIR}/tbe_toolkits/run.sh"
RESULT_SUMMARY="${CUR_DIR}/result_summary.txt"
RESULT_CSV_FILE="${CUR_DIR}/st_plus_result.csv"
RESULT_TAR_FILE="${CUR_DIR}/result_st_plus.tar.gz"
PARAMS="${CUR_DIR}/params"
PROGRESS_OUTPUT="${CUR_DIR}/progress.txt"

run() {
  local op_dir="${CASE_SOURCE}"
  local params=""
  if [[ -f "${PARAMS}" ]]; then
    params=`cat ${PARAMS}`
  fi
  params="${params} --fatbin-parallel=false --progress-output=${PROGRESS_OUTPUT}"

  echo "[INFO]===============now run st_plus ==================="

  local cases=$(find "${op_dir}" -name "*.csv" | head -1 2>/dev/null)
  if [[ -n "$cases" ]]; then
    echo "[INFO] run case file: $cases, with parameters: ${params}"
    "${TOOLKIT_RUN_FILE}" "$cases" "${RESULT_CSV_FILE} ${params}"
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

parse_result_csv() {
  if [[ ! -f "${RESULT_CSV_FILE}" ]]; then
    echo "[ERROR] ${RESULT_CSV_FILE##*/} is not found in ${CUR_DIR}"
    return
  fi

  local lines=$(wc -l "${RESULT_CSV_FILE}" 2>/dev/null| awk '{print $1}')
  if [[ $lines -le 1 ]]; then
    echo "[ERROR] ${RESULT_CSV_FILE##*/} is empty or only titiles in it."
    return
  fi
  local total_count=$(expr $lines - 1)

  # parse result.csv
  local fail_case=`grep -E "FAIL|CRASH|EXCEPTION" "${RESULT_CSV_FILE}" | awk -F',' '{print $1}'`
  local arr=($fail_case)
  local fail_count=${#arr[@]}
  local succ_count=$(expr $total_count - $fail_count)

  echo "SUCCESS: $succ_count" > "${RESULT_SUMMARY}"
  echo "FAIL: $fail_count" >> "${RESULT_SUMMARY}"
  echo "FAIL_CASE: ${arr[@]}" >> "${RESULT_SUMMARY}"
}

tar_results () {
  local files=""

  cd "${CUR_DIR}"
  parse_result_csv
  if [[ -f "${RESULT_SUMMARY}" ]]; then
    files=${files}" ${RESULT_SUMMARY##*/}"
  fi

  if [[ -f "${RESULT_CSV_FILE}" ]]; then
    files=${files}" ${RESULT_CSV_FILE##*/}"
  fi

  local log_count=`ls ${CUR_DIR}/tbetoolkits-*.log 2>/dev/null | wc -l`
  if [[ $log_count -gt 0 ]]; then
    files=${files}" tbetoolkits-*.log"
  fi

  if [[ -d ~/ascend/log/plog ]]; then
    tar czf ${RESULT_TAR_FILE} $files ~/ascend/log/plog
  elif [[ -n "$files" ]]; then
    tar czf ${RESULT_TAR_FILE} $files
  fi
}

get_results() {
  if [[ ! -f "${RESULT_SUMMARY}" ]]; then
    echo "[ERROR] ${RESULT_CSV_FILE##*/} is not found or empty in ${CUR_DIR}"
    exit $STATUS_FAILED
  fi

  # parse result_summary.txt
  local fail_case=`cat ${RESULT_SUMMARY}| grep "^FAIL_CASE:" | awk -F: '{print $2}'`
  local fail_count=`cat ${RESULT_SUMMARY}| grep "^FAIL:" | awk -F: '{print $2}'`
  local succ_count=`cat ${RESULT_SUMMARY}| grep "^SUCCESS:" | awk -F: '{print $2}'`

  if [[ ${fail_count} -gt 0 ]]; then
    echo "[ERROR] Some TestCase(s) failed (${fail_count}): ${fail_case}"
    exit $STATUS_FAILED
  fi

  echo "[INFO] ALL TestCases Pass ($succ_count) !"
}

main() {
  local start=$(date +%s)
  if [ ! -f "${TOOLKIT_RUN_FILE}" ]; then
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
