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

CUR_PATH=$(dirname $0)

source ${CUR_PATH}/config.ini
source ${CUR_PATH}/util/modules/generate_related_ops.sh
source ${CUR_PATH}/util/modules/generate_related_sch.sh

OPS_TESTCASE_DIR="ops_testcase"
TEST_BIN_PATH="${BUILD_PATH}/${OPS_TESTCASE_DIR}"
ST_INSTALL_PATH="${TEST_BIN_PATH}/st"
UT_INSTALL_PATH="${TEST_BIN_PATH}/ut"
ST_CHANGE_LOG="${ST_INSTALL_PATH}/get_change.log"
UT_CHANGE_LOG="${UT_INSTALL_PATH}/get_change.log"

CANN_OUTPUT="${CANN_ROOT}/output"
TEST_TARGET="${CANN_OUTPUT}/${OPS_TESTCASE_DIR}.tar"

test ! -d "${CANN_OUTPUT}" && mkdir -p "${CANN_OUTPUT}"

CHANGE_LOG=""
TEST_INSTALL_PATH=""
OPS_SOURCE_DIR=""

install_stest() {
  local task_type="$1"
  local pr_file="$2"
  get_related_ops "${task_type}" "${pr_file}"
  all_cases=""
  install_related_ops "${pr_file}"
  case_not_found=""
  for op_case in $(echo "${all_cases}" | tr ',' ' '); do
    echo "[INFO] install testcase: ${op_case}"
    if [[ ! -d "${op_case}" ]]; then
      echo "[ERROR] cannot find testcase ${op_case}"
      case_not_found="${case_not_found} ${op_case##*/}"
      continue
    else
      cp -rf "${op_case}" "${TEST_INSTALL_PATH}"
    fi
  done

  if [ ! -z "${case_not_found}" ];then
    echo "[ERROR] st case not found:[${case_not_found}]"
    exit $STATUS_FAILED
  fi
}

install_sch_stest() {
  local task_type="$1"
  local pr_file="$2"
  get_related_sch "${task_type}" "${pr_file}"
  all_cases=""
  install_related_sch "${pr_file}"
  for op_case in $(echo "${all_cases}" | tr ',' ' '); do
    echo "[INFO] install testcase: ${op_case}"
    if [[ ! -d "${op_case}" ]]; then
      echo "[ERROR] cannot find testcase ${op_case}"
    else
      cp -rf "${op_case}" "${TEST_INSTALL_PATH}"
    fi
  done
}

install_all_stest() {
  cp -rf "${OPS_ST_SOURCE_DIR}/" "${TEST_INSTALL_PATH}"
  rm -rf "${TEST_INSTALL_PATH}/aicpu*"
}

install_sch_all_stest(){
  cp -rf "$SCH_ST_SOURCE_DIR" "${TEST_INSTALL_PATH}"
}

install_script() {
  echo "[INFO] install run_ops_test.sh"
  cp -f "${CUR_PATH}/util/run_ops_st.sh" "${TEST_BIN_PATH}"
}

install_sch_script() {
  echo "[INFO] install run_ops_test.sh"
  cp -f "${CUR_PATH}/util/run_sch_st.sh" "${TEST_BIN_PATH}/run_ops_st.sh"
  cp -f "${CANN_ROOT}/auto_schedule/python/tests/sch_run_st.py" "${TEST_BIN_PATH}"
  cp -r "${CANN_ROOT}/auto_schedule/python/tests/sch_test_frame" "${TEST_BIN_PATH}"
}

install_package() {
  echo "[INFO] install ${TEST_TARGET}"
  cd "${BUILD_PATH}" && tar cvf "${TEST_TARGET}" "${OPS_TESTCASE_DIR}"
}

main() {
  local task_type="$1"
  local pr_file="$2"
  if [[ ! -f "${pr_file}" ]]; then
    if [[ "${task_type}" == "st" ]]; then
      echo "[Info] pr_file contains nothing,install all st case"
      install_all_stest
      install_script
    else
      echo "[ERROR] A input file that contains files changed is required"
      exit $STATUS_SUCCESS
    fi
  elif [[ "${pr_file}" == "auto_schedule" ]];then
      install_sch_all_stest
      install_sch_script
  else
    ops_str=`cat ${pr_file} | awk -F\/ '{print $1}' | grep -v "auto_schedule"`
    if [[ -n "${ops_str}" ]]; then
      install_stest "${task_type}" "${pr_file}"
      install_script
    else
      install_sch_stest "${task_type}" "${pr_file}"
      install_sch_script
    fi
  fi
  install_package
}

task_type="$1"
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

test ! -d "${TEST_BIN_PATH}" && mkdir -p "${TEST_BIN_PATH}"

main $@
