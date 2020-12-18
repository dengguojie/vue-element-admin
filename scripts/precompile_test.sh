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
  for op_case in $(echo "${all_cases}" | tr ',' ' '); do
    echo "[INFO] install testcase: ${op_case}"
    if [[ ! -d "${op_case}" ]]; then
      echo "[ERROR] cannot find testcase ${op_case}"
      exit $STATUS_SUCCESS
    fi
    cp -rf "${op_case}" "${TEST_INSTALL_PATH}"
  done
  echo "[INFO] install run_ops_test.sh"
  cp -f "${CUR_PATH}/util/run_ops_st.sh" "${TEST_BIN_PATH}"
  cd "${BUILD_PATH}" && tar cvf "${TEST_TARGET}" "${OPS_TESTCASE_DIR}"
}

main() {
  local task_type="$1"
  local pr_file="$2"
  if [[ ! -f "${pr_file}" ]]; then
    echo "[ERROR] A input file that contains files changed is required"
    exit $STATUS_SUCCESS
  fi
  install_stest "${task_type}" "${pr_file}"
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

main $@
