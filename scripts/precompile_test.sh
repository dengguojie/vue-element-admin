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
SCH_TESTCASE_DIR="ops_testcase_schedule"
TEST_BIN_PATH="${BUILD_PATH}/${OPS_TESTCASE_DIR}"
ST_INSTALL_PATH="${TEST_BIN_PATH}/st"
UT_INSTALL_PATH="${TEST_BIN_PATH}/ut"
ST_PLUS_INSTALL_PATH="${TEST_BIN_PATH}/st_plus"
ST_CHANGE_LOG="${ST_INSTALL_PATH}/get_change.log"
UT_CHANGE_LOG="${UT_INSTALL_PATH}/get_change.log"
ST_PLUS_CHANGE_LOG="${ST_PLUS_INSTALL_PATH}/get_change.log"

CANN_OUTPUT="${CANN_ROOT}/output"
TEST_TARGET="${CANN_OUTPUT}/${OPS_TESTCASE_DIR}.tar"

test ! -d "${CANN_OUTPUT}" && mkdir -p "${CANN_OUTPUT}"

CHANGE_LOG=""
TEST_INSTALL_PATH=""
OPS_SOURCE_DIR=""

install_stest() {
  local pr_file="$1"
  get_related_ops "${pr_file}" "${CHANGE_LOG}"
  all_cases=""
  install_related_ops "${CHANGE_LOG}" "${OPS_SOURCE_DIR}" "*.json"
  case_not_found=""
  for op_case in $(echo "${all_cases}" | tr ',' ' '); do
    echo "[INFO] install testcase: ${op_case}"
    if [[ ! -d "${op_case}" ]]; then
      echo "[ERROR] cannot find testcase ${op_case}"
      case_not_found="${case_not_found} ${op_case##*/}"
      continue
    else
      mkdir -p "${TEST_INSTALL_PATH}"
      cp -rf "${op_case}" "${TEST_INSTALL_PATH}"
    fi
  done

  for line in `cat ${pr_file}`
  do
      line_realpath=${CANN_ROOT}/${line}
      line_dir=${line_realpath%/*}
      if [[ ${line_dir%/*} =~ "ops/built-in/tests/st" ]] && [[ ! $all_cases =~ ${line_dir} ]] && [[ ! ${line} =~ "st/aicpu_test" ]]
      then
        echo "[INFO]copy ${line_dir} to st dirctory"
        cp -rf "${line_dir}" "${TEST_INSTALL_PATH}"
      fi
  done

  if [ ! -z "${case_not_found}" ];then
    echo "[ERROR] st case not found:[${case_not_found}]"
    echo "exit $STATUS_FAILED"
  fi
}

install_st_plus_test() {
  local pr_file="$1"
  get_related_ops "${pr_file}" "${CHANGE_LOG}"
  all_cases=""
  install_related_ops "${CHANGE_LOG}" "${OPS_SOURCE_DIR}" "*.csv"
  case_not_found=""
  for op_case in $(echo "${all_cases}" | tr ',' ' '); do
    echo "[INFO] install testcase: ${op_case}"
    if [[ ! -d "${op_case}" ]]; then
      echo "[ERROR] cannot find testcase ${op_case}"
      case_not_found="${case_not_found} ${op_case##*/}"
      continue
    else
      cases=$(find "${op_case}" -name "*.csv")
      if [[ -z $cases ]]; then
        echo "[WARNING] no testcase file ending with .csv in ${op_case}"
        case_not_found="${case_not_found} ${op_case##*/}"
        continue
      fi
      mkdir -p "${TEST_INSTALL_PATH}"
      cp -rf "${op_case}" "${TEST_INSTALL_PATH}"
    fi
  done

  if [ ! -z "${case_not_found}" ];then
    echo "[ERROR] st case not found: [${case_not_found}]"
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
    if [[  -d "${op_case}" ]]; then
      cp -rf "${op_case}" "${TEST_INSTALL_PATH}"
    elif [[ -f "${op_case}" ]]; then
      case_dir=$(basename $(dirname ${op_case}))
      if [[ ! -d "${TEST_INSTALL_PATH}/${case_dir}" ]];then
          mkdir -p ${TEST_INSTALL_PATH}/${case_dir}
      fi
      cp -rf "${op_case}" "${TEST_INSTALL_PATH}/${case_dir}"
    else
      echo "[ERROR] cannot find testcase ${op_case}"
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

install_st_plus_script() {
  echo "[INFO] install tbe_toolkits"
  cp -f "${CUR_PATH}/util/run_ops_st_plus.sh" "${TEST_BIN_PATH}"
  cp -rf "${CANN_ROOT}/tools/tbe_toolkits" "${TEST_BIN_PATH}"
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
  if [[ "${pr_file}" == "auto_schedule" ]];then
      install_sch_all_stest
      install_sch_script
      TEST_TARGET="${CANN_OUTPUT}/${SCH_TESTCASE_DIR}.tar"
  elif [[ ! -f "${pr_file}" ]]; then
    if [[ "${task_type}" == "st" ]]; then
      echo "[Info] pr_file contains nothing,install all st case"
      install_all_stest
      install_script
    else
      echo "[ERROR] A input file that contains files changed is required"
      exit $STATUS_SUCCESS
    fi
  else
    ops_str=`cat ${pr_file} | awk -F\/ '{print $1}' | grep -v "auto_schedule"`
    if [[ -n "${ops_str}" ]]; then
      if [[ "${task_type}" == "st" ]]; then
        install_stest "${pr_file}"
        install_script
      elif [[ "${task_type}" == "st_plus" ]]; then
        install_st_plus_test "${pr_file}"
        install_st_plus_script
      fi
    else
      install_sch_stest "${task_type}" "${pr_file}"
      install_sch_script
      TEST_TARGET="${CANN_OUTPUT}/${SCH_TESTCASE_DIR}.tar"
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
elif [[ "${task_type}" == "st_plus" ]]; then
  CHANGE_LOG="${ST_PLUS_CHANGE_LOG}"
  TEST_INSTALL_PATH="${ST_PLUS_INSTALL_PATH}"
  OPS_SOURCE_DIR="${OPS_ST_SOURCE_DIR}"
else
  echo "[ERROR] unsuported task type ${task_type}"
  exit $STATUS_FAILED
fi

test ! -d "${TEST_BIN_PATH}" && mkdir -p "${TEST_BIN_PATH}"

main $@
