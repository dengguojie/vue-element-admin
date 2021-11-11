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
CURR_PATH=$(cd $(dirname $0); pwd)
CANN_ROOT=$(cd ${CURR_PATH}/..; pwd)

STATUS_SUCCESS=0
STATUS_FAILED=1

# the binrary path for tests
CANN_TEST_OUT="$CANN_ROOT/build/test"
test ! -d "${CANN_TEST_OUT}" && mkdir -p "${CANN_TEST_OUT}"
CANN_UT_OUT="$CANN_TEST_OUT/ut"
CANN_ST_OUT="$CANN_TEST_OUT/st"

OPS_UT_COV_REPORT="${CANN_UT_OUT}/cov_report/ops"
SCH_ST_DIR="${CANN_ROOT}/auto_schedule/python/tests/st"
SCH_UT_DIR="${CANN_ROOT}/auto_schedule/python/tests/ut"

SCH_TEST_FRAME_INSTALL_HOME="${CANN_ROOT}/auto_schedule/python/tests"

set_ut_env() {
  local install_path="$1"
  export BASE_HOME="$install_path"
  export OPS_SOURCE_PATH="${CANN_ROOT}/ops/built-in/tbe"
  export ASCEND_OPP_PATH=$install_path/opp
  export PYTHONPATH=${OPS_SOURCE_PATH}:${SCH_TEST_FRAME_INSTALL_HOME}:${SCH_UT_DIR}:${PYTHONPATH}
  export LD_LIBRARY_PATH=$install_path/atc/lib64:${CANN_ROOT}/lib:$install_path/compiler/lib64:$LD_LIBRARY_PATH
  export PATH=$PATH:$install_path/atc/ccec_compiler/bin:$install_path/compiler/ccec_compiler/bin
}

run_ut() {
  local pr_file="$1"
  local supported_soc="Ascend910A,Ascend310"

  python3 "${CANN_ROOT}/auto_schedule/python/tests/sch_run_ut.py"                     \
                --soc_version="${supported_soc}"                            \
                --simulator_lib_path="${BASE_HOME}/toolkit/tools/simulator" \
                --pr_changed_file="${pr_file}"                              \
                --cov_path="${OPS_UT_COV_REPORT}/python_utest"              \
                --report_path="${OPS_UT_COV_REPORT}/report"
  if [[ $? -ne 0 ]]; then
    echo "run sch python utest failed."
    exit $STATUS_FAILED
  fi

  if [[ "x$pr_file" == "x" ]]; then
    echo "run all ut case successfully."
  else
    echo "run inc ut case successfully, start generate inc report."
    coverage_file=$(find $OPS_UT_COV_REPORT -name ".coverage" | head -n1)
    if [[ -f "$coverage_file" ]]; then
      coverage_dir="$(dirname $coverage_file)"
      cd $coverage_dir && coverage xml -o ${OPS_UT_COV_REPORT}/coverage.xml >/dev/null 2>&1
      diff-cover --compare-branch=origin/master ${OPS_UT_COV_REPORT}/coverage.xml --html-report ${OPS_UT_COV_REPORT}/report.html
      echo "diff-cover generated."
    else
      echo "coverage data is not exist, do not need generate inc report."
    fi
  fi
}

set_st_env() {
  local install_path="$1"
  # atc
  export PATH=$install_path/atc/ccec_compiler/bin:$install_path/atc/bin:$PATH
  export PYTHONPATH=${SCH_ST_DIR}:${SCH_TEST_FRAME_INSTALL_HOME}:${PYTHONPATH}
  export LD_LIBRARY_PATH=$install_path/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=$install_path/opp
  # acl
  export DDK_PATH=$install_path
  export NPU_HOST_LIB=$install_path/acllib/lib64/stub
  export LD_LIBRARY_PATH=$install_path/acllib/lib64:$install_path/add-ons:$LD_LIBRARY_PATH
}

run_st() {
  local sch_run_py="${CANN_ROOT}/auto_schedule/python/tests/sch_run_st.py"
  local supported_soc="Ascend310"

  if [[ -d "$CANN_ST_OUT" ]]; then
    rm -rf "$CANN_ST_OUT" >/dev/null 2>&1
  fi
  mkdir -p "$CANN_ST_OUT"

  python3.7 "$sch_run_py" "${SCH_ST_DIR}" "${supported_soc}" "${CANN_ST_OUT}"
  if [[ $? -ne 0 ]]; then
    echo "run ops python stest failed."
    exit $STATUS_FAILED
  fi
}

main() {
  local task="$1"
  local base_path="$2"
  local pr_file="$3"
  if [[ "$task" == "ut" ]]; then
      set_ut_env "${base_path}"
      run_ut "${pr_file}"
  elif [[ "$task" == "st" ]]; then
      set_st_env "${base_path}"
      run_st
  else
      echo "unknown task type: $task"
      exit $STATUS_FAILED
  fi
}


main $@

exit $STATUS_SUCCESS
