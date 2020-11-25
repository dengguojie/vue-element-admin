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
CANN_ROOT=$(cd ${CURR_PATH}/../..; pwd)

STATUS_SUCCESS=0
STATUS_FAILED=1

CANN_LLT_OUT="$CANN_ROOT/build/llt"
CANN_UT_OUT="$CANN_LLT_OUT/ut"
CANN_ST_OUT="$CANN_LLT_OUT/st"


set_ut_env() {
  local install_path="$1"
  export BASE_HOME="${install_path}"
  export OP_TEST_FRAME_INSTALL_HOME="${CANN_ROOT}/tools/op_test_frame/python"
  export OPS_SOURCE_PATH="${CANN_ROOT}/ops/built-in/tbe"
  export ASCEND_OPP_PATH=$BASE_HOME/opp
  export PYTHONPATH=$OPS_SOURCE_PATH:$OP_TEST_FRAME_INSTALL_HOME:$PYTHONPATH
  export LD_LIBRARY_PATH=$BASE_HOME/atc/lib64:${CANN_ROOT}/lib:$LD_LIBRARY_PATH
  export PATH=$PATH:$BASE_HOME/atc/ccec_compiler/bin
}

run_ut() {
  local pr_file="$1"
  local supported_soc="Ascend910,Ascend310"
  python3.7 run_ut.py --soc_version="${supported_soc}" \
                      --simulator_lib_path="${BASE_HOME}/toolkit/tools/simulator" \
                      --pr_changed_file="${pr_file}"
  if [[ $? -ne 0 ]]; then
    echo "run ops python utest failed."
    exit $STATUS_FAILED
  fi
  if [ "x$pr_file" == "x" ]; then
    echo "run all ut case successfully."
  else
    echo "run inc ut case successfully, start generate inc report."
    coverage=$(find $CURR_PATH/cov_report -name ".coverage" | head -n1)
    if [ -f "$coverage" ]; then
      cp $coverage $CANN_ROOT/.coverage
      cd $CANN_ROOT
      coverage xml -o $CURR_PATH/cov_report/coverage.xml >/dev/null 2>&1
      diff-cover --compare-branch=origin/development $CURR_PATH/cov_report/coverage.xml --html-report $CURR_PATH/cov_report/report.html
    else
      echo "coverage data is not exist, do not need generate inc report."
    fi
  fi
}

set_st_env() {
  local install_path="$1"
  # atc
  export PATH=$install_path/atc/ccec_compiler/bin:$install_path/atc/bin:$PATH
  export PYTHONPATH=$install_path/atc/python/site-packages:$install_path/toolkit/python/site-packages:$PYTHONPATH
  export LD_LIBRARY_PATH=$install_path/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=$install_path/opp
  # acl
  export DDK_PATH=$install_path
  export NPU_HOST_LIB=$install_path/acllib/lib64/stub
  export LD_LIBRARY_PATH=$install_path/acllib/lib64:$install_path/add-ons:$LD_LIBRARY_PATH
}

run_st() {
  local msopst="$DDK_PATH/toolkit/python/site-packages/bin/msopst"
  local supported_soc="Ascend310"
  \which msopst >/dev/null 2>&1
  if [[ $? -eq 0 ]]; then
    msopst="$(which msopst)"
  fi
  if [[ -d "$CANN_ST_OUT" ]]; then
    rm -rf "$CANN_ST_OUT" >/dev/null 2>&1
  fi
  mkdir -p "$CANN_ST_OUT"
  for file in $(find st -name *.json); do
    python3.7 "$msopst" run -i "$file" -soc "$supported_soc" -out "$CANN_ST_OUT"
    if [[ $? -ne 0 ]]; then
      echo "run ops python stest failed."
      exit $STATUS_FAILED
    fi
  done
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
