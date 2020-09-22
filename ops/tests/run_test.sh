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


set_env() {
  local base_path="$1"
  export BASE_HOME="${base_path}"
  export OP_TEST_FRAME_INSTALL_HOME="${CANN_ROOT}/tools/python"
  export OPS_SOURCE_PATH="${CANN_ROOT}/ops/built-in/tbe"
  export PYTHONPATH=$PYTHONPATH:$OPS_SOURCE_PATH:$OP_TEST_FRAME_INSTALL_HOME
  export LD_LIBRARY_PATH=$BASE_HOME/lib64:${CANN_ROOT}/lib:$LD_LIBRARY_PATH
  export PATH=$PATH:$BASE_HOME/ccec_compiler/bin
}


main() {
  local task="$1"
  local base_path="$2"
  local pr_file="$3"
  set_env "${base_path}"
  if [[ "$task" == "ut" || "$task" == "all" ]]; then
    local supported_soc="Ascend310,Ascend910"
    python3.7 run_ut.py --soc_version="${supported_soc}" --simulator_lib_path="${BASE_HOME}/simulator" --pr_changed_file="${pr_file}"
    if [[ $? -ne 0 ]]; then
      exit $STATUS_FAILED
    fi
    if [ "x$pr_file" == "x" ]; then
      echo "run all ut case successfully."
    else
      echo "run inc ut case successfully, start generate inc report."
      if [ -f "$CURR_PATH/cov_report/.coverage" ]; then
        cp $CURR_PATH/cov_report/.coverage $CANN_ROOT/.coverage
        cd $CANN_ROOT
        coverage xml -o $CURR_PATH/cov_report/coverage.xml
        diff-cover $CURR_PATH/cov_report/coverage.xml --html-report $CURR_PATH/cov_report/report.html
      else
        echo "coverage data is not exist, do not need generate inc report."
      fi
    fi
  fi;
  if [[ "$task" == "st" || "$task" == "all" ]]; then
    echo "run st"
  fi;
}


main $@

exit $STATUS_SUCCESS
