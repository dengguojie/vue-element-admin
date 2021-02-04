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
TOOLS_PATH=$(cd ${CURR_PATH}/..; pwd)

STATUS_SUCCESS=0
STATUS_FAILED=1



set_env() {
  export MSOPGEN_PATH="${TOOLS_PATH}/msopgen/"
  export MSOPST_PATH="${TOOLS_PATH}/op_test_frame/python/"
  export PYTHONPATH=$MSOPGEN_PATH:$MSOPST_PATH:$PYTHONPATH
}

run_ut() {
  python3.7 "${TOOLS_PATH}/test/run_ut.py"
  if [[ $? -ne 0 ]]; then
    echo "run can tools ut failed."
    exit $STATUS_FAILED
  fi
  echo "run all ut case successfully."
}

run_st() {
  python3.7 "${TOOLS_PATH}/test/run_st.py"
  if [[ $? -ne 0 ]]; then
    echo "run can tools st failed."
    exit $STATUS_FAILED
  fi
  echo "run all st case successfully."
}

main() {
  local task="$1"
  set_env
  if [[ "$task" == "ut" ]]; then
      run_ut
  elif [[ "$task" == "st" ]]; then
      run_st
  else
      echo "unknown task type: $task"
      exit $STATUS_FAILED
  fi
}

main $@

exit $STATUS_SUCCESS