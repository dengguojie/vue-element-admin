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


set_env() {
  local atc_path="$1"
  export ATC_HOME="${atc_path}"
  export OP_TEST_FRAME_INSTALL_HOME="${CANN_ROOT}/tools/python"
  export OPS_SOURCE_PATH="${CANN_ROOT}/ops/built-in/tbe"
  export PYTHONPATH=$PYTHONPATH:$OPS_SOURCE_PATH:$OP_TEST_FRAME_INSTALL_HOME
  export LD_LIBRARY_PATH=$ATC_HOME/lib64:${CANN_ROOT}/lib:$LD_LIBRARY_PATH
  export PATH=$PATH:$ATC_HOME/ccec_compiler/bin
}


main() {
  local task="$1"
  local atc_path="$2"
  set_env "${atc_path}"
  if [[ "$task" == "ut" || "$task" == "all" ]]; then
    local supported_soc="Ascend310 Ascend910"
    for version in $(echo ${supported_soc}); do
      python3.7 run_ut.py --soc_version="${version}"
    done
  fi;
  if [[ "$task" == "st" || "$task" == "all" ]]; then
    echo "run st"
  fi;
}


main $@
