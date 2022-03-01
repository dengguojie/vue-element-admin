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
CANN_ROOT=$(cd $(dirname $0); pwd)

STATUS_SUCCESS=0
STATUS_FAILED=1

# the binrary path for tests
CANN_TEST_OUT="$CANN_ROOT/result"
CANN_ST_OUT="$CANN_TEST_OUT/st"

CANN_ST_SOURCE="${CANN_ROOT}/st"

test ! -d "${CANN_ST_OUT}" && mkdir -p "${CANN_ST_OUT}"

ALL_CASES="cases.txt"
RESULT="result.txt"

set_st_env() {
  local install_path="$1"
  # atc
  export PATH=$install_path/atc/ccec_compiler/bin:$install_path/atc/bin:$PATH
  export PYTHONPATH=$install_path/atc/python/site-packages:${install_path}/opp/op_impl/built-in/ai_core/tbe:${CANN_ST_SOURCE}:${PYTHONPATH}
  export LD_LIBRARY_PATH=$install_path/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=$install_path/opp
  # acl ascend310
  export DDK_PATH=$install_path
  export NPU_HOST_LIB=$install_path/acllib/lib64/stub
  export LD_LIBRARY_PATH=$install_path/acllib/lib64:$install_path/add-ons:$LD_LIBRARY_PATH
  # for ascend 910
  export DDK_PATH=$install_path
  export NPU_HOST_LIB=$install_path/fwkacllib/lib64/stub
  export LD_LIBRARY_PATH=$install_path/fwkacllib/lib64:$install_path/add-ons:$LD_LIBRARY_PATH
}

all_in_one_set_st_evn() {
  local install_path="$1"
  export PATH=$install_path/compiler/ccec_compiler/bin:$install_path/compiler/bin:$PATH
  export PYTHONPATH=$install_path/compiler/python/site-packages:${install_path}/opp/op_impl/built-in/ai_core/tbe:${CANN_ST_SOURCE}:${PYTHONPATH}
  export LD_LIBRARY_PATH=$install_path/compiler/lib64:$install_path/runtime/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=$install_path/opp
}

run_st() {
  local op_type="$1"
  local sch_run_py="${CANN_ROOT}/sch_run_st.py"
  local supported_soc="$2"

  if [[ -d "$CANN_ST_OUT" ]]; then
    rm -rf "$CANN_ST_OUT" >/dev/null 2>&1
  fi
  mkdir -p "$CANN_ST_OUT"

  if [[ "${op_type}" == "all" ]]; then
    echo "[INFO] Run all testcases"
    case_dir="${CANN_ST_SOURCE}"
  elif [[ -d "${CANN_ST_SOURCE}/${op_type}" ]]; then
    echo "[INFO] Only run testcases for ${op_type}"
    case_dir="${CANN_ST_SOURCE}/${op_type}"
  else
    echo "[ERROR] testcase is missing under ${CANN_ST_SOURCE}/${op_type}"
    exit $STATUS_FAILED
  fi

  python3.7 "$sch_run_py" "${case_dir}" "${supported_soc}" "${CANN_ST_OUT}"

  if [[ $? -ne 0 ]]; then
      echo "[ERROR] run sch stest failed, case file is: $op_case."
      #exit $STATUS_FAILED
  else
      echo "ALL TestCases Pass."
  fi

}

gen_all_cases() {
  touch "${ALL_CASES}"
  find "${CANN_ST_SOURCE}" -name "test_*_impl.py" |
    xargs basename -a |
    sed 's/.py//g' > "${ALL_CASES}" 2>/dev/null
  echo "[INFO] find cases to execute:" && cat "${ALL_CASES}"
}

get_results() {
  touch "${RESULT}"
  find "${CANN_TEST_OUT}" -name "result.txt" |
    xargs grep -v "Test Result" |
    awk '{print $1" : "$2}' > "${RESULT}" 2>/dev/null
  echo "[INFO] get results for all:" && cat "${RESULT}"
}

main() {
  local base_path="$1"
  local op_type="$2"
  local soc_version="$3"

  if [[ -z "${op_type}" ]]; then
    op_type="all"
  fi

  if [[ -z "${soc_version}" ]]; then
     soc_version="Ascend310"
  fi

  gen_all_cases
  all_in_one_set_st_evn "${base_path}"
  run_st "${op_type}" "${soc_version}"
  get_results
}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ASCEND_PATH [OP_TYPE]" && exit $STATUS_FAILED
fi

main $@

exit $STATUS_SUCCESS
