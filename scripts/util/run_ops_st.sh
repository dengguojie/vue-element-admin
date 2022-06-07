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
export CANN_ROOT=$(cd $(dirname $0); pwd)

STATUS_SUCCESS=0
STATUS_FAILED=1

# the binrary path for tests
CANN_TEST_OUT="$CANN_ROOT/result"
CANN_ST_OUT="$CANN_TEST_OUT/st"

CANN_ST_SOURCE="${CANN_ROOT}/st"

test ! -d "${CANN_ST_OUT}" && mkdir -p "${CANN_ST_OUT}"

set_st_env() {
  local install_path="$1"
  local short_soc_version="$2"
  # atc
  export PATH=$install_path/compiler/ccec_compiler/bin:$install_path/compiler/bin:$PATH
  export ASCEND_OPP_PATH=$install_path/opp
  export PYTHONPATH=$install_path/compiler/python/site-packages:$install_path/toolkit/python/site-packages:$PYTHONPATH:${ASCEND_OPP_PATH}/op_impl/built-in/ai_core/tbe
  export LD_LIBRARY_PATH=$install_path/runtime/lib64:$install_path/compiler/lib64:$LD_LIBRARY_PATH
  # acl
  if [[ $short_soc_version == "Ascend310" ]]; then
    export DDK_PATH=$install_path
    export NPU_HOST_LIB=$install_path/runtime/lib64/stub
    export LD_LIBRARY_PATH=$install_path/runtime/lib64:$install_path/add-ons:$LD_LIBRARY_PATH
  fi
  #slog
  export ASCEND_SLOG_RPINT_TO_STDOUT=1
  alias atc="atc log=info"
}

modify_for_cov() {
  manager_files=`find /usr/local/ -name fusion_manager.py|grep te_fusion`
  for manager_file in $manager_files
  do
    if [[ `grep "st_cover.start()" $manager_file|wc -l` -eq 0 ]]; then
      echo "fusion_manager has changed for st_cover"
      cp -f $manager_file $manager_file".bak"
      sed -i 's#compile_info = call_op()#import coverage,sys,time;cov_file = os.path.join(os.getenv("CANN_ROOT") or "", "cov_result", ".coverage.%s.%s" % (op_func_name, time.time()));st_cover = coverage.Coverage(source=[op_module, "impl"], data_file=cov_file);st_cover.start();compile_info = call_op();st_cover.stop;st_cover.save();#g' $manager_file
      sed -i 's#return op_func(\*inputs, \*outputs, \*attrs)#import coverage,sys,time;cov_file = os.path.join(os.getenv("CANN_ROOT") or "", "cov_result", ".coverage.%s.%s" % (op_func_name, time.time()));st_cover = coverage.Coverage(source=[op_module, "impl"], data_file=cov_file);st_cover.start();res = op_func(*inputs, *outputs, *attrs);st_cover.stop;st_cover.save();return res;#g' $manager_file
      sed -i 's#generalize_res = generalize_func(\*inputs, \*outputs, \*attrs, impl_mode, generalize_config)#import coverage,sys,time;cov_file = os.path.join(os.getenv("CANN_ROOT") or "", "cov_result", ".coverage.%s.fuzzy.%s" % (op_type, time.time()));st_cover = coverage.Coverage(source=["impl"], data_file=cov_file);st_cover.start();generalize_res = generalize_func(*inputs, *outputs, *attrs, impl_mode, generalize_config);st_cover.stop;st_cover.save();#g' $manager_file
    fi
  done
}

clear_tmp() {
  unset CANN_ROOT
  manager_files=`find /usr/local/ -name fusion_manager.py|grep te_fusion`
  for manager_file in $manager_files
  do
    if [[ `grep "st_cover.start()" $manager_file|wc -l` -gt 0 ]]; then
      mv -f $manager_file".bak" $manager_file
    fi
  done
}

run_st() {
  local op_type="$1"
  local msopst="$DDK_PATH/toolkit/python/site-packages/bin/msopst"
  local supported_soc="$2"
  echo "[INFO]===============now run st on ${supported_soc}==================="
  \which msopst >/dev/null 2>&1
  if [[ $? -eq 0 ]]; then
    msopst="$(which msopst)"
  fi
  if [[ -d "$CANN_ST_OUT" ]]; then
    rm -rf "$CANN_ST_OUT" >/dev/null 2>&1
  fi
  mkdir -p "$CANN_ST_OUT"

  if [[ "${op_type}" == "all" ]]; then
    echo "[INFO] Run all testcases"
    op_dir="${CANN_ST_SOURCE}"
  elif [[ -d "${CANN_ST_SOURCE}/${op_type}" ]]; then
    echo "[INFO] Only run testcases for ${op_type}"
    op_dir="${CANN_ST_SOURCE}/${op_type}"
  else
    echo "[ERROR] testcase is missing under ${CANN_ST_SOURCE}/${op_type}"
    exit $STATUS_FAILED
  fi

  json_cases=$(find "${op_dir}" -name "*.json" 2>/dev/null)
  for op_case in $(echo $json_cases); do
    echo "[INFO] run case file: $op_case"
    custom_ini=${op_case/.json/.ini}
    if [[ ! -e ${custom_ini} ]];then
      timeout 10m python3.7 "$msopst" run -i "$op_case" -soc "$supported_soc" -out "${CANN_ST_OUT}_${op_case}"
    else
      echo "[INFO] use custom_ini:${custom_ini}"
      timeout 10m python3.7 "$msopst" run -i "$op_case" -soc "$supported_soc" -out "${CANN_ST_OUT}_${op_case}" -conf "${custom_ini}"
    fi
    if [[ $? -ne 0 ]]; then
      echo "[ERROR] run ops stest failed, case file is: $op_case."
      st_failed="true"
    fi
  done

  custom_cases=$(find "${op_dir}" -name "*custom.py" 2>/dev/null)

  if [[ ! -d "cov_result" ]]; then
    mkdir cov_result
  fi

  for custom_case in $(echo $custom_cases); do
    echo "[INFO] run case file: $custom_case"
    coverage run $custom_case
    if [[ $? -ne 0 ]]; then
      echo "[ERROR] run ops custom case failed, case file is: $custom_case."
      st_failed="true"
    fi
    file_name=`basename $custom_case`
    flag=${file_name%.*}
    mv .coverage cov_result/.coverage.$flag
  done
}

delete_unmatch_cases() {
  supported_soc="$1"
  if [[ $supported_soc == "Ascend310" ]]; then
      find "${CANN_ST_SOURCE}" -iname "*custom.py" | xargs rm -rf
      find "${CANN_ST_SOURCE}" \( -iname "*.json" -a ! -iname "*310*.py" \) | xargs rm -rf
      find "${CANN_ST_SOURCE}" -iname "*310[b-z]*.json" | xargs rm -rf
  elif [[ $supported_soc == "Ascend910" ]]; then
      find "${CANN_ST_SOURCE}" \( -iname "*.json" -a ! -iname "*910*.py" \) | xargs rm -rf
      find "${CANN_ST_SOURCE}" -iname "*910[b-z]*.json" | xargs rm -rf
  elif [[ $supported_soc == "Ascend310P3" ]]; then
      find "${CANN_ST_SOURCE}" -iname "*custom.py" | xargs rm -rf
      find "${CANN_ST_SOURCE}" \( -iname "*.json" -a ! -iname "*310p3*.json" \) | xargs rm -rf
  fi
}

get_results() {
  st_reports=$(find "${CANN_TEST_OUT}" -name "st_report.json" 2>/dev/null)
  for report in $(echo $st_reports); do
      fail_count=`grep "failed count" $report| awk '{print $3}'`
      if [[ 10#${fail_count} -gt 0 ]]; then
        echo "[ERROR]Some TestCase(s) failed! Please check log with keyword \"failed\" or \"FAILED\""
        st_failed="true"
      fi
  done

  if [[ `ls cov_result/.coverage*|wc -l` -gt 0 ]]; then
    echo "[INFO] find coverage files,tar them."
    cd cov_result && coverage combine && cd -
    tar -cvzf cov_result.tar.gz cov_result/.coverage*
  fi

  if [[ "$st_failed" = "true" ]];then
    echo "[ERROR]Some TestCase(s) failed! Please check log with keyword \"fail\" or \"ERROR\""
    exit $STATUS_FAILED
  else
    echo "[INFO]ALL TestCases Pass!"
  fi
}

main() {
  local base_path="$1"
  local op_type="$2"
  local short_soc_version="$3"
  st_failed="false"

  if [[ -z "${op_type}" ]]; then
     op_type="all"
  fi

  if [[ -z "${short_soc_version}" ]]; then
     short_soc_version="Ascend310"
  fi
  modify_for_cov
  delete_unmatch_cases $short_soc_version
  set_st_env "${base_path}"  "${short_soc_version}"
  run_st "${op_type}" "${short_soc_version}"
  get_results
  clear_tmp
}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ASCEND_PATH OP_TYPE [SHORT_SOC_VERSION]" && exit $STATUS_FAILED
fi

main $@

exit $STATUS_SUCCESS
