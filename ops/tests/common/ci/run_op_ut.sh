#! /bin/bash
# -------------------------------------------------
# run op python ut
# call this shell need params:
# 1) top_dir, the root of the repo
# 2) out_dir, like: out/onetrack/
# 3) mode:
#           1: run all ut,
#           2: analysis changes then run related ut case, check coverage 80%
#           3: run one op
# 4) op_name
# -------------------------------------------------

TOP_DIR="$1"
OUT_DIR="$2"
RUN_MODE="$3"
OP_NAME="$4"

UT_TMP_REPORT_PATH=${OUT_DIR}/llt/tmp_report/op_impl
UT_REPORT_PATH=${OUT_DIR}/llt/
COVERAGE_REPORT_PATH=${OUT_DIR}/llt/coverage_result

OPS_UT_SCAN_DIR=${TOP_DIR}/llt/ops/llt_new/ut/ops_test

OPS_SOURCE_DIR="${TOP_DIR}/ops"

PYTHON_CMD="python3.7"
COVERAGE_CMD="/opt/buildtools/python3.7/bin/coverage3"
DIFF_COVER_CMD="diff-cover"
UT_RUN_PY=${TOP_DIR}/llt/ops/llt_new/common/ci/run_ut.py
OP_CFG_UT_RUN_PY=${TOP_DIR}/llt/ops/llt_new/common/ci/run_op_cfg_ut.py
MERGE_REPORT_PY=${TOP_DIR}/llt/ops/llt_new/common/ci/merge_op_report.py
CHECK_COVERAGE_PY=${TOP_DIR}/llt/ops/llt_new/common/ci/check_python_coverage.py

# source env
if [ -x $TOP_DIR/llt/ops/llt_new/common/ci/set_python_env.sh ];then
    source $TOP_DIR/llt/ops/llt_new/common/ci/set_python_env.sh $TOP_DIR "ut" "onetrack"
fi;

ut_run_starttime=`date +'%Y-%m-%d %H:%M:%S'`

if [ ${RUN_MODE} = "1" ];then
  # run all ut
  $PYTHON_CMD $UT_RUN_PY --case_dir=${TOP_DIR}/llt/ops/llt_new/ut/ops_test --soc_version=Ascend910 --cov_path=${COVERAGE_REPORT_PATH} --report_path=${UT_REPORT_PATH} --simulator_mode=pv --simulator_lib_path=${OUT_DIR}/llt/ut/obj/lib/simulator
  ut_run_res=$?
elif [ ${RUN_MODE} = "2" ]; then
  $PYTHON_CMD $UT_RUN_PY --auto_analyse --soc_version=Ascend910 --cov_path=${COVERAGE_REPORT_PATH} --report_path=${UT_REPORT_PATH} --simulator_mode=pv --simulator_lib_path=${OUT_DIR}/llt/ut/obj/lib/simulator
  ut_run_res=$?
else
  # run one op's all ut
  $PYTHON_CMD $UT_RUN_PY --case_dir=${TOP_DIR}/llt/ops/llt_new/ut/ops_test/$OP_NAME --soc_version=Ascend910 --cov_path=${COVERAGE_REPORT_PATH} --report_path=${UT_REPORT_PATH}  --simulator_mode=pv --simulator_lib_path=${OUT_DIR}/llt/ut/obj/lib/simulator
  ut_run_res=$?
fi

echo "classify_rule: ops_python" > "${OUT_DIR}/llt/classify_rule_info.log"

# check coverage report
if [ ${RUN_MODE} = "1" ]; then
  echo "not need check full coverage percent"
elif [ ${RUN_MODE} = "2" ]; then
  echo "check coverage is coding"
#  cp ${OUT_DIR}/llt/coverage_result/.coverage ./.coverage
#  COVERAGE_XML_PATH=${OUT_DIR}/llt/coverage.xml
#  $COVERAGE_CMD xml -o $COVERAGE_XML_PATH
#  MANIFEST_PTH=.repo/manifests
#  MANIFEST_BRANCH=$(echo $(cd  "$TOP_DIR"/"${MANIFEST_PTH}" && git config branch.default.merge)|cut -d "/" -f 3)
#  GIT_BRANCH=m/"${MANIFEST_BRANCH}"
#  echo "GIT_BRANCH is ${GIT_BRANCH}"
#  cd "$OUT_DIR"
#  cd $OPS_SOURCE_DIR && $DIFF_COVER_CMD "$COVERAGE_XML_PATH" --compare-branch="${GIT_BRANCH}"  --html-report "$COVERAGE_REPORT_PATH"/report.html
#  diff_cover_res=$?
#  if [ "$diff_cover_res" != 0 ]; then
#    echo "ERROR: exec increase coverage [diff-cover fail]"
#    exit 1
#  fi
#  $PYTHON_CMD $CHECK_COVERAGE_PY "$COVERAGE_REPORT_PATH" "80" "false"
#  if [ $? -ne 0 ]; then
#    echo "check coverage percent failed"
#    exit 1
#  fi
else
  echo "not need check one op coverage percent"
fi

$PYTHON_CMD $OP_CFG_UT_RUN_PY
if [ $? -ne 0 ]; then
  echo "run test op config failed"
  exit 1
fi

if [ $ut_run_res != 0 ]
then
    echo "op ut run failed"
    exit 1
fi
echo "over"
