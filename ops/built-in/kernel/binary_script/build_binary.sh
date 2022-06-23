#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
# bash build_binary.sh {opp_run_path} {output_path} {short_soc_version}
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=1

OPP_RUN_PATH=$1
OUTPUT_PATH=$2
SHORT_SOC_VERSION=$3
CCACHE_ARGS=$4
_CURR_PATH=$(cd $( dirname "$BASE_SOURCE[0]" ) && pwd)
ROOT_DEFAULT_PATH=/usr/local/Ascend/latest
USER_DEFAULT_PATH=${HOME}/Ascend/latest
#use gen_opcinfo_for_socversion to gen csv
short_soc_version=$(echo $SHORT_SOC_VERSION | tr 'a' 'A')
csv_file="opc_info_${short_soc_version}.csv"

if [ $# -lt 3 ]; then
    echo "ERROR: Input Args Nums Not Equal Three, Please Check The Num."
    exit 1
fi

# source env
if [ ! -f ${ROOT_DEFAULT_PATH}/compiler/bin/setenv.bash ] && [ ! -f ${USER_DEFAULT_PATH}/compiler/bin/setenv.bash ]; then
    echo "ERROR: Compiler Setenv File Not Exists, Please Check."
    exit 1
fi

if [ ! -f ${ROOT_DEFAULT_PATH}/runtime/bin/setenv.bash ] && [ ! -f ${USER_DEFAULT_PATH}/runtime/bin/setenv.bash ]; then
    echo "ERROR: Runtime Setenv File Not Exists, Please Check."
    exit 1
fi

if [ $(id -u) -ne 0 ];then
    python_path=`find ${HOME} -name "tbe" | grep python | grep site-packages`
    sitepackage=${python_path%/*}
    tbe_path="$_CURR_PATH""/../../tbe"
    export PYTHONPATH=$PYTHONPATH:${sitepackage}:${tbe_path}
    for type in "compiler" "runtime"; do
        source ${USER_DEFAULT_PATH}/${type}/bin/setenv.bash
        if [ "$?" != 0 ]; then
            echo "ERROR: Source ${type} Env Falied."
            exit 1
        else
            echo "SUCCESS: Source ${type} Env Successfully."
        fi
    done
else
    for type in "compiler" "runtime"; do
        source ${ROOT_DEFAULT_PATH}/${type}/bin/setenv.bash
        if [ "$?" != 0 ]; then
            echo "ERROR: Source ${type} Env Falied."
            exit 1
        else
            echo "SUCCESS: Source ${type} Env Successfully."
        fi
    done

fi

# gen ccec
gen_ccec(){
  CCEC_PATH=`which ccec`
  $(> ccec)
  echo "#!/bin/bash" >> ccec
  echo "ccache_args=""\"""${CCACHE_ARGS} ${CCEC_PATH}""\"" >> ccec
  echo "args=""$""@" >> ccec
  echo "eval ""\"""$""{ccache_args} ""$""args""\"" >> ccec
  chmod +x ccec
}

gen_ccec
export PATH=${_CURR_PATH}:$PATH

echo "**************Start to Generate Opc Info*****************"
binary_csv_file="$_CURR_PATH/../binary_config/binary_config.csv"
bash gen_opcinfo_for_socversion.sh ${short_soc_version} ${csv_file}
if [ "$?" != 0 ]; then
    echo "ERROR: Gen opc info failed, Please Check!"
    exit 1
else
    echo "SUCCESS: Generate Opc Info Successfully"
fi
echo "**************End to Generate Opc Info*****************"

echo "**************Start to Generate Single Op*****************"
# 并行编译适配
# 设置并行编译数, 当前默认16个进程进行算子编译, 即同时16个算子进行二进制编译
PRONUM=16

# gen compile task
source build_env.sh
task_path="build_binary_all"
[ -d "./${task_path}/" ] || mkdir -p ./${task_path}/
rm -f ./${task_path}/*

op_list=`cat ${binary_csv_file} | tail -n +3 | awk -F, '{print $1}'`
for op_type in ${op_list}; do
{
    bash build_binary_single_op_gen_task.sh ${op_type} ${short_soc_version} ${OPP_RUN_PATH}/opp ${OUTPUT_PATH} ${task_path} ${csv_file}
}
done

# exe task
bash build_binary_single_op_exe_task.sh ${task_path} ${PRONUM}

echo "**************Start to Generate Fusion Ops*****************"
FUSION_OPS_FOLDER_PATH="${_CURR_PATH}/../binary_config/${short_soc_version,,}/FusionOps/"
if [ -d "${FUSION_OPS_FOLDER_PATH}" ]; then
    bash build_binary_fusion_ops.sh ${short_soc_version} ${FUSION_OPS_FOLDER_PATH} ${OUTPUT_PATH}
fi

echo "run build_binary.sh SUCCESS"
