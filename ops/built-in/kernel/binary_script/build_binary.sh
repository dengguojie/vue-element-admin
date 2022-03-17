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
# bash build_binary.sh {opp_run_path} {output_path} {soc_version}
OPP_RUN_PATH=$1
OUTPUT_PATH=$2
SOC_VERSION=$3
_CURR_PATH=$(cd $( dirname "$BASE_SOURCE[0]" ) && pwd)
ROOT_DEFAULT_PATH=/usr/local/Ascend/latest
USER_DEFAULT_PATH=${HOME}/Ascend/latest
#use gen_opcinfo_for_socversion to gen csv
soc_version=$(echo $SOC_VERSION | tr 'a' 'A')
csv_file="opc_info_${soc_version}.csv"

if [ $# -lt 3 ]; then
    echo "ERROR: Input Args Nums Not Equal Three, Please Check The Num."
    exit 1
fi

# source env
if [ ! -f ${ROOT_DEFAULT_PATH}/compiler/bin/setenv.bash ] && [ ! -f ${USER_DEFAULT_PATH}/compiler/bin/setenv.bash ]; then
    echo "ERROR: Compiler Setenv File Not Exists, Please Check."
    exit 1
fi

if [ $(id -u) -ne 0 ];then
    python_path=`find ${HOME} -name "tbe" | grep python | grep site-packages`
    sitepackage=${python_path%/*}
    tbe_path="$_CURR_PATH""/../../tbe"
    export PYTHONPATH=$PYTHONPATH:${sitepackage}:${tbe_path}
    for type in "compiler"; do
        source ${USER_DEFAULT_PATH}/${type}/bin/setenv.bash
        if [ "$?" != 0 ]; then
            echo "ERROR: Source ${type} Env Falied."
            exit 1
        else
            echo "SUCCESS: Source ${type} Env Successfully."
        fi
    done
else
    for type in "compiler"; do
        source ${ROOT_DEFAULT_PATH}/${type}/bin/setenv.bash
        if [ "$?" != 0 ]; then
            echo "ERROR: Source ${type} Env Falied."
            exit 1
        else
            echo "SUCCESS: Source ${type} Env Successfully."
        fi
    done

fi

echo "**************Start to Generate Opc Info*****************"
binary_csv_file="$_CURR_PATH/../binary_config/binary_config.csv"
bash gen_opcinfo_for_socversion.sh ${soc_version} ${csv_file}
if [ "$?" != 0 ]; then
    echo "ERROR: Gen opc info failed, Please Check!"
    exit 1
else
    echo "SUCCESS: Generate Opc Info Successfully"
fi
echo "**************End to Generate Opc Info*****************"

echo "**************Start to Generate Single Op*****************"
op_list=`cat ${binary_csv_file} | tail -n +3 | awk -F, '{print $1}'`
for op_type in ${op_list}; do
    bash build_binary_single_op.sh ${op_type} ${soc_version} ${OPP_RUN_PATH}/opp ${OUTPUT_PATH}
    if [ "$?" != 0 ]; then
        echo "ERROR: Build binary single op ${op_type} failed."
        exit 1
    else
        echo "SUCCESS: Build binary single op ${op_type} successfully."
    fi
done
