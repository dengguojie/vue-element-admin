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
# bash build_binary_fusion_ops.sh {soc_version} {task_path} {output_path}
_START_PATH=$(cd $( dirname "$BASE_SOURCE[0]" ) && pwd)
ROOT_DEFAULT_PATH=/usr/local/Ascend/latest
USER_DEFAULT_PATH=${HOME}/Ascend/latest

main()
{
    if [ $# -ne 3 ]; then
        echo "[ERROR] input error"
        echo "[ERROR] bash $0 {soc_version} {task_path} {output_path}"
        exit 1
    fi

    SOC_VERSION=$1
    TASK_PATH=$2
    OUTPUT_PATH=$3

    # source env
    export ASCEND_RUN_PATH=${ROOT_DEFAULT_PATH}
    if [ $(id -u) -ne 0 ]; then
        export ASCEND_RUN_PATH=${USER_DEFAULT_PATH}
    fi

    cd ${TASK_PATH}
    echo "last path: ${_START_PATH}, current path: ${TASK_PATH}"

    # Step1 Generate graph for fusion operator
    mkdir build
    cd ./build && cmake .. && make
    echo "[INFO] Built fwk_ir_build finished. "

    ./fwk_ir_build ${SOC_VERSION} && cd ${_START_PATH}
    echo "[INFO] Generating graph finished. "

    # Step2 Generate .json .o using OPC
    graph_file_path="${TASK_PATH}/build/ge_proto_00000_conv2d_net.txt"
    if [ ! -f "${graph_file_path}" ]; then
        echo "${graph_file_path} does not exist. "
        exit 1
    fi
    bin_filename="transdata_conv2d_transdata"
    soc_version_lower=${SOC_VERSION,,}
    binary_compile_full_path="${OUTPUT_PATH}/op_impl/built-in/ai_core/tbe/kernel/${soc_version_lower}/fusion_ops/"
    [ -d ${binary_compile_full_path} ] || mkdir -p ${binary_compile_full_path}
    echo "[INFO] Fusion_ops binary bin full path is ${binary_compile_full_path}"
    opc --graph=${graph_file_path} --soc_version=${SOC_VERSION} --output=${binary_compile_full_path} --bin_filename=${bin_filename}
    echo "[INFO] Generate .json .o using OPC finished. "

    # Step3 Add json info to fusion_ops.json
    binary_config_full_path="${OUTPUT_PATH}/op_impl/built-in/ai_core/tbe/kernel/config/${soc_version_lower}/"
    binary_config_file_name="fusion_ops"
    python3 gen_output_fusion_op_json.py "${binary_compile_full_path}/${bin_filename}.json" "${binary_config_full_path}/${binary_config_file_name}.json"
    echo "[INFO] Add json info to fusion_ops.json finished. "
}
set -o pipefail
main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
