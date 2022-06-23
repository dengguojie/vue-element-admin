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
"""
gen_output_fusion_op_json.py
"""
import sys
import os
import json
from pathlib import Path
from binary_util.util import wr_json


def main(opc_json, output_json):
    """
    gen output_json by opc_json
    """
    # Step 1: Prepare config json object
    opc_json_path = Path(opc_json)
    if not opc_json_path.is_file():
        print("[ERROR] The opc json doesn't exist. ")
        return
    # Get supportInfo from opc json file
    with open(opc_json) as file:
        opc_json_info = json.load(file)
        config_json_info = opc_json_info.get("supportInfo")
    # Add binInfo
    json_file_path = opc_json.split("kernel/")[1]
    bin_json_item = {"jsonFilePath": json_file_path}
    config_json_info["binInfo"] = bin_json_item

    # Step 2: Get output json object
    output_json_path = Path(output_json)
    if output_json_path.is_file():
        with open(output_json) as file:
            output_json_info = json.load(file)
    else:
        output_json_dir, _ = os.path.split(output_json_path)
        if not os.path.exists(output_json_dir):
            os.makedirs(output_json_dir)
        output_json_info = dict()
        output_json_info["binList"] = []

    # Step 3: Add config json object to output json object
    output_json_info.get("binList").append(config_json_info)

    # Step 4: Write info the .json file
    wr_json(output_json_info, output_json_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
