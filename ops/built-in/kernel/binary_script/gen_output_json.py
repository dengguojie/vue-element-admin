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
gen_output_json.py
"""
import sys
import os
import json
import stat

def main(binary_file, kernel_output, output_json):
    """
    gen output json by binary_file and opc json file
    """
    if not os.path.exists(kernel_output):
        print("[ERROR]the kernel binary_file doesnt exist")
        return False
    with open(binary_file, "r") as file_wr:
        binary_json = json.load(file_wr)
    binary_new_json = dict()
    binary_new_json["binList"] = []
    list_index = []
    for i, item in enumerate(binary_json.get("op_list")):
        opc_json_file_name = item.get("bin_filename")
        opc_json_file_path = kernel_output + opc_json_file_name + ".json"
        json_file_path = opc_json_file_path.split("kernel/")[1]
        if not os.path.exists(opc_json_file_path):
            print("[INFO]the opc_json_file desont exsit")
            list_index.append(i)
            continue
        with open(opc_json_file_path, "r") as file_opc:
            opc_info_json = json.load(file_opc)
        one_binary_case_info = opc_info_json.get("supportInfo")
        bin_json_item = {}
        bin_json_item["jsonFilePath"] = json_file_path
        one_binary_case_info["binInfo"] = bin_json_item
        binary_new_json.get("binList").append(one_binary_case_info)
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(output_json, flags, modes), "w") as f:
        json.dump(binary_new_json, f, indent=2)
    #remove item to failed json when deosnt match by bin_filename
    if len(list_index) != 0:
        list_index.reverse()
        for i in list_index:
            item = binary_json.get("op_list").pop(i)
            bin_filename = item.get("bin_filename")
            failed_json_file_name = bin_filename + "_failed"
            failed_json_file = kernel_output + failed_json_file_name + ".json"
            with os.fdopen(os.open(failed_json_file, flags, modes), "w") as f:
                json.dump(item, f, indent=2)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
