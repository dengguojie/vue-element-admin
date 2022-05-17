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
gen_opc_json_with_impl_mode.py
"""
import sys
import os
import json
from binary_util.util import wr_json


def main(ori_json, split_num):
    """
    gen output json by binary_file and opc json file
    """
    split_num = int(split_num)
    if split_num <= 1:
        split_num = 1
    if not os.path.exists(ori_json):
        print("[ERROR]the ori_json doesnt exist")
        return False
    with open(ori_json, "r") as file_wr:
        binary_json = json.load(file_wr)

    op_type = binary_json.get("op_type")
    op_list = binary_json.get("op_list", list())
    op_list_num = len(op_list)
    split_value = (op_list_num + split_num - 1) // split_num
    split_real_num = (op_list_num + split_value - 1) // split_value
    split_value_list = [split_value] * split_real_num + [0] * (split_num - split_real_num)
    split_tail_value = op_list_num - (split_real_num - 1) * split_value
    split_value_list[split_real_num - 1] = split_tail_value
    for idx, _split_value in enumerate(split_value_list):
        new_binary_json = {"op_type": op_type}
        new_binary_json["op_list"] = list()
        new_binary_file = ori_json + "_" + str(idx)
        start_idx = idx * split_value
        if start_idx < op_list_num:
            end_idx = start_idx + _split_value
            new_op_list = op_list[start_idx:end_idx]
            new_binary_json["op_list"] = new_op_list
        wr_json(new_binary_json, new_binary_file)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

