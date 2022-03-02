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
binary_json.py
"""
import json
import os
import ast
import configparser
from binary_op import BinaryBase


# 'pylint: disable=too-many-locals,too-many-arguments
class BinaryCfg:
    """
    BinaryCfg
    """
    CONFIG_PATH = "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/config/"
    BINARY_CONFIG_PATH = "../binary_config/"

    def __init__(self, op_info, tensor_num, tensor_list, attrs, nd_info, soc_version):
        self.op_info = op_info
        self.tensor_num = tensor_num
        self.tensor_list = tensor_list
        self.attrs = attrs
        self.nd_info = nd_info
        self.soc_version = soc_version

    def fuzz_mode(self):
        """
        generate format_mod_type, dtype_mod_type and op_info_case
        """
        format_type = self.op_info.get("format_type")
        dtype_type = self.op_info.get("dtype_type")
        # 0: default, 1: agnostic, 2:fixed
        format_mod_type = [format_type] * self.tensor_num
        # 0: default, 1: byte
        dtype_mod_type = [dtype_type] * self.tensor_num
        #op_info_case
        op_info_case = None
        if self.tensor_list is not None:
            tensor_size = len(self.tensor_list)
            if  tensor_size == 1:
                format_before = self.tensor_list[0].get("format")
                dtype_before = self.tensor_list[0].get("dtype")
                format_size = len(format_before)
                dtype_size = len(dtype_before)
                support_format = ",".join(format_before * dtype_size)
                support_dtype = ",".join([val for val in dtype_before for _ in range(format_size)])
                op_info_case = [[support_dtype] * self.tensor_num, [support_format] * self.tensor_num]
            elif tensor_size == self.tensor_num:
                support_dtype = []
                support_format = []
                for tensor in self.tensor_list:
                    format_before = tensor.get("format")
                    dtype_before = tensor.get("dtype")
                    format_size = len(format_before)
                    dtype_size = len(dtype_before)
                    format_after = ",".join(format_before * dtype_size)
                    support_format.append(format_after)
                    dtype_after = ",".join([val for val in dtype_before for _ in range(format_size)])
                    support_dtype.append(dtype_after)
                op_info_case = [support_dtype, support_format]
            else:
                raise RuntimeError(
                    "[ERROR] tensor_num is {}, tensor_list size is {}, not equal".format(self.tensor_num, tensor_size))
        return [format_mod_type, dtype_mod_type, self.attrs, self.nd_info, op_info_case]

    def fuzz_opinfo_cfg(self):
        """
        generate binary json
        """
        mod_lst = self.fuzz_mode()
        op_type = self.op_info.get("op_type")
        op_name = self.op_info.get("op_name")
        for soc in self.soc_version:
            op_ob = BinaryBase(op_type)
            op_info_cfg = BinaryCfg.CONFIG_PATH + soc + "/aic-" + soc + "-ops-info.json"
            op_info_binary_cfg_path = BinaryCfg.BINARY_CONFIG_PATH + soc + "/" + op_type + "/"

            try:
                is_gen = op_ob.gen_binary_json(op_info_cfg, mod_lst[0], mod_lst[1], mod_lst[2], mod_lst[3], mod_lst[4])
                if is_gen:
                    os.makedirs(op_info_binary_cfg_path)
                    op_info_binary_cfg = op_info_binary_cfg_path + op_name + ".json"
                    op_ob.dump_binary_json_to_file(op_info_binary_cfg)
                else:
                    raise RuntimeError("[ERROR]{} dump binary json fail".format(op_type))

            except FileExistsError:
                print("[warning]{} {} binary config already exists".format(op_type, soc))

def binary_cfg(op_type, soc_version):
    """
    read binary_json_cfg
    """
    format_mode = {"DEFAULT": BinaryBase.FORMAT_MODE_DEFAULT, "AGNOSTIC": BinaryBase.FORMAT_MODE_AGNOSTIC,
                   "FIXED": BinaryBase.FORMAT_MODE_FIXED}
    dtype_mode = {"DEFAULT": BinaryBase.DTYPE_MODE_DEFAULT, "BYTE": BinaryBase.DTYPE_MODE_BYTE}
    #读取ini文件
    cfg = configparser.ConfigParser()
    cfg.read("binary_json_cfg.ini")
    #获取需要生成json的算子
    if op_type == "all":
        op_type = cfg.sections()
    else:
        op_type = op_type.strip(',').split(',')
    #获取需要生成json的平台
    if soc_version == "all":
        soc_version = ["ascend310", "ascend320", "ascend610", "ascend615", "ascend710",
                       "ascend910", "ascend920", "hi3796cv300cs", "hi3796cv300es", "sd3403"]
    else:
        soc_version = soc_version.strip(',').split(',')
    #生成输入算子的json
    for operator in op_type:
        items = dict(cfg.items(operator))
        op_name = items.get("op_name")

        format_type = items.get("format_type")
        format_type = format_mode.get(format_type)

        dtype_type = items.get("dtype_type")
        dtype_type = dtype_mode.get(dtype_type)

        tensor_num = items.get("tensor_num")
        tensor_num = int(tensor_num)

        tensor_list = items.get("tensor_list")
        if tensor_list is not None:
            tensor_list = [ast.literal_eval(tensor_list)]

        attrs = items.get("attrs")
        if attrs is not None:
            attrs = ast.literal_eval(attrs)

        nd_info = items.get("nd_info")

        op_info = {"op_type": operator, "op_name": op_name, "format_type": format_type, "dtype_type": dtype_type}
        fuc = BinaryCfg(op_info, tensor_num, tensor_list, attrs, nd_info, soc_version)
        fuc.fuzz_opinfo_cfg()

def mate_json(op_type, binary_file, input_tensors):
    op_ob = BinaryBase(op_type)
    with open(input_tensors, 'r') as tensours:
        input_tensour = json.load(tensours)
    print(binary_file)
    op_ob.update_tensor(binary_file, input_tensour)
