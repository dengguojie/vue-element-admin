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
import sys
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

    def __init__(self, op_info, tensor_num, select_format, attrs, nd_info, soc_version):

        self.op_type = op_info.get("op_type")
        self.op_name = op_info.get("op_name")
        self.format_type = op_info.get("format_type")
        self.dtype_type = op_info.get("dtype_type")

        self.tensor_num = tensor_num
        self.select_format = select_format
        self.attrs = None if attrs == {} else attrs
        self.nd_info = nd_info
        self.soc_version = soc_version

    def fuzz_mode(self):
        """
        generate format_mod_type, dtype_mod_type and op_info_case
        """

        # 0: default, 1: agnostic, 2:fixed
        format_mod_type = [self.format_type] * self.tensor_num
        # 0: default, 1: byte
        dtype_mod_type = [self.dtype_type] * self.tensor_num
        return [format_mod_type, dtype_mod_type]

    def get_select_format(self, soc):
        """
        get op_info_case
        """
        if self.select_format is None:
            op_info_case = None
        else:
            os.system(
                "python3 ./binary_get_select_format.py {} {} {} {}".format(
                    soc, self.op_type, self.tensor_num, self.select_format))

            select_json = BinaryCfg.BINARY_CONFIG_PATH + self.op_type + "-" + soc + ".json"
            with open(select_json, "r") as sl:
                dynamic_json = json.load(sl)

            support_format, support_dtype = [], []
            for k in dynamic_json:
                tensor = dynamic_json.get(k)
                support_format.append(tensor.get("format"))
                support_dtype.append(tensor.get("dtype"))
            op_info_case = [support_dtype, support_format]
            os.remove(select_json)
        return op_info_case

    def fuzz_opinfo_cfg(self):
        """
        generate binary json
        """
        mod_lst = self.fuzz_mode()
        for soc in self.soc_version:
            op_ob = BinaryBase(self.op_type)
            op_info_case = self.get_select_format(soc)
            op_info_cfg = BinaryCfg.CONFIG_PATH + soc + "/aic-" + soc + "-ops-info.json"
            op_info_binary_cfg_path = BinaryCfg.BINARY_CONFIG_PATH + soc + "/" + self.op_type + "/"

            try:
                is_gen = op_ob.gen_binary_json(op_info_cfg, mod_lst[0], mod_lst[1],
                                               self.attrs, self.nd_info, op_info_case)
                if is_gen:
                    os.makedirs(op_info_binary_cfg_path)
                    op_info_binary_cfg = op_info_binary_cfg_path + self.op_name + ".json"
                    op_ob.dump_binary_json_to_file(op_info_binary_cfg)
                    print("[INFO] success, {} dump binary json in {}".format(self.op_type, soc))
                else:
                    print("[ERROR]{} dump binary json fail in {}".format(self.op_type, soc))

            except FileExistsError:
                print("[Warning]{} binary config already exists in {}".format(self.op_type, soc))

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
        try:
            items = dict(cfg.items(operator))
        except configparser.NoSectionError:
            sys.exit("[ERROR]{} not in binary json config".format(operator))
        # REQUIRED
        op_name = items.get("op_name")
        format_type = items.get("format_type")
        dtype_type = items.get("dtype_type")
        tensor_num = items.get("tensor_num")
        # OPTION
        select_format = items.get("select_format")
        var_attrs = items.get("var_attrs")
        enumerate_attrs = items.get("enumerate_attrs")
        nd_info = items.get("nd_info")

        if None in (op_name, format_type, dtype_type, tensor_num):
            sys.exit("[ERROR]Please check the {} binary json config".format(operator))
        format_type = format_mode.get(format_type)
        dtype_type = dtype_mode.get(dtype_type)
        tensor_num = int(tensor_num)

        attrs = {}
        if var_attrs is not None:
            var_attrs = var_attrs.strip(',').split(',')
            values = [None] * len(var_attrs)
            var_attrs = dict(zip(var_attrs, values))
            attrs.update(var_attrs)
        if enumerate_attrs is not None:
            enumerate_attrs = ast.literal_eval(enumerate_attrs)
            attrs.update(enumerate_attrs)

        op_info = {"op_type": operator, "op_name": op_name, "format_type": format_type, "dtype_type": dtype_type}
        fuc = BinaryCfg(op_info, tensor_num, select_format, attrs, nd_info, soc_version)
        fuc.fuzz_opinfo_cfg()

def mate_json(op_type, binary_file, input_tensors):
    op_ob = BinaryBase(op_type)
    with open(input_tensors, 'r') as tensours:
        input_tensour = json.load(tensours)
    op_ob.update_tensor(binary_file, input_tensour)
