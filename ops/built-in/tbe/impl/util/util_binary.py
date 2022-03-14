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
binary_op.py
"""
import os
import json
from pathlib import Path
from impl.util.platform_adapter import tbe_platform
from impl.util.util_tensor_dict import FormatConstant

_ASCEND_OPP_PATH_ENV = "ASCEND_OPP_PATH"
_ASCEND_OPP_PATH_DEFAULT = "/usr/local/Ascend/opp"
_BUILTIN_KERNEL_CONFIG_PATH = "op_impl/built-in/ai_core/tbe/kernel/config"
_BUILTIN_OPP_CONFIG_PATH = "op_impl/built-in/ai_core/tbe/config"


def get_bit_len(data_type_string):
    """
    get_bit_len
    """
    dtype_dict = {
        'float16': 16,
        'int16': 16,
        'uint16': 16,
        'float': 32,
        'int32': 32,
        'uint32': 32,
        'int64': 64,
        'uint64': 64,
        'int8': 8,
        'uint8': 8,
        'bool': 8,
    }
    return dtype_dict.get(data_type_string, 0)


def convert_to_snake(op_name):
    """
    convert the op name to snake dtype
    the rule if from func GetModeleName in opcompiler/te_fusion/source/fusion_api.cc
    """
    new_name = ""
    sub_head = False
    name_list = list(op_name)
    for _idx, _char in enumerate(name_list):
        if _char.islower():
            sub_head = False
        if _char.isdigit():
            sub_head = True
        if _char.isupper() and _idx != 0:
            if not sub_head:
                new_name += "_"
                sub_head = True
            else:
                _idx_next = _idx + 1
                if _idx_next < len(name_list):
                    if name_list[_idx_next].islower():
                        new_name += "_"
        new_name += _char

    return new_name.lower()


def get_module_name(op_type, soc_version_lower=None, opp_path=None):
    """
    get_module_name
    """
    if soc_version_lower is None:
        soc_version = tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION)
        soc_version_lower = soc_version.lower()
    op_json_file_name = "aic-{0}-ops-info.json".format(soc_version_lower)
    if opp_path is None:
        opp_path = Path(os.environ.get(_ASCEND_OPP_PATH_ENV, _ASCEND_OPP_PATH_DEFAULT))
    opp_config_path = opp_path.joinpath(_BUILTIN_OPP_CONFIG_PATH)
    opp_config_path = opp_config_path.joinpath(soc_version_lower)
    opp_config_path = opp_config_path.joinpath(op_json_file_name)
    module_name = None
    if opp_config_path.is_file():
        opp_config_js = json.loads(opp_config_path.read_text())
        op_type_info = opp_config_js.get(op_type)
        op_file = op_type_info.get("opFile")
        if op_file is None:
            module_name = convert_to_snake(op_type)
        else:
            module_name = op_file.get("value")
    return module_name


# 'pylint: disable=too-many-instance-attributes
class BinaryMatchBase:
    """
    Class: class that BinaryMatchBase
    """
    SPECIAL_FORMAT = FormatConstant.SPECIAL_FORMAT
    # key string in request from tefusion
    GENERALIZATIO_KEY_NAME = "generalize_config"
    GENERALIZATIO_MODE_KEY_NAME = "mode"
    GENERALIZATIO_MODE_COMPILE = "keep_rank"
    GENERALIZATIO_MODE_BINARY = "all_shape"
    # key string in kernel config
    KERNEL_CONFIG_KEY = "binList"
    FORMAT_MODE_KEY = "format_match_mode"
    DTYPE_MODE_KEY = "dtype_match_mode"
    DTYPE_MODE_DEFAULT = "DtypeDefault"
    DTYPE_MODE_BYTE = "DtypeByte"
    FORMAT_MODE_DEFAULT = "FormatDefault"
    FORMAT_MODE_AGNOSTIC = "FormatAgnostic"
    FORMAT_MODE_FIXED = "FormatFixed"

    def __init__(self, op_type, json_name=None):
        self.op_type = op_type
        self.json_name = json_name
        self.special_format = list(BinaryMatchBase.SPECIAL_FORMAT)
        self.input_num = 0
        self.output_num = 0
        self.attr_num = 0
        self.arg_minest_num = 0
        self.binary_rule_list = None
        self.match_time = 0

    def get_binary_rule(self):
        """
        get_binary_rule: get binary rule json file from op_impl/built-in/ai_core/tbe/kernel/config
        """
        if self.binary_rule_list is not None:
            # mean the binary case is init, return directly
            return
        opp_path = Path(os.environ.get(_ASCEND_OPP_PATH_ENV, _ASCEND_OPP_PATH_DEFAULT))
        soc_version = tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION)
        soc_version_lower = soc_version.lower()
        if self.json_name is None:
            # will get file name from op ini file: aic-{soc_version_lower}-ops-info.json
            op_module_name = get_module_name(self.op_type, soc_version_lower=soc_version_lower, opp_path=opp_path)
            if op_module_name is None:
                # do not find op in aic-{soc_version_lower}-ops-info.json, so no binary rule, return directly
                self.binary_rule_list = list()
                return
            self.json_name = "{0}.json".format(op_module_name)

        kernel_config_path = opp_path.joinpath(_BUILTIN_KERNEL_CONFIG_PATH)
        kernel_config_path = kernel_config_path.joinpath(soc_version_lower)
        kernel_config_path = kernel_config_path.joinpath(self.json_name)
        if not kernel_config_path.is_file():
            # do not find the kernel config in op_impl/built-in/ai_core/tbe/kernel/config, return directly
            self.binary_rule_list = list()
            return
        binary_rule_json = json.loads(kernel_config_path.read_text())
        self.binary_rule_list = binary_rule_json.get(BinaryMatchBase.KERNEL_CONFIG_KEY, list())
        if self.binary_rule_list:
            first_rule = self.binary_rule_list[0]
            self.input_num = len(first_rule.get("inputs", []))
            self.output_num = len(first_rule.get("outputs", []))
            self.attr_num = len(first_rule.get("attrs", []))
            self.arg_minest_num = self.input_num + self.output_num + self.attr_num

    def match_result(self, *args, **kwargs):
        """
        match_result: get the operator from kwargs, and do match binary with args
        """
        compile_mode = kwargs.get(BinaryMatchBase.GENERALIZATIO_KEY_NAME,
                                  {BinaryMatchBase.GENERALIZATIO_MODE_KEY_NAME: None})
        if self.binary_rule_list is None or compile_mode.get(BinaryMatchBase.GENERALIZATIO_MODE_KEY_NAME,
                                                             "None") != BinaryMatchBase.GENERALIZATIO_MODE_BINARY:
            # only the match req will do this binary match
            return None

        input_args = len(args)
        if input_args < self.arg_minest_num:
            # the args for compile must be > arg_minest_num
            # arg_minest_num is from ini config, value = input + output + attr
            return None

        for op_rule_item in self.binary_rule_list:
            is_match = True
            # check_tensors
            inputs = op_rule_item.get("inputs", list())
            # check_input_tensors
            for idx, input_tensor in enumerate(inputs):
                if not match_tenser(args[idx], input_tensor):
                    is_match = False
                    break

            if not is_match:
                continue

            # check_attrs
            attrs = op_rule_item.get("attrs", list())
            for idx, attr_info in enumerate(attrs):
                if not match_attr(args[idx + self.input_num + self.output_num], attr_info):
                    is_match = False
                    break

            if not is_match:
                continue
            # match kernel, will return the update info
            match_kernel_res = update_args(args, op_rule_item)
            return [match_kernel_res]

        # do not match the kernel, return None
        return None


def match_format(input_tensor, target_tensor, special_format=BinaryMatchBase.SPECIAL_FORMAT):
    """
    do_match format
    """
    format_type = target_tensor.get(BinaryMatchBase.FORMAT_MODE_KEY, BinaryMatchBase.FORMAT_MODE_DEFAULT)

    def _match(input_format, target_format):
        if target_format is None:
            return True

        is_legal_format = input_format == target_format
        if format_type == BinaryMatchBase.FORMAT_MODE_AGNOSTIC:
            # do not care format, all format use ND kernel
            is_legal_format = True
        elif format_type == BinaryMatchBase.FORMAT_MODE_DEFAULT:
            # when target_format is ND
            # special_format(5HD/6HD) can not use ND kernel
            # NHWC/NCHW/.... can use ND kernel
            if target_format == "ND" and input_format not in special_format:
                is_legal_format = True

        return is_legal_format

    input_real_format = input_tensor.get("format")
    target_real_format = target_tensor.get("format")
    is_match_real_format = _match(input_real_format, target_real_format)
    # ori_format match
    # to do
    return is_match_real_format


def match_dtype(input_tensor, target_tensor):
    """
    do_match dtype
    """
    dtype_type = target_tensor.get(BinaryMatchBase.DTYPE_MODE_KEY, BinaryMatchBase.DTYPE_MODE_DEFAULT)

    def _match(input_dtype, target_dtype):
        if input_dtype is None or target_dtype is None:
            return False

        is_match_dtype = True
        if dtype_type == BinaryMatchBase.DTYPE_MODE_BYTE:
            # the same byte use the same mode
            is_match_dtype = get_bit_len(input_dtype) == get_bit_len(target_dtype)
        else:
            is_match_dtype = input_dtype == target_dtype

        return is_match_dtype

    input_dtype = input_tensor.get("dtype")
    target_dtype = target_tensor.get("dtype")
    is_match_dtype = _match(input_dtype, target_dtype)
    return is_match_dtype


def match_shape(input_tensor, target_tensor):
    """
    do_match shape
    """
    def _match(input_shape, target_shape):
        if target_shape == (-2,):
            return True
        return input_shape == target_shape

    input_real_shape = tuple(input_tensor.get("shape", tuple()))
    target_real_shape = tuple(target_tensor.get("shape", tuple()))
    is_match_shape = _match(input_real_shape, target_real_shape)
    # ori_shape match will be done here
    return is_match_shape


def match_tenser(input_tensor, target_tensor):
    """match_tenser"""
    if not match_shape(input_tensor, target_tensor):
        return False
    if not match_dtype(input_tensor, target_tensor):
        return False
    if not match_format(input_tensor, target_tensor):
        return False

    return True


def match_attr(input_attr, target_attr):
    """match_attr"""
    if target_attr is None:
        return False
    if "value" not in target_attr.keys():
        return False
    target_attr_value = target_attr.get("value")
    if target_attr_value is None:
        return True
    return target_attr_value == input_attr

def update_tenser(input_tensor, target_tensor):
    """
    update_tenser "shape", "ori_shape", "dtype", "format", "ori_format"
    """
    def _update(update_key):
        update_info = target_tensor.get(update_key)
        if update_info is not None:
            input_tensor.update({update_key: update_info})
    for _key in ("shape", "ori_shape", "dtype", "format", "ori_format"):
        _update(_key)

    return input_tensor


def update_attr(target_attr):
    """
    update_attr
    """
    return target_attr.get("value")


def update_args(args, op_rule_item):
    # update input tensors
    args = list(args)
    inputs = op_rule_item.get("inputs", list())
    arg_idx = 0
    for idx, input_tensor in enumerate(inputs):
        args[idx + arg_idx] = update_tenser(args[idx + arg_idx], input_tensor)

    # update output tensors
    arg_idx += len(inputs)
    outputs = op_rule_item.get("outputs", list())
    for idx, output_tensor in enumerate(outputs):
        args[idx + arg_idx] = update_tenser(args[idx + arg_idx], output_tensor)

    # update attrs
    arg_idx += len(outputs)
    attrs = op_rule_item.get("attrs", list())
    for idx, attr_info in enumerate(attrs):
        args[idx + arg_idx] = update_attr(attr_info)
    return args


def binary_match(op_name, json_name=None):
    """binary_match"""
    rule_op = BinaryMatchBase(op_name, json_name)

    def binary_match_inner(*args, **kwargs):
        rule_op.get_binary_rule()
        return rule_op.match_result(*args, **kwargs)

    return binary_match_inner
