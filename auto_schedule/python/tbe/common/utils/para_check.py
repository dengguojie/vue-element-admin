#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
common function for check ops parameter
"""
import math
import re
import warnings
from enum import Enum
from functools import reduce as _reduce
from functools import wraps

from .errormgr import error_manager_util as error_manager

SHAPE_SIZE_LIMIT = 2 ** 31 - 1
SHAPE_SIZE_ZERO = 0
DIM_LIMIT = SHAPE_SIZE_LIMIT
MIN_UNKNOWN_SHAPE_RANK = 0
MAX_UNKNOWN_SHAPE_NUM = 2 ** 31 - 1
DEFAULT_MIN_SHAPE_DIM = 1
DEFAULT_MAX_SHAPE_DIM = 8
DEFAULT_MAX_SHAPE_NUM = 200000000
DYNAMIC_SHAPE_FLAG = -1

RANK_ZERO = 0
RANK_LIMIT = 8
ZERO_DIM = 0
NONE_TYPE = type(None)

# the max len of kernel_name
MAX_KERNEL_NAME_LEN = 200
KERNEL_NAME = "kernel_name"

CONST = "const"
SPECIAL = "special"
ORIGINAL = "original"
SPECIAL_SCALAR = "special_scalar"
COMMON = "common"
BROADCAST = "broadcast"

REQUIRED_INPUT = "required_input"
OPTION_INPUT = "option_input"
DYNAMIC_INPUT = "dynamic_input"

REQUIRED_OUTPUT = "required_output"
OPTION_OUTPUT = "option_output"
DYNAMIC_OUTPUT = "dynamic_output"

# in proto attr can be a Tensor/BYTES/LIST_TYPE Type, but not te fusion don't support this type
REQUIRED_ATTR_INT = "REQUIRED_ATTR_INT"
REQUIRED_ATTR_FLOAT = "REQUIRED_ATTR_FLOAT"
REQUIRED_ATTR_STR = "REQUIRED_ATTR_STR"
REQUIRED_ATTR_BOOL = "REQUIRED_ATTR_BOOL"
REQUIRED_ATTR_TYPE = "REQUIRED_ATTR_TYPE"
REQUIRED_ATTR_LIST_INT = "REQUIRED_ATTR_LIST_INT"
REQUIRED_ATTR_LIST_FLOAT = "REQUIRED_ATTR_LIST_FLOAT"
REQUIRED_ATTR_LIST_BOOL = "REQUIRED_ATTR_LIST_BOOL"
REQUIRED_ATTR_LIST_LIST_INT = "REQUIRED_ATTR_LIST_LIST_INT"

OPTION_ATTR_INT = "OPTION_ATTR_INT"
OPTION_ATTR_FLOAT = "OPTION_ATTR_FLOAT"
OPTION_ATTR_STR = "OPTION_ATTR_STR"
OPTION_ATTR_BOOL = "OPTION_ATTR_BOOL"
OPTION_ATTR_TYPE = "OPTION_ATTR_TYPE"
OPTION_ATTR_LIST_INT = "OPTION_ATTR_LIST_INT"
OPTION_ATTR_LIST_FLOAT = "OPTION_ATTR_LIST_FLOAT"
OPTION_ATTR_LIST_BOOL = "OPTION_ATTR_LIST_BOOL"
OPTION_ATTR_LIST_LIST_INT = "OPTION_ATTR_LIST_LIST_INT"

OP_ERROR_CODE_000 = 'E80000'
OP_ERROR_CODE_001 = 'E80001'
OP_ERROR_CODE_002 = 'E80002'
OP_ERROR_CODE_003 = 'E80003'
OP_ERROR_CODE_004 = 'E80004'
OP_ERROR_CODE_005 = 'E80005'
OP_ERROR_CODE_006 = 'E80006'
OP_ERROR_CODE_007 = 'E80007'
OP_ERROR_CODE_008 = 'E80008'
OP_ERROR_CODE_009 = 'E80009'
OP_ERROR_CODE_010 = 'E80010'
OP_ERROR_CODE_011 = 'E80011'
OP_ERROR_CODE_012 = 'E80012'
OP_ERROR_CODE_013 = 'E80013'
OP_ERROR_CODE_014 = 'E80014'
OP_ERROR_CODE_015 = 'E80015'
OP_ERROR_CODE_016 = 'E80016'
OP_ERROR_CODE_017 = 'E80017'
OP_ERROR_CODE_018 = 'E80018'
OP_ERROR_CODE_019 = 'E80019'
OP_ERROR_CODE_020 = 'E80020'
OP_ERROR_CODE_021 = 'E80021'
OP_ERROR_CODE_022 = 'E80022'
OP_ERROR_CODE_023 = 'E80023'
OP_ERROR_CODE_024 = 'E80024'
OP_ERROR_CODE_025 = 'E80025'
OP_ERROR_CODE_026 = 'E80026'
OP_ERROR_CODE_027 = 'E80027'


# OpParamInfoKey && TensorFormat :Internal Use Only
class OpParamInfoKey(Enum):
    SHAPE = "shape"
    FORMAT = "format"
    ORI_SHAPE = "ori_shape"
    ORI_FORMAT = "ori_format"
    D_TYPE = "dtype"
    RANGE = "range"


class TensorFormat(Enum):
    ND = "ND"
    NCHW = "NCHW"
    NHWC = "NHWC"
    NDHWC = "NDHWC"
    NCDHW = "NCDHW"
    CHWN = "CHWN"
    NC1HWC0 = "NC1HWC0"
    NC1HWC0_C04 = "NC1HWC0_C04"
    NDC1HWC0 = "NDC1HWC0"
    FRACTAL_NZ = "FRACTAL_NZ"
    HWCN = "HWCN"
    DHWCN = "DHWCN"
    FRACTAL_Z = "FRACTAL_Z"
    FRACTAL_Z_C04 = "FRACTAL_Z_C04"
    C1HWNCoC0 = "C1HWNCoC0"
    FRACTAL_Z_3D = "FRACTAL_Z_3D"
    FRACTAL_ZN_LSTM = "FRACTAL_ZN_LSTM"
    FRACTAL_ZN_RNN = "FRACTAL_ZN_RNN"
    ND_RNN_BIAS = "ND_RNN_BIAS"


ALL_FORMAT_LIST = [entry.value for entry in TensorFormat]
ALL_DTYPE_LIST = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "bfloat16",
                  "int64", "uint64", "float16", "float32", "float64", "bool", "uint1","bfloat16")
OP_NAME = ""
PARAM_NAME = ""


def check_op_params(*type_args, **type_kwargs):
    """
    check op params
    """
    from tbe.dsl.base import operation
    input_params = [REQUIRED_INPUT, OPTION_INPUT, DYNAMIC_INPUT]
    output_params = [REQUIRED_OUTPUT, OPTION_OUTPUT, DYNAMIC_OUTPUT]
    required_attr_params = [REQUIRED_ATTR_STR, REQUIRED_ATTR_FLOAT,
                            REQUIRED_ATTR_INT, REQUIRED_ATTR_BOOL,
                            REQUIRED_ATTR_TYPE, REQUIRED_ATTR_LIST_INT,
                            REQUIRED_ATTR_LIST_BOOL, REQUIRED_ATTR_LIST_FLOAT,
                            REQUIRED_ATTR_LIST_LIST_INT]
    list_type_attr = [REQUIRED_ATTR_LIST_BOOL, REQUIRED_ATTR_LIST_INT,
                      REQUIRED_ATTR_LIST_FLOAT, REQUIRED_ATTR_LIST_LIST_INT,
                      OPTION_ATTR_LIST_BOOL, OPTION_ATTR_LIST_INT,
                      OPTION_ATTR_LIST_FLOAT, OPTION_ATTR_LIST_LIST_INT]

    def _check_input_output_key(op_param, param_name, op_name=OP_NAME):
        # check all necessary information
        # (shape, format, ori_shape, ori_format, dtype)
        if not isinstance(op_param, dict):
            error_info = {
                'errCode': OP_ERROR_CODE_003, 'op_name': op_name,
                'param_name': param_name, 'param_type': 'dict',
                'actual_type': op_param.__class__.__name__}
            raise RuntimeError(
                error_info,
                "In op[%s], the parameter[%s]'s type should be [%s], "
                "but actually is [%s]." % (error_info['op_name'],
                                           error_info['param_name'],
                                           error_info['param_type'],
                                           error_info['actual_type']))
        error_info = {}

        if OpParamInfoKey.SHAPE.value not in op_param.keys():
            error_info['key'] = OpParamInfoKey.SHAPE.value
        elif OpParamInfoKey.FORMAT.value not in op_param.keys():
            error_info['key'] = OpParamInfoKey.FORMAT.value
        elif OpParamInfoKey.ORI_SHAPE.value not in op_param.keys():
            error_info['key'] = OpParamInfoKey.ORI_SHAPE.value
        elif OpParamInfoKey.ORI_FORMAT.value not in op_param.keys():
            error_info['key'] = OpParamInfoKey.ORI_FORMAT.value
        elif OpParamInfoKey.D_TYPE.value not in op_param.keys():
            error_info['key'] = OpParamInfoKey.D_TYPE.value
        elif operation.in_dynamic() and OpParamInfoKey.RANGE.value not in op_param.keys() :
            shape = op_param.get(OpParamInfoKey.ORI_SHAPE.value)
            if isinstance(shape, (tuple, list)) and DYNAMIC_SHAPE_FLAG in shape:
                error_info['key'] = OpParamInfoKey.RANGE.value

        if "key" in error_info.keys():
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            raise RuntimeError(
                error_info,
                "In op[%s], the input[%s] does not contain the item[%s]."
                % (error_info['op_name'], error_info['param_name'],
                   error_info['key']))

    def _check_input_output_dict(op_param, param_name, op_name=OP_NAME):
        _check_input_output_key(op_param, param_name, op_name)
        check_shape(op_param[OpParamInfoKey.SHAPE.value],
                    param_name=param_name)
        check_shape(op_param[OpParamInfoKey.ORI_SHAPE.value],
                    param_name=param_name)
        if operation.in_dynamic() and DYNAMIC_SHAPE_FLAG in op_param.get(OpParamInfoKey.ORI_SHAPE.value):
            _check_range(op_param[OpParamInfoKey.SHAPE.value],
                         op_param[OpParamInfoKey.RANGE.value],
                         param_name=param_name)

        if op_param[OpParamInfoKey.FORMAT.value] not in ALL_FORMAT_LIST:
            error_info = {
                'errCode': OP_ERROR_CODE_015, 'op_name': op_name,
                'param_name': param_name,
                'excepted_format_list': ",".join(ALL_FORMAT_LIST),
                'format': op_param[OpParamInfoKey.FORMAT.value]}

            raise RuntimeError(
                error_info,
                "In op[%s], the format of input[%s] should be one of [%s],"
                " but actually is [%s]."
                % (error_info['op_name'], error_info['param_name'],
                   error_info['excepted_format_list'], error_info['format']))

        if op_param[OpParamInfoKey.ORI_FORMAT.value] not in ALL_FORMAT_LIST:
            error_info = {
                'errCode': OP_ERROR_CODE_014, 'op_name': op_name,
                'param_name': param_name,
                'excepted_format_list': ",".join(ALL_FORMAT_LIST),
                'format': op_param[OpParamInfoKey.ORI_FORMAT.value]}
            raise RuntimeError(
                error_info,
                "In op[%s], the ori format of input[%s] should be one of [%s]"
                ", but actually is [%s]."
                % (error_info['op_name'],
                   error_info['param_name'], ",".join(ALL_FORMAT_LIST),
                   error_info['format']))

        if not isinstance(op_param[OpParamInfoKey.D_TYPE.value], str):
            error_info = {
                'errCode': OP_ERROR_CODE_003, 'op_name': op_name,
                'param_name': param_name, 'param_type': 'str',
                'actual_type':
                    op_param[OpParamInfoKey.D_TYPE.value].__class__.__name__}
            raise RuntimeError(
                error_info,
                "In op[%s], the parameter[%s]'s type should be [%s],  "
                "but actually is [%s]."
                % (error_info['op_name'], error_info['param_name'],
                   error_info['param_type'], error_info['actual_type']))

        if op_param[OpParamInfoKey.D_TYPE.value] is None or \
                op_param[OpParamInfoKey.D_TYPE.value].lower() not in \
                ALL_DTYPE_LIST:
            error_info = {
                'errCode': OP_ERROR_CODE_008, 'op_name': op_name,
                'param_name': param_name,
                'excepted_dtype_list': ",".join(ALL_DTYPE_LIST),
                'dtype': op_param[OpParamInfoKey.D_TYPE.value]}
            raise RuntimeError(
                error_info,
                "In op[%s], the parameter[%s]'s dtype should be one of [%s], "
                "but actually is [%s]."
                % (error_info['op_name'], error_info['param_name'],
                   error_info['excepted_dtype_list'], error_info['dtype']))
        if "param_name" not in op_param.keys():
            op_param["param_name"] = param_name

    def _check_input(op_param, param_name, param_type, op_name=OP_NAME):
        if param_type == REQUIRED_INPUT:
            error_info = {
                'errCode': OP_ERROR_CODE_001, 'op_name': op_name,
                'param_name': param_name}
            if op_param is None:
                raise RuntimeError(
                    error_info,
                    "In op[%s], the mandatory parameter[%s] is missed."
                    % (error_info['op_name'], error_info['param_name']))
            _check_input_output_dict(op_param, param_name, op_name)
        elif param_type == OPTION_INPUT:
            if op_param is not None:
                _check_input_output_dict(op_param, param_name, op_name)
        else:
            if not isinstance(op_param, (list, tuple)):
                error_info = {
                    'errCode': OP_ERROR_CODE_003, 'op_name': op_name,
                    'param_name': param_name, 'param_type': "list truple",
                    'actual_type': op_param.__class__.__name__}
                raise RuntimeError(
                    error_info,
                    "In op[%s], the parameter[%s]'s type should be [%s],  "
                    "but actually is [%s]."
                    % (op_name, param_name, error_info['param_type'],
                       error_info['actual_type']))
            if not op_param:
                error_info = {
                    'errCode': OP_ERROR_CODE_001, 'op_name': op_name,
                    'param_name': param_name}
                raise RuntimeError(
                    error_info,
                    "In op[%s], the mandatory parameter[%s] is missed."
                    % (op_name, param_name))
            for one_input in op_param:
                _check_input_output_dict(one_input, param_name, op_name)

    def _check_output(op_param, param_name, param_type, op_name=OP_NAME):
        if param_type == REQUIRED_OUTPUT:
            if op_param is None:
                error_info = {
                    'errCode': OP_ERROR_CODE_001, 'op_name': op_name,
                    'param_name': param_name}
                raise RuntimeError(
                    error_info,
                    "In op[%s], the mandatory parameter[%s] is missed."
                    % (op_name, param_name))

            _check_input_output_dict(op_param, param_name, op_name)
        elif param_type == OPTION_OUTPUT:
            if op_param is not None:
                _check_input_output_dict(op_param, param_name, op_name)
        else:
            if not isinstance(op_param, (list, tuple)):
                error_info = {
                    'errCode': OP_ERROR_CODE_003, 'op_name': op_name,
                    'param_name': param_name, 'param_type': "list tuple",
                    'actual_type': op_param.__class__.__name__}
                raise RuntimeError(
                    error_info,
                    "In op[%s], the parameter[%s]'s type should be [%s],  "
                    "but actually is [%s]."
                    % (op_name, param_name, error_info['param_type'],
                       error_info['actual_type']))
            if not op_param:
                error_info = {
                    'errCode': OP_ERROR_CODE_001, 'op_name': op_name,
                    'param_name': param_name}
                raise RuntimeError(
                    error_info,
                    "In op[%s], the mandatory  parameter[%s] is missed."
                    % (op_name, param_name))
            for one_input in op_param:
                _check_input_output_dict(one_input, param_name, op_name)

    def _check_attr_type(op_param, param_name, py_type, py_type_name,
                         op_name=OP_NAME):
        if not isinstance(op_param, py_type):
            error_info = {
                'errCode': OP_ERROR_CODE_003, 'op_name': op_name,
                'param_name': param_name, 'param_type': str(py_type),
                'actual_type': op_param.__class__.__name__}
            raise RuntimeError(
                error_info,
                "In op[%s], the parameter[%s]'s type should be [%s],"
                " but actually is [%s]."
                % (error_info['op_name'], error_info['param_name'],
                   error_info['param_type'], error_info['actual_type']))
        if py_type_name == "float":
            if math.isinf(op_param) or math.isnan(op_param):
                error_info = {
                    'errCode': OP_ERROR_CODE_000, 'op_name': op_name,
                    'param_name': param_name,
                    'excepted_value': "float range data",
                    'real_value': str(op_param)}
                raise RuntimeError(
                    error_info,
                    "In op[%s], the parameter[%s] should be [%s], "
                    "but actually is [%s]."
                    % (error_info['op_name'], error_info['param_name'],
                       error_info['excepted_value'], error_info['real_value']))

    def _check_list_attr(op_param, param_name, param_type, op_name=OP_NAME):
        if not isinstance(op_param, (list, tuple)):
            error_info = {
                'errCode': OP_ERROR_CODE_003, 'op_name': op_name,
                'param_name': param_name, 'param_type': "list tuple",
                'actual_type': op_param.__class__.__name__}
            raise RuntimeError(
                error_info,
                "In op[%s], the parameter[%s]'s type should be [%s],"
                "  but actually is [%s]."
                % (error_info['op_name'], error_info['param_name'],
                   error_info['param_type'], error_info['actual_type']))

        if param_type in [REQUIRED_ATTR_LIST_BOOL, OPTION_ATTR_LIST_BOOL]:
            for one_attr in op_param:
                _check_attr_type(one_attr, param_name, bool, "bool", op_name)

        if param_type in [REQUIRED_ATTR_LIST_INT, OPTION_ATTR_LIST_INT]:
            for one_attr in op_param:
                _check_attr_type(one_attr, param_name, int, "int", op_name)

        if param_type in [REQUIRED_ATTR_LIST_FLOAT, OPTION_ATTR_LIST_FLOAT]:
            for one_attr in op_param:
                _check_attr_type(one_attr, param_name, float, "float", op_name)

        if param_type in [REQUIRED_ATTR_LIST_LIST_INT,
                          OPTION_ATTR_LIST_LIST_INT]:
            for one_attr in op_param:
                if not isinstance(one_attr, (list, tuple)):
                    error_info = {
                        'errCode': OP_ERROR_CODE_003, 'op_name': op_name,
                        'param_name': param_name, 'param_type': "list tuple",
                        'actual_type': op_param.__class__.__name__}
                    raise RuntimeError(
                        error_info,
                        "In op[%s], the parameter[%s]'s type should be [%s],"
                        " but actually is [%s]."
                        % (error_info['op_name'], error_info['param_name'],
                           error_info['param_type'], error_info['actual_type']))

                for ele in one_attr:
                    _check_attr_type(ele, param_name, int, "int",  op_name)

    def _check_attr(op_param, param_name, param_type, op_name=OP_NAME):
        if op_param is None and param_type in required_attr_params:
            error_info = {'errCode': OP_ERROR_CODE_001, 'op_name': op_name,
                          'param_name': param_name}
            raise RuntimeError(
                error_info,
                "In op[%s], the mandatory parameter[%s] is missed."
                % (op_name, param_name))
        if not op_param:
            return

        if param_type in [REQUIRED_ATTR_INT, OPTION_ATTR_INT]:
            _check_attr_type(op_param, param_name, int, "int", op_name)

        if param_type in [REQUIRED_ATTR_FLOAT, OPTION_ATTR_FLOAT]:
            _check_attr_type(op_param, param_name, float, "float", op_name)

        if param_type in [REQUIRED_ATTR_STR, OPTION_ATTR_STR]:
            _check_attr_type(op_param, param_name, str, "string", op_name)

        if param_type in [REQUIRED_ATTR_BOOL, OPTION_ATTR_BOOL]:
            _check_attr_type(op_param, param_name, bool, "bool", op_name)

        if param_type in [REQUIRED_ATTR_TYPE, OPTION_ATTR_TYPE]:
            if op_param not in ALL_DTYPE_LIST:
                error_info = {
                    'errCode': OP_ERROR_CODE_003, 'op_name': op_name,
                    'param_name': param_name,
                    'param_type': " ".join(ALL_DTYPE_LIST),
                    'actual_type': op_param.__class__.__name__}
                raise RuntimeError(
                    error_info,
                    "In op[%s], the parameter[%s]'s dtype "
                    "should be one of [%s], but actually is [%s]."
                    % (error_info['op_name'], error_info['param_name'],
                       error_info['param_type'], error_info['actual_type']))

        if param_type in list_type_attr:
            _check_list_attr(op_param, param_name, param_type, op_name)

    def _check_kernel_name(kernel_name, param_name, op_name):
        """
        check kernel_name
        """
        if not isinstance(kernel_name, str):
            error_info = {
                'errCode': OP_ERROR_CODE_003, 'op_name': op_name,
                'param_name': param_name, 'param_type': "str",
                'actual_type': kernel_name.__class__.__name__}
            raise RuntimeError(
                error_info,
                "In op[%s], the parameter[%s]'s type should be [%s], "
                "but actually is [%s]."
                % (error_info['op_name'], error_info['param_name'],
                   error_info['param_type'], error_info['actual_type']))

        if len(kernel_name) > MAX_KERNEL_NAME_LEN:
            error_info = {
                'errCode': OP_ERROR_CODE_002, 'op_name': op_name,
                'param_name': param_name, 'min_value': '0',
                'max_value': str(MAX_KERNEL_NAME_LEN),
                'real_value': str(len(kernel_name))}
            raise RuntimeError(
                error_info,
                "In op[%s], the parameter[%s] should be in "
                "the range of [%s, %s], but actually is [%s]."
                % (error_info['op_name'], error_info['param_name'],
                   error_info['min_value'], error_info['max_value'],
                   error_info['real_value']))

        pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        if not pattern.match(kernel_name):
            error_info = {'errCode': OP_ERROR_CODE_020, 'op_name': op_name}
            raise RuntimeError(
                error_info,
                "In op[%s],kernel_name can only contain letters, numbers and "
                "underscores, and begin with underscores or letters"
                % (error_info['op_name']))

    def _check_one_op_param(op_param, param_name, param_type, op_name=OP_NAME):

        if param_type in input_params:
            _check_input(op_param, param_name, param_type, op_name)
        elif param_type in output_params:
            _check_output(op_param, param_name, param_type, op_name)
        elif param_type == KERNEL_NAME:
            if op_param is None:
                return
            _check_kernel_name(op_param, param_name, op_name)
        else:  # else is attr_params:
            _check_attr(op_param, param_name, param_type, op_name)

    def _out_wrapper(func):
        formal_parameter = func.__code__.co_varnames
        formal_parameter_list = list(zip(formal_parameter, type_args))

        @wraps(func)
        def _in_wrapper(*args, **kwargs):
            global OP_NAME
            OP_NAME = func.__name__
            for i, one_args in enumerate(args):
                op_name = func.__name__
                _check_one_op_param(one_args, formal_parameter_list[i][0],
                                    formal_parameter_list[i][1], op_name)

            for arg_key in kwargs:
                op_name = func.__name__
                for name_type in formal_parameter_list:
                    if arg_key == name_type[0]:
                        _check_one_op_param(kwargs[arg_key], arg_key,
                                            name_type[1], op_name)
                        break

            return func(*args, **kwargs)

        return _in_wrapper

    return _out_wrapper


def _check_range(shape, shape_range, min_dim=0, max_dim=RANK_LIMIT,
                 max_shape_num=MAX_UNKNOWN_SHAPE_NUM, param_name=PARAM_NAME):
    """
    check rule for tensor shape
    """
    if not isinstance(shape_range, (tuple, list)):
        error_info = {
            'errCode': OP_ERROR_CODE_003, 'op_name': OP_NAME,
            'param_name': param_name, 'param_type': "list tuple",
            'actual_type': shape_range.__class__.__name__}
        raise RuntimeError(
            error_info,
            "In op, the parameter[%s]'s type should be [%s],"
            "but actually is [%s]."
            % (error_info['param_name'], error_info['param_type'],
               error_info['actual_type']))
    if len(shape) != len(shape_range):
        error_info = {
            'errCode': OP_ERROR_CODE_021, 'op_name': OP_NAME,
            'param_name': param_name, 'shape_len': len(shape),
            'range_len': len(shape_range)}
        raise RuntimeError(
            error_info,
            "In op, the length of shape[%s] and the length of range[%s] "
            "must be the same."
            % (error_info['shape_len'], error_info['range_len']))

    for range_i in shape_range:
        if len(range_i) == 2 and (range_i[1] is None) \
                and isinstance(range_i[0], int) \
                and 0 <= range_i[0] <= max_shape_num:
            continue
        if not isinstance(range_i[0], int):
            error_info = {
                'errCode': OP_ERROR_CODE_003, 'op_name': OP_NAME,
                'param_name': param_name, 'param_type': 'int',
                'actual_type': range_i[0].__class__.__name__}
            raise RuntimeError(
                error_info,
                "In op, the parameter[%s]'s type should be [%s], "
                "but actually is [%s]."
                % (error_info['param_name'], error_info['param_type'],
                   error_info['actual_type']))
        if not isinstance(range_i[1], int):
            error_info = {
                'errCode': OP_ERROR_CODE_003, 'op_name': OP_NAME,
                'param_name': param_name, 'param_type': 'int',
                'actual_type': range_i[1].__class__.__name__}
            raise RuntimeError(
                error_info,
                "In op, the parameter[%s]'s type should be [%s],"
                "but actually is [%s]."
                % (error_info['param_name'], error_info['param_type'],
                   error_info['actual_type']))
        valid_type = isinstance(range_i[0], int) and isinstance(range_i[1], int)
        if len(range_i) != 2:
            error_info = {'errCode': OP_ERROR_CODE_023, 'op_name': OP_NAME,
                          'param_name': param_name}
            raise RuntimeError(
                error_info,
                "In op[%s],the length of each element in the range must be two"
                % (error_info['op_name']))
        valid_range = \
            len(range_i) == 2 and 0 <= range_i[0] <= range_i[1] <= max_shape_num
        if valid_type and valid_range:
            continue
        else:
            error_info = {
                'errCode': OP_ERROR_CODE_022, 'op_name': OP_NAME,
                'param_name': param_name, 'first_real_value': range_i[0],
                'second_real_value': range_i[1], 'min_range_value': 0,
                'max_range_value': max_shape_num}
            raise RuntimeError(
                error_info,
                "In op, the dim of first range input[%s] is less than "
                "that of the second range input[%s], and the dim of range "
                "should be in the range of [%s, %s]."
                % (error_info['first_real_value'],
                   error_info['second_real_value'], 0, max_shape_num))


def _check_dynamic_shape(shape, max_dim=DIM_LIMIT, max_rank=RANK_LIMIT,
                         param_name=PARAM_NAME):

    _check_shape_range(max_rank, MIN_UNKNOWN_SHAPE_RANK, param_name, shape)
    for _, dim in enumerate(shape):
        valid_dim = -2 <= dim <= max_dim
        if not valid_dim:
            error_info = {
                'errCode': OP_ERROR_CODE_002, 'op_name': OP_NAME,
                'param_name': param_name, 'min_value': "-2",
                'max_value': max_dim, 'real_value': dim}
            raise RuntimeError(
                error_info,
                "In op, the parameter[%s] should be in "
                "the range of [%s, %s], "
                "but actually is [%s]."
                % (error_info['param_name'], -2, max_dim, dim))


def check_shape(shape, min_dim=0, max_dim=DIM_LIMIT, min_rank=0,
                max_rank=RANK_LIMIT, min_size=0,
                max_size=SHAPE_SIZE_LIMIT, param_name=PARAM_NAME):
    """
    check shape size
    """
    from tbe.dsl.base import operation
    if not isinstance(shape, (tuple, list)):
        error_info = {'errCode': OP_ERROR_CODE_003, 'op_name': OP_NAME,
                      'param_name': param_name,
                      'param_type': "list tuple",
                      'actual_type': shape.__class__.__name__}
        raise RuntimeError(
            error_info,
            "In op, the parameter[%s]'s type should be [%s], "
            "but actually is [%s]."
            % (error_info['param_name'], error_info['param_type'],
               error_info['actual_type']))

    for dim in shape:
        if not isinstance(dim, int):
            error_info = {'errCode': OP_ERROR_CODE_003, 'op_name': OP_NAME,
                          'param_name': param_name,
                          'param_type': 'int',
                          'actual_type': dim.__class__.__name__}
            raise RuntimeError(
                error_info,
                "In op, the parameter[%s]'s type should be [%s],  "
                "but actually is [%s]."
                % (error_info['param_name'], error_info['param_type'],
                   error_info['actual_type']))

    if operation.in_dynamic():
        _check_dynamic_shape(shape, max_dim, max_rank, param_name)
    else:
        _check_shape_range(max_rank, min_rank, param_name, shape)

        for _, dim in enumerate(shape):
            if dim < min_dim:
                error_info = {'errCode': OP_ERROR_CODE_002, 'op_name': OP_NAME,
                              'param_name': param_name,
                              'min_value': min_dim,
                              'real_value': dim}
                raise RuntimeError(
                    error_info,
                    "In op, the dim value[%s] should more than [%s],"
                    " but actually is [%s]."
                    % (error_info['param_name'], min_dim, dim))
        if shape:
            shape_size = _reduce(lambda x, y: x * y, shape[:])
        else:
            shape_size = 1
        if shape_size < min_size:
            error_info = {'errCode': OP_ERROR_CODE_011, 'op_name': OP_NAME,
                          'param_name': param_name,
                          'min_value': min_size, 'real_value': shape_size}
            raise RuntimeError(
                error_info,
                "In op, the shape size(product of all dimensions) of "
                "input[%s] should more than [%s], but actually is [%s]."
                % (error_info['min_value'], min_size, shape_size))


def _check_shape_range(max_rank, min_rank, param_name, shape):
    if len(shape) < min_rank or len(shape) > max_rank:
        error_info = {
            'errCode': OP_ERROR_CODE_012, 'op_name': OP_NAME,
            'param_name': param_name, 'min_value': min_rank,
            'max_value': max_rank, 'real_value': len(shape)}
        raise RuntimeError(
            error_info,
            "In op, the num of dimensions of input/output[%s] should be in"
            "the range of [%s, %s], but actually is [%s]."
            % (error_info['param_name'], min_rank, max_rank, len(shape)))


def check_dtype(dtype, check_list=ALL_DTYPE_LIST, param_name=PARAM_NAME):
    """
    The common check rule for tensor dtype
    """
    if dtype is None:
        error_info = {'errCode': OP_ERROR_CODE_007, 'op_name': OP_NAME,
                      'param_name': param_name}
        raise RuntimeError(error_info,
                           "In op, the input[%s]'s dtype could not be none."
                           % (error_info['param_name']))

    if not isinstance(dtype, str):
        error_info = {'errCode': OP_ERROR_CODE_003, 'op_name': OP_NAME,
                      'param_name': param_name, 'param_type': 'str',
                      'actual_type': dtype.__class__.__name__}
        raise RuntimeError(
            error_info,
            "In op, the parameter[%s]'s type should be [%s],  "
            "but actually is [%s]."
            % (error_info['param_name'], error_info['param_type'],
               error_info['actual_type']))
    if dtype.lower() not in check_list:
        error_info = {'errCode': OP_ERROR_CODE_008, 'op_name': OP_NAME,
                      'param_name': param_name,
                      'excepted_dtype_list': check_list, 'dtype': dtype.lower()}
        raise RuntimeError(
            error_info,
            "In op, the parameter[%s]'s dtype should be one of [%s]"
            ", but actually is [%s]."
            % (error_info['param_name'], error_info['excepted_dtype_list'],
               error_info['dtype']))


def check_format(data_format, check_list=None, param_name=PARAM_NAME):
    """
    The common check rule for tensor dtype
    """

    if check_list is None:
        check_list = ALL_FORMAT_LIST
    if data_format is None:
        error_info = {'errCode': OP_ERROR_CODE_017, 'op_name': OP_NAME,
                      'param_name': param_name}
        raise RuntimeError(
            error_info,
            "In op, the input[%s]'s format could not be none"
            % (error_info['param_name']))

    if data_format not in check_list:
        error_info = {'errCode': OP_ERROR_CODE_015, 'op_name': OP_NAME,
                      'param_name': param_name,
                      'excepted_format_list': ",".join(check_list),
                      'format': data_format}
        raise RuntimeError(
            error_info,
            "In op, the format of input[%s] should be one of [%s], "
            "but actually is [%s]."
            % (error_info['param_name'], error_info['excepted_format_list'],
               error_info['format']))


def check_elewise_shape_range(inputs: list, support_broadcast=False):
    """
    :param support_broadcast:True or False
    :param inputs: list, all inputs of operator
    :return:
    """
    from tbe.dsl.base import operation
    if not operation.in_dynamic():
        return

    def _has_intersection(range0, range1):
        _range0 = list(range0)
        _range1 = list(range1)
        if _range0[1] is None:
            _range0[1] = MAX_UNKNOWN_SHAPE_NUM
        if _range1[1] is None:
            _range1[1] = MAX_UNKNOWN_SHAPE_NUM
        return max(_range0[0], _range1[0]) <= min(_range0[1], _range1[1])

    def _check_range_relu(shape_x, shape_y, range_x, range_y):
        size_x = len(shape_x)
        size_y = len(shape_y)
        min_size = min(size_x, size_y)
        for i in range(1, min_size + 1):
            if len(range_x[-i]) != 2 or len(range_y[-i]) != 2:
                err_info = {'errCode': OP_ERROR_CODE_023,
                            'op_name': operation.get_context().get_op_type(),
                            'param_name': PARAM_NAME}
                raise RuntimeError(
                    err_info,
                    "In op[%s],the range of each element must be two"
                    % (err_info['op_name']))
            if support_broadcast:
                if (shape_x[-i] != 1 and shape_y[-i] != 1) and \
                        not (_has_intersection(range_x[-i], range_y[-i])
                             or range_x[-i][0] <= 1 or range_y[-i][0] <= 1):
                    err_info = {'errCode': OP_ERROR_CODE_024,
                                'op_name': operation.get_context().get_op_type(),
                                'param_name': PARAM_NAME}
                    raise RuntimeError(
                        err_info,
                        "In op[%s],the range at the same location "
                        "must have intersections" % (err_info['op_name']))
            else:
                if not _has_intersection(range_x[-i], range_y[-i]):
                    err_info = {'errCode': OP_ERROR_CODE_024,
                                'op_name': operation.get_context().get_op_type(),
                                'param_name': PARAM_NAME}
                    raise RuntimeError(
                        err_info,
                        "In op[%s],the range at the same location "
                        "must have intersections" % (err_info['op_name']))

    if len(inputs) <= 1:
        return
    last_shape = None
    last_range = None
    inputs_keys = (OpParamInfoKey.SHAPE.value, OpParamInfoKey.RANGE.value)
    for index, _input in enumerate(inputs):
        if not isinstance(_input, dict):
            error_info = {'errCode': OP_ERROR_CODE_003,
                          'op_name': operation.get_context().get_op_type(),
                          'param_name': PARAM_NAME, 'param_type': 'dict',
                          'actual_type': _input.__class__.__name__}
            raise RuntimeError(
                error_info,
                "In op[%s], the parameter[%s]'s type should be [%s],  "
                "but actually is [%s]."
                % (error_info['op_name'], error_info['param_name'],
                   error_info['param_type'], error_info['actual_type']))
        for key in inputs_keys:
            if key not in _input.keys():
                error_info = {'errCode': OP_ERROR_CODE_004,
                              'op_name': operation.get_context().get_op_type(),
                              'param_name': PARAM_NAME,
                              'key': OpParamInfoKey.RANGE.value}
                raise RuntimeError(
                    error_info,
                    "In op[%s], the input[%s] does not contain the item[%s]."
                    % (error_info['op_name'], error_info['param_name'],
                       error_info['key']))
        shape = _input.get("shape")
        _range = _input.get("range")
        if index > 0:
            _check_range_relu(shape, last_shape, _range, last_range)
        last_shape = shape
        last_range = _range


def _check_input_type_dict(input_dict, input_key, input_name):
    """
    check input parameter type for new type: dict
    rule1: key of input_dict should be in the input_key
    rule2: type of input_dict[shape] should be in (list, tuple), if have shape
    rule3: type of input_dict[dtype] should be in (str), if have dtype

    Parameters
    ----------
    input_dict: dict
        input_dict
    input_key: list or tuple
        all input key list, the key of input must in input_key
    input_name: str
        input param name, only used for error print

    Returns
    -------
    None
    """

    def _check_input_type(input_key, input_type):
        if not isinstance(input_dict[input_key], input_type):
            args_dict = {
                "errCode": "E60037",
                "param_name": "{}".format(input_key),
                "type_list": "{}".format(input_type),
                "type": "{}".format(type(input_dict[input_key]))
            }
            raise RuntimeError(
                args_dict,
                error_manager.get_error_message(args_dict)
            )

    for key in input_dict.keys():
        if key not in input_key:
            args_dict = {
                "errCode": "E60038",
                "desc": "input parameter value must have property {}".format(key)
            }
            raise RuntimeError(
                args_dict,
                error_manager.get_error_message(args_dict)
            )
        # check shape's type of input_dict, if have shape
        if key == "shape":
            _check_input_type(key, (list, tuple))

        # check dtype's type of input_dict, if have dtype
        if key == "dtype":
            _check_input_type(key, (str,))


def check_input_type(*type_args, **type_kwargs):
    """
    check input parameter type
    """

    def out_wrapper(func):
        """
        out_wrapper

        :param func: func
        :return: None
        """
        formal_parameter = func.__code__.co_varnames
        formal_parameter_list = list(zip(formal_parameter, type_args))

        @wraps(func)
        def in_wrapper(*args, **kwargs):
            """
            in_wrapper
            :param args: args
            :param kwargs: kwargs
            :return: None
            """
            for i in range(len(args)):
                # add for new input dict, if dict, will check shape and dtype
                if isinstance(args[i], dict):
                    _check_input_type_dict(args[i], args[i].keys(),
                                          formal_parameter_list[i][0])
                if not isinstance(args[i], formal_parameter_list[i][1]):
                    args_dict = {
                        "errCode": "E60038",
                        "desc":
                            "Input parameter type error, expected type is {}, "
                            "actual input is {}".format(
                                formal_parameter_list[i][1],
                                type(args[i])
                            )
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager.get_error_message(args_dict)
                    )
            for i in kwargs:
                for j in formal_parameter_list:
                    if i in j:
                        if not isinstance(kwargs[i], j[1]):
                            args_dict = {
                                "errCode": "E60038",
                                "desc":
                                    "Input {} type error, "
                                    "expected type is {}, actual input is {}".format(
                                        i,
                                        j[1],
                                        type(kwargs[i])
                                    )
                            }
                            raise RuntimeError(
                                args_dict,
                                error_manager.get_error_message(args_dict)
                            )
                        break
            return func(*args, **kwargs)

        return in_wrapper

    return out_wrapper


def check_dtype_rule(dtype, check_list, param_name="default"):
    """
    The common check rule for tensor dtype
    """
    if dtype is None:
        args_dict = {
            "errCode": "E60038",
            "desc": "dtype is None"
        }
        raise RuntimeError(
            args_dict,
            error_manager.get_error_message(args_dict)
        )

    if dtype.lower() not in check_list:
        if param_name == "default":
            args_dict = {
                "errCode": "E60038",
                "desc": "The data type is not supported. Please check the data type"
            }
            raise RuntimeError(
                args_dict,
                error_manager.get_error_message(args_dict)
            )
        else:
            args_dict = {
                "errCode": "E60005",
                "param_name": param_name,
                "expected_dtype_list": "{}".format(check_list),
                "dtype": "{}".format(dtype.lower()),
            }
            raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))


def check_shape_rule(shape, min_dim=None, max_dim=None, max_shape_num=None):
    """
    The common check rule for tensor shape
    """
    if min_dim is None:
        min_dim = DEFAULT_MIN_SHAPE_DIM

    if max_dim is None:
        max_dim = DEFAULT_MAX_SHAPE_DIM

    if max_shape_num is None:
        max_shape_num = DEFAULT_MAX_SHAPE_NUM

    if not isinstance(shape, (tuple, list)):
        args_dict = {
            "errCode": "E60037",
            "param_name": "shape",
            "type_list": "[tuple, list]",
            "type": "{}".format(type(shape))
        }
        raise RuntimeError(
            args_dict,
            error_manager.get_error_message(args_dict)
        )

    if len(shape) < min_dim or len(shape) > max_dim:
        args_dict = {
            "errCode": "E60011",
            "attr_name": "shape dim",
            "range": "[{}, {}]".format(min_dim, max_dim),
            "value": "{}".format(len(shape))
        }
        raise RuntimeError(
            args_dict,
            error_manager.get_error_message(args_dict)
        )

    for i in range(len(shape)):
        if type(shape[i]) != int:
            args_dict = {
                "errCode": "E60037",
                "param_name": "shape axis",
                "type_list": "[int]",
                "type": "{}".format(type(shape[i]))
            }
            raise RuntimeError(
                args_dict,
                error_manager.get_error_message(args_dict)
            )
        if shape[i] <= 0:
            args_dict = {
                "errCode": "E60039",
                "attr_name": "axis",
                "param_name": "shape",
                "comparator": "more",
                "expected_value": "0",
                "input_value": "{}".format(shape[i])
            }
            raise RuntimeError(
                args_dict,
                error_manager.get_error_message(args_dict)
            )


def check_kernel_name(kernel_name):
    """
    check kernel_name
    ----------
    kernel_name: str or None

    Returns
    -------
    None
    """
    if kernel_name is None:
        return

    if not isinstance(kernel_name, str):
        try:
            kernel_name = str(kernel_name)
        except ValueError:
            args_dict = {
                "errCode": "E60037",
                "param_name": "kernel_name",
                "type_list": "[str, None]",
                "type": "{}".format(type(kernel_name))
            }
            raise RuntimeError(
                args_dict,
                error_manager.get_error_message(args_dict)
            )

    if len(kernel_name) > MAX_KERNEL_NAME_LEN:
        args_dict = {
            "errCode": "E60039",
            "attr_name": "length",
            "param_name": "kernel_name",
            "comparator": "less",
            "expected_value": "{}".format(MAX_KERNEL_NAME_LEN),
            "input_value": "{}".format(len(kernel_name))
        }
        raise RuntimeError(
            args_dict,
            error_manager.get_error_message(args_dict)
        )

    pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
    if not pattern.match(kernel_name):
        args_dict = {
            "errCode": "E60038",
            "desc": "kernel_name can only contain letters, numbers and underscores,"
                    "and begin with underscores or letters"
        }
        raise RuntimeError(
            args_dict,
            error_manager.get_error_message(args_dict)
        )


def check_and_init_5hdc_reduce_support(input_tensor, axis):
    """5HD Special param for 5hd schedule"""
    if "format" in input_tensor and input_tensor["format"] == "NC1HWC0" and \
            1 in axis and 4 in axis and input_tensor["dtype"] == "float16":
        if "ori_shape" in input_tensor and "ori_format" in input_tensor:
            return True
        else:
            args_dict = {
                "errCode": "E60040",
                "param_name": "input tensor",
                "attr_name": "ori_shape and ori_format"
            }
            raise RuntimeError(
                args_dict,
                error_manager.get_error_message(args_dict)
            )
    return False


def is_scalar(shape):
    """
    verify that tensor is scalar
    ----------
    shape: shape of data

    Returns
    -------
    True or False
    """
    if isinstance(shape, (list, tuple)):
        if len(shape) == 1 and shape[0] == 1:
            return True
    return False


def check_shape_size(shape, limit=SHAPE_SIZE_LIMIT+1):
    """
    if get all shape size, use get_shape_size function.
    ----------
    shape: shape of data

    limit: limit of the product of all dimension

    Returns
    -------
    None
    """
    from functools import reduce
    product = reduce(lambda x, y: x * y, shape[:])  # product of all dimension
    if product >= limit:
        args_dict = {
            "errCode": "E60039",
            "attr_name": "size",
            "param_name": "shape",
            "comparator": "less",
            "expected_value": "{}".format(limit),
            "input_value": "{}".format(product)
        }
        raise RuntimeError(
            args_dict,
            error_manager.get_error_message(args_dict)
        )


def check_tensor_shape_size(shape):
    """
    The function is deprecated.
    if check shape size, use check_shape_size function.
    if get all shape size,use get_shape_size function.
    """
    warnings.warn("check_tensor_shape_size is deprecated", DeprecationWarning)
    from functools import reduce
    product = reduce(lambda x, y: x * y, shape[:])  # product of all dimension

    return product


def check_reduce_shape_rule(shape):
    """
    check the shape of reduce axis must be less than MAX_REDUCE_SHAPE_NUM
    :param shape: inout shape
    """
    # the shape of reduce axis must be less than MAX_REDUCE_SHAPE_NUM
    warnings.warn("check_reduce_shape_rule is deprecated", DeprecationWarning)
    from functools import reduce
    product = reduce(lambda x, y: x * y, shape[:])  # product of all dimension