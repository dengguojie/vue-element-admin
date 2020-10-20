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
common function
"""
import re
import math
from functools import reduce as functools_reduce
from functools import wraps
from te.platform.fusion_manager import fusion_manager
from te.platform import operation
from te import tvm

SHAPE_SIZE_LIMIT = 2 ** 31 - 1
SHAPE_SIZE_ZERO = 0
RANK_ZERO = 0
RANK_LIMIT = 8
DIM_LIMIT = 2 ** 31 - 1
ZERO_DIM = 0
# the max len of kernel_name
MAX_KERNEL_NAEM_LEN = 200
MIN_UNKOWN_SHAPE_RANK = 0
MAX_UNKOWN_SHAPE_NUM = 2 ** 31 - 1

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

KERNEL_NAME = "kernel_name"

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


class OpParamInfoKey:  # pylint: disable=too-few-public-methods
    """
    Define op params
    """

    def __init__(self):
        pass

    SHAPE = "shape"
    FORMAT = "format"
    ORI_SHAPE = "ori_shape"
    ORI_FORMAT = "ori_format"
    D_TYPE = "dtype"
    RANGE = "range"


class TensorFormat:  # pylint: disable=too-few-public-methods
    """
    Define op params
    """

    def __init__(self):
        pass

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


ALL_FORMAT_LIST = [TensorFormat.__dict__[d_key]
                   for d_key in TensorFormat.__dict__ if "__" not in d_key]
ALL_DTYPE_LIST = ("int8", "uint8", "int16", "uint16", "int32", "uint32",
                  "int64", "uint64", "float16", "float32", "float64", "bool")
OP_NAME = ""
PARAM_NAME = ""


def check_op_params(*type_args,  # pylint: disable=too-many-locals,too-many-statements
                    **type_kwargs):  # pylint: disable=unused-argument,
    """
    check op params
    """
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
        # check all necessary information(shape, format, ori_shape, ori_format, dtype)
        if not isinstance(op_param, dict):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = 'dict'
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s],  "
                                           "but actually is [%s]." % (error_info['op_name'],
                                                                      error_info['param_name'],
                                                                      error_info['param_type'],
                                                                      error_info['actual_type']))
        if OpParamInfoKey.SHAPE not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.SHAPE
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))
        if OpParamInfoKey.FORMAT not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.FORMAT
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))
        if OpParamInfoKey.ORI_SHAPE not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.ORI_SHAPE
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))
        if OpParamInfoKey.ORI_FORMAT not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.ORI_FORMAT
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))
        if OpParamInfoKey.D_TYPE not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.D_TYPE
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['key']))

        if operation.in_dynamic():
            if OpParamInfoKey.RANGE not in op_param.keys():
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_004
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['key'] = OpParamInfoKey.RANGE
                raise RuntimeError(error_info,
                                   "In op[%s], the input[%s] does not contain the item[%s]."
                                   % (error_info['op_name'], error_info['param_name'], error_info['key']))

    def _check_input_output_dict(op_param, param_name, op_name=OP_NAME):
        _check_input_output_key(op_param, param_name, op_name)
        if operation.in_dynamic():
            check_range(op_param[OpParamInfoKey.SHAPE], op_param[OpParamInfoKey.RANGE], param_name=param_name)
        check_shape(op_param[OpParamInfoKey.SHAPE], param_name=param_name)
        check_shape(op_param[OpParamInfoKey.ORI_SHAPE], param_name=param_name)

        if op_param[OpParamInfoKey.FORMAT] not in TensorFormat.__dict__.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_015
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['excepted_format_list'] = ",".join(ALL_FORMAT_LIST)
            error_info['format'] = op_param[OpParamInfoKey.FORMAT]

            raise RuntimeError(error_info, "In op[%s], the format of input[%s] "
                                           "should be one of [%s], but actually is [%s]."
                               % (error_info['op_name'],
                                  error_info['param_name'],
                                  error_info['excepted_format_list'],
                                  error_info['format']))

        if op_param[OpParamInfoKey.ORI_FORMAT] not in TensorFormat.__dict__.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_014
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['excepted_format_list'] = ",".join(ALL_FORMAT_LIST)
            error_info['format'] = op_param[OpParamInfoKey.ORI_FORMAT]
            raise RuntimeError(error_info,
                               "In op[%s], the ori format of input[%s] should be one of [%s]"
                               ", but actually is [%s]."
                               % (error_info['op_name'],
                                  error_info['param_name'],
                                  ",".join(ALL_FORMAT_LIST),
                                  error_info['format']))

        if not isinstance(op_param[OpParamInfoKey.D_TYPE], str):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = 'str'
            error_info['actual_type'] = op_param[OpParamInfoKey.D_TYPE].__class__.__name__
            raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s],  "
                                           "but actually is [%s]." % (error_info['op_name'],
                                                                      error_info['param_name'],
                                                                      error_info['param_type'],
                                                                      error_info['actual_type']))

        if op_param[OpParamInfoKey.D_TYPE] is None or op_param[OpParamInfoKey.D_TYPE].lower() not in ALL_DTYPE_LIST:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_008
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['excepted_dtype_list'] = ",".join(ALL_DTYPE_LIST)
            error_info['dtype'] = op_param[OpParamInfoKey.D_TYPE]
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s dtype should be "
                               "one of [%s], but actually is [%s]." %
                               (error_info['op_name'],
                                error_info['param_name'],
                                error_info['excepted_dtype_list'],
                                error_info['dtype']))

    def _check_input(op_param, param_name, param_type, op_name=OP_NAME):
        if param_type == REQUIRED_INPUT:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_001
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            if op_param is None:
                raise RuntimeError(error_info, "In op[%s], the mandatory "
                                               "parameter[%s] is missed."
                                   % (error_info['op_name'], error_info['param_name']))
            _check_input_output_dict(op_param, param_name, op_name)
        elif param_type == OPTION_INPUT:
            if op_param is not None:
                _check_input_output_dict(op_param, param_name, op_name)
        else:
            if not isinstance(op_param, (list, tuple)):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_003
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['param_type'] = "list truple"
                error_info['actual_type'] = op_param.__class__.__name__
                raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s]"
                                               ",  but actually is [%s]."
                                   % (op_name, param_name, error_info['param_type'], error_info['actual_type']))
            if not op_param:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_001
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                raise RuntimeError(error_info, "In op[%s], the mandatory parameter[%s]"
                                               " is missed." % (op_name, param_name))
            for one_input in op_param:
                _check_input_output_dict(one_input, param_name, op_name)

    def _check_output(op_param, param_name, param_type, op_name=OP_NAME):
        if param_type == REQUIRED_OUTPUT:
            if op_param is None:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_001
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                raise RuntimeError(error_info, "In op[%s], the mandatory parameter[%s]"
                                               " is missed." % (op_name, param_name))

            _check_input_output_dict(op_param, param_name, op_name)
        elif param_type == OPTION_OUTPUT:
            if op_param is not None:
                _check_input_output_dict(op_param, param_name, op_name)
        else:
            if not isinstance(op_param, (list, tuple)):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_003
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['param_type'] = "list tuple"
                error_info['actual_type'] = op_param.__class__.__name__
                raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s "
                                               "type should be [%s],  but actually is [%s]."
                                   % (op_name, param_name, error_info['param_type'], error_info['actual_type']))
            if not op_param:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_001
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                raise RuntimeError(error_info, "In op[%s], the mandatory"
                                               " parameter[%s] is missed."
                                   % (op_name, param_name))
            for one_input in op_param:
                _check_input_output_dict(one_input, param_name, op_name)

    def _check_attr_type(op_param, param_name, py_type, py_type_name, op_name=OP_NAME):
        if not isinstance(op_param, py_type):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = str(py_type)
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s],"
                               " but actually is [%s]."
                               % (error_info['op_name'], error_info['param_name'],
                                  error_info['param_type'], error_info['actual_type']))
        if py_type_name == "float":
            if math.isinf(op_param) or math.isnan(op_param):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_000
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['excepted_value'] = "float range data"
                error_info['real_value'] = str(op_param)
                raise RuntimeError(error_info,
                                   "In op[%s], the parameter[%s] should be [%s], but actually is [%s]."
                                   % (error_info['op_name'], error_info['param_name'],
                                      error_info['excepted_value'], error_info['real_value']))

    def _check_list_attr_element(op_param, param_name, py_type, py_type_name, op_name=OP_NAME):
        if not isinstance(op_param, py_type):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = str(py_type)
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s],"
                               " but actually is [%s]."
                               % (error_info['op_name'], error_info['param_name'],
                                  error_info['param_type'], error_info['actual_type']))
        if py_type_name == "float":
            if math.isinf(op_param) or math.isnan(op_param):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_000
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['excepted_value'] = "float range data"
                error_info['real_value'] = str(op_param)
                raise RuntimeError(error_info,
                                   "In op[%s], the parameter[%s] should be [%s], but actually is [%s]."
                                   % (error_info['op_name'], error_info['param_name'],
                                      error_info['excepted_value'], error_info['real_value']))

    def _check_list_attr(op_param, param_name, param_type, op_name=OP_NAME):
        if not isinstance(op_param, (list, tuple)):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = "list tuple"
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s],"
                               "  but actually is [%s]."
                               % (error_info['op_name'], error_info['param_name'],
                                  error_info['param_type'], error_info['actual_type']))

        if param_type in [REQUIRED_ATTR_LIST_BOOL, OPTION_ATTR_LIST_BOOL]:
            for one_attr in op_param:
                _check_list_attr_element(one_attr, param_name, bool, "bool", op_name)

        if param_type in [REQUIRED_ATTR_LIST_INT, OPTION_ATTR_LIST_INT]:
            for one_attr in op_param:
                _check_list_attr_element(one_attr, param_name, int, "int", op_name)

        if param_type in [REQUIRED_ATTR_LIST_FLOAT, OPTION_ATTR_LIST_FLOAT]:
            for one_attr in op_param:
                _check_list_attr_element(one_attr, param_name, float, "float", op_name)

        if param_type in [REQUIRED_ATTR_LIST_LIST_INT, OPTION_ATTR_LIST_LIST_INT]:
            for one_attr in op_param:
                if not isinstance(one_attr, (list, tuple)):
                    error_info = {}
                    error_info['errCode'] = OP_ERROR_CODE_003
                    error_info['op_name'] = op_name
                    error_info['param_name'] = param_name
                    error_info['param_type'] = "list tuple"
                    error_info['actual_type'] = op_param.__class__.__name__
                    raise RuntimeError(error_info,
                                       "In op[%s], the parameter[%s]'s type should be [%s],"
                                       " but actually is [%s]."
                                       % (error_info['op_name'],
                                          error_info['param_name'],
                                          error_info['param_type'],
                                          error_info['actual_type']))

                for ele in one_attr:
                    _check_list_attr_element(ele, param_name, int, "int", op_name)

    def _check_attr(op_param, param_name, param_type, op_name=OP_NAME):
        if op_param is None and param_type in required_attr_params:

            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_001
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            raise RuntimeError(error_info,
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
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_003
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['param_type'] = " ".join(ALL_DTYPE_LIST)
                error_info['actual_type'] = op_param.__class__.__name__
                raise RuntimeError(error_info,
                                   "In op[%s], the parameter[%s]'s dtype should"
                                   " be one of [%s], but actually is [%s]."
                                   % (error_info['op_name'],
                                      error_info['param_name'],
                                      error_info['param_type'],
                                      error_info['actual_type']))

        if param_type in list_type_attr:
            _check_list_attr(op_param, param_name, param_type, op_name)

    def _check_kernel_name(kernel_name, param_name, op_name):
        """
        check kernel_name
        """
        if not isinstance(kernel_name, str):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = "str"
            error_info['actual_type'] = kernel_name.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s], "
                               "but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['param_type'], error_info['actual_type']))

        if len(kernel_name) > MAX_KERNEL_NAEM_LEN:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_002
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['min_value'] = '0'
            error_info['max_value'] = str(MAX_KERNEL_NAEM_LEN)
            error_info['real_value'] = str(len(kernel_name))
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s] should be in the range of [%s, %s],"
                               "but actually is [%s]." % (error_info['op_name'],
                                                          error_info['param_name'],
                                                          error_info['min_value'],
                                                          error_info['max_value'],
                                                          error_info['real_value']))

        pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        if not pattern.match(kernel_name):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_020
            error_info['op_name'] = op_name
            raise RuntimeError(error_info,
                               "In op[%s],kernel_name can only contain letters, numbers and underscores,"
                               " and begin with underscores or letters" % (error_info['op_name']))

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
            for i, one_args in enumerate(args):
                op_name = func.__name__
                _check_one_op_param(one_args, formal_parameter_list[i][0],
                                    formal_parameter_list[i][1], op_name)

            for arg_key in kwargs:
                op_name = func.__name__
                for name_type in formal_parameter_list:
                    if arg_key == name_type[0]:
                        _check_one_op_param(kwargs[arg_key], arg_key, name_type[1], op_name)
                        break

            return func(*args, **kwargs)

        return _in_wrapper

    return _out_wrapper


def check_range(shape, shape_range, min_dim=0,  # pylint: disable=too-many-arguments
                max_dim=RANK_LIMIT, max_shape_num=MAX_UNKOWN_SHAPE_NUM,  # pylint: disable=too-many-arguments
                param_name=PARAM_NAME):  # pylint: disable=too-many-arguments
    """
    check rule for tensor shape
    """
    if not isinstance(shape_range, (tuple, list)):
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_003
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['param_type'] = "list tuple"
        error_info['actual_type'] = shape_range.__class__.__name__
        raise RuntimeError(error_info,
                           "In op, the parameter[%s]'s type should be [%s],"
                           "but actually is [%s]." %
                           (error_info['param_name'],
                            error_info['param_type'], error_info['actual_type']))
    if len(shape) != len(shape_range):
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_021
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['shape_len'] = len(shape)
        error_info['range_len'] = len(shape_range)
        raise RuntimeError(error_info,
                           "In op, the length of shape[%s] and"
                           "the length of range[%s] must be the same." %
                           (error_info['shape_len'], error_info['range_len']))

    for range_i in shape_range:
        if len(range_i) == 2 and (range_i[1] is None) \
                and isinstance(range_i[0], int) \
                and 0 < range_i[0] <= max_shape_num:
            continue
        if not isinstance(range_i[0], int):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['param_type'] = 'int'
            error_info['actual_type'] = range_i[0].__class__.__name__
            raise RuntimeError(error_info,
                               "In op, the parameter[%s]'s type should be [%s],"
                               "but actually is [%s]." %
                               (error_info['param_name'], error_info['param_type'],
                                error_info['actual_type']))
        if not isinstance(range_i[1], int):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['param_type'] = 'int'
            error_info['actual_type'] = range_i[1].__class__.__name__
            raise RuntimeError(error_info,
                               "In op, the parameter[%s]'s type should be [%s],"
                               "but actually is [%s]." %
                               (error_info['param_name'], error_info['param_type'],
                                error_info['actual_type']))
        valid_type = isinstance(range_i[0], int) and \
            isinstance(range_i[1], int)
        if len(range_i) != 2:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_023
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            raise RuntimeError(error_info,
                               "In op[%s],the length of each element"
                               "in the range must be two" %
                               (error_info['op_name']))
        valid_range = len(range_i) == 2 and 0 < range_i[0] <= range_i[1] <= max_shape_num
        if valid_type and valid_range:
            continue
        else:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_022
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['first_real_value'] = range_i[0]
            error_info['second_real_value'] = range_i[1]
            error_info['min_range_value'] = 0
            error_info['max_range_value'] = max_shape_num
            raise RuntimeError(error_info,
                               "In op, the ndim of first range input[%s] "
                               "is less than that of the second range input[%s], "
                               "and the ndim of range should be in the range of [%s, %s]."
                               % (error_info['first_real_value'],
                                  error_info['second_real_value'],
                                  0,
                                  max_shape_num))


def check_dynamic_shape(shape, max_dim=DIM_LIMIT, max_rank=RANK_LIMIT, param_name=PARAM_NAME):
    if len(shape) < MIN_UNKOWN_SHAPE_RANK or len(shape) > max_rank:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_012
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['min_value'] = MIN_UNKOWN_SHAPE_RANK
        error_info['max_value'] = max_rank
        error_info['real_value'] = len(shape)
        raise RuntimeError(error_info,
                           "In op, the num of dimensions of input[%s] should be in"
                           "the range of [%s, %s], but actually is [%s]."
                           % (error_info['param_name'], MIN_UNKOWN_SHAPE_RANK, max_rank, len(shape)))
    for _, dim in enumerate(shape):
        valid_dim = -1 <= dim <= max_dim and dim != 0
        if not valid_dim:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_002
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['min_value'] = "-1"
            error_info['max_value'] = max_dim
            error_info['real_value'] = dim
            raise RuntimeError(error_info,
                               "In op, the parameter[%s] should be in the range of [%s, %s] and cannot be zero,"
                               "but actually is [%s]."
                               % (error_info['param_name'], -1, max_dim, dim))


def check_shape(shape, min_dim=0, max_dim=DIM_LIMIT,  # pylint: disable=too-many-arguments
                min_rank=0, max_rank=RANK_LIMIT,  # pylint: disable=too-many-arguments
                min_size=0, max_size=SHAPE_SIZE_LIMIT, param_name=PARAM_NAME):  # pylint: disable=too-many-arguments
    """
    check shape size
    """
    if not isinstance(shape, (tuple, list)):

        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_003
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['param_type'] = "list tuple"
        error_info['actual_type'] = shape.__class__.__name__
        raise RuntimeError(error_info,
                           "In op, the parameter[%s]'s type should be [%s], "
                           "but actually is [%s]." %
                           (error_info['param_name'],
                            error_info['param_type'], error_info['actual_type']))

    for dim in shape:
        if not isinstance(dim, int):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['param_type'] = 'int'
            error_info['actual_type'] = dim.__class__.__name__
            raise RuntimeError(error_info,
                               "In op, the parameter[%s]'s type should be [%s],  "
                               "but actually is [%s]." %
                               (error_info['param_name'], error_info['param_type'],
                                error_info['actual_type']))

    if operation.in_dynamic():
        check_dynamic_shape(shape, max_dim, max_rank, param_name)
    else:
        if len(shape) < min_rank or len(shape) > max_rank:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_012
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['min_value'] = min_rank
            error_info['max_value'] = max_rank
            error_info['real_value'] = len(shape)
            raise RuntimeError(error_info,
                               "In op, the num of dimensions of input[%s] should be in"
                               "the range of [%s, %s], but actually is [%s]."
                               % (error_info['param_name'], min_rank, max_rank, len(shape)))

        for _, dim in enumerate(shape):
            if dim < min_dim:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_002
                error_info['op_name'] = OP_NAME
                error_info['param_name'] = param_name
                error_info['min_value'] = min_dim
                error_info['real_value'] = dim
                raise RuntimeError(error_info,
                                   "In op, the dim value[%s] should more than [%s],"
                                   "but actually is [%s]."
                                   % (error_info['param_name'], min_dim, dim))
        if shape:
            shape_size = functools_reduce(lambda x, y: x * y, shape[:])
        else:
            shape_size = 1
        if shape_size < min_size:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_011
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = param_name
            error_info['min_value'] = min_size
            error_info['real_value'] = shape_size
            raise RuntimeError(error_info,
                               "In op, the shape size(product of all dimensions) of "
                               "input[%s] should more than [%s], but actually is [%s]."
                               % (error_info['min_value'], min_size, shape_size))


def check_dtype(dtype, check_list=ALL_DTYPE_LIST, param_name=PARAM_NAME):
    """
    The common check rule for tensor dtype
    """
    if dtype is None:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_007
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        raise RuntimeError(error_info, "In op, the input[%s]'s dtype could not be none." %
                           (error_info['param_name']))

    if not isinstance(dtype, str):
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_003
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['param_type'] = 'str'
        error_info['actual_type'] = dtype.__class__.__name__
        raise RuntimeError(error_info, "In op, the parameter[%s]'s type should be [%s],  "
                                       "but actually is [%s]." %
                           (error_info['param_name'], error_info['param_type'], error_info['actual_type']))
    if dtype.lower() not in check_list:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_008
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['excepted_dtype_list'] = check_list
        error_info['dtype'] = dtype.lower()
        raise RuntimeError(error_info, "In op, the parameter[%s]'s dtype should be one of [%s]"
                                       ", but actually is [%s]."
                           % (error_info['param_name'],
                              error_info['excepted_dtype_list'], error_info['dtype']))


def check_format(data_format, check_list=ALL_FORMAT_LIST, param_name=PARAM_NAME):
    """
    The common check rule for tensor dtype
    """

    if data_format is None:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_017
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        raise RuntimeError(error_info, "In op, the input[%s]'s format could not be none" %
                           (error_info['param_name']))

    if data_format not in check_list:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_015
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = param_name
        error_info['excepted_format_list'] = ",".join(check_list)
        error_info['format'] = data_format
        raise RuntimeError(error_info, "In op, the format of input[%s] should "
                                       "be one of [%s], but actually is [%s]."
                           % (error_info['param_name'],
                               error_info['excepted_format_list'], error_info['format']))


def check_elewise_shape_range(inputs: list, support_broadcast=False):
    """
    :param inputs: list, all inputs of operator
    :return:
    """
    def _has_intersection(range0, range1):
        _range0 = list(range0)
        _range1 = list(range1)
        if _range0[1] is None:
            _range0[1] = MAX_UNKOWN_SHAPE_NUM
        if _range1[1] is None:
            _range1[1] = MAX_UNKOWN_SHAPE_NUM
        return max(_range0[0], _range1[0]) <= min(_range0[1], _range1[1])

    def _check_range_relu(shape_x, shape_y, range_x, range_y):
        size_x = len(shape_x)
        size_y = len(shape_y)
        min_size = min(size_x, size_y)
        for i in range(1, min_size + 1):
            if len(range_x[-i]) != 2 or len(range_y[-i]) != 2:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_023
                error_info['op_name'] = operation.get_context().get_op_type()
                error_info['param_name'] = PARAM_NAME
                raise RuntimeError(error_info,
                                   "In op[%s],the range of each element must be two" % (error_info['op_name']))
            if support_broadcast:
                if (shape_x[-i] != 1 and shape_y[-i] != 1) and \
                        not (_has_intersection(range_x[-i], range_y[-i])
                             or range_x[-i][0] <= 1 or range_y[-i][0] <= 1):
                    error_info = {}
                    error_info['errCode'] = OP_ERROR_CODE_024
                    error_info['op_name'] = operation.get_context().get_op_type()
                    error_info['param_name'] = PARAM_NAME
                    raise RuntimeError(error_info,
                                       "In op[%s],the range at the same location "
                                       "must have intersections" % (error_info['op_name']))
            else:
                if not _has_intersection(range_x[-i], range_y[-i]):
                    error_info = {}
                    error_info['errCode'] = OP_ERROR_CODE_024
                    error_info['op_name'] = operation.get_context().get_op_type()
                    error_info['param_name'] = PARAM_NAME
                    raise RuntimeError(error_info,
                                       "In op[%s],the range at the same location "
                                       "must have intersections" % (error_info['op_name']))

    if len(inputs) <= 1:
        return
    last_shape = None
    last_range = None
    inputs_keys = (OpParamInfoKey.SHAPE, OpParamInfoKey.RANGE)
    for index, _input in enumerate(inputs):
        if not isinstance(_input, dict):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = operation.get_context().get_op_type()
            error_info['param_name'] = PARAM_NAME
            error_info['param_type'] = 'dict'
            error_info['actual_type'] = _input.__class__.__name__
            raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s],  "
                                           "but actually is [%s]." % (error_info['op_name'],
                                                                      error_info['param_name'],
                                                                      error_info['param_type'],
                                                                      error_info['actual_type']))
        for key in inputs_keys:
            if key not in _input.keys():
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_004
                error_info['op_name'] = operation.get_context().get_op_type()
                error_info['param_name'] = PARAM_NAME
                error_info['key'] = OpParamInfoKey.RANGE
                raise RuntimeError(error_info,
                                   "In op[%s], the input[%s] does not contain the item[%s]."
                                   % (error_info['op_name'], error_info['param_name'], error_info['key']))
        shape = _input.get("shape")
        _range = _input.get("range")
        if index > 0:
            _check_range_relu(shape, last_shape, _range, last_range)
        last_shape = shape
        last_range = _range


def squeeze_shape(shape):
    """
    squeeze shape
    """
    squeezed_shape = [i for i in shape if i > 1]
    if not squeezed_shape:
        squeezed_shape = [1]

    return squeezed_shape


def wrap_axes_to_positive(axes, rank):
    """
    wrap axis to positive
    """
    if isinstance(axes, (tuple, list)):
        local_axes = axes
    else:
        local_axes = [axes]
    res_axes = []
    for axis in local_axes:
        if rank <= axis or axis < -rank:
            raise RuntimeError("Axis must between [-%d, %d)." % (rank, rank))
        if axis < 0:
            laxis = axis + rank
        else:
            laxis = axis
        res_axes.append(laxis)

    return res_axes


def refine_shape_axes(shape, axes):
    """
    refine shape and axes for reduce ops, fused reduced axes, and fused not reduced axes
    result is a tuple of (shape, axes)
    for example:
        input: shape is (2,3,4,5,6), axes is (1, -3)
        output: (2, 12, 30), (1,)

    Parameters
    ----------
    shape : shape which need refine

    axes : axes which need refine

    Returns
    -------
    shape : list
        refined shape

    axes : list
        refined axes

    """
    if len(shape) == 1:
        return shape, axes
    wrapped_axes = wrap_axes_to_positive(axes, len(shape))
    wrapped_axes = sorted(wrapped_axes)
    refined_axes = []
    reduce_flag = -1
    refined_shape = []
    for idx, dim in enumerate(shape):
        if dim == 1:
            # dim is one, not need reduce skip
            continue
        tmp_flag = 1 if idx in wrapped_axes else 0
        if reduce_flag == 1 and tmp_flag == 1:
            # continues reduce
            refined_shape[-1] *= dim
        elif reduce_flag == 0 and tmp_flag == 0:
            # continues no reduce
            refined_shape[-1] *= dim
        else:
            refined_shape.append(dim)
            if tmp_flag == 1:
                refined_axes.append(idx)
            reduce_flag = tmp_flag

    if not refined_shape:
        refined_shape.append(1)

    return refined_shape, refined_axes


def broadcast_shapes(shape1, shape2, op_name=OP_NAME, param_name_input1='', param_name_input2=''):
    """
    two input shapes produce three output shape
    """
    def _generate_dynamic_output(_shape1_i, _shape2_i, out_shape, index):
        if not _equal(_shape1_i, _shape2_i):
            if isinstance(_shape1_i, int):
                if _shape1_i == 1:
                    out_shape.append(_shape2_i)
                else:
                    out_shape.append(_shape1_i)
            elif isinstance(_shape2_i, int):
                if _shape2_i == 1:
                    out_shape.append(_shape1_i)
                else:
                    out_shape.append(_shape2_i)
            else:
                var_name = "dim_" + str(index) + "_2"
                _var = operation.get_te_var(var_name)
                if _var is None:
                    bound_x = operation.get_te_var(_shape1_i.name).get_bound()
                    bound_y = operation.get_te_var(_shape2_i.name).get_bound()
                    bound = (min(bound_x[0], bound_y[0]),
                             max(bound_x[1], bound_y[1]))
                    _var = operation.var(var_name, bound)
                else:
                    _var = _var.tvm_var
                out_shape.append(_var)
        else:
            out_shape.append(_shape1_i)

    shape1 = list(shape1)
    shape2 = list(shape2)
    swapped = False
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        swapped = True

    _dv = len(shape1) - len(shape2)
    shape2 = [1] * _dv + shape2

    out_shape = []
    for i, (shape1_i, shape2_i) in enumerate(zip(shape1, shape2)):
        if not _equal(shape1_i, shape2_i) and \
                (isinstance(shape1_i, int) and shape1_i != 1) \
                and (isinstance(shape2_i, int) and shape2_i != 1):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_013
            error_info['op_name'] = op_name
            error_info['input1_name'] = param_name_input1
            error_info['input2_name'] = param_name_input2
            error_info['input1_shape'] = ",".join(str(i) for i in shape1)
            error_info['input2_shape'] = ",".join(str(i) for i in shape2)
            raise RuntimeError(error_info, "In op[%s], the inputs[%s][%s] could "
                                           "not be broadcast together with shapes[%s][%s]."
                               % (op_name, param_name_input1, param_name_input2,
                                  error_info['input1_shape'], error_info['input2_shape']))
        if operation.in_dynamic():
            _generate_dynamic_output(shape1_i, shape2_i, out_shape, i)
        else:
            out_shape.append(shape1_i if _equal(shape2_i, 1) else shape2_i)

    if swapped:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape


def refine_shapes_for_broadcast(shape1, shape2):
    """
    Fusing the axes for the input shapes
    """
    def _dynamic_refine_shapes_for_broadcast(shape1, shape2):
        """
        Fusing the axes for the input shapes
        """
        def _equals_one(_x):
            if isinstance(_x, tvm.expr.ConstExpr):
                return _x.value == 1
            if isinstance(_x, int):
                return _x == 1
            return False

        def _get_state(_a, _b):
            if _equal(_a, _b):
                return 1
            if _equals_one(_a):
                return 2
            if _equals_one(_b):
                return 3
            return 4

        fused_shape1 = [1]
        fused_shape2 = [1]
        fusion_index = []
        current_index = []
        state = None
        mode = operation.get_context().get("mode")
        if mode != ORIGINAL:
            return shape1, shape2
        for index, (i_a, i_b) in enumerate(zip(shape1, shape2)):
            if _equals_one(i_a) and _equals_one(i_b):
                pass
            elif state is None:
                fused_shape1[-1] *= i_a
                fused_shape2[-1] *= i_b
                state = _get_state(i_a, i_b)
                current_index.append(index)
            elif _get_state(i_a, i_b) == 4:
                fused_shape1.append(i_a)
                fused_shape2.append(i_b)
                state = _get_state(i_a, i_b)
                fusion_index.append(current_index)
                current_index = [index]
            elif state == _get_state(i_a, i_b):
                fused_shape1[-1] *= i_a
                fused_shape2[-1] *= i_b
                current_index.append(index)
            else:
                fused_shape1.append(i_a)
                fused_shape2.append(i_b)
                state = _get_state(i_a, i_b)
                fusion_index.append(current_index)
                current_index = [index]

        fusion_index.append(current_index)
        operation.add_compile_info("_fusion_index", fusion_index)

        return fused_shape1, fused_shape2

    def _const_refine_shapes_for_broadcast(shape1, shape2):
        def _delete_one(shape1, shape2):
            # delete 1 when both 1
            shape1_new = []
            shape2_new = []
            for i, (shape1_i, shape2_i) in enumerate(zip(shape1, shape2)):
                if (shape1_i != shape2_i) or \
                        (shape1_i == shape2_i and shape1_i != 1):
                    shape1_new.append(shape1[i])
                    shape2_new.append(shape2[i])
            if shape1_new == [] and shape2_new == []:
                shape1_new = [1]
                shape2_new = [1]
            return shape1_new, shape2_new

        shape1, shape2 = _delete_one(shape1, shape2)

        fused_shape1 = []
        fused_shape2 = []
        fused_shape1.append(shape1[0])
        fused_shape2.append(shape2[0])
        j = 0
        for i, (shape1_i, shape2_i) in enumerate(zip(shape1, shape2)):
            if i == 0:
                pass
            elif shape1_i == shape2_i and shape1[i - 1] == shape2[i - 1]:
                fused_shape1[j] *= shape1[i]
                fused_shape2[j] *= shape2[i]
            elif shape1_i != shape2_i and shape1[i - 1] != shape2[i - 1] \
                    and (shape1_i == shape1[i - 1] or shape2_i == shape2[i - 1]):
                fused_shape1[j] *= shape1[i]
                fused_shape2[j] *= shape2[i]
            else:
                j += 1
                if i != 0:
                    fused_shape1.append(shape1[i])
                    fused_shape2.append(shape2[i])

        return fused_shape1, fused_shape2

    if fusion_manager.get_build_cfg() == "disable":
        return shape1, shape2

    shape1, shape2 = list(shape1), list(shape2)
    swapped = False
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        swapped = True

    _dv = len(shape1) - len(shape2)
    shape2 = [1] * _dv + shape2

    if operation.in_dynamic():
        operation.add_compile_info("_fusion", 2)
        fused_shape1, fused_shape2 = \
            _dynamic_refine_shapes_for_broadcast(shape1, shape2)
    else:
        fused_shape1, fused_shape2 = \
            _const_refine_shapes_for_broadcast(shape1, shape2)

    if swapped:
        fused_shape1, fused_shape2 = fused_shape2, fused_shape1

    return fused_shape1, fused_shape2


def _equal(expr_a, expr_b):
    """
    :param expr_a:
    :param expr_b:
    :return:
    """
    elements1 = {}
    elements2 = {}

    single_types = (int, float, tvm.expr.Var)
    const_types = (tvm.expr.IntImm,)
    for expr, elements in zip((expr_a, expr_b), (elements1, elements2)):
        if isinstance(expr, single_types):
            elements[expr] = elements.get(expr, 0) + 1
        elif isinstance(expr, const_types):
            elements[expr.value] = elements.get(expr.value, 0) + 1
        elif isinstance(expr, tvm.expr.Expr):
            _parse_expr(expr, elements)
        else:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_025
            error_info['op_name'] = operation.get_context().get_op_type()
            error_info['param_expr'] = expr
            raise RuntimeError(error_info,
                               "In op[%s], unsupported expr: [%s]" % (error_info['op_name'],
                                                                      error_info['param_expr']))

    return elements1 == elements2


def _parse_expr(expr, elements: dict):
    if isinstance(expr, tvm.expr.Mul):
        _parse_mul(expr, elements)
    else:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_025
        error_info['op_name'] = operation.get_context().get_op_type()
        error_info['param_expr'] = expr
        raise RuntimeError(error_info,
                           "In op[%s], unsupported expr: [%s]" % (error_info['op_name'],
                                                                  error_info['param_expr']))


def _parse_mul(expr, elements: dict):
    if not isinstance(expr, tvm.expr.Mul):
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_026
        error_info['op_name'] = operation.get_context().get_op_type()
        error_info['param_expr'] = expr
        raise RuntimeError(error_info,
                           "In op[%s], it is not mul expr: [%s]" % (error_info['op_name'],
                                                                    error_info['param_expr']))

    const_types = (tvm.expr.IntImm,)
    var_types = (tvm.expr.Var,)
    for _x in (expr.a, expr.b):
        if isinstance(_x, const_types):
            elements[_x.value] = elements.get(_x.value, 0) + 1
        elif isinstance(_x, var_types):
            elements[_x] = elements.get(_x, 0) + 1
        else:
            _parse_mul(_x, elements)


def variable_shape(inputs: list, support_broadcast=False):
    """
    :param inputs: all inputs
    :param support_broadcast: whether to support broadcast
    :return:
    """
    def _has_intersection(range0, range1):
        _range0 = list(range0)
        _range1 = list(range1)
        if _range0[1] is None:
            _range0[1] = MAX_UNKOWN_SHAPE_NUM
        if _range1[1] is None:
            _range1[1] = MAX_UNKOWN_SHAPE_NUM
        return max(_range0[0], _range1[0]) <= min(_range0[1], _range1[1])

    def _select(cond, then_case, else_case):
        if cond:
            return then_case
        else:
            return else_case

    def _update_range(shape0, range0, shape1, range1):
        for index in range(len(range0)):
            verify_shape = (shape0[index] != -1 and shape1[index] != -1) or \
                shape0[index] == 1 or shape1[index] == 1
            if verify_shape:
                continue
            range_x = list(range0[index])
            range_y = list(range1[index])
            for j, (_rx, _ry) in enumerate(zip(range_x, range_y)):
                if _rx is None:
                    range_x[j] = MAX_UNKOWN_SHAPE_NUM
                if _ry is None:
                    range_y[j] = MAX_UNKOWN_SHAPE_NUM
            x_const = shape0[index] != -1 and shape1[index] == -1
            y_const = shape0[index] == -1 and shape1[index] != -1
            variable_intersection = _has_intersection(range_x, range_y) and \
                range_x[0] > 1 and range_y[0] > 1
            if x_const:
                range_y = (_select(range_y[0] <= 1, range_y[0], shape0[index]),
                           _select(range_y[1] >= shape0[index], shape0[index], 1))
            elif y_const:
                range_y = (_select(range_x[0] <= 1, range_x[0], shape1[index]),
                           _select(range_x[1] >= shape1[index], shape1[index], 1))
            elif variable_intersection:
                range_x = (max(range_x[0], range_y[0]),
                           min(range_x[1], range_y[1]))
                range_y = range_x
            elif not _has_intersection(range_x, range_y):
                if range_x[0] <= 1:
                    range_x = (1, 1)
                if range_y[0] <= 1:
                    range_y = (1, 1)
            range0[index] = tuple(range_x)
            range1[index] = tuple(range_y)
            if range_x[0] == range_x[1]:
                shape0[index] = range_x[0]
            if range_y[0] == range_y[1]:
                shape1[index] = range_y[0]

    def _fill(_inputs):
        if support_broadcast:
            if len(inputs) != 2:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_027
                error_info['op_name'] = operation.get_context().get_op_type()
                error_info['param_name'] = PARAM_NAME
                raise RuntimeError(error_info,
                                   "In op[%s], only support two inputs for broadcast" % (error_info['op_name']))
            x_0, x_1 = _inputs
            shape0, range0 = list(x_0["shape"]), list(x_0["range"])
            shape1, range1 = list(x_1["shape"]), list(x_1["range"])
            swapped = False
            if len(shape0) < len(shape1):
                shape0, range0, shape1, range1 = shape1, range1, shape0, range0
                swapped = True
            d_v = len(shape0) - len(shape1)
            shape1 = [1] * d_v + shape1
            range1 = [(1, 1)] * d_v + range1
            if swapped:
                shape0, range0, shape1, range1 = shape1, range1, shape0, range0
            _update_range(shape0, range0, shape1, range1)
            return [shape0, shape1], [range0, range1]

        _shapes, _ranges = [], []
        for _input in inputs:
            _shapes.append(_input["shape"])
            _ranges.append(_input["range"])
        return _shapes, _ranges

    def _maybe_broadcast():
        if support_broadcast:
            for _r in ranges:
                if _r[i][0] <= 1:
                    return True
        return False

    def _mode_process():
        if mode == CONST:
            if support_broadcast:
                input1 = inputs[0]["shape"]
                input2 = inputs[1]["shape"]
                const_shape = [a & b for a, b in zip(input1, input2)]
            else:
                const_shape = inputs[0]["shape"]
            operation.get_context().get_current_compute(). \
                add("const_shape", const_shape)
        elif mode == SPECIAL:
            pattern = inputs[0].get("pattern")
            operation.get_context().\
                get_current_compute().add("pattern", pattern)
            if support_broadcast:
                for i, _pattern in enumerate(pattern):
                    if _pattern == COMMON:
                        for j in range(len(shapes)):
                            shapes[j][i] = -77
        elif mode == SPECIAL_SCALAR:
            pattern = inputs[0].get("pattern")
            operation.get_context(). \
                get_current_compute().add("pattern", pattern)

    if len(inputs) < 1:
        return []
    mode = inputs[0].get("mode")
    if mode is None:
        mode = ORIGINAL
    operation.get_context().add("mode", mode)
    current_compute = operation.get_context().get_current_compute()
    if current_compute:
        current_compute.add("mode", mode)
    operation.get_context().add("support_broadcast", support_broadcast)

    shapes, ranges = _fill(inputs)
    _mode_process()

    d_shapes = [[] for _ in shapes]
    for i in range(len(shapes[0])):
        _var = None
        need_two_vars = _maybe_broadcast()
        _suffix = 0
        for d_shape, shape, _range in zip(d_shapes, shapes, ranges):
            if shape[i] == -1 and _range[i][0] == _range[i][1]:
                d_shape.append(_range[i][0])
            elif shape[i] == -1:
                if _var is None or need_two_vars:
                    _var = operation.var("dim_" + str(i) + "_" + str(_suffix),
                                         _range[i])
                d_shape.append(_var)
            elif shape[i] == -77:
                if _var is None:
                    _var = operation.var("dim_" + str(i) + "_" + str(_suffix),
                                         _range[i])
                d_shape.append(_var)
            else:
                d_shape.append(shape[i])
            _suffix += 1

    return d_shapes
