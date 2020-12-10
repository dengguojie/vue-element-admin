#!/usr/bin/python3
# coding=utf-8

"""
Function:
This file mainly involves class for IR operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

from . import utils
from .op_info_tf import TFOpInfo
from .op_info_parser import ArgParser

MS_TF_INPUT_OUTPUT_DTYPE_MAP = {
    "bool": "BOOL_Default",
    "DT_BOOL": "BOOL_Default",
    "float16": "F16_Default",
    "float32": "F32_Default",
    "float64": "F64_Default",
    "int8": "I8_Default",
    "DT_INT8": "F8_Default",
    "int16": "F16_Default",
    "DT_INT16": "F16_Default",
    "int32": "F32_Default",
    "DT_INT32": "F32_Default",
    "int64": "F64_Default",
    "DT_INT64": "F64_Default",
    "uint8": "U8_Default",
    "DT_UINT8": "U8_Default",
    "uint16": "F16_Default",
    "DT_UINT16": "F16_Default",
    "uint32": "F32_Default",
    "DT_UINT32": "F32_Default",
    "uint64": "F64_Default",
    "DT_UINT64": "F64_Default",
}


class MSTFOpInfo(TFOpInfo):

    def __init__(self, argument: ArgParser):
        super().__init__(argument)

    @staticmethod
    def _mapping_input_output_type(tf_type, name):
        if tf_type in MS_TF_INPUT_OUTPUT_DTYPE_MAP:
            return MS_TF_INPUT_OUTPUT_DTYPE_MAP.get(tf_type)
        else:
            utils.print_warn_log("The '%s' type '%s' in "
                                 "the .txt file is unsupported. Please "
                                 "check. If you aren't having problems, "
                                 "just ignore the warning."
                                 % (name, tf_type))
        return ""
