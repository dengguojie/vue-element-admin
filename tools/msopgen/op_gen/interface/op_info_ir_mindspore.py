#!/usr/bin/python3
# coding=utf-8

"""
Function:
This file mainly involves class for IR operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

from . import utils
from .arg_parser import ArgParser
from .op_info_ir import IROpInfo

MS_INPUT_OUTPUT_DTYPE_LIST = [
    "None_None", "BOOL_None", "BOOL_Default", "BOOL_5HD", "BOOL_FracZ",
    "BOOL_FracNZ", "BOOL_C1HWNCoC0", "BOOL_NCHW", "BOOL_NHWC", "BOOL_NDHWC",
    "I8_None", "I8_Default", "I8_5HD", "I8_FracZ", "I8_FracNZ", "I8_C1HWNCoC0",
    "I8_NCHW", "I8_NHWC", "I8_HWCN", "I8_NDHWC", "U8_None", "U8_Default",
    "U8_5HD", "U8_FracZ", "U8_FracNZ", "U8_C1HWNCoC0", "U8_NCHW", "U8_NHWC",
    "U8_HWCN", "U8_NDHWC", "I16_None", "I16_Default", "I16_5HD", "I16_FracZ",
    "I16_FracNZ", "I16_C1HWNCoC0", "I16_NCHW", "I16_NHWC", "I16_HWCN",
    "I16_NDHWC", "U16_None", "U16_Default", "U16_5HD", "U16_FracZ",
    "U16_FracNZ", "U16_C1HWNCoC0", "U16_NCHW", "U16_NHWC", "U16_HWCN",
    "U16_NDHWC", "I32_None", "I32_Default", "I32_5HD", "I32_FracZ",
    "I32_FracNZ", "I32_C1HWNCoC0", "I32_NCHW", "I32_NHWC", "I32_HWCN",
    "I32_NDHWC", "U32_None", "U32_Default", "U32_5HD", "U32_FracZ",
    "U32_FracNZ", "U32_C1HWNCoC0", "U32_NCHW", "U32_NHWC", "U32_HWCN",
    "U32_NDHWC", "I64_None", "I64_Default", "I64_5HD", "I64_FracZ",
    "I64_FracNZ", "I64_C1HWNCoC0", "I64_NCHW", "I64_NHWC", "I64_HWCN",
    "I64_NDHWC", "U64_None", "U64_Default", "U64_5HD", "U64_FracZ",
    "U64_FracNZ", "U64_C1HWNCoC0", "U64_NCHW", "U64_NHWC", "U64_HWCN",
    "U64_NDHWC", "F16_None", "F16_Default", "F16_5HD", "F16_FracZ",
    "F16_FracNZ", "F16_C1HWNCoC0", "F16_NCHW", "F16_NHWC", "F16_HWCN",
    "F16_NDHWC", "F16_FracZNLSTM", "F32_None", "F32_Default", "F32_5HD",
    "F32_FracZ", "F32_FracNZ", "F32_C1HWNCoC0", "F32_NCHW", "F32_NHWC",
    "F32_HWCN", "F32_NDHWC", "F32_FracZNLSTM", "F64_None",
    "F64_Default", "F64_5HD", "F64_FracZ", "F64_FracNZ", "F64_C1HWNCoC0",
    "F64_NCHW", "F64_NHWC", "F64_HWCN", "F64_NDHWC"
]


class MSIROpInfo(IROpInfo):

    def __init__(self, argument: ArgParser):
        super().__init__(argument)

    @staticmethod
    def _mapping_input_output_type(ir_type, ir_name):
        if ir_type in MS_INPUT_OUTPUT_DTYPE_LIST:
            return ir_type
        else:
            utils.print_warn_log("The %s 'TypeRange' '%s' in the .xlsx file "
                                 "is unsupported. Please check. If you "
                                 "aren't having problems, "
                                 "just ignore the warning."
                                 % (ir_name, ir_type))
        return ""
