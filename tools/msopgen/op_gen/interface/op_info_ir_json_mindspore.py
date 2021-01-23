#!/usr/bin/python3
# coding=utf-8

"""
Function:
This file mainly involves class for IR JSON for mindspore operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

from . import utils
from .arg_parser import ArgParser
from .op_info_ir_json import JsonIROpInfo


class JsonMSIROpInfo(JsonIROpInfo):

    def __init__(self, argument: ArgParser):
        super().__init__(argument)

    @staticmethod
    def _mapping_input_output_type(ir_type, ir_name):
        if ir_type in utils.MS_INPUT_OUTPUT_DTYPE_LIST:
            return ir_type
        else:
            utils.print_warn_log("The %s 'TypeRange' '%s' in the .json file "
                                 "is unsupported. Please check. If you "
                                 "aren't having problems, "
                                 "just ignore the warning."
                                 % (ir_name, ir_type))
        return ""
