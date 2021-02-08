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


class MSIROpInfo(IROpInfo):

    def __init__(self, argument: ArgParser):
        super().__init__(argument)

    @staticmethod
    def _mapping_input_output_type(ir_type, ir_name):
        file_type = utils.INPUT_FILE_XLSX
        return utils.CheckFromConfig().trans_ms_io_dtype(ir_type, ir_name,
                                                         file_type)
