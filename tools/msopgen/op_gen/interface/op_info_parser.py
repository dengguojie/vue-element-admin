#!/usr/bin/python3
# coding=utf-8

"""
Function:
This file mainly involves class for parsing operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import sys

from .arg_parser import ArgParser
from .op_info_ir import IROpInfo
from .op_info_tf import TFOpInfo
from .op_info_ir_mindspore import MSIROpInfo
from .op_info_tf_mindspore import MSTFOpInfo
from . import utils


class OpInfoParser:
    """
    CLass for parsing operator info
    """

    def __init__(self, argument: ArgParser):
        self.op_info = self._create_op_info(argument)
        self.op_info.parse()

    @staticmethod
    def _create_op_info(argument: ArgParser):
        if argument.input_path.endswith(".xlsx") \
                or argument.input_path.endswith(".xls"):
            if argument.gen_flag and argument.framework in utils.FMK_MS:
                return MSIROpInfo(argument)
            else:
                return IROpInfo(argument)

        if argument.gen_flag and argument.framework in utils.FMK_MS:
            return MSTFOpInfo(argument)
        return TFOpInfo(argument)
