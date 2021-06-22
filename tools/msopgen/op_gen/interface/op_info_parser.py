#!/usr/bin/python3
# coding=utf-8

"""
Function:
This file mainly involves class for parsing operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

from .arg_parser import ArgParser
from .op_info_ir import IROpInfo
from .op_info_tf import TFOpInfo
from .op_info_ir_mindspore import MSIROpInfo
from .op_info_tf_mindspore import MSTFOpInfo
from .op_info_ir_json import JsonIROpInfo
from .op_info_ir_json_mindspore import JsonMSIROpInfo
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
        if argument.input_path.endswith(utils.INPUT_FILE_EXCEL):
            utils.print_warn_log("Excel as input will be removed in future "
                                 "version, it is recommended to use json "
                                 "file as input. ")
            if argument.gen_flag and argument.framework in utils.FMK_MS:
                return MSIROpInfo(argument)
            return IROpInfo(argument)
        if argument.input_path.endswith(utils.INPUT_FILE_JSON):
            if argument.gen_flag and argument.framework in utils.FMK_MS:
                return JsonMSIROpInfo(argument)
            return JsonIROpInfo(argument)
        if argument.gen_flag and argument.framework in utils.FMK_MS:
            return MSTFOpInfo(argument)
        return TFOpInfo(argument)

    def get_op_info(self):
        """
        get op info
        """
        return self.op_info

    @staticmethod
    def get_gen_flag():
        """
        get gen flag
        """
        return None
