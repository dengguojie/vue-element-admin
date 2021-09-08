#!/usr/bin/env python
# coding=utf-8

"""
Function:
This file mainly involves class for IR JSON for mindspore operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

from . import utils
from .op_info_ir_json import JsonIROpInfo
from .const_manager import ConstManager


class JsonMSIROpInfo(JsonIROpInfo):
    """
    CLass for IR OP Info from Json for Mindspore.
    """

    @staticmethod
    def _mapping_input_output_type(ir_type, ir_name):
        file_type = ConstManager.INPUT_FILE_JSON
        return utils.CheckFromConfig().trans_ms_io_dtype(ir_type, ir_name,
                                                         file_type)

    @staticmethod
    def _init_op_format(input_output_map, prefix, input_output_name,
                        ir_type_list):
        op_format = ",".join("ND" for _ in ir_type_list)
        op_format = op_format.split(",")
        return op_format

    def get_op_path(self):
        """
        get op path
        """
        return self.op_path

    def get_gen_flag(self):
        """
        get gen flag
        """
        return self.gen_flag
