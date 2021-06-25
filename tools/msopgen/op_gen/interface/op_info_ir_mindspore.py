#!/usr/bin/env python
# coding=utf-8

"""
Function:
This file mainly involves class for IR operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

from . import utils
from .op_info_ir import IROpInfo


class MSIROpInfo(IROpInfo):
    """
    CLass for IR row for Mindspore.
    """

    @staticmethod
    def _mapping_input_output_type(ir_type, ir_name):
        file_type = utils.INPUT_FILE_XLSX
        return utils.CheckFromConfig().trans_ms_io_dtype(ir_type, ir_name,
                                                         file_type)

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
