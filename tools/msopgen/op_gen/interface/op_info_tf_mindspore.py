#!/usr/bin/env python
# coding=utf-8

"""
Function:
This file mainly involves class for IR operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

from . import utils
from .op_info_tf import TFOpInfo


class MSTFOpInfo(TFOpInfo):
    """
    CLass representing operator info for Mindspore.
    """

    @staticmethod
    def _mapping_input_output_type(tf_type, name):
        return utils.CheckFromConfig().trans_ms_tf_io_dtype(tf_type, name)

    def get_op_path(self):
        """
        get op path
        """
        return self.op_path

    def get_op_type(self):
        """
        get op type
        """
        return self.op_type
