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


class MSTFOpInfo(TFOpInfo):

    def __init__(self, argument: ArgParser):
        super().__init__(argument)

    @staticmethod
    def _mapping_input_output_type(tf_type, name):
        return utils.CheckFromConfig().trans_ms_tf_io_dtype(tf_type, name)