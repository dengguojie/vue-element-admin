#!/usr/bin/python3
# coding=utf-8

"""
Function:
This file mainly involves class for parsing operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
try:
    import sys
    from .arg_parser import ArgParser
    from .op_info_ir import IROpInfo
    from .op_info_tf import TFOpInfo
except (ImportError,) as import_error:
    sys.exit("[ERROR][op_info_parser]Unable to import module: %s." % str(
        import_error))


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
            return IROpInfo(argument)
        return TFOpInfo(argument.input_path)
