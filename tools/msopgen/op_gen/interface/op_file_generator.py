#!/usr/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves class for generating operator files.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
try:
    import sys
    from .arg_parser import ArgParser
    from .op_file_aicore import OpFileAiCore
    from .op_file_aicpu import OpFileAiCpu
    from . import utils
except (ImportError,) as import_error:
    sys.exit("[ERROR][op_file_generator]Unable to import module: %s." % str(
        import_error))


class OpFileGenerator:
    """
    CLass for generating operator files
    """

    def __init__(self, argument: ArgParser):
        self.op_file = self._create_op_file(argument)

    @staticmethod
    def _create_op_file(argument: ArgParser):
        if argument.core_type == utils.CoreType.AICORE:
            return OpFileAiCore(argument)
        elif argument.core_type == utils.CoreType.AICPU:
            utils.print_info_log("Start to generator aicpu operator files.")
            return OpFileAiCpu(argument)
        else:
            raise utils.MsOpGenException(
                utils.MS_OP_GEN_UNKNOWN_CORE_TYPE_ERROR)

    def generate(self):
        self.op_file.generate()


