#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves class for generating operator files.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
from op_gen.interface.arg_parser import ArgParser
from op_gen.interface.op_file_aicore import OpFileAiCore
from op_gen.interface.op_file_aicpu import OpFileAiCpu
from op_gen.interface.op_file_mindspore_aicore import OpFileMindSporeAiCore
from op_gen.interface.op_file_mindspore_aicpu import OpFileMindSporeAICPU
from op_gen.interface import utils
from op_gen.interface.const_manager import ConstManager


class OpFileGenerator:
    """
    CLass for generating operator files
    """

    def __init__(self: any, argument: ArgParser) -> None:
        self.op_file = self._create_op_file(argument)

    @staticmethod
    def _create_op_file(argument: ArgParser) -> any:
        if argument.framework in ConstManager.FMK_MS:
            if argument.core_type == ConstManager.AICORE:
                utils.print_info_log(
                    "Start to generate MindSpore AI core operator files.")
                return OpFileMindSporeAiCore(argument)

            if argument.core_type == ConstManager.AICPU:
                utils.print_info_log(
                    "Start to generate MindSpore AI CPU operator files.")
                return OpFileMindSporeAICPU(argument)

        if argument.core_type == ConstManager.AICORE:
            utils.print_info_log(
                "Start to generate AI Core operator files.")
            return OpFileAiCore(argument)
        utils.print_info_log("Start to generate AI CPU operator files.")
        return OpFileAiCpu(argument)

    def generate(self: any) -> None:
        """
        generate op files
        """
        self.op_file.generate()

    def get_op_file(self: any) -> any:
        """
        get op files
        """
        return self.op_file
