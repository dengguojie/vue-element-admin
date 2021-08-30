#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves main function of op generation module.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import sys
from op_gen.interface.arg_parser import ArgParser
from op_gen.interface.op_file_generator import OpFileGenerator
from op_gen.interface.op_info_parser import OpInfoParser
from op_gen.interface import utils


def _do_gen_cmd(argument):
    try:
        op_file_generator = OpFileGenerator(argument)
        if not op_file_generator.op_file:
            utils.print_error_log(
                "AICPU is not supported by MindSpore operators.")
            raise utils.MsOpGenException(utils.MS_OP_GEN_NONE_ERROR)
        op_file_generator.generate()
    except utils.MsOpGenException as ex:
        sys.exit(ex.error_info)
    finally:
        pass


def _do_mi_cmd(argument):
    try:
        if argument.mi_cmd == utils.INPUT_ARGUMENT_CMD_MI_QUERY:
            OpInfoParser(argument)
    except utils.MsOpGenException as ex:
        sys.exit(ex.error_info)
    finally:
        pass


def main():
    """main function"""
    # 1.parse input argument and check arguments valid
    try:
        argument = ArgParser()
    except utils.MsOpGenException as ex:
        sys.exit(ex.error_info)
    finally:
        pass
    # 2.generate file, according to gen and mi
    if argument.gen_flag:
        _do_gen_cmd(argument)
    else:
        _do_mi_cmd(argument)
    utils.print_info_log("Generation completed.")
    sys.exit(utils.MS_OP_GEN_NONE_ERROR)


if __name__ == "__main__":
    main()
