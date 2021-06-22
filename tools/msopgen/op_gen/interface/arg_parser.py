#!/usr/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves class for parsing input arguments.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
try:
    import os
    import sys
    import argparse
    from . import utils
except (ImportError,) as import_error:
    sys.exit("[ERROR][arg_parser]Unable to import module: %s." % str(
        import_error))


class ArgParser:
    """
    CLass for parsing input arguments
    """

    def __init__(self):
        parse = argparse.ArgumentParser()
        subparsers = parse.add_subparsers(help='commands')
        mi_parser = subparsers.add_parser(
            utils.INPUT_ARGUMENT_CMD_MI, help='Machine interface for IDE.',
            allow_abbrev=False)
        gen_parser = subparsers.add_parser(
            utils.INPUT_ARGUMENT_CMD_GEN, help=' Generator operator project.',
            allow_abbrev=False)
        self._gen_parse_add_arguments(gen_parser)
        self._mi_parse_add_arguments(mi_parser)
        args = parse.parse_args(sys.argv[1:])
        if len(sys.argv) <= 1:
            parse.print_usage()
            sys.exit(utils.MS_OP_GEN_INVALID_PARAM_ERROR)
        if sys.argv[1] == utils.INPUT_ARGUMENT_CMD_GEN:
            self.gen_flag = True
            self.input_path = ""
            self.framework = ""
            self.compute_unit = ""
            self.output_path = ""
            self.mode = 0
            self.core_type = -1
            self.op_type = ""
            self._check_input_path(args.input)
            self._check_framework(args.framework)
            self._check_compute_unit_valid(args.compute_unit)
            self._check_output_path(args.output)
            self._check_mode_valid(args.mode)
            self._check_op_type_valid(args.operator)
            return
        if sys.argv[1] == utils.INPUT_ARGUMENT_CMD_MI:
            self.gen_flag = False
            if len(sys.argv) <= 2:
                mi_parser.print_usage()
                sys.exit(utils.MS_OP_GEN_INVALID_PARAM_ERROR)
            self.mi_cmd = sys.argv[2]
            self._check_mi_cmd_param(args)

    @staticmethod
    def _mi_parse_add_arguments(mi_parser):
        mi_subparsers = mi_parser.add_subparsers(help='commands')
        query_parser = mi_subparsers.add_parser(
            utils.INPUT_ARGUMENT_CMD_MI_QUERY, help='Query the operators from '
                                                    'the \"Op\" sheet in IR '
                                                    'excel.')
        # query Op parser and add arguments
        query_parser.add_argument("-i", "--input",
                                  dest="input",
                                  default="",
                                  help="<Required> the input file, %s file, "
                                       "which needs to be existed and "
                                       "readable." % (utils.MI_VALID_TYPE,),
                                  required=True)
        query_parser.add_argument("-out", "--output",
                                  dest="output",
                                  default=os.getcwd(),
                                  help="<Optional> output path",
                                  required=False)

    @staticmethod
    def _gen_parse_add_arguments(gen_parser):
        gen_parser.add_argument("-i", "--input",
                                dest="input",
                                default="",
                                help="<Required> the input file, %s file, "
                                     "which needs to be existed and readable." % (utils.GEN_VALID_TYPE,),
                                required=True)
        gen_parser.add_argument("-f", "--framework",
                                dest="framework",
                                default="TF",
                                help="<Required> op framework type(case "
                                     "insensitive) tf, tensorflow, caffe, "
                                     "ms, mindspore, onnx, pytorch.",
                                required=True)
        gen_parser.add_argument("-c", "--compute_unit",
                                dest="compute_unit",
                                default="",
                                help="<Required> compute unit, of which the "
                                     "format should be like "
                                     "ai_core-ascend310 or aicpu.",
                                required=True)
        gen_parser.add_argument("-out", "--output",
                                dest="output",
                                default="",
                                help="<Optional> output path.",
                                required=False)
        gen_parser.add_argument("-m", "--mode",
                                dest="mode",
                                default='0',
                                help="<Optional> 0:default, generator "
                                     "project;1: add a new operator.",
                                required=False)
        gen_parser.add_argument("-op", "--operator",
                                dest="operator",
                                default="",
                                help="<Optional> op type in IR excel.",
                                required=False)

    def _check_mi_cmd_param(self, args):
        if self.mi_cmd == utils.INPUT_ARGUMENT_CMD_MI_QUERY:
            if not args.input.endswith(utils.MI_VALID_TYPE):
                utils.print_error_log(
                    'The file "%s" is invalid, only supports %s file. Please '
                    'modify it.' % (args.input, utils.MI_VALID_TYPE))
                raise utils.MsOpGenException(
                    utils.MS_OP_GEN_INVALID_PATH_ERROR)
            utils.check_path_valid(args.input)
            self.input_path = args.input
            self._check_output_path(args.output)

    def _check_op_type_valid(self, args_operator):
        if args_operator != '':
            utils.check_name_valid(args_operator)
            self.op_type = args_operator

    def _check_framework(self, args_framework):
        lower_args_framework = args_framework.lower()
        if lower_args_framework in utils.FMK_LIST:
            self.framework = lower_args_framework
        else:
            utils.print_error_log(
                "Unsupported framework type: " + args_framework)
            sys.exit(utils.MS_OP_GEN_CONFIG_UNSUPPORTED_FMK_TYPE_ERROR)

    def _check_output_path(self, args_output_path):
        args_output_path = os.path.realpath(args_output_path)
        if not os.path.exists(args_output_path):
            utils.make_dirs(args_output_path)
        if os.path.exists(args_output_path) and os.access(args_output_path,
                                                          os.W_OK):
            self.output_path = args_output_path
        else:
            utils.print_error_log(args_output_path +
                                  " does not exist or is not permitted to "
                                  "write.")
            sys.exit(utils.MS_OP_GEN_CONFIG_INVALID_OUTPUT_PATH_ERROR)

    def _check_input_path(self, args_input):
        if not args_input.endswith(utils.GEN_VALID_TYPE):
            utils.print_error_log(
                'The file "%s" is invalid, only supports %s file. Please '
                'modify it.' % (args_input, utils.GEN_VALID_TYPE))
            raise utils.MsOpGenException(utils.MS_OP_GEN_INVALID_PATH_ERROR)
        args_op_info = os.path.realpath(args_input)
        if os.path.isfile(args_op_info) and os.access(args_op_info, os.R_OK):
            self.input_path = args_op_info
        else:
            utils.print_error_log("Input path: " + args_input +
                                  " error, please check if it is an existed "
                                  "and readable file.")
            sys.exit(utils.MS_OP_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)

    def _init_core_type(self, unit_parse_list, type_list, core_type):
        if unit_parse_list[0].lower() in type_list:
            if self.core_type == -1:
                self.core_type = core_type
            else:
                if self.core_type != core_type:
                    utils.print_error_log("Invalid compute unit "
                                          "format. Only support one "
                                          "core type.")
                    raise utils.MsOpGenException(
                        utils.MS_OP_GEN_CONFIG_INVALID_COMPUTE_UNIT_ERROR)
        else:
            self._print_compute_unit_invalid_log()

    def _check_compute_unit_valid(self, args_compute_unit):
        compute_unit_list = args_compute_unit.split(",")
        for unit in compute_unit_list:
            unit_parse_list = unit.split("-", 1)
            if len(unit_parse_list) == 1:
                self._init_core_type(unit_parse_list,
                                     utils.AICPU_CORE_TYPE_LIST,
                                     utils.AICPU)
            elif len(unit_parse_list) == 2:
                self._init_core_type(unit_parse_list,
                                     utils.AICORE_CORE_TYPE_LIST,
                                     utils.AICORE)
            else:
                self._print_compute_unit_invalid_log()
        self.compute_unit = compute_unit_list
        return utils.MS_OP_GEN_NONE_ERROR

    @staticmethod
    def _print_compute_unit_invalid_log():
        utils.print_error_log("Invalid compute unit format. "
                              "Please check the format of input "
                              "compute unit is ${core_type}-${"
                              "unit_type}, like ai_core-ascend310 or aicpu.")
        raise utils.MsOpGenException(
            utils.MS_OP_GEN_CONFIG_INVALID_COMPUTE_UNIT_ERROR)

    def _check_mode_valid(self, mode):
        if str(mode) not in utils.GEN_MODE_LIST:
            utils.print_error_log('Unsupported mode: %s, only %s supported. '
                                  'Please check the input mode.' %
                                  (str(mode), ','.join(utils.GEN_MODE_LIST)))
            raise utils.MsOpGenException(
                utils.MS_OP_GEN_CONFIG_UNSUPPORTED_MODE_ERROR)
        self.mode = mode
        return utils.MS_OP_GEN_NONE_ERROR

    def get_gen_flag(self):
        """
        get gen flag
        """
        return self.gen_flag

    @staticmethod
    def get_gen_result():
        """
        get gen result
        """
        return None
