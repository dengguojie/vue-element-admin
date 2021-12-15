#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves class for parsing input arguments.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import ast
import sys
import argparse
from . import utils
from .const_manager import ConstManager


class MsopstArgParser:
    """
    class MsopstArgParser
    """
    def __init__(self):
        # parse input argument
        parse = argparse.ArgumentParser()
        subparsers = parse.add_subparsers(help='commands')
        create_parser = subparsers.add_parser(
            'create', help='Create test case json file.', allow_abbrev=False)
        run_parser = subparsers.add_parser(
            'run', help='Run the test case on the aihost.', allow_abbrev=False)
        mi_parser = subparsers.add_parser(
            'mi', help='Interaction with the IDE.', allow_abbrev=False)
        if len(sys.argv) <= 1:
            parse.print_usage()
            sys.exit(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)
        self._create_parser(create_parser)
        self._mi_parser(mi_parser)
        self._run_parser(run_parser)
        self.input_file = ""
        self.output_path = ""
        self.case_name = ''
        self.model_path = ''
        self.device_id = 0
        self.soc_version = ''
        self.err_thr = [0.01, 0.05]
        self.config_file = ''
        self.report_path = ''
        self.expect_path = ''
        self.result_path = ''
        args = parse.parse_args(sys.argv[1:])
        if sys.argv[1] == 'create':
            self.input_file = args.input_file
            self.model_path = args.model_path
            self.quiet = args.quiet
            self.output_path = args.output_path
        elif sys.argv[1] == 'mi':
            if len(sys.argv) <= 2:
                mi_parser.print_usage()
                sys.exit(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)
            else:
                self._check_mi_args(args)
        else:
            self._check_run_args(args)

    def _check_mi_args(self, args):
        if sys.argv[2] == 'get_shape':
            self.model_path = args.model_path
            self.output_path = args.output_path
        elif sys.argv[2] == 'change_shape':
            self.input_file = args.input_file
            self.model_path = args.model_path
            self.output_path = args.output_path
        if sys.argv[2] == 'gen':
            self.input_file = args.input_file
            self.case_name = args.case_name
            self.output_path = args.output_path
        if sys.argv[2] == 'gen_testcase':
            self.input_file = args.input_file
            self._check_device_id(args.device_id)
            self.output_path = args.output_path
        if sys.argv[2] == 'compare':
            self.report_path = args.report_path
            self._gen_error_threshold(args.error_threshold)
            self.output_path = args.output_path
        if sys.argv[2] == 'compare_by_path':
            self.result_path = args.result_path
            self.expect_path = args.expect_path

    def _check_run_args(self, args):
        self.input_file = args.input_file
        self.case_name = args.case_name
        self._check_soc_version(args.soc_version)
        self._check_device_id(args.device_id)
        self._gen_error_threshold(args.error_threshold)
        self.config_file = args.config_file
        self.output_path = args.output_path

    def _check_soc_version(self, soc_version):
        if soc_version == '':
            utils.print_error_log(
                'The value of "soc_version" is empty. Please modify it.')
        self.soc_version = soc_version

    def _check_device_id(self, device_id):
        if not device_id.isdigit():
            utils.print_error_log(
                'please enter an integer number for device id,'
                ' now is %s.' % device_id)
            sys.exit(ConstManager.OP_TEST_GEN_INVALID_DEVICE_ID_ERROR)
        self.device_id = device_id

    def _gen_error_threshold(self, err_thr):
        if err_thr == "":
            err_thr = [0.01, 0.05]
        else:
            try:
                err_thr = ast.literal_eval(err_thr)
            except ValueError:
                utils.print_error_log(
                    "Error_threshold is unsupported. Example [0.01, 0.01].")
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_ERROR_THRESHOLD_ERROR)
            finally:
                pass
        self._check_error_threshold(err_thr)

    def _check_error_threshold(self, err_thr):
        if isinstance(err_thr, list):
            if len(err_thr) == 0:
                self.err_thr = err_thr
                return
            if len(err_thr) == 2:
                self.err_thr = utils.check_list_float(err_thr, "Error_threshold")
                return
        utils.print_error_log(
            "Error_threshold is unsupported. Example [0.01, 0.01].")
        raise utils.OpTestGenException(
            ConstManager.OP_TEST_GEN_INVALID_ERROR_THRESHOLD_ERROR)

    @staticmethod
    def _create_parser(create_parser):
        """
        parse create cmd
        :param create_parser:
        """
        create_parser.add_argument(
            "-i", "--input", dest="input_file", default="",
            help="<Required> the input file, .ini or .py file", required=True)
        create_parser.add_argument(
            "-out", "--output", dest="output_path", default="",
            help="<Optional> the output path", required=False)
        create_parser.add_argument(
            "-m", "--model", dest="model_path", default="",
            help="<Optional> the model path", required=False)
        create_parser.add_argument(
            '-q', "--quiet", dest="quiet", action="store_true", default=False,
            help="<Optional> quiet mode, skip human-computer interactions",
            required=False)

    @staticmethod
    def _run_parser(run_parser):
        """
        parse run cmd
        :param run_parser:
        """
        run_parser.add_argument(
            "-i", "--input", dest="input_file", default="",
            help="<Required> the input file, .json file, ", required=True)
        run_parser.add_argument(
            '-soc', "--soc_version", dest="soc_version",
            help="<Required> the soc version to run", required=True)
        run_parser.add_argument(
            "-out", "--output", dest="output_path", default="",
            help="<Optional> the output path", required=False)
        run_parser.add_argument(
            '-c', "--case_name", dest="case_name", default='all',
            help="<Optional> the case name to run or gen, splits with ',', "
                 "like 'case0,case1'.", required=False)
        run_parser.add_argument(
            '-d', "--device_id", dest="device_id", default="0",
            help="<Optional> input device id, default is 0.",
            required=False)
        run_parser.add_argument(
            '-conf', "--config_file", dest="config_file", default="",
            help="<Optional> config_file, msopst advance config file.",
            required=False)
        run_parser.add_argument(
            '-err_thr', "--error_threshold", dest="error_threshold",
            default="[0.01, 0.05]",
            help="<Optional> error_threshold, Error threshold of result"
                 "comparison, like [0.001, 0.001].",
            required=False)

    def _mi_parser(self, mi_parser):
        """
        parse mi cmd
        :param mi_parser:
        """
        subparsers = mi_parser.add_subparsers(help='commands')
        get_shape_parser = subparsers.add_parser(
            'get_shape', help='Get shape.')
        change_shape_parser = subparsers.add_parser(
            'change_shape', help='Change shape.')
        gen_json_parser = subparsers.add_parser(
            'gen', help='Generate acl_op.json for the ascend operator.')
        gen_testcase_parser = subparsers.add_parser(
            'gen_testcase', help='Generate testcase resource and data.')
        compare_parser = subparsers.add_parser(
            'compare', help='Compare result data with expect output.')
        compare_by_path_parser = subparsers.add_parser(
            'compare_by_path', help='Compare result data with expect output by st '
                                    'report.')

        # get shape parser
        get_shape_parser.add_argument(
            "-m", "--model", dest="model_path", default="",
            help="<Required> the model path", required=True)
        get_shape_parser.add_argument(
            "-out", "--output", dest="output_path", default="",
            help="<Optional> the output path", required=False)

        # change shape parser
        change_shape_parser.add_argument(
            "-m", "--model", dest="model_path", default="",
            help="<Required> the model path", required=True)
        change_shape_parser.add_argument(
            "-i", "--input", dest="input_file", default="",
            help="<Required> the input file, .json file", required=True)
        change_shape_parser.add_argument(
            "-out", "--output", dest="output_path", default="",
            help="<Optional> the output path", required=False)

        # gen parser
        self._mi_gen_parser(gen_json_parser, gen_testcase_parser)

        # compare parse
        self._mi_compare_parser(compare_parser, compare_by_path_parser)

    @staticmethod
    def _mi_gen_parser(gen_json_parser, gen_testcase_parser):
        # gen acl_op.json
        gen_json_parser.add_argument(
            "-i", "--input", dest="input_file", default="",
            help="<Required> the input file, .json file, ", required=True)
        gen_json_parser.add_argument(
            "-out", "--output", dest="output_path", default="",
            help="<Optional> the output path", required=False)
        gen_json_parser.add_argument(
            '-c', "--case_name", dest="case_name", default='all',
            help="<Optional> the case name to run or gen, splits with ',', like "
                 "'case0,case1'.", required=False)
        # gen_testcase_parser
        gen_testcase_parser.add_argument(
            "-i", "--input", dest="input_file", default="",
            help="<Required> the input file, st_report.json file, ", required=True)
        gen_testcase_parser.add_argument(
            "-out", "--output", dest="output_path", default="",
            help="<Optional> the output path", required=False)
        gen_testcase_parser.add_argument(
            '-d', "--device_id", dest="device_id", default="0",
            help="<Optional> input device id, default is 0.", required=False)

    @staticmethod
    def _mi_compare_parser(compare_parser, compare_by_path_parser):
        compare_parser.add_argument(
            "-i", "--report path", dest="report_path", default="",
            help="<Required> the st report file path",
            required=False)
        compare_parser.add_argument(
            "-out", "--output", dest="output_path", default="",
            help="<Optional> the output path", required=False)
        compare_parser.add_argument(
            '-err_thr', "--error_threshold", dest="error_threshold",
            default="[0.01, 0.05]",
            help="<Optional> error_threshold, Error threshold of result"
                 "comparison, like [0.001, 0.001].",
            required=False)

        # compare_by_path parse
        compare_by_path_parser.add_argument(
            "-result", "--result path", dest="result_path", default="",
            help="<Required> the result file path, ", required=True)
        compare_by_path_parser.add_argument(
            "-expect", "--expect path", dest="expect_path", default="",
            help="<Required> the expect result file path",
            required=True)

    def get_input_file(self):
        """
        get input file
        :return: input file
        """
        return self.input_file

    def get_output_path(self):
        """
        get output path
        :return: output path
        """
        return self.output_path
