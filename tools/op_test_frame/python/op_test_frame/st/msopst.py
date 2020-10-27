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
    from interface import utils
    from interface.acl_op_runner import AclOpRunner
    from interface.acl_op_generator import AclOpGenerator
    from interface.case_design import CaseDesign
    from interface.case_generator import CaseGenerator
    from interface.data_generator import DataGenerator
    from interface.model_parser import change_shape
    from interface.model_parser import get_shape
except (ImportError,) as import_error:
    sys.exit("[msopst] Unable to import module: %s." % str(import_error))


def _do_mi_cmd(args, cmd):
    output_path = os.path.realpath(args.output_path)
    try:
        if cmd == "get_shape":
            get_shape(args)
        elif cmd == "change_shape":
            change_shape(args)
        elif cmd == "gen":
            # design test case list from json file
            design = CaseDesign(args.input_file, args.case_name)
            case_list = design.design()

            # create acl_op project path and generate acl_op test case code
            acl_op_generator_instance = AclOpGenerator(case_list, output_path,
                                                       True)
            acl_op_generator_instance.generate()

            # generate data
            data_generator = DataGenerator(case_list, output_path, True)
            data_generator.generate()
        else:
            pass
    except utils.OpTestGenException as ex:
        sys.exit(ex.error_info)


def _do_run_cmd(args):
    output_path = os.path.realpath(args.output_path)
    try:
        if args.soc_version == '':
            utils.print_error_log(
                'The value of "soc_version" is empty. Please modify it.')
            sys.exit(utils.OP_TEST_GEN_INVALID_PARAM_ERROR)
        # design test case list from json file
        design = CaseDesign(args.input_file, args.case_name)
        case_list = design.design()

        # create acl_op project path and generate acl_op test case code
        acl_op_generator_instance = AclOpGenerator(case_list,
                                                   output_path, False)
        acl_op_generator_instance.generate()

        # generate data
        data_generator = DataGenerator(case_list, output_path, False)
        data_generator.generate()

        # run acl op
        path = os.path.join(output_path,
                            case_list[0]['op'].replace('/', '_'))
        runner = AclOpRunner(path, args.soc_version)
        runner.process()
    except utils.OpTestGenException as ex:
        sys.exit(ex.error_info)


def _create_parser(create_parser):
    """
    parse create cmd
    :param create_parser:
    """
    create_parser.add_argument(
        "-i", "--input", dest="input_file", default="",
        help="<Required> the input file, .ini file", required=True)
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


def _mi_parser(mi_parser):
    """
    parse mi cmd
    :param mi_parser:
    """
    subparsers = mi_parser.add_subparsers(help='commands')
    get_shape_parser = subparsers.add_parser(
        'get_shape', help='Get shape.')
    change_shape_parser = subparsers.add_parser(
        'change_shape', help='Change shape.')
    gen_parser = subparsers.add_parser(
        'gen', help='Generate test case resource.')

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
    gen_parser.add_argument(
        "-i", "--input", dest="input_file", default="",
        help="<Required> the input file, .json file, ", required=True)
    gen_parser.add_argument(
        "-out", "--output", dest="output_path", default="",
        help="<Optional> the output path", required=False)
    gen_parser.add_argument(
        '-c', "--case_name", dest="case_name", default='all',
        help="<Optional> the case name to run or gen, splits with ',', like "
             "'case0,case1'.", required=False)


def main():
    """
    main function
    :return:
    """
    # parse input argument
    parse = argparse.ArgumentParser()
    subparsers = parse.add_subparsers(help='commands')
    create_parser = subparsers.add_parser(
        'create', help='Create test case json file.')
    run_parser = subparsers.add_parser(
        'run', help='Run the test case on the aihost.')
    mi_parser = subparsers.add_parser(
        'mi', help='Interaction with the IDE.')
    _create_parser(create_parser)
    _run_parser(run_parser)
    _mi_parser(mi_parser)

    if len(sys.argv) <= 1:
        parse.print_usage()
        sys.exit(utils.OP_TEST_GEN_INVALID_PARAM_ERROR)
    args = parse.parse_args(sys.argv[1:])
    if sys.argv[1] == 'create':
        # generate test_case.json
        try:
            generator = CaseGenerator(args)
            generator.generate()
        except utils.OpTestGenException as ex:
            sys.exit(ex.error_info)
    elif sys.argv[1] == 'run':
        _do_run_cmd(args)
    else:
        _do_mi_cmd(args, sys.argv[2])
    sys.exit(utils.OP_TEST_GEN_NONE_ERROR)


if __name__ == "__main__":
    main()
