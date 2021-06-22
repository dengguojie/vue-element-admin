#! /usr/bin/env python
# coding=utf-8
"""
Function:
MsOpGenerator class.
This class mainly implements mindspore op src code generation.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved Â© 2020
Change History: 2020-07-11 file Created
"""
try:
    import os
    import sys
    import numpy as np
    import json
    import importlib
    from shutil import copytree
    from shutil import copy2
    from shutil import Error
    from op_test_frame.st.interface import utils
    from op_test_frame.st.interface import op_st_case_info
    from op_test_frame.st.template import code_snippet
    from op_test_frame.common import op_status
except (ImportError,) as import_error:
    sys.exit(
        "[ms_op_generator]Unable to import module: %s." % str(import_error))


# pylint: disable=too-many-arguments
def _create_ms_op_json_content(testcase_list):
    content = []
    for testcase_struct in testcase_list:
        # init dic with op name
        tmp_dic = {'op': testcase_struct['op']}

        # process input desc
        if "input_desc" in testcase_struct.keys():
            tmp_dic['input_desc'] = []
            for input_desc_input_dic in testcase_struct['input_desc']:
                input_desc_dic = {'type': input_desc_input_dic['type'],
                                  'shape': input_desc_input_dic['shape']}
                tmp_dic['input_desc'].append(input_desc_dic)

        # process output desc
        if "output_desc" in testcase_struct.keys():
            tmp_dic['output_desc'] = []
            for output_desc_input_dic in testcase_struct['output_desc']:
                output_desc_dic = {
                    'type': output_desc_input_dic['type'],
                    'shape': output_desc_input_dic['shape']}
                tmp_dic['output_desc'].append(output_desc_dic)

        # process attr
        if "attr" in testcase_struct.keys():
            tmp_dic['attr'] = []
            for attr_dic in testcase_struct['attr']:
                tmp_dic['attr'].append(attr_dic)

        # only append non-repetitive json struct
        if tmp_dic not in content:
            content.append(tmp_dic)

    try:
        return str(json.dumps(content, sort_keys=True, indent=2))
    except TypeError:
        utils.print_error_log("")
    return None


def _write_content_to_file(content, file_path):
    try:
        with os.fdopen(os.open(file_path, utils.WRITE_FLAGS,
                               utils.WRITE_MODES), 'w+') as file_object:
            file_object.write(content)
    except OSError as err:
        utils.print_error_log("Unable to write file(%s): %s." % file_path
                              % str(err))
        raise utils.OpTestGenException(utils.OP_TEST_GEN_WRITE_FILE_ERROR)
    utils.print_info_log("Successfully Generated file %s." % file_path)


def _append_content_to_file(content, file_path):
    try:
        with open(file_path, 'a+') as file_object:
            file_object.write(content)
    except OSError as err:
        utils.print_error_log("Unable to write file(%s): %s." % file_path
                              % str(err))
        raise utils.OpTestGenException(utils.OP_TEST_GEN_WRITE_FILE_ERROR)
    utils.print_info_log("Successfully appended content to " + file_path)


def copy_template(src, dst):
    """
    copy template from src dir to dst dir
    :param src: template src dir
    :param dst: dst dir
    """
    names = os.listdir(src)
    errors = []
    for name in names:
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.isdir(srcname):
                if os.path.isdir(dstname) and os.listdir(dstname):
                    utils.print_error_log(
                        dstname + " is not empty,please settle it and retry .")
                    sys.exit()
                copytree(srcname, dstname)
            else:
                copy2(srcname, dstname)
        except (IOError, OSError) as why:
            errors.append((srcname, dstname, str(why)))
    if errors:
        raise Error(errors)


class MsOpGenerator:
    """
    Class for generating mindspore op testcode.
    """

    def __init__(self, testcase_list, output_path, device_id, machine_type,
                 report):
        self.testcase_list = testcase_list
        self.machine_type = machine_type
        self._check_output_path(output_path, testcase_list)
        self.report = report
        self.device_id = device_id

    def _check_output_path(self, output_path, testcase_list):
        formalized_path = os.path.realpath(output_path)
        utils.check_path_valid(formalized_path, True)
        if self.machine_type:
            self.output_path = output_path
        else:
            op_name_path = os.path.join(output_path, testcase_list[0]['op'])
            if not os.path.exists(op_name_path):
                try:
                    os.makedirs(op_name_path, mode=0o750)
                except OSError as err:
                    utils.print_error_log(
                        "Failed to create \"" + op_name_path + "\". " + str(
                            err))
                    sys.exit(utils.OP_TEST_GEN_INVALID_PATH_ERROR)
            else:
                utils.print_error_log(
                    "Specified output path already has \"" + testcase_list[0][
                        'op'] +
                    "\" directory, please delete or move it and retry.")
            self.output_path = op_name_path

    def _mkdir_output_dir(self):
        ####### [step 1]
        ####### create output_path dir
        run_dir = os.path.join(self.output_path, 'run', 'out', 'test_data')
        utils.make_dirs(run_dir)
        src_dir = os.path.join(self.output_path, 'src')
        utils.make_dirs(src_dir)

    @staticmethod
    def _get_mindspore_input_param_type(op_name_lower):
        op_name_impl = "{}_impl".format(op_name_lower)
        op_name_op_info = "{}_op_info".format(op_name_lower)
        ms_input_param_type_list = []
        try:
            params = importlib.import_module(op_name_impl)
            mindspore_ops_info = getattr(params, op_name_op_info)
        except Exception as error:
            utils.print_error_log(
                'Failed to import "%s" to get operation information of "%s",'
                ' the reason is %s.' % (op_name_impl, op_name_op_info, error))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        ms_ops_input_list = mindspore_ops_info.get('inputs')
        # get input param type
        for input_list in ms_ops_input_list:
            ms_input_param_type_list.append(input_list.get('param_type'))
        return ms_input_param_type_list

    @staticmethod
    def _get_no_attr_input_args(ms_param_type_list,
                                count_input, input_name_list):
        param_type_list_length = len(ms_param_type_list)
        # input param type is required.
        if utils.DYNAMIC_INPUT not in ms_param_type_list:
            input_args = ','.join(input_name_list)
            input_name = input_args
            return input_args, input_name

        # input param type has only dynamic input.
        if param_type_list_length == 1:
            return utils.DYNAMIC_INPUT_ARGS, utils.DYNAMIC_INPUT_NAME
        # input param type have dynamic input
        # and other input(optional or required).
        else:
            dynamic_input_count = count_input - param_type_list_length
            input_index = 0
            input_name_tensor_list = []
            for input_param_type in ms_param_type_list:
                if input_param_type == utils.DYNAMIC_INPUT:
                    dynamic_input_name_str = ','.join(
                        input_name_list[input_index:
                                        input_index + dynamic_input_count])
                    input_name_tuple = "({})".format(dynamic_input_name_str)
                    input_name_tensor_list.append(input_name_tuple)
                    input_index = input_index + dynamic_input_count + 1
                else:
                    input_name_tensor_list.append(input_name_list[input_index])
                    input_index += 1
            input_args = ','.join(input_name_list)
            input_name = ','.join(input_name_tensor_list)
            return input_args, input_name

    def _rewrite_files_for_output_dir(self):
        testcase_test_net_content = ''
        op_name = self.testcase_list[0]['op']
        op_name_lower = utils.fix_name_lower_with_under(op_name)
        ms_param_type_list = self._get_mindspore_input_param_type(op_name_lower)
        testcase_test_net_content += code_snippet. \
            TESTCASE_IMPORT_CONTENT.format(
            import_op=op_name_lower,
            op_name=op_name,
            device_id=self.device_id)
        # generate test_sub_case function
        testcase_test_net_func_content = ''
        for testcase_struct in self.testcase_list:
            testcase_name = testcase_struct['case_name']
            output_paths = []
            output_num = 0
            for _ in testcase_struct['input_desc']:
                output_data_name = testcase_struct['case_name'] + '_input_' \
                                   + str(output_num)
                output_data_path = os.path.join("test_data/data/",
                                                output_data_name)
                output_paths.append(output_data_path)
                output_num = output_num + 1

            # deal with report
            last_filedir = os.path.split(self.output_path)[1]
            output_abs_paths = [
                os.path.join(last_filedir, 'run', 'out', x + ".bin") for x
                in output_paths]
            case_report = self.report.get_case_report(testcase_name)
            case_report.trace_detail.st_case_info.planned_output_data_paths = \
                output_abs_paths
            count_input = 1
            inputs = ''
            tensor_list = []
            input_name_list = []
            for input_desc in testcase_struct['input_desc']:
                input_name = 'input{}'.format(count_input)
                tensor_desc = code_snippet.TESTCASE_TEST_TENSOR.format(
                    input_name=input_name)
                tensor_list.append(tensor_desc)
                bin_file = output_abs_paths[count_input - 1]
                ms_type = input_desc['type']
                inputs += code_snippet.TESTCASE_TEST_NET_INPUT.format(
                    input_name=input_name,
                    file=bin_file,
                    np_type=ms_type,
                    op_shape=input_desc.get('shape'))
                input_name_list.append(input_name)
                count_input += 1
            tensor_total = ','.join(tensor_list)
            input_name_join = ','.join(input_name_list)
            count_output = 1
            outputs = ''
            for output_desc in testcase_struct['output_desc']:
                output_name = 'output{}'.format(count_output)
                ms_type = getattr(np, output_desc.get('type'))
                outputs += code_snippet.TESTCASE_TEST_NET_OUTPUT.format(
                    output_name=output_name,
                    op_lower=op_name_lower,
                    tensor=tensor_total,
                    np_type=ms_type)
                count_output += 1
            testcase_test_def_name = testcase_name[0].lower() \
                                     + testcase_name[1:]

            testcase_test_net_func_content += code_snippet.TESTCASE_TEST_NET \
                .format(
                subcase=testcase_test_def_name,
                op_lower=op_name_lower,
                inputs=inputs,
                outputs=outputs)

        if not self.testcase_list[0].get('attr') or \
                utils.DYNAMIC_INPUT in ms_param_type_list:
            input_args, input_name = self._get_no_attr_input_args(
                ms_param_type_list, count_input, input_name_list)
            testcase_test_net_func_content += code_snippet \
                .TESTCASE_CLASS_CONTENT_NO_ATTR.format(
                op_lower=op_name_lower,
                op_name=op_name,
                input_args=input_args,
                inputs=input_name_join)
        else:
            attr_value_list = []
            for attr_info in self.testcase_list[0]['attr']:
                attr_value_list.append(str(attr_info['value']))
            attr_value = ", ".join(list(attr_value_list))
            attr_construct = code_snippet \
                .TESTCASE_CLASS_CONTENT_WITH_ATTR_CONSTRUCT.format(
                inputs=input_name_join,
                op_lower=op_name_lower)
            testcase_test_net_content += code_snippet \
                .TESTCASE_CLASS_CONTENT_WITH_ATTR.format(
                op_name=op_name,
                op_lower=op_name_lower,
                attr_value=attr_value,
                attr_constrct=attr_construct)
        testcase_test_net_content += testcase_test_net_func_content
        output_testcase_py_path = self.output_path + \
                                  utils.TESTCASE_PY_RELATIVE_PATH \
                                      .format(op_name=op_name_lower)

        _append_content_to_file(testcase_test_net_content,
                                output_testcase_py_path)
        output_pytest_ini_path = self.output_path \
                                 + utils.PYTEST_INI_RELATIVE_PATH
        # create pytest.ini format
        pytest_ini_content = code_snippet.PYTEST_INI_CONTEN
        _append_content_to_file(pytest_ini_content,
                                output_pytest_ini_path)

        # deal with report
        gen_ms_result = op_st_case_info.OpSTStageResult(
            op_status.SUCCESS,
            "gen_ms_code",
            output_testcase_py_path)
        for case_report in self.report.report_list:
            case_report.trace_detail.add_stage_result(gen_ms_result)

    def generate(self):
        """
        Function Description:
            generate mindspore op python files containing info of testcases
        :return:
        """
        self._mkdir_output_dir()
        self._rewrite_files_for_output_dir()
        utils.print_info_log("mindspore operator test code files for specified"
                             " test cases have been successfully generated.")
