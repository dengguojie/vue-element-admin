#!/usr/bin/env python
# coding=utf-8
"""
Function:
MsOpGenerator class.
This class mainly implements mindspore op src code generation.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved Â© 2020
Change History: 2020-07-11 file Created
"""

import os
import json
import importlib

import numpy as np

from op_test_frame.st.interface import utils
from op_test_frame.st.interface.const_manager import ConstManager
from op_test_frame.st.interface import op_st_case_info
from op_test_frame.st.template.code_snippet import CodeTemplate
from op_test_frame.common import op_status


def _add_op_info_in_tmp_dict(testcase_struct, tmp_dict, op_key):
    if op_key in testcase_struct.keys():
        tmp_dict[op_key] = []
        for input_desc_input_dic in testcase_struct.get(op_key):
            op_desc_dict = {'type': input_desc_input_dic.get('type'),
                            'shape': input_desc_input_dic.get('shape')}
            tmp_dict.get(op_key).append(op_desc_dict)


def _add_attr_info_in_tmp_dict(testcase_struct, tmp_dict, op_key):
    if op_key in testcase_struct.keys():
        tmp_dict[op_key] = []
        for attr_dic in testcase_struct.get(op_key):
            tmp_dict.get(op_key).append(attr_dic)


def _create_ms_op_json_content(testcase_list):
    content = []
    for testcase_struct in testcase_list:
        # init dic with op name
        tmp_dic = {'op': testcase_struct.get('op')}
        # process input desc
        _add_op_info_in_tmp_dict(testcase_struct, tmp_dic, ConstManager.INPUT_DESC)
        # process output desc
        _add_op_info_in_tmp_dict(testcase_struct, tmp_dic, ConstManager.OUTPUT_DESC)
        # process attr
        _add_attr_info_in_tmp_dict(testcase_struct, tmp_dic, ConstManager.ATTR)
        # only append non-repetitive json struct
        if tmp_dic not in content:
            content.append(tmp_dic)

    try:
        return str(json.dumps(content, sort_keys=True, indent=2))
    except TypeError:
        utils.print_error_log("")
    finally:
        pass
    return ""


def _append_content_to_file(content, file_path):
    try:
        with os.fdopen(os.open(file_path, ConstManager.WRITE_FLAGS,
                               ConstManager.WRITE_MODES), 'a+') as file_object:
            file_object.write(content)
    except OSError as err:
        utils.print_error_log("Unable to write file(%s): %s." % file_path
                              % str(err))
        raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)
    finally:
        pass
    utils.print_info_log("Content appended to %s successfully." % file_path)


class MsOpGenerator:
    """
    Class for generating mindspore op testcode.
    """

    def __init__(self, testcase_list, path_and_device_id, machine_type,
                 report):
        self.testcase_list = testcase_list
        self.machine_type = machine_type
        self.output_path = utils.check_output_path(
            path_and_device_id[0], testcase_list, self.machine_type)
        self.report = report
        self.device_id = path_and_device_id[1]

    @staticmethod
    def _get_ms_ops_info(op_name_impl, op_name_op_info):
        params = importlib.import_module(op_name_impl)
        return getattr(params, op_name_op_info)

    def _mkdir_output_dir(self):
        ####### [step 1]
        ####### create output_path dir
        run_dir = os.path.join(self.output_path, 'run', 'out', 'test_data')
        utils.make_dirs(run_dir)
        src_dir = os.path.join(self.output_path, 'src')
        utils.make_dirs(src_dir)

    def _get_mindspore_input_param_type(self):
        _, op_name_lower = self._get_op_name()
        op_name_impl = "{}_impl".format(op_name_lower)
        op_name_op_info = "{}_op_info".format(op_name_lower)
        ms_input_param_type_list = []
        try:
            mindspore_ops_info = self._get_ms_ops_info(op_name_impl, op_name_op_info)
        except Exception as error:
            utils.print_error_log(
                'Failed to import "%s" to get operation information of "%s",'
                ' the reason is %s.' % (op_name_impl, op_name_op_info, error))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        finally:
            pass
        ms_ops_input_list = mindspore_ops_info.get('inputs')
        # get input param type
        for input_list in ms_ops_input_list:
            ms_input_param_type_list.append(input_list.get('param_type'))
        return ms_input_param_type_list

    @staticmethod
    def _get_dynamic_input_info(ms_param_type_list,
                                count_input, input_name_list):
        param_type_list_length = len(ms_param_type_list)
        # input param type has only dynamic input.
        if param_type_list_length == 1:
            return ConstManager.DYNAMIC_INPUT_ARGS, ConstManager.DYNAMIC_INPUT_NAME
        # input param type have dynamic input
        # and other input(optional or required).
        dynamic_input_count = count_input - param_type_list_length
        input_index = 0
        input_name_tensor_list = []
        for input_param_type in ms_param_type_list:
            if input_param_type == ConstManager.DYNAMIC_INPUT:
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

    def _get_no_attr_input_args(self, ms_param_type_list,
                                count_input, input_name_list):
        # input param type is required.
        if ConstManager.DYNAMIC_INPUT not in ms_param_type_list:
            input_args = ','.join(input_name_list)
            input_name = input_args
            return input_args, input_name
        input_args, input_name = self._get_dynamic_input_info(ms_param_type_list,
                                     count_input, input_name_list)
        return input_args, input_name

    def _mkdir_input_data_path(self, testcase_struct):
        input_paths = []
        testcase_name = testcase_struct['case_name']
        input_count = 0
        for _ in testcase_struct['input_desc']:
            input_data_name = '{}_input_{}'.format(
                testcase_struct['case_name'], str(input_count))
            input_data_path = os.path.join("test_data/data/",
                                           input_data_name)
            input_paths.append(input_data_path)
            input_count += 1

        # deal with report
        path_name = os.path.split(self.output_path)[1]
        input_data_abs_paths_tuple = (os.path.join(path_name, 'run', 'out', x + ".bin") for x in input_paths)
        input_data_abs_paths = list(input_data_abs_paths_tuple)
        case_report = self.report.get_case_report(testcase_name)
        case_report.trace_detail.st_case_info.planned_output_data_paths = input_data_abs_paths
        return input_data_abs_paths

    def _generate_output_content(self, testcase_struct, tensor_list,
                                 inputs_str):
        _, op_name_lower = self._get_op_name()
        testcase_test_func_content = ''
        testcase_name = testcase_struct.get('case_name')
        outputs_str = ''
        output_count = 1
        for output_desc in testcase_struct.get('output_desc'):
            output_name = 'output{}'.format(output_count)
            data_type = getattr(np, output_desc.get('type'))
            outputs_str += CodeTemplate.TESTCASE_TEST_NET_OUTPUT.format(
                output_name=output_name,
                op_lower=op_name_lower,
                tensor=','.join(tensor_list),
                np_type=data_type)
            output_count += 1
        testcase_test_func_content += CodeTemplate.TESTCASE_TEST_NET \
            .format(subcase=testcase_name[0].lower() + testcase_name[1:],
                    op_lower=op_name_lower,
                    inputs=inputs_str,
                    outputs=outputs_str)
        return testcase_test_func_content

    def _generate_function_content(self, testcase_struct, input_data_abs_paths):
        input_count = 1
        inputs_str = ''
        tensor_content_list = []
        input_name_list = []
        for input_desc in testcase_struct.get('input_desc'):
            input_name = 'input{}'.format(input_count)
            tensor_content_list.append(CodeTemplate.TESTCASE_TEST_TENSOR.format(
                input_name=input_name))
            inputs_str += CodeTemplate.TESTCASE_TEST_NET_INPUT.format(
                input_name=input_name,
                file=input_data_abs_paths[input_count - 1],
                np_type=input_desc.get('type'),
                op_shape=input_desc.get('shape'))
            input_name_list.append(input_name)
            input_count += 1
        func_content = self._generate_output_content(testcase_struct,
                                                     tensor_content_list,
                                                     inputs_str)
        return func_content, input_count, input_name_list

    def _generate_net_content(self, input_count, input_name_list,
                              ms_param_type_list):
        testcase_net_content = ''
        inputs_str = ','.join(input_name_list)
        op_name, op_name_lower = self._get_op_name()
        if not self.testcase_list[0].get('attr') or \
                ConstManager.DYNAMIC_INPUT in ms_param_type_list:
            input_args, inputs_str = self._get_no_attr_input_args(
                ms_param_type_list, input_count, input_name_list)
            testcase_net_content += \
                CodeTemplate.TESTCASE_CLASS_CONTENT_NO_ATTR.format(
                    op_lower=op_name_lower,
                    op_name=op_name,
                    input_args=input_args,
                    inputs=inputs_str)
        else:
            attr_value_list = []
            for attr_info in self.testcase_list[0].get('attr'):
                attr_value_list.append(str(attr_info.get('value')))
            attr_value = ", ".join(list(attr_value_list))
            attr_construct = \
                CodeTemplate.TESTCASE_CLASS_CONTENT_WITH_ATTR_CONSTRUCT.format(
                    inputs=inputs_str,
                    op_lower=op_name_lower)
            testcase_net_content += \
                CodeTemplate.TESTCASE_CLASS_CONTENT_WITH_ATTR.format(
                    op_name=op_name,
                    op_lower=op_name_lower,
                    attr_value=attr_value,
                    attr_constrct=attr_construct)
        return testcase_net_content

    def _generate_test_function(self):
        ms_param_type_list = self._get_mindspore_input_param_type()
        testcase_py_func_content = ''
        func_content = ''
        input_count = 1
        input_name_list = []
        for testcase_struct in self.testcase_list:
            # create input data path
            input_data_abs_paths = self._mkdir_input_data_path(testcase_struct)
            # generate function content with input and output info of operator
            sub_case_func_content, input_count, input_name_list = \
                self._generate_function_content(testcase_struct,
                                                input_data_abs_paths)
            func_content += sub_case_func_content
        testcase_py_func_content += self._generate_net_content(
            input_count, input_name_list, ms_param_type_list)
        testcase_py_func_content += func_content
        return testcase_py_func_content

    def _generate_test_template_content(self):
        """
        generate test_op python file template
        """
        # generate import module template
        op_name, op_name_lower = self._get_op_name()
        mindspore_test_py_content = ''
        mindspore_test_py_content += \
            CodeTemplate.TESTCASE_IMPORT_CONTENT.format(
                import_op=op_name_lower,
                op_name=op_name,
                device_id=self.device_id)

        # generate test sub case function contents
        mindspore_test_py_content += self._generate_test_function()
        return mindspore_test_py_content

    def _get_op_name(self):
        op_name = self.testcase_list[0].get('op')
        op_name_lower = utils.fix_name_lower_with_under(op_name)
        return op_name, op_name_lower

    def _rewrite_files_for_output_dir(self):
        # generate test_op.py template content.
        mindspore_test_py_content = self._generate_test_template_content()
        # create test_op.py path
        _, op_name_lower = self._get_op_name()
        output_testcase_py_path = \
            self.output_path + ConstManager.TESTCASE_PY_RELATIVE_PATH.format(
                op_name=op_name_lower)
        # write test_op.py to path
        _append_content_to_file(mindspore_test_py_content,
                                output_testcase_py_path)
        # create pytest.ini for format pytest result
        output_pytest_ini_path = \
            self.output_path + ConstManager.PYTEST_INI_RELATIVE_PATH
        pytest_ini_content = CodeTemplate.PYTEST_INI_CONTEN
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
        utils.print_info_log("MindSpore operator test code files for specified"
                             " test cases generated successfully.")

    def get_device_id(self):
        """
        Function Description: get device id
        :return:
        """
        return self.device_id
