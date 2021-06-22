#! /usr/bin/env python
# coding=utf-8
"""
Function:
AclOpGenerator class. This class mainly implements acl op src code generation.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
Change History: 2020-07-11 file Created
"""
try:
    import os
    import sys
    import json
    from shutil import copytree
    from shutil import copy2
    from shutil import Error
    from . import utils
    from . import op_st_case_info
    from ..template import code_snippet
    from op_test_frame.common import op_status
    from . import dynamic_handle
    from op_test_frame.st.interface.global_config_parser import GlobalConfig as GC
except (ImportError,) as import_error:
    sys.exit(
        "[acl_op_generator]Unable to import module: %s." % str(import_error))


def _get_desc_dic(tmp_dic, key_desc, testcase_struct):
    tmp_dic[key_desc] = []
    input_name_list = []
    for desc_dic in testcase_struct.get(key_desc):
        if desc_dic.get('ori_format') is not None and \
                desc_dic.get('ori_shape') is not None:
            res_desc_dic = {
                'format': desc_dic.get('format'),
                'origin_format': desc_dic.get('ori_format'),
                'type': desc_dic.get('type'),
                'shape': desc_dic.get('shape'),
                'origin_shape': desc_dic.get('ori_shape')}
        else:
            res_desc_dic = {
                'format': desc_dic.get('format'),
                'type': desc_dic.get('type'),
                'shape': desc_dic.get('shape')}
        # Add name field for input*.paramType = optional or dynamic scenarios.
        input_name = desc_dic.get('name')
        if input_name is not None:
            # check the input_desc has the same name.
            if input_name in input_name_list:
                utils.print_error_log(
                    "The input name: (%s) has already exist." % input_name)
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_INPUT_NAME_ERROR)
            input_name_list.append(input_name)
            res_desc_dic.update(
                {'name': desc_dic.get('name')})
        # Add shape_range in acl.json for dynamic shape of operators
        if desc_dic.get(utils.SHAPE_RANGE):
            res_desc_dic.update(
                {utils.SHAPE_RANGE: desc_dic.get(utils.SHAPE_RANGE)})
        tmp_dic[key_desc].append(res_desc_dic)


def _create_acl_op_json_content(testcase_list):
    content = []
    for testcase_struct in testcase_list:
        # init dic with op name
        tmp_dic = {'op': testcase_struct['op']}
        # process input desc
        if "input_desc" in testcase_struct.keys():
            _get_desc_dic(tmp_dic, "input_desc", testcase_struct)
        # process output desc
        if "output_desc" in testcase_struct.keys():
            _get_desc_dic(tmp_dic, "output_desc", testcase_struct)
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


def _write_content_to_file(content, file_path):
    try:
        with os.fdopen(os.open(file_path, utils.WRITE_FLAGS,
                               utils.WRITE_MODES), 'w+') as file_object:
            file_object.write(content)
    except OSError as err:
        utils.print_error_log("Unable to write file(%s): %s." % (file_path,
                                                                 str(err)))
        raise utils.OpTestGenException(utils.OP_TEST_GEN_WRITE_FILE_ERROR)
    utils.print_info_log("Successfully Generated file %s." % file_path)


def _append_content_to_file(content, file_path):
    try:
        with open(file_path, 'a+') as file_object:
            file_object.write(content)
    except OSError as err:
        utils.print_error_log("Unable to write file(%s): %s." % (file_path,
                                                                 str(err)))
        raise utils.OpTestGenException(utils.OP_TEST_GEN_WRITE_FILE_ERROR)
    utils.print_info_log("Successfully appended content to " + file_path)


def _map_to_acl_format_enum(format_list):
    """
    map format to acl format enum
    :param format_list: input format list
    :return: acl format enum list str
    """
    result_str = ""
    acl_format_list = []
    for acl_format in format_list:
        acl_format_list.append(
            "(aclFormat)" + str(GC.instance().white_lists.format_map.get(acl_format)))
    result_str += ", ".join(acl_format_list)
    return result_str


def _create_exact_testcase_content(testcase_struct, device_id):
    input_shape_list = []
    input_data_type_list = []
    input_format_list = []
    for input_desc_dic in testcase_struct['input_desc']:
        # consider dynamic shape scenario
        input_shape = dynamic_handle.replace_shape_to_typical_shape(
            input_desc_dic)
        if input_desc_dic.get('format') in utils.OPTIONAL_TYPE_LIST or \
                input_desc_dic.get('type') == utils.TYPE_UNDEFINED:
            input_desc_dic['shape'] = []
            input_shape_list.append(input_desc_dic['shape'])
            input_data_type_list.append("DT_UNDEFINED")
            input_format_list.append("UNDEFINED")
        else:
            input_shape_list.append(input_shape)
            input_data_type_list.append(input_desc_dic['type'])
            input_format_list.append(input_desc_dic['format'])

    input_shape_data = utils.format_list_str(input_shape_list)
    input_data_type = utils.map_to_acl_datatype_enum(input_data_type_list)
    input_format = _map_to_acl_format_enum(input_format_list)

    input_file_path_list = []
    input_num = 0
    for input_desc_dic in testcase_struct['input_desc']:
        if input_desc_dic.get('format') in utils.OPTIONAL_TYPE_LIST or \
                input_desc_dic.get('type') == utils.TYPE_UNDEFINED:
            input_data_path = ""
            input_file_path_list.append(input_data_path)
            continue
        input_data_name = testcase_struct['case_name'] + '_input_' + str(
            input_num)
        input_data_path = os.path.join("test_data", "data", input_data_name)
        input_file_path_list.append(input_data_path)
        input_num = input_num + 1
    input_file_path = str(
        ', '.join('"' + item + '"' for item in input_file_path_list))

    output_shape_list = []
    output_data_type_list = []
    output_format_list = []
    for output_desc_dic in testcase_struct['output_desc']:
        # consider dynamic shape scenario
        output_shape = dynamic_handle.replace_shape_to_typical_shape(
            output_desc_dic)
        output_shape_list.append(output_shape)
        output_data_type_list.append(output_desc_dic['type'])
        output_format_list.append(output_desc_dic['format'])

    output_shape_data = utils.format_list_str(output_shape_list)
    output_data_type = utils.map_to_acl_datatype_enum(output_data_type_list)
    output_format = _map_to_acl_format_enum(output_format_list)

    output_file_path_list = []
    output_num = 0
    for _ in testcase_struct.get('output_desc'):
        output_data_name = testcase_struct.get('case_name') + '_output_' + str(
            output_num)
        output_data_path = os.path.join("result_files", output_data_name)
        output_file_path_list.append(output_data_path)
        output_num = output_num + 1
    output_file_path = str(
        ', '.join('"' + item + '"' for item in output_file_path_list))

    # do acl attr code generation
    all_attr_code_snippet = ""
    if "attr" in testcase_struct.keys():
        attr_index = 0
        for attr_dic in testcase_struct.get('attr'):
            attr_code_str = "    OpTestAttr attr" + str(attr_index) \
                            + " = {" \
                            + utils.OP_ATTR_TYPE_MAP.get(attr_dic.get('type')) \
                            + ", \"" + attr_dic.get('name') + "\"};\n"
            attr_code_str += "    attr" + str(attr_index) + "." \
                             + utils.ATTR_MEMBER_VAR_MAP.get(attr_dic.get('type')) \
                             + " = " \
                             + utils.create_attr_value_str(attr_dic.get('value')) \
                             + ';\n'
            # deal with the list_list_int attr
            if attr_dic.get('type') == "list_list_int":
                if isinstance(attr_dic.get('value'), list):
                    if isinstance(attr_dic.get('value')[0], list):
                        number_list = list()
                        for num_list in attr_dic.get('value'):
                            number_list.append(len(num_list))
                        num_str = str(number_list).replace('[', '{') \
                            .replace(']', '}')
                        attr_code_str += "    attr" + str(attr_index) + \
                                         ".listIntNumValues" +\
                                         " = " + num_str + ";\n"
            attr_code_str += "    opTestDesc.opAttrVec.push_back(attr" + \
                             str(attr_index) + ");\n"
            all_attr_code_snippet += attr_code_str
            attr_index = attr_index + 1

    testcase_content = code_snippet.TESTCASE_CONTENT.format(
        op_name=testcase_struct.get('op'),
        input_shape_data=input_shape_data,
        input_data_type=input_data_type,
        input_format=input_format,
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        output_shape_data=output_shape_data,
        output_data_type=output_data_type,
        output_format=output_format,
        all_attr_code_snippet=all_attr_code_snippet,
        device_id=device_id,
        testcase_name=testcase_struct.get('case_name'))
    return testcase_content, output_file_path_list


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


class AclOpGenerator:
    """
    Class for generating acl op testcode.
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

    def _copy_entire_template_dir(self):
        ####### [step 1]
        ####### copy entire template dir to output path
        template_path = os.path.realpath(
            os.path.split(os.path.realpath(__file__))[
                0] + utils.SRC_RELATIVE_TEMPLATE_PATH)

        copy_template(template_path, self.output_path)

    def _rewrite_files_for_output_dir(self):
        testcase_cpp_content = ""
        for testcase_struct in self.testcase_list:
            testcase_content, output_paths = \
                _create_exact_testcase_content(testcase_struct, self.device_id)
            testcase_name = testcase_struct['case_name']
            testcase_function_content = code_snippet.TESTCASE_FUNCTION.format(
                op_name=testcase_struct['op'],
                testcase_name=testcase_name,
                testcase_content=testcase_content)
            testcase_cpp_content += testcase_function_content
            # deal with report
            output_abs_paths = [
                os.path.join(self.output_path, 'run', 'out', x + ".bin") for x
                in output_paths]
            case_report = self.report.get_case_report(testcase_name)
            case_report.trace_detail.st_case_info.planned_output_data_paths = \
                output_abs_paths
        output_testcase_cpp_path = self.output_path + \
                                   utils.TESTCASE_CPP_RELATIVE_PATH
        _append_content_to_file(testcase_cpp_content,
                                output_testcase_cpp_path)

        ## f3.prepare acl json content and write file
        acl_json_content = _create_acl_op_json_content(self.testcase_list)
        output_acl_op_json_path = self.output_path + \
                                  utils.ACL_OP_JSON_RELATIVE_PATH
        _write_content_to_file(acl_json_content,
                               output_acl_op_json_path)
        # deal with report
        gen_acl_result = op_st_case_info.OpSTStageResult(
            op_status.SUCCESS,
            "gen_acl_code",
            output_testcase_cpp_path)
        for case_report in self.report.report_list:
            case_report.trace_detail.add_stage_result(gen_acl_result)

    def generate(self):
        """
        Function Description:
            generate acl op c++ files containing info of testcases
        :return:
        """
        self._copy_entire_template_dir()
        self._rewrite_files_for_output_dir()
        utils.print_info_log("acl op test code files for specified "
                             "test cases have been successfully generated.")
