#!/usr/bin/env python
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
        # add is_const in acl_op.json
        utils.ConstInput.add_const_info_in_acl_json(desc_dic, res_desc_dic)
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


def _create_acl_op_json_content(testcase_list, compile_flag):
    content = []
    if compile_flag is not None:
        compile_dic = {'compile_flag': compile_flag}
        content.append(compile_dic)
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
                if attr_dic.get('type') == 'data_type':
                    if attr_dic.get(
                            'value') in utils.ATTR_TYPE_SUPPORT_TYPE_MAP.keys():
                        attr_dic['value'] = utils.ATTR_TYPE_SUPPORT_TYPE_MAP.get(
                            attr_dic.get('value'))
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
        utils.print_error_log("Unable to write file(%s): %s." % (file_path,
                                                                 str(err)))
        raise utils.OpTestGenException(utils.OP_TEST_GEN_WRITE_FILE_ERROR)
    utils.print_info_log("File %s generated successfully." % file_path)


def _append_content_to_file(content, file_path):
    try:
        with open(file_path, 'a+') as file_object:
            file_object.write(content)
    except OSError as err:
        utils.print_error_log("Unable to write file(%s): %s." % (file_path,
                                                                 str(err)))
        raise utils.OpTestGenException(utils.OP_TEST_GEN_WRITE_FILE_ERROR)
    utils.print_info_log("Content appended to %s successfully." % file_path)


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
            "(aclFormat){}".format(str(
                GC.instance().white_lists.format_map.get(acl_format))))
    result_str += ", ".join(acl_format_list)
    return result_str


def _get_input_desc(testcase_struct):
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
        input_data_name = "{}_input_{}".format(testcase_struct['case_name'],
                                               str(input_num))
        input_data_path = os.path.join("test_data", "data", input_data_name)
        input_file_path_list.append(input_data_path)
        input_num = input_num + 1
    input_file_path = str(
        ', '.join('"{}"'.format(item) for item in input_file_path_list))
    return input_shape_data, input_data_type, input_format, input_file_path


def _get_output_desc(testcase_struct):
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
        output_data_name = "{}_output_{}".format(testcase_struct['case_name'],
                                                 str(output_num))
        output_data_path = os.path.join("result_files", output_data_name)
        output_file_path_list.append(output_data_path)
        output_num = output_num + 1
    return output_shape_data, output_data_type, output_format, output_file_path_list


def _replace_dict_list(attr_dic, attr_code_str, attr_index):
    if isinstance(attr_dic.get('value'), list):
        if isinstance(attr_dic.get('value')[0], list):
            number_list = list()
            for num_list in attr_dic.get('value'):
                number_list.append(len(num_list))
            num_str = str(number_list).replace('[', '{') \
                .replace(']', '}')
            attr_code_str += "    attr{}.listIntNumValues = {} ;\n".format(
                str(attr_index), num_str)
    return attr_code_str


def _check_attr_value(attr_dic):
    # deal with the type
    attr_value = attr_dic.get('value')
    if attr_dic.get('type') == 'data_type':
        if attr_value in utils.ATTR_TYPE_SUPPORT_TYPE_MAP.keys():
            dtype = utils.adapt_acl_datatype(attr_value)
            attr_value = "ACL_{}".format(str(dtype).upper())
    return attr_value


def _get_attr_desc(testcase_struct):
    all_attr_code_snippet = ""
    if "attr" in testcase_struct.keys():
        attr_index = 0
        for attr_dic in testcase_struct.get('attr'):
            attr_code_str = "    OpTestAttr attr{attr_index} = " \
                            "{{{type}, \"{name}\"}};\n".format(
                                attr_index=str(attr_index),
                                type=utils.OP_ATTR_TYPE_MAP.get(attr_dic.get('type')),
                                name=attr_dic.get('name'))
            attr_value = _check_attr_value(attr_dic)
            attr_code_str += "    attr{attr_index}.{type} = {value};\n"\
                .format(
                    attr_index=str(attr_index),
                    type=utils.ATTR_MEMBER_VAR_MAP.get(attr_dic.get('type')),
                    value=utils.create_attr_value_str(attr_value))

            # deal with the list_list_int attr
            if attr_dic.get('type') == "list_list_int":
                if isinstance(attr_value, list):
                    attr_code_str = _replace_dict_list(
                        attr_dic, attr_code_str, attr_index)
            attr_code_str += "    opTestDesc.opAttrVec.push_back(" \
                             "attr{attr_index});\n".format(
                                 attr_index=str(attr_index))
            all_attr_code_snippet += attr_code_str
            attr_index = attr_index + 1
    return all_attr_code_snippet


def _create_exact_testcase_content(testcase_struct, device_id):
    # do acl input op description
    input_shape_data, input_data_type, input_format, input_file_path = \
        _get_input_desc(testcase_struct)
    # do acl const_status op description
    const_status = utils.ConstInput.get_acl_const_status(testcase_struct)
    # do acl output op description
    output_shape_data, output_data_type, output_format, output_file_path_list = \
        _get_output_desc(testcase_struct)
    output_file_path = str(
        ', '.join('"{}"'.format(item) for item in output_file_path_list))
    # do acl attr code generation
    all_attr_code_snippet = _get_attr_desc(testcase_struct)

    testcase_content = code_snippet.TESTCASE_CONTENT.format(
        op_name=testcase_struct.get('op'),
        input_shape_data=input_shape_data,
        input_data_type=input_data_type,
        input_format=input_format,
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        is_const=const_status,
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
                    utils.print_error_log("%s is not empty,please settle it "
                                          "and retry ." % dstname)
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

    def __init__(self, testcase_list, path_and_device_id, machine_type,
                 report, compile_flag):
        self.testcase_list = testcase_list
        self.machine_type = machine_type
        self._check_output_path(path_and_device_id[0], testcase_list)
        self.report = report
        self.device_id = path_and_device_id[1]
        self.compile_flag = compile_flag

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
                        "Failed to create %s. %s" % (op_name_path, str(err)))
                    sys.exit(utils.OP_TEST_GEN_INVALID_PATH_ERROR)
            else:
                utils.print_error_log("Specified output path already has \"%s\""
                                      " directory, please delete or move it "
                                      "and retry." % testcase_list[0]['op'])
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
        acl_json_content = _create_acl_op_json_content(self.testcase_list, self.compile_flag)
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

    def get_device_id(self):
        """
        Function Description:
            get device_id
        :return: device_id
        """
        return self.device_id
