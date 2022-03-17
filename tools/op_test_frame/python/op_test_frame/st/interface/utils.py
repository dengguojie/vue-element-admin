#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves the common function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import os.path
import subprocess
import sys
import time
import re
import json

import numpy as np

from op_test_frame.st.interface.data_generator import DataGenerator
from op_test_frame.st.interface.const_manager import ConstManager


def create_attr_list_str(attr_value):
    """
    create attribute exact value string based on list type
    :param attr_value: attr value variable
    :return: res_str
    """
    if not attr_value:
        return "{}"
    if isinstance(attr_value[0], list):
        res_str = "{"
        num_list_str_list = []
        for num_list in attr_value:
            num_list_str_list.append(
                "new int64_t[" + str(len(num_list)) +
                "]{" + str(", ".join(str(item) for item in num_list)) + "}")
        res_str += str(
            ", ".join(str(item) for item in num_list_str_list)) + "}"
    elif isinstance(attr_value[0], str):
        res_str = "{" + str(
            ", ".join('"' + item + '"' for item in attr_value)) + "}"
    elif isinstance(attr_value[0], bool):
        bool_int_list = []
        for bool_val in attr_value:
            # 0 for false, others for true
            if bool_val:
                bool_int_list.append(1)
            else:
                bool_int_list.append(0)
        res_str = "{" + str(
            ", ".join(str(item) for item in bool_int_list)) + "}"
    else:
        res_str = "{" + str(
            ", ".join(str(item) for item in attr_value)) + "}"
    return res_str


def create_attr_value_str(attr_value):
    """
    create attribute exact value string based on type
    :param attr_value: attr value variable
    :return: none
    """
    if isinstance(attr_value, list):
        res_str = create_attr_list_str(attr_value)
    elif isinstance(attr_value, str):
        # string
        if attr_value.startswith("ACL"):
            res_str = attr_value
        else:
            res_str = '"' + attr_value + '"'
    elif isinstance(attr_value, bool):
        # bool
        if attr_value:
            res_str = str(1)
        else:
            res_str = str(0)
    else:
        # num
        res_str = str(attr_value)
    return res_str


def format_list_str(list_input):
    """
    format list string of json type to cpp type
    convert [[1, 64], [1, 64]] to {1, 64}, {1, 64}
    :param list_input:
    :return: list string of cpp type
    """
    return str(list_input)[1:-1].replace("[", "{").replace("]", "}")


def format_dict_to_list(dict_input):
    """
    format list type of head file to python type
    convert {1, 64} to [1, 64]
    :param dict_input:
    :return: list string of python type
    """
    return str(dict_input).replace("{", "[").replace("}", "]")


def map_type_to_expect_type(dtype):
    """
    mapping float to float32, because of <class 'numpy.float'> is
    <class 'numpy.float64'>.
    :param dtype: input dtype
    :return: dtype
    """
    dtype_map = {"float": "float32"}
    if dtype in dtype_map:
        dtype = dtype_map.get(dtype)
    return dtype


def adapt_acl_datatype(dtype):
    """
    To adapt aclDataType, mapping float32 to float.
    :param dtype: input dtype
    :return: dtype
    """
    dtype_map = {"float32": "float"}
    if dtype in dtype_map:
        dtype = dtype_map.get(dtype)
    return dtype


def map_to_acl_datatype_enum(dtype_list):
    """
    map datatype to acl datatype enum
    :param dtype_list: input dtype list
    :return: acl datatype enum list str
    """
    result_str = ""
    acl_dtype_list = []
    for dtype in dtype_list:
        dtype = adapt_acl_datatype(dtype)
        acl_dtype_list.append("ACL_{}".format(str(dtype).upper()))
    result_str += ", ".join(acl_dtype_list)
    return result_str


class OpTestGenException(Exception):
    """
    The class for Op Gen Exception
    """

    def __init__(self, error_info):
        super(OpTestGenException, self).__init__(error_info)
        self.error_info = error_info


def _print_log(level, msg):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(int(time.time())))
    pid = os.getpid()
    print(current_time + " (" + str(pid) + ") - [" + level + "] " + msg)
    sys.stdout.flush()


def print_error_log(error_msg_info):
    """
    print error log
    @param error_msg: the error message
    @return: none
    """
    _print_log("ERROR", error_msg_info)


def print_warn_log(warn_msg):
    """
    print warn log
    @param warn_msg: the warn message
    @return: none
    """
    _print_log("WARNING", warn_msg)


def print_info_log(info_msg):
    """
    print info log
    @param info_msg: the info message
    @return: none
    """
    _print_log("INFO", info_msg)


class CallingCounter:
    """
    Class CallingCounter
    """
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

    def get_count(self):
        """
        get self.count
        """
        return self.count

    def get_func(self):
        """
        get self.func
        """
        return self.func


@CallingCounter
def print_step_log(step_msg):
    """
    print step log
    @param step_msg: the info message
    @return: none
    """
    step_count = str(print_step_log.count)
    msg_format = "".join(["[STEP", step_count, "] ", step_msg])
    _print_log("INFO", msg_format)


def check_path_valid(path, isdir=False):
    """
    Function Description:
    check path valid
    Parameter:
    path: the path to check
    isdir: the path is dir or file
    """
    if path == "":
        print_error_log("The path is null. Please check whether the argument is valid.")
        raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)
    path = os.path.realpath(path)
    if isdir and not os.path.exists(path):
        try:
            os.makedirs(path, mode=0o700)
        except OSError as ex:
            print_error_log(
                'Failed to create {}. Please check the path permission or '
                'disk space. {} '.format(path, str(ex)))
            raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)
        finally:
            pass
    if not os.path.exists(path):
        print_error_log('The path {} does not exist. Please check whether '
                        'the path exists.'.format(path))
        raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log('The path {} does not have permission to read.'
                        ' Please check the path permission.'.format(path))
        raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    if isdir and not os.access(path, os.W_OK):
        print_error_log('The path {} does not have permission to write.'
                        ' Please check the path permission.'.format(path))
        raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    if isdir:
        if not os.path.isdir(path):
            print_error_log('The path {} is not a directory.'
                            ' Please check the path.'.format(path))
            raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log('The path {} is not a file.'
                            ' Please check the path.'.format(path))
            raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)


def check_output_path(output_path, testcase_list, machine_type):
    """
    Function Description: check output path
    Parameter:
    output_path: output path
    testcase_list: testcase list
    machine_type: machine type
    Return: output path
    """
    formalized_path = os.path.realpath(output_path)
    check_path_valid(formalized_path, True)
    if not machine_type:
        op_name_path = os.path.join(output_path, testcase_list[0].get('op'))
        if not os.path.exists(op_name_path):
            try:
                os.makedirs(op_name_path, mode=0o750)
            except OSError as err:
                print_error_log(
                    "Failed to create %s. %s" % (op_name_path, str(err)))
                sys.exit(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)
            finally:
                pass
        else:
            src_path = os.path.join(op_name_path, 'src')
            if os.path.exists(src_path):
                print_error_log("Specified output path already has %s directory, "
                                "please delete or move it and retry." % testcase_list[0]['op'])
                sys.exit(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)
        output_path = op_name_path
    return output_path


def check_name_valid(name):
    """
    Function Description: check name valid
    Parameter:
    name: the name to check
    Return Value:
    VectorComparisonErrorCode
    """
    if name == "":
        print_error_log("The input name is \"\"")
        return ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR
    name_pattern = re.compile(ConstManager.SUPPORT_PATH_PATTERN)
    match = name_pattern.match(name)
    if match is None:
        return ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR
    return ConstManager.OP_TEST_GEN_NONE_ERROR


def get_content_from_double_quotes(line):
    """
    Function Description:
    get content list between two double quotes
    Parameter:
    path: content line containing double quotes
    Return Value:
    VectorComparisonErrorCode
    """
    pattern = re.compile('"(.*)"')
    match = pattern.findall(line)
    if match:
        return match
    print_error_log("(\" \") format error. Please check.")
    sys.exit(ConstManager.OP_TEST_GEN_CONFIG_OP_DEFINE_ERROR)


def check_value_valid(fe_type, value, name, prefix=""):
    """
    Function Description:
    check path valid
    Parameter:
    fe_type: the type of attr
    value: the value of attr
    name: the name of attr
    prefix: the type of attr prefix
    """
    value_type = int
    if fe_type == 'int':
        value_type = int
    elif fe_type == 'bool':
        value_type = bool
    elif fe_type == 'string':
        value_type = str
    elif fe_type == 'float':
        value_type = float
    elif fe_type == 'data_type':
        if value not in ConstManager.ATTR_TYPE_SUPPORT_TYPE_MAP.keys():
            print_error_log(
                'The value (%s) is invalid. The value of "%s" for "attr" '
                'only supports in %s. Please modify it.'
                % (value, name, ConstManager.ATTR_TYPE_SUPPORT_TYPE_MAP.keys()))
            raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        value_type = str
    elif fe_type == 'list_int':
        if not isinstance(value, list):
            print_error_log(
                'The value (%s) is invalid. The value of "%s" for "attr" '
                'only supports list_list_int. Please modify it.'
                % (value, name))
            raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        if len(value) == 0:
            print_error_log(
                'The value (%s) is empty. The value of "%s" for "attr" '
                'only supports list_list_int. Please modify it.'
                % (value, name))
            raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        for item in value:
            check_value_valid('int', item, name, 'list_list_')
        return

    if not isinstance(value, value_type):
        print_error_log(
            'The value (%s) is invalid. The value of "%s" for "attr" only '
            'supports %s%s. Please modify it.'
            % (value, name, prefix, fe_type))
        raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)


def check_attr_value_valid(attr):
    """
    check attr value valid
    :param attr: the attr to check
    :return:
    """
    attr_type = attr.get('type')
    if attr_type.startswith('list_'):
        if not isinstance(attr.get('value'), list):
            print_error_log(
                'The value (%s) is invalid. The value of "%s" for "attr" '
                'only supports %s. Please modify it.'
                % (attr.get('value'), attr.get('name'), attr_type))
            raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        for value in attr.get('value'):
            check_value_valid(
                attr_type[len('list_'):], value, attr.get('name'), 'list_')
    else:
        check_value_valid(attr_type, attr.get('value'), attr.get('name'))


def load_json_file(json_path):
    """
    load json file to json object
    :param json_path: the json path
    :return: the json object
    """

    try:
        with open(json_path, 'r') as input_file:
            try:
                return json.load(input_file)
            except Exception as ex:
                print_error_log(
                    'Failed to load json file %s. Please modify it. %s'
                    % (json_path, str(ex)))
                raise OpTestGenException(ConstManager.OP_TEST_GEN_PARSE_JSON_FILE_ERROR)
            finally:
                pass
    except IOError as io_error:
        print_error_log(
            'Failed to open file %s. %s' % (json_path, str(io_error)))
        raise OpTestGenException(ConstManager.OP_TEST_GEN_OPEN_FILE_ERROR)
    finally:
        pass


def read_file(op_file):
    """
    read content_txt from op_file
    :param op_file:op file
    :return: None
    """
    try:
        with open(op_file) as op_file_object:
            content_txt = op_file_object.read()
            return content_txt
    except IOError as io_err:
        print_error_log(
            'Failed to open file %s. %s, please check it.' % (op_file, str(io_err)))
        raise OpTestGenException(ConstManager.OP_TEST_GEN_OPEN_FILE_ERROR)
    finally:
        pass


def write_json_file(json_path, content):
    """
    write  content to json file
    :param content:
    :param json_path: the json path
    """
    try:
        if os.path.exists(json_path) and os.path.isfile(json_path):
            os.remove(json_path)
        with os.fdopen(os.open(json_path, ConstManager.WRITE_FLAGS,
                               ConstManager.WRITE_MODES), 'w') as file_object:
            file_object.write(
                json.dumps(content, sort_keys=False, indent=4))
    except IOError as io_error:
        print_error_log(
            'Failed to generate file %s. %s' % (json_path, str(io_error)))
        raise OpTestGenException(ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)
    finally:
        pass
    print_info_log(
        "Generate file %s successfully." % os.path.realpath(json_path))


def make_dirs(op_dir):
    """
    make dirs
    :param op_dir:dirs
    """
    try:
        if not os.path.isdir(op_dir) or not os.path.exists(op_dir):
            os.makedirs(op_dir, ConstManager.FOLDER_MASK)
    except OSError as err:
        print_error_log("Unable to make dir: %s." % str(err))
        raise OpTestGenException(ConstManager.OP_TEST_GEN_MAKE_DIRS_ERROR)
    finally:
        pass


def fix_name_lower_with_under(name):
    """
    change name to lower_with_under style,
    eg: "Abc" -> abc
    eg: "AbcDef" -> abc_def
    eg: "ABCDef" -> abc_def
    eg: "Abc2DEf" -> abc2d_ef
    eg: "Abc2DEF" -> abc2def
    eg: "ABC2dEF" -> abc2d_ef
    :param name: op type/input/out_put/attribute name to be fix
    :return: name has been fixed
    """
    fix_name = ""
    for index, name_str in enumerate(name):
        if index == 0:
            fix_name += name_str.lower()
        elif name_str.isupper() and index != len(name) - 1:
            if (index != len(name) - 1 and name[index + 1].islower()) or \
                    (index != 0 and name[index - 1].islower()):
                # If a capital letter is surrounded by lowercase letters, convert to "_" + lowercase letter
                # In addition, all are converted to lowercase letters
                # eg: "Abc2DEf"  ->   "abc2d_ef"
                fix_name += "_{}".format(name_str.lower())
            else:
                fix_name += name_str.lower()
        else:
            fix_name += name_str.lower()
    return fix_name


def check_list_float(input_list, param_name):
    """
    Check whether the list consists of floating point numbers
    eg: [0.01, 0.1]
    :param input_list: list to be check
    :param param_name: list param name
    :return: input_list
    """
    support_flag = True
    for i in input_list:
        if not isinstance(i, float) or i > 1 or i < 0:
            support_flag = False
            break
    if support_flag:
        return input_list
    print_error_log("%s is unsupported. Example [0.01, 0.01]." % str(param_name))
    raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_ERROR_THRESHOLD_ERROR)


def add_new_key_to_cross_list(tensor, cross_key_list):
    """
    Function: Add new key in cross key list.
    return:
    """
    new_key_list = [ConstManager.SHAPE_RANGE, ConstManager.VALUE, ConstManager.IS_CONST]
    for key in new_key_list:
        if tensor.get(key):
            cross_key_list.append(key)


def execute_command(cmd):
    """
    Execute command
    """
    print_info_log('Execute command: %s' % cmd)
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.strip()
        if line:
            print(line)
    if process.returncode != 0:
        print_error_log('Failed to execute command: %s' % cmd)
        raise OpTestGenException(
            ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)


class ScanFile:
    """
    The class for scanning path to get subdirectories.
    """
    def __init__(self, directory, first_prefix=None, second_prefix=None):
        self.directory = directory
        self.first_prefix = first_prefix
        self.second_prefix = second_prefix

    def _check_second_prefix_dir(self, each_file_path, files_list):
        dir_info = os.path.split(each_file_path)
        if self.first_prefix:
            if dir_info[1].startswith(self.second_prefix):
                files_list.append(each_file_path)

    def _get_files_list(self, file_path, files_list):
        all_files = os.listdir(file_path)
        for each_file in all_files:
            each_file_path = os.path.join(file_path, each_file)
            if os.path.isdir(each_file_path):
                self._check_second_prefix_dir(each_file_path, files_list)
        return files_list

    def scan_subdirs(self):
        """
        scan the specified path to get the list of subdirectories.
        :return: list of subdirectories
        """
        files_list = []
        if os.path.exists(self.directory):
            all_files = os.listdir(self.directory)
            for each_file in all_files:
                file_path = os.path.join(self.directory, each_file)
                files_list = self._get_files_list(file_path, files_list)
        else:
            print_error_log("The scanned directory does not exist: %s"
                            % self.directory)
        return files_list

    def get_prefix(self):
        """
        get prefix
        :return: prefix
        """
        return self.prefix


class ConstInput:
    """
    The class for dealing with const input.
    attr: is_const is a bool type.
    """
    def __init__(self, is_const=None):
        self.is_const = is_const

    def _check_is_const(self):
        if self.is_const is None:
            return False
        if not isinstance(self.is_const, bool):
            print_error_log('The value of "is_const" only support bool '
                            'type: true or false.')
            raise OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        return True

    def deal_with_const(self, input_desc, for_fuzz):
        """
        Function: update is_const field in input_desc dict.
        return:
        """
        if self._check_is_const():
            if for_fuzz:
                input_desc.update({ConstManager.IS_CONST: self.is_const})
            else:
                input_desc.update({ConstManager.IS_CONST: [self.is_const]})

    @staticmethod
    def add_const_info_in_acl_json(desc_dict, res_desc_dic, output_path, case_name, index):
        """
        Function: check whether there is an is_const field in the desc_dict,
        and then check whether there is a value field. Otherwise, use the
        data distribution and value range to generate constant value.
        Finally, deposit is_const and constant value to acl_op.json.
        Return:
        desc_dict-> input or output information description
        res_desc_dic-> a dict for format,shape,type information
        output_path-> output path
        case_name-> case name
        index-> input/output index
        """
        input_shape = desc_dict.get('shape')
        dtype = desc_dict.get('type')
        if desc_dict.get(ConstManager.IS_CONST) is True:
            case_value = desc_dict.get(ConstManager.VALUE)
            if case_value:
                if isinstance(case_value, str):
                    np_type = getattr(np, dtype)
                    data = np.fromfile(case_value, np_type)
                    const_value = data.tolist()
                else:
                    const_value = np.array(
                        desc_dict.get(ConstManager.VALUE)).flatten().tolist()
            else:
                # generate const value with data_distribute
                range_min, range_max = desc_dict.get(ConstManager.VALUE_RANGE)
                data = DataGenerator.gen_data(
                    input_shape, range_min, range_max, dtype,
                    desc_dict.get(ConstManager.DATA_DISTRIBUTE))
                const_value = ConstInput._generate_const_value(data, output_path, case_name, index)
            const_value_dict = {
                ConstManager.IS_CONST: desc_dict.get(ConstManager.IS_CONST),
                ConstManager.CONST_VALUE: const_value}
            res_desc_dic.update(const_value_dict)

    @staticmethod
    def get_acl_const_status(testcase_struct):
        """
        Function: check input whether is a constant, and generate constant status
        for inputConst variable in testcase.cpp.
        Return: const_status
        """
        input_const_list = []
        const_status = ""
        acl_const_list = []
        for input_desc_dic in testcase_struct.get(ConstManager.INPUT_DESC):
            if input_desc_dic.get(ConstManager.IS_CONST):
                input_const_list.append(
                    str(input_desc_dic.get(ConstManager.IS_CONST)).lower())
            else:
                input_const_list.append(ConstManager.FALSE)
        for acl_const in input_const_list:
            acl_const_list.append(acl_const)
        const_status += ", ".join(acl_const_list)
        return const_status

    @staticmethod
    def _generate_const_value(data, output_path, case_name, index):
        const_data = data.flatten()
        # generate bin file to save const value and used by data generator module
        output_path = os.path.join(output_path, 'run', 'out', 'test_data', 'data')
        make_dirs(output_path)
        file_path = os.path.join(output_path,
                                 case_name + '_input_' + str(index) + '.bin')
        const_data.tofile(file_path)
        os.chmod(file_path, ConstManager.WRITE_MODES)
        const_value = const_data.tolist()
        return const_value
