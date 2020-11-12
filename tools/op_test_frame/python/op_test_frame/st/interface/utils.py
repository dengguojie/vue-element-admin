#!/usr/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves the common function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import os.path
import sys
import time
import re
import stat
import json
import numpy as np

# error code for user:success
OP_TEST_GEN_NONE_ERROR = 0
# error code for user: config error
OP_TEST_GEN_CONFIG_UNSUPPORTED_FMK_TYPE_ERROR = 11
OP_TEST_GEN_CONFIG_INVALID_OUTPUT_PATH_ERROR = 12
OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR = 13
OP_TEST_GEN_CONFIG_INVALID_COMPUTE_UNIT_ERROR = 14
OP_TEST_GEN_CONFIG_UNSUPPORTED_OPTION_ERROR = 15
OP_TEST_GEN_CONFIG_OP_DEFINE_ERROR = 16
OP_TEST_GEN_INVALID_PARAM_ERROR = 17
OP_TEST_GEN_INVALID_SHEET_PARSE_ERROR = 18
OP_TEST_GEN_INVALID_DATA_ERROR = 18
# error code for user: generator error
OP_TEST_GEN_INVALID_PATH_ERROR = 101
OP_TEST_GEN_PARSE_JSON_FILE_ERROR = 102
OP_TEST_GEN_OPEN_FILE_ERROR = 103
OP_TEST_GEN_CLOSE_FILE_ERROR = 104
OP_TEST_GEN_OPEN_DIR_ERROR = 105
OP_TEST_GEN_INDEX_OUT_OF_BOUNDS_ERROR = 106
OP_TEST_GEN_PARSER_JSON_FILE_ERROR = 107
OP_TEST_GEN_WRITE_FILE_ERROR = 108
OP_TEST_GEN_READ_FILE_ERROR = 109
OP_TEST_GEN_MAKE_DIR_ERROR = 110
OP_TEST_GEN_GET_KEY_ERROR = 111
OP_TEST_GEN_MAKE_DIRS_ERROR = 112
# error code for user: un know error
OP_TEST_GEN_UNKNOWN_ERROR = 1001
OP_TEST_GEN_TF_LOAD_ERROR = 1002
OP_TEST_GEN_TF_GET_OPERATORS_ERROR = 1003
OP_TEST_GEN_TF_WRITE_GRAPH_ERROR = 1004
OP_TEST_GEN_TF_GET_PLACEHOLDER_ERROR = 1005
OP_TEST_GEN_TF_CHANGE_PLACEHOLDER_ERROR = 1006

ACL_TEST_GEN_NONE_ERROR = 0
ACL_TEST_GEN_ERROR = 255

WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

FMK_LIST = "tf tensorflow caffe"
SUPPORT_PATH_PATTERN = r"^[A-Za-z0-9_\./:()=\\-]+$"
EMPTY = ""
SRC_RELATIVE_TEMPLATE_PATH = "/../template/acl_op_src"

MAIN_CPP_RELATIVE_PATH = "/src/main.cpp"
TESTCASE_CPP_RELATIVE_PATH = "/src/testcase.cpp"
ACL_OP_JSON_RELATIVE_PATH = "/run/out/test_data/config/acl_op.json"


FILE_AUTHORITY = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR
FOLDER_MASK = 0o700

WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

DTYPE_TO_NUMPY_MAP = {
    'float16': np.float16,
    'float': np.float32,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'bool': np.bool
}

ATTR_TYPE_MAP = {
    'int': 'int',
    'float': 'float',
    'bool': 'bool',
    'str': 'string',
    'listInt': 'list_int',
    'listFloat': 'list_float',
    'listBool': 'list_bool',
    'listStr': 'list_string',
    'listListInt': 'list_list_int'
}

OP_ATTR_TYPE_MAP = {
    'int': 'OP_INT',
    'float': 'OP_FLOAT',
    'bool': 'OP_BOOL',
    'string': 'OP_STRING',
    'list_int': 'OP_LIST_INT',
    'list_float': 'OP_LIST_FLOAT',
    'list_bool': 'OP_LIST_BOOL',
    'list_string': 'OP_LIST_STRING',
    'list_list_int': 'OP_LIST_INT_PTR'
}

ATTR_MEMBER_VAR_MAP = {
    'int': 'intAttr',
    'float': 'floatAttr',
    'bool': 'boolAttr',
    'string': 'stringAttr',
    'list_int': 'listIntAttr',
    'list_float': 'listFloatAttr',
    'list_bool': 'listBoolAttr',
    'list_string': 'listStringAttr',
    'list_list_int': 'listIntPtrAttr'
}

DATA_DISTRIBUTION_LIST = ['uniform', 'normal', 'beta', 'laplace', 'triangular',
                          'relu', 'sigmoid', 'softmax', 'tanh']

FORMAT_LIST = ['NCHW', 'NHWC', 'ND', 'NC1HWC0', 'FRACTAL_Z', 'NC1C0HWPAD',
               'NHWC1C0', 'FSR_NCHW', 'FRACTAL_DECONV', 'C1HWNC0',
               'FRACTAL_DECONV_TRANSPOSE', 'FRACTAL_DECONV_SP_STRIDE_TRANS',
               'NC1HWC0_C04', 'FRACTAL_Z_C04', 'CHWN',
               'DECONV_SP_STRIDE8_TRANS', 'NC1KHKWHWC0', 'BN_WEIGHT',
               'FILTER_HWCK', 'HWCN', 'LOOKUP_LOOKUPS', 'LOOKUP_KEYS',
               'LOOKUP_VALUE', 'LOOKUP_OUTPUT', 'LOOKUP_HITS', 'MD', 'NDHWC',
               'C1HWNCoC0', 'FRACTAL_NZ']


def create_attr_value_str(attr_value):
    """
    create attribute exact value string based on type
    :param attr_value: attr value variable
    :return: none
    """
    if isinstance(attr_value, list):
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
    elif isinstance(attr_value, str):
        # string
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


def map_to_acl_datatype_enum(dtype_list):
    """
    map datatype to acl datatype enum
    :param dtype_list: input dtype list
    :return: acl datatype enum list str
    """
    result_str = ""
    acl_dtype_list = []
    for dtype in dtype_list:
        acl_dtype_list.append("ACL_" + str(dtype).upper())
    result_str += ", ".join(acl_dtype_list)
    return result_str


def map_to_acl_format_enum(format_list):
    """
    map format to acl format enum
    :param format_list: input format list
    :return: acl format enum list str
    """
    result_str = ""
    acl_format_list = []
    for acl_format in format_list:
        acl_format_list.append("ACL_FORMAT_" + str(acl_format).upper())
    result_str += ", ".join(acl_format_list)
    return result_str


class OpTestGenException(Exception):
    """
    The class for Op Gen Exception
    """

    def __init__(self, error_info):
        super().__init__(error_info)
        self.error_info = error_info


def _print_log(level, msg):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(int(time.time())))
    pid = os.getpid()
    print(current_time + " (" + str(pid) + ") - [" + level + "] " + msg)
    sys.stdout.flush()


def print_error_log(error_msg):
    """
    print error log
    @param error_msg: the error message
    @return: none
    """
    _print_log("ERROR", error_msg)


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


def check_path_valid(path, isdir=False):
    """
    Function Description:
        check path valid
    Parameter:
        path: the path to check
        isdir: the path is dir or file
    """
    if path == "":
        print_error_log("The path is null. Please check the argument valid.")
        raise OpTestGenException(OP_TEST_GEN_INVALID_PARAM_ERROR)
    path = os.path.realpath(path)
    if isdir and not os.path.exists(path):
        try:
            os.makedirs(path, mode=0o700)
        except OSError as ex:
            print_error_log(
                'Failed to create {}. Please check the path permission or '
                'disk space. {} '.format(path, str(ex)))
            raise OpTestGenException(OP_TEST_GEN_INVALID_PATH_ERROR)
    if not os.path.exists(path):
        print_error_log('The path {} does not exist. Please check whether '
                        'the path exists.'.format(path))
        raise OpTestGenException(OP_TEST_GEN_INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log('The path {} does not have permission to read.'
                        ' Please check the path permission.'.format(path))
        raise OpTestGenException(OP_TEST_GEN_INVALID_PATH_ERROR)

    if isdir and not os.access(path, os.W_OK):
        print_error_log('The path {} does not have permission to write.'
                        ' Please check the path permission.'.format(path))
        raise OpTestGenException(OP_TEST_GEN_INVALID_PATH_ERROR)

    if isdir:
        if not os.path.isdir(path):
            print_error_log('The path {} is not a directory.'
                            ' Please check the path.'.format(path))
            raise OpTestGenException(OP_TEST_GEN_INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log('The path {} is not a file.'
                            ' Please check the path.'.format(path))
            raise OpTestGenException(OP_TEST_GEN_INVALID_PATH_ERROR)


def check_name_valid(name):
    """
    Function Description:
        check name valid
    Parameter:
        name: the name to check
    Return Value:
        VectorComparisonErrorCode
    """
    if name == "":
        print_error_log("The input name is \"\"")
        return OP_TEST_GEN_INVALID_PARAM_ERROR
    name_pattern = re.compile(SUPPORT_PATH_PATTERN)
    match = name_pattern.match(name)
    if match is None:
        return OP_TEST_GEN_INVALID_PARAM_ERROR
    return OP_TEST_GEN_NONE_ERROR


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
    print_error_log("(\" \") format error , please check!")
    sys.exit(OP_TEST_GEN_CONFIG_OP_DEFINE_ERROR)


def _check_value_valid(fe_type, value, name, prefix=""):
    value_type = int
    if fe_type == 'int':
        value_type = int
    elif fe_type == 'bool':
        value_type = bool
    elif fe_type == 'string':
        value_type = str
    elif fe_type == 'float':
        value_type = float
    elif fe_type == 'list_int':
        if not isinstance(value, list):
            print_error_log(
                'The value (%s) is invalid. The value of "%s" for "attr" '
                'only supports list_list_int. Please modify it.'
                % (value, name))
            raise OpTestGenException(OP_TEST_GEN_INVALID_DATA_ERROR)
        if len(value) == 0:
            print_error_log(
                'The value (%s) is empty. The value of "%s" for "attr" '
                'only supports list_list_int. Please modify it.'
                % (value, name))
            raise OpTestGenException(OP_TEST_GEN_INVALID_DATA_ERROR)
        for item in value:
            _check_value_valid('int', item, name, 'list_list_')
        return

    if not isinstance(value, value_type):
        print_error_log(
            'The value (%s) is invalid. The value of "%s" for "attr" only '
            'supports %s%s. Please modify it.'
            % (value, name, prefix, fe_type))
        raise OpTestGenException(OP_TEST_GEN_INVALID_DATA_ERROR)


def check_attr_value_valid(attr):
    """
    check attr value valid
    :param attr: the attr to check
    :return:
    """
    attr_type = attr['type']
    if attr_type.startswith('list_'):
        if not isinstance(attr['value'], list):
            print_error_log(
                'The value (%s) is invalid. The value of "%s" for "attr" '
                'only supports %s. Please modify it.'
                % (attr['value'], attr['name'], attr_type))
            raise OpTestGenException(OP_TEST_GEN_INVALID_DATA_ERROR)
        if len(attr['value']) == 0:
            print_error_log(
                'The value (%s) is empty. The value of "%s" for "attr" '
                'only supports %s. Please modify it.'
                % (attr['value'], attr['name'], attr_type))
            raise OpTestGenException(OP_TEST_GEN_INVALID_DATA_ERROR)
        for value in attr['value']:
            _check_value_valid(
                attr_type[len('list_'):], value, attr['name'], 'list_')
    else:
        _check_value_valid(attr_type, attr['value'], attr['name'])


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
                raise OpTestGenException(OP_TEST_GEN_PARSE_JSON_FILE_ERROR)
    except IOError as io_error:
        print_error_log(
            'Failed to open file %s. %s' % (json_path, str(io_error)))
        raise OpTestGenException(OP_TEST_GEN_OPEN_FILE_ERROR)


def read_file(op_file):
    """
    read new_str from op_file
    :param op_file:the file
    :return:
    """
    try:
        with open(op_file) as file_object:
            txt = file_object.read()
            return txt
    except IOError as io_error:
        print_error_log(
            'Failed to open file %s. %s' % (op_file, str(io_error)))
        raise OpTestGenException(OP_TEST_GEN_OPEN_FILE_ERROR)

def write_json_file(json_path, content):
    """
    write  content to json file
    :param content:
    :param json_path: the json path
    """
    try:
        if os.path.exists(json_path) and os.path.isfile(json_path):
            os.remove(json_path)
        with os.fdopen(os.open(json_path, WRITE_FLAGS,
                               WRITE_MODES), 'w') as file_object:
            file_object.write(
                json.dumps(content, sort_keys=False, indent=4))
    except IOError as io_error:
        print_error_log(
            'Failed to generate file %s. %s' % (json_path, str(io_error)))
        raise OpTestGenException(OP_TEST_GEN_WRITE_FILE_ERROR)
    print_info_log(
        "Generate file %s successfully." % os.path.realpath(json_path))


def make_dirs(op_dir):
    """
    make dirs
    :param op_dir:dirs
    """
    try:
        if not os.path.isdir(op_dir) or not os.path.exists(op_dir):
            os.makedirs(op_dir, FOLDER_MASK)
    except OSError as err:
        print_error_log("Unable to make dir: %s." % str(err))
        raise OpTestGenException(OP_TEST_GEN_MAKE_DIRS_ERROR)

