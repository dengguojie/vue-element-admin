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
import sys
import time
import re
import stat
import json

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
OP_TEST_GEN_AND_RUN_ERROR = 19
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
OP_TEST_GEN_INVALID_DEVICE_ID_ERROR = 113
OP_TEST_GEN_INVALID_INPUT_NAME_ERROR = 114
OP_TEST_GEN_NONE_TYPICAL_SHAPE_ERROR = 115
# error code for user: un know error
OP_TEST_GEN_UNKNOWN_ERROR = 1001
OP_TEST_GEN_TF_LOAD_ERROR = 1002
OP_TEST_GEN_TF_GET_OPERATORS_ERROR = 1003
OP_TEST_GEN_TF_WRITE_GRAPH_ERROR = 1004
OP_TEST_GEN_TF_GET_PLACEHOLDER_ERROR = 1005
OP_TEST_GEN_TF_CHANGE_PLACEHOLDER_ERROR = 1006

ACL_TEST_GEN_NONE_ERROR = 0
ACL_TEST_GEN_ERROR = 255

BOTH_GEN_AND_RUN_ACL_PROJ = 0
ONLY_GEN_WITHOUT_RUN_ACL_PROJ = 1
ONLY_RUN_WITHOUT_GEN_ACL_PROJ = 2
ONLY_RUN_WITHOUT_GEN_ACL_PROJ_PERFORMANCE = 3
BOTH_GEN_AND_RUN_ACL_PROJ_PERFORMANCE = 4

WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

FMK_LIST = "tf tensorflow caffe"
SUPPORT_PATH_PATTERN = r"^[A-Za-z0-9_\./:()=\\-]+$"
EMPTY = ""
SRC_RELATIVE_TEMPLATE_PATH = "/../template/acl_op_src"
MAIN_CPP_RELATIVE_PATH = "/src/main.cpp"
TESTCASE_CPP_RELATIVE_PATH = "/src/testcase.cpp"
ACL_OP_JSON_RELATIVE_PATH = "/run/out/test_data/config/acl_op.json"
TESTCASE_PY_RELATIVE_PATH = "/src/test_{op_name}.py"
PYTEST_INI_RELATIVE_PATH = "/src/pytest.ini"
INPUT_SUFFIX_LIST = ['.ini', '.py']
BIN_FILE = '.bin'
PY_FILE = '.py'
FILE_AUTHORITY = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR
FOLDER_MASK = 0o700
TYPE_UNDEFINED = "UNDEFINED"

INPUT_DESC = 'input_desc'
OUTPUT_DESC = 'output_desc'
# dynamic shape scenario add keys as follows
SHAPE_RANGE = 'shape_range'
TYPICAL_SHAPE = 'typical_shape'
VALUE = 'value'
# Two dynamic scenarios: shape value is -1 or -2
SHAPE_DYNAMIC_SCENARIOS_ONE = -1
SHAPE_DYNAMIC_SCENARIOS_TWO = -2
# dynamic shape scenario, shape_range default value.
SHAPE_RANGE_DEFAULT_VALUE = [[1, -1]]

ONLY_GEN_WITHOUT_RUN = 'only_gen_without_run'
ONLY_RUN_WITHOUT_GEN = 'only_run_without_gen'
ASCEND_GLOBAL_LOG_LEVEL = 'ascend_global_log_level'
ASCEND_SLOG_PRINT_TO_STDOUT = 'ascend_slog_print_to_stdout'
ATC_SINGLEOP_ADVANCE_OPTION = 'atc_singleop_advance_option'
PERFORMACE_MODE = 'performance_mode'

# dynamic input.
DYNAMIC_INPUT = 'dynamic'
DYNAMIC_INPUT_ARGS = '*dynamic_input'
DYNAMIC_INPUT_NAME = 'dynamic_input'

SPACE = ' '
EMPTY = ''
NEW_LINE_MARK = "\\"
QUOTATION_MARK = "\""
COMMA = ','
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
IN_OUT_OP_KEY_MAP = {
    'INPUT': 'input',
    'DYNAMIC_INPUT': 'input',
    'OPTIONAL_INPUT': 'input',
    'OUTPUT': 'output',
    'DYNAMIC_OUTPUT': 'output'
}

AICPU_ATTR_LIST = ['ATTR', 'REQUIRED_ATTR']
ATTR_TYPE_MAP = {
    'int': 'int',
    'float': 'float',
    'bool': 'bool',
    'str': 'string',
    'type': 'data_type',
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
    'data_type': 'OP_DTYPE',
    'list_int': 'OP_LIST_INT',
    'list_float': 'OP_LIST_FLOAT',
    'list_bool': 'OP_LIST_BOOL',
    'list_string': 'OP_LIST_STRING',
    'list_list_int': 'OP_LIST_INT_PTR'
}

OP_PROTO_PARSE_ATTR_TYPE_MAP = {
    "Int": "int",
    "Float": "float",
    "String": "str",
    "Bool": "bool",
    "Type": "type",
    "ListInt": "listInt",
    "ListFloat": "listFloat",
    "ListString": "listStr",
    "ListBool": "listBool",
    "ListListInt": "listListInt"
}

ATTR_MEMBER_VAR_MAP = {
    'int': 'intAttr',
    'float': 'floatAttr',
    'bool': 'boolAttr',
    'string': 'stringAttr',
    'data_type': 'dtypeAttr',
    'list_int': 'listIntAttr',
    'list_float': 'listFloatAttr',
    'list_bool': 'listBoolAttr',
    'list_string': 'listStringAttr',
    'list_list_int': 'listIntPtrAttr'
}

ATTR_TYPE_SUPPORT_TYPE_MAP = {
    "int8": "DT_INT8",
    "int32": "DT_INT32",
    "int16": "DT_INT16",
    "int64": "DT_INT64",
    "uint8": "DT_UINT8",
    "uint16": "DT_UINT16",
    "uint32": "DT_UINT32",
    "uint64": "DT_UINT64",
    "float": "DT_FLOAT",
    "float16": "DT_FLOAT16",
    "float32": "DT_FLOAT",
    "bool": "DT_BOOL",
    "double": "DT_DOUBLE",
    "complex64": "DT_COMPLEX64",
    "complex128": "DT_COMPLEX128"
}

OPTIONAL_TYPE_LIST = ['UNDEFINED', 'RESERVED']
TRUE_OR_FALSE_LIST = ['True', 'False']


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
        print_error_log("The path is null. Please check whether the argument is valid.")
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
    print_error_log("(\" \") format error. Please check.")
    sys.exit(OP_TEST_GEN_CONFIG_OP_DEFINE_ERROR)


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
        if value not in ATTR_TYPE_SUPPORT_TYPE_MAP.keys():
            print_error_log(
                'The value (%s) is invalid. The value of "%s" for "attr" '
                'only supports in %s. Please modify it.'
                % (value, name, ATTR_TYPE_SUPPORT_TYPE_MAP.keys()))
            raise OpTestGenException(OP_TEST_GEN_INVALID_DATA_ERROR)
        value_type = str
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
            check_value_valid('int', item, name, 'list_list_')
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
    attr_type = attr.get('type')
    if attr_type.startswith('list_'):
        if not isinstance(attr.get('value'), list):
            print_error_log(
                'The value (%s) is invalid. The value of "%s" for "attr" '
                'only supports %s. Please modify it.'
                % (attr.get('value'), attr.get('name'), attr_type))
            raise OpTestGenException(OP_TEST_GEN_INVALID_DATA_ERROR)
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
            if name[index + 1].islower() or (
                    index != 0 and name[index - 1].islower()):
                # If a capital letter is surrounded by lowercase letters, convert to "_" + lowercase letter
                # In addition, all are converted to lowercase letters
                # eg: "Abc2DEf"  ->   "abc2d_ef"
                fix_name += "_{}".format(name_str.lower())
            else:
                fix_name += name_str.lower()
        else:
            fix_name += name_str.lower()
    return fix_name


class ScanFile:
    """
    The class for scanning path to get subdirectories.
    """
    def __init__(self, directory, prefix=None):
        self.directory = directory
        self.prefix = prefix

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
                if os.path.isdir(file_path):
                    dir_info = os.path.split(file_path)
                    if self.prefix:
                        if dir_info[1].startswith(self.prefix):
                            files_list.append(dir_info[1])
                    else:
                        files_list.append(dir_info[1])
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
