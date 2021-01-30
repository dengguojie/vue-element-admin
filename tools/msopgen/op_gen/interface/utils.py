#!/usr/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves the common function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
try:
    import os
    import os.path
    import sys
    import time
    import re
    import stat
    import json
    from shutil import copytree
    from shutil import copy2
except (ImportError,) as import_error:
    sys.exit("[ERROR][utils]Unable to import module: %s." % str(
        import_error))

# error code for user:success
MS_OP_GEN_NONE_ERROR = 0
# error code for user: config error
MS_OP_GEN_CONFIG_UNSUPPORTED_FMK_TYPE_ERROR = 11
MS_OP_GEN_CONFIG_INVALID_OUTPUT_PATH_ERROR = 12
MS_OP_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR = 13
MS_OP_GEN_CONFIG_INVALID_COMPUTE_UNIT_ERROR = 14
MS_OP_GEN_CONFIG_UNSUPPORTED_MODE_ERROR = 15
MS_OP_GEN_CONFIG_OP_DEFINE_ERROR = 16
MS_OP_GEN_INVALID_PARAM_ERROR = 17
MS_OP_GEN_INVALID_SHEET_PARSE_ERROR = 18
# error code for user: generator error
MS_OP_GEN_INVALID_PATH_ERROR = 101
MS_OP_GEN_PARSE_DUMP_FILE_ERROR = 102
MS_OP_GEN_OPEN_FILE_ERROR = 103
MS_OP_GEN_CLOSE_FILE_ERROR = 104
MS_OP_GEN_OPEN_DIR_ERROR = 105
MS_OP_GEN_INDEX_OUT_OF_BOUNDS_ERROR = 106
MS_OP_GEN_PARSER_JSON_FILE_ERROR = 107
MS_OP_GEN_WRITE_FILE_ERROR = 108
MS_OP_GEN_READ_FILE_ERROR = 109
MS_OP_GEN_UNKNOWN_CORE_TYPE_ERROR = 110
MS_OP_GEN_PARSER_EXCEL_FILE_ERROR = 111
MS_OP_GEN_JSON_DATA_ERROR = 112
MS_OP_GEN_INVALID_FILE_ERROR = 113
# error code for user: un know error
MS_OP_GEN_UNKNOWN_ERROR = 1001
# call os/sys error:
MS_OP_GEN_MAKE_DIRS_ERROR = 1002
MS_OP_GEN_COPY_DIRS_ERROR = 1003

LEFT_BRACES = "{"
RIGHT_BRACES = "}"
SUPPORT_PATH_PATTERN = r"^[A-Za-z0-9_\./:()=\\-]+$"
FMK_MS = ["ms", "mindspore"]
FMK_LIST = ["tf", "tensorflow", "caffe", "pytorch", "ms", "mindspore"]
PROJ_MS_NAME = "mindspore"
GEN_MODE_LIST = ['0', '1']
OP_TEMPLATE_PATH = "../template/op_project_tmpl"
MS_PROTO_PATH = "op_proto"
OP_TEMPLATE_MS_OP_PROTO_PATH = "../template/op_project_tmpl/op_proto"
OP_TEMPLATE_AICPU_PATH = "../template/cpukernel"
OP_TEMPLATE_TBE_PATH = "../template/tbe"
SPACE = ' '
EMPTY = ''
FILE_AUTHORITY = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR
FOLDER_MASK = 0o700

WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

# path
IMPL_DIR = "tbe/impl/"
MS_IMPL_DIR = "mindspore/impl"
IMPL_NAME = "_impl"
IMPL_SUFFIX = ".py"

# input arguments
INPUT_ARGUMENT_CMD_GEN = 'gen'
INPUT_ARGUMENT_CMD_MI = 'mi'
INPUT_ARGUMENT_CMD_MI_QUERY = 'query'

OP_INFO_WITH_PARAM_TYPE_LEN = 3
OP_INFO_WITH_FORMAT_LEN = 4
AICPU_CORE_TYPE_LIST = ['aicpu', 'ai_cpu']
AICORE_CORE_TYPE_LIST = ['aicore', 'ai_core', 'vectorcore', 'vector_core']
PARAM_TYPE_DYNAMIC = "dynamic"
PARAM_TYPE_REQUIRED = "required"
PARAM_TYPE_OPTIONAL = "optional"
PARAM_TYPE_MAP_INI = {"1": PARAM_TYPE_REQUIRED, "0": PARAM_TYPE_OPTIONAL}
INPUT_OUTPUT_PARAM_TYPE = [PARAM_TYPE_DYNAMIC, PARAM_TYPE_REQUIRED,
                           PARAM_TYPE_OPTIONAL]
ATTR_PARAM_TYPE = [PARAM_TYPE_REQUIRED, PARAM_TYPE_OPTIONAL]

# input file type
INPUT_FILE_XLSX = ".xlsx"
INPUT_FILE_XLS = ".xls"
INPUT_FILE_TXT = ".txt"
INPUT_FILE_JSON = ".json"
INPUT_FILE_EXCEL = (INPUT_FILE_XLSX, INPUT_FILE_XLS)
MI_VALID_TYPE = (INPUT_FILE_XLSX, INPUT_FILE_XLS, INPUT_FILE_JSON)
GEN_VALID_TYPE = (INPUT_FILE_XLSX, INPUT_FILE_XLS, INPUT_FILE_TXT,
                  INPUT_FILE_JSON)

# keys in map
INFO_IR_TYPES_KEY = "ir_type_list"
INFO_PARAM_TYPE_KEY = "param_type"
INFO_PARAM_FORMAT_KEY = "format_list"

# mindspore support dtype
MS_INPUT_OUTPUT_DTYPE_LIST = [
    "None_None", "BOOL_None", "BOOL_Default", "BOOL_5HD", "BOOL_FracZ",
    "BOOL_FracNZ", "BOOL_C1HWNCoC0", "BOOL_NCHW", "BOOL_NHWC", "BOOL_NDHWC",
    "I8_None", "I8_Default", "I8_5HD", "I8_FracZ", "I8_FracNZ", "I8_C1HWNCoC0",
    "I8_NCHW", "I8_NHWC", "I8_HWCN", "I8_NDHWC", "U8_None", "U8_Default",
    "U8_5HD", "U8_FracZ", "U8_FracNZ", "U8_C1HWNCoC0", "U8_NCHW", "U8_NHWC",
    "U8_HWCN", "U8_NDHWC", "I16_None", "I16_Default", "I16_5HD", "I16_FracZ",
    "I16_FracNZ", "I16_C1HWNCoC0", "I16_NCHW", "I16_NHWC", "I16_HWCN",
    "I16_NDHWC", "U16_None", "U16_Default", "U16_5HD", "U16_FracZ",
    "U16_FracNZ", "U16_C1HWNCoC0", "U16_NCHW", "U16_NHWC", "U16_HWCN",
    "U16_NDHWC", "I32_None", "I32_Default", "I32_5HD", "I32_FracZ",
    "I32_FracNZ", "I32_C1HWNCoC0", "I32_NCHW", "I32_NHWC", "I32_HWCN",
    "I32_NDHWC", "U32_None", "U32_Default", "U32_5HD", "U32_FracZ",
    "U32_FracNZ", "U32_C1HWNCoC0", "U32_NCHW", "U32_NHWC", "U32_HWCN",
    "U32_NDHWC", "I64_None", "I64_Default", "I64_5HD", "I64_FracZ",
    "I64_FracNZ", "I64_C1HWNCoC0", "I64_NCHW", "I64_NHWC", "I64_HWCN",
    "I64_NDHWC", "U64_None", "U64_Default", "U64_5HD", "U64_FracZ",
    "U64_FracNZ", "U64_C1HWNCoC0", "U64_NCHW", "U64_NHWC", "U64_HWCN",
    "U64_NDHWC", "F16_None", "F16_Default", "F16_5HD", "F16_FracZ",
    "F16_FracNZ", "F16_C1HWNCoC0", "F16_NCHW", "F16_NHWC", "F16_HWCN",
    "F16_NDHWC", "F16_FracZNLSTM", "F32_None", "F32_Default", "F32_5HD",
    "F32_FracZ", "F32_FracNZ", "F32_C1HWNCoC0", "F32_NCHW", "F32_NHWC",
    "F32_HWCN", "F32_NDHWC", "F32_FracZNLSTM", "F64_None",
    "F64_Default", "F64_5HD", "F64_FracZ", "F64_FracNZ", "F64_C1HWNCoC0",
    "F64_NCHW", "F64_NHWC", "F64_HWCN", "F64_NDHWC"
]


class GenModeType:
    """
       CLass for gen mode type
       """
    GEN_PROJECT = '0'
    GEN_OPERATOR = '1'


INPUT_OUTPUT_DTYPE_MAP = {
    "float": "DT_FLOAT",
    "bool": "DT_BOOL",
    "int32": "DT_INT32",
    "int64": "DT_INT64",
    "half": "DT_FLOAT16",
    "uint32": "DT_UINT32",
    "uint64": "DT_UINT64",
    "numbertype": "DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, "
                  "DT_INT32,DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,"
                  "DT_UINT64, DT_QINT8,DT_QUINT8, DT_QINT16, DT_QUINT16, "
                  "DT_QINT32, DT_BOOL",
    "realnumbertype": "DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, "
                      "DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, "
                      "DT_UINT64, DT_QINT8, DT_QUINT8, DT_QINT16, "
                      "DT_QUINT16, DT_QINT32, DT_BOOL",
    "quantizedtype": "DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32",
    "all": "DT_FLOAT,DT_DOUBLE,DT_INT32,DT_UINT8,DT_INT16,DT_INT8,DT_STRING,"
           "DT_COMPLEX64,DT_INT64,DT_BOOL,DT_QINT8, DT_QUINT8,DT_QINT32,"
           "DT_Bfp16,DT_QINT16,DT_QUINT16,DT_UINT16,DT_COMPLEX128,DT_HALF,"
           "DT_RESOURCE, DT_VARIANT,DT_UINT32,DT_UINT64",
    "BasicType": "DT_FLOAT,DT_DOUBLE,DT_INT32,DT_UINT8,DT_INT16,DT_INT8,"
                 "DT_COMPLEX64,DT_INT64,DT_QINT8, DT_QUINT8,DT_QINT32, "
                 "DT_QINT16,DT_QUINT16,DT_UINT16,DT_COMPLEX128,DT_HALF,"
                 "DT_UINT32,DT_UINT64",
    "IndexNumberType": "DT_INT32,DT_INT64 ",
    "int8": "DT_INT8",
    "int16": "DT_INT16",
    "uint8": "DT_UINT8",
    "uint16": "DT_UINT16",
    "qint8": "DT_QINT8",
    "qint16": "DT_QINT16",
    "qint32": "DT_QINT32",
    "quint8": "DT_QUINT8",
    "quint16": "DT_QUINT16",
    "fp16": "DT_FLOAT16",
    "fp32": "DT_FLOAT32",
    "double": "DT_DOUBLE",
    "complex64": "DT_COMPLEX64",
    "complex128": "DT_COMPLEX128",
    "string": "DT_STRING",
    "resource": "DT_RESOURCE"
}

IR_ATTR_TYPE_MAP = {
    "int": "Int",
    "float": "Float",
    "string": "String",
    "bool": "Bool",
    "type": "Type",
    "list_int": "ListInt",
    "list_string": "ListString",
    "list_bool": "ListBool",
    "list_list_int": "ListListInt",
    "tensor": "Tensor",
    "list_float": "ListFloat",
    "list_tensor": "ListTensor",
    "list_type": "ListType"
}

INI_ATTR_TYPE_MAP = {
    "Int": "int",
    "Float": "float",
    "String": "str",
    "Bool": "bool",
    "Tensor": "tensor",
    "Type": "type",
    "ListInt": "listInt",
    "ListFloat": "listFloat",
    "ListString": "listStr",
    "ListBool": "listBool",
    "ListTensor": "listTensor",
    "ListType": "listType",
    "ListListInt": "listListInt"
}

TF_ATTR_TYPE_MAP = {
    "int": "Int",
    "float": "Float",
    "string": "String",
    "bool": "Bool",
    "type": "Type",
    "list(int)": "ListInt",
    "list(string)": "ListString",
    "list(bool)": "ListBool",
    "list(list(int))": "ListListInt",
    "list(float)": "ListFloat"
}


class CoreType:
    """
    The index of Core type
    """
    AICORE = 0
    AICPU = 1


class PathType:
    """
    The enum for path type
    """
    All = 0
    File = 1
    Directory = 2


class MsOpGenException(Exception):
    """
    The class for compare error
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


def check_name_valid(name):
    """
    Function Description:
        check name valid
    Parameter:
        name: the name to check
    Return Value:
        MsOpGenException
    """
    if name == "":
        print_warn_log("The input name is \"\"")
        return MS_OP_GEN_INVALID_PARAM_ERROR
    name_pattern = re.compile(SUPPORT_PATH_PATTERN)
    match = name_pattern.match(name)
    if match is None:
        print_warn_log("The op type is invalid %s" % name)
        return MS_OP_GEN_INVALID_PARAM_ERROR
    return MS_OP_GEN_NONE_ERROR


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
        raise MsOpGenException(MS_OP_GEN_INVALID_PATH_ERROR)
    path = os.path.realpath(path)
    if isdir and not os.path.exists(path):
        try:
            os.makedirs(path, mode=0o700)
        except OSError as ex:
            print_error_log(
                'Failed to create {}. Please check the path permission or '
                'disk space. {} '.format(path, str(ex)))
            raise MsOpGenException(MS_OP_GEN_INVALID_PATH_ERROR)
    if not os.path.exists(path):
        print_error_log('The path {} does not exist. Please check whether '
                        'the path exists.'.format(path))
        raise MsOpGenException(MS_OP_GEN_INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log('The path {} does not have permission to read.'
                        ' Please check the path permission.'.format(path))
        raise MsOpGenException(MS_OP_GEN_INVALID_PATH_ERROR)

    if isdir and not os.access(path, os.W_OK):
        print_error_log('The path {} does not have permission to write.'
                        ' Please check the path permission.'.format(path))
        raise MsOpGenException(MS_OP_GEN_INVALID_PATH_ERROR)

    if isdir:
        if not os.path.isdir(path):
            print_error_log('The path {} is not a directory.'
                            ' Please check the path.'.format(path))
            raise MsOpGenException(MS_OP_GEN_INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log('The path {} is not a file.'
                            ' Please check the path.'.format(path))
            raise MsOpGenException(MS_OP_GEN_INVALID_PATH_ERROR)


def copy_template(src, dst, is_skip_exist=False):
    """
    copy template files  from src dir to dest dir
    :param src: source dir
    :param dst: dest dir
    :param is_skip_exist: True:skip when dir is exist
    """
    make_dirs(dst)
    names = os.listdir(src)
    errors = []
    for name in names:
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.isdir(srcname):
                if os.path.isdir(dstname) and len(os.listdir(dstname)) != 0:
                    if is_skip_exist:
                        continue
                    print_error_log(
                        dstname + " is not empty,please check.")
                    sys.exit(MS_OP_GEN_INVALID_PATH_ERROR)
                copytree(srcname, dstname)
            else:
                copy2(srcname, dstname)
        except (OSError, os.error) as why:
            errors.append((srcname, dstname, str(why)))

    if errors:
        print_error_log(errors)
        raise MsOpGenException(MS_OP_GEN_WRITE_FILE_ERROR)


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
    if not match:
        print_warn_log("line = %s, (\"key:value\") format error , please "
                       "check the .txt file! " % line)
    return match


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
        raise MsOpGenException(MS_OP_GEN_MAKE_DIRS_ERROR)


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
        raise MsOpGenException(MS_OP_GEN_READ_FILE_ERROR)


def write_files(op_file, new_str):
    """
    write new_str to op_file
    :param op_file:the file
    :param new_str:the string to be written
    :return:
    """
    try:
        if os.path.exists(op_file):
            print_warn_log(op_file + " already exists!")
            return
        with os.fdopen(os.open(op_file, WRITE_FLAGS, WRITE_MODES), 'w') as \
                fout:
            fout.write(new_str)
    except OSError as err:
        print_error_log("Unable to write file(%s): %s." % op_file % str(err))
        raise MsOpGenException(MS_OP_GEN_WRITE_FILE_ERROR)
    print_info_log("Generate file %s successfully." % op_file)


def write_json_file(json_path, content):
    """
    write  content to json file
    :param content:
    :param json_path: the json path
    :return: the json object
    """
    try:
        with os.fdopen(os.open(json_path, WRITE_FLAGS,
                               WRITE_MODES), 'w+') as file_object:
            file_object.write(
                json.dumps(content, sort_keys=False, indent=4))
    except IOError as io_error:
        print_error_log(
            'Failed to generate json file %s. %s' % (json_path, str(io_error)))
        raise MsOpGenException(MS_OP_GEN_WRITE_FILE_ERROR)
    print_info_log(
        "Generate file %s successfully." % json_path)


def read_json_file(json_path):
    try:
        with open(json_path, 'rb') as jsonfile:
            try:
                return json.load(jsonfile)
            except Exception as ex:
                print_error_log(
                    'Failed to load json file %s. Please modify it. %s'
                    % (json_path, str(ex)))
                raise MsOpGenException(MS_OP_GEN_READ_FILE_ERROR)
    except IOError as io_error:
        print_error_log(
            'Failed to open json file %s. %s' % (json_path, str(io_error)))
        raise MsOpGenException(MS_OP_GEN_OPEN_FILE_ERROR)


def fix_name_lower_with_under(name):
    """
    change name to lower_with_under style,
    eg: "ConcatOffset" -> concat_offset
    :param name: op type/input/out_put/attribute name to be fix
    :return: name has been fixed
    """
    fix_name = ""
    for index, name_str in enumerate(name):
        if name_str.isupper():
            if index == 0:
                fix_name += name_str.lower()
            else:
                fix_name += "_" + name_str.lower()
        else:
            fix_name += name_str
    return fix_name
