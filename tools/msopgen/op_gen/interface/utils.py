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
from shutil import copytree
from shutil import copy2

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
FMK_LIST = ["tf", "tensorflow", "caffe", "pytorch", "ms", "mindspore", "onnx"]
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

# GenModeType
GEN_PROJECT = '0'
GEN_OPERATOR = '1'

# CoreType
AICORE = 0
AICPU = 1


class MsOpGenException(Exception):
    """
    The class for compare error
    """

    def __init__(self, error_info):
        super(MsOpGenException, self).__init__(error_info)
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


def read_json_file(json_path):
    """
    read json file to get json object
    @param json_path: the path of json file
    @return: json object
    """
    try:
        with open(json_path, 'rb') as jsonfile:
            try:
                return json.load(jsonfile)
            except Exception as ex:
                print_error_log(
                    'Failed to load json file %s. Please modify it. %s'
                    % (json_path, str(ex)))
                raise MsOpGenException(MS_OP_GEN_READ_FILE_ERROR)
            finally:
                pass
    except IOError as io_error:
        print_error_log(
            'Failed to open json file %s. %s' % (json_path, str(io_error)))
        raise MsOpGenException(MS_OP_GEN_OPEN_FILE_ERROR)
    finally:
        pass


class CheckFromConfig:
    """
    The class for check param from config file
    """
    def __init__(self):
        # verification limit
        self.ms_io_dtype_list = \
            self.get_trans_value("MS_INPUT_OUTPUT_DTYPE_LIST")
        self.io_dtype_map = self.get_trans_value("INPUT_OUTPUT_DTYPE_MAP")
        self.ir_attr_type_map = self.get_trans_value("IR_ATTR_TYPE_MAP")
        self.ini_attr_type_map = self.get_trans_value("INI_ATTR_TYPE_MAP")
        self.check_attr_type_map = self.get_trans_value("CHECK_PARAM_ATTR_TYPE_MAP")
        self.tf_attr_type_map = self.get_trans_value("TF_ATTR_TYPE_MAP")
        self.ms_tf_io_dtype_map = \
            self.get_trans_value("MS_TF_INPUT_OUTPUT_DTYPE_MAP")
        self.tf_io_dtype_map = \
            self.get_trans_value("TF_INPUT_OUTPUT_DTYPE_MAP")

    @staticmethod
    def get_trans_value(key):
        """
        get verification limit from config file
        @param key: key of config json file
        @return: value of config json file
        """
        current_path = os.path.abspath(__file__)
        transform_json_path = os.path.join(
            os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."), "config", "transform.json")
        trans_data = read_json_file(transform_json_path)

        if trans_data is None:
            print_error_log("The Config file is empty or invalid. Please check.")
            raise MsOpGenException(MS_OP_GEN_READ_FILE_ERROR)
        trans_data_value = trans_data.get(key)
        if trans_data_value is None:
            print_error_log(
                "%s in Config file is None or invalid. Please check."
                % key)
            raise MsOpGenException(MS_OP_GEN_READ_FILE_ERROR)
        return trans_data_value

    def trans_ms_io_dtype(self, ir_type, ir_name, file_type):
        """
        transform input output type for mindspore
        @param ir_type: type from template file
        @param ir_name: name from template file
        @param file_type: template file type
        @return: type for mindspore
        """
        if ir_type in self.ms_io_dtype_list:
            return ir_type
        print_warn_log("The %s 'TypeRange' '%s' in the %s file is "
                       "not supported. Please check. If you do not have "
                       "this problems, ignore the warning."
                       % (ir_name, ir_type, file_type))
        return ""

    def trans_io_dtype(self, ir_type, ir_name, file_type):
        """
        transform input output type for tf,caffee,pytorch
        @param ir_type: type from template file
        @param ir_name: name from template file
        @param file_type: template file type
        @return: type for tf,caffee,pytorch
        """
        if ir_type in self.io_dtype_map:
            return self.io_dtype_map.get(ir_type)
        print_warn_log("The %s 'TypeRange' '%s' in the %s file is "
                       "not supported. Please check. If you do not have "
                       "this problems, ignore the warning."
                       % (ir_name, ir_type, file_type))
        return ""

    def trans_ir_attr_type(self, attr_type, file_type):
        """
        transform attr type for ir.h
        @param attr_type: type from template file
        @param file_type: template file type
        @return: attr type for ir.h
        """
        if attr_type in self.ir_attr_type_map:
            return self.ir_attr_type_map.get(attr_type)
        print_warn_log("The attr type '%s' specified in the %s file is "
                       "not supported. Please check the input or output type. "
                       "If you not have this problem, ignore the "
                       "warning." % (attr_type, file_type))
        return ""

    def trans_ini_attr_type(self, attr_type):
        """
        transform attr type for .ini
        @param attr_type: attr type from template file
        @return: attr type for .ini
        """
        if attr_type in self.ini_attr_type_map:
            return self.ini_attr_type_map.get(attr_type)
        print_warn_log("The attr type '%s' is not supported in the .ini file. "
                       "Please check the attr type. If you do not have this "
                       "problem, ignore the warning." % attr_type)
        return ""

    def trans_tf_attr_type(self, tf_type):
        """
        transform attr type from tf .txt
        @param tf_type: tf type from template file
        @return: attr type for .ini
        """
        if tf_type in self.tf_attr_type_map:
            return self.tf_attr_type_map.get(tf_type)
        print_warn_log("The attr type '%s' in the .txt file is not supported. "
                       "Please check the input or output type. If you do not "
                       "have this problem, ignore the warning." % tf_type)
        return ""

    def trans_ms_tf_io_dtype(self, tf_type, name):
        """
        transform tf type from tf mindspore .txt
        @param tf_type: tf type from template file
        @param name: tf name from template file
        @return: type for tf mindspore
        """
        if tf_type in self.ms_tf_io_dtype_map:
            return self.ms_tf_io_dtype_map.get(tf_type)
        print_warn_log("The '%s' type '%s' in the .txt file is "
                       "not supported. Please check. If you do not "
                       "have this problem, ignore the warning."
                       % (name, tf_type))
        return ""

    def trans_tf_io_dtype(self, tf_type, name):
        """
        transform tf type from tf  .txt
        @param tf_type: tf type from template file
        @param name: tf name from template file
        @return: type for tf
        """
        if tf_type in self.tf_io_dtype_map:
            return self.tf_io_dtype_map.get(tf_type)
        print_warn_log("The '%s' type '%s' in the .txt file is not supported. "
                       "Please check. If you do not have this problems, just "
                       "ignore the warning." % (name, tf_type))
        return ""

    def trans_check_attr_type(self, attr_type):
        """
        transform attr type for check_op_params
        @param attr_type: attr type from template file
        @return: attr type for .ini
        """
        if attr_type in self.check_attr_type_map:
            return self.check_attr_type_map.get(attr_type)
        print_warn_log("The attr type '%s' is not supported in check_op_params. "
                       "Please check the attr type. If you do not have this "
                       "problem, ignore the warning." % attr_type)
        return ""


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
        print_error_log("The path is null. Please check whether the argument is valid.")
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
        finally:
            pass
    if not os.path.exists(path):
        print_error_log('The path {} does not exist. Please check whether '
                        'the path exists.'.format(path))
        raise MsOpGenException(MS_OP_GEN_INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log('You do not have the read permission on the path {} .'
                        'Please check.'.format(path))
        raise MsOpGenException(MS_OP_GEN_INVALID_PATH_ERROR)

    if isdir and not os.access(path, os.W_OK):
        print_error_log('You do not have the write permission on the path {} .'
                        'Please check.'.format(path))
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
            if copy_src_to_dst(srcname, dstname, is_skip_exist):
                continue
        except (OSError, os.error) as why:
            errors.append((srcname, dstname, str(why)))
        finally:
            pass

    if errors:
        print_error_log(errors)
        raise MsOpGenException(MS_OP_GEN_WRITE_FILE_ERROR)


def copy_src_to_dst(srcname, dstname, is_skip_exist):
    """
    copy sub template files  from src dir to dest dir
    :param srcname: source sub dir
    :param dstname: dest sub dir
    :param is_skip_exist: skip when dir is exist
    """
    if os.path.isdir(srcname):
        if copy_exist_file(dstname, is_skip_exist):
            return True
        copytree(srcname, dstname)
    else:
        copy2(srcname, dstname)


def copy_exist_file(dstname, is_skip_exist):
    """
    copy file is exist
    :param dstname: dest sub dir
    :param is_skip_exist: skip when dir is exist
    """
    if os.path.isdir(dstname) and len(os.listdir(dstname)) != 0:
        if is_skip_exist:
            return True
        print_error_log("{} is not empty. Please check.".format(dstname))
        sys.exit(MS_OP_GEN_INVALID_PATH_ERROR)


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
        print_warn_log("line = %s, (\"key:value\") format error. Please "
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
    finally:
        pass


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
    finally:
        pass


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
    finally:
        pass
    print_info_log("File %s generated successfully." % op_file)


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
    finally:
        pass
    print_info_log(
        "Generate file %s successfully." % json_path)


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
        elif name_str.isupper():
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
