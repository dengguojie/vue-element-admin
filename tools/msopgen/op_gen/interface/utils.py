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
import json
from shutil import copytree
from shutil import copy2
from .const_manager import ConstManager


class MsOpGenException(Exception):
    """
    The class for compare error
    """

    def __init__(self, error_info):
        super(MsOpGenException, self).__init__(error_info)
        self.error_info = error_info


def _print_log(level: str, msg: str) -> None:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(int(time.time())))
    pid = os.getpid()
    print(current_time + " (" + str(pid) + ") - [" + level + "] " + msg)
    sys.stdout.flush()


def print_error_log(error_msg: str) -> None:
    """
    print error log
    @param error_msg: the error message
    @return: none
    """
    _print_log("ERROR", error_msg)


def print_warn_log(warn_msg: str) -> None:
    """
    print warn log
    @param warn_msg: the warn message
    @return: none
    """
    _print_log("WARNING", warn_msg)


def print_info_log(info_msg: str) -> None:
    """
    print info log
    @param info_msg: the info message
    @return: none
    """
    _print_log("INFO", info_msg)


def read_json_file(json_path: str) -> any:
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
                raise MsOpGenException(ConstManager.MS_OP_GEN_READ_FILE_ERROR)
            finally:
                pass
    except IOError as io_error:
        print_error_log(
            'Failed to open json file %s. %s' % (json_path, str(io_error)))
        raise MsOpGenException(ConstManager.MS_OP_GEN_OPEN_FILE_ERROR)
    finally:
        pass


class CheckFromConfig:
    """
    The class for check param from config file
    """
    def __init__(self: any) -> None:
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
    def get_trans_value(key: str) -> str:
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
            raise MsOpGenException(ConstManager.MS_OP_GEN_READ_FILE_ERROR)
        trans_data_value = trans_data.get(key)
        if trans_data_value is None:
            print_error_log(
                "%s in Config file is None or invalid. Please check."
                % key)
            raise MsOpGenException(ConstManager.MS_OP_GEN_READ_FILE_ERROR)
        return trans_data_value

    def trans_ms_io_dtype(self: any, ir_type: str, ir_name: str, file_type: str) -> str:
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

    def trans_io_dtype(self: any, ir_type: str, ir_name: str, file_type: str) -> str:
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

    def trans_ir_attr_type(self: any, attr_type: str, file_type: str) -> str:
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

    def trans_ini_attr_type(self: any, attr_type: str) -> str:
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

    def trans_tf_attr_type(self: any, tf_type: str) -> str:
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

    def trans_ms_tf_io_dtype(self: any, tf_type: str, name: str) -> str:
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

    def trans_tf_io_dtype(self: any, tf_type: str, name: str) -> str:
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

    def trans_check_attr_type(self: any, attr_type: str) -> str:
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


def check_name_valid(name: str) -> int:
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
        return ConstManager.MS_OP_GEN_INVALID_PARAM_ERROR
    name_pattern = re.compile(ConstManager.SUPPORT_PATH_PATTERN)
    match = name_pattern.match(name)
    if match is None:
        print_warn_log("The op type is invalid %s" % name)
        return ConstManager.MS_OP_GEN_INVALID_PARAM_ERROR
    return ConstManager.MS_OP_GEN_NONE_ERROR


def check_path_valid(path: str, isdir: bool = False) -> None:
    """
    Function Description:
    check path valid
    Parameter:
    path: the path to check
    isdir: the path is dir or file
    """
    if path == "":
        print_error_log("The path is null. Please check whether the argument is valid.")
        raise MsOpGenException(ConstManager.MS_OP_GEN_INVALID_PATH_ERROR)
    path = os.path.realpath(path)
    if isdir and not os.path.exists(path):
        try:
            os.makedirs(path, mode=0o700)
        except OSError as ex:
            print_error_log(
                'Failed to create {}. Please check the path permission or '
                'disk space. {} '.format(path, str(ex)))
            raise MsOpGenException(ConstManager.MS_OP_GEN_INVALID_PATH_ERROR)
        finally:
            pass
    if not os.path.exists(path):
        print_error_log('The path {} does not exist. Please check whether '
                        'the path exists.'.format(path))
        raise MsOpGenException(ConstManager.MS_OP_GEN_INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log('You do not have the read permission on the path {} .'
                        'Please check.'.format(path))
        raise MsOpGenException(ConstManager.MS_OP_GEN_INVALID_PATH_ERROR)

    if isdir and not os.access(path, os.W_OK):
        print_error_log('You do not have the write permission on the path {} .'
                        'Please check.'.format(path))
        raise MsOpGenException(ConstManager.MS_OP_GEN_INVALID_PATH_ERROR)

    if isdir:
        if not os.path.isdir(path):
            print_error_log('The path {} is not a directory.'
                            ' Please check the path.'.format(path))
            raise MsOpGenException(ConstManager.MS_OP_GEN_INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log('The path {} is not a file.'
                            ' Please check the path.'.format(path))
            raise MsOpGenException(ConstManager.MS_OP_GEN_INVALID_PATH_ERROR)


def copy_template(src: str, dst: str, is_skip_exist: bool = False) -> None:
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
        raise MsOpGenException(ConstManager.MS_OP_GEN_WRITE_FILE_ERROR)


def copy_src_to_dst(srcname: str, dstname: str, is_skip_exist: bool) -> bool:
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
    return False


def copy_exist_file(dstname: str, is_skip_exist: bool) -> bool:
    """
    copy file is exist
    :param dstname: dest sub dir
    :param is_skip_exist: skip when dir is exist
    """
    if os.path.isdir(dstname) and len(os.listdir(dstname)) != 0:
        if is_skip_exist:
            return True
        print_error_log("{} is not empty. Please check.".format(dstname))
        sys.exit(ConstManager.MS_OP_GEN_INVALID_PATH_ERROR)
    return False


def get_content_from_double_quotes(line: str) -> any:
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


def make_dirs(op_dir: str) -> None:
    """
    make dirs
    :param op_dir:dirs
    """
    try:
        if not os.path.isdir(op_dir) or not os.path.exists(op_dir):
            os.makedirs(op_dir, ConstManager.FOLDER_MASK)
    except OSError as err:
        print_error_log("Unable to make dir: %s." % str(err))
        raise MsOpGenException(ConstManager.MS_OP_GEN_MAKE_DIRS_ERROR)
    finally:
        pass


def read_file(op_file: str) -> None:
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
        raise MsOpGenException(ConstManager.MS_OP_GEN_READ_FILE_ERROR)
    finally:
        pass


def write_files(op_file: str, new_str: str) -> None:
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
        with os.fdopen(os.open(op_file, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES), 'w') as \
                fout:
            fout.write(new_str)
    except OSError as err:
        print_error_log("Unable to write file(%s): %s." % op_file % str(err))
        raise MsOpGenException(ConstManager.MS_OP_GEN_WRITE_FILE_ERROR)
    finally:
        pass
    print_info_log("File %s generated successfully." % op_file)


def write_json_file(json_path: str, content: str) -> None:
    """
    write  content to json file
    :param content:
    :param json_path: the json path
    :return: the json object
    """
    try:
        with os.fdopen(os.open(json_path, ConstManager.WRITE_FLAGS,
                               ConstManager.WRITE_MODES), 'w+') as file_object:
            file_object.write(
                json.dumps(content, sort_keys=False, indent=4))
    except IOError as io_error:
        print_error_log(
            'Failed to generate json file %s. %s' % (json_path, str(io_error)))
        raise MsOpGenException(ConstManager.MS_OP_GEN_WRITE_FILE_ERROR)
    finally:
        pass
    print_info_log(
        "Generate file %s successfully." % json_path)


def fix_name_lower_with_under(name: str) -> str:
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
