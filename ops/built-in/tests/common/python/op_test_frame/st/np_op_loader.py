import os
import re
import sys
import types


def _build_np_op_key(np_func_name):
    return re.sub(r'[^0-9a-z]', "", np_func_name)


def _load_np_op_func(np_op_file, np_op_func_map):
    np_op_dir = os.path.dirname(np_op_file)
    np_op_module_name = os.path.basename(np_op_file)[:-3]
    sys_patch_added = False
    if np_op_dir not in sys.path:
        sys.path.insert(0, np_op_dir)
        sys_patch_added
    try:
        __import__(np_op_module_name)
    except BaseException as e:
        print(e)
        return False, "Load numpy op file failed, file: %" % np_op_file
    np_op_module = sys.modules[np_op_module_name]
    for key, val in np_op_module.__dict__.items():
        if str(key).startswith("np_") and isinstance(val, types.FunctionType):
            store_key = _build_np_op_key(key)
            np_op_func_map[store_key] = val
    if sys_patch_added:
        sys.path.remove(np_op_dir)

    return True, ""


def _load_np_op_func_list(np_op_file_list):
    if not np_op_file_list:
        return {}
    error_msg_list = []
    np_op_func_map = {}
    load_failed = False
    for np_op_file in np_op_file_list:
        np_op_func, error_msg = _load_np_op_func(np_op_file, np_op_func_map)
        if not np_op_func:
            error_msg_list.append(error_msg)
            load_failed = True
    if load_failed:
        for error_msg in error_msg_list:
            print(error_msg)
        raise RuntimeError("Load np op funcs failed, please see previous message!")

    return np_op_func_map


def _scan_case_json_np_op(case_dir):
    case_files = os.listdir(case_dir)
    np_op_file_list = []
    for case_file in case_files:
        if os.path.isdir(os.path.join(case_dir, case_file)):
            if case_file.endswith(os.path.sep + "out"):
                # is st out tmp file, ignore it
                continue
            else:
                np_op_file_list_in_sub = _scan_case_json_np_op(os.path.join(case_dir, case_file))
                for np_op_file in np_op_file_list_in_sub:
                    np_op_file_list.append(np_op_file)
        else:
            if re.match(r'np_.*\.py$', case_file):
                np_op_file_list.append(os.path.join(case_dir, case_file))
    return np_op_file_list


def load_np_op_file(np_op_file):
    np_op_file_list = [np_op_file, ]
    np_op_map = _load_np_op_func_list(np_op_file_list)
    return np_op_map


def load_np_op(np_op_dir):
    np_op_file_list = _scan_case_json_np_op(np_op_dir)
    np_op_map = _load_np_op_func_list(np_op_file_list)
    return np_op_map
