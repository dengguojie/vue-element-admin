import os
import shutil
import filecmp
import difflib


def clear_out_path(out_path):
    path = os.path.relpath(out_path)
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def compare_with_golden(result_path, golden_path):
    result_path = os.path.relpath(result_path)
    golden_path = os.path.join(os.path.relpath(golden_path))
    names = os.listdir(golden_path)
    shield_list = ['scripts', 'build.sh', 'CMakeLists', 'toolchain.cmake', 'cmake']
    for name in names:
        if any(shield in name for shield in shield_list):
            continue
        src_name = os.path.join(result_path, name)
        dst_name = os.path.join(golden_path, name)
        if os.path.isdir(src_name):
            if not compare_with_golden(src_name, dst_name):
                print(" %s VS %s return false." % (src_name, dst_name))
                return False
        else:
            if os.path.isfile(src_name):
                if not filecmp.cmp(src_name, dst_name):
                    print(" %s VS %s return false." % (src_name, dst_name))
                    return False
            else:
                print("The %s is not exist." % src_name)
                return False
    return True


def check_result(result_path, golden_path):
    if not compare_with_golden(result_path, golden_path):
        return False
    return True


def check_file_context(result_path, golden_path):
    result_path = os.path.relpath(result_path)
    golden_path = os.path.join(os.path.relpath(golden_path))
    golden_names = os.listdir(golden_path)
    result_names = os.listdir(result_path)
    src_name = os.path.join(result_path, result_names[0])
    dst_name = os.path.join(golden_path, golden_names[0])
    if not filecmp.cmp(src_name, dst_name):
        print(" %s VS %s return false." % (src_name, dst_name))
        return False
    return True
