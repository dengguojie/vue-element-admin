import json
import os
import stat
import sys

tbe_ops = {}
dir_path = os.path.dirname(os.path.abspath(__file__))


def parse_ini_files(ini_files):
    tbe_ops_info = {}
    parse_ini_to_obj(ini_files, tbe_ops_info)
    return tbe_ops_info


def parse_ini_to_obj(ini_file, tbe_ops_info):
    print("FFF %s"%ini_file)
    with open(ini_file) as ini_file:
        lines = ini_file.readlines()
        op = {}
        op_name = ""
        for line in lines:
            line = line.rstrip()
            if line.startswith("["):
                op_name = line[1:-1]
                op = {}
                tbe_ops_info[op_name] = op
            else:
                key1 = line[:line.index("=")]
                key2 = line[line.index("=")+1:]
                key1_0, key1_1 = key1.split(".")
                if not key1_0 in op:
                    op[key1_0] = {}
                if key1_1 in op[key1_0]:
                  raise RuntimeError("Op:" + op_name + " " + key1_0 + " " + key1_1 + " is repeated!")
                op[key1_0][key1_1] = key2

def check_op_info(tbe_ops):
    print("\n\n==============check valid for ops info start==============")
    not_valid_op=[]
    required_op_input_info_keys = ["paramType", "name"]
    required_op_output_info_keys = ["paramType", "name"]
    shape_type_valid_value = ["fix", "range", "list"]
    need_compile_valid_value = ["true", "false"]
    param_type_valid_value = ["dynamic", "optional", "required"]
    required_attr_key = ["type", "value", "paramType"]
    is_valid = True
    for op_key in tbe_ops:
        op = tbe_ops[op_key]

        for op_info_key in op:
            if op_info_key.startswith("input"):
                op_input_info = op[op_info_key]
                missing_keys=[]
                for  required_op_input_info_key in required_op_input_info_keys:
                    if not required_op_input_info_key in op_input_info:
                        missing_keys.append(required_op_input_info_key)
                if len(missing_keys) > 0:
                    print("op: " + op_key + " " + op_info_key +" missing: " + ",".join(missing_keys))
                    is_valid = False
                else:
                    if not op_input_info["paramType"] in param_type_valid_value:
                        print("op: " + op_key + " " + op_info_key +" paramType not valid, valid key:[dynamic, optional, required]")
                        is_valid = False
            if op_info_key.startswith("output"):
                op_input_info = op[op_info_key]
                missing_keys=[]
                for  required_op_input_info_key in required_op_output_info_keys:
                    if not required_op_input_info_key in op_input_info:
                        missing_keys.append(required_op_input_info_key)
                if len(missing_keys) > 0:
                    print("op: " + op_key + " " + op_info_key +" missing: " + ",".join(missing_keys))
                    is_valid = False
                else:
                    if not op_input_info["paramType"] in param_type_valid_value:
                        print("op: " + op_key + " " + op_info_key +" paramType not valid, valid key:[fix, range, list]")
                        is_valid = False
    print("==============check valid for ops info end================\n\n")
    return is_valid


def write_json_file(tbe_ops_info, json_file_path):
    json_file_real_path = os.path.realpath(json_file_path)
    with open(json_file_real_path, "w") as f:
        # Only the owner and group have rights
        os.chmod(json_file_real_path, stat.S_IWGRP + stat.S_IWUSR + stat.S_IRGRP + stat.S_IRUSR)
        json.dump(tbe_ops_info, f, sort_keys=True, indent=4, separators=(',', ':'))
    print("Compile op info cfg successfully.")


def parse_ini_to_json(ini_file_paths, outfile_path):
    tbe_ops_info = parse_ini_files(ini_file_paths)
    if not check_op_info(tbe_ops_info):
        print("Compile op info cfg failed.")
        return False
    write_json_file(tbe_ops_info, outfile_path)
    return True


if __name__ == '__main__':
    args = sys.argv

    outfile_path = "tbe_ops_info.json"
    ini_file_paths = []

    if len(args) != 2:
        print("Command format: python parser_ini.py dir")
        sys.exit(-1)

    folder_path = args[1]
    print("Parse ini file in %s"%folder_path)
    if not os.path.exists(folder_path):
        print("The folder %s is not exists"%folder_path)
        sys.exit(-1)

    parse_file_list = dict()
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            if name.endswith("ini"):
                file_path = os.path.join(root, name)
                parse_file_list[name.replace('ini', 'json')] = file_path.replace('\\', '/').replace('\\\\', '/')

    if parse_file_list is None:
        print("Fail: cannot find ini file in %s"%folder_path)
        sys.exit(0)

    print(parse_file_list)

    for outfile_path, ini_file_paths in parse_file_list.items():
        if not parse_ini_to_json(ini_file_paths, outfile_path):
            sys.exit(1)

    sys.exit(0)
