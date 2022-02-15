import json
import os
import stat
import sys


def parse_ini_files(ini_files):
    """ Parse ini files. """
    aicpu_ops_info = {}
    print(ini_files)
    parse_ini_to_obj(ini_files, aicpu_ops_info)
    return aicpu_ops_info


def parse_ini_to_obj(ini_file, aicpu_ops_info):
    """ Parse ini file to object. """
    with open(ini_file) as inif:
        lines = inif.readlines()
        op = {}
        for line in lines:
            line = line.rstrip()
            if line.startswith("["):
                op_name = line[1:-1]
                op = {}
                aicpu_ops_info[op_name] = op
            else:
                key1 = line[:line.index("=")]
                key2 = line[line.index("=")+1:]
                key1_0, key1_1 = key1.split(".")
                if key1_0 not in op:
                    op[key1_0] = {}
                op[key1_0][key1_1] = key2


def check_op_info(aicpu_ops):
    """ Check op info. """
    print("\n==============check valid for ops info start==============")
    required_op_info_keys = ["computeCost", "engine", "flagAsync", "flagPartial", "opKernelLib"]

    for op_key in aicpu_ops:
        op = aicpu_ops[op_key]
        for key in op:
            if key == "opInfo":
                op_info = op["opInfo"]
                missing_keys = []
                for  required_op_info_key in required_op_info_keys:
                    if required_op_info_key not in op_info:
                        missing_keys.append(required_op_info_key)
                if len(missing_keys) > 0:
                    print("op: " + op_key + " opInfo missing: " + ",".join(missing_keys))
            elif (key[:5] == "input") and (key[5:].isdigit()):
                for op_sets in op[key]:
                    if op_sets not in ('format', 'type', 'name'):
                        print("input should has format type or name as the key, "
                              "but getting %s" % op_sets)
                        raise KeyError("bad op_sets key")
            elif (key[:6] == "output") and (key[6:].isdigit()):
                for op_sets in op[key]:
                    if op_sets not in ('format', 'type', 'name'):
                        print("output should has format type or name as the key, "
                              "but getting %s" % op_sets)
                        raise KeyError("bad op_sets key")
            elif (key[:13] == "dynamic_input") and (key[13:].isdigit()):
                for op_sets in op[key]:
                    if op_sets not in ('format', 'type', 'name'):
                        print("output should has format type or name as the key, "
                              "but getting %s" % op_sets)
                        raise KeyError("bad op_sets key")
            elif (key[:14] == "dynamic_output") and (key[14:].isdigit()):
                for op_sets in op[key]:
                    if op_sets not in ('format', 'type', 'name'):
                        print("output should has format type or name as the key, "
                              "but getting %s" % op_sets)
                        raise KeyError("bad op_sets key")
            else:
                print("Only opInfo, input[0-9], output[0-9] can be used as a key, "
                      "but op %s has the key %s" % (op_key, key))
                raise KeyError("bad key value")
    print("==============check valid for ops info end================\n")


def write_json_file(aicpu_ops_info, json_file_path):
    """ Write json file. """
    json_file_real_path = os.path.realpath(json_file_path)
    with open(json_file_real_path, "w") as f:
        # Only the owner and group have rights
        os.chmod(json_file_real_path, stat.S_IWGRP + stat.S_IWUSR + stat.S_IRGRP + stat.S_IRUSR)
        json.dump(aicpu_ops_info, f, sort_keys=True, indent=4, separators=(',', ':'))


def parse_ini_to_json(inf_paths, ouf_path):
    """ Parse ini file to json. """
    aicpu_ops_info = parse_ini_files(inf_paths)
    try:
        check_op_info(aicpu_ops_info)
    except KeyError:
        print("bad format key value, failed to generate json file")
    finally:
        write_json_file(aicpu_ops_info, ouf_path)
        print("parse_ini_to_json try except normal")

if __name__ == '__main__':
    args = sys.argv

    outfile_path = "tf_kernel.json"
    ini_file_paths = []


    if len(args) != 2:
        print("Command format: python parser_ini.py dir")
        sys.exit(-1)

    folder_path = args[1]
    print("Parse ini file in %s" % folder_path)
    if not os.path.exists(folder_path):
        print("The folder %s is not exists" % folder_path)
        sys.exit(-1)

    parse_file_list = {}
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            if name.endswith("ini"):
                file_path = os.path.join(root, name)
                parse_file_list[name.replace('ini', 'json')] = file_path.replace('\\', '/').replace('\\\\', '/')

    if parse_file_list is None:
        print("Fail: cannot find ini file in %s" % folder_path)
        sys.exit(0)

    for outf_path, inif_paths in parse_file_list.items():
        parse_ini_to_json(inif_paths, outf_path)

    sys.exit(0)
