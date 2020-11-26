import os

cur_file_path = __file__
ini_path = os.path.dirname(cur_file_path)  # llt/ops/llt_new/ut/cfg_format_check
ini_path = os.path.dirname(ini_path)  # llt/ops/llt_new/ut
ini_path = os.path.dirname(ini_path)  # llt/ops/llt_new
ini_path = os.path.dirname(ini_path)  # llt/ops
ini_path = os.path.dirname(ini_path)  # llt
ini_path = os.path.dirname(ini_path)  #
ini_path = os.path.join(ini_path, "cann/ops/built-in/tbe/op_info_cfg/ai_core")  #


def fix_cfg(lines):
    has_dynamic = False
    has_pattern = False
    for line in lines:
        if "dynamicFormat.flag" in line :
           has_dynamic = True
        if "op.pattern" in line:
            has_pattern = True
    new_lines = []
    for line in lines:
        if "needCompile" in line:
            continue
        if "compute.cost" in line or "partial.flag" in line or "async.flag" in line:
            continue
        if ".format" in line:
            if has_dynamic or has_pattern:
                continue
        new_lines.append(line+"\n")
    return new_lines


def fix_file(cfg_file):
    with open(cfg_file) as c_f:
        lines = c_f.readlines()

    op_type = None
    op_lines = []
    new_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("["):
            if op_type is not None:
                new_lines += fix_cfg(op_lines)
            op_type = line[line.index("[")+1: line.index("]")]
            op_lines = [line, ]
        else:
            op_lines.append(line)

    with open(cfg_file, "w+") as c_f:
        c_f.writelines(new_lines)


for soc_dir in os.listdir(ini_path):
    soc_path = os.path.join(ini_path, soc_dir)
    for file_name in os.listdir(soc_path):
        if file_name.endswith(".ini"):
            # if "910" not in file_name:
            #     continue
            file_path = os.path.join(soc_path, file_name)
            print(file_name)
            fix_file(file_path)
