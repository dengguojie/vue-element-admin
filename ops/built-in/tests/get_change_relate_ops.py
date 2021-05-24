import os
import sys
from typing import List, Dict

cur_dir = os.path.realpath(__file__)
repo_root = os.path.sep.join(cur_dir.split(os.path.sep)[:-4])
ini_cfg_root = os.path.join(repo_root, "ops", "built-in", "tbe", "op_info_cfg", "ai_core")
impl_file_root = os.path.join(repo_root, "ops", "built-in", "tbe", "impl")


def get_op_info_from_ini_file(ini_file_full_path, no_sep_module_map: Dict[str, str]):
    with open(ini_file_full_path) as ini_f:
        lines = ini_f.readlines()
    op_list = {}
    op_type = None
    for line in lines:
        if line.startswith("["):
            if op_type is not None:
                static_name = no_sep_module_map.get("impl." + op_type.lower().replace("_", "").strip())
                unknown_shape_name = no_sep_module_map.get("impl.dynamic." + op_type.lower().replace("_", "").strip())
                op_module = static_name if static_name else unknown_shape_name
                if op_module:
                    op_module = op_module.split(".")[-1]
                op_list[op_type] = op_module

            op_type = line.strip()[1:-1]
        elif line.startswith("opFile.value"):
            last_name = line[line.index("=") + 1:].strip()
            op_list[op_type] = last_name
            op_type = None

        if op_type is not None:
            static_name = no_sep_module_map.get("impl." + op_type.lower().replace("_", "").strip())
            unknown_shape_name = no_sep_module_map.get("impl.dynamic" + op_type.lower().replace("_", "").strip())
            op_module = static_name if static_name else unknown_shape_name
            if op_module:
                op_module = op_module.split(".")[-1]
            op_list[op_type] = op_module

    return op_list


def get_op_list(no_sep_module_map: Dict[str, str]):
    total_op_module_map = {}
    for ini_path, _, ini_file_list in os.walk(ini_cfg_root):
        for ini_file_name in ini_file_list:
            op_type_module_map = get_op_info_from_ini_file(os.path.join(ini_path, ini_file_name), no_sep_module_map)
            for key, value in op_type_module_map.items():
                if key not in total_op_module_map.keys():
                    total_op_module_map[key] = []
                if value not in total_op_module_map[key]:
                    total_op_module_map[key].append(value)

    return total_op_module_map


def impl_file_op_relation_map():
    depends_map = {}
    module_name_file_map = {}
    for file_path, _, file_list in os.walk(impl_file_root):
        if file_path.endswith("__pycache__"):
            continue
        parent_module_name = file_path[len(impl_file_root) - 4:].replace(os.path.sep, ".")
        for file_name in file_list:
            if file_name == "__init__.py":
                continue
            module_name = parent_module_name + "." + file_name[:-3].strip()
            module_name_file_map[module_name] = os.path.join(file_path, file_name)
            depend_modules = analysis_module_relation(os.path.join(file_path, file_name), parent_module_name)
            depend_impl_modules = []
            for depend_module in depend_modules:
                if depend_module.startswith("impl"):
                    depend_impl_modules.append(depend_module)
            depends_map[module_name] = depend_impl_modules
    recursion_depend_map = recursion_depend(depends_map)

    module_name_depend_files = {}
    for module_name, depend_modules in recursion_depend_map.items():
        depend_files = []
        for m_name in module_name_file_map.keys():
            for depend_module in depend_modules:
                if m_name == depend_module or (m_name + ".") in depend_module:
                    depend_file = module_name_file_map.get(m_name)
                    if depend_file not in depend_files:
                        depend_files.append(depend_file)
                    depend_modules.remove(depend_module)
        module_name_depend_files[module_name] = depend_files
    return module_name_depend_files


def get_depend_module(module_name, depends_map, recursion_path, all_depend_modules: List):
    if module_name in recursion_path:
        return
    recursion_path.append(module_name)
    depend_modules = depends_map.get(module_name, None)
    if depend_modules is None:
        module_tree = str(module_name).split(".")
        if len(module_tree) > 1:
            parent_module_name = ".".join(module_tree[:-1])
            get_depend_module(parent_module_name, depends_map, recursion_path, all_depend_modules)
            if parent_module_name in depends_map.keys() and parent_module_name not in all_depend_modules:
                all_depend_modules.append(parent_module_name)
            return
    else:
        for depend_module in depend_modules:
            if depend_module in depends_map.keys() and depend_module not in all_depend_modules:
                all_depend_modules.append(depend_module)
            get_depend_module(depend_module, depends_map, recursion_path, all_depend_modules)


def recursion_depend(depends_map):
    recursion_depend_map = {}
    for module_name, depend_modules in depends_map.items():
        all_depends_modules = [module_name, ]
        recursion_path = []
        get_depend_module(module_name, depends_map, recursion_path, all_depends_modules)
        recursion_depend_map[module_name] = all_depends_modules
    return recursion_depend_map


def analysis_module_relation(file_path, module_name):
    with open(file_path) as py_f:
        lines = py_f.readlines()
    depend_modules = []
    pre_line = None
    for line in lines:
        line = line.rstrip()
        if line.endswith("\\"):
            if pre_line is not None:
                pre_line += line[:-1]
            else:
                pre_line = line[:-1]
            continue
        else:
            if pre_line is not None:
                line = pre_line + line
            pre_line = None
        if line.startswith("import "):
            if " as " in line:
                depend_modules.append(line[7:line.index(" as ")].strip())
            else:
                import_modules = line[7:].strip().split(",")
                for import_module in import_modules:
                    depend_modules.append(import_module)
        elif line.startswith("from "):
            from_module = line[4:line.index(" import ")].strip()
            if from_module.startswith("."):
                from_module = module_name + from_module
            import_modules = line[line.index(" import ") + 8:].strip()
            if " as " in import_modules:
                import_modules = import_modules[:import_modules.index(" as ")].strip()
            import_modules = import_modules.split(",")
            for import_module in import_modules:
                depend_modules.append(from_module + "." + import_module.strip())
        else:
            continue
    return depend_modules


def op_file_map():
    module_file_map = impl_file_op_relation_map()
    no_sep_module_map = {}
    for key in module_file_map.keys():
        no_sep_module_map[key.replace("_", "").lower().strip()] = key
    all_op_list = get_op_list(no_sep_module_map)
    op_type_file_map = {}
    for op_type, op_modules in all_op_list.items():
        op_file_list = []
        for op_module in op_modules:
            if not op_module:
                print("%s op type has not found any impl file." % op_type)
                continue
            for op_file in module_file_map.get("impl." + op_module, []):
                if op_file not in op_file_list:
                    op_file_list.append(op_file)
            for op_file in module_file_map.get("impl.dynamic." + op_module, []):
                if op_file not in op_file_list:
                    op_file_list.append(op_file)

        op_type_file_map[op_type] = op_file_list
    return op_type_file_map


def check_op_not_used_file():
    op_type_file_map = op_file_map()
    for file_path, _, file_list in os.walk(impl_file_root):
        if file_path.endswith("__pycache__"):
            continue
        for file_name in file_list:
            if file_name == "__init__.py":
                continue

            file_full_path = os.path.join(file_path, file_name)
            find_op = False
            for op_type, op_files in op_type_file_map.items():
                if file_full_path in op_files:
                    find_op = True
                    break
            if not find_op:
                print("File %s not used by any op." % file_full_path)


def get_change_file_op(change_file_list):
    if not change_file_list:
        return []
    op_type_file_map = op_file_map()
    effect_ops = []
    for op_type, op_files in op_type_file_map.items():
        for op_file in op_files:
            if op_file in change_file_list:
                effect_ops.append(op_type)
                break
    return effect_ops

def get_case_change_op(change_dir_list):
    print("[INFO]change_dir_list is ", str(change_dir_list))
    if not change_dir_list:
        return []
    op_type_file_map = op_file_map()
    effect_ops = []
    for op_type, _ in op_type_file_map.items():
        op_type_dir = os.path.join(repo_root, "ops", "built-in", "tests", "st", str(op_type))
        if op_type_dir in change_dir_list:
            effect_ops.append(str(op_type))
            break
    return effect_ops


class FileChangeInfo:
    def __init__(self, ops_files=[], op_test_frame_files=[], test_files=[], other_files=[]):
        self.op_files = ops_files
        self.op_test_frame_files = op_test_frame_files
        self.test_files = test_files
        self.other_files = other_files

    def print_change_info(self):
        print("=========================================================================")
        print("changed file info")
        print("-------------------------------------------------------------------------")
        print("op changed files: \n%s" % "\n".join(self.op_files))
        print("-------------------------------------------------------------------------")
        print("op test frame changed files: \n%s" % "\n".join(self.op_test_frame_files))
        print("-------------------------------------------------------------------------")
        print("op test changed files: \n%s" % "\n".join(self.test_files))
        print("-------------------------------------------------------------------------")
        print("other changed files: \n%s" % "\n".join(self.other_files))
        print("=========================================================================")


def get_file_change_info_from_ci(changed_file_info_from_ci):
    """
    get file change info from ci, ci will write `git diff > /pr_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or FileChangeInf
    """
    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        print("[ERROR] %s file is not exist, can not get file change info in this pull request." % or_file_path)
        return None
    with open(or_file_path) as or_f:
        lines = or_f.readlines()
    ops_changed_files = []
    test_change_files = []
    op_test_frame_changed_files = []
    other_changed_files = []
    for line in lines:
        line = line.strip()
        if line.startswith(os.path.join("ops", "built-in", "tests")):
            test_change_files.append(line)
        elif line.startswith(os.path.join("ops", "built-in")):
            ops_changed_files.append(line)
        elif line.startswith(os.path.join("tools")):
            op_test_frame_changed_files.append(line)
        else:
            other_changed_files.append(line)
    return FileChangeInfo(ops_files=ops_changed_files, op_test_frame_files=op_test_frame_changed_files,
                          test_files=test_change_files, other_files=other_changed_files)


def get_change_relate_op_type_list(changed_file_info_from_ci):
    file_change_info = get_file_change_info_from_ci(changed_file_info_from_ci)
    if not file_change_info:
        print("[ERROR] not found file change info, run failed.")
        return None
    file_change_info.print_change_info()

    def _get_relate_op_list_by_file_change():
        relate_op_type_list = []

        def _deal_ops_file_change():
            ops_changed_files = file_change_info.op_files
            if not ops_changed_files:
                return
            impl_changed_files = []
            for ops_changed_file in ops_changed_files:
                ops_changed_file = str(ops_changed_file)
                if ops_changed_file.startswith(os.path.join("ops", "built-in", "tbe", "impl")) \
                        and ops_changed_file.endswith("py"):
                    ops_changed_file = os.path.join(repo_root, ops_changed_file)
                    impl_changed_files.append(ops_changed_file)

            op_types = get_change_file_op(impl_changed_files)
            for op_type in op_types:
                if op_type not in relate_op_type_list:
                    relate_op_type_list.append(op_type)

        _deal_ops_file_change()

        def _deal_op_test_frame_change():
            case_changed_files = file_change_info.test_files
            if not case_changed_files:
                return
            case_changed_dirs = []

            for case_changed_file in case_changed_files:
                case_changed_file = str(case_changed_file)
                if case_changed_file.startswith(os.path.join("ops", "built-in", "tests", "st")) \
                   and (case_changed_file.endswith("json") or case_changed_file.endswith("py")):
                    case_changed_file = os.path.join(repo_root, case_changed_file)
                    case_changed_dir,_ = os.path.split(case_changed_file)
                    case_changed_dirs.append(case_changed_dir)

            op_types = get_case_change_op(case_changed_dirs)
            for op_type in op_types:
                if op_type not in relate_op_type_list:
                    relate_op_type_list.append(op_type)

            if not relate_op_type_list:
                # if test frame change, test one op at least
                relate_op_type_list.append("Add")

        _deal_op_test_frame_change()

        return relate_op_type_list

    try:
        relate_directory_list = _get_relate_op_list_by_file_change()
    except BaseException as e:
        print(e.args)
        return None
    if relate_directory_list:
        print("[INFO] relate op directory list is: [%s]" % ", ".join(relate_directory_list))
    else:
        print("[INFO] relate directory list is empty")
    return relate_directory_list


if __name__ == '__main__':
    pr_changed_file = ""
    if len(sys.argv) >= 2:
        pr_changed_file = sys.argv[1]
    else:
        pr_changed_file = "pr_filelist.txt"
    case_dirs = get_change_relate_op_type_list(pr_changed_file)
    print("related_ops_dirs=%s" % ' '.join(case_dirs))
