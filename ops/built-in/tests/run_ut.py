import os
import shutil
from absl import flags, app
from typing import List, Dict

import cube_ut_runner
from op_test_frame.ut import op_ut_runner
from op_test_frame.common import op_status

FLAGS = flags.FLAGS

flags.DEFINE_string("soc_version", None, "SOC VERSION")
flags.DEFINE_string("op", None, "Test case directory name of test op")
flags.DEFINE_string("case_name", None, "Case name, run which case")
flags.DEFINE_string("case_dir", None, "Case directory, test case in which directory")
flags.DEFINE_string("report_path", None, "which directory to save ut test report")
flags.DEFINE_string("cov_path", None, "which dirctory to save coverage report")
flags.DEFINE_string("simulator_lib_path", None, "the path to simulator libs")
flags.DEFINE_string("pr_changed_file", None, "git diff result file by ci, analyse relate ut by this file")
flags.DEFINE_integer("process_num", None, "process number")

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
    get file change info from ci, ci will write `git diff > /or_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or FileChangeInf
    """
    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        print("[ERROR] %s file is not exist, can not get file change info in this pull request.")
        return None
    with open(or_file_path) as or_f:
        lines = or_f.readlines()
    ops_changed_files = []
    test_change_files = []
    op_test_frame_changed_files = []
    other_changed_files = []
    print("----------ci changed file content----------")
    print("".join(lines))
    print("-------------------------------------------")
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


def get_change_relate_ut_dir_list(changed_file_info_from_ci):
    file_change_info = get_file_change_info_from_ci(changed_file_info_from_ci)
    if not file_change_info:
        print("[ERROR] not found file change info, run ut failed.")
        return None
    file_change_info.print_change_info()

    def _get_relate_ut_list_by_file_change():
        relate_ut_dir_list = []

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
            if not op_types:
                print("[INFO] op file changes affect none op.")
                return
            print("[INFO] op file changes affect ops: [%s]" % ",".join(op_types))
            for op_type in op_types:
                op_ut_test_dir = os.path.join(repo_root, "ops", "built-in", "tests", "ut", "ops_test", op_type)
                if not os.path.exists(op_ut_test_dir):
                    error_msg = "[ERROR] This commit will affect op: " + op_type
                    error_msg += ", but this op has not found test case file. "
                    error_msg += "Should has test case file like:"
                    error_msg += " built-in/tests/ut/ops_test/" + op_type + "/test_*_impl.py"
                    print(error_msg)
                    #raise RuntimeError("Not found test case directory")
                relate_ut_dir_list.append(op_ut_test_dir)

        _deal_ops_file_change()

        def _deal_test_file_change():
            test_changed_files = file_change_info.test_files
            if not test_changed_files:
                return
            for test_changed_file in test_changed_files:
                test_changed_file = str(test_changed_file).strip()
                test_case_dir = os.path.join("ops", "built-in", "tests", "ut", "ops_test")
                in_ut_dir = test_changed_file.startswith(test_case_dir)
                test_changed_file_split = test_changed_file.split(os.path.sep)
                test_changed_file_name = test_changed_file_split[-1]
                file_name_match = test_changed_file_name.startswith("test_")
                file_name_match = file_name_match and test_changed_file_name.endswith("_impl.py")

                if in_ut_dir and file_name_match:
                    if not len(test_changed_file_split) == 7:
                        raise RuntimeError(
                            "Can only add test case file like: built-in/tests/ut/ops_test/Add/test_*_impl.py.")
                    op_ut_test_dir = os.path.join(repo_root, test_case_dir, test_changed_file.split(os.path.sep)[-2])
                    if op_ut_test_dir not in relate_ut_dir_list:
                        relate_ut_dir_list.append(os.path.join(op_ut_test_dir, test_changed_file_name))
                        # not need test all test in dir, relate_ut_dir_list.append(op_ut_test_dir)

        _deal_test_file_change()

        def _deal_op_test_frame_change():
            if not relate_ut_dir_list:
                relate_ut_dir_list.append(os.path.join(repo_root, "ops", "built-in", "tests", "ut", "ops_test", "Add"))

        _deal_op_test_frame_change()

        return relate_ut_dir_list

    try:
        relate_ut_directory_list = _get_relate_ut_list_by_file_change()
    except BaseException as e:
        print(e.args)
        return None
    if relate_ut_directory_list:
        print("[INFO] relate ut directory list is: [%s]" % ", ".join(relate_ut_directory_list))
    else:
        print("[INFO] relate ut directory list is empty")
    return relate_ut_directory_list

def get_cube_case_dir(case_dir):
    cube_case_dir = []
    vector_case_dir = []
    cube_ops = ['AvgPool', 'AvgPool3D', 'AvgPool3DD',
                'AvgPool3DGrad', 'AvgPool3DGradD',
                'AvgPoolGrad', 'AvgPoolGradD',
                'AvgPoolV2', 'AvgPoolV2GradD', 'Avg_pool_v2_grad_d',
                'BatchMatmul', 'BatchMatmulV2',
                'Conv2D', 'Conv2DAipp',
                'Conv2DBackpropFilter', 'Conv2DBackpropFilterD',
                'Conv2DBackpropInput', 'Conv2DBackpropInputD',
                'Conv2DBn1', 'Conv2DCompress', 'Conv2DConv', 'Conv2DDim',
                'Conv2DErr', 'Conv2DSreadSwrite',
                'Conv2DTranspose', 'Conv2DTransposeD',
                'Conv2DV200', 'Conv2D_Compress',
                'Conv2D_Lx_Fusion', 'Conv2D_Vector_Fused',
                'Deconvolution',
                'Conv3D',
                'Conv3DBackpropFilter', 'Conv3DBackpropFilterD',
                'Conv3DBackpropInput', 'Conv3DBackpropInputD',
                'Conv3DTranspose', 'Conv3DTransposeD',
                'DepthwiseConv2D',
                'DepthwiseConv2DBackpropFilter',
                'DepthwiseConv2DBackpropFilterD',
                'DepthwiseConv2DBackpropInput',
                'DepthwiseConv2DBackpropInputD',
                'DepthwiseConv2D_Fused'
                ]
    base_ut_dir = os.path.join(repo_root, "ops", "built-in", "tests", "ut", "ops_test")
    if isinstance(case_dir, str):
        for item in os.listdir(base_ut_dir):
            item_dir = os.path.join(base_ut_dir, item)
            if item in cube_ops:
                cube_case_dir.append(item_dir)
            elif os.path.isdir(item_dir):
                vector_case_dir.append(item_dir)
    else:
        for item in case_dir:
            if os.path.basename(item) in cube_ops or os.path.basename(os.path.dirname(item)) in cube_ops:
                cube_case_dir.append(item)
            else:
                vector_case_dir.append(item)

    return cube_case_dir, vector_case_dir


def main(argv):
    _ = argv
    soc_version = FLAGS.soc_version
    soc_version = [soc.strip() for soc in str(soc_version).split(",")]
    pr_changed_file = FLAGS.pr_changed_file
    if not pr_changed_file or not str(pr_changed_file).strip():
        case_dir = FLAGS.case_dir
        if not case_dir:
            case_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ut", "ops_test")
    else:
        case_dir = get_change_relate_ut_dir_list(pr_changed_file)
        if case_dir is None:
            # get relate ut failed.
            exit(-1)
        for dir_item in case_dir:
            if not os.path.exists(dir_item):
                case_dir.remove(dir_item)
        if not case_dir:
            # has no relate ut, not need run ut.
            exit(0)

    if not pr_changed_file or not str(pr_changed_file).strip():
        print("Enter all op.")
        cov_report_path = FLAGS.cov_path + '_cube' if FLAGS.cov_path else "./cov_report/ops/python_utest_cube"
        report_path = FLAGS.report_path + '_cube'  if FLAGS.report_path else "./report/ops/python_report_cube"
        simulator_lib_path = FLAGS.simulator_lib_path if FLAGS.simulator_lib_path else "/usr/local/Ascend/toolkit/tools/simulator"
        process_num = FLAGS.process_num
        cube_res = cube_ut_runner.run_ut(case_dir,
                                         soc_version=soc_version,
                                         test_report="json",
                                         test_report_path=report_path,
                                         cov_report="html",
                                         cov_report_path=cov_report_path,
                                         simulator_mode="pv",
                                         simulator_lib_path=simulator_lib_path,
                                         process_num=process_num)
        dst_cov_report_path = FLAGS.cov_path if FLAGS.cov_path else "./cov_report/ops/python_utest"
        dst_report_path = FLAGS.report_path if FLAGS.report_path else "./report/ops/python_report"
        shutil.copytree(cov_report_path, dst_cov_report_path)
        shutil.copytree(report_path, dst_report_path)
        if cube_res != op_status.SUCCESS:
            exit(-1)
        exit(0)

    cube_case_dir, vector_case_dir = get_cube_case_dir(case_dir)
    if cube_case_dir:
        print("Enter cube op.")
        cov_report_path = FLAGS.cov_path + '_cube' if FLAGS.cov_path else "./cov_report/ops/python_utest_cube"
        report_path = FLAGS.report_path + '_cube'  if FLAGS.report_path else "./report/ops/python_report_cube"
        simulator_lib_path = FLAGS.simulator_lib_path if FLAGS.simulator_lib_path else "/usr/local/Ascend/toolkit/tools/simulator"
        process_num = FLAGS.process_num
        cube_res = cube_ut_runner.run_ut(cube_case_dir,
                                         soc_version=soc_version,
                                         test_report="json",
                                         test_report_path=report_path,
                                         cov_report="html",
                                         cov_report_path=cov_report_path,
                                         simulator_mode="pv",
                                         simulator_lib_path=simulator_lib_path,
                                         process_num=process_num)
        cube_cov_file = os.path.join(cov_report_path, ".coverage")
        if os.path.exists(cube_cov_file):
            if not os.path.exists(FLAGS.cov_path):
                os.makedirs(FLAGS.cov_path)
            dst_path = os.path.join(FLAGS.cov_path, '.coverage.cube')
            shutil.move(cube_cov_file, dst_path)
        if cube_res != op_status.SUCCESS:
            exit(-1)
  
    if vector_case_dir:
        print("Enter vector op.")
        cov_report_path = FLAGS.cov_path if FLAGS.cov_path else "./cov_report/ops/python_utest"
        report_path = FLAGS.report_path if FLAGS.report_path else "./report/ops/python_report"
        simulator_lib_path = FLAGS.simulator_lib_path if FLAGS.simulator_lib_path else "/usr/local/Ascend/toolkit/tools/simulator"
        process_num = FLAGS.process_num
        res = op_ut_runner.run_ut(vector_case_dir,
                                  soc_version=soc_version,
                                  test_report="json",
                                  test_report_path=report_path,
                                  cov_report="html",
                                  cov_report_path=cov_report_path,
                                  simulator_mode="pv",
                                  simulator_lib_path=simulator_lib_path,
                                  process_num=process_num)
        if res != op_status.SUCCESS:
            exit(-1)

    exit(0)


if __name__ == "__main__":
    app.run(main)
