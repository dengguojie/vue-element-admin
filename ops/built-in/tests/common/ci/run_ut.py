import os
import sys
import traceback
from typing import List
from typing import Dict
from absl import flags, app
from op_test_frame.ut import op_ut_runner
from op_test_frame.common import op_status

FLAGS = flags.FLAGS
flags.DEFINE_string("soc_version", None, "SOC VERSION")
flags.DEFINE_string("op", None, "Test case directory name of test op")
flags.DEFINE_string("case_name", None, "Case Name")
flags.DEFINE_string("case_dir", None, "Case directory")
flags.DEFINE_string("report_path", None, "report_path directory")
flags.DEFINE_string("cov_path", None, "cov_path directory")
flags.DEFINE_bool("auto_analyse", False, "cov_path directory")
flags.DEFINE_string("simulator_mode", None, "simulator_mode")
flags.DEFINE_string("simulator_lib_path", None, "simulator_lib_path")

cur_dir = os.path.realpath(__file__)
repo_root = os.path.sep.join(cur_dir.split(os.path.sep)[:-9])
ini_cfg_root = os.path.join(repo_root, "asl", "ops", "cann", "ops", "built-in", "tbe", "op_info_cfg", "ai_core")
impl_file_root = os.path.join(repo_root, "asl", "ops", "cann", "ops", "built-in", "tbe", "impl")


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
            for op_type, value in op_type_module_map.items():
                if op_type not in total_op_module_map.keys():
                    total_op_module_map[op_type] = []
                if value not in total_op_module_map[op_type]:
                    total_op_module_map[op_type].append(value)
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


# all_op_list = get_op_list()
#
# module_file_map = impl_file_op_relation_map()


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


class ChangeFileInfo:
    def __init__(self, ops_files=[], op_test_frame_files=[], tbe_files=[], llt_files=[], model_files=[],
                 other_files=[]):
        self.ops_files = ops_files
        self.op_test_frame_files = op_test_frame_files
        self.tbe_files = tbe_files
        self.llt_files = llt_files
        self.model_files = model_files
        self.other_files = other_files

    def print_change_info(self):
        print("========================================================")
        print("changed file info")
        print("--------------------------------------------------------")
        print("ops changed files: \n%s" % "\n".join(self.ops_files))
        print("--------------------------------------------------------")
        print("tbe changed files: \n%s" % "\n".join(self.tbe_files))
        print("--------------------------------------------------------")
        print("llt changed files: \n%s" % "\n".join(self.llt_files))
        print("--------------------------------------------------------")
        print("simulator model changed files: \n%s" % "\n".join(self.model_files))
        print("--------------------------------------------------------")
        print("other changed files: \n%s" % "\n".join(self.other_files))
        print("========================================================")


def get_changed_file():
    change_files_list_f = os.path.join(repo_root, "vendor", "hisi", "llt", "ci", "script", "changed_files_list")
    if not os.path.exists(change_files_list_f):
        print("[WARNING] vendor/hisi/llt/ci/script/changed_files_list is not exist, use default")
        change_files_list_f = os.path.join(os.path.dirname(__file__), "default_chnage_file.txt")
    with open(change_files_list_f) as c_f:
        lines = c_f.readlines()
    ops_files = []
    op_test_frame_files = []
    tbe_files = []
    llt_files = []
    simulator_model_files = []
    other_files = []
    for line in lines:
        line = line.strip()
        repo_and_file = line.split(":")
        if len(repo_and_file) != 2:
            print("[ERROR] changed_files_list info is wrong. file info is: \n%s" % ",".join(lines))
            return None
        repo_name, file_path = [x.strip() for x in repo_and_file]
        if not file_path:
            print("[ERROR] changed_files_list info is wrong. file info is: \n%s" % ",".join(lines))
            return None
        if repo_name == "toolchain/tensor_utils":
            op_test_frame_files.append(file_path)
        elif repo_name == "ops":
            ops_files.append(file_path)
        elif repo_name == "llt/ops":
            llt_files.append(file_path)
        elif repo_name == "tensor_engine":
            tbe_files.append(file_path)
        elif repo_name == "model":
            simulator_model_files.append(file_path)
        else:
            other_files.append(file_path)
    return ChangeFileInfo(ops_files=ops_files, op_test_frame_files=op_test_frame_files,
                          llt_files=llt_files, tbe_files=tbe_files, model_files=simulator_model_files,
                          other_files=other_files)


def get_schedule_file_relate_op(schedule_file_name):
    element_wise_ops = ["ReduceMeanD", "ReduceProductD", "ReduceMinD", "ReduceMaxD", "BNTrainingReduce",
                        "ReduceSum", "ReduceSumD", "ReduceAnyD", "ReduceAllD", "Reduction", "BNTrainingReduceGrad",
                        "INTrainingReduceV2", "GNTrainingReduce"]
    inplace_ops = ["InplaceUpdateD", "InplaceAddD", "InplaceSubD"]
    pool_ops = ["MaxPool", "MaxPoolExt2", "AvgPool", "Pooling"]
    conv2d_backprop_input_ops = ["Conv2DBackpropInput", "Conv2DBackpropInputD", "Conv2DTranspose",
                                 "Conv2DTransposeD", "Deconvolution"]
    mapping_config = [{
        "files": ["max_pool2d_schedule", "pooling2d_schedule"],
        "ops": pool_ops
    }, {
        "files": ["inplace_schedule"],
        "ops": inplace_ops
    }, {
        "files": ["elewise_multi_schedule", "elewise_schedule", "elewise_schedule_new", "elewise_speel_schedule",
                  "pure_broadcast_schedule", "pure_broadcast_intrin", "reduce_5hdc_intrin", "reduce_5hdc_schedule",
                  "reduce_atomic_schedule", "reduce_mean_mid_reduce_high_performance_schedule", "vector_schedule",
                  "reduce_multi_schedule"],
        "ops": element_wise_ops
    }, {
        "files": ["conv2d_backprop_input_schedule", "conv2d_backprop_input_opti_schedule",
                  "conv2d_backprop_input_general_schedule"],
        "ops": conv2d_backprop_input_ops
    }]
    for map_info in mapping_config:
        if schedule_file_name in map_info.get("files"):
            return map_info.get("ops")
    return []


def accurate_tbe_file_change(change_file_info: ChangeFileInfo, op_type_list):
    tbe_changed_files = change_file_info.tbe_files
    if not tbe_changed_files:
        return
    for tbe_file in tbe_changed_files:
        tbe_file = str(tbe_file).strip()
        if not ("auto_schedule/python/tbe/dsl/static_schedule" in tbe_file or "python/te/lang/dynamic/schedule" in tbe_file):
            continue
        if not tbe_file.endswith(".py"):
            continue
        file_name = os.path.basename(tbe_file)[:-3]
        op_types = get_schedule_file_relate_op(file_name)
        if not op_types:
            continue
        for op_type in op_types:
            if op_type not in op_type_list:
                op_type_list.append(op_type)


def accurate_ops_file_change(change_file_info: ChangeFileInfo, op_type_list):
    ops_file_list = change_file_info.ops_files
    if not ops_file_list:
        return
    impl_files = []
    for ops_file in ops_file_list:
        ops_file = str(ops_file).strip()
        if ops_file.startswith("ops/built-in/tbe/impl"):
            ops_file = os.path.join(repo_root, "ops", ops_file)
            if ops_file not in impl_files:
                impl_files.append(ops_file)
    ops_types = get_change_file_op(impl_files)
    for op_type in ops_types:
        if op_type not in op_type_list:
            op_type_list.append(op_type)


def accurate_llt_file_change(change_file_info: ChangeFileInfo, op_type_list):
    llt_files = change_file_info.llt_files
    if not llt_files:
        return
    has_other_py = False
    for llt_file in llt_files:
        llt_file = str(llt_file).strip()
        if llt_file.startswith("llt_new/ut/ops_test") and llt_file.endswith("py"):
            op_type = llt_file.split(os.path.sep)[-2]
            if op_type not in op_type_list:
                op_type_list.append(op_type)
        else:
            if llt_file.endswith("py"):
                has_other_py = True
    if not op_type_list and has_other_py:
        # not set
        op_type_list.append("Add")


def accurate_op_test_frame_change(change_file_info: ChangeFileInfo, op_type_list):
    op_test_frame_files = change_file_info.op_test_frame_files
    if not op_test_frame_files:
        return
    has_py = False
    for op_test_frame_file in op_test_frame_files:
        op_test_frame_file = str(op_test_frame_file).strip()
        if op_test_frame_file.endswith("py"):
            has_py = True
    if has_py and not op_type_list:
        op_type_list.append("Add")


def accurate_simulator_model_change(change_file_info: ChangeFileInfo, op_type_list):
    simulator_model_files = change_file_info.model_files
    if not simulator_model_files:
        return
    model_need_test_op = ["Abs", "Add", "Asin", "BatchMatmul", "Bias", "Eltwise", "Maximum", "Relu", "Conv2D", "Conv3D"]
    for op_type in model_need_test_op:
        if op_type not in op_type_list:
            op_type_list.append(op_type)


def accurate_to_op_type(change_file_info: ChangeFileInfo) -> List[str]:
    change_file_info.print_change_info()
    op_type_list = []
    print(">>>> add tbe file relate op types")
    accurate_tbe_file_change(change_file_info, op_type_list)
    print(">>>> need test op type: [%s]" % ", ".join(op_type_list))
    print(">>>> add ops file relate op types")
    accurate_ops_file_change(change_file_info, op_type_list)
    print(">>>> need test op type: [%s]" % ", ".join(op_type_list))
    print(">>>> add llt file relate op types")
    accurate_llt_file_change(change_file_info, op_type_list)
    print(">>>> need test op type: [%s]" % ", ".join(op_type_list))
    print(">>>> add op test frame file relate op types")
    accurate_op_test_frame_change(change_file_info, op_type_list)
    print(">>>> need test op type: [%s]" % ", ".join(op_type_list))
    print(">>>> add simulator model file relate op types")
    accurate_simulator_model_change(change_file_info, op_type_list)
    print(">>>> need test op type: [%s]" % ", ".join(op_type_list))
    if "Axpy" not in op_type_list:
        op_type_list.append("Axpy")
    if not op_type_list:
        print(">>>> not need test op ut")
    return op_type_list


def get_need_test_op_type():
    op_list = []
    try:
        op_list = get_need_test_op_type_with_error()
    except BaseException as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        trace_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
        err_trace_str = ""
        for t_i in trace_info:
            err_trace_str += t_i
        print(err_trace_str)
        return []

    return op_list


def get_need_test_op_type_with_error():
    changed_file_info = get_changed_file()
    if not changed_file_info:
        return []
    return accurate_to_op_type(changed_file_info)


def main(args):
    soc_version = FLAGS.soc_version
    soc_version = [soc.strip() for soc in str(soc_version).split(",")]
    case_dir = FLAGS.case_dir
    if FLAGS.auto_analyse:
        op_type_list = get_need_test_op_type()
        if op_type_list is None:
            exit(-1)
        if not op_type_list:
            exit(0)
        cur_path = os.path.realpath(__file__)
        ut_path = os.path.join(os.path.sep.join(cur_path.split(os.path.sep)[:-3]), "ut", "ops_test")
        case_dir = [os.path.join(ut_path, op_type) for op_type in op_type_list]
    cov_report_path = FLAGS.cov_path if FLAGS.cov_path else "./cov_report"
    report_path = FLAGS.report_path if FLAGS.report_path else "./report"
    print(case_dir)
    res = op_ut_runner.run_ut(case_dir, soc_version=soc_version, test_report="json", test_report_path=report_path,
                              simulator_mode="pv" if FLAGS.simulator_lib_path else None,
                              simulator_lib_path=FLAGS.simulator_lib_path,
                              cov_report="html",
                              cov_report_path=cov_report_path)
    if res == op_status.SUCCESS:
        exit(0)
    else:
        exit(-1)


if __name__ == "__main__":
    app.run(main)
