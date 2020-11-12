#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from common.config.llt_config import LLTConf

op_ut_root_dir = os.path.join(LLTConf.llt_root_path, "ut/ops_test")


def add_cases_to_file(op_type, case_class, op_module_name, op_func_name, case_str, soc_info=""):
    op_ut_path = os.path.join(op_ut_root_dir, op_type)
    if not os.path.exists(op_ut_path):
        os.makedirs(op_ut_path)
    op_ut_file = os.path.join(op_ut_path, "test_"+op_type+"_impl.py")
    op_ut_init_file = os.path.join(op_ut_path, "__init__.py")

    if not os.path.exists(op_ut_init_file):
        init_info_str="""#!/usr/bin/python
# -*- coding: UTF-8 -*-
#这个文件的作用是：
#包含这个文件，python ut工程就认为这个目录是python ut用例的目录，会自动包含这个目录下的所有用例，否则这个目录下的用例不会被执行
"""
        with open(op_ut_init_file, "w+") as init_f:
            init_f.write(init_info_str)

    case_added = False
    case_file_info = ""
    # already exist, update auto gen cases, don't change cases added artificial
    if os.path.exists(op_ut_file):
        with open(op_ut_file) as ut_f:
            lines = ut_f.readlines()

        old_case = False
        for line in lines:
            if "don't change me" in line:
                return
            if "auto gen %s test cases start" % soc_info in line:
                case_file_info += line
                old_case = True
                case_file_info += case_str
                case_added = True
            elif "auto gen %s test cases end" % soc_info in line:
                old_case = False
                case_file_info += line
            else:
                if not old_case:
                    case_file_info += line
    else:
        case_file_info = """#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import %s

ut_case = %s("%s", %s, %s)

""" % (case_class, case_class, op_type, "\"" + op_module_name + "\"" if op_module_name else "None",
       "\"" + op_func_name + "\"" if op_func_name else "None")
        case_file_info += "\n# ============ auto gen %s test cases start ===============\n" % soc_info
        case_file_info += case_str
        case_file_info += "# ============ auto gen %s test cases end =================\n\n" % soc_info

        case_file_info += """if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
"""

    with open(op_ut_file, "w+") as ut_f:
        ut_f.write(case_file_info)


def auto_gen_broadcast_op_ut(op_type, soc_dtypes=None, op_module_name=None, op_func_name=None):
    if soc_dtypes is None:
        soc_dtypes = [
            {
                "soc": ["Ascend910", "Ascend310"],
                "dtype": ["float16", "float32", "int32"],
            }, {
                "soc": ["Hi3796CV300ES"],
                "dtype": ["float16", "int32"],
            }]

    cases_str = ""
    for soc_dtypes in soc_dtypes:
        soc_str = "[\"" + "\", \"".join(soc_dtypes.get("soc")) +"\"]"
        dtype_str = "[\"" + "\", \"".join(soc_dtypes.get("dtype")) +"\"]"
        params = tuple([soc_str, dtype_str] * 17)
        cases_str += """ut_case.add_broadcast_case_simple(%s, %s, (1,), (1,))
ut_case.add_broadcast_case_simple(%s, %s, (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(%s, %s, (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(%s, %s, (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(%s, %s, (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(%s, %s, (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(%s, %s, (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(%s, %s, (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(%s, %s, (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(%s, %s, (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(%s, %s, (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(%s, %s, (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(%s, %s, (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(%s, %s, (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(%s, %s, (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(%s, %s, (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(%s, %s, (10, 13), (10, 11, 12), expect=RuntimeError)

""" % params

    add_cases_to_file(op_type, "BroadcastOpUT", op_module_name, op_func_name, cases_str, soc_str)


def auto_gen_grad_broadcast_op_ut(op_type, soc_dtypes=None, op_module_name=None, op_func_name=None):
    if soc_dtypes is None:
        soc_dtypes = [
            {
                "soc": ["Ascend910", "Ascend310"],
                "dtype": ["float16", "float32", "int32"],
            }, {
                "soc": ["Hi3796CV300ES"],
                "dtype": ["float16", "int32"],
            }]

    cases_str=""
    for soc_dtypes in soc_dtypes:
        soc_str = "[\"" + "\", \"".join(soc_dtypes.get("soc")) +"\"]"
        dtype_str = "[\"" + "\", \"".join(soc_dtypes.get("dtype")) +"\"]"
        params = tuple([soc_str, dtype_str] * 14)
        cases_str += """ut_case.add_broadcast_case_simple(%s, %s, (1,), (1,))
ut_case.add_broadcast_case_simple(%s, %s, (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(%s, %s, (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(%s, %s, (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(%s, %s, (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(%s, %s, (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(%s, %s, (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(%s, %s, (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(%s, %s, (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(%s, %s, (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(%s, %s, (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(%s, %s, (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(%s, %s, (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(%s, %s, (10, 13), (10, 11, 12), expect=RuntimeError)

""" % params

    add_cases_to_file(op_type, "BroadcastOpUT", op_module_name, op_func_name, cases_str, soc_str)


def auto_gen_reduce_op_ut(op_type, soc_dtypes=None, op_module_name=None, op_func_name=None):
    if soc_dtypes is None:
        soc_dtypes = [
            {
                "soc": ["Ascend910", "Ascend310"],
                "dtype": ["float16", "float32", "int32"],
            }, {
                "soc": ["Hi3796CV300ES"],
                "dtype": ["float16", "int32"],
            }]

    cases_str=""
    for soc_dtypes in soc_dtypes:
        soc_str = "[\"" + "\", \"".join(soc_dtypes.get("soc")) +"\"]"
        dtype_str = "[\"" + "\", \"".join(soc_dtypes.get("dtype")) +"\"]"
        params = tuple([soc_str, dtype_str] * 20)
        cases_str += """ut_case.add_reduce_case_simple(%s, %s, (1,), (0,), True)
ut_case.add_reduce_case_simple(%s, %s, (1,), 0, False)
ut_case.add_reduce_case_simple(%s, %s, (1, 1), (1,), True)
ut_case.add_reduce_case_simple(%s, %s, (1, 1), (1,), False)
ut_case.add_reduce_case_simple(%s, %s, (101, 10241), (-1, ), True)
ut_case.add_reduce_case_simple(%s, %s, (101, 10241), (-1, ), False)
ut_case.add_reduce_case_simple(%s, %s, (1023*255, ), (-1, ), True)
ut_case.add_reduce_case_simple(%s, %s, (1023*255, ), (-1, ), False)
ut_case.add_reduce_case_simple(%s, %s, (51, 101, 1023), (1, 2), True)
ut_case.add_reduce_case_simple(%s, %s, (51, 101, 1023), (1, 2), False)
ut_case.add_reduce_case_simple(%s, %s, (51, 101, 1023), (1, ), True)
ut_case.add_reduce_case_simple(%s, %s, (51, 101, 1023), (1, ), False)
ut_case.add_reduce_case_simple(%s, %s, (51, 101, 1023), (0, 1, 2), True)
ut_case.add_reduce_case_simple(%s, %s, (51, 101, 1023), (0, 1, 2), False)
ut_case.add_reduce_case_simple(%s, %s, (99991, 10), (0, ), True)
ut_case.add_reduce_case_simple(%s, %s, (99991, 10), (0, ), False)
ut_case.add_reduce_case_simple(%s, %s, (1, 99991), (1, ), True)
ut_case.add_reduce_case_simple(%s, %s, (1, 99991), (1, ), False)
ut_case.add_reduce_case_simple(%s, %s, (1, 99991, 10), (1, ), True)
ut_case.add_reduce_case_simple(%s, %s, (1, 99991, 10), (1, ), False)

""" % params

    add_cases_to_file(op_type, "ReduceOpUT", op_module_name, op_func_name, cases_str, soc_str)


def auto_gen_single_elewise_op_ut(op_type, soc_dtypes=None, op_module_name=None, op_func_name=None):
    if soc_dtypes is None:
        soc_dtypes = [
            {
                "soc": ["Ascend910", "Ascend310"],
                "dtype": ["float16", "float32", "int32"],
            }, {
                "soc": ["Hi3796CV300ES"],
                "dtype": ["float16", "int32"],
            }]

    cases_str = ""
    for soc_dtypes in soc_dtypes:
        soc_str = "[\"" + "\", \"".join(soc_dtypes.get("soc")) +"\"]"
        dtype_str = "[\"" + "\", \"".join(soc_dtypes.get("dtype")) +"\"]"
        params = tuple([soc_str, dtype_str] * 17)
        cases_str += """ut_case.add_elewise_case_simple(%s, %s, (1,))
ut_case.add_elewise_case_simple(%s, %s, (1, 1))
ut_case.add_elewise_case_simple(%s, %s, (16, 32))
ut_case.add_elewise_case_simple(%s, %s, (16, 2, 32))
ut_case.add_elewise_case_simple(%s, %s, (16, 2, 4, 32))
ut_case.add_elewise_case_simple(%s, %s, (512, 1024))
ut_case.add_elewise_case_simple(%s, %s, (2, 1024))
ut_case.add_elewise_case_simple(%s, %s, (4096, 1024))
ut_case.add_elewise_case_simple(%s, %s, (32, 128, 1024))
ut_case.add_elewise_case_simple(%s, %s, (100, 100))
ut_case.add_elewise_case_simple(%s, %s, (1, 512, 1))
ut_case.add_elewise_case_simple(%s, %s, (1, 16, 512, 512))
ut_case.add_elewise_case_simple(%s, %s, (9973, 1))
ut_case.add_elewise_case_simple(%s, %s, (1024, 1024, 256))
ut_case.add_elewise_case_simple(%s, %s, (11, 33))
ut_case.add_elewise_case_simple(%s, %s, (10, 12))
ut_case.add_elewise_case_simple(%s, %s, (10, 13))

""" % params

    add_cases_to_file(op_type, "ElementwiseOpUT", op_module_name, op_func_name, cases_str, soc_str)
