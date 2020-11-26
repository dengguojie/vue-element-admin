import os
import shutil
from typing import List
from op_test_frame.st.st_case_info import STSingleCaseInfo


def generate_run_py_list(case_root_dir, case_info_list: List[STSingleCaseInfo]):
    run_one_tpl = os.path.join(os.path.dirname(os.path.realpath(__file__)), "case_run.pytpl")
    for case_info in case_info_list:
        case_dir = os.path.join(case_root_dir, case_info.op)
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        run_file_full_path = os.path.join(case_dir, "run.py")
        shutil.copy(run_one_tpl, run_file_full_path)
        with open(os.path.join(case_dir, "__init__.py"), 'w+') as init_f:
            init_f.write("")
    run_all_tpl = os.path.join(os.path.dirname(os.path.realpath(__file__)), "case_run_all.pytpl")
    run_all_full_path = os.path.join(case_root_dir, "run.py")
    shutil.copy(run_all_tpl, run_all_full_path)
    with open(os.path.join(case_root_dir, "__init__.py"), 'w+') as init_f:
        init_f.write("")
