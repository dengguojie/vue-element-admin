# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
ut load for load ut cases
"""
import os
import sys
import fnmatch
from typing import List
from op_test_frame.common import logger
from op_test_frame.ut import op_ut_case_info


def load_ut_cases(case_dir, file_name_pattern="test_*_impl.py") -> List[op_ut_case_info.UTCaseFileInfo]:
    test_case_file_list = []

    def _find_all_test_case_file(case_path_list):
        case_file_list_inner = []
        for case_path in case_path_list:
            case_path_inner = os.path.realpath(case_path)
            if os.path.isfile(case_path_inner) and case_path not in case_file_list_inner:
                case_file_list_inner.append(case_path)
                continue
            for path, dir_names, file_names in os.walk(case_path_inner):
                for file_name in file_names:
                    if fnmatch.fnmatch(file_name, file_name_pattern):
                        test_case_file = os.path.join(path, file_name)
                        if test_case_file not in case_file_list_inner:
                            case_file_list_inner.append(test_case_file)
        return case_file_list_inner

    if not isinstance(case_dir, (tuple, list)):
        case_dir = [case_dir, ]

    case_file_list = _find_all_test_case_file(case_dir)
    if len(case_file_list) > 4:
        import multiprocessing
        from multiprocessing import Pool
        cpu_cnt = multiprocessing.cpu_count() - 1
        with Pool(processes=cpu_cnt) as pool:
            all_res = pool.map(_get_op_module_info, case_file_list)
        for res in all_res:
            if not res or not res[1]:
                continue
            test_case_file_list.append(op_ut_case_info.UTCaseFileInfo(*res))
    else:
        for case_file in case_file_list:
            res = _get_op_module_info(case_file)
            if not res or not res[1]:
                continue
            test_case_file_list.append(op_ut_case_info.UTCaseFileInfo(*res))
    logger.log_debug("load_ut_cases end, load case files total count: %d" % len(test_case_file_list))
    return test_case_file_list


def _get_op_module_info(case_file):
    logger.log_debug("_get_op_module_info start, case file: %s" % case_file)
    case_dir = os.path.dirname(case_file)
    case_module_name = os.path.basename(case_file)[:-3]
    sys.path.insert(0, case_dir)
    try:
        __import__(case_module_name)
    except BaseException as e:
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        trace_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
        err_trace_str = ""
        for t_i in trace_info:
            err_trace_str += t_i
        logger.log_warn("import case file failed,case File \"%s\", error msg: %s, \n trace detail:\n %s"
                        "" % (case_file, e.args[0], err_trace_str))

        sys.path.remove(case_dir)
        logger.log_debug("_get_op_module_info end, case file: %s " % case_file)
        return None

    case_module = sys.modules[case_module_name]
    ut_case = getattr(case_module, "ut_case", None)
    if not ut_case:
        logger.log_warn("Not found ut_case in case_file: %s" % case_file)
        sys.path.remove(case_dir)
        del sys.modules[case_module_name]
        logger.log_debug("_get_op_module_info end, case file: %s " % case_file)
        return None

    op_module_name = None
    if "op_module_name" in ut_case.__dict__.keys():
        op_module_name = ut_case.op_module_name
    else:
        logger.log_warn("'ut_case' object type is not base on 'OpUT' in case_file: %s" % case_file)

    sys.path.remove(case_dir)
    del sys.modules[case_module_name]
    logger.log_debug("_get_op_module_info end, case file: %s " % case_file)
    return case_file, op_module_name


if __name__ == '__main__':
    file_name_pattern = "test_*_impl.py"


    def find_all_test_case_file(case_path_list):
        case_file_list_inner = []
        for case_path in case_path_list:
            case_path_inner = os.path.realpath(case_path)
            if os.path.isfile(case_path_inner) and case_path not in case_file_list_inner:
                case_file_list_inner.append(case_path)
                continue
            for path, dir_names, file_names in os.walk(case_path_inner):
                for file_name in file_names:
                    if fnmatch.fnmatch(file_name, file_name_pattern):
                        test_case_file = os.path.join(path, file_name)
                        if test_case_file not in case_file_list_inner:
                            case_file_list_inner.append(test_case_file)
        return case_file_list_inner


    import time

    start = time.time()
    find_all_test_case_file(["/home/allan/tewsp/repo_simple/llt/ops/llt_new/ut/ops_test", ])
    end = time.time()
    print(end - start)
