#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
import re
import fnmatch
import multiprocessing
from typing import List
from multiprocessing import Pool
from op_test_frame.common import logger
from op_test_frame.ut import op_ut_case_info
from op_test_frame.ut import OpUT


def _find_all_test_case_file(case_path_list, file_name_pattern):
    case_file_list_inner = []
    for case_path in case_path_list:
        case_path_inner = os.path.realpath(case_path)
        if os.path.isfile(case_path_inner) and case_path not in case_file_list_inner:
            case_file_list_inner.append(case_path)
            continue
        for path, _, file_names in os.walk(case_path_inner):
            for file_name in file_names:
                if fnmatch.fnmatch(file_name, file_name_pattern):
                    test_case_file = os.path.join(path, file_name)
                    if test_case_file not in case_file_list_inner:
                        case_file_list_inner.append(test_case_file)
    return case_file_list_inner


def load_ut_cases(case_dir, file_name_pattern="test_*_impl.py") -> List[op_ut_case_info.UTCaseFileInfo]:
    """
    load ut cases
    :param case_dir: case diretory
    :param file_name_pattern: file name pattern, default: "test_*_impl.py"
    :return: List of UTCaseFileInfo
    """
    test_case_file_list = []
    has_error = False

    if not isinstance(case_dir, (tuple, list)):
        case_dir = [case_dir, ]

    case_file_list = _find_all_test_case_file(case_dir, file_name_pattern)
    logger.log_debug("load_ut_cases, case_file_list: %s" % ",".join(case_file_list))
    if len(case_file_list) > 4:
        cpu_cnt = max(multiprocessing.cpu_count() - 1, 1)
        with Pool(processes=cpu_cnt) as pool:
            all_res = pool.map(_get_op_module_info, case_file_list)
        for res in all_res:
            status, c_f, c_m = res
            if not status:
                has_error = True
            if not c_f or not c_m:
                continue
            test_case_file_list.append(op_ut_case_info.UTCaseFileInfo(c_f, c_m))
    else:
        for case_file in case_file_list:
            status, c_f, c_m = _get_op_module_info(case_file)
            if not status:
                has_error = True
            if not c_f or not c_m:
                continue
            test_case_file_list.append(op_ut_case_info.UTCaseFileInfo(c_f, c_m))
    test_case_file_list.sort(key=lambda x:x.case_file)
    logger.log_debug("load_ut_cases end, load case files total count: %d" % len(test_case_file_list))
    return test_case_file_list, has_error


def _get_op_module_info(case_file):
    logger.log_debug("_get_op_module_info start, case file: %s" % case_file)
    with open(case_file, 'r') as f:
        content = f.read()
    match = re.search('\s*ut_case\s*=\s*[^(]+\(\s*([^,)]+)[^)]*\)', content)
    if not match:
        logger.log_warn("Not found ut_case in case_file: %s" % case_file)
        return True, None, None
    op_type = match.groups()[0].strip()
    op_type = op_type[1:-1]
    op_module_name = None
    match = re.search('\s*ut_case\s*=\s*[^(]+\(\s*([^,)]+)\s*,\s*([^,)]+)\s*,?[^,)]*\)', content)
    if match:
        module_name = match.groups()[1].strip()
        if module_name != 'None':
            op_module_name = module_name[1:-1]
    logger.log_debug(f"op_type is %s, op_module_name is %s, case file: %s" % (op_type, op_module_name, case_file))
    ut_case = OpUT(op_type, op_module_name, None)
    logger.log_debug("_get_op_module_info end, case file: %s " % case_file)
    return True, case_file, ut_case.op_module_name
