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
op ut helper, apply helper function, get case name
"""
import os
import sys
import json
from op_test_frame.ut import ut_loader
from op_test_frame.common import logger


def get_case_name_from_file(params):
    case_file_path = params.get("case_file_path")
    soc_version = params.get("soc_version")
    try:
        case_dir = os.path.dirname(os.path.realpath(case_file_path))
        case_module_name = os.path.basename(os.path.realpath(case_file_path))[:-3]
        sys.path.insert(0, case_dir)
        __import__(case_module_name)
        case_module = sys.modules[case_module_name]
        ut_case = getattr(case_module, "ut_case", None)
        case_name_list_inner = ut_case.get_all_test_case_name(soc_version)
    except BaseException as e:
        logger.log_err("Get case name failed, error msg: %s" % e.args[0])
        return None
    return {case_file_path: case_name_list_inner}


def get_case_name(case_file, dump_file=None, soc_version=None):
    def _save_to_file(info_str, dump_file_inner):
        dump_file_dir = os.path.dirname(dump_file_inner)
        if not os.path.exists(dump_file_dir):
            os.makedirs(dump_file_dir)
        with open(dump_file_inner, 'w+') as d_f:
            d_f.write(info_str)

    def _print_case_name_info(json_obj):
        for key, name_list in json_obj.items():
            print("Case File: %s, Case names as follows" % key)
            for idx, name in enumerate(name_list):
                print("        %d) %s" % (idx, name))

    total_case_info = {}
    ut_case_file_info_list = ut_loader.load_ut_cases(case_file)

    if len(ut_case_file_info_list) > 4:
        multi_process_args = [
            {
                "case_file_path": ut_file_info.case_file,
                "soc_version": soc_version
            } for ut_file_info in ut_case_file_info_list
        ]
        import multiprocessing
        from multiprocessing import Pool
        cpu_cnt = multiprocessing.cpu_count() - 1
        with Pool(processes=cpu_cnt) as pool:
            all_res = pool.map(get_case_name_from_file, multi_process_args)
        for res in all_res:
            if not res:
                continue
            for key, val in res.items():
                total_case_info[key] = val
    else:
        for ut_file_info in ut_case_file_info_list:
            params = {
                "case_file_path": ut_file_info.case_file,
                "soc_version": soc_version
            }
            res = get_case_name_from_file(params)
            if not res:
                continue
            for key, val in res.items():
                total_case_info[key] = val

    if not dump_file:
        _print_case_name_info(total_case_info)
    else:
        dump_file = os.path.realpath(dump_file)
        json_str = json.dumps(total_case_info, indent=4)
        _save_to_file(json_str, dump_file)
