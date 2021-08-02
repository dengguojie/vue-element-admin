# Copyright 2021 Huawei Technologies Co., Ltd
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
for schedule ut test
"""
import os
import json
import sys

from sch_test_frame.ut import op_ut_runner
from sch_test_frame.common import op_status


cur_dir = os.path.realpath(__file__)


def run_st(case_dir, soc_version, out_dir):
    soc_version = soc_version
    if os.path.isdir(case_dir):
        case_list_file = os.listdir(case_dir)
        if not case_list_file or case_list_file == ['get_change.log']:
            # has no relate ut, not need run ut.
            exit(0)
    
    cov_report_path = "./cov_result"
    report_path = "./report/sch/python_report"

    res = op_ut_runner.run_ut(case_dir,
                              soc_version=soc_version,
                              test_report="json",
                              test_report_path=report_path,
                              cov_report="html",
                              cov_report_path=cov_report_path)
    result = {}  
    for combine_rpt in os.listdir(os.path.join(report_path, "combine_rpt_path")):
        case_name = "_".join(combine_rpt.split(".")[0].split("_")[2:])
        case_res = "Pass"
        with open(os.path.join(report_path, "combine_rpt_path", combine_rpt), "r") as rpt_file:
            rpt_data = json.load(rpt_file)
            for case_rpt in rpt_data["report_list"]:
                #case_name = case_rpt.get("case_name")
                if case_rpt.get("status") != "success":
                    case_res = "Failed"
            result[case_name] = case_res

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, "result.txt"), "w") as res_file:
        for k, v in result.items():
            res_str="{} {}".format(k,v)
            res_file.write(res_str)
            res_file.write("\n")

    if res == op_status.SUCCESS:
        exit(0)
    else:
        exit(-1)


if __name__ == "__main__":
    pr_changed_file = ""
    if len(sys.argv) >= 4:
        case_dir = sys.argv[1]
        soc_version = sys.argv[2]
        out_dir = sys.argv[3]
    elif len(sys.argv) == 3:
        case_dir = sys.argv[1]
        soc_version = sys.argv[2]
        out_dir = './'
    else:
        print("args error!!")
        exit(-1)
    run_st(case_dir, soc_version, out_dir)
