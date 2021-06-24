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
from absl import flags, app
from typing import List, Dict

from sch_test_frame.ut import op_ut_runner
from sch_test_frame.common import op_status

FLAGS = flags.FLAGS

flags.DEFINE_string("soc_version", None, "SOC VERSION")
flags.DEFINE_string("op", None, "Test case directory name of test op")
flags.DEFINE_string("case_name", None, "Case name, run which case")
flags.DEFINE_string("case_dir", None,
                    "Case directory, test case in which directory")
flags.DEFINE_string("report_path", None,
                    "which directory to save ut test report")
flags.DEFINE_string("cov_path", None, "which dirctory to save coverage report")
flags.DEFINE_string("simulator_lib_path", None, "the path to simulator libs")
flags.DEFINE_string(
    "pr_changed_file", None,
    "git diff result file by ci, analyse relate ut by this file")
flags.DEFINE_integer("process_num", None, "process number")

cur_dir = os.path.realpath(__file__)
repo_root = os.path.sep.join(cur_dir.split(os.path.sep)[:-4])
ini_cfg_root = os.path.join(repo_root, "ops", "built-in", "tbe", "op_info_cfg",
                            "ai_core")
impl_file_root = os.path.join(repo_root, "ops", "built-in", "tbe", "impl")


class FileChangeInfo:
    def __init__(self, schedule_files: list, sch_test_frame_files: list,
                 test_files: list, other_files: list):
        self.schedule_files = schedule_files
        self.sch_test_frame_files = sch_test_frame_files
        self.test_files = test_files
        self.other_files = other_files

    def print_change_info(self):
        print(
            "========================================================================="
        )
        print("changed file info")
        print(
            "-------------------------------------------------------------------------"
        )
        print("sch changed files: \n%s" % "\n".join(self.schedule_files))
        print(
            "-------------------------------------------------------------------------"
        )
        print("sch test frame changed files: \n%s" %
              "\n".join(self.sch_test_frame_files))
        print(
            "-------------------------------------------------------------------------"
        )
        print("sch test changed files: \n%s" % "\n".join(self.test_files))
        print(
            "-------------------------------------------------------------------------"
        )
        print("other changed files: \n%s" % "\n".join(self.other_files))
        print(
            "========================================================================="
        )


def get_file_change_info_from_ci(changed_file_info_from_ci):
    """
    get file change info from ci, ci will write `git diff > /or_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or FileChangeInf
    """
    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        print(
            "[ERROR] %s file is not exist, can not get file change info in this pull request."
        )
        return None
    with open(or_file_path) as or_f:
        lines = or_f.readlines()
    sch_changed_files = []
    test_change_files = []
    sch_test_frame_changed_files = []
    other_changed_files = []
    print("----------ci changed file content----------")
    print("".join(lines))
    print("-------------------------------------------")
    for line in lines:
        line = line.strip()
        if line.startswith(os.path.join("auto_schedule", "python", "tests")):
            test_change_files.append(line)
        elif line.startswith(os.path.join("auto_schedule", "python")):
            sch_changed_files.append(line)
        elif line.startswith(os.path.join("tools")):
            sch_test_frame_changed_files.append(line)
        else:
            other_changed_files.append(line)
    return FileChangeInfo(schedule_files=sch_changed_files,
                          sch_test_frame_files=sch_test_frame_changed_files,
                          test_files=test_change_files,
                          other_files=other_changed_files)


def get_change_relate_ut_dir_list(changed_file_info_from_ci):
    file_change_info = get_file_change_info_from_ci(changed_file_info_from_ci)
    if not file_change_info:
        print("[ERROR] not found file change info, run ut failed.")
        return None
    file_change_info.print_change_info()

    def _get_relate_ut_list_by_file_change():
        relate_ut_dir_list = []

        def _deal_auto_schedule_file_change():
            schedule_changed_files = file_change_info.schedule_files
            if not schedule_changed_files:
                return
            map_file = os.path.join(os.path.dirname(cur_dir),
                                    "sch_case_map.json")
            with open(map_file, "r") as json_file:
                schedule_case_map = json.load(json_file)
            for schedule_changed_file in schedule_changed_files:
                schedule_ut_case = schedule_case_map.get(
                    schedule_changed_file, [])
                if schedule_ut_case:
                    for sch_case in schedule_ut_case:
                        sch_ut_test_dir = os.path.join(repo_root,
                                                       "auto_schedule",
                                                       "python", "tests", "ut",
                                                       sch_case)
                        if sch_ut_test_dir not in relate_ut_dir_list:
                            relate_ut_dir_list.append(sch_ut_test_dir)

        _deal_auto_schedule_file_change()

        def _deal_test_file_change():
            test_changed_files = file_change_info.test_files
            if not test_changed_files:
                return
            for test_changed_file in test_changed_files:
                test_changed_file = str(test_changed_file).strip()
                test_case_dir = os.path.join("auto_schedule", "python",
                                             "tests", "ut")
                in_ut_dir = test_changed_file.startswith(test_case_dir)
                test_changed_file_split = test_changed_file.split(os.path.sep)
                test_changed_file_name = test_changed_file_split[-1]
                file_name_match = test_changed_file_name.startswith("test_")
                file_name_match = file_name_match and test_changed_file_name.endswith(
                    "_impl.py")

                if in_ut_dir and file_name_match:
                    if not len(test_changed_file_split) == 6:
                        raise RuntimeError(
                            "Can only add test case file like: auto_schedule/python/tests/ut/vadd/test_*_impl.py."
                        )
                    op_ut_test_dir = os.path.join(
                        repo_root, test_case_dir,
                        test_changed_file.split(os.path.sep)[-2])
                    if op_ut_test_dir not in relate_ut_dir_list:
                        relate_ut_dir_list.append(
                            os.path.join(op_ut_test_dir,
                                         test_changed_file_name))
                        # not need test all test in dir, relate_ut_dir_list.append(op_ut_test_dir)

        _deal_test_file_change()

        def _deal_sch_test_frame_change():
            if not relate_ut_dir_list:
                relate_ut_dir_list.append(
                    os.path.join(repo_root, "auto_schedule", "python", "tests",
                                 "ut", "vadd"))

        _deal_sch_test_frame_change()

        return relate_ut_dir_list

    try:
        relate_ut_directory_list = _get_relate_ut_list_by_file_change()
    except BaseException as e:
        print(e.args)
        return None
    if relate_ut_directory_list:
        print("[INFO] relate ut directory list is: [%s]" %
              ", ".join(relate_ut_directory_list))
    else:
        print("[INFO] relate ut directory list is empty")
    return relate_ut_directory_list


def main(argv):
    _ = argv
    soc_version = FLAGS.soc_version
    soc_version = [soc.strip() for soc in str(soc_version).split(",")]
    pr_changed_file = FLAGS.pr_changed_file
    if not pr_changed_file or not str(pr_changed_file).strip():
        case_dir = FLAGS.case_dir
        if not case_dir:
            case_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "ut")
    else:
        case_dir = get_change_relate_ut_dir_list(pr_changed_file)
        if case_dir is None:
            # get relate ut failed.
            exit(-1)
        if not case_dir:
            # has no relate ut, not need run ut.
            exit(0)
    cov_report_path = FLAGS.cov_path if FLAGS.cov_path else "./cov_report/ops/python_utest"
    report_path = FLAGS.report_path if FLAGS.report_path else "./report/ops/python_report"
    simulator_lib_path = FLAGS.simulator_lib_path if FLAGS.simulator_lib_path else  \
                            "/usr/local/Ascend/toolkit/tools/simulator"
    res = op_ut_runner.run_ut(case_dir,
                              soc_version=soc_version,
                              test_report="json",
                              test_report_path=report_path,
                              cov_report="html",
                              cov_report_path=cov_report_path,
                              simulator_mode="pv",
                              simulator_lib_path=simulator_lib_path)
    if res == op_status.SUCCESS:
        exit(0)
    else:
        exit(-1)


if __name__ == "__main__":
    app.run(main)
