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
UT test entry for Ops
"""
import os
from op_test_frame.ut import op_ut_runner
from op_test_frame.common import op_status
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string("soc_version", None, "SOC VERSION")
flags.DEFINE_string("op", None, "Test case directory name of test op")
flags.DEFINE_string("case_name", None, "Case name, run which case")
flags.DEFINE_string("case_dir", None, "Case directory, test case in which directory")
flags.DEFINE_string("report_path", None, "which directory to save ut test report")
flags.DEFINE_string("cov_path", None, "which dirctory to save coverage report")
flags.DEFINE_string("simulator_lib_path", None, "the path to simulator libs")


def main(argv):
    _ = argv
    soc_version = FLAGS.soc_version
    soc_version = [soc.strip() for soc in str(soc_version).split(",")]
    case_dir = FLAGS.case_dir
    if not case_dir:
        case_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ut", "ops_test")
    cov_report_path = FLAGS.cov_path if FLAGS.cov_path else "./cov_report"
    report_path = FLAGS.report_path if FLAGS.report_path else "./report"
    simulator_lib_path= FLAGS.simulator_lib_path if FLAGS.simulator_lib_path else "/usr/local/Ascend/toolkit/tools/simulator"
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

