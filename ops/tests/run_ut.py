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

