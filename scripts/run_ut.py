# encoding: utf-8
import os
import subprocess
from absl import flags, app

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

cur_dir = os.path.realpath(__file__)


def main(argv):
    del argv
    sch_tag = 0
    ops_tag = 0
    pr_changed_files = FLAGS.pr_changed_file
    if not pr_changed_files or not str(pr_changed_files).strip():
        ops_tag = 1
    else:
        pr_changed_files = os.path.realpath(FLAGS.pr_changed_file)
        with open(pr_changed_files) as pr_f:
            lines = pr_f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith(os.path.join("auto_schedule", "python")) or   \
                    line.startswith(os.path.join("tools", "sch_test_frame")):
                sch_tag = 1
            else:
                ops_tag = 1

    params_dict = {
        "--soc_version=": FLAGS.soc_version,
        "--simulator_lib_path=": FLAGS.simulator_lib_path,
        "--pr_changed_file=": FLAGS.pr_changed_file,
        "--cov_path=": FLAGS.cov_path,
        "--report_path=": FLAGS.report_path
    }
    params = []
    for input_key in params_dict.keys():
        params.append(input_key + str(params_dict[input_key]))

    root_path = os.path.dirname(os.path.dirname(cur_dir))
    python_path = os.getenv('PYTHONPATH')

    if sch_tag:
        print("[INFO]Run schedule ut case!!!")
        frame_path = os.path.join(root_path, 'auto_schedule', 'python',
                                  'tests')
        case_path = os.path.join(root_path, 'auto_schedule', 'python', 'tests',
                                 'ut')
        os.environ['PYTHONPATH'] = ':'.join(
            [case_path, frame_path, python_path])
        run_file_path = os.path.join(root_path, "auto_schedule", "python",
                                     "tests", "sch_run_ut.py")
        cmd = ["python3", run_file_path] + params
        print("[INFO]cmd is ", str(cmd))
        res_msg = os.system(" ".join(cmd))
        if res_msg != 0:
            exit(res_msg)

    if ops_tag:
        print("[INFO]Run ops ut case!!!")
        frame_path = os.path.join(root_path, 'tools', 'op_test_frame',
                                  'python')
        os.environ['PYTHONPATH'] = ':'.join([frame_path, python_path])
        run_file_path = os.path.join(root_path, "ops", "built-in", "tests",
                                     "run_ut.py")
        cmd = ["python3", run_file_path] + params
        print("[INFO]cmd is ", str(cmd))
        res_msg = os.system(" ".join(cmd))
        if res_msg != 0:
            exit(res_msg)

    exit(0)


if __name__ == "__main__":
    app.run(main)
