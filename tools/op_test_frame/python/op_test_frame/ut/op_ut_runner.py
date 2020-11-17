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

"""op ut runner, apply run ut function"""
import unittest
import time
import os
import sys
import stat
import shutil
import multiprocessing
from datetime import datetime
from multiprocessing import Pool
from typing import List
from functools import reduce

import coverage
from op_test_frame.common import logger
from op_test_frame.ut import ut_loader
from op_test_frame.ut import ut_report
from op_test_frame.utils import file_util
from op_test_frame.ut.op_ut_case_info import OpUTSuite
from op_test_frame.ut.op_ut_case_info import CaseUsage
from op_test_frame.model_run_utils import model_run_utils

DATA_DIR_MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP


class OpUTTestRunner:  # pylint: disable=too-few-public-methods
    """
    Op ut runner
    """

    def __init__(self, print_summary=True, verbosity=2, simulator_mode=None, simulator_lib_path=None):
        self.print_summary = print_summary
        self.verbosity = verbosity

        if simulator_mode:
            if not simulator_lib_path:
                simulator_lib_path = os.environ.get("SIMULATOR_PATH")
            if not simulator_lib_path:
                raise RuntimeError(
                    "Not configured simulator path, when run simulator. "
                    "Please set simulator_lib_path arg, or set ENV SIMULATOR_PATH")
            if simulator_mode not in model_run_utils.SUPPORT_MODEL_LIST:
                raise RuntimeError("Not support this simulator_mode: %s, not suppot [%s]" % (
                    simulator_mode, ",".join(model_run_utils.SUPPORT_MODEL_LIST)))

        self.simulator_mode = simulator_mode
        self.simulator_lib_path = simulator_lib_path

    def _excute_one_soc(self, op_ut: OpUTSuite):
        # to avoid import te before tensorflow, can't import te outside function
        from te.platform import te_set_version  # pylint: disable=import-outside-toplevel
        te_set_version(op_ut.soc)
        suite = op_ut.soc_suite
        op_ut.clear_test_trace()
        if self.simulator_mode:
            model_run_utils.set_run_mode(self.simulator_mode, op_ut.soc, self.simulator_lib_path)
        unittest.TextTestRunner(verbosity=self.verbosity).run(suite)
        return op_ut.get_test_trace()

    def run(self, op_uts: List[OpUTSuite]):
        """
        run ut suites
        :param op_uts: op ut suite list
        :return: run report
        """
        report = ut_report.OpUTReport()
        start_time = time.time()
        print(">>>> start run test case")
        for op_ut in op_uts:
            trace_list = self._excute_one_soc(op_ut)
            for case_trace in trace_list:
                case_rpt = ut_report.OpUTCaseReport(case_trace)
                report.add_case_report(case_rpt)
        end_time = time.time()
        time_taken = end_time - start_time
        print(">>>> end run test case, op_type:%s cost time: %d " % (
            "None" if not op_uts else op_uts[0].op_type, time_taken))
        if self.print_summary:
            report.console_print()

        return report


class RunUTCaseFileArgs:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    run ut case file args for multiprocess run
    """

    def __init__(self, case_file, op_module_name, soc_version,  # pylint: disable=too-many-arguments
                 case_name, test_report, test_report_data_path,
                 cov_report, cov_data_path, simulator_mode, simulator_lib_path,
                 data_dir, dump_model_dir):
        self.case_file = case_file
        self.op_module_name = op_module_name
        self.soc_version = soc_version
        self.case_name = case_name
        self.test_report = test_report
        self.test_report_data_path = test_report_data_path
        self.cov_report = cov_report
        self.cov_data_path = cov_data_path
        self.simulator_mode = simulator_mode
        self.simulator_lib_path = simulator_lib_path
        self.data_dir = data_dir
        self.dump_model_dir = dump_model_dir


def _run_ut_case_file(run_arg: RunUTCaseFileArgs):
    logger.log_info("start run: %s" % run_arg.case_file)
    res = True
    if run_arg.cov_report:
        ut_cover = coverage.Coverage(source=[run_arg.op_module_name, "impl"], data_file=run_arg.cov_data_path)
        ut_cover.start()

    try:
        case_dir = os.path.dirname(os.path.realpath(run_arg.case_file))
        case_module_name = os.path.basename(os.path.realpath(run_arg.case_file))[:-3]
        sys.path.insert(0, case_dir)
        __import__(case_module_name)
        case_module = sys.modules[case_module_name]
        ut_case = getattr(case_module, "ut_case", None)
        case_usage_list = [CaseUsage.IMPL, CaseUsage.CUSTOM, CaseUsage.CFG_COVERAGE_CHECK,
                           CaseUsage.CHECK_SUPPORT, CaseUsage.SELECT_FORMAT, CaseUsage.PRECISION]

        if not run_arg.simulator_mode:
            case_usage_list.remove(CaseUsage.PRECISION)

        soc_case_list = ut_case.get_test_case(run_arg.soc_version, case_name=run_arg.case_name,
                                              case_usage_list=case_usage_list)
        for soc_case in soc_case_list:
            soc_case.clear_test_trace()
            soc_case.set_test_data_dir(run_arg.data_dir)
            soc_case.set_dump_model_dir(run_arg.dump_model_dir)
            if run_arg.simulator_mode:
                soc_case.set_simulator_mode(run_arg.simulator_mode)
        case_runner = OpUTTestRunner(print_summary=False, simulator_mode=run_arg.simulator_mode,
                                     simulator_lib_path=run_arg.simulator_lib_path)
        ut_rpt = case_runner.run(soc_case_list)
        ut_rpt.save(run_arg.test_report_data_path)
        del sys.modules[case_module_name]
    except BaseException as run_err:  # pylint: disable=broad-except
        logger.log_err("Test Failed! case_file: %s, error_msg: %s" % (run_arg.case_file, run_err.args[0]))
        res = False

    if run_arg.cov_report:
        ut_cover.stop()
        ut_cover.save()
    logger.log_info("end run: %s" % run_arg.case_file)
    return res


SUCCESS = "success"
FAILED = "failed"


def _check_args(case_dir, test_report, cov_report):
    if not case_dir:
        logger.log_err("Not set case dir")
        return False
    if test_report and test_report not in ("json", "console"):
        logger.log_err("'test_report' only support 'json/console'.")
        return False
    if cov_report and cov_report not in ("html", "json", "xml"):
        logger.log_err("'cov_report' only support 'html/json/xml'.")
        return False
    return True


def _build_cov_data_path(cov_report_path):
    cov_combine_path = os.path.join(os.path.realpath(cov_report_path), "combine_data_path")
    if os.path.exists(cov_combine_path):
        shutil.rmtree(cov_combine_path)
    file_util.makedirs(cov_combine_path, mode=DATA_DIR_MODES)
    return cov_combine_path


def _build_report_data_path(test_report_path):
    rpt_combine_path = os.path.join(os.path.realpath(test_report_path), "combine_rpt_path")
    if os.path.exists(rpt_combine_path):
        shutil.rmtree(rpt_combine_path)
    file_util.makedirs(rpt_combine_path, mode=DATA_DIR_MODES)
    return rpt_combine_path


def run_ut(case_dir, soc_version, case_name=None,  # pylint: disable=too-many-arguments, too-many-locals
           test_report="json", test_report_path="./report",
           cov_report=None, cov_report_path="./cov_report",
           simulator_mode=None, simulator_lib_path=None,
           simulator_data_path="./model", test_data_path="./data"):
    """
    run ut test case
    :param case_dir: a test case dir or a test case file
    :param soc_version: like "Ascend910", "Ascend310"
    :param case_name: run case name, default is None, run all test case
    :param test_report: support console/json, report format type
    :param test_report_path: test report save path
    :param cov_report: support html/json/xml type, if None means not need coverage report
    :param cov_report_path: coverage report save path
    :param simulator_mode: simulator_mode can be None/pv/ca/tm
    :param simulator_lib_path: simulator library path
    :param simulator_data_path: test data directory, input, output and expect output data
    :param test_data_path: when run ca or tm mode, dump data save in this dirctory

    :return: success or failed
    """
    print("start run ops ut time: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    if not _check_args(case_dir, test_report, cov_report):
        return FAILED

    case_file_info_list, load_has_err = ut_loader.load_ut_cases(case_dir)
    if not case_file_info_list:
        logger.log_err("Not found any test cases.")
        return FAILED

    cov_combine_dir = _build_cov_data_path(cov_report_path)
    rpt_combine_dir = _build_report_data_path(test_report_path)

    def _build_multiprocess_run_args():
        total_run_arg_list = []
        ps_count = 1
        if not isinstance(soc_version, (tuple, list)):
            soc_version_list = str(soc_version).split(",")
        else:
            soc_version_list = soc_version
        for case_file_info in case_file_info_list:
            case_file_tmp = os.path.basename(case_file_info.case_file)[:-3]
            for one_soc_version in soc_version_list:
                single_cov_data_path = os.path.join(cov_combine_dir, ".coverage_" + str(ps_count) + "_" + case_file_tmp)
                single_rpt_data_path = os.path.join(rpt_combine_dir,
                                                    "rpt_" + str(ps_count) + "_" + case_file_tmp + ".data")
                run_arg = RunUTCaseFileArgs(case_file=case_file_info.case_file,
                                            op_module_name=case_file_info.op_module_name,
                                            soc_version=one_soc_version,
                                            case_name=case_name,
                                            test_report=test_report,
                                            test_report_data_path=single_rpt_data_path,
                                            cov_report=cov_report,
                                            cov_data_path=single_cov_data_path,
                                            simulator_mode=simulator_mode,
                                            simulator_lib_path=simulator_lib_path,
                                            data_dir=test_data_path,
                                            dump_model_dir=simulator_data_path)
                total_run_arg_list.append(run_arg)
                ps_count += 1
        return total_run_arg_list

    multiprocess_run_args = _build_multiprocess_run_args()

    cpu_count = multiprocessing.cpu_count() - 1
    logger.log_info("multiprocessing cpu count: %d" % cpu_count)
    logger.log_info("multiprocess_run_args count: %d" % len(multiprocess_run_args))

    if len(multiprocess_run_args) == 1:
        run_success = _run_ut_case_file(multiprocess_run_args[0])
    else:
        with Pool(processes=cpu_count) as pool:
            results = pool.map(_run_ut_case_file, multiprocess_run_args)
        run_success = reduce(lambda x, y: x and y, results)

    test_report = ut_report.OpUTReport()
    test_report.combine_report(rpt_combine_dir)
    report_data_path = os.path.join(test_report_path, ".ut_test_report")
    test_report.save(report_data_path)
    if test_report:
        test_report.console_print()

    if cov_report and len(multiprocess_run_args) > 0:
        total_cov_data_file = os.path.join(cov_report_path, ".coverage")
        cov = coverage.Coverage(source="impl", data_file=total_cov_data_file)
        combine_files = [os.path.join(cov_combine_dir, cov_file) for cov_file in os.listdir(cov_combine_dir)]
        cov.combine(combine_files)
        cov.save()
        cov.load()
        cov.html_report(directory=cov_report_path)
        os.removedirs(cov_combine_dir)
    print("end run ops ut time: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    if load_has_err:
        logger.log_err("Has error in case files, you can see error log by key word 'import case file failed'.")
    run_result = SUCCESS if run_success and not load_has_err else FAILED
    if test_report.err_cnt > 0 or test_report.failed_cnt > 0:
        run_result = FAILED
    return run_result
