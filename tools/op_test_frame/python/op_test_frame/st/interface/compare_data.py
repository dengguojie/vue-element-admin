#!/usr/bin/env python
# coding=utf-8
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
compare data
"""
import csv
import os
import numpy as np

from op_test_frame.st.interface import utils
from op_test_frame.st.interface.const_manager import ConstManager


class CompareData:
    """
    class CompareData
    """
    def __init__(self, op_params, err_thr, error_report, run_dir):
        self.op_params = op_params
        self.real_data = None
        self.data_compare = None
        self.err_thr = err_thr
        self.error_report = error_report
        self.run_dir = run_dir

    @staticmethod
    def _cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
        if 'nan' in str(expect_data) or 'inf' in str(expect_data):
            if type_str.lower() == 'fp16':
                expect_data = 65504
            else:
                expect_data = 3.4028e38
        diff = abs(float(real_data) - float(expect_data))
        if abs(float(real_data) - float(expect_data)) < diff_thd:
            rate_diff = diff
        else:
            rate_diff = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
        return rate_diff

    @staticmethod
    def _cal_relative_diff_np(real_data, expect_data, diff_thd):
        uint_data = np.abs(np.subtract(real_data, expect_data))
        max_data = np.maximum(np.abs(real_data), (np.abs(expect_data)))
        diff_data = float((1.0 / (1 << 14)) / diff_thd)
        data_add = np.add(np.maximum(max_data, diff_data), 10e-10)
        result = np.where(uint_data < diff_thd, uint_data, uint_data / data_add)
        return result

    @staticmethod
    def _write_err_report(csv_path, csv_data):
        try:
            with os.fdopen(os.open(csv_path, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES),
                           'a+', newline='') as file_object:
                writer = csv.writer(file_object)
                writer.writerow(csv_data)
        except IOError as io_error:
            utils.print_warn_log("Failed to save the error report data, the reason is %s" % io_error)
        finally:
            pass

    def compare(self, npu_output, cpu_output):
        """
        compare
        """
        self.real_data = npu_output.flatten()
        self.data_compare = cpu_output.flatten()
        if self.real_data.size == 0 and self.real_data.size == self.data_compare.size:
            utils.print_info_log(
                'The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
            return "Pass", 0.0, 0
        return self._get_compare_result()

    def _get_data_size(self):
        start = 0
        end = self.real_data.size - 1
        if end < start:
            end = start
        real_data_size = int(end - start + 1) if end != start else 1
        return start, end, real_data_size

    def _check_overflows_count(self):
        overflows_count = self.data_compare[np.isinf(self.data_compare)].size + self.data_compare[
            np.isnan(self.data_compare)].size
        if overflows_count > 0:
            utils.print_info_log('Overflow,size:%s,benchmark_output:%s, %s' % (
                overflows_count, self.data_compare[np.isinf(self.data_compare)][0:10],
                self.data_compare[np.isnan(self.data_compare)][0:10]))

    def _get_compare_result(self):
        diff_thd, pct_thd, max_diff_hd = self.err_thr[0], self.err_thr[1], 0.1
        max_error = 0
        result = "Failed"
        if self.real_data.size != self.data_compare.size:
            utils.print_error_log(
                'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (
                    self.real_data.size, self.data_compare.size))
            return result, 0.0, max_error
        start, end, real_data_size = self._get_data_size()
        self._check_overflows_count()
        utils.print_info_log('total_count:%s; max_diff_thd:%s;' % (real_data_size, max_diff_hd))
        try:
            diff_abs = np.abs(np.subtract(self.real_data.astype(np.float32), self.data_compare.astype(np.float32)))
        except MemoryError:
            return result, 0.0, max_error
        finally:
            pass
        self._display_output(start, end, diff_thd)
        result, err_list, error_percent = self._get_error_percent(
            [diff_abs, diff_thd, max_diff_hd], real_data_size, pct_thd)
        if result == "Failed":
            self._display_error_output(err_list)
        return result, error_percent, max_error

    def _display_output(self, start, end, diff_thd):
        utils.print_info_log(
            '---------------------------------------------------------------------------------------')
        utils.print_info_log('{:<15} {:<15} {:<15} {:<15} {:<15}'.format('Index', 'ExpectOut', 'RealOut',
                                                                         'FpDiff', 'RateDiff'))
        utils.print_info_log(
            '---------------------------------------------------------------------------------------')
        real_data_size = int(end - start)
        if real_data_size <= 20:
            for index in range(real_data_size + 1):
                self._display_data_by_index(index, start, diff_thd)
        else:
            for index in range(10):
                self._display_data_by_index(index, start, diff_thd)
            dot_3 = '...'
            utils.print_info_log('{dot:<15} {dot:<15} {dot:<15} {dot:<15} {dot:<15}'.format(dot=dot_3))
            for i in range(real_data_size - 10 + 1, real_data_size + 1):
                self._display_data_by_index(i, start, diff_thd)

    def _display_data_by_index(self, index, start, diff_thd):
        index = index + start
        data_index = '%08d' % (index + 1)
        expect_out = '%.7f' % self.data_compare[index]
        real_out = '%.7f' % self.real_data[index]
        fp_diff = '%.7f' % abs(np.float64(self.data_compare[index]) - np.float64(self.real_data[index]))
        rate_diff = '%.7f' % self._cal_relative_diff(self.data_compare[index], self.real_data[index], diff_thd)
        utils.print_info_log('{:<15} {:<15} {:<15} {:<15} {:<15}'.format(data_index, expect_out, real_out,
                                                                         fp_diff, rate_diff))

    def _get_error_percent(self, diff_list, split_count, pct_thd):
        diff_index = np.where(diff_list[0] > 0)
        rdiff = self._cal_relative_diff_np(self.real_data[diff_index].astype(np.float32),
                                           self.data_compare[diff_index].astype(np.float32),
                                           diff_list[1])
        err_diff = rdiff[rdiff > diff_list[1]]
        diff_idx_list = diff_index[0]
        err_idx = diff_idx_list[np.where(rdiff > diff_list[1])]

        fulfill_num = split_count - err_diff.size
        fulfill_percent = float(fulfill_num) / float(split_count)
        pct_thd = 1 - pct_thd
        result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
        if len(err_diff) > 0:
            max_error = max(err_diff)
            if max(err_diff) >= diff_list[2]:
                result = "Failed"
        utils.print_info_log(
            '---------------------------------------------------------------------------------------')
        utils.print_info_log('{:<15} {:<15} {:<15} {:<15}'.format('DiffThd', 'PctThd', 'PctRlt', 'Result'))
        utils.print_info_log(
            '---------------------------------------------------------------------------------------')
        utils.print_info_log('{:<15.4f} {:<15.2%} {:<15.6%} {:<15}'.format(diff_list[1], float(pct_thd),
                                                                           fulfill_percent, result))
        if len(err_diff) > 0:
            utils.print_info_log(
                'Maximum error is: %s. Tolerance threshold is: %s.' % (
                    max_error, diff_list[2]))
        return result, [err_idx, err_diff], fulfill_percent * 100

    def _display_error_output(self, err_list):
        err_idx, relative_diff = err_list
        # Get err report path
        csv_path = self._get_err_report_path()
        # If error_report is true, write header to .csv
        if self.error_report == 'true':
            utils.print_warn_log("For failed test cases, it may take a long time to save the data "
                                 "results whose expected data is inconsistent with the actual data.")
            self._write_err_report(csv_path, ConstManager.ERR_REPORT_HEADER)
        # Print inconsistent data for the first 10 and the last 10.
        utils.print_info_log('Error Line-----------------------------------------------------------------------------')
        utils.print_info_log('{:<15} {:<15} {:<15} {:<15} {:<15}'.format('Index', 'ExpectOut', 'RealOut',
                                                                         'FpDiff', 'RateDiff'))
        utils.print_info_log('---------------------------------------------------------------------------------------')
        # Show Error line and if error_report is true, write error line to .csv
        self._show_and_write_err_report(err_idx, relative_diff, csv_path)
        if self.error_report == 'true':
            utils.print_warn_log(
                "For failed test case of %s, the data results whose expected data is inconsistent with "
                "the actual data saved in: %s" % (self.op_params.get(ConstManager.CASE_NAME), csv_path))
        utils.print_info_log('---------------------------------------------------------------------------------------')

    def _show_and_write_err_report(self, err_idx, relative_diff, csv_path):
        count = 0
        len_err = len(err_idx)
        for i in err_idx:
            count += 1
            data_index = '%08d' % (i + 1)
            expect_out = '%.7f' % self.data_compare[i]
            real_out = '%.7f' % self.real_data[i]
            fp_diff = '%.7f' % abs(np.float64(self.data_compare[i]) - np.float64(self.real_data[i]))
            rate_diff = '%.7f' % float(relative_diff[count - 1])
            if len_err <= 20 or count < 10 or count > len_err - 10:
                utils.print_info_log('{:<15} {:<15} {:<15} {:<15} {:<15}'.format(data_index, expect_out, real_out,
                                                                                 fp_diff, rate_diff))
            elif count == 10:
                dot_3 = '...'
                utils.print_info_log('{dot:<15} {dot:<15} {dot:<15} {dot:<15} {dot:<15}'.format(dot=dot_3))
            if self.error_report == 'true':
                self._write_err_report(csv_path, [data_index, expect_out, real_out, fp_diff, rate_diff])

    def _get_err_report_path(self):
        csv_path = ''
        if self.error_report == 'true':
            case_name = self.op_params.get(ConstManager.CASE_NAME)
            st_error_reports = os.path.join(self.run_dir, 'run', 'out', 'test_data', 'data', 'st_error_reports')
            if not os.path.exists(st_error_reports):
                os.makedirs(st_error_reports)
            csv_file = case_name + '_error_report.csv'
            csv_path = os.path.join(st_error_reports, csv_file)
        return csv_path
