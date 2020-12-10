
import os
from . import utils
import time
import numpy as np
from op_test_frame.common import op_status
from . import op_st_case_info


def _cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (
                float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result


def _cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def _parse_dtype_by_filename(file_name):
    file_str_list = file_name.split("_")
    file_str = file_str_list[-1]  # eg:int32.bin
    str_type = file_str.split(".")[0]
    return _get_np_dtype(str_type)


def _get_np_dtype(type_str):
    type_dict = {
        'fp64': np.float64, 'fp32': np.float32,  'float32': np.float32,
        'float': np.float32, 'fp16': np.float16, 'float16': np.float16,
        'int64': np.int64, 'int32': np.int32, 'int16': np.int16,
        'int8': np.int8,
        'uint64': np.uint64, 'uint32': np.uint32, 'uint16': np.uint16,
        'uint8': np.uint8,
        'bool': np.bool, 'complex64': np.complex64,
        'complex128': np.complex128,
    }
    return type_dict.get(type_str)


def _display_output(real_data, expect_data, start, end, diff_thd):
    utils.print_info_log(
        '---------------------------------------------------------------------------------------')
    utils.print_info_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    utils.print_info_log(
        '---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            j = i + start
            utils.print_info_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j],
                abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                _cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
    else:
        for i in range(10):
            j = i + start
            utils.print_info_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j],
                abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                _cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
        utils.print_info_log('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            j = i + start
            utils.print_info_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j],
                abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                _cal_relative_diff(expect_data[j], real_data[j], diff_thd)))


def _display_error_output(real_data, expect_data, err_idx, relative_diff, start, end, diff_thd):
    utils.print_info_log('Error '
               'Line-----------------------------------------------------------------------------')
    utils.print_info_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    utils.print_info_log('---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if len_err <= 20 or count < 10 or count > len_err - 10:
            utils.print_info_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i], abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10:
            dot_3 = '...'
            utils.print_info_log('%08s \t %07s \t %07s \t %07s \t %07s  \t %07s ' % (dot_3, dot_3, dot_3, dot_3, dot_3, dot_3))
    utils.print_info_log('---------------------------------------------------------------------------------------')


def _data_compare(npu_output, cpu_output, diff_thd=0.01, pct_thd=0.05,
                 max_diff_hd=0.1):
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        utils.print_info_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
        return "Pass", 0.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        utils.print_error_log(
            'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (
            real_data.size, data_compe.size))
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[
        np.isnan(data_compe)].size
    if overflows_count > 0:
        utils.print_info_log('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[np.isinf(data_compe)][0:10],
            data_compe[np.isnan(data_compe)][0:10]))

    split_count = int(end - start + 1) if end != start else 1
    utils.print_info_log(
        'split_count:%s; max_diff_hd:%s;' % (float(split_count), max_diff_hd))
    try:
        diff_abs = np.abs(np.subtract(real_data.astype(np.float32),
                                      data_compe.astype(np.float32)))
    except MemoryError:
        return result, 0.0, max_error
    diff_index = np.where(diff_abs > 0)
    rdiff = _cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                 data_compe[diff_index].astype(np.float32),
                                 diff_thd)
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    error_cnt = err_diff.size

    fulfill_num = split_count - error_cnt
    fulfill_percent = float(fulfill_num) / float(split_count) * 100.0
    _display_output(real_data, data_compe, start, end, diff_thd)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff)
        if max(err_diff) >= max_diff_hd:
            result = "Failed"
    utils.print_info_log(
        '---------------------------------------------------------------------------------------')
    utils.print_info_log('DiffThd  \t PctThd   \t PctRlt   \t Result')
    utils.print_info_log(
        '---------------------------------------------------------------------------------------')
    utils.print_info_log('%.4f     \t %.2f%%   \t %.6f%%   \t %s' % (
    diff_thd, pct_thd, fulfill_percent, result))
    if len(err_diff) > 0:
        utils.print_info_log('Maximum error is: %s. Tolerance threshold is: %s.' % (
        max_error, max_diff_hd))
    if result == "Failed":
        _display_error_output(real_data, data_compe, err_idx, err_diff, start,
                             end, diff_thd)
    return result, fulfill_percent, max_error


def compare2(result_dir, expect_dir):
    """
    compare output data with expect data by path
    :param result_dir: result data path
    :param expect_dir: expecet data path
    :return:
    """
    start_time = time.time()
    utils.print_info_log(
        'Step:------>>>>>> Start to compare result <<<<<<------ ')
    names = os.listdir(result_dir)
    result_list = list()
    for name in names:
        result_file = os.path.join(result_dir, name)
        expect_name = name.replace("_output_", "_expect_output_")
        expect_file = os.path.join(expect_dir, expect_name)
        if not os.path.isfile(result_file):
            utils.print_warn_log("There is no result file :%s" %
                                 result_file)
            continue
        if not os.path.isfile(expect_file):
            utils.print_warn_log("There is no expect output file"
                                 ":%s" % expect_file)
            continue

        np_type = _parse_dtype_by_filename(result_file)
        if not np_type:
            utils.print_warn_log("Failed to get numpy data type from file "
                                 "name(%s),the np_type = %s")
            continue
        npu_output = np.fromfile(result_file, np_type)
        cpu_output = np.fromfile(expect_file, np_type)
        result, error_percent, max_error = _data_compare(npu_output,
                                                        cpu_output)
        result_list.append([result, error_percent, max_error])

    utils.print_info_log(
        'End to compare result. Duration:%0.2f second.' % (
                    time.time() - start_time))
    return


def compare(report, run_dir):
    """
    compare output data with expect data by report
    :param report: the st report object
    :param run_dir: the run dir ,the parent dir of inc\run, etc ...
    :return:
    """
    start_time = time.time()
    utils.print_info_log(
        'Step:------>>>>>> Start to compare result <<<<<<------ ')
    # 1. check run result , if failed , record failed and skip compare
    result_txt = os.path.join(run_dir, 'run', 'out', 'result_files',
                              'result.txt')
    if not os.path.exists(result_txt) or \
            not os.access(result_txt, os.R_OK):
        utils.print_error_log("Failed to get %s, please check "
                              "run result." % result_txt)
        # add run failed stage result to report
        run_acl_result = op_st_case_info.OpSTStageResult(
            op_status.FAILED, "run_acl_code", None)
        for case_report in report.report_list:
            case_report.trace_detail.add_stage_result(run_acl_result)
        return

    # 2. get case run result,
    txt = utils.read_file(result_txt)
    run_result_list = txt.split('\n')
    for line in run_result_list:
        if len(line.split("  ")) != 3:
            continue
        index, case_name, result = line.split("  ")
        if not index.isdigit():
            utils.print_warn_log("The result line '%s' format error." %
                                 line)
            continue
        utils.print_info_log(
            'Index %s:------>>>>>> Start to compare %s result '
            '<<<<<<------ '
            % (index, case_name))
        case_report = report.get_case_report(case_name)
        if not case_report:
            continue
        if not case_report.trace_detail.st_case_info.op_params.get(
            "calc_expect_func_file_func"):
            utils.print_info_log("There is no 'calc_expect_func_file_func'"
                                 "for compare, skip compare.")
            return
        if result == "[fail]":
            utils.print_info_log("There case '%s' run failed, "
                                 "no result data for compare, compare "
                                 "skipped." % case_name)
            run_acl_result = op_st_case_info.OpSTStageResult(
                op_status.FAILED, "run_acl_code", None)
            case_report.trace_detail.add_stage_result(run_acl_result)
        elif result == "[pass]":
            result_list = list()
            run_acl_result = op_st_case_info.OpSTStageResult(
                op_status.SUCCESS, "run_acl_code", None)
            case_report.trace_detail.add_stage_result(run_acl_result)
            case_info = case_report.trace_detail.st_case_info
            if not case_info:
                utils.print_warn_log("There is no case info for '%s'."
                                     % case_name)
                _add_compare_failed_stage(case_report)
                continue
            if not case_info.expect_data_paths:
                utils.print_warn_log("There is no expect_data_paths for '%s'."
                                     % case_name)
                _add_compare_failed_stage(case_report)
                continue
            for idx, expect_file in enumerate(case_info.expect_data_paths):
                result_file = case_info.planned_output_data_paths[idx]
                utils.print_info_log(
                    "The result file %s comapre vs expect data %s" % (
                        os.path.basename(result_file),
                        os.path.basename(expect_file)))
                if not os.path.isfile(result_file):
                    utils.print_warn_log("There is no result file :%s,"
                                         "skip compare." %
                                         result_file)
                    continue
                if not os.path.isfile(expect_file):
                    utils.print_warn_log("There is no expect output file"
                                         ":%s,skip compare." % expect_file)
                    continue
                ouput_configs = case_info.op_params.get("output_desc")
                if not ouput_configs:
                    utils.print_warn_log("Failed to output data type.")
                    continue
                str_type = ouput_configs[idx].get("type")
                np_type = _get_np_dtype(str_type)
                utils.print_info_log(
                    "The data type is {}, the numpy type is {}".format(
                        str_type, np_type))
                if not np_type:
                    utils.print_warn_log(
                        "Failed to get numpy data type. Skip compare")
                    continue
                npu_output = np.fromfile(result_file, np_type)
                cpu_output = np.fromfile(expect_file, np_type)
                result, error_percent, max_error = _data_compare(npu_output,
                                                                 cpu_output)
                result_list.append([result, error_percent, max_error])

            # add compare report
            compare_status = op_status.SUCCESS
            if not result_list:
                compare_status = op_status.FAILED
            for result_out in result_list:
                if result_out[0] == "Failed":
                    compare_status = op_status.FAILED
            compare_data_result = op_st_case_info.OpSTStageResult(
                compare_status, "compare_data", None)
            case_report.trace_detail.add_stage_result(compare_data_result)

        else:
            utils.print_warn_log("The result in result.txt only support "
                                 "'[pass]' and '[fail]', '%s' is "
                                 "unsupported." % result)
    utils.print_info_log(
        'End to compare result. Duration:%0.2f second.' % (
                    time.time() - start_time))


def _add_compare_failed_stage(case_report):
    compare_data_result = op_st_case_info.OpSTStageResult(
        op_status.FAILED, "compare_data", None)
    case_report.trace_detail.add_stage_result(compare_data_result)
