#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Comparison utilities for dynamic shape
"""
# Standard Packages
import logging
import os
import time
from typing import Tuple, Union, Any

# Third-party Packages
import numpy
from ..tbe_logging import add_level
from ...utilities import get
from ...utilities import check_equal_length
from ...utilities import bfloat16_conversion


add_level("debugc", "DEBUGC", logging.DEBUG - 5)


def comparing(dyn_kernel_name: str, stc_kernel_name: str, cst_kernel_name: str, bin_kernel_name: str,
              dyn_outputs: Tuple[Union[str, Any], ...],
              stc_outputs: Tuple[Union[str, Any], ...],
              cst_outputs: Tuple[Union[str, Any], ...],
              bin_outputs: Tuple[Union[str, Any], ...],
              goldens: Tuple[Union[str, numpy.ndarray], ...],
              percentage_thresholds: Tuple[Tuple[float, float]],
              output_dtypes: tuple,
              strict_precision_mode=True,
              absolute_precision=0,
              fork_count=1) -> tuple:
    """
    Compare dynamic static and golden outputs
    :param bin_kernel_name:
    :param bin_outputs:
    :param dyn_kernel_name:
    :param stc_kernel_name:
    :param cst_kernel_name:
    :param dyn_outputs:
    :param stc_outputs:
    :param cst_outputs:
    :param goldens:
    :param percentage_thresholds:
    :param output_dtypes:
    :param strict_precision_mode:
    :param absolute_precision:
    :param fork_count: parallel fork count:
    :return:
    """
    # Transform bytes to numpy.array if possible
    dyn_outputs = list(dyn_outputs)
    stc_outputs = list(stc_outputs)
    cst_outputs = list(cst_outputs)
    bin_outputs = list(bin_outputs)
    output_dtypes = tuple(output_dtype for output_dtype in output_dtypes if output_dtype is not None)
    __outputs_to_numpy_arrays(dyn_outputs, output_dtypes)
    __outputs_to_numpy_arrays(stc_outputs, output_dtypes)
    __outputs_to_numpy_arrays(cst_outputs, output_dtypes)
    __outputs_to_numpy_arrays(bin_outputs, output_dtypes)
    goldens = list(goldens)
    for _idx, golden in enumerate(goldens):
        if hasattr(golden, "dtype") and "bfloat16" in str(golden.dtype):
            goldens[_idx] = golden.astype("float32")
    goldens = tuple(goldens)
    dyn_outputs = tuple(dyn_outputs)
    stc_outputs = tuple(stc_outputs)
    cst_outputs = tuple(cst_outputs)
    bin_outputs = tuple(bin_outputs)
    relative_percentage_thresholds = ((0.001, 0),) if strict_precision_mode else percentage_thresholds
    # noinspection PyBroadException
    try:
        logging_data = "\n"
        # DYN
        logging_data += "Comparing %s with numpy\n" % dyn_kernel_name
        dyn_precision, _logging_data, d_passed = __compare(goldens, absolute_precision,
                                                           percentage_thresholds, dyn_outputs, fork_count)
        logging_data += _logging_data
        # STC
        logging_data += "Comparing %s with numpy\n" % stc_kernel_name
        stc_precision, _logging_data, s_passed = __compare(goldens, absolute_precision,
                                                           percentage_thresholds, stc_outputs, fork_count)
        logging_data += _logging_data
        # REL
        logging_data += "Comparing %s with %s\n" % (dyn_kernel_name, stc_kernel_name)
        rel_precision, _logging_data, r_passed = __compare(stc_outputs, absolute_precision,
                                                           relative_percentage_thresholds, dyn_outputs, fork_count)
        logging_data += _logging_data
        # CST
        logging_data += "Comparing %s with numpy\n" % cst_kernel_name
        cst_precision, _logging_data, _ = __compare(goldens, absolute_precision,
                                                    percentage_thresholds, cst_outputs, fork_count)
        logging_data += _logging_data
        # RST
        logging_data += "Comparing %s with %s\n" % (cst_kernel_name, stc_kernel_name)
        rst_precision, _logging_data, _ = __compare(stc_outputs, absolute_precision,
                                                    relative_percentage_thresholds, cst_outputs, fork_count)
        logging_data += _logging_data
        # BIN
        logging_data += "Comparing %s with numpy\n" % bin_kernel_name
        bin_precision, _logging_data, _ = __compare(goldens, absolute_precision,
                                                    relative_percentage_thresholds, bin_outputs, fork_count)
        logging_data += _logging_data
        # BRE
        logging_data += "Comparing %s with %s\n" % (bin_kernel_name, dyn_kernel_name)
        bre_precision, _logging_data, _ = __compare(dyn_outputs, absolute_precision,
                                                    relative_percentage_thresholds, bin_outputs, fork_count)
        logging_data += _logging_data
        logging.debugc(logging_data)
        if "DYN_OFF" in dyn_precision or "DYN_INPUT_MISSING" in dyn_precision:
            if s_passed:
                passed = "STC_PASS"
            else:
                passed = "STC_FAIL"
        else:
            if d_passed and s_passed and not r_passed:
                passed = "PART"
            elif d_passed or r_passed:
                passed = "PASS"
            else:
                passed = "FAIL"
    except:
        (dyn_precision, stc_precision, rel_precision,
         cst_precision, rst_precision, bin_precision, bre_precision, passed) = ("COMPARE_FAILURE",) * 8
        logging.exception("Comparison failed")
    return (dyn_precision, stc_precision, rel_precision,
            cst_precision, rst_precision,
            bin_precision, bre_precision, passed)


def __compare(goldens: Tuple[Union[str, numpy.array], ...], absolute_precision: int,
              percentage_thresholds: Tuple[Tuple[float, float]],
              outputs: Tuple[Union[numpy.array, str], ...],
              fork_count: int):
    before_compare = time.time()
    is_pass = True
    if not check_equal_length((goldens, outputs)):
        raise RuntimeError("Comparison error: number of golden arrays not match with output arrays\n%s\n%s"
                           % (str(goldens), str(outputs)))
    for percentage_threshold in percentage_thresholds:
        if not isinstance(percentage_threshold, (tuple, list)):
            raise RuntimeError("Invalid precision tolerences: %s" % str(percentage_thresholds))
        if len(percentage_threshold) != 2:
            raise RuntimeError("Invalid precision tolerences: %s" % str(percentage_thresholds))
    if fork_count <= 0:
        raise RuntimeError("Fork count should be larger than zero!!! Received %d" % fork_count)
    elif fork_count == 1:
        precision, logging_data = __inner_compare(goldens, outputs, percentage_thresholds, absolute_precision, 1, 0)
    else:
        my_index = 0
        # Start forking
        forked_workers = []
        for i in range(fork_count - 1):
            read_pipe, write_pipe = os.pipe2(os.O_NONBLOCK)
            fork_return = os.fork()
            if fork_return == 0:
                # Child forked, break out
                os.close(read_pipe)
                fd = os.fdopen(write_pipe, 'w')
                my_index += 1
                try:
                    compare_result = __inner_compare(goldens, outputs, percentage_thresholds, absolute_precision,
                                                     fork_count, my_index)
                except BaseException as e:
                    fd.write("str(%s)\n" % str(e.args))
                else:
                    fd.write(str(compare_result))
                finally:
                    fd.close()
                # Child completed, exit
                # noinspection PyProtectedMember
                os._exit(0)
                raise RuntimeError("Placeholder")
            else:
                # Master forked, continue
                os.close(write_pipe)
                my_index += 1
                forked_workers.append(read_pipe)
        # Master index is always zero
        compare_result = __inner_compare(goldens, outputs, percentage_thresholds, absolute_precision, fork_count, 0)
        # Master completed, waiting for child
        for _ in forked_workers:
            pid, return_code = os.wait()
            if return_code != 0:
                raise RuntimeError("Forked comparison unit %d returned %d" % (pid, return_code))
        # Get child results
        child_results = [compare_result]
        for forked_read_pipe in forked_workers:
            fd = os.fdopen(forked_read_pipe)
            compare_result = eval(fd.read())
            fd.close()
            child_results.append(compare_result)
        res = child_results.pop()
        if isinstance(res, str):
            raise RuntimeError(res)
        else:
            precision, logging_data = res
        # Collect sub precisions
        for res in child_results:
            _precision, _logging_data = res
            logging_data += _logging_data
            temp_precision = ""
            for sub_pre, _sub_pre in zip(precision.split(","), _precision.split(",")):
                if temp_precision != "":
                    temp_precision += ","
                # noinspection PyBroadException
                try:
                    sub_pre = float(sub_pre)
                    _sub_pre = float(_sub_pre)
                except:
                    if sub_pre == _sub_pre:
                        temp_precision += sub_pre
                    else:
                        temp_precision += sub_pre + _sub_pre
                else:
                    temp_precision += str(eval("%s+%s" % (sub_pre, _sub_pre)))
            precision = temp_precision
    # Change precision to percentage
    temp_precision = ""
    for idx, sub_pre in enumerate(precision.split(",")):
        if temp_precision != "":
            temp_precision += ","
        # noinspection PyBroadException
        try:
            num = eval(sub_pre)
            num /= fork_count
            if (1 - num) > get(percentage_thresholds, idx)[1]:
                is_pass = False
            num_sub_pre = str(num * 100) + "%"
        except:
            temp_precision += sub_pre
            is_pass = False
        else:
            temp_precision += num_sub_pre
    precision = temp_precision
    logging.debugc("Comparison Unit costs %s" % str(time.time() - before_compare))
    return precision, logging_data, is_pass


def __inner_compare(goldens, outputs, percentage_thresholds, absolute_precision, part_count, part_index):
    precision = ""
    logging_data = ""
    if outputs:
        for idx, data_pair in enumerate(zip(outputs, goldens)):
            output, golden = data_pair
            if precision != "":
                precision += ","
            if isinstance(output, str):
                precision += output
                continue
            if isinstance(golden, str):
                precision += golden + "_GOLDEN"
                continue
            if golden.size <= 0:
                precision += "ZERO_GOLDEN"
                continue
            if golden.size != output.size:
                precision += "%d vs %d" % (len(output), len(golden))
                continue
            # Get start index and end index
            if part_count < golden.size:
                part_length = golden.size // part_count
                start_index = part_length * part_index
                end_index = start_index + part_length
                if end_index > golden.size:
                    end_index = golden.size
            else:
                if part_index >= golden.size:
                    precision += "1"
                    logging_data += "Output %d Part %d skipped\n" % (idx, part_index)
                    continue
                else:
                    start_index = part_index
                    end_index = part_index + 1
            logging_data += ("Output %d Part %d starts from %d to %d\n" % (idx, part_index, start_index, end_index))
            before_isclose = time.time()
            different_element_results = numpy.isclose(output[start_index: end_index],
                                                      golden[start_index: end_index],
                                                      rtol=get(percentage_thresholds, idx)[0],
                                                      atol=absolute_precision, equal_nan=True)
            logging.debugc("Output %d Part %d Numpy Function Call isclose() costs %s"
                           % (idx, part_index, str(time.time() - before_isclose)))
            before_where = time.time()
            different_element_indexes = numpy.where(different_element_results != numpy.array((True,)))[0]
            logging.debugc("Output %d Part %d Numpy Function Call where() costs %s" % (idx, part_index,
                                                                                       str(time.time() - before_where)))
            del different_element_results
            logging_data += ("Output %d Part %d Compare Difference length %d:\n%s\n" % (idx, part_index,
                                                                                        different_element_indexes.size,
                                                                                        different_element_indexes.size))
            compare_results = ""
            for index in range(different_element_indexes.size):
                real_index = different_element_indexes[index]
                golden_data = golden[start_index: end_index][real_index]
                actual_data = output[start_index: end_index][real_index]
                if "bool" not in str(golden.dtype):
                    compare_results += "Index: %03d RealIndex: %06d Expected: %-14.09f " \
                                       "Actual: %-14.09f Diff: %-14.06f\n" \
                                       % (index, start_index + real_index, golden_data,
                                          actual_data, (actual_data - golden_data) / golden_data)
                else:
                    compare_results += "Index: %03d RealIndex: %06d Expected: %s " \
                                       "Actual: %s Diff: FAIL\n" \
                                       % (index, start_index + real_index, str(golden_data),
                                          str(actual_data))
                if index == 100:
                    break
            logging_data += (compare_results[:-1] + "\n")
            precision += "%s" % str((golden.size - different_element_indexes.size)
                                    / golden.size)
            logging_data += "Precision is %s\n" % precision
    else:
        logging_data += "Output or Golden data is empty, compare result UNKNOWN\n"
        precision += "UNKNOWN"
    return precision, logging_data


def __outputs_to_numpy_arrays(dyn_outputs, output_dtypes):
    output_dtypes = bfloat16_conversion(output_dtypes)
    for idx, dyn_output in enumerate(dyn_outputs):
        if isinstance(dyn_output, str):
            continue
        else:
            # Probably ctypes char array
            dyn_outputs[idx] = numpy.frombuffer(dyn_output, get(output_dtypes, idx))
            if "bfloat16" in str(get(output_dtypes, idx)):
                dyn_outputs[idx] = dyn_outputs[idx].astype(numpy.float32)
