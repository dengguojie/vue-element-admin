#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("CumulativeLogsumexpD", "impl.cumulativelogsumexp_d", "cumulative_logsumexp_d")

def calc_expect_func(input, output, axis, exclusive, reverse):
    value_exp = np.exp(input["value"])
    value_exp = np.swapaxes(value_exp, 0, axis)
    if reverse:
        value_exp = value_exp[::-1, ...]
    cum_exp = np.cumsum(value_exp, axis=0)
    np_log = np.log(cum_exp)
    if exclusive:
        value = 1
        if input["dtype"] == "float16":
            value = -2 ** 15 * 1.9991
        elif input["dtype"] == "float32":
            value = -2 ** 127 * 1.9999999
        np_to_broad_cast = np.ones_like(np_log[0:1]) * value
        np_log = np.concatenate([np_to_broad_cast, np_log])[:-1, ...]

    if reverse:
        np_log = np_log[::-1, ...]

    np_res = np.swapaxes(np_log, 0, axis)

    return np_res

ut_case.add_precision_case("all", {
    "params": [
        {"shape": (1,), "dtype": "float16", "format": "ND",
         "ori_shape": (1,), "ori_format": "ND", "param_type":"input"},
        {"shape": (1,), "dtype": "float16", "format": "ND",
         "ori_shape": (1,), "ori_format": "ND", "param_type":"output"},
        0, False, False],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_precision_case("Ascend910", {
    "params": [
        {"shape": (1, 16, 16), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND", "param_type":"input"},
        {"shape": (1, 16, 16), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND", "param_type":"output"},
        0, False, False],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_precision_case("all", {
    "params": [
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND", "param_type":"input"},
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND", "param_type":"output"},
        0, True, False],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_precision_case("all", {
    "params": [
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND", "param_type":"input"},
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND", "param_type":"output"},
        0, True, True],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_precision_case("all", {
    "params": [
        {"shape": (1, 16, 13), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 13), "ori_format": "ND", "param_type":"input"},
        {"shape": (1, 16, 13), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 13), "ori_format": "ND", "param_type":"output"},
        0, True, True],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_case("all", {
    "params": [
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        0, False, True],
    "expect": "success",
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 12, 13), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13), "ori_format": "ND"},
        {"shape": (1, 12, 13), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13), "ori_format": "ND"},
        0, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 12, 13), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13), "ori_format": "ND"},
        {"shape": (1, 12, 13), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13), "ori_format": "ND"},
        -2, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 12, 13, 15), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13, 15), "ori_format": "ND"},
        {"shape": (1, 12, 13, 15), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13, 15), "ori_format": "ND"},
        2, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 12, 13, 15), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13, 15), "ori_format": "ND"},
        {"shape": (1, 12, 13, 15), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13, 15), "ori_format": "ND"},
        3, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        1, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        1, False, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        1, True, False],
    "expect": "success"
})
ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        1, False, False],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {"shape": (10, 1, 13, 15, 223), "dtype": "float16", "format": "ND",
         "ori_shape": (10, 1, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 1, 13, 15, 223), "dtype": "float16", "format": "ND",
         "ori_shape": (10, 1, 13, 15, 223), "ori_format": "ND"},
        1, True, False],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (35, 1, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (35, 1, 13, 15, 223), "ori_format": "ND"},
        {"shape": (35, 1, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (35, 1, 13, 15, 223), "ori_format": "ND"},
        1, True, False],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (35, 1, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (35, 1, 13, 15, 223), "ori_format": "ND"},
        {"shape": (35, 1, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (35, 1, 13, 15, 223), "ori_format": "ND"},
        -1, True, False],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {"shape": (1, 2, 3, 4), "dtype": "int16", "format": "ND",
         "ori_shape": (1, 2, 3, 4), "ori_format": "ND"},
        {"shape": (1, 2, 3, 4), "dtype": "int16", "format": "ND",
         "ori_shape": (1, 2, 3, 4), "ori_format": "ND"},
        2, False, False],
    "expect": RuntimeError
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 2, 4, 5), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 2, 3, 4), "ori_format": "ND"},
        {"shape": (1, 2, 3, 4), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 2, 3, 4), "ori_format": "ND"},
        10, False, False],
    "expect": RuntimeError
})

ut_case.add_case("Ascend310", {
    "params": [
        {"shape": (10, 12, 13, 15, 223), "dtype": "float16", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 12, 13, 15, 223), "dtype": "float16", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        1, False, False],
    "expect": "success"
})

