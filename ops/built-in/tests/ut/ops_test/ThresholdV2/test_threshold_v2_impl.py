# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import torch

ut_case = OpUT("ThresholdV2", "impl.threshold_v2", "threshold_v2")

def calc_expect_func(input_x, threshold, value, output):
    x = input_x["value"]
    threshold_val = threshold["value"][0]
    y = output

    if value is None:
        value_val = 0
    else:
        value_val = value["value"][0]

    if input_x["dtype"] == "float16":
        for i in np.nditer(x, op_flags=['readwrite']):
            if i <= threshold_val:
                i[...] = value_val
        return [x, ]

    x = torch.tensor(x)
    res = torch.threshold(x, threshold_val, value_val).numpy()
    return [res, ]

def gen_use_value_case(shape_val, dtype_val, precision):
    return{
        "params": [
            {"ori_shape": shape_val, "shape": shape_val,
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val,
            "value_range": [1.0, 8.0], "param_type": "input"},

            {"ori_shape": (1,), "shape": (1,),
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val,
            "value": np.array([5], dtype=dtype_val), "param_type": "input"},

            {"ori_shape": (1,), "shape": (1,),
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val,
            "value": np.array([10], dtype=dtype_val), "param_type": "input"},

            {"ori_shape": shape_val, "shape": shape_val,
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val, "param_type": "output"}],

        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(precision, precision),
        "case_name": "use_value_" + dtype_val
    }

def gen_no_value_case(shape_val, dtype_val, precision):
    return{
        "params": [
            {"ori_shape": shape_val, "shape": shape_val,
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val,
            "value_range": [1.0, 8.0], "param_type": "input"},

            {"ori_shape": (1,), "shape": (1,),
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val,
            "value": np.array([5], dtype=dtype_val), "param_type": "input"},

            None,

            {"ori_shape": shape_val, "shape": shape_val,
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val, "param_type": "output"}],

        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(precision, precision),
        "case_name": "no_value_" + dtype_val
    }

ut_case.add_case("all", {
    "params": [
        {"ori_shape": (32,), "shape": (32,), "ori_format": "ND", "format": "ND", "dtype": "float16"},
        {"ori_shape": (2, 2), "shape": (2, 2), "ori_format": "ND", "format": "ND", "dtype": "float16"},
        {"ori_shape": (2, 2), "shape": (2, 2), "ori_format": "ND", "format": "ND", "dtype": "float16"},
        {"ori_shape": (32,), "shape": (32,), "ori_format": "ND", "format": "ND", "dtype": "float16"}],
    "case_name": "para_check",
    "expect": RuntimeError
})

ut_case.add_precision_case("Ascend910A", gen_use_value_case((4, 5), "float32", 0.0001))
ut_case.add_precision_case("Ascend910A", gen_use_value_case((5, 6, 7, 8), "float32", 0.0001))
ut_case.add_precision_case("all", gen_use_value_case((2, 8), "float16", 0.001))
ut_case.add_precision_case("all", gen_use_value_case((2, 8), "int32", 0.001))
ut_case.add_precision_case("all", gen_use_value_case((2, 8), "uint8", 0.001))
ut_case.add_precision_case("all", gen_use_value_case((2, 8), "int8", 0.001))

ut_case.add_precision_case("Ascend910A", gen_no_value_case((4, 5), "float32", 0.0001))
ut_case.add_precision_case("Ascend910A", gen_no_value_case((5, 6, 7, 8), "float32", 0.0001))
ut_case.add_precision_case("all", gen_no_value_case((2, 8), "float16", 0.001))
ut_case.add_precision_case("all", gen_no_value_case((2, 8), "int32", 0.001))
ut_case.add_precision_case("all", gen_no_value_case((2, 8), "uint8", 0.001))
ut_case.add_precision_case("all", gen_no_value_case((2, 8), "int8", 0.001))
