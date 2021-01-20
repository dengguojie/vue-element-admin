#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("Crop", None, None)


def calc_expect_func(input_x1, input_x2, input_y, axis=2, offsets=(0), kernel_name="crop", impl_mode="high_performance"):
    input_tensor = input_x1.get("value")
    shape_y = input_x2.get("value").shape
    offsets_new = [0] * len(shape_y)
    offsets_new[axis:] = offsets
    out_tensor = input_tensor[
                 offsets_new[0]:offsets_new[0] + shape_y[0],
                 offsets_new[1]:offsets_new[1] + shape_y[1],
                 offsets_new[2]:offsets_new[2] + shape_y[2],
                 offsets_new[3]:offsets_new[3] + shape_y[3]]
    return out_tensor


def gen_crop_precision_case(shape_x, shape_y, dtype, expect="success", axis=2, offsets=(0), kernel_name="crop", impl_mode="high_performance"):
    return {"params": [{"dtype": dtype, "shape": shape_x, "format": "NCHW", "ori_shape": shape_x, "ori_format": "NCHW",
                        "param_type": "input", "value_range": [-10, 10]},
                       {"dtype": dtype, "shape": shape_y, "format": "NCHW", "ori_shape": shape_y, "ori_format": "NCHW",
                        "param_type": "input", "value_range": [-10, 10]},
                       {"dtype": dtype, "shape": shape_y, "format": "NCHW", "ori_shape": shape_y, "ori_format": "NCHW",
                        "param_type": "output", "value_range": [-10, 10]},
                       axis, offsets],
            "case_name": kernel_name,
            "expect": expect,
            "calc_expect_func": calc_expect_func}


case1 = {"params": [{"shape": (2, 3, 2, 16, 6, 7), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 3, 2, 16, 6, 7), "ori_format": "NCHW"},
                    {"shape": (2, 2, 1, 8, 5, 6), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 1, 8, 5, 6), "ori_format": "NCHW"},
                    {"shape": (2, 2, 1, 8, 5, 6), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 1, 8, 5, 6), "ori_format": "NCHW"},
                    0, [0, 0, 0, 0, 0, 0]],
         "case_name": "crop_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 3, 2, 3), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 3, 2, 3), "ori_format": "NCHW"},
                    {"shape": (2, 3, 2, 3), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 3, 2, 3), "ori_format": "NCHW"},
                    {"shape": (2, 3, 2, 3), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 3, 2, 3), "ori_format": "NCHW"},
                    1, [5]],
         "case_name": "crop_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1, 21, 568, 568), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 21, 568, 568), "ori_format": "NCHW"},
                    {"shape": (1, 3, 500, 500), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 3, 500, 500), "ori_format": "NCHW"},
                    {"shape": (1, 3, 500, 500), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 3, 500, 500), "ori_format": "NCHW"},
                    0, [0, 3, 4, 5]],
         "case_name": "crop_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (1, 21, 568, 568), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 21, 568, 568), "ori_format": "NCHW"},
                    {"shape": (1, 3, 500, 500), "dtype": "float32", "format": "NCHW", "ori_shape": (1, 3, 500, 500), "ori_format": "NCHW"},
                    {"shape": (1, 3, 500, 500), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 500, 500), "ori_format": "NCHW"},
                    0, [0, 3, 4, 5]],
         "case_name": "crop_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (1, 21, 568, 568), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 21, 568, 568), "ori_format": "NCHW"},
                    {"shape": (1, 3, 500, 500), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 3, 500, 500), "ori_format": "NCHW"},
                    {"shape": (1, 3, 500, 500), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 3, 500, 500), "ori_format": "NCHW"},
                    1, [0, 3, 4, 5]],
         "case_name": "crop_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

def test_get_op_support_info(test_arg):
    from impl.crop import get_op_support_info
    get_op_support_info({"shape": (1, 3, 568, 568), "dtype": "int32", "format": "ND", "ori_shape": (1, 3, 568, 568), "ori_format": "ND"},
                        {"shape": (1, 3, 500, 500), "dtype": "int32", "format": "ND", "ori_shape": (1, 3, 500, 500), "ori_format": "ND"},
                        {"shape": (1, 3, 500, 500), "dtype": "int32", "format": "ND", "ori_shape": (1, 3, 500, 500), "ori_format": "ND"},
                        2, [0, 0, 0, 0])

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_case("all", case4)
ut_case.add_case("all", case5)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)

ut_case.add_precision_case("all",
                           gen_crop_precision_case((10000, 4, 5, 6), (1849, 1, 1, 1), "float16", "success", 0,
                                                   (0, 0, 0, 0), kernel_name="crop_1"))
ut_case.add_precision_case("all",
                           gen_crop_precision_case((2, 23, 149, 179), (1, 14, 134, 134), "float16", "success", 0,
                                                   (1, 1, 1, 1), kernel_name="crop_2"))
# ut_case.add_precision_case("all",
#                            gen_crop_precision_case((2048, 8, 12, 14), (1024, 4, 3, 7), "float16", "success", 0,
#                                                    (0, 0, 0, 0), kernel_name="crop_3"))
# ut_case.add_precision_case("all",
#                            gen_crop_precision_case((130, 8, 12, 14), (128, 4, 3, 7), "float16", "success", 0,
#                                                    (0, 0, 0, 0), kernel_name="crop_4"))
# ut_case.add_precision_case("all",
#                            gen_crop_precision_case((64, 88, 140, 120), (32, 40, 70, 60), "float16", "success", -3,
#                                                    (0, 0, 0), kernel_name="crop_5"))


