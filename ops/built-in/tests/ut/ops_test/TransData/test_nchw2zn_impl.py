#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", None, None)


def calc_expect_func(src, dst, src_format, dst_format):
    input_shape = src.get("shape")
    input_tensor = src.get("value")
    dst_shape = dst.get("shape")
    axis_n = dst_shape[0]
    axis_c = dst_shape[1]
    axis_d = dst_shape[2]
    axis_h = dst_shape[3]
    axis_w = dst_shape[4]
    axis_dc1hw = input_shape[0]
    axis_no = input_shape[1]
    axis_ni = input_shape[2]
    axis_c0 = input_shape[3]
    axis_c1 = axis_dc1hw // axis_d // axis_h // axis_w
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0

    tmp_input_tensor = input_tensor.reshape(axis_d, axis_c1, axis_h, axis_w,
                                            axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(4, 5, 1, 6, 0, 2, 3))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_no * axis_ni, axis_c1 * axis_c0,
                                                axis_d, axis_h, axis_w)
    output_tensor = tmp_input_tensor[:n_pad, :c_pad, :, :, :]

    return output_tensor


case1 = {"params": [{"shape": (36, 2, 16, 106), "dtype": "float16",
                     "ori_shape": (36, 2, 16, 106), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (1696, 3, 16, 16), "dtype": "float16",
                     "ori_shape": (1696, 3, 16, 16), "format": "FRACTAL_Z", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z", 1],
         "expect": "success",
         "format_expect": ["FRACTAL_Z"],
         "support_expect": False}

case2 = {"params": [{"shape": (3, 17, 16, 16), "dtype": "float16",
                     "ori_shape": (3, 17, 16, 16), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (512, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (512, 1, 16, 16), "format": "FRACTAL_Z", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z", 1],
         "expect": "success",
         "format_expect": ["FRACTAL_Z"],
         "support_expect": False}


case3 = {"params": [{"shape": (3, 17, 1, 10001), "dtype": "float16",
                     "ori_shape": (3, 17, 1, 10001), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (20002, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (20002, 1, 16, 16), "format": "FRACTAL_Z", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z", 1],
         "expect": "success",
         "format_expect": ["FRACTAL_Z"],
         "support_expect": False}

case4 = {"params": [{"shape": (36, 2, 16, 106), "dtype": "float32",
                     "ori_shape": (36, 2, 16, 106), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (1696, 3, 16, 16), "dtype": "float32",
                     "ori_shape": (1696, 3, 16, 16), "format": "FRACTAL_Z", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z", 1],
         "expect": "success",
         "format_expect": ["FRACTAL_Z"],
         "support_expect": False}

case5 = {"params": [{"shape": (3, 17, 16, 16), "dtype": "float32",
                     "ori_shape": (3, 17, 16, 16), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (512, 1, 16, 16), "dtype": "float32",
                     "ori_shape": (512, 1, 16, 16), "format": "FRACTAL_Z", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z", 1],
         "expect": "success",
         "format_expect": ["FRACTAL_Z"],
         "support_expect": False}


case6 = {"params": [{"shape": (3, 17, 1, 10001), "dtype": "float32",
                     "ori_shape": (3, 17, 1, 10001), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (20002, 1, 16, 16), "dtype": "float32",
                     "ori_shape": (20002, 1, 16, 16), "format": "FRACTAL_Z", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z", 1],
         "expect": "success",
         "format_expect": ["FRACTAL_Z"],
         "support_expect": False}

case7 = {"params": [{"shape": (36, 2, 16, 106), "dtype": "int8",
                     "ori_shape": (36, 2, 16, 106), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (1696, 3, 16, 32), "dtype": "int8",
                     "ori_shape": (1696, 3, 16, 32), "format": "FRACTAL_Z", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z", 1],
         "expect": "success",
         "format_expect": ["FRACTAL_Z"],
         "support_expect": False}

case8 = {"params": [{"shape": (3, 17, 16, 16), "dtype": "int8",
                     "ori_shape": (3, 17, 16, 16), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (256, 1, 16, 32), "dtype": "int8",
                     "ori_shape": (256, 1, 16, 32), "format": "FRACTAL_Z", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z", 1],
         "expect": "success",
         "format_expect": ["FRACTAL_Z"],
         "support_expect": False}


case9 = {"params": [{"shape": (3, 17, 1, 10001), "dtype": "int8",
                     "ori_shape": (3, 17, 1, 10001), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (10001, 1, 16, 32), "dtype": "int8",
                     "ori_shape": (10001, 1, 16, 32), "format": "FRACTAL_Z", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z", 1],
         "expect": "success",
         "format_expect": ["FRACTAL_Z"],
         "support_expect": False}

ut_case.add_case(["Ascend310", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend910A"], case9)
