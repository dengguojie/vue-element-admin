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
    axis_c1hw = input_shape[0]
    axis_no = input_shape[1]
    axis_ni = input_shape[2]
    axis_c0 = input_shape[3]
    n_pad = None if axis_no * axis_ni == dst_shape[0] else dst_shape[0] - axis_no * axis_ni

    tmp_input_tensor = np.transpose(input_tensor, axes=(1, 2, 0, 3))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_no * axis_ni, axis_c1hw, axis_c0)
    output_tensor = tmp_input_tensor[:n_pad, :, :].reshape(dst_shape)

    return output_tensor


err1 = {"params": [{"shape": (24, 1, 16, 16), "dtype": "float16",
                    "ori_shape": (24, 1, 16, 16), "format": "FRACTAL_NZ",
                    "ori_format": "FRACTAL_NZ"},
                   {"shape": (16, 2, 3, 4, 16), "dtype": "int8",
                    "ori_shape": (16, 2, 3, 4, 16), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                   "FRACTAL_NZ", "NC1HWC0"],
        "expect": RuntimeError,
        "format_expect": ["NC1HWC0"],
        "support_expect": False}

err2 = {"params": [{"shape": (24, 1, 16, 17), "dtype": "float16",
                    "ori_shape": (24, 1, 16, 17), "format": "FRACTAL_NZ",
                    "ori_format": "FRACTAL_NZ"},
                   {"shape": (16, 2, 3, 4, 16), "dtype": "float16",
                    "ori_shape": (16, 2, 3, 4, 16), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                   "FRACTAL_NZ", "NC1HWC0"],
        "expect": RuntimeError,
        "format_expect": ["NC1HWC0"],
        "support_expect": False}

err3 = {"params": [{"shape": (24, 1, 17, 16), "dtype": "float16",
                    "ori_shape": (24, 1, 17, 16), "format": "FRACTAL_NZ",
                    "ori_format": "FRACTAL_NZ"},
                   {"shape": (16, 2, 3, 4, 16), "dtype": "float16",
                    "ori_shape": (16, 2, 3, 4, 16), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                   "FRACTAL_NZ", "NC1HWC0"],
        "expect": RuntimeError,
        "format_expect": ["NC1HWC0"],
        "support_expect": False}

err4 = {"params": [{"shape": (24, 1, 16, 16), "dtype": "float16",
                    "ori_shape": (24, 1, 16, 16), "format": "FRACTAL_NZ",
                    "ori_format": "FRACTAL_NZ"},
                   {"shape": (16, 2, 3, 4, 17), "dtype": "float16",
                    "ori_shape": (16, 2, 3, 4, 17), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                   "FRACTAL_NZ", "NC1HWC0"],
        "expect": RuntimeError,
        "format_expect": ["NC1HWC0"],
        "support_expect": False}

err5 = {"params": [{"shape": (24, 2, 16, 16), "dtype": "float16",
                    "ori_shape": (24, 2, 16, 16), "format": "FRACTAL_NZ",
                    "ori_format": "FRACTAL_NZ"},
                   {"shape": (16, 2, 3, 4, 16), "dtype": "float16",
                    "ori_shape": (16, 2, 3, 4, 16), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                   "FRACTAL_NZ", "NC1HWC0"],
        "expect": RuntimeError,
        "format_expect": ["NC1HWC0"],
        "support_expect": False}

err6 = {"params": [{"shape": (25, 1, 16, 16), "dtype": "float16",
                    "ori_shape": (25, 1, 16, 16), "format": "FRACTAL_NZ",
                    "ori_format": "FRACTAL_NZ"},
                   {"shape": (16, 2, 3, 4, 16), "dtype": "float16",
                    "ori_shape": (16, 2, 3, 4, 16), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                   "FRACTAL_NZ", "NC1HWC0"],
        "expect": RuntimeError,
        "format_expect": ["NC1HWC0"],
        "support_expect": False}

case1 = {"params": [{"shape": (24, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (24, 1, 16, 16), "format": "FRACTAL_NZ",
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (16, 2, 3, 4, 16), "dtype": "float16",
                     "ori_shape": (16, 2, 3, 4, 16), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                    "FRACTAL_NZ", "NC1HWC0"],
         "expect": "success",
         "format_expect": ["NC1HWC0"],
         "support_expect": True}

case2 = {"params": [{"shape": (32, 8, 16, 16), "dtype": "float16",
                     "ori_shape": (32, 8, 16, 16), "format": "FRACTAL_NZ",
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (113, 2, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (113, 2, 1, 16, 16), "format": "NC1HWC0",
                     "ori_format": "NC1HWC0"},
                    "FRACTAL_NZ", "NC1HWC0"],
         "expect": "success",
         "format_expect": ["NC1HWC0"],
         "support_expect": True}

case3 = {"params": [{"shape": (128, 2, 16, 16), "dtype": "float16",
                     "ori_shape": (128, 2, 16, 16), "format": "FRACTAL_NZ",
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (17, 4, 1, 32, 16), "dtype": "float16",
                     "ori_shape": (17, 4, 1, 32, 16), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                    "FRACTAL_NZ", "NC1HWC0"],
         "expect": "success",
         "format_expect": ["NC1HWC0"],
         "support_expect": True}

case4 = {"params": [{"shape": (4082, 248, 16, 16), "dtype": "float16",
                     "ori_shape": (4082, 248, 16, 16), "format": "FRACTAL_NZ",
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (3953, 2, 1, 2041, 16), "dtype": "float16",
                     "ori_shape": (3953, 2, 1, 2041, 16), "format": "NC1HWC0",
                     "ori_format": "NC1HWC0"},
                    "FRACTAL_NZ", "NC1HWC0"],
         "expect": "success",
         "format_expect": ["NC1HWC0"],
         "support_expect": True}

case5 = {"params": [{"shape": (4082, 256, 16, 16), "dtype": "float16",
                     "ori_shape": (4082, 256, 16, 16), "format": "FRACTAL_NZ",
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (4081, 2, 1, 2041, 16), "dtype": "float16",
                     "ori_shape": (4081, 2, 1, 2041, 16), "format": "NC1HWC0",
                     "ori_format": "NC1HWC0"},
                    "FRACTAL_NZ", "NC1HWC0"],
         "expect": "success",
         "format_expect": ["NC1HWC0"],
         "support_expect": True}

case6 = {"params": [{"shape": (24, 1, 16, 32), "dtype": "int8",
                     "ori_shape": (24, 1, 16, 32), "format": "FRACTAL_NZ",
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (16, 2, 3, 4, 32), "dtype": "int8",
                     "ori_shape": (16, 2, 3, 4, 32), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                    "FRACTAL_NZ", "NC1HWC0"],
         "expect": "success",
         "format_expect": ["NC1HWC0"],
         "support_expect": True}

case7 = {"params": [{"shape": (32, 8, 16, 32), "dtype": "int8",
                     "ori_shape": (32, 8, 16, 32), "format": "FRACTAL_NZ",
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (113, 2, 1, 16, 32), "dtype": "int8",
                     "ori_shape": (113, 2, 1, 16, 32), "format": "NC1HWC0",
                     "ori_format": "NC1HWC0"},
                    "FRACTAL_NZ", "NC1HWC0"],
         "expect": "success",
         "format_expect": ["NC1HWC0"],
         "support_expect": True}

case8 = {"params": [{"shape": (128, 2, 16, 32), "dtype": "int8",
                     "ori_shape": (128, 2, 16, 32), "format": "FRACTAL_NZ",
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (17, 4, 1, 32, 32), "dtype": "int8",
                     "ori_shape": (17, 4, 1, 32, 32), "format": "NC1HWC0", "ori_format": "NC1HWC0"},
                    "FRACTAL_NZ", "NC1HWC0"],
         "expect": "success",
         "format_expect": ["NC1HWC0"],
         "support_expect": True}

case9 = {"params": [{"shape": (4082, 248, 16, 32), "dtype": "int8",
                     "ori_shape": (4082, 248, 16, 32), "format": "FRACTAL_NZ",
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (3953, 2, 1, 2041, 32), "dtype": "int8",
                     "ori_shape": (3953, 2, 1, 2041, 32), "format": "NC1HWC0",
                     "ori_format": "NC1HWC0"},
                    "FRACTAL_NZ", "NC1HWC0"],
         "expect": "success",
         "format_expect": ["NC1HWC0"],
         "support_expect": True}

case10 = {"params": [{"shape": (4082, 256, 16, 32), "dtype": "int8",
                      "ori_shape": (4082, 256, 16, 32), "format": "FRACTAL_NZ",
                      "ori_format": "FRACTAL_NZ"},
                     {"shape": (4081, 2, 1, 2041, 32), "dtype": "int8",
                      "ori_shape": (4081, 2, 1, 2041, 32), "format": "NC1HWC0",
                      "ori_format": "NC1HWC0"},
                     "FRACTAL_NZ", "NC1HWC0"],
          "expect": "success",
          "format_expect": ["NC1HWC0"],
          "support_expect": True}

case11 = {"params": [{"shape": (32, 13, 16, 16), "dtype": "float16",
                      "ori_shape": (32, 13, 16, 16), "format": "FRACTAL_NZ",
                      "ori_format": "FRACTAL_NZ"},
                     {"shape": (200, 2, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (200, 2, 1, 16, 16), "format": "NC1HWC0",
                      "ori_format": "NC1HWC0"},
                     "FRACTAL_NZ", "NC1HWC0"],
          "expect": "success",
          "format_expect": ["NC1HWC0"],
          "support_expect": True}

ut_case.add_case(["Ascend910"], err1)
ut_case.add_case(["Ascend910"], err2)
ut_case.add_case(["Ascend910"], err3)
ut_case.add_case(["Ascend910"], err4)
ut_case.add_case(["Ascend910"], err5)
ut_case.add_case(["Ascend910"], err6)
ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910"], case3)
ut_case.add_case(["Ascend910"], case4)
ut_case.add_case(["Ascend910"], case5)
ut_case.add_case(["Ascend910"], case6)
ut_case.add_case(["Ascend910"], case7)
ut_case.add_case(["Ascend910"], case8)
ut_case.add_case(["Ascend910"], case9)
ut_case.add_case(["Ascend910"], case10)
ut_case.add_case(["Ascend910"], case11)

# add precision cases
case12 = {"params": [{"shape": (24, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (24, 1, 16, 16), "format": "FRACTAL_NZ",
                      "ori_format": "FRACTAL_NZ",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (16, 2, 3, 4, 16), "dtype": "float16",
                      "ori_shape": (16, 2, 3, 4, 16), "format": "NC1HWC0", "ori_format": "NC1HWC0",
                      "param_type": "output"},
                     "FRACTAL_NZ", "NC1HWC0"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case13 = {"params": [{"shape": (32, 8, 16, 16), "dtype": "float16",
                      "ori_shape": (32, 8, 16, 16), "format": "FRACTAL_NZ",
                      "ori_format": "FRACTAL_NZ",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (113, 2, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (113, 2, 1, 16, 16), "format": "NC1HWC0",
                      "ori_format": "NC1HWC0",
                      "param_type": "output"},
                     "FRACTAL_NZ", "NC1HWC0"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case14 = {"params": [{"shape": (128, 2, 16, 16), "dtype": "float16",
                      "ori_shape": (128, 2, 16, 16), "format": "FRACTAL_NZ",
                      "ori_format": "FRACTAL_NZ",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (17, 4, 1, 32, 16), "dtype": "float16",
                      "ori_shape": (17, 4, 1, 32, 16), "format": "NC1HWC0", "ori_format": "NC1HWC0",
                      "param_type": "output"},
                     "FRACTAL_NZ", "NC1HWC0"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case15 = {"params": [{"shape": (24, 1, 16, 32), "dtype": "int8",
                      "ori_shape": (24, 1, 16, 32), "format": "FRACTAL_NZ",
                      "ori_format": "FRACTAL_NZ",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (16, 2, 3, 4, 32), "dtype": "int8",
                      "ori_shape": (16, 2, 3, 4, 32), "format": "NC1HWC0", "ori_format": "NC1HWC0",
                      "param_type": "output"},
                     "FRACTAL_NZ", "NC1HWC0"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case16 = {"params": [{"shape": (32, 8, 16, 32), "dtype": "int8",
                      "ori_shape": (32, 8, 16, 32), "format": "FRACTAL_NZ",
                      "ori_format": "FRACTAL_NZ",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (113, 2, 1, 16, 32), "dtype": "int8",
                      "ori_shape": (113, 2, 1, 16, 32), "format": "NC1HWC0",
                      "ori_format": "NC1HWC0",
                      "param_type": "output"},
                     "FRACTAL_NZ", "NC1HWC0"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case17 = {"params": [{"shape": (128, 2, 16, 32), "dtype": "int8",
                      "ori_shape": (128, 2, 16, 32), "format": "FRACTAL_NZ",
                      "ori_format": "FRACTAL_NZ",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (17, 4, 1, 32, 32), "dtype": "int8",
                      "ori_shape": (17, 4, 1, 32, 32), "format": "NC1HWC0", "ori_format": "NC1HWC0",
                      "param_type": "output"},
                     "FRACTAL_NZ", "NC1HWC0"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case18 = {"params": [{"shape": (32, 13, 16, 16), "dtype": "float16",
                      "ori_shape": (32, 13, 16, 16), "format": "FRACTAL_NZ",
                      "ori_format": "FRACTAL_NZ",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (200, 2, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (200, 2, 1, 16, 16), "format": "NC1HWC0",
                      "ori_format": "NC1HWC0",
                      "param_type": "output"},
                     "FRACTAL_NZ", "NC1HWC0"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

# ut_case.add_precision_case(["Ascend910"], case12)
# ut_case.add_precision_case(["Ascend910"], case13)
# ut_case.add_precision_case(["Ascend910"], case14)
# ut_case.add_precision_case(["Ascend910"], case15)
# ut_case.add_precision_case(["Ascend910"], case16)
# ut_case.add_precision_case(["Ascend910"], case17)
# ut_case.add_precision_case(["Ascend910"], case18)

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
