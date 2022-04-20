#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def calc_expect_func(src, dst, src_format, dst_format):
    input_shape = src.get("shape")
    input_tensor = src.get("value")
    dst_shape = dst.get("shape")
    axis_c = dst_shape[1]
    axis_n = input_shape[0]
    axis_d = input_shape[1]
    axis_c1 = input_shape[2]
    axis_h = input_shape[3]
    axis_w = input_shape[4]
    axis_c0 = input_shape[5]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0

    tmp_input_tensor = np.transpose(input_tensor, axes=(0, 2, 5, 1, 3, 4))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_n, axis_c1 * axis_c0,
                                                axis_d, axis_h, axis_w)
    output_tensor = tmp_input_tensor[:, :c_pad, :, :, :]

    return output_tensor


err1 = {"params": [{"shape": (1, 1, 16, 1, 1, 16), "dtype": "float16",
                    "ori_shape": (1, 1, 16, 1, 1, 16), "format": "NDC1HWC0",
                    "ori_format": "NDC1HWC0"},
                   {"shape": (1, 256, 1, 1, 1), "dtype": "float32",
                    "ori_shape": (1, 256, 1, 1, 1), "format": "NCDHW", "ori_format": "NCDHW"},
                   "NDC1HWC0", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

err2 = {"params": [{"shape": (1, 1, 16, 1, 1, 17), "dtype": "float16",
                    "ori_shape": (1, 1, 16, 1, 1, 17), "format": "NDC1HWC0",
                    "ori_format": "NDC1HWC0"},
                   {"shape": (1, 256, 1, 1, 1), "dtype": "float16",
                    "ori_shape": (1, 256, 1, 1, 1), "format": "NCDHW", "ori_format": "NCDHW"},
                   "NDC1HWC0", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

err3 = {"params": [{"shape": (2, 1, 16, 1, 1, 16), "dtype": "float16",
                    "ori_shape": (2, 1, 16, 1, 1, 16), "format": "NDC1HWC0",
                    "ori_format": "NDC1HWC0"},
                   {"shape": (1, 256, 1, 1, 1), "dtype": "float16",
                    "ori_shape": (1, 256, 1, 1, 1), "format": "NCDHW", "ori_format": "NCDHW"},
                   "NDC1HWC0", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

err4 = {"params": [{"shape": (1, 1, 15, 1, 1, 16), "dtype": "float16",
                    "ori_shape": (1, 1, 15, 1, 1, 16), "format": "NDC1HWC0",
                    "ori_format": "NDC1HWC0"},
                   {"shape": (1, 256, 1, 1, 1), "dtype": "float16",
                    "ori_shape": (1, 256, 1, 1, 1), "format": "NCDHW", "ori_format": "NCDHW"},
                   "NDC1HWC0", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

err5 = {"params": [{"shape": (1, 1, 16, 1, 1, 16), "dtype": "float16",
                    "ori_shape": (1, 1, 16, 1, 1, 16), "format": "NDC1HWC0",
                    "ori_format": "NDC1HWC0"},
                   {"shape": (1, 256, 1, 1, 1), "dtype": "float16",
                    "ori_shape": (1, 256, 1, 1, 1), "format": "NCDHW", "ori_format": "NCDHW"},
                   "NDC1HWC0", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

case1 = {"params": [{"shape": (1, 2, 8, 126, 126, 16), "dtype": "float32",
                     "ori_shape": (1, 2, 8, 126, 126, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (1, 128, 2, 126, 126), "dtype": "float32",
                     "ori_shape": (1, 128, 2, 126, 126), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "NDC1HWC0", "NCDHW"],
         "case_name": "ndc1hwc0_2_ncdhw_001",
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case2 = {"params": [{"shape": (2, 3, 2, 16, 3, 16), "dtype": "float16",
                     "ori_shape": (2, 3, 2, 16, 3, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (2, 32, 3, 16, 3), "dtype": "float16",
                     "ori_shape": (2, 32, 3, 16, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "NDC1HWC0", "NCDHW"],
         "case_name": "ndc1hwc0_2_ncdhw_002",
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case3 = {"params": [{"shape": (2, 3, 4096, 16, 2, 16), "dtype": "float16",
                     "ori_shape": (2, 3, 4096, 16, 2, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (2, 65536, 3, 16, 2), "dtype": "float16",
                     "ori_shape": (2, 65536, 3, 16, 2), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "NDC1HWC0", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case4 = {"params": [{"shape": (2, 3, 2, 1, 5120, 16), "dtype": "float16",
                     "ori_shape": (2, 3, 2, 1, 5120, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (2, 32, 3, 1, 5120), "dtype": "float16",
                     "ori_shape": (2, 32, 3, 1, 5120), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "NDC1HWC0", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

# split big
case5 = {"params": [{"shape": (2, 3, 4375, 1, 5120, 16), "dtype": "float16",
                     "ori_shape": (2, 3, 4375, 1, 5120, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (2, 70000, 3, 1, 5120), "dtype": "float16",
                     "ori_shape": (2, 70000, 3, 1, 5120), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "NDC1HWC0", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case6 = {"params": [{"shape": (2, 3, 4094, 1, 3960, 16), "dtype": "float16",
                     "ori_shape": (2, 3, 4094, 1, 3960, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (2, 65500, 3, 1, 3960), "dtype": "float16",
                     "ori_shape": (2, 65500, 3, 1, 3960), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "NDC1HWC0", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

# hwc0_core
case7 = {"params": [{"shape": (2, 3, 3952, 1, 3952, 16), "dtype": "float16",
                     "ori_shape": (2, 3, 3952, 1, 3952, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (2, 63232, 3, 1, 3952), "dtype": "float16",
                     "ori_shape": (2, 63232, 3, 1, 3952), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "NDC1HWC0", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case8 = {"params": [{"shape": (2, 3, 3952, 1, 3951, 16), "dtype": "float16",
                     "ori_shape": (2, 3, 3952, 1, 3951, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (2, 63232, 3, 1, 3951), "dtype": "float16",
                     "ori_shape": (2, 63232, 3, 1, 3951), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "NDC1HWC0", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case9 = {"params": [{"shape": (2, 3, 4096, 15, 2, 16), "dtype": "float16",
                     "ori_shape": (2, 3, 4096, 15, 2, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (2, 65536, 3, 15, 2), "dtype": "float16",
                     "ori_shape": (2, 65536, 3, 15, 2), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "NDC1HWC0", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case10 = {"params": [{"shape": (2, 3, 2, 15, 3, 16), "dtype": "float16",
                      "ori_shape": (2, 3, 2, 15, 3, 16), "format": "NDC1HWC0",
                      "ori_format": "NDC1HWC0",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (2, 32, 3, 15, 3), "dtype": "float16",
                      "ori_shape": (2, 32, 3, 15, 3), "format": "NCDHW", "ori_format": "NCDHW",
                      "param_type": "output"},
                     "NDC1HWC0", "NCDHW"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case11 = {"params": [{"shape": (1, 1, 128, 1, 1, 16), "dtype": "float16",
                      "ori_shape": (1, 1, 128, 1, 1, 16), "format": "NDC1HWC0",
                      "ori_format": "NDC1HWC0",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (1, 2048, 1, 1, 1), "dtype": "float16",
                      "ori_shape": (1, 2048, 1, 1, 1), "format": "NCDHW", "ori_format": "NCDHW",
                      "param_type": "output"},
                     "NDC1HWC0", "NCDHW"],
          "case_name": "ndc1hwc0_2_ncdhw_011",
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case12 = {"params": [{"shape": (1, 1, 32, 1, 1, 16), "dtype": "float16",
                      "ori_shape": (1, 1, 32, 1, 1, 16), "format": "NDC1HWC0",
                      "ori_format": "NDC1HWC0",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (1, 512, 1, 1, 1), "dtype": "float16",
                      "ori_shape": (1, 512, 1, 1, 1), "format": "NCDHW", "ori_format": "NCDHW",
                      "param_type": "output"},
                     "NDC1HWC0", "NCDHW"],
          "case_name": "ndc1hwc0_2_ncdhw_012",
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


ut_case.add_case(["Ascend310","Ascend910A"], case1)
ut_case.add_case(["Ascend310","Ascend910A"], case2)
ut_case.add_case(["Ascend310","Ascend910A"], case11)
ut_case.add_case(["Ascend310","Ascend910A"], case12)


if __name__ == '__main__':
    ut_case.run()
    exit(0)
