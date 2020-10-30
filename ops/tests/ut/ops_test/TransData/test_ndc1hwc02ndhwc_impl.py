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
    axis_c = dst_shape[4]
    axis_n = input_shape[0]
    axis_d = input_shape[1]
    axis_c1 = input_shape[2]
    axis_h = input_shape[3]
    axis_w = input_shape[4]
    axis_c0 = input_shape[5]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0

    tmp_input_tensor = np.transpose(input_tensor, axes=(0, 1, 3, 4, 2, 5))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_n, axis_d, axis_h, axis_w, axis_c1 * axis_c0)
    output_tensor = tmp_input_tensor[:, :, :, :, :c_pad]

    return output_tensor


err1 = {"params": [{"shape": (1, 1, 16, 1, 1, 16), "dtype": "float16",
                    "ori_shape": (1, 1, 16, 1, 1, 16), "format": "NDC1HWC0",
                    "ori_format": "NDC1HWC0"},
                   {"shape": (1, 1, 1, 1, 256), "dtype": "float32",
                    "ori_shape": (1, 1, 1, 1, 256), "format": "NDHWC", "ori_format": "NDHWC"},
                   "NDC1HWC0", "NDHWC"],
        "expect": RuntimeError,
        "format_expect": ["NDHWC"],
        "support_expect": False}

err2 = {"params": [{"shape": (1, 1, 16, 1, 1, 17), "dtype": "float16",
                    "ori_shape": (1, 1, 16, 1, 1, 17), "format": "NDC1HWC0",
                    "ori_format": "NDC1HWC0"},
                   {"shape": (1, 1, 1, 1, 256), "dtype": "float16",
                    "ori_shape": (1, 1, 1, 1, 256), "format": "NDHWC", "ori_format": "NDHWC"},
                   "NDC1HWC0", "NDHWC"],
        "expect": RuntimeError,
        "format_expect": ["NDHWC"],
        "support_expect": False}

err3 = {"params": [{"shape": (2, 1, 16, 1, 1, 16), "dtype": "float16",
                    "ori_shape": (2, 1, 16, 1, 1, 16), "format": "NDC1HWC0",
                    "ori_format": "NDC1HWC0"},
                   {"shape": (1, 1, 1, 1, 256), "dtype": "float16",
                    "ori_shape": (1, 1, 1, 1, 256), "format": "NDHWC", "ori_format": "NDHWC"},
                   "NDC1HWC0", "NDHWC"],
        "expect": RuntimeError,
        "format_expect": ["NDHWC"],
        "support_expect": False}

err4 = {"params": [{"shape": (1, 1, 15, 1, 1, 16), "dtype": "float16",
                    "ori_shape": (1, 1, 15, 1, 1, 16), "format": "NDC1HWC0",
                    "ori_format": "NDC1HWC0"},
                   {"shape": (1, 1, 1, 1, 256), "dtype": "float16",
                    "ori_shape": (1, 1, 1, 1, 256), "format": "NDHWC", "ori_format": "NDHWC"},
                   "NDC1HWC0", "NDHWC"],
        "expect": RuntimeError,
        "format_expect": ["NDHWC"],
        "support_expect": False}

case1 = {"params": [{"shape": (1, 1, 16, 1, 1, 16), "dtype": "float16",
                     "ori_shape": (1, 1, 16, 1, 1, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (1, 1, 1, 1, 256), "dtype": "float16",
                     "ori_shape": (1, 1, 1, 1, 256), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "NDC1HWC0", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case2 = {"params": [{"shape": (4, 5, 8, 26, 26, 16), "dtype": "float16",
                     "ori_shape": (4, 5, 8, 26, 26, 16), "format": "NDC1HWC0",
                     "ori_format": "NDC1HWC0",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (4, 5, 26, 26, 128), "dtype": "float16",
                     "ori_shape": (4, 5, 26, 26, 128), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "NDC1HWC0", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_case(["Ascend910"], err1)
ut_case.add_case(["Ascend910"], err2)
ut_case.add_case(["Ascend910"], err3)
ut_case.add_case(["Ascend910"], err4)
ut_case.add_precision_case(["Ascend910"], case1)
# ut_case.add_precision_case(["Ascend910"], case2)

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
