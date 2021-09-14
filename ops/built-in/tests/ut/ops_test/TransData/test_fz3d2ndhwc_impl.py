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
    axis_d = dst_shape[1]
    axis_h = dst_shape[2]
    axis_w = dst_shape[3]
    axis_c = dst_shape[4]
    axis_dc1hw = input_shape[0]
    axis_no = input_shape[1]
    axis_ni = input_shape[2]
    axis_c0 = input_shape[3]
    axis_c1 = axis_dc1hw // axis_d // axis_h // axis_w
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0

    tmp_input_tensor = input_tensor.reshape(axis_d, axis_c1, axis_h, axis_w,
                                            axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(4, 5, 0, 2, 3, 1, 6))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_no * axis_ni, axis_d,
                                                axis_h, axis_w, axis_c1 * axis_c0)
    output_tensor = tmp_input_tensor[:n_pad, :, :, :, :c_pad]

    return output_tensor


err1 = {"params": [{"shape": (120, 4, 16, 16), "dtype": "float16",
                    "ori_shape": (120, 4, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 3, 4, 5, 32), "dtype": "float32",
                    "ori_shape": (64, 3, 4, 5, 32), "format": "NDHWC", "ori_format": "NDHWC"},
                   "FRACTAL_Z_3D", "NDHWC"],
        "expect": RuntimeError,
        "format_expect": ["NDHWC"],
        "support_expect": False}

err2 = {"params": [{"shape": (120, 4, 32, 16), "dtype": "float16",
                    "ori_shape": (120, 4, 32, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 3, 4, 5, 32), "dtype": "float16",
                    "ori_shape": (64, 3, 4, 5, 32), "format": "NDHWC", "ori_format": "NDHWC"},
                   "FRACTAL_Z_3D", "NDHWC"],
        "expect": RuntimeError,
        "format_expect": ["NDHWC"],
        "support_expect": False}

err3 = {"params": [{"shape": (120, 4, 16, 32), "dtype": "float16",
                    "ori_shape": (120, 4, 16, 32), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 3, 4, 5, 32), "dtype": "float16",
                    "ori_shape": (64, 3, 4, 5, 32), "format": "NDHWC", "ori_format": "NDHWC"},
                   "FRACTAL_Z_3D", "NDHWC"],
        "expect": RuntimeError,
        "format_expect": ["NDHWC"],
        "support_expect": False}

err4 = {"params": [{"shape": (120, 3, 16, 16), "dtype": "float16",
                    "ori_shape": (120, 3, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 3, 4, 5, 32), "dtype": "float16",
                    "ori_shape": (64, 3, 4, 5, 32), "format": "NDHWC", "ori_format": "NDHWC"},
                   "FRACTAL_Z_3D", "NDHWC"],
        "expect": RuntimeError,
        "format_expect": ["NDHWC"],
        "support_expect": False}

err5 = {"params": [{"shape": (100, 4, 16, 16), "dtype": "float16",
                    "ori_shape": (100, 4, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 3, 4, 5, 32), "dtype": "float16",
                    "ori_shape": (64, 3, 4, 5, 32), "format": "NDHWC", "ori_format": "NDHWC"},
                   "FRACTAL_Z_3D", "NDHWC"],
        "expect": RuntimeError,
        "format_expect": ["NDHWC"],
        "support_expect": False}

case1 = {"params": [{"shape": (27, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (27, 1, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (15, 3, 3, 3, 5), "dtype": "float16",
                     "ori_shape": (15, 3, 3, 3, 5), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case2 = {"params": [{"shape": (27, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (27, 1, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (16, 3, 3, 3, 16), "dtype": "float16",
                     "ori_shape": (16, 3, 3, 3, 16), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case3 = {"params": [{"shape": (27, 2, 16, 16), "dtype": "float16",
                     "ori_shape": (27, 2, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (32, 3, 3, 3, 16), "dtype": "float16",
                     "ori_shape": (32, 3, 3, 3, 16), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case4 = {"params": [{"shape": (216, 8, 16, 16), "dtype": "float16",
                     "ori_shape": (216, 8, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (128, 3, 3, 3, 128), "dtype": "float16",
                     "ori_shape": (128, 3, 3, 3, 128), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case5 = {"params": [{"shape": (24, 8, 16, 16), "dtype": "float16",
                     "ori_shape": (24, 8, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (128, 1, 1, 3, 128), "dtype": "float16",
                     "ori_shape": (128, 1, 1, 3, 128), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case6 = {"params": [{"shape": (27, 1, 16, 16), "dtype": "float32",
                     "ori_shape": (27, 1, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (16, 3, 3, 3, 5), "dtype": "float32",
                     "ori_shape": (16, 3, 3, 3, 5), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case7 = {"params": [{"shape": (27, 1, 16, 16), "dtype": "float32",
                     "ori_shape": (27, 1, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (16, 3, 3, 3, 16), "dtype": "float32",
                     "ori_shape": (16, 3, 3, 3, 16), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case8 = {"params": [{"shape": (27, 2, 16, 16), "dtype": "float32",
                     "ori_shape": (27, 2, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (32, 3, 3, 3, 16), "dtype": "float32",
                     "ori_shape": (32, 3, 3, 3, 16), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case9 = {"params": [{"shape": (216, 8, 16, 16), "dtype": "float32",
                     "ori_shape": (216, 8, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (128, 3, 3, 3, 128), "dtype": "float32",
                     "ori_shape": (128, 3, 3, 3, 128), "format": "NDHWC", "ori_format": "NDHWC",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NDHWC"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case10 = {"params": [{"shape": (24, 8, 16, 16), "dtype": "float32",
                      "ori_shape": (24, 8, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (128, 1, 1, 3, 128), "dtype": "float32",
                      "ori_shape": (128, 1, 1, 3, 128), "format": "NDHWC", "ori_format": "NDHWC",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "NDHWC"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_case(["Ascend910"], err1)
ut_case.add_case(["Ascend910"], err2)
ut_case.add_case(["Ascend910"], err3)
ut_case.add_case(["Ascend910"], err4)
ut_case.add_case(["Ascend910"], err5)
#ut_case.add_precision_case(["Ascend910A"], case1)
#ut_case.add_precision_case(["Ascend910"], case2)
#ut_case.add_precision_case(["Ascend910"], case3)
#ut_case.add_precision_case(["Ascend910"], case4)
#ut_case.add_precision_case(["Ascend910"], case5)
# ut_case.add_precision_case(["Ascend910"], case6)
# ut_case.add_precision_case(["Ascend910"], case7)
# ut_case.add_precision_case(["Ascend910"], case8)
# ut_case.add_precision_case(["Ascend910"], case9)
# ut_case.add_precision_case(["Ascend910"], case10)

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
