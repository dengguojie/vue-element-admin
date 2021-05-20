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


err1 = {"params": [{"shape": (120, 4, 16, 16), "dtype": "float16",
                    "ori_shape": (120, 4, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 32, 3, 4, 5), "dtype": "float32",
                    "ori_shape": (64, 32, 3, 4, 5), "format": "NCDHW", "ori_format": "NCDHW"},
                   "FRACTAL_Z_3D", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

err2 = {"params": [{"shape": (120, 4, 32, 16), "dtype": "float16",
                    "ori_shape": (120, 4, 32, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 32, 3, 4, 5), "dtype": "float16",
                    "ori_shape": (64, 32, 3, 4, 5), "format": "NCDHW", "ori_format": "NCDHW"},
                   "FRACTAL_Z_3D", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

err3 = {"params": [{"shape": (120, 4, 16, 32), "dtype": "float16",
                    "ori_shape": (120, 4, 16, 32), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 32, 3, 4, 5), "dtype": "float16",
                    "ori_shape": (64, 32, 3, 4, 5), "format": "NCDHW", "ori_format": "NCDHW"},
                   "FRACTAL_Z_3D", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

err4 = {"params": [{"shape": (120, 3, 16, 16), "dtype": "float16",
                    "ori_shape": (120, 3, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 32, 3, 4, 5), "dtype": "float16",
                    "ori_shape": (64, 32, 3, 4, 5), "format": "NCDHW", "ori_format": "NCDHW"},
                   "FRACTAL_Z_3D", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

err5 = {"params": [{"shape": (100, 4, 16, 16), "dtype": "float16",
                    "ori_shape": (100, 4, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (64, 32, 3, 4, 5), "dtype": "float16",
                    "ori_shape": (64, 32, 3, 4, 5), "format": "NCDHW", "ori_format": "NCDHW"},
                   "FRACTAL_Z_3D", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

err6 = {"params": [{"shape": (35100, 8, 16, 16), "dtype": "float16",
                    "ori_shape": (35100, 8, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (128, 200, 3, 30, 30), "dtype": "float16",
                    "ori_shape": (128, 200, 3, 30, 30), "format": "NCDHW", "ori_format": "NCDHW"},
                   "FRACTAL_Z_3D", "NCDHW"],
        "expect": RuntimeError,
        "format_expect": ["NCDHW"],
        "support_expect": False}

case1 = {"params": [{"shape": (27, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (27, 1, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (16, 5, 3, 3, 3), "dtype": "float16",
                     "ori_shape": (16, 5, 3, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case2 = {"params": [{"shape": (27, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (27, 1, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (16, 16, 3, 3, 3), "dtype": "float16",
                     "ori_shape": (16, 16, 3, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case3 = {"params": [{"shape": (27, 2, 16, 16), "dtype": "float16",
                     "ori_shape": (27, 2, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (32, 16, 3, 3, 3), "dtype": "float16",
                     "ori_shape": (32, 16, 3, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case4 = {"params": [{"shape": (216, 8, 16, 16), "dtype": "float16",
                     "ori_shape": (216, 8, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (128, 128, 3, 3, 3), "dtype": "float16",
                     "ori_shape": (128, 128, 3, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case5 = {"params": [{"shape": (24, 8, 16, 16), "dtype": "float16",
                     "ori_shape": (24, 8, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (128, 128, 1, 1, 3), "dtype": "float16",
                     "ori_shape": (128, 128, 1, 1, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case6 = {"params": [{"shape": (27, 1, 16, 16), "dtype": "float32",
                     "ori_shape": (27, 1, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (16, 5, 3, 3, 3), "dtype": "float32",
                     "ori_shape": (16, 5, 3, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case7 = {"params": [{"shape": (48, 3, 16, 16), "dtype": "float32",
                     "ori_shape": (48, 3, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (1, 2, 2, 8, 126), "dtype": "float32",
                     "ori_shape": (1, 2, 2, 8, 126), "format": "FRACTAL_Z_3D", "ori_format": "DHWCN",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "DHWCN", 21],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": False}

case8 = {"params": [{"shape": (27, 2, 16, 16), "dtype": "float32",
                     "ori_shape": (27, 2, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (32, 16, 3, 3, 3), "dtype": "float32",
                     "ori_shape": (32, 16, 3, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case9 = {"params": [{"shape": (216, 8, 16, 16), "dtype": "float32",
                     "ori_shape": (216, 8, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (128, 128, 3, 3, 3), "dtype": "float32",
                     "ori_shape": (128, 128, 3, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case10 = {"params": [{"shape": (24, 8, 16, 16), "dtype": "float32",
                      "ori_shape": (24, 8, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (128, 128, 1, 1, 3), "dtype": "float32",
                      "ori_shape": (128, 128, 1, 1, 3), "format": "NCDHW", "ori_format": "NCDHW",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "NCDHW"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case11 = {"params": [{"shape": (216, 8, 16, 16), "dtype": "float32",
                      "ori_shape": (216, 8, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (128, 127, 3, 3, 3), "dtype": "float32",
                      "ori_shape": (128, 127, 3, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "NCDHW"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case12 = {"params": [{"shape": (351, 8, 16, 16), "dtype": "float16",
                      "ori_shape": (351, 8, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (128, 200, 3, 3, 3), "dtype": "float16",
                      "ori_shape": (128, 200, 3, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "NCDHW"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case15 = {"params": [{"shape": (21600, 8, 16, 16), "dtype": "float32",
                     "ori_shape": (21600, 8, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (128, 128, 3, 3, 300), "dtype": "float32",
                     "ori_shape": (128, 128, 3, 3, 300), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": False}

case16 = {"params": [{"shape": (21600, 8, 16, 16), "dtype": "float32",
                     "ori_shape": (21600, 8, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (128, 128, 300, 3, 3), "dtype": "float32",
                     "ori_shape": (128, 128, 300, 3, 3), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": False}

case17 = {"params": [{"shape": (34300, 4, 16, 16), "dtype": "float32",
                     "ori_shape": (34300, 4, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (64, 3, 7, 700, 7), "dtype": "float32",
                     "ori_shape": (64, 3, 7, 700, 7), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": False}

case18 = {"params": [{"shape": (384, 32, 16, 16), "dtype": "float32",
                     "ori_shape": (384, 32, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D",
                     "param_type": "input", "value_range": [-10.0, 10.0]},
                    {"shape": (512, 2048, 3, 1, 1), "dtype": "float32",
                     "ori_shape": (512, 2048, 3, 1, 1), "format": "NCDHW", "ori_format": "NCDHW",
                     "param_type": "output"},
                    "FRACTAL_Z_3D", "NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": False}

ut_case.add_case(["Ascend910"], err1)
ut_case.add_case(["Ascend910"], err2)
ut_case.add_case(["Ascend910"], err6)
ut_case.add_precision_case(["Ascend910"], case1)
ut_case.add_precision_case(["Ascend910"], case5)
ut_case.add_precision_case(["Ascend910"], case6)
# ut_case.add_precision_case(["Ascend910"], case10)
# ut_case.add_precision_case(["Ascend910"], case11)

ut_case.add_case(["Ascend910"], err3)
ut_case.add_case(["Ascend910"], err4)
ut_case.add_case(["Ascend910"], err5)
ut_case.add_precision_case(["Ascend910"], case2)
ut_case.add_precision_case(["Ascend910"], case3)
ut_case.add_precision_case(["Ascend910"], case4)
ut_case.add_case(["Ascend310","Ascend910A"], case7)
ut_case.add_case(["Ascend310","Ascend910A"], case15)
ut_case.add_case(["Ascend310","Ascend910A"], case16)
ut_case.add_case(["Ascend310","Ascend910A"], case17)
ut_case.add_case(["Ascend310","Ascend910A"], case18)
#ut_case.add_precision_case(["Ascend910"], case8)
#ut_case.add_precision_case(["Ascend910"], case9)
#ut_case.add_precision_case(["Ascend910"], case12)

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
