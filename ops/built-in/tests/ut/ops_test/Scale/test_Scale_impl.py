#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("Scale", None, None)


def scale_cce(shape_x, shape_scale, shape_bias, shape_y,
              shape_x_ori, shape_scale_ori, shape_bias_ori, shape_y_ori,
              dtype_x, dtype_scale, dtype_bias, dtype_y,
              format_x, format_scale, format_bias, format_y,
              format_x_ori, format_scale_ori, format_bias_ori, format_y_ori,
              axis=1, num_axes=1, scale_from_blob=True, expect="success", kernel_name_val="scale"):

    x_input = {"shape": shape_x, "ori_shape": shape_x_ori,
               "format": format_x, "ori_format": format_x_ori, "dtype": dtype_x}
    scale_input = {"shape": shape_scale, "ori_shape": shape_scale_ori,
                   "format": format_scale, "ori_format": format_scale_ori, "dtype": dtype_scale}
    bias_input = None
    if len(shape_bias) > 0:
        bias_input = {"shape": shape_bias, "ori_shape": shape_bias_ori,
                      "format": format_bias, "ori_format": format_bias_ori, "dtype": dtype_bias}

    y = {"shape": shape_y, "ori_shape": shape_y_ori,
         "format": format_y, "ori_format": format_y_ori, "dtype": dtype_y}

    return {"params": [x_input, scale_input, bias_input, y, axis, num_axes, scale_from_blob],
            "case_name":kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = scale_cce((2, 3, 2, 3), (1, 3, 1, 1), (1, 3, 1, 1), (2, 3, 2, 3),
                  (2, 3, 2, 3), (1, 3, 1, 1), (1, 3, 1, 1), (2, 3, 2, 3),
                  "float32", "float32", "float32", "float32",
                  "ND", "ND", "ND", "ND",
                  "ND", "ND", "ND", "ND",
                  1, 1, True, "success", "scale_1")

case2 = scale_cce((1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16),
                  (1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16),
                  "float16", "float16", "float16", "float16",
                  "ND", "ND", "ND", "ND",
                  "ND", "ND", "ND", "ND",
                  0, -1, True, "success", "scale_2")

case3 = scale_cce((1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16),
                  (1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16),
                  "float32", "float32", "float32", "float32",
                  "ND", "ND", "ND", "ND",
                  "ND", "ND", "ND", "ND",
                  0, -1, True, RuntimeError, "scale_2")

ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)
ut_case.add_case("Hi3796CV300ES", case3)

def calc_expect_func(x, scale, bias, y, axis, num_axes, scale_from_blob):
    if bias is not None:
        res = x['value'] * scale['value'] + bias['value']
    else:
        res = x['value'] * scale['value']
    return res

precision_case1 = {"params": [{"shape": (2, 3, 2, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 2, 3),"ori_format": "ND","param_type":"input"}, #x
                              {"shape": (2, 3, 2, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 2, 3),"ori_format": "ND","param_type":"input"},
                              {"shape": (2, 3, 2, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 2, 3),"ori_format": "ND","param_type":"input"},
                              {"shape": (2, 3, 2, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 2, 3),"ori_format": "ND","param_type":"output"},
                              1, 1, True
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}
precision_case2 = {"params": [{"shape": (1,1,4,8,16), "dtype": "float16", "format": "ND", "ori_shape": (1,1,4,8,16),"ori_format": "ND","param_type":"input"}, #x
                              {"shape": (1,1,4,8,16), "dtype": "float16", "format": "ND", "ori_shape": (1,1,4,8,16),"ori_format": "ND","param_type":"input"},
                              {"shape": (1,1,4,8,16), "dtype": "float16", "format": "ND", "ori_shape": (1,1,4,8,16),"ori_format": "ND","param_type":"input"},
                              {"shape": (1,1,4,8,16), "dtype": "float16", "format": "ND", "ori_shape": (1,1,4,8,16),"ori_format": "ND","param_type":"output"},
                              0, -1, True
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910A", precision_case1)
ut_case.add_precision_case("Ascend910A", precision_case2)
