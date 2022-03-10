#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from impl.util import util_common
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
                  0, -1, True, RuntimeError, "scale_3")

case4 = scale_cce((2, 3, 2, 3), (2), (1, 3, 1, 1), (2, 3, 2, 3),
                  (2, 3, 2, 3), (1, 3, 1, 1), (1, 3, 1, 1), (2, 3, 2, 3),
                  "float32", "float32", "float32", "float32",
                  "ND", "ND", "ND", "ND",
                  "ND", "ND", "ND", "ND",
                  1, 1, True, RuntimeError, "scale_4")

case5 = scale_cce((2, 3, 2, 3), (2, 3, 2, 5), (1, 3, 1, 1), (2, 3, 2, 3),
                  (2, 3, 2, 3), (1, 3, 1, 1), (1, 3, 1, 1), (2, 3, 2, 3),
                  "float32", "float32", "float32", "float32",
                  "ND", "ND", "ND", "ND",
                  "ND", "ND", "ND", "ND",
                  1, 1, True, RuntimeError, "scale_5")

case6 = scale_cce((2, 3, 2, 3, 4), (1, 3, 1, 1, 1), (1, 3, 1, 1, 1), (2, 3, 2, 3, 4),
                  (2, 3, 2, 3, 4), (1, 3, 1, 1, 1), (1, 3, 1, 1, 1), (2, 3, 2, 3, 4),
                  "float32", "float32", "float32", "float32",
                  "ND", "ND", "ND", "ND",
                  "ND", "ND", "ND", "ND",
                  4, 1, True, RuntimeError, "scale_6")

case7 = scale_cce((2, 3, 2, 3, 4), (1, 3, 1, 1, 1), (1, 3, 1, 1, 1), (2, 3, 2, 3, 4),
                  (2, 3, 2, 3, 4), (1, 3, 1, 1, 1), (1, 3, 1, 1, 1), (2, 3, 2, 3, 4),
                  "float32", "float32", "float32", "float32",
                  "ND", "ND", "ND", "ND",
                  "ND", "ND", "ND", "ND",
                  1, -2, True, RuntimeError, "scale_7")

case8 = scale_cce((2, 3, 2, 3, 4), (1, 3, 1, 1, 1), (1, 3, 1, 1, 1), (2, 3, 2, 3, 4),
                  (2, 3, 2, 3, 4), (1, 3, 1, 1, 1), (1, 3, 1, 1, 1), (2, 3, 2, 3, 4),
                  "float32", "float32", "float32", "float32",
                  "ND", "ND", "ND", "ND",
                  "ND", "ND", "ND", "ND",
                  -1, 1, True, RuntimeError, "scale_8")

case9 = scale_cce((1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16),
                  (1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16), (1, 1, 448, 448, 16),
                  "float16", "float16", "float16", "float16",
                  "ND", "ND", "ND", "ND",
                  "ND", "ND", "ND", "ND",
                  0, -1, False, "success", "scale_9")

ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)
ut_case.add_case("Hi3796CV300ES", case3)
ut_case.add_case("Ascend910A", case4)
ut_case.add_case("Ascend910A", case5)
ut_case.add_case("Ascend910A", case6)
ut_case.add_case("Ascend910A", case7)
ut_case.add_case("Ascend910A", case8)
ut_case.add_case("Ascend910A", case9)


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
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
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


# pylint: disable=unused-argument
# ut_case.scale_test_cfg_cov_case("all")
def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.scale import op_select_format
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     "test_scale_op_select_format_1")
    op_select_format({"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     "test_scale_op_select_format_2")
    op_select_format({"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     "test_scale_op_select_format_3")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     "test_scale_op_select_format_4")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     "test_scale_op_select_format_5")
    op_select_format({"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     "test_scale_op_select_format_6")
    op_select_format({"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     "test_scale_op_select_format_7")
    op_select_format({"shape": (16, 16, 32, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 32, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 32, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 32, 16),
                      "ori_format": "NCHW"},
                     "test_scale_op_select_format_8")
    op_select_format({"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,), "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     "test_scale_op_select_format_9")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     "test_scale_op_select_format_10")
    op_select_format({"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     "test_scale_op_select_format_11")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     "test_scale_op_select_format_12")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "int8", "format": "NHWC", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NHWC"},
                     {"shape": (1,), "dtype": "int8", "format": "NHWC", "ori_shape": (1,),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 16, 1), "dtype": "int8", "format": "NHWC", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NHWC"},
                     "test_scale_op_select_format_13")
    op_select_format({"shape": (4,64,200,320), "dtype": "float16", "format": "NCHW", "ori_shape": (4,64,200,320),
                      "ori_format": "NCHW"},
                     {"shape": (1,64,1,1), "dtype": "float16", "format": "NCHW", "ori_shape": (1,64,1,1),
                      "ori_format": "NCHW"},
                     {"shape": (4,64,200,320), "dtype": "float16", "format": "NCHW", "ori_shape": (4,64,200,320),
                      "ori_format": "NCHW"},
                     "test_scale_op_select_format_14")
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     "test_scale_op_select_format_15")
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN"},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN"},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN"},
                     "test_scale_op_select_format_16")
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 8},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 8},
                     "test_scale_op_select_format_17")
    op_select_format({"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (),
                      "ori_format": "NHWC", "sub_format" : 0},
                     {"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (),
                      "ori_format": "NHWC", "sub_format" : 0},
                     {"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (),
                      "ori_format": "NHWC", "sub_format" : 0},
                     "test_scale_op_select_format_18")

    def __test_util_commom():
        input_parm = ({"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,), "ori_format": "NHWC", "sub_format" : 0},
                      {"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,), "ori_format": "NHWC", "sub_format" : 0},
                      {"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,), "ori_format": "NHWC", "sub_format" : 0})
        util_common.is_support_fractal_z_inputs(input_parm)

    __test_util_commom()


def test_op_select_format_001(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.scale import op_select_format
    te_set_version("SD3403")
    op_select_format(
        {
            "shape": (16, 16, 16, 16),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (16, 16, 16, 16),
            "ori_format": "NCHW"
        }, {
            "shape": (16, 16, 16, 16),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (16, 16, 16, 16),
            "ori_format": "NCHW"
        }, None, None)
    op_select_format({
        "shape": (2,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2,),
        "ori_format": "ND"
    }, {
        "shape": (2,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2,),
        "ori_format": "ND"
    }, None, None)

ut_case.add_cust_test_func(test_func=test_op_select_format_001)
ut_case.add_cust_test_func(support_soc="all", test_func=test_op_select_format)