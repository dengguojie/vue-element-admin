#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.layer_norm_x_backprop_v2 import op_select_format

ut_case = OpUT("LayerNormXBackpropV2", None, None)

case1 = {"params": [{"shape": (64, 768, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (12288,), "dtype": "float32", "format": "ND", "ori_shape": (12288,), "ori_format": "ND"},
                    {"shape": (12288,), "dtype": "float32", "format": "ND", "ori_shape": (12288,), "ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float32", "format": "ND", "ori_shape": (256,), "ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"}, ],
         "case_name": "layer_norm_x_backprop_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (12288,), "dtype": "float16", "format": "ND", "ori_shape": (12288,), "ori_format": "ND"},
                    {"shape": (12288,), "dtype": "float16", "format": "ND", "ori_shape": (12288,), "ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (256,), "ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"}, ],
         "case_name": "layer_norm_x_backprop_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"}],
         "case_name": "layer_norm_x_backprop_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"}],
         "case_name": "layer_norm_x_backprop_v2_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (5120, 32, 4), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (5120, 32, 4), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (5120, 32, 1), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (5120, 32, 1), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (4,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (5120, 32, 4), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (5120, 32, 4), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND"}],
         "case_name": "layer_norm_x_backprop_v2_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {
    "params": [{"shape": (48, 2304, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                "ori_format": "ND"},
               {"shape": (48, 2304, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                "ori_format": "ND"},
               {"shape": (36864, 1), "dtype": "float16", "format": "ND", "ori_shape": (12288,), "ori_format": "ND"},
               {"shape": (36864, 1), "dtype": "float16", "format": "ND", "ori_shape": (12288,), "ori_format": "ND"},
               {"shape": (768,), "dtype": "float16", "format": "ND", "ori_shape": (256,), "ori_format": "ND"},
               {"shape": (48, 2304, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                "ori_format": "ND"},
               {"shape": (48, 2304, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                "ori_format": "ND"}, ],
    "case_name": "layer_norm_x_backprop_v2_6",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)


def check_format(support_dtype, support_format, format_json):
    import json
    obj = json.loads(format_json)

    def _check_param_format(param_name):
        result_dtype = set(obj.get(param_name).get("dtype").split(","))
        if result_dtype != support_dtype:
            raise RuntimeError("dtype of {} expected:{} actual:{}".format(param_name, support_dtype, result_dtype))

        result_format = set(obj.get(param_name).get("format").split(","))
        if result_format != support_format:
            raise RuntimeError(
                "format of {} expected:{} actual:{}".format(param_name, support_format, result_format))

    _check_param_format("input0")
    _check_param_format("output0")


def _test_op_select_format_func(x_shape, mean_shape, gamma_shape, expect_dtype, expect_format, case_name):
    res = op_select_format(
        {"shape": x_shape, "dtype": "float32", "format": "ND", "ori_shape": x_shape, "ori_format": "ND"},
        {"shape": x_shape, "dtype": "float32", "format": "ND", "ori_shape": x_shape, "ori_format": "ND"},
        {"shape": mean_shape, "dtype": "float32", "format": "ND", "ori_shape": mean_shape, "ori_format": "ND"},
        {"shape": mean_shape, "dtype": "float32", "format": "ND", "ori_shape": mean_shape, "ori_format": "ND"},
        {"shape": gamma_shape, "dtype": "float32", "format": "ND", "ori_shape": gamma_shape, "ori_format": "ND"},
        {"shape": x_shape, "dtype": "float32", "format": "ND", "ori_shape": x_shape, "ori_format": "ND"},
        {"shape": x_shape, "dtype": "float32", "format": "ND", "ori_shape": x_shape, "ori_format": "ND"},
        case_name)

    check_format(expect_dtype, expect_format, res)


def test_op_select_format(test_arg):
    x_shape = (64, 64, 1024)
    mean_shape = (64, 64, 1)
    gamma_shape = (1024,)
    case_name = "test_beta_gamma_v2_op_select_format_1"
    expect_dtype = {'float16', ' float', 'float16', 'float16', 'float16', 'float', 'float', 'float'}
    expect_format = {"FRACTAL_NZ", "FRACTAL_NZ", "NCHW", "NHWC", "ND", "NCHW", "NHWC", "ND"}
    _test_op_select_format_func(x_shape, mean_shape, gamma_shape, expect_dtype, expect_format, case_name)

    x_shape = (5120, 16384)
    mean_shape = (5120, 1)
    gamma_shape = (16384,)
    case_name = "test_beta_gamma_v2_op_select_format_2"
    expect_dtype = {'float16', 'float16', 'float16', 'float', 'float', 'float'}
    expect_format = {"NCHW", "NHWC", "ND", "NCHW", "NHWC", "ND"}
    _test_op_select_format_func(x_shape, mean_shape, gamma_shape, expect_dtype, expect_format, case_name)


ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
