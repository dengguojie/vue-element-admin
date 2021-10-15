#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("LayerNormBetaGammaBackpropV2", None, None)

case1 = {"params": [{"shape": (64, 768, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float32", "format": "ND", "ori_shape": (1024,), "ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float32", "format": "ND", "ori_shape": (1024,), "ori_format": "ND"},
                    [1024]],
         "case_name": "layer_norm_beta_gamma_backprop_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [
    {"shape": (12288, 1024), "dtype": "float32", "format": "ND", "ori_shape": (12288, 1024), "ori_format": "ND"},
    {"shape": (12288, 1024), "dtype": "float32", "format": "ND", "ori_shape": (12288, 1024), "ori_format": "ND"},
    {"shape": (1024,), "dtype": "float32", "format": "ND", "ori_shape": (1024,), "ori_format": "ND"},
    {"shape": (1024,), "dtype": "float32", "format": "ND", "ori_shape": (1024,), "ori_format": "ND"},
    [1024]],
         "case_name": "layer_norm_beta_gamma_backprop_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,), "ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,), "ori_format": "ND"},
                    [1024]],
         "case_name": "layer_norm_beta_gamma_backprop_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [
    {"shape": (12288, 1024), "dtype": "float16", "format": "ND", "ori_shape": (12288, 1024), "ori_format": "ND"},
    {"shape": (12288, 1024), "dtype": "float16", "format": "ND", "ori_shape": (12288, 1024), "ori_format": "ND"},
    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,), "ori_format": "ND"},
    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,), "ori_format": "ND"},
    [1024]],
         "case_name": "layer_norm_beta_gamma_backprop_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    [1]],
         "case_name": "layer_norm_beta_gamma_backprop_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (2, 2, 2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2, 2, 2), "ori_format": "ND"},
                    {"shape": (2, 2, 2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2, 2, 2), "ori_format": "ND"},
                    {"shape": (2, 2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2, 2), "ori_format": "ND"},
                    {"shape": (2, 2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2, 2), "ori_format": "ND"},
                    [2, 2, 2]],
         "case_name": "layer_norm_beta_gamma_backprop_v2_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend710", "Ascend910A"], case6)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
