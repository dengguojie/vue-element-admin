#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ScaledMaskedSoftmaxGrad", None, None)

case1 = {"params": [{"shape": (16,6,8,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,128),"ori_format": "ND"},
                    {"shape": (16,6,8,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,128),"ori_format": "ND"},
                    {"shape": (16,6,8,8,16,16), "dtype": "bool", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,128),"ori_format": "ND"},
                    {"shape": (16,6,8,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,128),"ori_format": "ND"},
                    2.0,
                    True],
         "case_name": "ScaledMaskedSoftmaxGrad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (16,6,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,512,512),"ori_format": "ND"},
                    {"shape": (16,6,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,512,512),"ori_format": "ND"},
                    {"shape": (16,6,32,32,16,16), "dtype": "bool", "format": "FRACTAL_NZ", "ori_shape": (16,6,512,512),"ori_format": "ND"},
                    {"shape": (16,6,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,512,512),"ori_format": "ND"},
                    3.0,
                    False],
         "case_name": "ScaledMaskedSoftmaxGrad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (16,6,32,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    {"shape": (16,6,32,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    {"shape": (16,6,32,8,16,16), "dtype": "bool", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    {"shape": (16,6,32,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    2.0,
                    False],
         "case_name": "ScaledMaskedSoftmaxGrad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (16,6,32,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    {"shape": (16,6,32,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    {"shape": (16,1,32,8,16,16), "dtype": "bool", "format": "FRACTAL_NZ", "ori_shape": (16,1,128,512),"ori_format": "ND"},
                    {"shape": (16,6,32,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    3.0,
                    False],
         "case_name": "ScaledMaskedSoftmaxGrad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (16,6,32,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    {"shape": (16,6,32,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    {"shape": (16,1,32,8,16,16), "dtype": "bool", "format": "FRACTAL_NZ", "ori_shape": (16,1,128,512),"ori_format": "ND"},
                    {"shape": (16,6,32,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,6,128,512),"ori_format": "ND"},
                    4.0,
                    False],
         "case_name": "ScaledMaskedSoftmaxGrad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A",], case1)
ut_case.add_case(["Ascend910A",], case2)
ut_case.add_case(["Ascend910A",], case3)
ut_case.add_case(["Ascend910A",], case4)
ut_case.add_case(["Ascend910A",], case5)