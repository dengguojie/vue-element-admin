#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("AxpyWithSoftmaxAndDropOutDoMask", None, None)

case1 = {"params": [{"shape": (96,12,24,24,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,384,384),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,24,24,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,384,384),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,24,24,16,16), "dtype": "uint8", "format": "ND", "ori_shape": (96,12,384,384),"ori_format": "ND"},
                    {"shape": (96,12,24,24,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,384,384),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,24,24,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,384,384),"ori_format": "FRACTAL_NZ"},
                    0.125, 0.5, [-1]],
         "case_name": "AxpyWithSoftmaxAndDropOutDoMask_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (96,12,8,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,128,128),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,8,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,128,128),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,8,8,16,16), "dtype": "uint8", "format": "ND", "ori_shape": (96,12,128,128),"ori_format": "ND"},
                    {"shape": (96,12,8,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,128,128),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,8,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,128,128),"ori_format": "FRACTAL_NZ"},
                    0.125, 0.5, [-1]],
         "case_name": "AxpyWithSoftmaxAndDropOutDoMask_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (96,12,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,512,512),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,512,512),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,32,32,16,16), "dtype": "uint8", "format": "ND", "ori_shape": (96,12,512,512),"ori_format": "ND"},
                    {"shape": (96,12,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,512,512),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,512,512),"ori_format": "FRACTAL_NZ"},
                    0.125, 0.5, [-1]],
         "case_name": "AxpyWithSoftmaxAndDropOutDoMask_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (96,12,16,16,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,256,256),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,16,16,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,256,256),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,16,16,16,16), "dtype": "uint8", "format": "ND", "ori_shape": (96,12,256,256),"ori_format": "ND"},
                    {"shape": (96,12,16,16,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,256,256),"ori_format": "FRACTAL_NZ"},
                    {"shape": (96,12,16,16,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96,12,256,256),"ori_format": "FRACTAL_NZ"},
                    0.125, 0.5, [-1]],
         "case_name": "AxpyWithSoftmaxAndDropOutDoMask_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A",], case1)
ut_case.add_case(["Ascend910A",], case2)
ut_case.add_case(["Ascend910A",], case3)
ut_case.add_case(["Ascend910A",], case4)
