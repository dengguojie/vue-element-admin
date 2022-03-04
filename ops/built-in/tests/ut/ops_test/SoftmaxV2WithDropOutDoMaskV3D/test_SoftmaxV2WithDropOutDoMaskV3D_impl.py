#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("SoftmaxV2WithDropOutDoMaskV3D", None, None)

case1 = {"params": [{"shape": (24,16,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24,16,512,512),"ori_format": "FRACTAL_NZ"},
                    {"shape": (24,16,32,32,16,16), "dtype": "uint8", "format": "FRACTAL_NZ", "ori_shape": (24,16,512,512),"ori_format": "FRACTAL_NZ"},
                    {"shape": (24,16,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24,16,512,512),"ori_format": "FRACTAL_NZ"},
                    {"shape": (24,16,32,32,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24,16,512,512),"ori_format": "FRACTAL_NZ"},0.5,[-1]],
         "case_name": "softmaxdomask_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


case2 = {"params": [{"shape": (24,16,24,24,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24,16,384,384),"ori_format": "FRACTAL_NZ"},
                    {"shape": (24,16,24,24,16,16), "dtype": "uint8", "format": "FRACTAL_NZ", "ori_shape": (24,16,384,384),"ori_format": "FRACTAL_NZ"},
                    {"shape": (24,16,24,24,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24,16,384,384),"ori_format": "FRACTAL_NZ"},
                    {"shape": (24,16,24,24,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24,16,384,384),"ori_format": "FRACTAL_NZ"},0.5,[-1]],
         "case_name": "softmaxdomask_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A",], case1)
ut_case.add_case(["Ascend910A",], case2)
