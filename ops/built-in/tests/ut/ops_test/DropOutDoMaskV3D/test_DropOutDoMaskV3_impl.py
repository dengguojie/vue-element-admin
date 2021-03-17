#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("DropOutDoMaskV3D", None, None)

case1 = {"params": [{"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128), "ori_format": "ND"},
                    0.1],
         "case_name": "drop_out_do_mask_v3_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (8, 8, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 128),
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (16384,), "dtype": "uint8", "format": "FRACTAL_NZ", "ori_shape": (16384,),
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (8, 8, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 128),
                     "ori_format": "FRACTAL_NZ"},
                    0.1],
         "case_name": "drop_out_do_mask_v3_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
