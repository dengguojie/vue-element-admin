#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np


ut_case = OpUT("TransData", "impl.trans_data",
               "trans_data")

case1 = {"params": [{"shape": (36, 2, 16, 106), "dtype": "float16",
                     "ori_shape": (36, 2, 16, 106), "format": "NCHW",
                     "ori_format": "NCHW"},
                    {"shape": (1696, 3, 16, 16), "dtype": "float16",
                     "ori_shape": (1696, 3, 16, 16), "format": "FRACTAL_Z_C04", "ori_format": "NCHW"},
                    "NCHW", "FRACTAL_Z_C04"],
         "expect": "success",
         "format_expect": ["FRACTAL_Z_C04"],
         "support_expect": False}


ut_case.add_case(["Ascend310", "Ascend910A"], case1)
