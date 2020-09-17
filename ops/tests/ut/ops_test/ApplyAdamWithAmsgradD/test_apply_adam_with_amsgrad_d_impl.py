#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyAdamWithAmsgradD", None, None)

# ============ auto gen  test cases start ===============
ut_case.add_case(["Ascend910", "Ascend310"], {"params": [
    {"shape": (16, 32), "format": "NC1HWC0", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "NC1HWC0",},
    {"shape": (16, 32), "format": "NC1HWC0", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "NC1HWC0",},
    {"shape": (16, 32), "format": "NC1HWC0", "dtype": "float32", 
     "ori_shape": (16, 32), "ori_format": "NC1HWC0",},
    {"shape": (16, 32), "format": "NC1HWC0", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "NC1HWC0",},
    {"shape": (1,), "format": "ND", "dtype": "float32",
     "ori_shape": (1,), "ori_format": "ND",},
    {"shape": (1,), "format": "ND", "dtype": "float32",
     "ori_shape": (1,), "ori_format": "ND",},
    {"shape": (1,), "format": "ND", "dtype": "float32",
     "ori_shape": (1,), "ori_format": "ND",},
    {"shape": (16, 32), "format": "NC1HWC0", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "NC1HWC0",},
    {"shape": (16, 32), "format": "NC1HWC0", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "NC1HWC0",},
    {"shape": (16, 32), "format": "NC1HWC0", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "NC1HWC0",},
    {"shape": (16, 32), "format": "NC1HWC0", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "NC1HWC0",},
    {"shape": (16, 32), "format": "NC1HWC0", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "NC1HWC0",},
    0.9,
    0.999,
    1.0,
    False,
    ],
    "expect":
        "success"})

# ============ auto gen test cases end =================

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
