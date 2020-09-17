#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("LayerNorm", None, None)

case1 = {"params": [{"shape": (64,128,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1023),"ori_format": "NCHW"},
                    {"shape": (1023,), "dtype": "float16", "format": "NCHW", "ori_shape": (1023,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1023),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    1, 2],
         "case_name": "layer_norm_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (64,128,1024), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1024), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    1, 2],
         "case_name": "layer_norm_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)



if __name__ == '__main__':
    ut_case.run("Ascend910")
