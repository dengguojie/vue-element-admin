#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("BninferenceD", None, None)

case1 = {"params": [{"shape": (1,16,10,10), "dtype": "float16", "format": "NCHW", "ori_shape": (1,16,10,10),"ori_format": "NCHW"},
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None, None,
                    {"shape": (1,16,10,10), "dtype": "float16", "format": "NCHW", "ori_shape": (1,16,10,10),"ori_format": "NCHW"},
                    0.999, 0.001, True, 1],
         "case_name": "bn_inference_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None, None,
                    {"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                    0.999, 0.001, True, 1],
         "case_name": "bn_inference_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1,16,1,1), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,16,1,1),"ori_format": "NC1HWC0"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None, None,
                    {"shape": (1,16,1,1), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,16,1,1),"ori_format": "NC1HWC0"},
                    0.999, 0.001, True, 1],
         "case_name": "bn_inference_d_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (1,16,10,10), "dtype": "float16", "format": "NCHW", "ori_shape": (1,16,10,10),"ori_format": "NCHW"},
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    {"shape": (16,1), "dtype": "float16", "format": "ND", "ori_shape": (16,1),"ori_format": "ND"},
                    None,
                    {"shape": (1,16,10,10), "dtype": "float16", "format": "NCHW", "ori_shape": (1,16,10,10),"ori_format": "NCHW"},
                    0.999, 0.001, True, 1],
         "case_name": "bn_inference_d_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310"], case4)

if __name__ == '__main__':
    ut_case.run()
    exit(0)

