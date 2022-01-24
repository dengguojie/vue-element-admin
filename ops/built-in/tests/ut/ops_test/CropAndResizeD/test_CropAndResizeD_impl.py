#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("CropAndResizeD", "impl.crop_and_resize", "crop_and_resize")

case1 = {"params": [{"shape": (1,32,16,16,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,16,16,16*32),"ori_format": "NHWC"},
                    {"shape": (10, 4), "dtype": "float32", "format": "ND", "ori_shape": (10,4),"ori_format": "ND"},
                    {"shape": (10,), "dtype": "int32", "format": "ND", "ori_shape": (10,),"ori_format": "ND"},
                    {"shape": (10,32,14,14,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (10,16,16,16*32),"ori_format": "NHWC"},
                    (14, 14), 0.0, "bilinear"],
         "case_name": "bilinear_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1,32,16,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,16,16*32),"ori_format": "NHWC"},
                    {"shape": (10, 4), "dtype": "float16", "format": "ND", "ori_shape": (10,4),"ori_format": "ND"},
                    {"shape": (10,), "dtype": "int32", "format": "ND", "ori_shape": (10,),"ori_format": "ND"},
                    {"shape": (10,32,14,14,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10,16,16,16*32),"ori_format": "NHWC"},
                    (14, 14), 0.0, "bilinear"],
         "case_name": "bilinear_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend910"], case2)
