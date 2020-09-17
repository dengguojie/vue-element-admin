#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ExtractImagePatches", None, None)

case1 = {"params": [{"shape": (1,2,4,1), "dtype": "float16", "format":"NCHW","ori_shape": (1,2,4,1), "ori_format": "ND"},
                    {"shape": (1,2,4,1), "dtype": "float16", "format":"NHWC","ori_shape": (1,2,4,1), "ori_format": "ND"},
                    (1,2,2,1), (1,3,3,1), (1,3,3,1), "SAME"],
         "case_name": "extract_image_patches_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,10,1), "dtype": "float16", "format":"NCHW","ori_shape": (1,2,10,1), "ori_format": "ND"},
                    {"shape": (1,2,10,1), "dtype": "float16", "format":"NHWC","ori_shape": (1,2,10,1), "ori_format": "ND"},
                    (1,4,4,1), (1,3,3,1), (1,3,3,1), "SAME"],
         "case_name": "extract_image_patches_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)