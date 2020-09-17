#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ClipBoxesD", None, None)

case1 = {"params": [{"shape": (4000, 4), "dtype": "float16", "format": "ND", "ori_shape": (4000, 4),"ori_format": "ND"},
                    {"shape": (4000, 4), "dtype": "float16", "format": "ND", "ori_shape": (4000, 4),"ori_format": "ND"},
                    [1024, 728]],
         "case_name": "clip_boxes_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (6000, 4), "dtype": "float16", "format": "ND", "ori_shape": (6000, 4),"ori_format": "ND"},
                    {"shape": (6000, 4), "dtype": "float16", "format": "ND", "ori_shape": (6000, 4),"ori_format": "ND"},
                    [1024, 728]],
         "case_name": "clip_boxes_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (17600, 4), "dtype": "float16", "format": "ND", "ori_shape": (17600, 4),"ori_format": "ND"},
                    {"shape": (17600, 4), "dtype": "float16", "format": "ND", "ori_shape": (17600, 4),"ori_format": "ND"},
                    [1024, 728]],
         "case_name": "clip_boxes_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (4000, 4), "dtype": "float32", "format": "ND", "ori_shape": (4000, 4),"ori_format": "ND"},
                    {"shape": (4000, 4), "dtype": "float32", "format": "ND", "ori_shape": (4000, 4),"ori_format": "ND"},
                    [1024, 728]],
         "case_name": "clip_boxes_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (128,3), "dtype": "float16", "format": "ND", "ori_shape": (128,3),"ori_format": "ND"},
                    {"shape": (128,3), "dtype": "float16", "format": "ND", "ori_shape": (128,3),"ori_format": "ND"},
                    [128, 256]],
         "case_name": "clip_boxes_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)




