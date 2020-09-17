#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("RoiPooling", None, None)

case1 = {"params": [{"shape": (1,2,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NCHW"},
                    {"shape": (1,2,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,2,16),"ori_format": "NCHW"},
                    {"shape": (1,2,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,2,16),"ori_format": "NCHW"},
                    {"shape": (1,2,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NCHW"},
                    1, 1, 0.5, 0.5],
         "case_name": "roi_pooling_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    {"shape": (32,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,2,16),"ori_format": "NCHW"},
                    {"shape": (32,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,2,16),"ori_format": "NCHW"},
                    {"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    1, 1, 0.5, 0.5],
         "case_name": "roi_pooling_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    {"shape": (32,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,2,16),"ori_format": "NCHW"},
                    {"shape": (32,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,2,16),"ori_format": "NCHW"},
                    {"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    1, 1, 0.5, -0.5],
         "case_name": "roi_pooling_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)