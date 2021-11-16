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
                    10, 10, 0.5, -0.5],
         "case_name": "roi_pooling_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (1,2,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NCHW"},
                    {"shape": (16, 5), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 5),"ori_format": "NCHW"},
                    None,
                    {"shape": (16,2,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,4,16,16),"ori_format": "NCHW"},
                    16, 16, 0.5, 0.5],
         "case_name": "roi_pooling_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    {"shape": (32,5), "dtype": "float16", "format": "NCHW", "ori_shape": (32,5),"ori_format": "NCHW"},
                    None,
                    {"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    16, 16, 0.5, 0.5],
         "case_name": "roi_pooling_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    {"shape": (32,5), "dtype": "float16", "format": "NCHW", "ori_shape": (32,5),"ori_format": "NCHW"},
                    None,
                    {"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    16, 16, 0.5, -0.5],
         "case_name": "roi_pooling_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    {"shape": (32,5), "dtype": "float16", "format": "NCHW", "ori_shape": (32,5),"ori_format": "NCHW"},
                    None,
                    {"shape": (32,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,4,16,16),"ori_format": "NCHW"},
                    16, 16, 0.5, -0.5],
         "case_name": "roi_pooling_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}  
case8 = {"params": [{"shape": (1,8,124,124,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,128,124,124),"ori_format": "NCHW"},
                    {"shape": (16,5), "dtype": "float16", "format": "NCHW", "ori_shape": (16,5),"ori_format": "NCHW"},
                    None,
                    {"shape": (16,80,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,80,5,5,16),"ori_format": "NCHW"},
                    5, 5, 0.5, -0.5],
         "case_name": "roi_pooling_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}  
case9 = {"params": [{"shape": (1,4,34,34,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,64,34,34),"ori_format": "NCHW"},
                    {"shape": (12,5), "dtype": "float16", "format": "NCHW", "ori_shape": (12,5),"ori_format": "NCHW"},
                    None,
                    {"shape": (12,4,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (12,64,5,5),"ori_format": "NCHW"},
                    5, 5, 0.5, -0.5],
         "case_name": "roi_pooling_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}  
case10 = {"params": [{"shape": (32,9,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,9,4,16,16),"ori_format": "NCHW"},
                    {"shape": (32,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,2,16),"ori_format": "NCHW"},
                    {"shape": (32,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,2,16),"ori_format": "NCHW"},
                    {"shape": (32,9,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,9,4,16,16),"ori_format": "NCHW"},
                    16, 16, 0.5, -0.5],
         "case_name": "roi_pooling_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case11 = {"params": [{"shape": (1,9,84,84,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,129,84,84),"ori_format": "NCHW"},
                    {"shape": (12,5), "dtype": "float16", "format": "NCHW", "ori_shape": (12,5),"ori_format": "NCHW"},
                    None,
                    {"shape": (12,9,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (12, 129,5,5),"ori_format": "NCHW"},
                    5, 5, 0.5, -0.5],
         "case_name": "roi_pooling_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True} 

case12 = {"params": [{"shape": (32,9,32,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,9,4,16,16),"ori_format": "NCHW"},
                    {"shape": (32,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,2,16),"ori_format": "NCHW"},
                    {"shape": (32,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,2,16),"ori_format": "NCHW"},
                    {"shape": (32,9,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,9,4,16,16),"ori_format": "NCHW"},
                    40, 30, 0.5, -0.5],
         "case_name": "roi_pooling_12",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
       
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case12)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)