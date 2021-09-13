#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf

ut_case = OpUT("ResizeBilinearV2D", None, None)

case1 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (2, 2), False, False],
         "case_name": "resize_bilinear_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 2), False, True],
         "case_name": "resize_bilinear_v2_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (25, 2), False, False],
         "case_name": "resize_bilinear_v2_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    (2, 2), False, False],
         "case_name": "resize_bilinear_v2_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (5,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (5,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (5,3,10,10,16), "dtype": "float32", "format": "NHWC", "ori_shape": (5,3,10,10,16),"ori_format": "NHWC"},
                    (10, 10), False, True],
         "case_name": "resize_bilinear_v2_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (2,260,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), False, False],
         "case_name": "resize_bilinear_v2_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
       
case8 = {"params": [{"shape": (2,260,1,1,16), "dtype": "int64", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), False, False],
         "case_name": "resize_bilinear_v2_d_8",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (2,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), False, False],
         "case_name": "resize_bilinear_v2_d_9",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (2,1,1,16,10), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10, 1), False, False],
         "case_name": "resize_bilinear_v2_d_10",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case11 = {"params": [{"shape": (2,260,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (3000, 10), False, False],
         "case_name": "resize_bilinear_v2_d_11",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case12 = {"params": [{"shape": (2,260,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), True, True],
         "case_name": "resize_bilinear_v2_d_12",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case13 = {"params": [{"shape": (16, 128, 1, 1, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), False, False],
         "case_name": "resize_bilinear_v2_d_13",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}    #check_supported_tik
case14 = {"params": [{"shape": (2,260,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), False, False],
         "case_name": "resize_bilinear_v2_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}    #entre resize_bilinear_v2_d_compute  start  line 6811
case15 = {"params": [{"shape": (25, 128, 1, 3, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_15",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case32 = {"params": [{"shape": (25, 128, 2, 2, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_32",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}     #cover line 300-306 
case33 = {"params": [{"shape": (25, 128, 2, 2, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_33",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}      
case16 = {"params": [{"shape": (25, 128, 2, 2, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_16",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}     #line253 first loop start  (share_in[-2] = shape_out[-2] && share_in[-3] = shape_out[-3])
case17 = {"params": [{"shape": (25, 128, 2, 2, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_17",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case18 = {"params": [{"shape": (25, 128, 1, 1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 300, 100, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_18",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}   # entre : first loop branch 2   line  360         second loop start!
case19 = {"params": [{"shape": (25, 128, 1, 1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 32, 2, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_19",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case20 = {"params": [{"shape": (150, 512, 1, 1, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 32, 2, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 50), True, False],
         "case_name": "resize_bilinear_v2_d_20",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}     #entre : second loop  line 417    third loop start!
case21 = {"params": [{"shape": (150, 512, 1, 1, 35), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 512, 10, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (5, 10), True, False],
         "case_name": "resize_bilinear_v2_d_21",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True} 
case22 = {"params": [{"shape": (150, 512, 1, 1, 35), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 1000, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (2, 10), True, False],
         "case_name": "resize_bilinear_v2_d_22",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True} 
case30 = {"params": [{"shape": (5, 168, 1, 1, 2), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 888, 10, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (10, 15), True, False],
         "case_name": "resize_bilinear_v2_d_30",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case23 = {"params": [{"shape": (25, 512, 1, 1, 35), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 100, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (3, 10), True, False],
         "case_name": "resize_bilinear_v2_d_23",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True} 
case24 = {"params": [{"shape": (25, 512, 1, 1, 35), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 10, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (3, 10), True, False],
         "case_name": "resize_bilinear_v2_d_24",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}   #line 757
case25 = {"params": [{"shape": (25, 512, 1, 1, 35), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 9.5, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_25",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case26 = {"params": [{"shape": (25, 512, 1, 1, 35), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 10, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_26",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case27 = {"params": [{"shape": (160, 55, 1, 1, 30), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 10, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_27",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}    #line 936 else
case28 = {"params": [{"shape": (25, 512, 1, 1, 35), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 200, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_28",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case29 = {"params": [{"shape": (25, 512, 1, 1, 35), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 1000, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_29",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}    #watch out case30 case32 case33 above
case31 = {"params": [{"shape": (25, 512, 1, 1, 35), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 10), True, False],
         "case_name": "resize_bilinear_v2_d_31",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case34 = {"params": [{"shape": (10, 2025, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (100, 15), True, False],
         "case_name": "resize_bilinear_v2_d_34",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case35 = {"params": [{"shape": (10, 2025, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (5, 15), True, False],
         "case_name": "resize_bilinear_v2_d_35",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case36 = {"params": [{"shape": (10, 2025, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (4, 7), True, False],
         "case_name": "resize_bilinear_v2_d_36",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case37 = {"params": [{"shape": (10, 50, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (6, 100), True, False],
         "case_name": "resize_bilinear_v2_d_37",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}   #kittle size start line 1907 
case38 = {"params": [{"shape": (10, 50, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (10, 300), True, False],
         "case_name": "resize_bilinear_v2_d_38",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True} 
case39 = {"params": [{"shape": (10, 50, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (9, 2), True, False],
         "case_name": "resize_bilinear_v2_d_39",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}

case40 = {"params": [{"shape": (10, 2025, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (3, 1000), True, False],
         "case_name": "resize_bilinear_v2_d_40",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}     #f32 line 2404 start
case41 = {"params": [{"shape": (10, 2025, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (3, 10), True, False],
         "case_name": "resize_bilinear_v2_d_41",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case42 = {"params": [{"shape": (10, 2025, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (3, 5), True, False],
         "case_name": "resize_bilinear_v2_d_42",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case43 = {"params": [{"shape": (10, 820, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (3, 5), True, False],
         "case_name": "resize_bilinear_v2_d_43",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case44 = {"params": [{"shape": (10, 40, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (3, 1000), True, False],
         "case_name": "resize_bilinear_v2_d_44",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case45 = {"params": [{"shape": (10, 50, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 50, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (3, 5), True, False],
         "case_name": "resize_bilinear_v2_d_45",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case46 = {"params": [{"shape": (10, 50, 3, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), False, True],
         "case_name": "resize_bilinear_v2_d_46",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}                # No.3 situation : output H/W = (1,1)   line 3612 start
case47 = {"params": [{"shape": (10, 50, 3, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (10, 50, 3, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_47",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case48 = {"params": [{"shape": (10, 50, 999, 999, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (10, 2025, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_48",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case49 = {"params": [{"shape": (10, 2025, 3, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (45, 500, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_49",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case50 = {"params": [{"shape": (10, 50, 655, 111, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (10, 50, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_50",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case51 = {"params": [{"shape": (10, 2025, 3, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (10, 50, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_51",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case52 = {"params": [{"shape": (10, 50, 655, 111, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (100, 260, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_52",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case53 = {"params": [{"shape": (10, 2025, 3, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (100, 132, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_53",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case54 = {"params": [{"shape": (10, 1, 655, 111, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (10, 13, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_54",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case55 = {"params": [{"shape": (10, 2025, 3, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (10, 13, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_55",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}
case56 = {"params": [{"shape": (10, 2025, 3, 3, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 13, 1, 16),"ori_format": "NHWC"},
                    {"shape": (10, 13, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2, 16),"ori_format": "NHWC"},
                    (5, 5), True, False],
         "case_name": "resize_bilinear_v2_d_56",
         "expect": "success",
         "format_expect": [], 
         "support_expect": True}



def test_check_supported_info(test_arg): 
    from impl.resize_bilinear_v2_d import check_supported
    check_supported({"shape": (1, 1, 1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1), "ori_format": "NHWC"},
                    {"shape": (1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1), "ori_format": "NHWC"},
                    [0, 3, 4, 1, 2],
                    "check_supported_1")
    check_supported({"shape": (1, 1, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1), "ori_format": "NCHW"},
                    {"shape": (1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1), "ori_format": "NCHW"},
                    [0, 3, 4, 1, 2],
                    "check_supported_2")
    check_supported({"shape": (1, 1, 1, 1), "dtype": "float16", "format": "NHW", "ori_shape": (1, 1), "ori_format": "NHW"},
                    {"shape": (1, 1), "dtype": "float16", "format": "NHW", "ori_shape": (1, 1), "ori_format": "NHW"},
                    [0, 3, 4, 1, 2],
                    "check_supported_3")
    check_supported({"shape": (1, 1, 1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1), "ori_format": "NHWC"},
                    {"shape": (1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1), "ori_format": "NHWC"},
                    [3000, 3, 4, 1, 2],
                    "check_supported_4")
    check_supported({"shape": (1, 1, 1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1), "ori_format": "NHWC"},
                    {"shape": (1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1), "ori_format": "NHWC"},
                    [-10, 3, 4, 1, 2],
                    "check_supported_5")
ut_case.add_cust_test_func(test_func=test_check_supported_info)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case13)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case14)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case15)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case16)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case17)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case18)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case19)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case20)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case21)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case22)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case23)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case24)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case26)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case27)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case28)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case29)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case30)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case31)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case32)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case33)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case34)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case35)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case36)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case37)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case38)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case39)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case40)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case41)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case42)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case43)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case44)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case45)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case46)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case47)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case48)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case49)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case50)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case51)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case52)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case53)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case54)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case55)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case56)



def trans_data_to_tf(data_nchwc0):
    out_size = data_nchwc0.shape
    nhwc = np.zeros((out_size[0],out_size[-3], out_size[-2], out_size[1]*out_size[-1]), dtype=data_nchwc0.dtype)

    for i in range(out_size[0]):
        for j in range(out_size[1]):
            for k in range(out_size[-1]):
                for h in range(out_size[-3]):
                    for w in range(out_size[-2]):
                        nhwc[i][h][w][j*out_size[-1] + k] = data_nchwc0[i][j][h][w][k]

    return nhwc
def trans_tf_data_out(data_nhwc):
    in_size = data_nhwc.shape
    if data_nhwc.dtype == "float16":
        c0 = 16
    else:
        c0 = 16

    nchwc0  = np.zeros((in_size[0], in_size[-1] // c0, in_size[-3], in_size[-2], c0), dtype=data_nhwc.dtype)

    for i in range(in_size[0]):
        for j in range(in_size[-1] // c0):
            for k in range(c0):
                for h in range(in_size[-3]):
                    for w in range(in_size[-2]):
                        nchwc0[i][j][h][w][k] = data_nhwc[i][h][w][j*c0+k]
    return nchwc0

def calc_expect_func(image, out, size):
    data_in = trans_data_to_tf(image['value'])
    x =  tf.placeholder(image['dtype'], shape=data_in.shape)
    y =  tf.placeholder("int32", shape=(2,))
    z =  tf.image.resize_bilinear(x, y, False)
    
    with tf.Session() as sess:
        res = sess.run(z, feed_dict={x: data_in, y: np.array(size)})
    res = trans_tf_data_out(res)
    return res

ut_case.add_precision_case("all", {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "ND", "ori_shape": (2,3,1,1,16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2,3,2,2,16), "dtype": "float32", "format": "ND", "ori_shape": (2,3,2,2,16),"ori_format": "ND", "param_type": "output"},
                                              (2,2)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })



