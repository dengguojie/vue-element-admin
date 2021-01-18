#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ResizeNearestNeighborV2D", None, None)

case1 = {"params": [{"shape": (34,2,1,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW"},
                    {"shape": (34,2,1,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW"},
                    (1, 1), False, False],
         "case_name": "resize_nearest_neighbor_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (34,2,300,200,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,300,200,16),"ori_format": "NCHW"},
                    {"shape": (34,2,300,200,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,300,200,16),"ori_format": "NCHW"},
                    (300, 200), False, True],
         "case_name": "resize_nearest_neighbor_v2_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (34,2,31,51,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,31,51,16),"ori_format": "NCHW"},
                    {"shape": (34,2,31,51,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,31,51,16),"ori_format": "NCHW"},
                    (31, 51), True, False],
         "case_name": "resize_nearest_neighbor_v2_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (34,2,1,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW"},
                    {"shape": (34,2,1,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW"},
                    (300, 300), False, False],
         "case_name": "resize_nearest_neighbor_v2_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (34,2,1,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW"},
                    {"shape": (34,2,1,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW"},
                    (31, 51), True, False],
         "case_name": "resize_nearest_neighbor_v2_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (34,2,1,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW"},
                    {"shape": (34,2,1,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW"},
                    (31, 51), False, True],
         "case_name": "resize_nearest_neighbor_v2_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (1088,188,3,72,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1088,3008,3,72),"ori_format": "NCHW"},
                    {"shape": (1088,188,2,2,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1088,3008,2,2),"ori_format": "NCHW"},
                    (2, 2), False, False],
         "case_name": "resize_nearest_neighbor_v2_d_7",
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

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)
