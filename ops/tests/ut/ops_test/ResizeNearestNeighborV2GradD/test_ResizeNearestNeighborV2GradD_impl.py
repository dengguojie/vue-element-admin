#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ResizeNearestNeighborV2GradD", None, None)

case1 = {"params": [{"shape": (1,1,1920,2160, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,3,2,2),"ori_format": "NCHW"},
                    {"shape": (1,1,1920,2160), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"},
                    (1920,2160), True, False],
         "case_name": "resize_nearest_neighbor_v2_grad_d_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,1,4,3, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 4, 3, 2),"ori_format": "NCHW"},
                    {"shape": (1,1,3,2), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"},
                    (3,2), True, True],
         "case_name": "resize_nearest_neighbor_v2_grad_d_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,1,1920,2160, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,3,2,2),"ori_format": "NCHW"},
                    {"shape": (1,1,1920,2160, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"},
                    (1920,2160), True, False],
         "case_name": "resize_nearest_neighbor_v2_grad_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,1,256,256,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,3,2,2),"ori_format": "NCHW"},
                    {"shape": (1,1,128,128,16), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"},
                    (128,128), False, False],
         "case_name": "resize_nearest_neighbor_v2_grad_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)