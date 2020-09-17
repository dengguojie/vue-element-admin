#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SpaceToBatchNdD", None, None)

case1 = {"params": [{"shape": (2,128,48,72,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 48, 72, 2048),"ori_format": "NHWC"},
                    {"shape": (288,128,6,8,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 6, 8, 2048),"ori_format": "NHWC"},
                    [12, 12],
                    [[12, 12], [12, 12]]],
         "case_name": "space_to_batch_nd_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,128,54,80,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 54, 80, 2048),"ori_format": "NHWC"},
                    {"shape": (338,128,6,8,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (338, 6, 8, 2048),"ori_format": "NHWC"},
                    [13, 13],
                    [[12, 12], [12, 12]]],
         "case_name": "space_to_batch_nd_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,128,516,72,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 516, 72, 2048),"ori_format": "NHWC"},
                    {"shape": (288,128,6,8,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 6, 8, 2048),"ori_format": "NHWC"},
                    [12, 12],
                    [[12, 12], [12, 12]]],
         "case_name": "space_to_batch_nd_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)


if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)
