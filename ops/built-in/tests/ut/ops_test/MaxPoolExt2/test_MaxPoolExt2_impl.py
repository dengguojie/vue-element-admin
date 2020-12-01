#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("MaxPoolExt2", None, None)

case1 = {"params": [{"shape": (1,3,35,49,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 35, 49, 16),"ori_format": "NHWC"},
                    {"shape": (1,3,17,24,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 17, 24, 16),"ori_format": "NHWC"},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID"],
         "case_name": "max_pool_ext2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,4,23,111,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 23, 111, 16),"ori_format": "NHWC"},
                    {"shape": (1,4,11,55,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 11, 55, 16),"ori_format": "NHWC"},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID"],
         "case_name": "max_pool_ext2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,4,23,111,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 23, 111, 16),"ori_format": "NHWC",
                     "addr_type": 1, "L1_fusion_type": 0},
                    {"shape": (1,4,11,55,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 11, 55, 16),"ori_format": "NHWC",
                     "addr_type": 1},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID"],
         "case_name": "max_pool_ext2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,4,23,111,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 23, 111, 16),"ori_format": "NHWC",
                     "L1_fusion_type": 0},
                    {"shape": (1,4,11,55,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 11, 55, 16),"ori_format": "NHWC",
                     "addr_type": 1},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID"],
         "case_name": "max_pool_ext2_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

def test_get_op_support_info(test_arg):
    from impl.max_pool_ext2 import get_op_support_info
    get_op_support_info({"shape": (1,4,23,111,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 23, 111, 16),"ori_format": "NHWC",
                     "L1_fusion_type": 0},
                    {"shape": (1,4,11,55,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 11, 55, 16),"ori_format": "NHWC",
                     "addr_type": 1},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID", "NHWC")
    get_op_support_info({"shape": (1,4,23,111,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 4, 23, 111, 16),"ori_format": "NHWC",
                        "L1_fusion_type": 0},
                        {"shape": (1,4,11,55,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 4, 11, 55, 16),"ori_format": "NHWC",
                        "addr_type": 1},
                        [1, 1, 3, 3],
                        [1, 1, 2, 2],
                        "VALID", "NC1HWC0")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)


if __name__ == '__main__':
    ut_case.run("Ascend910")
