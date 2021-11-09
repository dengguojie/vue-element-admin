#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.ascend_requant_s16 import get_op_support_info
ut_case = OpUT("AscendRequantS16", None, None)

case1 = {"params": [{"shape": (1,1,1,1,16), "dtype": "int16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    None,
                    {"shape": (1,1,1,1,32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1,1,1,1,32),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "int16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    True],
         "case_name": "ascend_requants16_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,4,4,16), "dtype": "int16", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (32,),"ori_format": "ND"},
                    {"shape": (1,2,1,1,16), "dtype": "int16", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,4,4,32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1,1,4,4,32),"ori_format": "NC1HWC0"},
                    None,
                    False],
         "case_name": "ascend_requants16_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

def test_get_op_support_info_001(test_arg):
    """
    test_get_op_support_info_001
    """
    get_op_support_info(
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "FRACTAL_NZ",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        }, None, None, None, None, False, False)

def test_get_op_support_info_002(test_arg):
    """
    test_get_op_support_info_002
    """
    get_op_support_info(
        {
            "shape": (3, 5, 5, 8, 16, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        }, None, 1, None, None, True, False)

def test_get_op_support_info_003(test_arg):
    """
    test_get_op_support_info_003
    """
    get_op_support_info(
        {
            "shape": (3, 5, 5, 8, 16, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        }, None, 1, None, None, False, False)

def test_get_op_support_info_004(test_arg):
    """
    test_get_op_support_info_004
    """
    get_op_support_info(
        {
            "shape": (3, 5, 5, 8, 16, 16),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        }, None, 1, None, None, False, False)

ut_case.add_cust_test_func(test_func=test_get_op_support_info_001)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_002)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_003)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_004)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)


