#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
segment_sum
'''
from op_test_frame.ut import OpUT
from impl.dynamic.segment_sum import check_supported
from impl.dynamic.segment_sum import op_select_format


# 'pylint: disable=unused-argument,invalid-name
ut_case = OpUT("SegmentSum", "impl.dynamic.segment_sum", "segment_sum")

def test_check_supported(test_arg):
    """
    func test_check_supported
    """
    check_supported({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                    {"shape": (28, 16, 16), "dtype": "int32", "format": "ND",
                     "ori_shape": (28, 16, 16), "ori_format": "ND"},
                    {"shape": (200, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (200, 28, 16, 16), "ori_format": "NCHW"})

    check_supported({"shape": (25, 28, 16, 1), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (25, 28, 16, 1), "ori_format": "NCHW"},
                    {"shape": (25,), "dtype": "int32", "format": "ND",
                     "ori_shape": (25,), "ori_format": "ND"},
                    {"shape": (25, 28, 16, 1), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (25, 28, 16, 1), "ori_format": "NCHW"})

    check_supported({"shape": (25, 28, -1, -1), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (25, 28, 16, 16), "ori_format": "NCHW"},
                    {"shape": (25,), "dtype": "int32", "format": "ND",
                     "ori_shape": (25,), "ori_format": "ND"},
                    {"shape": (25, 28, -1, -1), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (25, 28, 16, 16), "ori_format": "NCHW"})

    check_supported({"shape": (25, 28), "dtype": "uint8", "format": "ND", "ori_shape": (25, 28), "ori_format": "ND"},
                    {"shape": (25,), "dtype": "int64", "format": "ND", "ori_shape": (25,), "ori_format": "ND"},
                    {"shape": (25, 28), "dtype": "uint8", "format": "ND", "ori_shape": (25, 28), "ori_format": "ND"},)

    check_supported({"shape": (25, 28), "dtype": "int32", "format": "ND", "ori_shape": (25, 28), "ori_format": "ND"},
                    {"shape": (25,), "dtype": "int64", "format": "ND", "ori_shape": (25,), "ori_format": "ND"},
                    {"shape": (25, 28), "dtype": "int32", "format": "ND", "ori_shape": (25, 28), "ori_format": "ND"})


def test_op_select_format(test_arg):
    """
    func test_op_select_format
    """
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (28, 16, 16), "dtype": "int32", "format": "ND",
                      "ori_shape": (28, 16, 16), "ori_format": "ND"},
                     {"shape": (200, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (200, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "int8", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (28, 16, 16), "dtype": "int32", "format": "ND",
                      "ori_shape": (28, 16, 16), "ori_format": "ND"},
                     {"shape": (200, 28, 16, 16), "dtype": "int8", "format": "NCHW",
                      "ori_shape": (200, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "uint8", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20,), "dtype": "int32", "format": "ND", "ori_shape": (20,), "ori_format": "ND"},
                     {"shape": (200, 28, 16, 16), "dtype": "uint8", "format": "NCHW",
                      "ori_shape": (200, 28, 16, 16), "ori_format": "NCHW"})

case1 = {"params": [{"shape": (-1, -1,), "dtype": "float16", "format": "ND", "ori_shape": (150, 150),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (150,),
                     "ori_format": "ND", "range": [((1, 1),)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (150,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    ],
         "case_name": "segment_sum1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (150, 150),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (150,),
                     "ori_format": "ND", "range": [((1, 1),)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (150,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    ],
         "case_name": "segment_sum2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (150, 150),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (150,),
                     "ori_format": "ND", "range": [((1, 1),)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (150,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    ],
         "case_name": "segment_sum3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)

ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_check_supported)
if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
