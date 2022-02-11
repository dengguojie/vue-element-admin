#!/usr/bin/env/ python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from unittest.mock import MagicMock
from unittest.mock import patch

# pylint: disable=invalid-name
ut_case = OpUT("PRelu", "impl.dynamic.prelu", "prelu")


case1 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 4, 4),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]},
    {"shape": (-1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,),
     "ori_format": "NHWC", "range": [(1, 1)]},
    {"shape": (1, -1, -1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 4, 4),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]}],
         "case_name": "prelu_dynamic_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 4, 4),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]},
    {"shape": (-1, -1, -1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 4, 4),
     "ori_format": "NHWC", "range": [(1, 1), (1, 10), (1, 10)]},
    {"shape": (1, -1, -1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 4, 4),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]}],
          "case_name": "prelu_dynamic_2",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}

'''
case3 = {"params": [
    {"shape": (-1, 2, 6, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (64, 2, 4, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
    {"shape": (-1, 2, 6, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1, 2, 1, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
    {"shape": (-1, 2, 6, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (64, 2, 4, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]}],
         "case_name": "prelu_dynamic_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
'''
case4 = {"params": [
    {"shape": (-1, 2, -1, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (64, 2, 4, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
    {"shape": (-1,), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,),
     "ori_format": "NDC1HWC0", "range": [(1, 1)]},
    {"shape": (-1, 2, -1, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (64, 2, 4, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]}],
         "case_name": "prelu_dynamic_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [
    {"shape": (6, 1, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (6, 19, 20),
     "ori_format": "ND", "range": [(6, 6), (1, 1), (1, 1), (16, 16), (16, 16)]},
    {"shape": (1, 1, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 19, 20),
     "ori_format": "ND", "range": [(1, 1), (1, 1), (1, 1), (16, 16), (16, 16)]},
    {"shape": (6, 1, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 19, 20),
     "ori_format": "ND", "range": [(6, 6), (1, 1), (1, 1), (16, 16), (16, 16)]}],
         "case_name": "prelu_dynamic_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [
    {"shape": (32, 1, 4, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 1, 4, 16),
     "ori_format": "NHWC", "range": [(32, 32), (1, 1), (4, 4), (16, 16)]},
    {"shape": (1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16),
     "ori_format": "NHWC", "range": [(1, 1), (16, 16)]},
    {"shape": (32, 1, 4, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 1, 4, 16),
     "ori_format": "NHWC", "range": [(32, 32), (1, 1), (4, 4), (16, 16)]}],
         "case_name": "prelu_dynamic_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [
    {"shape": (3, 2, 4, 5, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (3, 4, 5, 19),
     "ori_format": "NHWC", "range": [(3, 3), (2, 2), (4, 4), (5, 5), (16, 16)]},
    {"shape": (2, 4, 5, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 1, 1, 16),
     "ori_format": "NHWC", "range": [(3, 3), (2, 2), (4, 4), (5, 5), (16, 16)]},
    {"shape": (3, 2, 4, 5, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 1, 5, 19),
     "ori_format": "NHWC", "range": [(3, 3), (2, 2), (4, 4), (5, 5), (16, 16)]}],
         "case_name": "prelu_dynamic_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [
    {"shape": (3, 2, 4, 5, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (3, 4, 5, 19),
     "ori_format": "NDHWC", "range": [(3, 3), (2, 2), (4, 4), (5, 5), (16, 16)]},
    {"shape": (2, 4, 5, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (2, 1, 1, 16),
     "ori_format": "NDHWC", "range": [(3, 3), (2, 2), (4, 4), (5, 5), (16, 16)]},
    {"shape": (3, 2, 4, 5, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (2, 1, 5, 19),
     "ori_format": "NDHWC", "range": [(3, 3), (2, 2), (4, 4), (5, 5), (16, 16)]}],
         "case_name": "prelu_dynamic_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [
    {"shape": (3, 2, 4, 5, 16), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 5, 19),
     "ori_format": "ND", "range": [(3, 3), (2, 2), (4, 4), (5, 5), (16, 16)]},
    {"shape": (4, 5, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 16),
     "ori_format": "ND", "range": [(3, 3), (2, 2), (4, 4), (5, 5), (16, 16)]},
    {"shape": (3, 2, 4, 5, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 5, 19),
     "ori_format": "ND", "range": [(3, 3), (2, 2), (4, 4), (5, 5), (16, 16)]}],
         "case_name": "prelu_dynamic_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


def test_op_select_format(test_arg):
    from impl.dynamic.prelu import op_select_format
    op_select_format(
        {"shape": (1, 1, 1, 1, 1, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1, 3, 4, 5, 19),
         "ori_format": "NDHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
        {"shape": (1, 1, 1, 1, 1, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1, 2, 1, 5, 19),
         "ori_format": "NDHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
        {"shape": (1, 1, 1, 1, 1, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1, 1, 1, 1, 1, 16),
         "ori_format": "NDHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]}
        )
    op_select_format(
        {"shape": (1, 1, 1, 1, 1, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 3, 4, 5, 19),
         "ori_format": "ND", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
        {"shape": (1, 1, 1, 1, 1, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 2, 1, 5, 19),
         "ori_format": "ND", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
        {"shape": (1, 1, 1, 1, 1, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1, 1, 1, 1, 16),
         "ori_format": "ND", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]}
        )
    op_select_format(
        {"shape": (3, 2, 4, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": ( 3, 4, 5, 19),
         "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2)]},
        {"shape": (3, 2, 4, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": ( 2, 1, 5, 19),
         "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2)]},
        {"shape": (3, 2, 4, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 1, 16),
         "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2)]}
        )
    op_select_format(
        {"shape": (3, 2, 4, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": ( 3, 4, 5, 19),
         "ori_format": "NCHW", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2)]},
        {"shape": (1, 16, 1, 1, 1), "dtype": "float16", "format": "NC1HWC0", "ori_shape": ( 1,),
         "ori_format": "NCHW", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2)]},
        {"shape": (3, 2, 4, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 16, 1, 1, 1),
         "ori_format": "NCHW", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2)]}
        )
    op_select_format(
        {"shape": (32,32), "dtype": "float16", "format": "ND", "ori_shape": (32,32),"ori_format": "ND"},
        {"shape": (32,), "dtype": "float16", "format": "NCHW", "ori_shape": (32,),"ori_format": "NCHW"},
        {"shape": (32,32), "dtype": "float16", "format": "ND", "ori_shape": (32,32),"ori_format": "ND"}
        )


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_cust_test_func(test_func=test_op_select_format)

def side_effects(*args):
    # return vals[args]
    return True

def test_v220_mock(test_arg):
    with patch("te.platform.api_check_support",MagicMock(side_effect=side_effects)):
        from impl.dynamic.prelu import op_select_format
        op_select_format({"shape": (32,32), "format": "NCHW", "dtype": "float16", "ori_shape": (32,32), "ori_format": "NCHW"},
                         {"shape": (32,), "format": "NCHW", "dtype": "float16", "ori_shape": (32,), "ori_format": "NCHW"},
                         {"shape": (32,32), "format": "NCHW", "dtype": "float16", "ori_shape": (32,32), "ori_format": "NCHW"})
ut_case.add_cust_test_func(test_func=test_v220_mock)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
