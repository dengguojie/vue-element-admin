#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Bias", "impl.dynamic.bias", "bias")

case1 = {
    "params": [
        {"shape": (-1,-1,-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4,5),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None),(1,None)]},
        {"shape": (-1,-1,-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4,5),"ori_format": "ND", "range":[(1,None), (2,None), (3,None),(4,None),(5,None)]},
        {"shape": (-1,-1,-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4,5),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None),(1,None)]},
        0,
        5
    ],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1,-1,-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4,5),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None),(1,None)]},
        {"shape": (-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (4,5),"ori_format": "ND", "range":[(4,None),(5,None)]},
        {"shape": (-1,-1,-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4,5),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None),(1,None)]},
        3,
        2
    ],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case3 = {"params": [{"shape": (-1,-1,-1,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(4,4)]},
                    {"shape": (1,2,3,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4),"ori_format": "ND", "range":[(1,1), (2,2), (3,3),(4,4)]},
                    {"shape": (1,2,3,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    0, 5
                    ],
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (2,3,4),"ori_format": "ND", "range":[(2,None),(3,None),(4,None)]},
                    {"shape": (4,), "dtype": "float32", "format": "ND", "ori_shape": (4,),"ori_format": "ND", "range":[(4,4)]},
                    {"shape": (2,3,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4,5),"ori_format": "ND", "range":[(2,None),(3,None),(4,None)]},
                    2, 1
                    ],
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {
    "params": [
        {"shape": (-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (2,3,4),"ori_format": "ND", "range":[(2,None),(3,None),(4,None)]},
        {"shape": (4,), "dtype": "float32", "format": "ND", "ori_shape": (4,),"ori_format": "ND", "range":[(4,4)]},
        {"shape": (2,3,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,3,4,5),"ori_format": "ND", "range":[(2,None),(3,None),(4,None)]},
        -1, -1
    ],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case6 = {
    "params": [
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND", "range":[(2,None)]},
        {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1,1)]},
        {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND", "range":[(2,None)]},
        0,
        0,
        False
    ],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)

def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.dynamic.bias import op_select_format
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     "test_bias_op_select_format_1")
                     
ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
