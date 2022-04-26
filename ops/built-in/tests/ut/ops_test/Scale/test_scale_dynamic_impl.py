#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Scale", "impl.dynamic.scale", "scale")

case1 = {
        "params": [{
        "shape": (2, 2, 3),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1, 2, 3),
        "ori_format": "ND",
        "range": [(1, 10), (1, 10), (1, 10)]
    }, {
        "shape": (2, 2, 3),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1, 2, 3),
        "ori_format": "ND",
        "range": [(1, 10), (1, 10), (1, 10)]
    }, {
        "shape": (2, 2, 3),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1, 2, 3),
        "ori_format": "ND",
        "range": [(1, 10), (1, 10), (1, 10)]
    }, {
        "shape": (-1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND",
        "range": [(1, 10)]
    }, 0, 3, True],
    "case_name": "Scale_1",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)

def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.dynamic.scale import op_select_format
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     "test_scale_op_select_format_1")

ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
