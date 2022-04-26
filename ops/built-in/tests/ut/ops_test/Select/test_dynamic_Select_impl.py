#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Select", "impl.dynamic.select", "select")

case1 = {"params": [
{"shape": (2, ), "dtype": "int8", "format": "ND", "ori_shape": (2, ),
     "ori_format": "ND", "range": [(2, 2)]},
    {"shape": (1, -1), "dtype": "int32", "format": "ND", "ori_shape": (1, 2),
     "ori_format": "ND", "range": [(1, 1), (1, 100)]},
    {"shape": (1, -1), "dtype": "int32", "format": "ND", "ori_shape": (1, 2),
     "ori_format": "ND", "range": [(1, 1), (1, 100)]},
    {"shape": (1, -1), "dtype": "int32", "format": "ND", "ori_shape": (1, 2),
     "ori_format": "ND", "range": [(1, 1), (1, 100)]},
],
    "case_name": "select_dynamic_1",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}

case2 = {"params": [
    {"shape": (-1, -1), "dtype": "int32", "format": "ND",
     "ori_shape": (-1, -1), "ori_format": "ND", "range": [(1, None),(1, None)]},
    {"shape": (-1, -1), "dtype": "int32", "format": "ND",
     "ori_shape": (-1, -1), "ori_format": "ND", "range": [(1, None),(1, None)]},
    {"shape": (-1, -1), "dtype": "int32", "format": "ND",
     "ori_shape": (-1, -1), "ori_format": "ND", "range": [(1, None),(1, None)]},
    {"shape": (-1, -1), "dtype": "int32", "format": "ND",
     "ori_shape": (-1, -1), "ori_format": "ND", "range": [(1, None),(1, None)]}
],
    "case_name": "select_dynamic_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case3 = {"params": [
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]}
],
    "case_name": "select_dynamic_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case4 = {"params": [
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]}
],
    "case_name": "select_dynamic_4",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case5 = {"params": [
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]}
],
    "case_name": "select_dynamic_5",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case6 = {"params": [
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND", "range": [(1, None)]}
],
    "case_name": "select_dynamic_6",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend310"], case3)
ut_case.add_case(["Ascend910A", "Ascend310"], case4)
ut_case.add_case(["Ascend910A", "Ascend310"], case5)
ut_case.add_case(["Ascend910A", "Ascend310"], case6)

def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.dynamic.select import op_select_format
    op_select_format({"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input", "value_range": [-1,2]},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                     "test_scale_op_select_format_1")

ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == '__main__':
    ut_case.run("Ascend910A")