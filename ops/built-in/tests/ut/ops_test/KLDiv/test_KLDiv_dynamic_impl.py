#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("KLDiv", "impl.dynamic.kl_div", "kl_div")

case1 = {
  "params": [
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    "batchmean"
  ],
  "case_name": "kl_div_dynamic_1",
  "expect": "success",
  "format_expect": [],
  "support_expect": True
}

case2 = {
  "params": [
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    "none"
  ],
  "case_name": "kl_div_dynamic_2",
  "expect": "success",
  "format_expect": [],
  "support_expect": True
}

case3 = {
  "params": [
    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    "sum"
  ],
  "case_name": "kl_div_dynamic_3",
  "expect": "success",
  "format_expect": [],
  "support_expect": True
}

case4 = {
  "params": [
    {'shape': (16, 1, 77), 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': (16, 1, 77), 'ori_format': 'NCHW'},
    {'shape': (16, 1, 77), 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': (16, 1, 77), 'ori_format': 'NCHW'},
    {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND'},
    "batchmean"
  ],
  "case_name": "kl_div_static_1",
  "expect": "success",
  "format_expect": [],
  "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)


def test_op_select_format(test_arg):
  from impl.dynamic.kl_div import op_select_format
  op_select_format(
    {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
     "ori_format": "NDC1HWC0"},
    {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
     "ori_format": "NDC1HWC0"},
    {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
     "ori_format": "NDC1HWC0"}, "none")
  op_select_format(
    {"shape": (1, 2, 1, 1), "ori_shape": (1, 2, 1, 1), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
    {"shape": (1, 2, 1, 1), "ori_shape": (1, 2, 1, 1), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
    {"shape": (1,), "ori_shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND"}, "sum")
  op_select_format(
    {"shape": (-1, -1, -1, -1, -1), "ori_shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NC1HWC0",
     "ori_format": "NCHW"},
    {"shape": (-1, -1, -1, -1, -1), "ori_shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NC1HWC0",
     "ori_format": "NCHW"},
    {"shape": (1,), "ori_shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND"}, "sum")


ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
