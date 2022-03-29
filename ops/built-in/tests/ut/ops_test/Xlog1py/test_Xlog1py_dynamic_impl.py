#!/usr/bin/env python
# -*- UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("Xlog1py", "impl.dynamic.xlog1py", "xlog1py")

case1 = {
    "params": [
        {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float16"},
        {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float16"},
        {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float16"}
    ],
    "case_name": "xlog1py_dynamic_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float32"},
        {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float32"},
        {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float32"}
    ],
    "case_name": "xlog1py_dynamic_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case3 = {
    "params": [
        {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float16"},
        {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float32"},
        {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float32"}
    ],
    "case_name": "xlog1py_dynamic_3",
    "expect": "failed",
    "format_expect": [],
    "support_expect": True
}

case4 = {
    "params": [
        {"shape": (1,), "ori_shape": (2,), "range": ((1, 100),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float16"},
        {"shape": (1,), "ori_shape": (1,), "range": ((1, 200),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float16"},
        {"shape": (1,), "ori_shape": (2,), "range": ((1, 200),), "format": "NHWC", "ori_format": "NHWC",
         'dtype': "float16"}
    ],
    "case_name": "xlog1py_dynamic_4",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case3)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case4)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710", "Ascend310"])

