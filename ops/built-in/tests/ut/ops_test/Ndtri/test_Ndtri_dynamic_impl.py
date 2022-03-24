#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Ndtri", "impl.dynamic.ndtri", "ndtri")

case1 = {
    "params": [
        {"shape": (-1, ), "dtype": "float32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND","range": [(1, 100)]},
        {"shape": (-1, ), "dtype": "float32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND","range": [(1, 100)]},
    ],
    "case_name": "Ndtri_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1, ), "dtype": "float16", "format": "ND", "ori_shape": (2, ),"ori_format": "ND","range": [(1, 100)]},
        {"shape": (-1, ), "dtype": "float16", "format": "ND", "ori_shape": (2, ),"ori_format": "ND","range": [(1, 100)]},
    ],
    "case_name": "Ndtri_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [
        {"shape": (-1, -1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float16"},
        {"shape": (-1, -1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float16"}
    ],
    "case_name": "Ndtri_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [
        {"shape": (-2,), "ori_shape": (-2,), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float32"},
        {"shape": (-2,), "ori_shape": (-2,), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float32"}
    ],
    "case_name": "Ndtri_4",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend610", "Ascend615", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend610", "Ascend615", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend610", "Ascend615", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend610", "Ascend615", "Ascend710", "Ascend910A"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend610", "Ascend615", "Ascend710", "Ascend910A")
