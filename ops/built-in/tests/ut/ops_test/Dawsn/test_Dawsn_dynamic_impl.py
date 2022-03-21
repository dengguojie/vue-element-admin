#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Dawsn", "impl.dynamic.dawsn", "dawsn")

case1 = {
    "params": [
        {"shape": (-1, ), "dtype": "float32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND","range": [(1, 100)]},
        {"shape": (-1, ), "dtype": "float32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND","range": [(1, 100)]},
    ],
    "case_name": "dawsn_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1, ), "dtype": "float16", "format": "ND", "ori_shape": (2, ),"ori_format": "ND","range": [(1, 100)]},
        {"shape": (-1, ), "dtype": "float16", "format": "ND", "ori_shape": (2, ),"ori_format": "ND","range": [(1, 100)]},
    ],
    "case_name": "dawsn_2",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend610", "Ascend615", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend610", "Ascend615", "Ascend710"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910A", "Ascend610", "Ascend615", "Ascend710")
