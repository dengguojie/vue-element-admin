#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("Invert", "impl.dynamic.invert", "invert")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "int16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "int16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
    ],
    "case_name": "Invert_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1, -1), "dtype": "uint16", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND","range":[(1, 100), (1, 100)]},
        {"shape": (-1, -1), "dtype": "uint16", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND","range":[(1, 100), (1, 100)]},
    ],
    "case_name": "Invert_2",
    "expect": "success",
    "support_expect": True
}
case3 = {
    "params": [
        {"shape": (-1, -1, -1), "dtype": "uint16", "format": "ND", "ori_shape": (2, 3, 4),"ori_format": "ND","range":[(1, 100), (1, 100), (1, 100)]},
        {"shape": (-1, -1, -1), "dtype": "uint16", "format": "ND", "ori_shape": (2, 3, 4),"ori_format": "ND","range":[(1, 100), (1, 100), (1, 100)]},
    ],
    "case_name": "Invert_3",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910A")
