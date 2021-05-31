#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Acosh", "impl.dynamic.acosh", "acosh")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND","range":[(1, 100)]},
    ],
    "case_name": "Acosh_1",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
