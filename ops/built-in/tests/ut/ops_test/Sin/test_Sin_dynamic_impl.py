#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("Sin", "impl.dynamic.sin", "sin")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND","range":[(1, 100)]},
    ],
    "case_name": "Sin_1",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case("Ascend910", case1)

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
