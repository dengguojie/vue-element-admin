#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("Assign", "impl.dynamic.assign", "assign")

case1 = {"params": [{"shape": (-2,), "dtype": "bool", "format": "ND", "ori_shape": (-2,),"ori_format": "ND","range":[(1, 100)]}, #x
                    {"shape": (-2,), "dtype": "bool", "format": "ND", "ori_shape": (-2,),"ori_format": "ND","range":[(1, 100)]}, #h
                    {"shape": (-2,), "dtype": "bool", "format": "ND", "ori_shape": (-2,),"ori_format": "ND","range":[(1, 100)]},
                    ],
         "case_name": "Assign_1",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")