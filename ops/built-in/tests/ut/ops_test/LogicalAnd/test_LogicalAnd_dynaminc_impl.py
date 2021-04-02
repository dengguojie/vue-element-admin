#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("LogicalAnd", "impl.dynamic.logical_and", "logical_and")

case1 = {"params": [{"shape": (-1,), "dtype": "int8", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #x
                    {"shape": (-1,), "dtype": "int8", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #h
                    {"shape": (-1,), "dtype": "int8", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]},
                    ],
         "case_name": "Assign_1",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])
