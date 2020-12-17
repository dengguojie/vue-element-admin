#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te

from op_test_frame.ut import OpUT
ut_case = OpUT("AssignAdd", "impl.dynamic.assign_add", "assign_add")

case1 = {"params": [{"shape": (-1,), "dtype": "uint8", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #x
                    {"shape": (-1,), "dtype": "uint8", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #h
                    {"shape": (-1,), "dtype": "uint8", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]},
                    ],
         "case_name": "Assign_1",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run(["Ascend310", "Ascend710", "Ascend910"])

