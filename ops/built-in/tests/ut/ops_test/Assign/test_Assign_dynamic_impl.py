#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te

from op_test_frame.ut import OpUT
ut_case = OpUT("Assign", "impl.dynamic.assign", "assign")

case1 = {"params": [{"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #x
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #h
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]},
                    ],
         "case_name": "Assign_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2,),"ori_format": "ND","range":[(1, 100)]}, #x
                    {"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2,),"ori_format": "ND","range":[(1, 100)]}, #h
                    {"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2,),"ori_format": "ND","range":[(1, 100)]},
                    ],
         "case_name": "Assign_2",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910A")