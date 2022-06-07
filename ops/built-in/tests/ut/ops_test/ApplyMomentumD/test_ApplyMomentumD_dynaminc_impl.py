#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyMomentumD", "impl.dynamic.apply_momentum_d", "apply_momentum_d")


case1 = {"params": [{"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]}],
         "case_name": "apply_momentum_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 100)]}, True],
         "case_name": "apply_momentum_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case2)
# ut_case.add_case(["Ascend310"], case2)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend310P3", "Ascend910A"])
