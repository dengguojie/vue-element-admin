#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
rsqrt_grad
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("RsqrtGrad", "impl.dynamic.rsqrt_grad", "rsqrt_grad")

case1 = {"params": [{"shape": (-1,), "dtype": "int8", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "int8", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "int8", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    ],
         "case_name": "rsqrt_grad1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    ],
         "case_name": "rsqrt_grad2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,),
                     "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,),
                     "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,),
                     "ori_format": "ND"},
                    ],
         "case_name": "rsqrt_grad3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
