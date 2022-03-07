#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SeluGrad", "impl.dynamic.selu_grad", "selu_grad")
    
case1 = {"params": [{"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #x
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #h
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]},
                    ],
         "case_name": "selu_grad_0",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2, ),"ori_format": "ND","range":[(1, None)]}, #x
                    {"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2, ),"ori_format": "ND","range":[(1, None)]}, #h
                    {"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2, ),"ori_format": "ND","range":[(1, None)]},
                    ],
         "case_name": "selu_grad_1",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #x
                    {"shape": (-1, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #h
                    {"shape": (-1, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]},
                    ],
         "case_name": "selu_grad_2",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #x
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #h
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]},
                    ],
         "case_name": "selu_grad_3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A","Ascend610","Ascend615","Ascend710","Ascend920A"], case1)
ut_case.add_case(["Ascend910A","Ascend610","Ascend615","Ascend710","Ascend920A"], case2)
ut_case.add_case(["Ascend910A","Ascend610","Ascend615","Ascend710","Ascend920A"], case3)
ut_case.add_case(["Ascend910A","Ascend610","Ascend615","Ascend710","Ascend920A"], case4)

if __name__ == '__main__':
    ut_case.run(["Ascend910A","Ascend610","Ascend615","Ascend710","Ascend920A"])