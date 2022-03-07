#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("SoftsignGrad", "impl.dynamic.softsign_grad", "softsign_grad")

case1 = {"params": [{"shape": (-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100),(1, 100)]},
                    {"shape": (-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100),(1, 100)]},
                    {"shape": (-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100),(1, 100)]},
                    ],
         "case_name": "softsign_grad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (32,112,15,112), "dtype": "float16", "format": "ND", "ori_shape": (32,112,15,112),"ori_format": "ND","range":[(32,32),(112,112),(15,15),(112,112)]},
                    {"shape": (32,112,15,112), "dtype": "float16", "format": "ND", "ori_shape": (32,112,15,112),"ori_format": "ND","range":[(32,32),(112,112),(15,15),(112,112)]},
                    {"shape": (32,112,15,112), "dtype": "float16", "format": "ND", "ori_shape": (32,112,15,112),"ori_format": "ND","range":[(32,32),(112,112),(15,15),(112,112)]},
                    ],
         "case_name": "softsign_grad_2",
         "expect": "success",
         "support_expect": True}
case3 = {"params": [{"shape": (32,112,15,-1), "dtype": "float32", "format": "ND", "ori_shape": (32,112,15,112),"ori_format": "ND","range":[(32,32),(112,112),(15,15),(110,120)]},
                    {"shape": (32,112,15,112), "dtype": "float32", "format": "ND", "ori_shape": (32,112,15,112),"ori_format": "ND","range":[(32,32),(112,112),(15,15),(112,112)]},
                    {"shape": (32,112,15,112), "dtype": "float32", "format": "ND", "ori_shape": (32,112,15,112),"ori_format": "ND","range":[(32,32),(112,112),(15,15),(112,112)]},
                    ],
         "case_name": "softsign_grad_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (32,112,15,112),"ori_format": "ND","range":[(32,32),(112,112),(15,15),(110,120)]},
                    {"shape": (32,112,15,112), "dtype": "float32", "format": "ND", "ori_shape": (32,112,15,112),"ori_format": "ND","range":[(32,32),(112,112),(15,15),(112,112)]},
                    {"shape": (32,112,15,112), "dtype": "float32", "format": "ND", "ori_shape": (32,112,15,112),"ori_format": "ND","range":[(32,32),(112,112),(15,15),(112,112)]},
                    ],
         "case_name": "softsign_grad_4",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A","Ascend610","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend610","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend610","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend610","Ascend710"], case4)

if __name__ == '__main__':
    ut_case.run(["Ascend610", "Ascend710", "Ascend910A"])
