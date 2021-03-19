#/user/bin/env python
# -*- coding: UTF-8 -*-
'''
apply_adadelta_d
'''
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyAdadeltaD", "impl.dynamic.apply_adadelta_d", "apply_adadelta_d")

case1 = {"params": [{"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    ],
         "case_name": "ApplyAdadeltaD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    ],
         "case_name": "ApplyAdadeltaD_2",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run(["Ascend910A"])
