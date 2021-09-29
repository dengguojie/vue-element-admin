#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
diag
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("Diag", "impl.dynamic.diag", "diag")

case1 = {"params": [{"shape": (-1, -1,), "dtype": "float16", "format": "ND", "ori_shape": (150,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (150, 150),
                     "ori_format": "ND", "range": [(1, 200)]},
                    ],
         "case_name": "diag1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (150,),
                     "ori_format": "ND", "range": [(1, 200)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (150, 150),
                     "ori_format": "ND", "range": [(1, 200)]},
                    ],
         "case_name": "diag2",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend610", "Ascend615", "Ascend710",
                  "HI3796CV300CS", "Hi3796CV300ES"], case1)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend610", "Ascend615", "Ascend710",
                  "HI3796CV300CS", "Hi3796CV300ES"], case2)
if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310", "Ascend610", "Ascend615", "Ascend710",
                 "HI3796CV300CS", "Hi3796CV300ES"])
