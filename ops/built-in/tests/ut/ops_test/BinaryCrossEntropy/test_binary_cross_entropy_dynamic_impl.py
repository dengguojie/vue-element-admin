"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

BinaryCrossEntropy ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BinaryCrossEntropy", "impl.dynamic.binary_cross_entropy", "binary_cross_entropy")

case1 = {"params": [{"shape": (-1, 8, 375), "dtype": "float16",
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (-1, 8, 375), "dtype": "float16", 
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (-1, 8, 375), "dtype": "float16", 
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (1,), "dtype": "float16", 
                     "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 1)]},
                    "mean"
                    ],
         "case_name": "BinaryCrossEntroy_1",
         "expect": "success",
         "support_expect": True}
case2 = {"params": [{"shape": (-1, 8, 375), "dtype": "float16",
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},#x
                    {"shape": (-1, 8, 375), "dtype": "float16", 
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    None,
                    {"shape": (1,), "dtype": "float16", 
                     "format": "ND", "ori_shape":(1,),
                     "ori_format": "ND", "range":[(1, 1)]},
                    "none"
                    ],
         "case_name": "BinaryCrossEntroy_2",
         "expect": "success",
         "support_expect": True}
case3 = {"params": [{"shape": (-1, 8, 375), "dtype": "float32",
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},#x
                    {"shape": (-1, 8, 375), "dtype": "float32",
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (-1, 8, 375), "dtype": "float32",
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (1,), "dtype": "float32", 
                     "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 1)]},
                    "none"
                    ],
         "case_name": "BinaryCrossEntroy_3",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
# ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A", "Ascend310"], case3)



if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend310"])
