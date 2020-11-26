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

SparseApplyProximalAdagradD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("SparseApplyProximalAdagradD", None, None)
case1 = {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                    {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    False,
                    ],
         "case_name": "SparseApplyProximalAdagradD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (8,1,2), "dtype": "float32", "format": "ND", "ori_shape": (8,1,2),"ori_format": "ND"},
                    {"shape": (8,1,2), "dtype": "float32", "format": "ND", "ori_shape": (8,1,2),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (8,), "dtype": "float16", "format": "ND", "ori_shape": (8,),"ori_format": "ND"},
                    {"shape": (8,1,2), "dtype": "float32", "format": "ND", "ori_shape": (8,1,2),"ori_format": "ND"},
                    False,
                    ],
         "case_name": "SparseApplyProximalAdagradD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1555, 65535, 3776), "dtype": "float32", "format": "ND", "ori_shape": (1555, 65535, 3776),"ori_format": "ND"},
                    {"shape": (1555, 65535, 3776), "dtype": "float32", "format": "ND", "ori_shape": (1555, 65535, 3776),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (3110,), "dtype": "float32", "format": "ND", "ori_shape": (3110,),"ori_format": "ND"},
                    {"shape": (1555,), "dtype": "int16", "format": "ND", "ori_shape": (1555,),"ori_format": "ND"},
                    {"shape": (1555, 65535, 3776), "dtype": "float16", "format": "ND", "ori_shape": (1555, 65535, 3776),"ori_format": "ND"},
                    {"shape": (1555, 65535, 3776), "dtype": "float32", "format": "ND", "ori_shape": (1555, 65535, 3776),"ori_format": "ND"},
                    False,
                    ],
         "case_name": "SparseApplyProximalAdagradD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (5999, 9999), "dtype": "float32", "format": "ND", "ori_shape": (5999, 9999),"ori_format": "ND"},
                    {"shape": (5999, 9999), "dtype": "float32", "format": "ND", "ori_shape": (5999, 9999),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (7554,), "dtype": "float32", "format": "ND", "ori_shape": (7554,),"ori_format": "ND"},
                    {"shape": (1555,), "dtype": "int32", "format": "ND", "ori_shape": (1555,),"ori_format": "ND"},
                    {"shape": (5999, 9999), "dtype": "float16", "format": "ND", "ori_shape": (5999, 9999),"ori_format": "ND"},
                    {"shape": (5999, 9999), "dtype": "float32", "format": "ND", "ori_shape": (5999, 9999),"ori_format": "ND"},
                    True,
                    ],
         "case_name": "SparseApplyProximalAdagradD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (64, 64), "dtype": "float32", "format": "ND", "ori_shape": (64, 64),"ori_format": "ND"},
                    {"shape": (64, 64), "dtype": "float32", "format": "ND", "ori_shape": (64, 64),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (77,), "dtype": "float32", "format": "ND", "ori_shape": (77,),"ori_format": "ND"},
                    {"shape": (13, 13), "dtype": "int32", "format": "ND", "ori_shape": (13, 13),"ori_format": "ND"},
                    {"shape": (64, 64), "dtype": "float16", "format": "ND", "ori_shape": (64, 64),"ori_format": "ND"},
                    {"shape": (64, 64), "dtype": "float32", "format": "ND", "ori_shape": (64, 64),"ori_format": "ND"},
                    False,
                    ],
         "case_name": "SparseApplyProximalAdagradD_5",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend710"], case5)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
