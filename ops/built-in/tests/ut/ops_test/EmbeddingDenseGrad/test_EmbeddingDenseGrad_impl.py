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

EmbeddingDenseGrad ut case
"""

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("EmbeddingDenseGrad","impl.embedding_dense_grad", "embedding_dense_grad")

case1 = {"params": [{"shape": (3, 3, 32), "dtype": "float32", "format": "ND", "ori_shape": (3, 3, 32),
                     "ori_format": "ND"},
                    {"shape": (3, 3), "dtype": "int32", "format": "ND",
                     "ori_shape": (3, 3),"ori_format": "ND"},
                    {"shape": (23, 32), "dtype": "float32", "format": "ND", "ori_shape": (23, 32),
                     "ori_format": "ND"}, 23, -1, False],
         "case_name": "embedding_dense_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (3, 3, 1), "dtype": "float32", "format": "ND",
                     "ori_shape": (3, 3, 1),"ori_format": "ND"},
                    {"shape": (3, 3), "dtype": "int32", "format": "ND",
                     "ori_shape": (3, 3), "ori_format": "ND"},
                    {"shape": (23, 1), "dtype": "float32", "format": "ND", "ori_shape": (23, 1),
                     "ori_format": "ND"}, 23, -1, False],
         "case_name": "embedding_dense_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (7000, 1), "dtype": "float32", "format": "ND", "ori_shape": (7000, 1),
                     "ori_format": "ND"},
                    {"shape": (7000,), "dtype": "int32", "format": "ND",
                     "ori_shape": (7000,),"ori_format": "ND"},
                    {"shape": (7000, 1), "dtype": "float32", "format": "ND", "ori_shape": (7000, 1),
                     "ori_format": "ND"}, 7000, -1, False],
         "case_name": "embedding_dense_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (7000, 64), "dtype": "float32", "format": "ND", "ori_shape": (7000, 64),
                     "ori_format": "ND"},
                    {"shape": (7000,), "dtype": "int32", "format": "ND",
                     "ori_shape": (7000,),"ori_format": "ND"},
                    {"shape": (7000, 64), "dtype": "float32", "format": "ND", "ori_shape": (7000, 64),
                     "ori_format": "ND"}, 7000, -1, True],
         "case_name": "embedding_dense_grad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (3, 3, 32), "dtype": "float32", "format": "ND", "ori_shape": (3, 3, 32),
                     "ori_format": "ND"},
                    {"shape": (3, 3), "dtype": "int32", "format": "ND",
                     "ori_shape": (3, 3),"ori_format": "ND"},
                    {"shape": (23, 32), "dtype": "float32", "format": "ND", "ori_shape": (23, 32),
                     "ori_format": "ND"}, 23, -1, True],
         "case_name": "embedding_dense_grad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (3, 3, 512), "dtype": "float32", "format": "ND",
                     "ori_shape": (3, 3, 512),"ori_format": "ND"},
                    {"shape": (3, 3), "dtype": "int32", "format": "ND",
                     "ori_shape": (3, 3), "ori_format": "ND"},
                    {"shape": (40, 512), "dtype": "float32", "format": "ND", "ori_shape": (40, 512),
                     "ori_format": "ND"}, 40, -1, True],
         "case_name": "embedding_dense_grad_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (40000, 513), "dtype": "float32", "format": "ND",
                     "ori_shape": (40000, 513),"ori_format": "ND"},
                    {"shape": (40000,), "dtype": "int32", "format": "ND",
                     "ori_shape": (40000,), "ori_format": "ND"},
                    {"shape": (40000, 513), "dtype": "float32", "format": "ND", "ori_shape": (40000, 513),
                     "ori_format": "ND"}, 40000, -1, False],
         "case_name": "embedding_dense_grad_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (41111, 1025), "dtype": "float32", "format": "ND",
                     "ori_shape": (41111, 1025),"ori_format": "ND"},
                    {"shape": (41111,), "dtype": "int32", "format": "ND",
                     "ori_shape": (41111,), "ori_format": "ND"},
                    {"shape": (41111, 1025), "dtype": "float32", "format": "ND", "ori_shape": (41111, 1025),
                     "ori_format": "ND"}, 41111, -1, False],
         "case_name": "embedding_dense_grad_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case_lis = [case1, case2, case3, case4, case5, case6, case7, case8]
for ele in case_lis:
    ut_case.add_case(["Ascend910A"], ele)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
