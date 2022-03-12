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

Muls ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
from te import tvm
import os

ut_case = OpUT("Muls", None, None)

case1 = {"params": [{"shape": (125, 125), "dtype": "float16", "format": "ND", "ori_shape": (125, 125),"ori_format": "ND"},
                    {"shape": (125, 125), "dtype": "float16", "format": "ND", "ori_shape": (125, 125),"ori_format": "ND"},
                    3.0],
         "case_name": "Muls_1",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

case2 = {"params": [{"shape": (3, 30, 100), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100),"ori_format": "ND"},
                    {"shape": (3, 30, 100), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100),"ori_format": "ND"},
                    3.0],
         "case_name": "Muls_2",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

case3 = {"params": [{"shape": (3, 30, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16),"ori_format": "ND"},
                    {"shape": (3, 30, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16),"ori_format": "ND"},
                    3.0],
         "case_name": "Muls_3",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

case4 = {"params": [{"shape": (3, 30, 100, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16, 17),"ori_format": "ND"},
                    {"shape": (3, 30, 100, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16, 17),"ori_format": "ND"},
                    3.0],
         "case_name": "Muls_4",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

def test_muls_compute_with_batchmatmul_1(test_arg):
    from impl.muls import muls_compute
    x = tvm.placeholder((2, 2, 2, 16, 16), name="x", dtype="float16",
                        attrs={'format': "FRACTAL_NZ", "ori_shape": (2, 32, 32)})
    tensor_x = tvm.compute((2, 2, 2, 16, 16), lambda *i: x(*i), name="tensor_x", tag="matmul",
                           attrs={'format': "FRACTAL_NZ", "ori_shape": (2, 32, 32), "batch_shape": (2,)})
    output = {"shape": (2, 2, 2, 16, 16), "dtype": "float16", "ori_shape": (2, 32, 32), "format": "FRACTAL_NZ"}
    muls_compute(tensor_x, output, 2.0, "muls_kernel")

def test_muls_compute_with_batchmatmul_2(test_arg):
    from impl.muls import muls_compute
    x = tvm.placeholder((2, 2, 2, 16, 16), name="x", dtype="float16",
                        attrs={'format': "FRACTAL_NZ", "ori_shape": (2, 32, 32)})
    tensor_x = tvm.compute((2, 2, 2, 16, 16), lambda *i: x(*i), name="tensor_x", tag="matmul",
                           attrs={'format': "FRACTAL_NZ", "ori_shape": (2, 32, 32), "batch_shape": (2,), "para_name": "muls"})
    output = {"shape": (2, 2, 2, 16, 16), "dtype": "float16", "ori_shape": (2, 32, 32), "format": "FRACTAL_NZ"}
    muls_compute(tensor_x, output, 2.0, "muls_kernel")

ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910"], case3)
ut_case.add_case(["Ascend910"], case4)
ut_case.add_cust_test_func(test_func=test_muls_compute_with_batchmatmul_1)
ut_case.add_cust_test_func(test_func=test_muls_compute_with_batchmatmul_2)