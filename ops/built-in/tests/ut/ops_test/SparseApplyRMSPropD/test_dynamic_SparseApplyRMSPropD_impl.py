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

SparseApplyRMSPropD ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
import random

ut_case = OpUT("SparseApplyRmsPropD",
               "impl.dynamic.sparse_apply_rms_prop_d",
               "sparse_apply_rms_prop_d")

case_small_shape_scalar_fp32 = {
    "params":
        [
            {
                "shape": (1, ),  # var
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # lr
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1,),  # grad
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),   # indices
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),  # var
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            0.5, 0.99,
            0.0001,
            False
        ],
    "case_name": 'test_sparse_apply_rms_prop_d_small_shape_scalar_fp32',
    "expect": "success"
}

ut_case.add_case(["Ascend910A"], case_small_shape_scalar_fp32)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
