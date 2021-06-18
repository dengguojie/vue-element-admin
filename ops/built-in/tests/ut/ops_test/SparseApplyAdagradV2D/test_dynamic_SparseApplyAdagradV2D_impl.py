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

SparseApplyAdagradV2D ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("SparseApplyAdagradV2D",
               "impl.dynamic.sparse_apply_adagrad_v2_d",
               "sparse_apply_adagrad_v2_d")

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
                "shape": (1, ),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1,),  # grad
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),   # indices
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # var
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            # lr,epsilon,use_locking,update_slots
            0.01,
            0.0001,
            False,
            False
        ],
    "case_name": 'test_sparse_apply_adagrad_v2_d_small_shape_scalar_fp32',
    "expect": "success"
}

ut_case.add_case(["Ascend910A"], case_small_shape_scalar_fp32)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
