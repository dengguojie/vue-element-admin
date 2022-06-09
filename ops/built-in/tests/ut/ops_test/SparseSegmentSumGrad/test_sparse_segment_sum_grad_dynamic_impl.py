"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

SparseSegmentSumGrad ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("SparseSegmentSumGrad", "impl.dynamic.sparse_segment_sum_grad", "sparse_segment_sum_grad")


def gen_dynamic_floormod_case(shape_x, shape_y, range_x, range_y, dtype_val, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": "float32", "ori_shape": shape_x, "ori_format": "ND", "format": "ND",
                        "range": range_x},
                       {"shape": shape_y, "dtype": "int32", "ori_shape": shape_y, "ori_format": "ND", "format": "ND",
                        "range": range_y},
                       {"shape": shape_y, "dtype": "int32", "ori_shape": shape_y, "ori_format": "ND", "format": "ND",
                        "range": range_y},
                       {"shape": shape_y, "dtype": "int32", "ori_shape": shape_y, "ori_format": "ND", "format": "ND",
                        "range": range_y},
                       {"shape": shape_x, "dtype": "float32", "ori_shape": shape_x, "ori_format": "ND", "format": "ND",
                        "range": range_x}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


ut_case.add_case("Ascend910A",
                 gen_dynamic_floormod_case((-1,), (1,), ((1, None),), ((1, 1),),
                                           "float32", "dynamic_sparse_segment_sum_grad_case", "success"))
if __name__ == '__main__':
    ut_case.run("Ascend910A")
