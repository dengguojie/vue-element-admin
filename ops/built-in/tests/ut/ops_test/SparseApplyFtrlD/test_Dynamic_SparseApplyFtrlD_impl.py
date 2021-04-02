#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

Dynamic SparseApplyFtrlD ut case
"""
from op_test_frame.ut import OpUT


ut_case = OpUT("SparseApplyFtrlD", "impl.dynamic.sparse_apply_ftrl_d", "sparse_apply_ftrl_d")


def gen_dynamic_sparse_apply_ftrl_d_case(dict_var, dict_accum, dict_linear, dict_grad, dict_indices,
                                         lr, l1, l2, lr_power, kernel_name_val, expect, output_dict_var=None):
    if output_dict_var:
        params = [dict_var, dict_accum, dict_linear, dict_grad, dict_indices,
                  output_dict_var, dict_accum, dict_linear,
                  lr, l1, l2, lr_power, False]
    else:
        params = [dict_var, dict_accum, dict_linear, dict_grad, dict_indices,
                  dict_var, dict_accum, dict_linear,
                  lr, l1, l2, lr_power, False]

    return {"params": params,
            "case_name": kernel_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_01", "success"))

ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_02", "success"))

ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (-1, 1), "dtype": "float32", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((21340, 21340), (1, 1))},
                     {"shape": (-1, 1), "dtype": "float32", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((21340, 21340), (1, 1))},
                     {"shape": (-1, 1), "dtype": "float32", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((21340, 21340), (1, 1))},
                     {"shape": (-1, 1), "dtype": "float32", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((2000, 2000), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((2000, 2000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_03", "success"))

# var dtype is invalid
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80), "dtype": "int32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_04", RuntimeError))

# indices dtype is invalid
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_05", RuntimeError))

# len(var_shape) != len(accum_shape)
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80, 2), "dtype": "float32", "ori_shape": (7800, 80, 2),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80), (2, 2))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80, 2), "dtype": "float32", "ori_shape": (7800, 80, 2),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80), (2, 2))},
                     {"shape": (-1, 80, 2), "dtype": "float32", "ori_shape": (-1, 80, 2),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80), (2, 2))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_06", RuntimeError))

# len(var_shape) != len(linear_shape)
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80, 2), "dtype": "float32", "ori_shape": (7800, 80, 2),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80), (2, 2))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_07", RuntimeError))

# len(var_shape) != len(grad_shape)
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80, 2), "dtype": "float32", "ori_shape": (7800, 80, 2),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80), (2, 2))},
                     {"shape": (7800, 80, 2), "dtype": "float32", "ori_shape": (7800, 80, 2),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80), (2, 2))},
                     {"shape": (7800, 80, 2), "dtype": "float32", "ori_shape": (7800, 80, 2),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80), (2, 2))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_08", RuntimeError))

# len(var_shape) != len(var_out_shape)
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_09", RuntimeError,
                     {"shape": (7800, 80, 80), "dtype": "float32", "ori_shape": (7800, 80, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80), (80, 80))},
                 ))

# len(indices_shape) == 1
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1, 8), "dtype": "int32", "ori_shape": (-1, 8),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (8, 8))},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_10", RuntimeError))

# len(var_shape) < 2
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800),)},
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800),)},
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800),)},
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800),)},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_11", RuntimeError))

# valid: lr_attr > 0
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     -2, 0, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_12", RuntimeError))

# valid: l1_attr >= 0
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, -1, 0, -0.5,
                     "dynamic_sparse_apply_ftrl_d_13", RuntimeError))

# valid: l2_attr >= 0
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, -1, -0.5,
                     "dynamic_sparse_apply_ftrl_d_14", RuntimeError))


# valid: lr_power_attr <= 0
ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_dynamic_sparse_apply_ftrl_d_case(
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (7800, 80), "dtype": "float32", "ori_shape": (7800, 80),
                      "format": "ND", "ori_format": "ND", "range": ((7800, 7800), (80, 80))},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((1000, 1000),)},
                     2, 0, 0, 0.5,
                     "dynamic_sparse_apply_ftrl_d_15", RuntimeError))


if __name__ == '__main__':
    ut_case.run("Ascend910A")
