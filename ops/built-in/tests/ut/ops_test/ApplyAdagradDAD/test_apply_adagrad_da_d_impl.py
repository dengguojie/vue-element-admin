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

ApplyAdagradDaD ut case
"""
# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name,missing-docstring
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
ut_case = OpUT("ApplyAdagradDaD", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    ],
         "case_name": "ApplyAdagradDaD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    ],
         "case_name": "ApplyAdagradDaD_2",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case2)

def _gen_outputs(input_var, input_grad_accum, input_grad_squa_accum,
                 input_grad, input_lr, input_l1,
                 input_l2, input_global_step, dtype):
    if dtype == "float16":
        input_var = input_var.astype(np.float32)
        input_grad_accum = input_grad_accum.astype(np.float32)
        input_grad_squa_accum = input_grad_squa_accum.astype(np.float32)
        input_grad = input_grad.astype(np.float32)
        input_lr = input_lr.astype(np.float32)
        input_l1 = input_l1.astype(np.float32)
        input_l2 = input_l2.astype(np.float32)

    # 1.grad_accum += grad
    output_grad_accum = input_grad_accum + input_grad

    # 2.grad_squared_accum += grad * grad
    output_grad_squa_accum = input_grad_squa_accum + input_grad * input_grad

    # 3.tmp_val = sign(grad_accum) * max(| grad_accum | -l1 * global_step, 0) if l1 > 0 else grad_accum
    l1_list = input_l1.flatten()
    if l1_list[0] > 0:
        sub_val = np.abs(output_grad_accum) - input_l1 * input_global_step
        tmp_val = np.sign(output_grad_accum) * np.maximum(sub_val, 0)
    else:
        tmp_val = output_grad_accum

    # 4.x_value = -1 * lr * tmp_val
    x_value = -1 * input_lr * tmp_val

    # 5.y_value = l2 * global_step * lr + sqrt(grad_squared_accum)
    y_value = input_l2 * input_global_step * input_lr + np.sqrt(output_grad_squa_accum)

    # 6.var = x_value / y_value
    output_var = x_value / y_value

    # 7.output_data = var
    output_data = output_var

    output_var = output_data.astype(dtype)
    output_grad_accum = output_grad_accum.astype(dtype)
    output_grad_squa_accum = output_grad_squa_accum.astype(dtype)

    return output_var, output_grad_accum, output_grad_squa_accum

def calc_expect_func(x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3):
    res1, res2, res3 = _gen_outputs(x1['value'], x2['value'], x3['value'], x4['value'],
                                    x5['value'], x6['value'], x7['value'], x8['value'],
                                    x1['dtype'])
    return res1, res2, res3

precision_case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "input"},
                              {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "output"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "output"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                               "ori_format": "ND", "param_type": "output"},
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910", precision_case1)
