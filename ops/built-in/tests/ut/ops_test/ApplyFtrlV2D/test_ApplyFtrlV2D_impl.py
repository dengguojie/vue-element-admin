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

ApplyFtrlV2D ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("ApplyFtrlV2D", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #x
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #c
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #w
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},  #b
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #mask
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #ht
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #it
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #jt
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #ft
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #ot
                    False,
                    ],
         "case_name": "ApplyFtrlV2D_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #x
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #c
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #w
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},  #b
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #mask
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #ht
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #it
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #jt
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #ft
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #ot
                    True,
                    ],
         "case_name": "ApplyFtrlV2D_2",
         "expect": "success",
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)

#precision cases
def _gen_outputs(input_var, input_accum, input_linear, input_grad, input_lr,
                 input_l1, input_l2, input_l2_shrinkage, input_lr_power, dtype, input_shape):
    element_num = 1
    for i in range(0, len(input_shape)):
        element_num = element_num * input_shape[i]

    if dtype == "float16":
        input_var = input_var.astype(np.float32)
        input_accum = input_accum.astype(np.float32)
        input_linear = input_linear.astype(np.float32)
        input_grad = input_grad.astype(np.float32)
        input_lr = input_lr.astype(np.float32)
        input_l1 = input_l1.astype(np.float32)
        input_l2 = input_l2.astype(np.float32)
        input_l2_shrinkage = input_l2_shrinkage.astype(np.float32)
        input_lr_power = input_lr_power.astype(np.float32)

    # 1.grad_with_shrinkage = grad + 2 * l2_shrinkage * var
    grad_with_shrinkage = input_grad + 2 * input_l2_shrinkage * input_var
    # 2.accum_new = accum + grad * grad
    accum_new = input_accum + input_grad * input_grad
    # 3.linear += grad_with_shrinkage-(accum_new^(-lr_power )- accum^(-lr_power ) )/ lr * var
    neg_lr_pow = 0 - input_lr_power
    new_linear = grad_with_shrinkage - ((
                                                (np.power(accum_new, neg_lr_pow)) - (np.power(input_accum, neg_lr_pow))) / input_lr) * input_var
    output_linear = input_linear + new_linear
    # 4.quadratic = accum_new^(-lr_power)/lr + 2*l2
    quadratic = (np.power(accum_new, neg_lr_pow)) / input_lr + 2 * input_l2
    # 5.var =((sign(linear)* l1 - linear))/quadratic if |linear|> l1 else 0.0
    x_res = np.sign(output_linear) * input_l1 - output_linear
    div_res = x_res / quadratic
    zero_res = input_var * 0.0
    abs_linear = np.abs(output_linear)

    a_list = abs_linear.flatten()
    b_list = input_l1.flatten()
    c1_list = div_res.flatten()
    c2_list = zero_res.flatten()
    d_list = []
    for i in range(0, element_num):
        if a_list[i] > b_list[0]:
            d_list.append(c1_list[i])
        else:
            d_list.append(c2_list[i])
    d_numpy = np.array(d_list)
    output_var = d_numpy.reshape(input_shape)

    # 6.accum = accum_new
    output_accum = accum_new

    output_data = output_var

    output_var = output_var.astype(dtype)
    output_accum = output_accum.astype(dtype)
    output_linear = output_linear.astype(dtype)
    output_data = output_data.astype(dtype)
    return output_var, output_accum, output_linear

def calc_expect_func(x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3):
    res1, res2, res3 = _gen_outputs(x1['value'], x2['value'], x3['value'], x4['value'],
                                    x5['value'], x6['value'], x7['value'], x8['value'],
                                    x9['value'], x1['dtype'], x1['shape'])
    return res1, res2, res3

precision_case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"}
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910", precision_case1)

