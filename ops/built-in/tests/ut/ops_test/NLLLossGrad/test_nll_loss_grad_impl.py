#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("NllLossGrad", None, None)

def calc_expect_func(x, y_grad, target, weight, total_weight, x_grad, reduction):
    x_shape = x["shape"]
    x_dtype = x["dtype"]
    y_grad_value = y_grad["value"]
    target_value = target["value"]
    weight_value = weight["value"]
    total_weight_value = total_weight["value"]
    x_grad_shape = x_grad["shape"]

    n_dim = x_shape[0]
    loss = np.zeros(x_shape).astype(x_dtype)

    for i in range(0, n_dim):
        valid_weight = weight_value[target_value[i]]

        if reduction == "none":
            loss[i][target_value[i]] = -1 * y_grad_value[i] * valid_weight
        elif reduction == "sum":
            loss[i][target_value[i]] =  -1 * y_grad_value[0] * valid_weight
        elif reduction == "mean":
            loss[i][target_value[i]] =  -1 * y_grad_value[0] * valid_weight / total_weight_value[0]

    loss = loss.reshape(x_grad_shape)

    return loss


case1 = {"params": [{"shape": (1, 5), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (5, ), "dtype": "float32", "format": "ND", "ori_shape": (5, ),"ori_format": "ND"},
                    {"shape": (1, 5), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"},
                    {"shape": (1, 5), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"}],
         "case_name": "nll_loss_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 16),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (16, ), "dtype": "float32", "format": "ND", "ori_shape": (16, ),"ori_format": "ND"},
                    {"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 16),"ori_format": "ND"},
                    {"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 16),"ori_format": "ND"}],
         "case_name": "nll_loss_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (128, ), "dtype": "float32", "format": "ND", "ori_shape": (128, ),"ori_format": "ND"},
                    {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128),"ori_format": "ND"},
                    {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128),"ori_format": "ND"}],
         "case_name": "nll_loss_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend710", "Ascend910"], case3)

ut_case.add_precision_case(["Ascend910A"], {
    "params": [
        {"shape": (3, 4), "dtype": "float32", "format": "ND", "ori_shape": (3, 4), "ori_format": "ND", "param_type": "input", "value_range": [0.0, 1.0]},
        {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "param_type": "input", "value_range": [0.0, 1.0]},
        {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND", "param_type": "input","value_range": [0, 3]},
        {"shape": (4,), "dtype": "float32", "format": "ND", "ori_shape": (4,), "ori_format": "ND", "param_type": "input","value_range": [0.0, 1.0]},
        {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "param_type": "input","value_range": [1.0, 10.0]},
        {"shape": (3, 4), "dtype": "float32", "format": "ND", "ori_shape": (3,4 ), "ori_format": "ND", "param_type": "output"}, 
        "mean"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)
})

if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/home/z00511265/.mindstudio/huawei/adk/1.75.T13.0.B130/toolkit/tools/simulator")
