#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("NllLoss", None, None)

def calc_expect_func(x, target, weight, y, total_weight, reduction):
    x_shape = x["shape"]
    x_value = x["value"]
    x_dtype = x["dtype"]
    target_value = target["value"]
    weight_value = weight["value"]
    y_shape = y["shape"]
    total_weight_shape = total_weight["shape"]

    n_dim = x_shape[0]
    total_weight = 0
    loss = np.zeros([n_dim]).astype(x_dtype)
    
    for i in range(0, n_dim):
        valid_weight = weight_value[target_value[i]]
        loss[i] = -1 * x_value[i][target_value[i]] * valid_weight
        total_weight += valid_weight

    if reduction == "sum":
        loss = np.sum(loss).reshape(y_shape)
    elif reduction == "mean":
        loss = (np.sum(loss) / total_weight).reshape(y_shape)
    
    total_weight = total_weight.reshape(total_weight_shape)
        
    return loss, total_weight

case1 = {"params": [{"shape": (2, 5), "dtype": "float32", "format": "ND", "ori_shape": (2, 5),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (5, ), "dtype": "float32", "format": "ND", "ori_shape": (5, ),"ori_format": "ND"},
                    {"shape": (2, ), "dtype": "float32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    "none"],
         "case_name": "nll_loss_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 16),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (16, ), "dtype": "float32", "format": "ND", "ori_shape": (16, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    "sum"],
         "case_name": "nll_loss_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (3, 128), "dtype": "float32", "format": "ND", "ori_shape": (3, 128),"ori_format": "ND"},
                    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,),"ori_format": "ND"},
                    {"shape": (128, ), "dtype": "float32", "format": "ND", "ori_shape": (128, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    "mean"],
         "case_name": "nll_loss_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (3072, 30528), "dtype": "float32", "format": "ND", "ori_shape": (3072, 30528),"ori_format": "ND"},
                    {"shape": (3072,), "dtype": "int32", "format": "ND", "ori_shape": (3072,),"ori_format": "ND"},
                    {"shape": (30528, ), "dtype": "float32", "format": "ND", "ori_shape": (30528, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    "mean"],
         "case_name": "nll_loss_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (3072, 30528), "dtype": "float32", "format": "ND", "ori_shape": (3072, 30528),"ori_format": "ND"},
                    {"shape": (3072,), "dtype": "int32", "format": "ND", "ori_shape": (3072,),"ori_format": "ND"},
                    {"shape": (30528, ), "dtype": "float32", "format": "ND", "ori_shape": (30528, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    "mean",
                    -1],
         "case_name": "nll_loss_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend710", "Ascend910A"], case5)

ut_case.add_precision_case(["Ascend910A"], {
    "params": [
        {"shape": (3, 4), "dtype": "float32", "format": "ND", "ori_shape": (3, 4), "ori_format": "ND", "param_type": "input", "value_range": [0.0, 1.0]},
        {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND", "param_type": "input","value_range": [0, 3]},
        {"shape": (4,), "dtype": "float32", "format": "ND", "ori_shape": (4,), "ori_format": "ND", "param_type": "input","value_range": [0.0, 1.0]},
        {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND", "param_type": "output"},
        {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND", "param_type": "output"},
        "mean"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)
})

def test_op_check_supported(test_arg):
    from impl.nll_loss import check_supported
    check_supported({"shape": (2, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 16), "ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    "sum", -100)

ut_case.add_cust_test_func(test_func=test_op_check_supported)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"], simulator_mode="pv",
                simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
