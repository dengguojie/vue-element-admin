#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("NllLoss", None, None)

case1 = {"params": [{"shape": (1, 5), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (5, ), "dtype": "float32", "format": "ND", "ori_shape": (5, ),"ori_format": "ND"},
                    {"shape": (1, 5), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"},
                    {"shape": (1, 5), "dtype": "float32", "format": "ND", "ori_shape": (1, 5),"ori_format": "ND"}],
         "case_name": "nll_loss_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 16),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (16, ), "dtype": "float32", "format": "ND", "ori_shape": (16, ),"ori_format": "ND"},
                    {"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 16),"ori_format": "ND"},
                    {"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 16),"ori_format": "ND"}],
         "case_name": "nll_loss_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (128, ), "dtype": "float32", "format": "ND", "ori_shape": (128, ),"ori_format": "ND"},
                    {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128),"ori_format": "ND"},
                    {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128),"ori_format": "ND"}],
         "case_name": "nll_loss_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

# if __name__ == '__main__':
    # ut_case.run("Ascend910")
    # exit(0)

def calc_expect_func(inputArr, target, weight, output_y, total_weight, reduction):
    if reduction != "none":
        shape_out = [1,]
    else:
        shape_out = list(inputArr["shape"][0])
    outputArr = np.zeros(shape_out).astype(inputArr["dtype"])
    fake_out = np.zeros(list(inputArr["shape"][0])).astype(inputArr["dtype"])
    total_weight = 0
    for i in range(0, shape_out[0]):
        fake_out[i] = inputArr["value"][i][target["value"][i]]
        fake_out[i] = fake_out[i] * weight["value"][target[i]]
        total_weight += weight["value"][target[i]]

    if reduction == "none":
        outputArr = fake_out*-1
    elif reduction == "sum":
        outputArr[0] = np.sum(fake_out) * -1
    elif reduction == "mean":
        outputArr[0] = np.sum(fake_out)/total_weight
        outputArr[0] = outputArr[0]*-1

    return outputArr


ut_case.add_precision_case("all", {
    "params": [
        {"shape": (3, 4), "dtype": "float32", "format": "ND", "ori_shape": (3, 4), "ori_format": "ND", "param_type": "input", "value_range": [0.0, 1.0]},
        {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND", "param_type": "input","value_range": [0, 3]},
        {"shape": (4,), "dtype": "float32", "format": "ND", "ori_shape": (4,), "ori_format": "ND", "param_type": "input","value_range": [0.0, 1.0]},
        {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND", "param_type": "output"},
        {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND", "param_type": "output"}, "mean",],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.00001)
})

if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/home/z00511265/.mindstudio/huawei/adk/1.75.T13.0.B130/toolkit/tools/simulator")
