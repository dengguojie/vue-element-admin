"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dot ut case
"""
# # -*- coding:utf-8 -*-
import sys
import numpy as np
import torch
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("soft_margin_loss")

#pylint: disable=unused-argument
def calc_expect_func(input_x, input_y, output2, dim):
    if input_x["dtype"] == "float16":
        input_x_temp = input_x["value"].astype(np.float32)
        input_y_temp = input_y["value"].astype(np.float32)
    else:
        input_x_temp = input_x["value"]
        input_y_temp = input_y["value"]

    if dim == "none":
        loss_mean = torch.nn.SoftMarginLoss(reduction='none')
        res = loss_mean(torch.from_numpy(input_x_temp), torch.from_numpy(input_y_temp)).numpy()

    elif dim == "sum":
        loss_mean = torch.nn.SoftMarginLoss(reduction='sum')
        res = loss_mean(torch.from_numpy(input_x_temp), torch.from_numpy(input_y_temp)).numpy()
        res = np.reshape(res, (1,))

    else:
        loss_mean = torch.nn.SoftMarginLoss()
        res = loss_mean(torch.from_numpy(input_x_temp), torch.from_numpy(input_y_temp)).numpy()
        res = np.reshape(res, (1,))

    if input_x["dtype"] == "float16":
        res = res.astype(np.float16)
        return [res, ]
    else:
        return [res, ]

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"},
                "none"],
    "case_name": "test_soft_margin_loss_case_1",
    "calc_expect_func": calc_expect_func
})

# ut_case.add_precision_case("Ascend910A", {
#     "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
#                 "param_type": "input"},
#                {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
#                 "param_type": "input"},
#                {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
#                 "param_type": "output"},
#                 "sum"],
#     "case_name": "test_soft_margin_loss_case_2",
#     "calc_expect_func": calc_expect_func
# })

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 32, 16), "shape": (2, 32, 16),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 32, 16), "shape": (2, 32, 16),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"},
                "mean"],
    "case_name": "test_soft_margin_loss_case_3",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"},
                "none"],
    "case_name": "test_soft_margin_loss_case_4",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"},
                "sum"],
    "case_name": "test_soft_margin_loss_case_5",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 32, 16), "shape": (2, 32, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 32, 16), "shape": (2, 32, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"},
                "mean"],
    "case_name": "test_soft_margin_loss_case_6",
    "calc_expect_func": calc_expect_func
})
