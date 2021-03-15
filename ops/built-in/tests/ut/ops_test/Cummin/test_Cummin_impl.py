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

ut_case = BroadcastOpUT("cummin")

#pylint: disable=unused-argument
def calc_expect_func(input, output1, output2, dim):
    if input["dtype"] == "float16":
        input_x_tmp = input["value"].astype(np.float32)
        res=torch.cummin(torch.from_numpy(input_x_tmp), dim)
        res1 = res[0].numpy().astype(np.float16)
        res2 = res[1].numpy().astype(np.float16)
        return [res1, res2]
    res=torch.cummin(torch.from_numpy(input["value"]), dim)
    res1 = res[0].numpy()
    res2 = res[1].numpy()
    return [res1, res2]

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"}, 0],
    "case_name": "test_cummin_case_1",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 8), "shape": (32, 8),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 8), "shape": (32, 8),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 8), "shape": (32, 8),
                "param_type": "output"}, 0],
    "case_name": "test_cummin_case_2",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 32, 16), "shape": (2, 32, 16),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 32, 16), "shape": (2, 32, 16),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 32, 16), "shape": (2, 32, 16),
                "param_type": "output"}, 0],
    "case_name": "test_cummin_case_3",
    "calc_expect_func": calc_expect_func
})
