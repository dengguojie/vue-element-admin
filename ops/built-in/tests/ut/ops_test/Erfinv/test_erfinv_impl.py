# # -*- coding:utf-8 -*-
import sys
import torch
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("erfinv")

#pylint: disable=unused-argument
def calc_expect_func(input_x, output_y):
    if input_x["dtype"] == "float16":
        input_x_temp = input_x["value"].astype(np.float32)
    else:
        input_x_temp = input_x["value"]
    x_tensor = torch.from_numpy(input_x_temp)
    res = torch.erfinv(x_tensor)
    res = res.numpy()
    if input_x["dtype"] == "float16":
        res = res.astype(np.float16)
        return [res, ]
    else:
        return [res, ]

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (4, 2),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (4, 2),
                "param_type": "output"}],
    "case_name": "test_is_erfinv_case_1",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.002, 0.002)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (4, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (4, 2),
                "param_type": "output"}],
    "case_name": "test_is_erfinv_case_2",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.002, 0.002),
})
