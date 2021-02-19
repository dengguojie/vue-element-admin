# # -*- coding:utf-8 -*-
import sys
from op_test_frame.common import precision_info
from op_test_frame.ut import BroadcastOpUT
import numpy as np
import torch
ut_case = BroadcastOpUT("hard_shrink_grad")
# pylint: disable=unused-argument,too-many-locals,invalid-name
def calc_expect_func(input_x1, input_x2, output_z, lambda_value):
    if input_x1["dtype"] == np.float16:
        input_x1_tmp = input_x1["value"].astype(np.float32)
    else:
        input_x1_tmp = input_x1["value"]
    if input_x2["dtype"] == np.float16:
        input_x2_tmp = input_x2["value"].astype(np.float32)
    else:
        input_x2_tmp = input_x2["value"]
    input1 = torch.from_numpy(input_x1_tmp).type(torch.float32)
    input2 = torch.from_numpy(input_x2_tmp).type(torch.float32)
    zero_tensor = torch.full_like(input2,0)
    one_tensor = torch.full_like(input2,1)
    lambda_tensor = torch.full_like(input2,lambda_value)
    ratio = torch.where(torch.abs(input2) > lambda_tensor,one_tensor,zero_tensor)
    result = torch.mul(input1, ratio)
    result = result.numpy()
    if input_x2["dtype"] == np.float16:
        result.astype(np.float16)
    return result

# [TODO] coding cases here
ut_case.add_precision_case("Ascend310", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "output"},0.5],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
   "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
               "param_type": "input"},
              {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
               "param_type": "input"},
              {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
               "param_type": "output"},0.5],
   "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend310", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "output"},0.5],
    "precision_standard": precision_info.PrecisionStandard(0.0005, 0.0005),
    "calc_expect_func": calc_expect_func
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input"},
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input"},
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "output"}, 0.5],
    "calc_expect_func": calc_expect_func,
    "expect": "success"
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input"}, 
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input"},
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND",  "ori_format": "ND",  "dtype": "float16",
                "param_type": "output"}, -0.5],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input"},
               {"shape": (100, 100,100), "ori_shape": (100, 100,100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input"}, 
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "output"}, 0.5],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})
