import numpy as np
import torch
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("elu_grad_v2")

# [TODO] coding expect function here
#pylint: disable=unused-argument
def calc_expect_func(input_x1, input_x2, output_z, alpha):
    flag = 0
    if input_x1["dtype"] == np.float16:
        flag = 1
        input_x1_tmp = input_x1["value"].astype(np.float32)
    else:
        input_x1_tmp = input_x1["value"]
    if input_x2["dtype"] == np.float16:
        flag = 1
        input_x2_tmp = input_x2["value"].astype(np.float32)
    else:
        input_x2_tmp = input_x2["value"]
    input1 = torch.from_numpy(input_x1_tmp).type(torch.float32)
    input2 = torch.from_numpy(input_x2_tmp).type(torch.float32)

    border = torch.zeros_like(input1)
    min_result = border.min(input2)
    one_tensor = torch.ones_like(min_result)
    add_result = torch.where(min_result >= 0.0, one_tensor, min_result + alpha)

    result = torch.mul(input1, add_result)
    result = result.numpy()

    if flag == 1:
        result.astype(np.float16)
    return result

# [TODO] coding cases here
ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "input","range_value":[0.1,2.0]},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "input","range_value":[0.1,2.0]},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "output"},1.0],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "input","range_value":[0.1,2.0]},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "input","range_value":[0.1,2.0]},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,100), "shape": (100,100),
                "param_type": "output"},1.0],
    "calc_expect_func": calc_expect_func
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input"},
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input"},
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output"}, 1.0],
    "calc_expect_func": calc_expect_func,
    "expect": "success"
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input"},
               {"shape": (100, 100), "ori_shape": (100, 100),"format": "ND", "ori_format": "ND","dtype": "float16",
                "param_type": "input"},
               {"shape": (100, 100),"ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "output"}, 1.0],
    "calc_expect_func": calc_expect_func,
    "expect": "success"
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (100, 100),"ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input"},
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input" },
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output"}, -1.0],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input"},
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input"}, 
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output"}, 1.0],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (100,100), "ori_shape": (100,100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input"},
               {"shape": (10,100), "ori_shape": (10,100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input"},
               {"shape": (100, 100), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output"}, 1.0],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})