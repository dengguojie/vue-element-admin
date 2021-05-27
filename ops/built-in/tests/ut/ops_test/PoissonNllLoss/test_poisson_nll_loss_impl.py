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

poisson_nll_loss ut case
"""
from op_test_frame.ut import BroadcastOpUT
import torch
import numpy as np
ut_case = BroadcastOpUT("poisson_nll_loss")

#[TODO] coding expect function here
#pylint: disable=unused-argument
def calc_expect_func(input_x, input_y, output_z, log_input, full, eps, reduction):
    if input_x["value"].dtype == np.float16:
        input_x_tmp = input_x["value"].astype(np.float32)
    else:
        input_x_tmp = input_x["value"]
    if input_y["value"].dtype == np.float16:
        input_y_tmp = input_y["value"].astype(np.float32)
    else:
        input_y_tmp = input_y["value"]
    m_sum = torch.nn.PoissonNLLLoss(log_input = log_input, full = full, eps = eps, reduction = reduction)
    res = m_sum(torch.from_numpy(input_x_tmp), torch.from_numpy(input_y_tmp))
    if input_x["value"].dtype == np.float16:
        retvalue = torch.reshape(res,(1,)).numpy()
        retvalue = retvalue.astype(np.float16)
    else:
        retvalue = torch.reshape(res,(1,)).numpy()
    return retvalue

def calc_expect_func_b(input_x, input_y, output_z, log_input, full, eps, reduction):
    if input_x["value"].dtype == np.float16:
        input_x_tmp = input_x["value"].astype(np.float32)
    else:
        input_x_tmp = input_x["value"]
    if input_y["value"].dtype == np.float16:
        input_y_tmp = input_y["value"].astype(np.float32)
    else:
        input_y_tmp = input_y["value"]
    m_sum = torch.nn.PoissonNLLLoss(log_input = log_input, full = full, eps = eps, reduction = reduction)
    res = m_sum(torch.from_numpy(input_x_tmp), torch.from_numpy(input_y_tmp))
    retvalue = torch.reshape(res,(1,)).numpy()
    return retvalue.astype(np.float16)

#[TODO] coding cases here
ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"}, False, True, 1e-8, "sum"],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
   "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
               "param_type": "input"},
              {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
               "param_type": "input"},
              {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
               "param_type": "output"}, False, True, 1e-8, "mean"],
   "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"}, False, True, 1e-8, "mean"],
    "calc_expect_func": calc_expect_func
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    }],
    "expect": "success"
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,1),
        "ori_shape": (1,1),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    }],
    "case_name":"error1",
    "expect": RuntimeError
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    }],
    "expect": RuntimeError
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    },
    True,False,1e-8, "xxx"],
    "expect": RuntimeError
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    },
        True, False, 0.0],
    "case_name":"eps0",
    "expect": RuntimeError
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    },
    False, True, 1e-8, "none"],
    "expect": "success"
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    },
    False, True, 1e-8, "sum"],
    "expect": "success"
})
