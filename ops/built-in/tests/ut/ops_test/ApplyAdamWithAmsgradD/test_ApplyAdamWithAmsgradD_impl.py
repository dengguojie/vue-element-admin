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

ApplyAdamWithAmsgradD ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyAdamWithAmsgradD", None, None)

# ============ auto gen  test cases start ===============
ut_case.add_case(["Ascend910", "Ascend310"], {"params": [
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (1,), "format": "ND", "dtype": "float32",
     "ori_shape": (1,), "ori_format": "ND",},
    {"shape": (1,), "format": "ND", "dtype": "float32",
     "ori_shape": (1,), "ori_format": "ND",},
    {"shape": (1,), "format": "ND", "dtype": "float32",
     "ori_shape": (1,), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    0.9,
    0.999,
    1.0,
    False,
    ],
    "expect":
        "success"})

# ============ auto gen test cases end =================

def calc_expect_func(var,
                     m,
                     v,
                     vhat,
                     beta1_power,
                     beta2_power,
                     lr,
                     grad,
                     var_output,
                     m_output,
                     v_output,
                     vhat_output,
                     beta1,
                     beta2,
                     epsilon):

    var_input = var["value"]
    m_input = m["value"]
    v_input = v["value"]
    vhat_input = vhat["value"]
    beta1_power_input = beta1_power["value"]
    beta2_power_input = beta2_power["value"]
    lr_input = lr["value"]
    grad_input = grad["value"]

    lr_t = lr_input * np.sqrt(1 - beta2_power_input) / (1 - beta1_power_input)
    m_t = beta1* m_input + (1 - beta1) * grad_input
    v_t = beta2 * v_input + (1 - beta2) * grad_input * grad_input
    vhat_t = np.maximum(vhat_input, v_t)
    var_t = var_input - lr_t * m_t / (np.sqrt(vhat_t) + epsilon)

    var_t = var_t.astype(var["dtype"])
    m_t = m_t.astype(var["dtype"])
    v_t = v_t.astype(var["dtype"])
    vhat_t = vhat_t.astype(var["dtype"])

    return [var_t, m_t, v_t, vhat_t]

ut_case.add_precision_case("all", {
    "params": [{"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "output"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "output"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "output"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "output"},
               0.9, 0.999, 1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


ut_case.add_precision_case("all", {
    "params": [{"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "output"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "output"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "output"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "output"},
               0.9, 0.999, 1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (27, 9, 7), "format": "ND", "dtype": "float32", "ori_shape": (27, 9, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (27, 9, 7), "format": "ND", "dtype": "float32", "ori_shape": (27, 9, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (27, 9, 7), "format": "ND", "dtype": "float32", "ori_shape": (27, 9, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (27, 9, 7), "format": "ND", "dtype": "float32", "ori_shape": (27, 9, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (27, 9, 7), "format": "ND", "dtype": "float32", "ori_shape": (27, 9, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (27, 9, 7), "format": "ND", "dtype": "float32", "ori_shape": (27, 9, 7), "ori_format": "ND", "param_type": "output"},
               {"shape": (27, 9, 7), "format": "ND", "dtype": "float32", "ori_shape": (27, 9, 7), "ori_format": "ND", "param_type": "output"},
               {"shape": (27, 9, 7), "format": "ND", "dtype": "float32", "ori_shape": (27, 9, 7), "ori_format": "ND", "param_type": "output"},
               {"shape": (27, 9, 7), "format": "ND", "dtype": "float32", "ori_shape": (27, 9, 7), "ori_format": "ND", "param_type": "output"},
               0.9, 0.999, 1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (17, 7), "format": "ND", "dtype": "float32", "ori_shape": (17, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (17, 7), "format": "ND", "dtype": "float32", "ori_shape": (17, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (17, 7), "format": "ND", "dtype": "float32", "ori_shape": (17, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (17, 7), "format": "ND", "dtype": "float32", "ori_shape": (17, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (17, 7), "format": "ND", "dtype": "float32", "ori_shape": (17, 7), "ori_format": "ND", "param_type": "input"},
               {"shape": (17, 7), "format": "ND", "dtype": "float32", "ori_shape": (17, 7), "ori_format": "ND", "param_type": "output"},
               {"shape": (17, 7), "format": "ND", "dtype": "float32", "ori_shape": (17, 7), "ori_format": "ND", "param_type": "output"},
               {"shape": (17, 7), "format": "ND", "dtype": "float32", "ori_shape": (17, 7), "ori_format": "ND", "param_type": "output"},
               {"shape": (17, 7), "format": "ND", "dtype": "float32", "ori_shape": (17, 7), "ori_format": "ND", "param_type": "output"},
               0.9, 0.999, 1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
