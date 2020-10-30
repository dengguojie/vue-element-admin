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

ApplyKerasMomentumD ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyKerasMomentumD", None, None)

# ============ auto gen  test cases start ===============
ut_case.add_case(["Ascend910", "Ascend310"], {"params": [
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (1,), "format": "ND", "dtype": "float32",
     "ori_shape": (1,), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (1,), "format": "ND", "dtype": "float32",
     "ori_shape": (1,), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    {"shape": (16, 32), "format": "ND", "dtype": "float32",
     "ori_shape": (16, 32), "ori_format": "ND",},
    False,
    True],
    "expect":
        "success"})

# ============ auto gen test cases end =================

def calc_expect_func(var,
                     accum,
                     lr,
                     grad,
                     momentum,
                     out_var,
                     out_accum,
                     use_locking,
                     use_nesterov):

    data_var = var["value"]
    data_accum = accum["value"]
    data_lr = lr["value"]
    data_grad = grad["value"]
    data_momentum = momentum["value"]

    if var["dtype"] == "float16":
        data_var = data_var.astype(np.float32)
        data_accum = data_accum.astype(np.float32)
        data_lr = data_lr.astype(np.float32)
        data_grad = data_grad.astype(np.float32)
        data_momentum = data_momentum.astype(np.float32)

    output_accum = (data_accum * data_momentum) - (data_grad * data_lr)
    if use_nesterov == True:
        output_var = data_var + \
                     (output_accum * data_momentum - data_grad * data_lr)
    else:
        output_var = data_var + output_accum

    return [output_var.astype(var["dtype"]), output_accum.astype(var["dtype"])]

ut_case.add_precision_case("all", {
    "params": [{"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "output"},
               {"shape": (16, 32), "format": "ND", "dtype": "float32", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "output"},
               False, True],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (17, 9, 7, 19), "format": "ND", "dtype": "float32", "ori_shape": (17, 9, 7, 19), "ori_format": "ND", "param_type": "input"},
               {"shape": (17, 9, 7, 19), "format": "ND", "dtype": "float32", "ori_shape": (17, 9, 7, 19), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (17, 9, 7, 19), "format": "ND", "dtype": "float32", "ori_shape": (17, 9, 7, 19), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (17, 9, 7, 19), "format": "ND", "dtype": "float32", "ori_shape": (17, 9, 7, 19), "ori_format": "ND", "param_type": "output"},
               {"shape": (17, 9, 7, 19), "format": "ND", "dtype": "float32", "ori_shape": (17, 9, 7, 19), "ori_format": "ND", "param_type": "output"},
               False, True],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (2, 3, 4, 5, 6, 1, 9), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5, 6, 1, 9), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5, 6, 1, 9), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5, 6, 1, 9), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5, 6, 1, 9), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5, 6, 1, 9), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5, 6, 1, 9), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5, 6, 1, 9), "ori_format": "ND", "param_type": "output"},
               {"shape": (2, 3, 4, 5, 6, 1, 9), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5, 6, 1, 9), "ori_format": "ND", "param_type": "output"},
               False, True],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "input"},
               {"shape": (1,), "format": "ND", "dtype": "float32", "ori_shape": (1,), "ori_format": "ND", "param_type": "input"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "output"},
               {"shape": (2, 3, 4, 5), "format": "ND", "dtype": "float32", "ori_shape": (2, 3, 4, 5), "ori_format": "ND", "param_type": "output"},
               False, True],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()