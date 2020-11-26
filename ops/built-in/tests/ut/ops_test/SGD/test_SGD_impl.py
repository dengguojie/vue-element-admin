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

SGD ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("SGD",
               "impl.sgd",
               "sgd")

case_small_shape_scalar_fp32 = {
    "params":
        [
            {
                "shape": (1, ),  # parameters
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # gradient
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # learning_rate
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1,),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),   # momentum
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),  # stat
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # parameters
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            # dampening,weight_decay,nesterov
            0.0,
            0.0,
            False
        ],
    "case_name": 'test_sgd_small_shape_scalar_fp32',
    "expect": "success"
}

case_medium_shape_fp32 = {
    "params":
        [
            {
                "shape": (9973, 13, 8297),  # parameters
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8297),  # gradient
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # learning_rate
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 13, 8297),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),   # momentum
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 13, 8297),  # stat
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8297),  # parameters
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            # dampening,weight_decay,nesterov
            0.0,
            0.0,
            False
        ],
    "case_name": 'test_sgd_small_shape_fp32',
    "expect": "success"
}

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_scalar_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_medium_shape_fp32)


def calc_expect_func(parameters, gradient, learning_rate, accum, momentum,
                     stat, update, dampening, weight_decay, nesterov):
    dtype = parameters["dtype"]
    input_var = parameters["value"]
    input_grad = gradient["value"]
    input_lr = learning_rate["value"]
    input_accum = accum["value"]
    input_momentum = momentum["value"]
    input_stat = stat["value"]

    if dtype == "float16":
        input_var = input_var.astype(np.float32)
        input_accum = input_accum.astype(np.float32)
        input_lr = input_lr.astype(np.float32)
        input_grad = input_grad.astype(np.float32)
        input_momentum = input_momentum.astype(np.float32)
        input_stat = input_stat.astype(np.float32)

    input_grad = input_grad + input_var*weight_decay
    # Calc output
    if input_stat.any() != 0.0:
        output_accum = input_accum * input_momentum + input_grad
    else:
        output_accum = input_accum * input_momentum + input_grad*(1-dampening)

    if nesterov is True:
        output_var = input_var - (input_grad * input_lr + output_accum * input_momentum * input_lr)
    else:
        output_var = input_var - output_accum * input_lr

    output_data = output_var.astype(np.float32)
    output_stat = input_stat * 0
    if dtype == "float16":
        output_accum = output_accum.astype(dtype)
        output_var = output_var.astype(dtype)
        output_data = output_data.astype(dtype)
        output_stat = output_stat.astype(dtype)

    return output_data

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 2), "shape": (33, 2), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 2), "shape": (33, 2), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 2), "shape": (33, 2), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 2), "shape": (33, 2), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 2), "shape": (33, 2), "param_type": "output"},
               0.0, 0.0, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 33, 2), "shape": (7, 7, 33, 2), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 33, 2), "shape": (7, 7, 33, 2), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 33, 2), "shape": (7, 7, 33, 2), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 33, 2), "shape": (7, 7, 33, 2), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 33, 2), "shape": (7, 7, 33, 2), "param_type": "output"},
               0.0, 0.0, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (777, 1), "shape": (777, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (777, 1), "shape": (777, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (777, 1), "shape": (777, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (777, 1), "shape": (777, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (777, 1), "shape": (777, 1), "param_type": "output"},
               0.0, 0.0, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33, 77, 1), "shape": (11, 33, 77, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33, 77, 1), "shape": (11, 33, 77, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33, 77, 1), "shape": (11, 33, 77, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33, 77, 1), "shape": (11, 33, 77, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33, 77, 1), "shape": (11, 33, 77, 1), "param_type": "output"},
               0.0, 0.0, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
