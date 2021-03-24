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

ApplyAdagradV2D ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("ApplyAdagradV2D", None, None)

case1 = {
    "params": [{
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, 0.0001],
    "case_name": "apply_adagrad_d_1",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case2 = {
    "params": [{
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, 0.0001],
    "case_name": "apply_adagrad_d_2",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case3 = {
    "params": [{
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, 0.0001],
    "case_name": "apply_adagrad_d_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case4 = {
    "params": [{
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (2,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, 0.0001],
    "case_name": "apply_adagrad_d_4",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case5 = {
    "params": [{
        "shape": (0,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (0,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (0,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (0,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, {
        "shape": (0,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }, 0.0001],
    "case_name": "apply_adagrad_d_5",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}


# pylint: disable=unbalanced-tuple-unpacking,too-many-arguments,too-many-locals,unused-argument
def calc_expect_func(var, accum, lr, grad, out_var, out_accum, epsilon):
    '''
    calc_expect_func
    '''
    input_var = var["value"]
    input_accum = accum["value"]
    input_lr = lr["value"]
    input_epsilon = epsilon
    input_grad = grad["value"]

    if var["dtype"] == "float16":
        input_var = input_var.astype(np.float32)
        input_accum = input_accum.astype(np.float32)
        input_lr = input_lr.astype(np.float32)
        input_epsilon = input_epsilon.astype(np.float32)
        input_grad = input_grad.astype(np.float32)

    update_slot = True
    if update_slot is True:
        grad_square = input_grad * input_grad
        accum = input_accum + grad_square
    else:
        accum = input_accum
    lr_grad = input_lr * input_grad
    sqrt_accum = np.sqrt(accum)
    sqrt_accum_epsilon = sqrt_accum + input_epsilon
    update = lr_grad / sqrt_accum_epsilon
    out_var = input_var - update
    if var["dtype"] == "float16":
        out_var = out_var.astype(np.float16)
        accum = accum.astype(np.float16)
    return [out_var, accum]


ut_case.add_precision_case(
    "Ascend910", {
        "params": [{
            "shape": (273,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (273,),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (273,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (273,),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (1,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (273,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (273,),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (273,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (273,),
            "ori_format": "ND",
            "param_type": "output"
        }, {
            "shape": (273,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (273,),
            "ori_format": "ND",
            "param_type": "output"
        }, 0.0001],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910", {
        "params": [{
            "shape": (1, 1, 16, 16),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1, 1, 16, 16),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (1, 1, 16, 16),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1, 1, 16, 16),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (1,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (1, 1, 16, 16),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1, 1, 16, 16),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (1, 1, 16, 16),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1, 1, 16, 16),
            "ori_format": "ND",
            "param_type": "output"
        }, {
            "shape": (1, 1, 16, 16),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1, 1, 16, 16),
            "ori_format": "ND",
            "param_type": "output"
        }, 0.0001],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910", {
        "params": [{
            "shape": (17, 9, 7, 19),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (17, 9, 7, 19),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (17, 9, 7, 19),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (17, 9, 7, 19),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (1,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (17, 9, 7, 19),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (17, 9, 7, 19),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (17, 9, 7, 19),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (17, 9, 7, 19),
            "ori_format": "ND",
            "param_type": "output"
        }, {
            "shape": (17, 9, 7, 19),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (17, 9, 7, 19),
            "ori_format": "ND",
            "param_type": "output"
        }, 0.0001],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910", {
        "params": [{
            "shape": (64, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (64, 128),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (64, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (64, 128),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (1,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (64, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (64, 128),
            "ori_format": "ND",
            "param_type": "input"
        }, {
            "shape": (64, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (64, 128),
            "ori_format": "ND",
            "param_type": "output"
        }, {
            "shape": (64, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (64, 128),
            "ori_format": "ND",
            "param_type": "output"
        }, 0.0001],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_case(["Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend710", "Ascend910"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910")
