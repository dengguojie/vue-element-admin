#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
static test
'''
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info


ut_case = OpUT("ApplyAdaMaxD", None, None)

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
    }, {
        "shape": (1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND"
    }],
    "case_name": "apply_ada_max_d_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": [{
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (2, 4),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
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
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }],
    "case_name": "apply_ada_max_d_2",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}

case3 = {
    "params": [{
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 2),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 2),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 4),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 4),
        "ori_format": "ND"
    }, {
        "shape": (1, 5),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 5),
        "ori_format": "ND"
    }, {
        "shape": (1, 6),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 6),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 3),
        "ori_format": "ND"
    }],
    "case_name": "apply_ada_max_d_3",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
def _gen_outputs(input_var, input_m, input_v, input_beta1_power, input_lr, input_beta1, input_beta2, input_epsilon,
                 input_grad, dtype):
    '''
    _gen_outputs
    '''
    if dtype == 'float16':
        input_var = input_var.astype('float32')
        input_m = input_m.astype('float32')
        input_v = input_v.astype('float32')
        input_beta1_power = input_beta1_power.astype('float32')
        input_lr = input_lr.astype('float32')
        input_beta1 = input_beta1.astype('float32')
        input_beta2 = input_beta2.astype('float32')
        input_epsilon = input_epsilon.astype('float32')
        input_grad = input_grad.astype('float32')

    # m.device(d) += (grad - m) * (T(1) - beta1())
    output_m = input_m + (input_grad - input_m) * (1 - input_beta1)
    # v.device(d) = (beta2() * v).cwiseMax(grad.abs())
    output_v = np.maximum(input_beta2 * input_v, np.abs(input_grad))
    # var.device(d) -= lr() / (T(1) - beta1_power()) * (m / (v + epsilon()))
    output_var = input_var - input_lr / (1 - input_beta1_power) * (output_m / (output_v + input_epsilon))

    if dtype == 'float16':
        output_m = output_m.astype(dtype)
        output_v = output_v.astype(dtype)
        output_var = output_var.astype(dtype)
    return output_var, output_m, output_v


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
def calc_expect_func(x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3):
    '''
    calc_expect_func
    '''
    res1, res2, res3 = _gen_outputs(x1['value'], x2['value'], x3['value'], x4['value'], x5['value'], x6['value'],
                                    x7['value'], x8['value'], x9['value'], x1['dtype'])
    return res1, res2, res3


precision_case1 = {
    "params": [
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "output"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "output"
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "output"
        },
    ],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
}

ut_case.add_precision_case("Ascend910", precision_case1)
