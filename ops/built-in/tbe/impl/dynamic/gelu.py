#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
gelu
"""
import functools
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator

CSVALUE = tvm.const(0.044715, "float32")


# pylint: disable=too-many-locals
def _tanh_parameter_compute(placeholders):
    mul_0 = tbe.vmul(placeholders, placeholders)
    pow_0 = tbe.vmul(mul_0, placeholders)
    mul_1 = tbe.vmuls(pow_0, CSVALUE)
    result = tbe.vadd(placeholders, mul_1)

    return result


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
@register_operator_compute("Gelu", op_mode="dynamic", support_fusion=True)
def gelu_compute(input_x, output_y, kernel_name="gelu"):
    """
    mathematical formula of gelu(x):
    gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    tanh(y) = 2/(1+exp(-2y)) - 1
    convert gelu to result(x) =
      x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is gelu

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    dtype = input_x.dtype
    has_improve_precision = False

    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        has_improve_precision = True
        input_x = tbe.cast_to(input_x, "float32")
        dtype = input_x.dtype
        
    # formula; gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    # formula; tanh(y) = 2/(1+exp(-2y)) - 1

    # simplize
    # formula; gelu(x) = x/(1+e^(-y))
    # the formula is y = 2*np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))

    # to avoid overflow, keep exp negative
    # formula; gelu(x) = x/(1+e^(-|y|)) * v_const
    # the formula is y = 2*np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))
    # formula; v_const = 1 if x > 0  , e^y  if x < 0
    # formula; 2*np.sqrt(2/np.pi)
    const_0 = tvm.const(1.5957691, "float32")
    const_1 = tvm.const(1.0, "float32")
    const_2 = tvm.const(-1.0, "float32")
    const_3 = tvm.const(0.0, "float32")
    tanh_parameter = _tanh_parameter_compute(input_x)
    mul_0 = tbe.vmuls(tanh_parameter, const_0)
    # y
    mul_0_min = tbe.vmins(mul_0, const_3)
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and dtype == "float32":
        mul_0_min_fp16 = tbe.cast_to(mul_0_min, "float16")
        right_mul = tbe.vexp(mul_0_min_fp16)
    else:
        right_mul = tbe.vexp(mul_0_min)
    right_mul_fp32 = tbe.cast_to(right_mul, dtype)

    # abs(y)
    mul_0_abs = tbe.vabs(mul_0)
    # -abs(y)
    mul_0_abs_neg = tbe.vmuls(mul_0_abs, const_2)

    # the formula is e^(-abs(y))
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and dtype == "float32":
        mul_0_abs_neg_fp16 = tbe.cast_to(mul_0_abs_neg, "float16")
        mul_0_abs_neg_exp = tbe.vexp(mul_0_abs_neg_fp16)
    else:
        mul_0_abs_neg_exp = tbe.vexp(mul_0_abs_neg)
    mul_0_abs_neg_exp_fp32 = tbe.cast_to(mul_0_abs_neg_exp, dtype)

    # the formula is e^(-abs(y)) + 1
    mul_0_abs_neg_exp_add = tbe.vadds(mul_0_abs_neg_exp_fp32, const_1)
    left_mul = tbe.vdiv(input_x, mul_0_abs_neg_exp_add)

    result = tbe.vmul(left_mul, right_mul_fp32)

    if has_improve_precision:
        result = tbe.cast_to(result, "float16")

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
@register_operator("Gelu")
def gelu(x, y, kernel_name="gelu"):
    """
    mathematical formula of gelu(x):
    gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    tanh(y) = 2/(1+exp(-2y)) - 1
    convert gelu to result(x) =
     x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support float16, float32
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is gelu

    Returns
    -------
    None.
    """
    dtype_x = x.get("dtype").lower()
    check_list = ("float16", "float32",)
    para_check.check_dtype(dtype_x, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x1,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([x1])

            input_data = tvm.placeholder(shape_x[0], name="input_data",
                                         dtype=dtype_x)
            res = gelu_compute(input_data, y, kernel_name)

            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
