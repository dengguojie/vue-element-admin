"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fast_gelu
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
# 'pylint: disable=too-many-locals,unused-variable
@register_operator_compute("FastGelu", op_mode="dynamic", support_fusion=True)
def fast_gelu_compute(input_x, output_y, kernel_name="fast_gelu", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    mathematical formula of fast_gelu(x):
    fast_gelu(x) = xe^(0.851x)(x-|x|)/(1+e^(-1.702|x|))
    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is fast_gelu

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    # const value
    const_1_value = 1
    attr = 1.702
    dtype = input_x.dtype.lower()
    attr_opp = 0 - attr
    attr_half = attr / 2
    check_support_flag = False
    if not (tbe_platform.api_check_support("tbe.dsl.vexp", "float32")) and \
            dtype == "float32":
        check_support_flag = True
        dtype = "float16"
        input_x = tbe.cast_to(input_x, dtype)

    const_0 = tvm.const(attr_opp, dtype)
    const_1 = tvm.const(const_1_value, dtype)
    abs_x = tbe.vabs(input_x)
    mul_abs_x = tbe.vmuls(abs_x, const_0)
    exp_abs_x = tbe.vexp(mul_abs_x)
    div_down = tbe.vadds(exp_abs_x, const_1)

    const_2 = tvm.const(attr_half, dtype)
    pn_x = tbe.vsub(input_x, abs_x)
    mul_pn_x = tbe.vmuls(pn_x, const_2)
    exp_pn_x = tbe.vexp(mul_pn_x)
    div_up = tbe.vmul(input_x, exp_pn_x)

    div_down_rec = tbe.vrec(div_down, impl_mode)
    result = tbe.vmul(div_up, div_down_rec)
    if check_support_flag:
        result = tbe.cast_to(result, "float32")

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
@register_operator("FastGelu")
def fast_gelu(input_x, output_y, kernel_name="fast_gelu", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    mathematical formula of fast_gelu(x):
    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_fast_gelu

    Returns
    -------
    none.
    """
    attr = 1.702
    shape = input_x.get("shape")
    para_check.check_shape(shape, param_name="input_x")

    check_list = ("float16", "float32")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            shape = shape_util.variable_shape([_input_x])
            data = tvm.placeholder(shape[0], name="data", dtype=input_dtype)
            result = fast_gelu_compute(data, output_y, kernel_name, impl_mode)

            tensors.append([data, result])

    with tvm.target.cce():
        schedules = tbe.auto_schedule(result)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
