"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

leaky_relu_grad
"""

import te.lang.cce as tbe
from te import tvm
import te.lang.base as tbe_base
from te.utils import shape_util
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.util.platform_adapter import register_operator

# define a scalar , value = 0
SCALAR_ZERO = 0
# define a scalar , value = -1
NEGATIVE_ONE = -1


# pylint: disable=unused-argument,invalid-name,too-many-locals
# pylint: disable=redefined-argument-from-local
def leaky_relu_grad_compute(g, x, y, negative_slope=0,
                            kernel_name="leaky_relu_grad"):
    """
    calculate the backpropagation of leaky_relu operation
    y = gradients(x>0) or negative_slope*gradients(x<=0).

    Parameters
    ----------
    g : TVM tensor
        the placeholder of input g
    x : TVM tensor
        the placeholder of input x
    y : dict
        dict of output y, include keys(shape and dtype)
    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization
    kernel_name : str
        kernel name, default value is "leaky_relu_grad"

    Returns
    -------
    res: TVM tensor
        the result of leaky_relu_grad_compute
    """

    shape_list = shape_util.broadcast_shapes(shape_util.shape_to_list(g.shape),
                                  shape_util.shape_to_list(x.shape))
    dtype = g.dtype
    g = tbe.broadcast(g, shape_list[2])
    x = tbe.broadcast(x, shape_list[2])

    if dtype == "float32":
        help_min = tvm.const(2 ** (-126), "float32")
        help_rec_one = tvm.const(2 ** 38, "float32")
        help_rec_sec = tvm.const(2 ** 44, "float32")
    elif dtype == "float16":
        help_min = tvm.const(2 ** (-24), "float16")
        help_rec_one = tvm.const(2 ** 12, "float16")
        help_rec_sec = help_rec_one

    tmp_min_x = tbe.vmins(x, help_min)
    tmp_max_x = tbe.vmaxs(tmp_min_x,
                                      tvm.const(SCALAR_ZERO, "float32"))
    tmp_mul_x = tbe.vmuls(tmp_max_x, help_rec_one)

    if dtype == "float32":
        tmp_mul_x = tbe.vmuls(tmp_mul_x, help_rec_sec)

    result_tmp_right = tbe.vmuls(tmp_mul_x, help_rec_sec)

    result_sub = tbe.vadds(result_tmp_right,
                                       tvm.const(NEGATIVE_ONE, "float32"))
    result_abs = tbe.vabs(result_sub)
    result_tmp_left = tbe.vmuls(result_abs, negative_slope)

    result_tmp = tbe.vadd(result_tmp_left, result_tmp_right)

    res = tbe.vmul(g, result_tmp)
    return res


@register_operator("LeakyReluGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT), para_check.KERNEL_NAME)
def leaky_relu_grad(g, x, y, negative_slope=0, kernel_name="leaky_relu_grad"):
    """
    calculate the backpropagation of leaky_relu operation
    y = gradients(x>0) or negative_slope*gradients(x<=0).
    support dtype:float16,float32

    Parameters
    ----------
    g : dict
        the backpropagated gradients to the corresponding leaky_relu operation
    x : dict
        the x passed as output of leaky_relu operation
    y : dict
        the output of leaky_relu back propagation
    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization
    kernel_name : str
        kernel name, default value is "leaky_relu_grad"

    Returns
    -------
    None
    """
    g_dtype = g.get("dtype").lower()
    x_dtype = x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(g_dtype, check_list, param_name="input_g")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_elewise_shape_range([g, x], support_broadcast=True)
    if g_dtype != x_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "g", "x",
                                                              g_dtype, x_dtype)
    ins = classify([g, x], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (g, x) in ins:
        with tbe_base.compute():
            g_shape, x_shape = shape_util.variable_shape([g, x], support_broadcast=True)
            g_shape, x_shape = shape_util.refine_shapes_for_broadcast(g_shape, x_shape)
            tensor_g = tvm.placeholder(g_shape, g_dtype, "tensor_g")
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            res = leaky_relu_grad_compute(tensor_g, tensor_x, y, negative_slope,
                                          kernel_name)
            tensors.append((tensor_g, tensor_x, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
