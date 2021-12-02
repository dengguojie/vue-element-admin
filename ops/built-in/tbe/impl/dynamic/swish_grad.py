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
dynamic swish_grad
`y = sigmoid(scale*x) + x*sigmoid'(scale*x)`
`sigmoid = sigmoid*(1 - sigmoid)`
let:
`A = fwd_input = x`                   # input of swish forward
`B = fwd_output = x*sigmoid(scale*x)` # output of swish forward
`y = scale*B + B/A*(1 - scale*B)`
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=invalid-name,too-many-locals,redefined-argument-from-local
def swish_grad_compute(input_gradients, fwd_input, fwd_output, bkwd_output, beta, kernel_name="swish_grad"):
    """
    algorithm : swish grad compute
    let:
    A = fwd_input = x                   # input of swish forward
    B = fwd_output = x*sigmoid(scale*x) # output of swish forward
    then,
    swish_grad = scale*B + B/A*(1 - scale*B)
    Parameters:
    ----------
    input_gradients : dictionary of gradient
    fwd_input : dictionary of swish input
    fwd_output : dictionary of swish output
    bkwd_output: dictionary of output
    beta: scale for exponent in sigmoid
    kernel_name : default value is "swish_grad"
    Returns
    -------
    a tenosr
    """
    dtype = fwd_input.dtype.lower()
    if dtype == "float16":
        fwd_input = tbe.cast_to(fwd_input, "float32")
        fwd_output = tbe.cast_to(fwd_output, "float32")
        input_gradients = tbe.cast_to(input_gradients, "float32")
    # calculate B/A
    sigmoid_value = tbe.vdiv(fwd_output, fwd_input)
    # calculate 1-beta*B
    one_tensor = tbe.broadcast(tvm.const(1, dtype="float32"), fwd_output.shape)
    beta_output = tbe.vmuls(fwd_output, tvm.const(beta, dtype="float32"))
    one_minus_b = tbe.vsub(one_tensor, beta_output)
    # calculate B/A*(1 - scale*B)
    swish_part = tbe.vmul(sigmoid_value, one_minus_b)
    # calculate scale*B + B/A*(1 - scale*B)
    grad_x = tbe.vadd(beta_output, swish_part)
    if dtype == "float16":
        res = tbe.vmul(input_gradients, grad_x)
        return tbe.cast_to(res, "float16")
    return tbe.vmul(input_gradients, grad_x)


def check_op_dtype(dtype_input, dtype_x0, dtype_x1):
    """
    check dtypes
    :param dtype_input: str
    :param dtype_x0: str
    :param dtype_x1: str
    :return: none
    """
    if dtype_input != dtype_x0:
        error_manager_vector.raise_err_two_input_dtype_invalid('swish_grad', "input_data", "x0",
                                                               "the dtype of input_data, x0, must be the same")
    if dtype_input != dtype_x1:
        error_manager_vector.raise_err_two_input_dtype_invalid('swish_grad', "input_data", "x1",
                                                               "the dtype of input_data, x1, must be the same")
    check_list = ["float16", "float32"]
    para_check.check_dtype(dtype_input, check_list)


@register_operator("SwishGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def swish_grad(input_data, x0, x1, output_y, scale=1.0, kernel_name="swish_grad"):
    """
    do swish grad

    let:
    A = fwd_input = x                   # input of swish forward
    B = fwd_output = x*sigmoid(scale*x) # output of swish forward
    then,
    swish_grad = scale*B + B/A*(1 - scale*B)
    Parameters:
    ----------
    input_data : dictionary of gradient
    x0 : dictionary of swish input
    x1 : dictionary of swish output
    y: dictionary of output
    scale: scale for exponent in sigmoid
    kernel_name : default value is "swish_grad"
    Returns
    -------
    None
    """
    shape_input = input_data.get("shape")
    shape_x0 = x0.get("shape")
    shape_x1 = x1.get("shape")
    dtype_input = input_data.get("dtype").lower()
    dtype_x0 = x0.get("dtype").lower()
    dtype_x1 = x1.get("dtype").lower()
    para_check.check_elewise_shape_range(
        [input_data, x0, x1], support_broadcast=True)

    para_check.check_kernel_name(kernel_name)
    check_op_dtype(dtype_input, dtype_x0, dtype_x1)

    ins = classify([input_data, x0, x1], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_input_data, _x0, _x1) in ins:
        with tbe.compute():
            shape_input, shape_x0, shape_x1 = shape_util.variable_shape(
                [_input_data, _x0, _x1])
            data_input = tvm.placeholder(
                shape_input, dtype=dtype_input, name="data_input")
            data_x0 = tvm.placeholder(shape_x0, dtype=dtype_x0, name="data_x0")
            data_x1 = tvm.placeholder(shape_x1, dtype=dtype_x1, name="data_x1")
            res = swish_grad_compute(
                data_input, data_x0, data_x1, output_y, scale, kernel_name)
            input_list = [data_input, data_x0, data_x1, res]
            tensors.append(input_list)

        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
