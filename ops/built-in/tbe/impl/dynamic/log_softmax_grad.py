# Copyright 2020 Huawei Technologies Co., Ltd
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
dynamic logsoftmax_grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator("LogSoftmaxGrad")
def log_softmax_grad_compute(input_dy, input_x, output_z, axis,
                             kernel_name="log_softmax_grad"):
    """
    TVM calculation process, used for fusion operation.
        dy - (exp(x) * sum(dy))

    Parameters
    ----------
    input_dy: TVM tensor
        the placeholder of input grad data
    input_x: TVM tensor
        the placeholder of input data
    output_z: dict
        shape and dtype of output, should be the same shape and type as input
    axis: int, list or tuple .
        the first axis to reduce, may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        cce kernel name, default value is log_softmax_grad

    Returns
    -------
    result: TVM tensor.
    """
    dtype = input_dy.dtype
    shape1 = shape_util.shape_to_list(input_dy.shape)
    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp",
                                           "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        input_dy = tbe.cast_to(input_dy, "float32")
        has_improve_precision = True

    data_exp = tbe.vexp(input_x)
    data_sum = tbe.reduce_sum(input_dy, axis, True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape1)
    data_softmax = tbe.vmul(data_exp, data_sum_broadcast)

    result = tbe.vsub(input_dy, data_softmax)
    if has_improve_precision:
        result = tbe.cast_to(result, "float16")

    return result


# 'pylint: disable=too-many-locals,variable_type_changed
@register_operator("LogSoftmaxGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def log_softmax_grad(input_dy, input_x, output_z, axis=-1,
                     kernel_name="log_softmax_grad"):
    """
    algorithm: log_softmax_grad
    calculating: gradient of log_softmax

    Parameters
    ----------
    input_dy : dict
        shape and dtype of grad input, only support float16, float32
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be the same shape and type as input
    axis: int, list or tuple .
        the first axis to reduce, may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        cce kernel name, default value is log_softmax_grad

    Returns
    -------
    None
    """

    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    if not isinstance(axis, int):
        axis = list(axis)

    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(dtype, ("float16", "float32"), param_name="x")
    if isinstance(axis, int):
        axis = [axis]

    schedules = []
    tensors = []
    ins = classify([input_dy, input_x, axis], "norm")

    for (dy, x, reduce_axis) in ins:
        with tbe.compute():
            dy_shape_var_new, x_shape_var_new = shape_util.variable_shape([dy, x], op_mode="norm")
            input_dy = tvm.placeholder(dy_shape_var_new, dtype=dtype, name="input_dy")
            input_x = tvm.placeholder(x_shape_var_new, dtype=dtype, name="input_x")
            output = log_softmax_grad_compute(input_dy, input_x, output_z, reduce_axis, kernel_name)
            tensors.append([input_dy, input_x, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
