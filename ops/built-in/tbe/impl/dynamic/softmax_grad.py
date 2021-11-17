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
dynamic softmaxgrad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify


# 'pylint: disable=locally-disabled,unused-argument
# 'pylint: disable=unused-variable
@register_operator("SoftmaxGrad")
def softmax_grad_compute(softmax, grad_softmax, grad_x, axis,
                         kernel_name="softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: TVM tensor
        the placeholder of first input data
    grad_softmax: TVM tensor
        the placeholder of second input data
    grad_x: dict
        the dict of output data
    axis: int, list or tuple .
        the first axis to reduce, may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        cce kernel name, default value is "softmax_grad"

    Returns
    -------
    res: TVM tensor
        the result of softmax_grad_compute
    """
    dtype = softmax.dtype
    shape_input2 = shape_util.shape_to_list(grad_softmax.shape)
    has_improve_precision = False

    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        grad_softmax = tbe.cast_to(grad_softmax, "float32")
        softmax = tbe.cast_to(softmax, "float32")
        has_improve_precision = True
    data_vmul = tbe.vmul(softmax, grad_softmax)
    data_sum = tbe.reduce_sum(data_vmul, axis=axis, keepdims=True)
    data_sum_tmp = tbe.broadcast(data_sum, shape_input2)
    data_sub = tbe.vsub(grad_softmax, data_sum_tmp)
    res = tbe.vmul(softmax, data_sub)
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint:disable=too-many-locals,invalid-name
@register_operator("SoftmaxGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def softmax_grad(softmax, grad_softmax, grad_x, axis=-1, kernel_name="softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: dict
        shape and dtype of first input, only support float16, float32
    grad_softmax: dict
        shape and dtype of second input, only support float16, float32
    grad_x: dict
        shape and dtype of output data, should be same shape and type as input
    axis: int, list or tuple .
        the first axis to reduce, may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        kernel name, default value is "softmax_grad"

    Returns
    -------
    None
    """

    shape = softmax.get("shape")
    grad_shape = grad_softmax.get("shape")
    dtype = softmax.get("dtype").lower()

    para_check.check_shape(shape, param_name="softmax")
    para_check.check_shape(grad_shape, param_name="grad_softmax")
    para_check.check_dtype(dtype, ("float16", "float32"), param_name="softmax")
    if not isinstance(axis, int):
        list_axis = list(axis)
    else:
        list_axis = [axis]
    input_axis = {"shape": [len(list_axis), ], "value": list_axis, "rel_pos_to_reduce": "axis"}

    schedules = []
    tensors = []
    ins = classify([softmax, grad_softmax, input_axis], "norm")

    for (x, grad, input_axis) in ins:
        with tbe.compute():
            shape_var_new, grad_shape_var_new, _ = shape_util.variable_shape([x, grad, input_axis], op_mode="norm")
            softmax = tvm.placeholder(shape_var_new, dtype=dtype, name="softmax")
            grad_softmax = tvm.placeholder(grad_shape_var_new, dtype=dtype, name="grad_softmax")
            output = softmax_grad_compute(softmax, grad_softmax, grad_x, input_axis.get("value"), kernel_name)
            tensors.append([softmax, grad_softmax, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
