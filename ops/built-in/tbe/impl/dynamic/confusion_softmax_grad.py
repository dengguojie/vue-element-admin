# Copyright 2019 Huawei Technologies Co., Ltd
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
confusion_softmax_grad
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from tbe.common.utils import shape_util
from tbe.dsl.base.operation import var


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = shape_util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = tbe.broadcast(tensor, temp_shape)
    tensor = tbe.broadcast(tensor, shape)
    return tensor

@register_operator_compute("ConfusionSoftmaxGrad", op_mode="dynamic", support_fusion=False)
def confusion_softmax_grad_compute(grad_dict, grad, x, y,
                                   kernel_name="confusion_softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    y = grad - sum(grad * x)

    Parameters
    ----------
    grad_dict: dict
        the dict of first input data
    grad: TVM tensor
        the placeholder of first input data
    x: TVM tensor
        the placeholder of second input data
    y: dict
        the dict of output data
    kernel_name: str
        cce kernel name, default value is "confusion_softmax_grad"

    Returns
    -------
    res: TVM tensor
        the result of confusion_softmax_grad_compute
    """
    dtype = grad.dtype
    shape_input1 = shape_util.shape_to_list(grad.shape)
    shape_input2 = shape_util.shape_to_list(x.shape)
    shape = shape_input2
    if list(shape_input1) != list(shape_input2):
        shape_input1, shape_input2, shape = shape_util.broadcast_shapes(shape_input1, shape_input2,
                                                                        param_name_input1="grad",
                                                                        param_name_input2="x")
        grad = _broadcast_nz(grad, shape)
        x = _broadcast_nz(x, shape)

    data_vmul = tbe.vmul(grad, x)
    if dtype == "float16":
        data_vmul = tbe.cast_to(data_vmul, "float32")

    axis = [-1]

    data_sum = tbe.reduce_sum(data_vmul, axis=axis, keepdims=True)

    if dtype == "float16":
        data_sum = tbe.cast_to(data_sum, "float16")

    data_sum_tmp = _broadcast_nz(data_sum, shape)

    res = tbe.vsub(grad, data_sum_tmp)

    return res

@register_operator("ConfusionSoftmaxGrad", "ConfusionSoftmaxGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def confusion_softmax_grad(grad, x, y, kernel_name="confusion_softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    y = grad - sum(grad * x)

    Parameters
    ----------
    grad: dict
        shape and dtype of first input, only support float16, float32
    x: dict
        shape and dtype of second input, only support float16, float32
    y: dict
        shape and dtype of output data, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "confusion_softmax_grad"

    Returns
    -------
    None
    """
    grad = util_common.update_shape_base_other_format(grad)
    x = util_common.update_shape_base_other_format(x)
    shape_grad = grad.get("shape")
    shape_x = x.get("shape")
    dtype_grad = grad.get("dtype")

    shape_util.compare_tensor_dict_key(grad, x, "dtype")
    para_check.check_shape(shape_grad, param_name="grad")
    para_check.check_shape(shape_x, param_name="x")

    check_list = ("float16", "float32")
    input_dtype = dtype_grad.lower()

    tensors = []
    schedules = []

    with tbe.compute():

        dim_0_0 = var("dim_0_0")
        dim_0_1 = var("dim_0_1")
        dim_0_2 = var("dim_0_2")

        shape_grad = [dim_0_0, dim_0_1, dim_0_2]

        para_check.check_dtype(input_dtype, check_list, param_name="grad")
        if list(shape_grad) != list(shape_x):
            shape_grad, shape_x, _ = \
                shape_util.broadcast_shapes(shape_grad, shape_x, param_name_input1="grad", param_name_input2="x")

        data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=input_dtype)
        data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_dtype)

        res = confusion_softmax_grad_compute(grad, data_grad, data_x, y,
                                            kernel_name=kernel_name)

        tensors.append([data_grad, data_x] + [res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
