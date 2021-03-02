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
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from te.lang.base import operation
from te.lang.base.operation import add_compile_info
from impl.util.platform_adapter import register_operator

# pylint: disable=locally-disabled,unused-argument
# pylint: disable=unused-variable
@register_operator("SoftmaxGrad")
def softmax_grad_compute(softmax, grad_softmax, grad_x,
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

    data_vmul = tbe.vmul(softmax, grad_softmax)
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.sum", "float32"):
        data_vmul = tbe.cast_to(data_vmul, "float32")
        grad_softmax = tbe.cast_to(grad_softmax, "float32")
        softmax = tbe.cast_to(softmax, "float32")
        has_improve_precision = True
    data_sum = tbe.sum(data_vmul, axis=-1, keepdims=True)
    data_sum_tmp = tbe.broadcast(data_sum, shape_input2)
    data_sub = tbe.vsub(grad_softmax, data_sum_tmp)
    res = tbe.vmul(softmax, data_sub)
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res

@register_operator("SoftmaxGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def softmax_grad(softmax, grad_softmax, grad_x, kernel_name="softmax_grad"):
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
    kernel_name: str
        kernel name, default value is "softmax_grad"

    Returns
    -------
    None
    """

    shape = softmax.get("shape")
    grad_shape = grad_softmax.get("shape")
    dtype = softmax.get("dtype").lower()

    axis = -1
    add_compile_info("ori_axis", axis)
    para_check.check_shape(shape, param_name="softmax")
    para_check.check_shape(grad_shape, param_name="grad_softmax")
    para_check.check_dtype(dtype, ("float16", "float32"), param_name="softmax")

    with tbe_base.compute():
        new_shape = []
        a = operation.var("a")
        new_shape.append(a)
        b = operation.var("b")
        new_shape.append(b)
        softmax = tvm.placeholder(new_shape, dtype=dtype, name="softmax")
        grad_softmaxgrad = tvm.placeholder(new_shape, dtype=dtype, name="grad_softmaxgrad")
        output = softmax_grad_compute(softmax, grad_softmaxgrad, grad_x, kernel_name)
    schedules = []
    with tvm.target.cce():
        sch = tbe.auto_schedule(output)
    schedules.append(sch)
    tensor_list = [softmax, grad_softmaxgrad, output]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}
    tbe.build(schedules, config)
