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
tanh_grad
"""
import functools

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("TanhGrad", op_mode="dynamic", support_fusion=True)
def tanh_grad_compute(y, dy, z, kernel_name="tanh_grad"):
    """
    do element-wise tanh_grad operation between two input tensors

    Parameters
    ----------
    y: TVM tensor
        the placeholder of y input data
    dy: TVM tensor
        the placeholder of dy input data
    z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh_grad

    Returns
    -------
    res : tvm.tensor
        the result of tanh_grad
    """
    dtype = y.dtype

    if dtype == "float16":
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")

    data1_square = tbe.vmul(y, y)
    data_mul = tbe.vmuls(data1_square, tvm.const(-1, dtype=dtype))
    anuminate = tbe.vadds(data_mul, tvm.const(1, dtype=dtype))
    res = tbe.vmul(anuminate, dy)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=too-many-locals
@register_operator("TanhGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def tanh_grad(y, dy, z, kernel_name="tanh_grad"):
    """
    do element-wise tanh_grad operation between two input tensors

    Parameters
    ----------
    y : dict
        shape and dtype of y input, only support float16, float32
    dy : dict
        shape and dtype of dy input, only support float16, float32
    z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is tanh_grad

    Returns
    -------
    None
    """
    dtype = y.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="y")
    ins = classify([y], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (classify_y,) in ins:
        with tbe.compute():
            shape = shape_util.variable_shape([classify_y])
            fuseshape = [1]
            fuseshape[0] = functools.reduce(lambda x, y: x * y, shape[0])
            data_y = tvm.placeholder(fuseshape, dtype=dtype, name="data1")
            data_dy = tvm.placeholder(fuseshape, dtype=dtype, name="data2")
            res = tanh_grad_compute(data_y, data_dy, z, kernel_name)
            tensors.append([data_y, data_dy, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "print_ir": False, "tensor_list": tensors}
    tbe.build(schedules, config)
