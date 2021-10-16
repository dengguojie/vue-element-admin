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
l2_loss
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator


# 'pylint: disable=unused-argument,invalid-name
def l2_loss_compute(x, axes, y, kernel_name="l2_loss"):
    """
    l2_loss compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    axes: int, list, tuple or NONETYPE
        the axes for reduce.
    y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "l2_loss".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same type as input tensor.
    """
    dtype = x.dtype
    coeff_sqrt = tvm.const(1.0 / (2**0.5), dtype=dtype)
    data_mul = tbe.vmuls(x, coeff_sqrt)
    data_sqr = tbe.vmul(data_mul, data_mul)
    res = tbe.reduce_sum(data_sqr, axis=axes)

    return res


# 'pylint: disable=too-many-locals,invalid-name
@register_operator("L2Loss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def l2_loss(x, y, kernel_name="l2_loss"):
    """
    reduce a tensor on a certain axes.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "l2_loss".

    Returns
    -------
    None
    """

    shape = x.get("shape")
    dtype_x = x["dtype"]
    dtype_lower_x = dtype_x.lower()
    check_list_x = ("float16", "float32")
    para_check.check_dtype(dtype_lower_x, check_list_x, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    axes = []
    for i in range(len(shape)):
        axes.append(i)
    input_axis = {"shape": [len(axes),], "value": axes, "rel_pos_to_reduce": "axis"}

    schedules = []
    ins = classify([x, input_axis], OpPatternMode.REDUCE, {"keepdims": False})
    tensors = []

    for (_x, axes) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x, axes], op_mode="reduce")[0]
            data_input_x = tvm.placeholder(shape_x, name="data_input_x",
                                           dtype=dtype_lower_x)
            res = l2_loss_compute(data_input_x, axes.get("value"), y)
            tensors.append([data_input_x, res])
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
