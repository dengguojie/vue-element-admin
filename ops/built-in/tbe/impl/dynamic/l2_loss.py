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
import te
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import shape_util
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.lang.base.operation import add_compile_info

NONETYPE = type(None)


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
    res = tbe.sum(data_sqr, axis=axes)

    return res


# 'pylint: disable=too-many-locals,invalid-name
@te.op.register_operator("L2Loss")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT,
                 KERNEL_NAME)
def l2_loss(x, y, kernel_name="l2_loss"):
    """reduce a tensor on a certain axes.

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
    check_dtype(dtype_lower_x, check_list_x, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    axes = []
    for i in range(len(shape)):
        axes.append(i)
    add_compile_info("_ori_axis", axes)
    input_axis = {"shape": [len(axes), ], "value": axes, "rel_pos_to_reduce": "axis"}

    schedules = []
    ins = classify([x, input_axis], Mode.REDUCE)
    tensors = []

    for (_x, axes) in ins:
        with tbe_base.compute():
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
