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
invert
"""

import functools
import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
import te.lang.base as tbe_base
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator


# pylint: disable=locally-disabled,unused-argument
@register_operator_compute("Invert", op_mode="dynamic", support_fusion=False)
def invert_compute(input_x, output_y, kernel_name="invert"):
    """Flips all bits elementwise.

    Parameters
    ----------
    input_x: TVM tensor
        input tensor.
    output_y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "invert".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    res = tbe.vnot(input_x)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
@register_operator("Invert")
def invert(input_x, output_y, kernel_name="invert"):
    """Flips all bits elementwise.

    Parameters
    ----------
    input_x: dict
        the dict of input tensor.
        Must be one of the following types: `int16`, `uint16`.
    output_y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "invert".

    Returns
    -------
    None.
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("int16", "uint16")
    para_check.check_dtype(input_dtype, check_list, param_name = "input_x")

    schedules, tensors = [], []
    ins = classify([input_x], Mode.ELEWISE)
    for (_input_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([_input_x])
            fuseshape = [1]
            fuseshape[0] = functools.reduce(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuseshape, dtype = input_dtype,
                                         name = "data_input")
            res = invert_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
