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
inv
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

# define a scalar , value = 1
SCALAR_ONE = 1


# pylint: disable=locally-disabled,unused-argument
@register_operator_compute("Inv", op_mode="dynamic", support_fusion=False)
def inv_compute(input_x, output_y, kernel_name="inv"):
    """
    compute inv

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: TVM tensor
        the placeholder of output data
    kernel_name: str
        kernel name, default value is "inv"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype
    shape = shape_util.shape_to_list(input_x.shape)

    temp_const = tvm.const(SCALAR_ONE, dtype=dtype)
    temp_tensor = tbe.broadcast(temp_const, shape, dtype)
    res = tbe.vdiv(temp_tensor, input_x)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
@register_operator("Inv")
def inv(input_x, output_y, kernel_name="inv"):
    """
    algorithm: inv
    calculating data's reciprocal, y = 1 / x

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "inv"

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
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
            res = inv_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
    