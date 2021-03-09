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
dynamic div
"""
from functools import reduce as reduceIns
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator


# pylint: disable=locally-disabled,unused-argument
def sqrt_compute(input_data, output_data, kernel_name="sqrt"):
    """
    calculating data sqrt,y= x**0.5,mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of input data
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    result: TVM tensor
        the result of sqrt
    """
    dtype = input_data.dtype
    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support(
            "te.lang.cce.vsqrt", "float32"):
        input_data = tbe.cast_to(input_data, "float32")
        has_improve_precision = True
    result = tbe.vsqrt(input_data)

    if has_improve_precision:
        result = tbe.cast_to(result, "float16")

    return result


@register_operator("Sqrt")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sqrt(input_x, output_y, kernel_name="sqrt"):
    """
    algorithm: sqrt
    calculating data sqrt,y= x**0.5, mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is sqrt

    Returns
    -------
    None
    """

    # check dtype
    x_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            # shape
            x_shape = shape_util.variable_shape([_input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            # div_compute
            input_data = tvm.placeholder(fuseshape, name="input_data",
                                         dtype=x_dtype)
            res = sqrt_compute(input_data, output_y, kernel_name)

            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
