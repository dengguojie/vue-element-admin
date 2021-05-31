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
dynamic cos
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute

# 2pi, the cycle of cosin
TWO_PI = 2 * 3.14159265358979


# pylint: disable=locally-disabled, unused-argument
@register_operator_compute("Cos", op_mode="dynamic", support_fusion=True)
def cos_compute(input_x, output_y, kernel_name="cos"):
    """
    algorithm: cos
    calculating data's cos x = 1 - x^2/2! + x^4/4! + ... + (-1)^k*x^2k/(2k)!

    Parameters
    ----------
    input_x : TVM tensor
              data of input
    output_y: dict
              shape and dtype of output, should be same shape and type as input
    kernel_name: str
              kernel name, default value is "cos"

    Returns
    -------
    res : TVM tensor
          the result of cos
    """

    dtype = input_x.dtype
    shape = shape_util.shape_to_list(input_x.shape)

    # cast to type float32 when type is float16
    has_improve_precision = False
    if dtype.lower() == "float16" and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    # round the input
    vmu_ = tbe.vmuls(input_x, 1.0 / TWO_PI)
    if not tbe_platform.api_check_support("tbe.dsl.round", "float32") and dtype == "float32":
        input_x_ = tbe.cast_to(vmu_, "float16")
        round_fp = tbe.round(input_x_)
    else:
        round_fp = tbe.round(vmu_)

    round_fp32 = tbe.cast_to(round_fp, dtype)
    input_x_round = tbe.vsub(input_x, tbe.vmuls(round_fp32, TWO_PI))

    # the initial value one
    const_res = tvm.const(1.0, dtype=dtype)
    res = tbe.broadcast(const_res, shape)
    # compute the rank 2
    input_x_power = tbe.vmul(input_x_round, input_x_round)
    iter_value = tbe.vmuls(input_x_power, -1.0 / 2.0)
    res = tbe.vadd(res, iter_value)
    # compute the rank 4~14
    iter_list = (4, 6, 8, 10, 12, 14)
    for i in iter_list:
        iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0 / (i * (i - 1)))
        res = tbe.vadd(res, iter_value)

    # cast the dtype to float16
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("Cos")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def cos(input_x, output_y, kernel_name="cos"):
    """
    algorithm: cos
    calculating data's cos x = 1 - x^2/2! + x^4/4! + ... + (-1)^k*x^2k/(2k)!

    Parameters
    ----------
    input_x : dict
              shape and dtype of input, only support float16, float32
    output_y: dict
              shape and dtype of output, should be same shape and type as input
    kernel_name : str
              kernel name, default value is "cos"

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_input = tvm.placeholder(x_shape[0], dtype=input_dtype,
                                         name="data_input")
            res = cos_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
