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
dynamic sinh
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute

class Constant:
    """
    The class for constant.
    """
    SCALER_NEGATIVE_ONE = -1
    SCALER_ZERO_POINT_FIVE = 0.5
    SCALAR_TWO = 2


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("Sinh", op_mode="dynamic", support_fusion=True)
def sinh_compute(input_data, output_data, kernel_name="sinh"):
    """
    algorithm: sinh
    calculating data's sinh = (exp(x) - exp(-x)) / 2

    Parameters
    ----------
    input_data: TVM tensor
        data of input.
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "sinh"

    Returns
    -------
    res: TVM tensor
        the res of sinh
    """

    dtype = input_data.dtype
    dtype_copy = input_data.dtype
    shape = input_data.shape

    # in order to get the precise calcuate result
    has_improve_precision = False
    if dtype.lower() == "float16" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_data = tbe.cast_to(input_data, "float32")
        dtype = "float32"
        has_improve_precision = True

    if dtype.lower() == "float32" and not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_data = tbe.cast_to(input_data, "float16")
        dtype = "float16"
        has_improve_precision = True

    data_mul = tbe.vmuls(input_data, tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype))
    data_exp = tbe.vexp(data_mul)
    data_exp_x = tbe.vmuls(data_exp, tvm.const(Constant.SCALER_ZERO_POINT_FIVE, dtype))

    tensor_two = tbe.broadcast(tvm.const(Constant.SCALAR_TWO, dtype), shape)
    data_ln2 = tbe.vlog(tensor_two)
    data_neg_ln2 = tbe.vmuls(data_ln2, tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype))
    data_x = tbe.vadd(input_data, data_neg_ln2)
    data_exp_data = tbe.vexp(data_x)

    res = tbe.vsub(data_exp_data, data_exp_x)

    # cast the dtype to float16
    if has_improve_precision:
        res = tbe.cast_to(res, dtype_copy)

    return res


@register_operator("Sinh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sinh(input_data, output_data, kernel_name="sinh"):
    """
    algorithm: sinh
    calculating data's sinh = (exp(x) - exp(-x)) / 2

    Parameters
    ----------
    input_data: dict
        shape and dtype of input, only support float16, float32
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "sinh"

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")
    dtype_input = input_data.get("dtype")
    input_dtype = dtype_input.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_data")

    schedules, tensors = [], []
    ins = classify([input_data], OpPatternMode.ELEWISE)
    for (_input_data,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_data])
            data_input = tvm.placeholder(x_shape[0], dtype=input_dtype,
                                         name="data_input")
            res = sinh_compute(data_input, output_data, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
