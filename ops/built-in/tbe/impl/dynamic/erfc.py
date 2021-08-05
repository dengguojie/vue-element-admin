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
dynamic erfc
"""
import functools

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute

# define a scaler, value = 1
SCALER_ONE = 1
# define a scaler, value = -1
SCALER_NEGATIVE_ONE = -1
# define a scaler, value = -0.47047, only used in compute of erfc and erf
SCALER_P = 0.47047
# define a scaler, value = 0.3480242, only used in compute of erfc and erf
SCALER_A = 0.3480242
# define a scaler, value = -0.0958798, only used in compute of erfc and erf
SCALER_B = -0.0958798
# define a scaler, value = 0.7478556, only used in compute of erfc and erf
SCALER_C = 0.7478556
# define a scaler, value = 32768
SCALER_FP16_MAX = 32768
# define a scaler, value = 2**(-15)
SCALER_FP16_MIN = 2 ** (-15)


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("Erfc", op_mode="dynamic", support_fusion=True)
def erfc_compute(input_x, output_y, kernel_name="erfc"):
    """
    compute erfc

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        he dict of output_data, include keys(shape and dtype)
    kernel_name: str
        kernel name, default value is "erfc"

    Returns
    -------
    erfc_result: TVM tensor
        the =result of compute
    """
    dtype = input_x.dtype
    dtype_ = input_x.dtype
    shape = shape_util.shape_to_list(input_x.shape)
    if dtype == "float16":
        dtype = "float32"
        input_x = tbe.cast_to(input_x, "float32")
    const_one = tvm.const(SCALER_ONE, dtype=dtype)
    const_negative_one = tvm.const(SCALER_NEGATIVE_ONE, dtype=dtype)
    const_p = tvm.const(SCALER_P, dtype=dtype)
    const_a = tvm.const(SCALER_A, dtype=dtype)
    const_b = tvm.const(SCALER_B, dtype=dtype)
    const_c = tvm.const(SCALER_C, dtype=dtype)
    fp16_max = tvm.const(SCALER_FP16_MAX, dtype=dtype)
    fp16_min = tvm.const(SCALER_FP16_MIN, dtype=dtype)
    data_sign_vmuls = tbe.vmuls(input_x, fp16_max)
    data_sign_abs = tbe.vabs(data_sign_vmuls)
    data_vadds = tbe.vadds(data_sign_abs, fp16_min)
    data_sign_div = tbe.vdiv(data_sign_vmuls, data_vadds)
    if not tbe_platform.api_check_support("tbe.dsl.round", "float32") and dtype == "float32":
        data_sign_div_16 = tbe.cast_to(data_sign_div, "float16")
        data_round_16 = tbe.round(data_sign_div_16)
        data_round = tbe.cast_to(data_round_16, dtype)
    else:
        data_round = tbe.round(data_sign_div)
    tensor_sign = tbe.cast_to(data_round, dtype)
    tensor_one = tbe.broadcast(const_one, shape, dtype)
    tensor_abs = tbe.vabs(input_x)
    erfc_t_vmuls = tbe.vmuls(tensor_abs, const_p)
    erfc_t_vadds = tbe.vadds(erfc_t_vmuls, const_one)
    erfc_data_t = tbe.vdiv(tensor_one, erfc_t_vadds)
    erfc_abs_square = tbe.vmul(tensor_abs, tensor_abs)
    erfc_data_vmuls = tbe.vmuls(erfc_abs_square, const_negative_one)
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and dtype == "float32":
        erfc_data_vmuls_16 = tbe.cast_to(erfc_data_vmuls, "float16")
        erfc_data_exp_16 = tbe.vexp(erfc_data_vmuls_16)
        erfc_data_exp = tbe.cast_to(erfc_data_exp_16, dtype)
    else:
        erfc_data_exp = tbe.vexp(erfc_data_vmuls)
    erfc_data_t_square = tbe.vmul(erfc_data_t, erfc_data_t)
    erfc_data_t_cube = tbe.vmul(erfc_data_t, erfc_data_t_square)
    erfc_t_vmuls = tbe.vmuls(erfc_data_t, const_a)
    erfc_t_square_vmuls = tbe.vmuls(erfc_data_t_square, const_b)
    erfc_t_cube_vmuls = tbe.vmuls(erfc_data_t_cube, const_c)
    erfc_square_vadd = tbe.vadd(erfc_t_vmuls, erfc_t_square_vmuls)
    erfc_cube_vadd_ = tbe.vadd(erfc_square_vadd, erfc_t_cube_vmuls)
    erfc_cube_vmuls = tbe.vmuls(erfc_cube_vadd_, const_negative_one)
    erfc_exp_vmul = tbe.vmul(erfc_cube_vmuls, erfc_data_exp)
    erfc_exp_vadds = tbe.vadds(erfc_exp_vmul, const_one)
    erfc_sign_vmul = tbe.vmul(tensor_sign, erfc_exp_vadds)
    erfc_sign_vmuls = tbe.vmuls(erfc_sign_vmul, const_negative_one)
    erfc_result = tbe.vadds(erfc_sign_vmuls, const_one)
    if dtype != dtype_:
        erfc_result = tbe.cast_to(erfc_result, dtype_)
    return erfc_result


@register_operator("Erfc")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def erfc(input_x, output_y, kernel_name="erfc"):
    """
    algorithm: erfc
    Computes the Gauss error function of `x` element-wise

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "erfc"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype_input,
                                         name="data_input")
            res = erfc_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
