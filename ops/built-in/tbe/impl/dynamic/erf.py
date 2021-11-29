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
dynamic erf
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,too-many-statements,invalid-name
@register_operator_compute("Erf", op_mode="dynamic", support_fusion=True)
def erf_compute(input_x, output_y, kernel_name="erf"):
    """
    compute erf

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        he dict of output_data, include keys(shape and dtype)
    kernel_name: str
        kernel name, default value is "erf"

    Returns
    -------
    erf_result: TVM tensor
        the =result of compute
    """
    # `define a scaler, value = 1`
    scalar_one = 1
    # `define a scaler, value = -1`
    scalar_negative_one = -1
    # `define a scaler, value = -0.47047, only used in compute of erf and erfc`
    scalar_p = 0.47047
    # `define a scaler, value = 0.3480242, only used in compute of erf and erfc`
    scalar_a = 0.3480242
    # `define a scaler, value = -0.0958798, only used in compute of erf and erfc`
    scalar_b = -0.0958798
    # `define a scaler, value = 0.7478556, only used in compute of erf and erfc`
    scalar_c = 0.7478556
    # `define a scaler, value = 32768`
    scalar_fp16_max = 32768
    # `define a scaler, value = 2**(-15)`
    scalar_fp16_min = 2 ** (-15)
    dtype = input_x.dtype
    dtype_ = input_x.dtype
    if dtype == "float16":
        dtype = "float32"
        input_x = tbe.cast_to(input_x, "float32")
    shape = shape_util.shape_to_list(input_x.shape)
    const_one = tvm.const(scalar_one, dtype=dtype)
    const_negative_one = tvm.const(scalar_negative_one, dtype=dtype)
    const_p = tvm.const(scalar_p, dtype=dtype)
    const_a = tvm.const(scalar_a, dtype=dtype)
    const_b = tvm.const(scalar_b, dtype=dtype)
    const_c = tvm.const(scalar_c, dtype=dtype)
    fp16_max = tvm.const(scalar_fp16_max, dtype=dtype)
    fp16_min = tvm.const(scalar_fp16_min, dtype=dtype)
    data_vmuls = tbe.vmuls(input_x, fp16_max)
    data_abs = tbe.vabs(data_vmuls)
    data_vadds = tbe.vadds(data_abs, fp16_min)
    data_div = tbe.vdiv(data_vmuls, data_vadds)
    if not tbe_platform.api_check_support("tbe.dsl.round", "float32") and dtype == "float32":
        data_div_16 = tbe.cast_to(data_div, "float16")
        data_round_16 = tbe.round(data_div_16)
        data_round = tbe.cast_to(data_round_16, dtype)
    else:
        data_round = tbe.round(data_div)
    tensor_sign = tbe.cast_to(data_round, dtype)
    tensor_one = tbe.broadcast(const_one, shape, dtype)
    tensor_abs = tbe.vabs(input_x)
    erf_t_vmuls = tbe.vmuls(tensor_abs, const_p)
    erf_t_vadds = tbe.vadds(erf_t_vmuls, const_one)
    erf_data_t = tbe.vdiv(tensor_one, erf_t_vadds)
    erf_abs_square = tbe.vmul(tensor_abs, tensor_abs)
    erf_data_vmuls = tbe.vmuls(erf_abs_square, const_negative_one)
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and dtype == "float32":
        data_div_16 = tbe.cast_to(erf_data_vmuls, "float16")
        erf_data_exp_16 = tbe.vexp(data_div_16)
        erf_data_exp = tbe.cast_to(erf_data_exp_16, dtype)
    else:
        erf_data_exp = tbe.vexp(erf_data_vmuls)
    erf_data_t_square = tbe.vmul(erf_data_t, erf_data_t)
    erf_data_t_cube = tbe.vmul(erf_data_t, erf_data_t_square)
    erf_t_vmuls = tbe.vmuls(erf_data_t, const_a)
    erf_t_square_vmuls = tbe.vmuls(erf_data_t_square, const_b)
    erf_t_cube_vmuls = tbe.vmuls(erf_data_t_cube, const_c)
    erf_square_vadd = tbe.vadd(erf_t_vmuls, erf_t_square_vmuls)
    erf_cube_vadd_ = tbe.vadd(erf_square_vadd, erf_t_cube_vmuls)
    erf_cube_vmuls = tbe.vmuls(erf_cube_vadd_, const_negative_one)
    erf_exp_vmul = tbe.vmul(erf_cube_vmuls, erf_data_exp)
    erf_exp_vadds = tbe.vadds(erf_exp_vmul, const_one)
    erf_result = tbe.vmul(tensor_sign, erf_exp_vadds)
    if dtype != dtype_:
        erf_result = tbe.cast_to(erf_result, dtype_)
    return erf_result


@register_operator("Erf")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def erf(input_x, output_y, kernel_name="erf"):
    """
    algorithm: erf
    Computes the Gauss error function of `x` element-wise

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "erf"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype")
    dtype_input = dtype_input.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype_input,
                                         name="data_input")
            res = erf_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
