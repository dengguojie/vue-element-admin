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
layer_norm_unify
"""
from tbe import tvm
from tbe.dsl.base import operation
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import classify


# 'pylint: disable = unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
def set_range(input_x, input_gamma, input_beta):
    """
    Set range information
    """
    range_x = []
    for dim in input_x.get("shape"):
        range_x.append((dim, dim))
    input_x["range"] = range_x
    range_gamma = []
    for dim in input_gamma.get("shape"):
        range_gamma.append((dim, dim))
    input_gamma["range"] = range_gamma
    range_beta = []
    for dim in input_beta.get("shape"):
        range_beta.append((dim, dim))
    input_beta["range"] = range_beta

    return input_x, input_gamma, input_beta


def is_special_cases(shape_x, shape_gamma, shape_beta):
    """
    Judge whether it is a special case
    """
    white_list_x = [[4, 300, 257, 12], [8, 512, 128], [8, 512, 4096], [64, 50, 768]]
    white_list_gamma = [[12], [128], [4096], [768]]
    white_list_beta = [[12], [128], [4096], [768]]
    shape_x = list(shape_x)
    shape_gamma = list(shape_gamma)
    shape_beta = list(shape_beta)

    if shape_x in white_list_x and shape_gamma in white_list_gamma and shape_beta in white_list_beta:
        return True

    return False


def layer_norm_compute(input_x,
                       input_gamma,
                       input_beta,
                       output_y,
                       output_mean,
                       output_variance,
                       reduce_axis,
                       begin_params_axis,
                       epsilon,
                       kernel_name="layer_norm",
                       impl_mode="high_performance"):
    """
    DSL description of the layernorm operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    output_data: dict
        shape and dtype of output
    reduce_axis: list
      the reduce axis
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "cce_layernorm"

    Returns
    -------
    res_tuple: tuple
        (mean, variance, result)
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    input_x1 = input_x
    cast_dtype = dtype
    cast_dtype_precision = dtype
    is_cast = False
    is_support_vexp = tbe_platform.api_check_support("te.lang.cce.vexp", "float32")
    tbe_context.get_context().add_compile_info("is_support_vexp", is_support_vexp)
    if dtype == "float16" and is_support_vexp and impl_mode == "high_performance":
        cast_dtype = "float32"
        cast_dtype_precision = "float32"
        input_x = tbe.cast_to(input_x, "float32")
        input_x1 = tbe.cast_to(input_x1, "float32")
        input_gamma = tbe.cast_to(input_gamma, "float32")
        input_beta = tbe.cast_to(input_beta, "float32")
        is_cast = True

    reduce_elts = 1.0
    for i in reduce_axis:
        reduce_elts *= shape_x[i]
    if isinstance(reduce_elts, float):
        mean_cofs = reduce_elts ** (-1)
        mean_cof = tvm.const(mean_cofs, dtype=cast_dtype)
    else:
        mean_cof = tbe.var("mean_cof", dtype=cast_dtype)
        operation.add_compile_info("reduce_mean_cof_dtype", cast_dtype)

    # DSL description of the mean calculation process
    mean_muls = tbe.vmuls(input_x, mean_cof)
    mean = tbe.reduce_sum(mean_muls, axis=reduce_axis, keepdims=True)
    # workspace special case
    if is_cast:
        mean_16 = tbe.cast_to(mean, "float16")
        mean = tbe.cast_to(mean_16, "float32")

    # DSL description of the variance calculation process
    mean_variance_broadcast = tbe.broadcast(mean, shape_x)
    variance_sub = tbe.vsub(input_x1, mean_variance_broadcast)
    variance_mul = tbe.vmul(variance_sub, variance_sub)
    variance_muls = tbe.vmuls(variance_mul, mean_cof)
    variance = tbe.reduce_sum(variance_muls, axis=reduce_axis, keepdims=True)
    if is_cast:
        variance_16 = tbe.cast_to(variance, "float16")
        variance = tbe.cast_to(variance_16, "float32")
    normalize_sub = variance_sub

    # DSL description of the normalize calculation process
    if impl_mode == "high_performance" and is_support_vexp:
        epsilon = tvm.const(epsilon, dtype=cast_dtype)
        variance_normalize_broadcast = tbe.broadcast(variance, shape_x)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_log = tbe.vlog(normalize_add)
        normalize_log_mul = tbe.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
        normalize_exp = tbe.vexp(normalize_log_mul)
        normalize_mul = tbe.vmul(normalize_sub, normalize_exp)
    else:
        tesor_one = tbe.broadcast(tvm.const(1, cast_dtype_precision), shape_x)
        variance_normalize_broadcast = tbe.broadcast(variance, shape_x)
        epsilon = tvm.const(epsilon, dtype=cast_dtype_precision)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_sqrt = tbe.vsqrt(normalize_add, 0)
        normalize_rsqrt = tbe.vdiv(tesor_one, normalize_sqrt)
        normalize_mul = tbe.vmul(normalize_sub, normalize_rsqrt)

    # DSL description of the scale and translate calculation process
    if begin_params_axis == 0:
        scale_mul = tbe.vmul(normalize_mul, input_gamma)
        res = tbe.vadd(scale_mul, input_beta)
    else:
        gamma_broadcast = tbe.broadcast(input_gamma, shape_x)
        beta_broadcast = tbe.broadcast(input_beta, shape_x)
        scale_mul = tbe.vmul(normalize_mul, gamma_broadcast)
        res = tbe.vadd(scale_mul, beta_broadcast)

    if is_cast:
        res = tbe.cast_to(res, "float16")
        return mean_16, variance_16, res

    return mean, variance, res


@register_operator("StaticLayerNorm")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def layer_norm(input_x,
               input_gamma,
               input_beta,
               output_y,
               output_mean,
               output_variance,
               begin_norm_axis,
               begin_params_axis,
               epsilon=1e-12,
               kernel_name="layer_norm",
               impl_mode="high_performance"):
    """
    layernorm operator interface implementation
    calculating: x, gamma, beta
        mean  = np.mean(x, reduce_axis, keepdims=True)
        variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True)
        result = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

    Parameters
    ----------
    input_x : dict
        shape and dtype of input x, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    input_beta: dict
        shape and dtype of input beta, only support float16, float32
    output_y: dict
        shape and dtype of output, only support float16, float32
    begin_norm_axis: int
      The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "layernorm"

    Returns
    -------
    None
    """
    shape_x = list(input_x.get("shape"))

    check_list = ("float16", "float32")
    dtype = input_x.get("dtype").lower()
    dtype_gamma = input_gamma.get("dtype").lower()
    dtype_beta = input_beta.get("dtype").lower()
    para_check.check_dtype(dtype, check_list, param_name="input_x")
    para_check.check_dtype(dtype_gamma, check_list, param_name="input_gamma")
    para_check.check_dtype(dtype_beta, check_list, param_name="input_beta")

    shape_gamma = list(input_gamma.get("shape"))
    shape_beta = list(input_beta.get("shape"))
    range_gamma = list(input_gamma.get("range"))
    range_beta = list(input_beta.get("range"))

    begin_norm_axis = shape_util.axis_check(len(shape_x), begin_norm_axis)
    begin_params_axis = shape_util.axis_check(len(shape_x), begin_params_axis)

    index_list = tuple(index for index, _ in enumerate(shape_x))
    reduce_axis = index_list[begin_norm_axis:]
    broadcast_axis = index_list[:begin_params_axis]

    input_gamma["shape"] = tuple(shape_gamma)
    input_beta["shape"] = tuple(shape_beta)
    input_gamma["range"] = tuple(range_gamma)
    input_beta["range"] = tuple(range_beta)

    ins = classify([input_x, input_gamma, input_beta, reduce_axis], "norm",
                   {"input_shape_type": [0, 1, 1], "same_input_shape_group": [[1, 2]],
                    "compile_broadcast_axes": {1: broadcast_axis, 2: broadcast_axis}})
    schedules = []
    tensors = []

    for (dy_shape_x, dy_shape_gamma, dy_shape_beta, dy_reduce_axis) in ins:
        with tbe.compute():
            x_var, gamma_var, beta_var = shape_util.variable_shape(
                [dy_shape_x, dy_shape_gamma, dy_shape_beta], op_mode="norm")
            data_x = tvm.placeholder(x_var, name="x", dtype=dtype)
            data_gamma = tvm.placeholder(gamma_var, name="gamma", dtype=dtype)
            data_beta = tvm.placeholder(beta_var, name="beta", dtype=dtype)

            mean, variance, res = layer_norm_compute(data_x, data_gamma, data_beta,
                                                     output_y, output_mean, output_variance,
                                                     dy_reduce_axis,
                                                     begin_params_axis, epsilon,
                                                     kernel_name, impl_mode)
            tensors.append([data_x, data_gamma, data_beta, res, mean, variance])
        with tvm.target.cce():
            sch = tbe.auto_schedule([res, mean, variance])

        schedules.append(sch)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensors
    }

    tbe.build(schedules, config)
