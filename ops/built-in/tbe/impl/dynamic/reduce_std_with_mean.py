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
reduce_std_with_mean
"""
import collections

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpPatternMode

SHAPE_SIZE_LIMIT = 2147483648


@register_operator_compute("ReduceStdWithMean", op_mode="dynamic", support_fusion=True)
def reduce_std_with_mean_compute(x, mean, dim, unbiased, keepdim, invert, epsilon, kernel_name="reduce_std_with_mean"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of X
    mean : TVM tensor
        the mean of X
    dim : int or intlist
        dimension to calculate, default value is None
    unbiased : bool
        control Bessel deviation, default value is True
    keepdim : bool
        hold dimension or not, default value is False
    kernel_name : str
        kernel name
    invert: bool
        controls whether the output is variance or inverse of variance, default value is False
    epsilon: float
        prevent division by 0
    kernel_name: str
        kernel name

    Returns
    -------
    output TVM tensor
    """
    x_type = x.dtype.lower()

    if x_type == "float16":
        x = tbe.cast_to(x, "float32")
        mean = tbe.cast_to(mean, "float32")

    kernel_name_var = kernel_name

    shape_x = shape_util.shape_to_list(x.shape)

    reduce_ele = 1.0
    for i in shape_x:
        reduce_ele *= i
    dtype = x.dtype
    # broadcast
    # mu_broadcast = tbe.broadcast(mean, shape_x)
    mu_broadcast = mean

    # calculate x-mubroadcast
    x_mu_sub = tbe.vsub(x, mu_broadcast)

    # calculate x_mu_sub^2
    var_mul = tbe.vmul(x_mu_sub, x_mu_sub)

    if unbiased:
        # Divided by N or (N-1)
        if isinstance(reduce_ele, float):
            cof_unbiased = (reduce_ele - 1.0) ** (-1)
            cof_unbiased = tvm.const(cof_unbiased, dtype=dtype)
        else:
            cof_unbiased = tbe.var("cof_unbiased", dtype=dtype)
            if dtype == "float16":
                tbe.var("cof_empty", dtype=dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", dtype)
            tbe_context.get_context().add_compile_info("attr_unbiased", "true")
        var_muls = tbe.vmuls(var_mul, cof_unbiased)
    else:
        if isinstance(reduce_ele, float):
            cof = reduce_ele ** (-1)
            cof = tvm.const(cof, dtype=dtype)
        else:
            cof = tbe.var("cof", dtype=dtype)
            if dtype == "float16":
                tbe.var("cof_empty", dtype=dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", dtype)
            tbe_context.get_context().add_compile_info("attr_unbiased", "false")
        var_muls = tbe.vmuls(var_mul, cof)

    # sum
    var = tbe.reduce_sum(var_muls, axis=dim, keepdims=keepdim)

    # calculate the square root
    y = tbe.vsqrt(var)

    # Determine invert: If the value is false, the variance is output.
    # If the value is true, the inverse of the variance is output.
    if not invert:
        if y.dtype != x_type:
            y = tbe.cast_to(y, dtype=x_type)

        # return variance
        return y
    else:
        epsilon_value = tvm.const(epsilon, dtype=y.dtype)
        y_epsilon = tbe.vadds(y, epsilon_value)
        y_invert = tbe.vrec(y_epsilon)
        if y_invert.dtype != x_type:
            y_invert = tbe.cast_to(y_invert, dtype=x_type)

        # return the inverse of variance
        return y_invert


@register_operator("ReduceStdWithMean")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def reduce_std_with_mean(x, mean, y, dim=None, unbiased=True, keepdims=False, invert=False, epsilon=0.001,
                         kernel_name="reduce_std_with_mean"):
    """
    Calculate variance or reciprocal of variance

    Parameters:
    ---------
    x: dict
        input tensor
    mean: dict
        mean value of input tensor
    y: dict
        output, variance or reciprocaal of variance
    dim: int or list[int]
        dimension to calculate, default value is None
    unbiased: bool
        control Bessel deviation, default value is True
    keepdims: bool
        hold dimension or not, default value is False
    invert: bool
        controls whether the output is variance or inverse of variance, default value is False
    epsilon: float
        prevent division by 0
    kernel_name: str
        cce kernel name, default value is reduce_std_with_mean
    
    Returns
    -------
    None
    """
    # calculating data parameters
    dtype = x["dtype"]
    dtype_y = y["dtype"]
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_lower, check_list)
    x["rel_pos_to_reduce"] = "before"
    mean["rel_pos_to_reduce"] = "before"

    shape = x["shape"]
    shape_len = len(shape)
    if not dim:
        dim = range(shape_len)
    if hasattr(dim, 'index'):
        dim = list(dim)
    dim = shape_util.axis_check(shape_len, dim)
    input_axis = {"shape": [len(dim), ], "value": dim, "rel_pos_to_reduce": "axis"}

    schedules = []
    tensors = []
    ins = classify([x, mean, input_axis], OpPatternMode.REDUCE, {"keepdims": keepdims is True})

    for (_input_x, _mean, _axes) in ins:
        with tbe.compute():
            x_var_new, mean_var_new = shape_util.variable_shape([_input_x, _mean, _axes],
                                                                op_mode="reduce")[0:2]
            data_x = tvm.placeholder(x_var_new, name="data_x",
                                     dtype=dtype_lower)
            data_mean = tvm.placeholder(mean_var_new, name="data_mean",
                                        dtype=dtype_lower)
            res = reduce_std_with_mean_compute(data_x, data_mean, _axes.get("value"), unbiased, keepdims,
                                               invert, epsilon, kernel_name)
            tensors.append([data_x, data_mean, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
