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
import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
from te.utils import shape_util
import te.platform as tbe_platform

SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=invalid-name,too-many-locals,unused-argument,too-many-arguments
# Analysis parameter dim
def reduce_std_check_dim(axis_dim, shape_x, dim):
    """
    reduce_std_check_dim
    """
    dims = len(shape_x)
    if isinstance(dim, int):
        axis_dim.append(dim)
    elif ((dim is None) or (len(dim) == 0)):
        for i in range(dims):
            axis_dim.append(i)
    else:
        for i in dim:
            if ((i < 0) and ((i + dims) in axis_dim)) or (i in axis_dim):
                continue
            axis_dim.append(int((i + dims) % dims))
    return axis_dim


@tbe_platform.fusion_manager.fusion_manager.register("reduce_std_with_mean")
def reduce_std_compute(x, mean, dim, unbiased, keepdim, invert, epsilon, kernel_name="reduce_std_with_mean"):
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
    output TVM tensors
    """

    # Analysis parameter dim
    x_type = x.dtype.lower()

    if x_type == "float16":
        x = tbe.cast_to(x, "float32")
        mean = tbe.cast_to(mean, "float32")

    shape_x = shape_util.shape_to_list(x.shape)

    axis_dim = []
    axis_dim = reduce_std_check_dim(axis_dim, shape_x, dim)

    # got total number of tensor
    reduce_ele = 1.0
    for i in axis_dim:
        reduce_ele *= shape_x[i]
    cof = reduce_ele ** (-1)

    # broadcast
    mu_broadcast = tbe.broadcast(mean, shape_x)

    # calculate x-mubroadcast
    x_mu_sub = tbe.vsub(x, mu_broadcast)

    # calculate x_mu_sub^2
    var_mul = tbe.vmul(x_mu_sub, x_mu_sub)

    # Divided by N or (N-1)
    if unbiased:
        cof_unbiased = (reduce_ele - 1.0) ** (-1)
        var_muls = tbe.vmuls(var_mul, cof_unbiased)
    else:
        var_muls = tbe.vmuls(var_mul, cof)

    # sum
    var = tbe.sum(var_muls, axis=axis_dim, keepdims=keepdim)

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


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def reduce_std_with_mean(x, mean, y, dim=None, unbiased=True, keepdims=False, invert=False, epsilon=0.001,
                         kernel_name="reduce_std_with_mean"):
    """
    calculating data

    Parameters
    ----------
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
    check_list = ("float16", "float32")

    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()
    para_check.check_dtype(dtype_x, check_list, param_name="x")
    para_check.check_shape(shape_x, param_name="x")

    shape_mean = mean.get("shape")
    dtype_mean = mean.get("dtype").lower()
    para_check.check_dtype(dtype_mean, check_list, param_name="mean")
    para_check.check_shape(shape_mean, param_name="mean")

    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    data_mean = tvm.placeholder(mean.get("shape"), dtype=mean.get("dtype"), name="data_mean")

    res = reduce_std_compute(data_x, data_mean, dim, unbiased, keepdims, invert, epsilon, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_mean, res]}
    tbe.cce_build_code(schedule, config)
