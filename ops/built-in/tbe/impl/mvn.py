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
mvn
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from impl.util import util_select_op_base

# const value
CONST_HALF = 0.5
CONST_SQRT_ITER = 3


# pylint: disable = unused-argument
# pylint: disable=invalid-name,too-many-arguments
def get_op_support_info(x, y, normalize_variance=True, across_channels=False, eps=1e-9, kernel_name="mvn"):
    """
    get_op_support_info
    """
    format_x = x.get("format")
    axis_split_list = []
    if format_x == "NCHW":
        if across_channels:
            split_0 = [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])]
            axis_split_list.append(split_0)
        else:
            split_0 = [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])]
            split_1 = [util_select_op_base.SplitInput([0, [1], [-1], [-1]]), util_select_op_base.SplitOutput([0, [1]])]
            axis_split_list= [split_0, split_1]
    else:
        axis_split_list = None

    axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_list, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# pylint: disable=too-few-public-methods
def _check_format_shape(data_format, shape):
    """
    Function to check format and shape of input data.

    Parameters
    ----------

    data_format: str
        format of input data
    shape: list or tuple
        data shape of input data
    Returns
    -------
    None
    """

    para_check.check_format(data_format, ("NCHW",), param_name="x")

    para_check.check_shape(shape, min_rank=4, max_rank=4, param_name="x")


def _check_dtype(input_dtype):
    """
    Function to check dtype of input data.

    Parameters
    ----------

    input_dtype: str
        dtype of input data
    Returns
    -------
    None
    """

    if tbe_platform.get_soc_spec("SOC_VERSION") in (
            "Hi3796CV300ES", "Hi3796CV300CS"):
        if input_dtype == "float32":
            error_info = {'errCode': 'E81006', 'param_name': 'dtype', 'op_name': 'mvn', 'real_value': input_dtype}
            raise RuntimeError("In op[%s], Hi3796CV300ES is not supported while the [%s] of input is [%s]."
                               % (error_info['op_name'], error_info['param_name'], error_info['real_value']))
        para_check.check_dtype(input_dtype, ("float16",), param_name="x")
    else:
        para_check.check_dtype(input_dtype, ("float16", "float32",), param_name="x")


# pylint: disable=too-many-arguments,too-many-locals,protected-access
# pylint: disable=too-many-branches,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("mvn")
def mvn_compute(x, y, normalize_variance, across_channels,
                eps, kernel_name="mvn"):
    """
    algorithm: MVN
    y = (x-mean(x))/(std(x) + eps)

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    normalize_variance: bool
        A bool value indicates the operation for normalize_variance.
    across_channels: bool
        A bool value indicates the operation for across_channels.
    eps: float
        A small float number added to the variance of x. Defaults to `1e-9`.
    kernel_name : str
        kernel name, default value is "mvn"

    Returns
    -------
    output tensor
    """

    dtype_x = x.dtype
    is_cast = False

    if tbe_platform.api_check_support("te.lang.cce.vmuls", "float32"):
        if dtype_x == "float16":
            is_cast = True
            x = tbe.cast_to(x, 'float32')

    shape_x = shape_util.shape_to_list(x.shape)

    if across_channels:
        axis = [1, 2, 3]
        num = shape_x[1] * shape_x[2] * shape_x[3]
    else:
        axis = [2, 3]
        num = shape_x[2] * shape_x[3]
    if num != 0:
        num_rec = 1.0/num

    # compute subtract mean
    mean_sum = tbe.sum(x, axis, True)    # sum
    mean_muls = tbe.vmuls(mean_sum, num_rec)
    mean_broad = tbe.broadcast(mean_muls, shape_x)  # mean

    if normalize_variance:
        mean_sub = tbe.vsub(x, mean_broad)   # x - mean

        var_mul = tbe.vmul(mean_sub, mean_sub)
        var_sum = tbe.sum(var_mul, axis, True)
        var_muls = tbe.vmuls(var_sum, num_rec)
        var = tbe.broadcast(var_muls, shape_x)   # var

        if tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
            y_sqrt = tbe.vsqrt(var)
            y_add = tbe.vadds(y_sqrt, eps)
            res = tbe.vdiv(mean_sub, y_add)
        elif tbe_platform.get_soc_spec("SOC_VERSION") in (
                "Hi3796CV300ES", "Hi3796CV300CS"):
            y_sqrt = tbe.vsqrt(var, priority_flag=1)
            y_add = tbe.vadds(y_sqrt, eps)
            res = tbe.vdiv(mean_sub, y_add)
        else:
            y_sqrt = tbe.vsqrt(var)

            for _ in range(CONST_SQRT_ITER):
                data_sqrt = tbe.vdiv(var, y_sqrt)
                data_sqrt = tbe.vadd(data_sqrt, y_sqrt)
                data_sqrt = tbe.vmuls(data_sqrt, tvm.const(CONST_HALF, var.dtype))
                y_sqrt = data_sqrt

            y_add = tbe.vadds(y_sqrt, eps)
            res = tbe.vdiv(mean_sub, y_add)
    else:
        res = tbe.vsub(x, mean_broad)   # x - mean

    if is_cast:
        res = tbe.cast_to(res, dtype_x)

    return res

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                 para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def mvn(x, y, normalize_variance=True, across_channels=False,
        eps=1e-9, kernel_name="mvn"):
    """
    algorithm: MVN
    y = (x-mean(x))/(std(x) + eps)

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    normalize_variance: bool
        A bool value indicates the operation for normalize_variance.
    across_channels: bool
        A bool value indicates the operation for across_channels.
    eps: float
        A small float number added to the variance of x. Defaults to `1e-9`.
    kernel_name : str
        kernel name, default value is "mvn"
    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    input_dtype = dtype_x.lower()

    para_check.check_shape(shape_x, param_name="x")

    _check_dtype(input_dtype)

    data_format = x.get("format")
    _check_format_shape(data_format, shape_x)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_dtype)
    res = mvn_compute(x_input, y, normalize_variance, across_channels,
                      eps, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [x_input, res]}

    tbe.cce_build_code(sch, config)
