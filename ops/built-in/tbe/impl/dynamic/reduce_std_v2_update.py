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
reduce_std_v2_update
"""
import operator as op
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpPatternMode


# 'pylint: disable=invalid-name,too-many-locals,unused-argument,too-many-arguments,too-many-branches
@register_operator_compute("reduce_std_v2_update", op_mode="dynamic", support_fusion=True)
def reduce_std_v2_update_compute(x, mean, dim, if_std, unbiased, keepdim, kernel_name="reduce_std_v2_update"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of X
    mean : TVM tensor
        the mean of X
    dim : intlist
        dimension to calculate
    if_std : bool
        control whether the output is standard deviation or variance, default value is False
    unbiased : bool
        control Bessel deviation, default value is True
    keepdim : bool
        hold dimension or not, default value is False
    kernel_name: str
        kernel name

    Returns
    -------
    output TVM tensors
    """
    x_type = x.dtype.lower()

    if x_type == "float16":
        x = tbe.cast_to(x, "float32")
        mean = tbe.cast_to(mean, "float32")

    shape_x = shape_util.shape_to_list(x.shape)

    reduce_ele = 1.0
    for i in shape_x:
        reduce_ele *= i
    dtype = x.dtype

    x_sub = tbe.vsub(x, mean)
    var_mul = tbe.vmul(x_sub, x_sub)

    if unbiased:
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

    var = tbe.reduce_sum(var_muls, axis=dim, keepdims=keepdim)

    if if_std:
        std = tbe.vsqrt(var, impl_mode="high_precision")
        if std.dtype != x_type:
            std = tbe.cast_to(std, x_type)
        return std

    if var.dtype != x_type:
        var = tbe.cast_to(var, x_type)
    return var


@register_operator("ReduceStdV2Update")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_std_v2_update(x, mean, output_var, dim, if_std=False, unbiased=True, keepdim=False,
                         kernel_name="reduce_std_v2_update"):
    """
    calculating data

    Parameters
    ----------
    x: dict
        input tensor
    mean: dict
        mean value of input tensor
    output_var: dict
        output, variance or standard deviation
    dim: list[int]
        dimension to calculate
    if_std : bool
        control whether the output is standard deviation or variance, default value is False
    unbiased: bool
        control Bessel deviation, default value is True
    keepdims: bool
        hold dimension or not, default value is False
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
    para_check.check_shape(shape_mean, param_name="mean")

    if not op.eq(shape_x, shape_mean):
        raise RuntimeError("the x and mean should have the same shape.")

    x["rel_pos_to_reduce"] = "before"
    mean["rel_pos_to_reduce"] = "before"

    dim = list(dim)
    dim = shape_util.axis_check(len(shape_x), dim)
    input_axis = {"shape": [len(dim), ], "value": dim, "rel_pos_to_reduce": "axis"}

    schedules, tensors = [], []
    ins = classify([x, mean, input_axis], OpPatternMode.REDUCE, {"keepdims": keepdim is True})

    for(_input_x, _mean, _axes) in ins:
        with tbe.compute():
            x_var_new, mean_var_new = shape_util.variable_shape([_input_x, _mean, _axes],
                                                                op_mode="reduce")[0:2]
            data_x = tvm.placeholder(x_var_new, name="data_x", dtype=dtype_x)
            data_mean = tvm.placeholder(mean_var_new, name="data_mean", dtype=dtype_x)
            res = reduce_std_v2_update_compute(data_x, data_mean, _axes.get("value"),
                                               if_std, unbiased, keepdim, kernel_name)
            tensors.append([data_x, data_mean, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)