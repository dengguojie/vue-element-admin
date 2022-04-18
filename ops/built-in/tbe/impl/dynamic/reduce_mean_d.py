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
dynamic reduce mean
"""
import collections

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpImplMode


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals
# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("ReduceMeanD", op_mode="dynamic", support_fusion=True)
def reduce_mean_d_compute(x,
                          y,
                          axes,
                          keepdims=None,
                          kernel_name="reduce_mean_d",
                          impl_mode=OpImplMode.HIGH_PERFORMANCE,
                          is_5hdc=False):
    """reduce_mean_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axes: int, list, tuple or NoneType
        the axes for reduce.
    keepdims: bool or NoneType
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_mean_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape_x = x.shape

    reduce_elts = 1.0
    if isinstance(axes, collections.Iterable):
        for i in axes:
            if isinstance(shape_x[i], tvm.expr.IntImm):
                reduce_elts *= shape_x[i].value
            else:
                reduce_elts *= shape_x[i]
    else:
        reduce_elts = shape_x[axes]

    dtype = x.dtype
    if dtype == "float32":
        calc_dtype = "float32"
    elif dtype == "float16":
        cce_product = tbe_platform.get_soc_spec("SOC_VERSION")
        if not tbe_platform.api_check_support("te.lang.cce.sum",
                                              "float32"):
            calc_dtype = "float16"
        elif cce_product == "Ascend310" and impl_mode == OpImplMode.HIGH_PERFORMANCE:
            calc_dtype = "float16"
        else:
            calc_dtype = "float32"
    else:
        # int8 and uint8
        calc_dtype = "float16"

    if isinstance(reduce_elts, float):
        if reduce_elts == 0:
            if calc_dtype == "float16":
                nan_data = tvm.const(65504, dtype=calc_dtype)
            else:
                nan_data = tvm.const(2**62, dtype=calc_dtype)
            sum_data_shape0 = tbe.reduce_sum(x, axis=axes, keepdims=keepdims)
            vadds_data_shape0 = tbe.vadds(sum_data_shape0, nan_data)
            res = tbe.cast_to(vadds_data_shape0, dtype)
            return res

        cof = reduce_elts ** (-1)
        cof = tvm.const(cof, dtype=calc_dtype)
    else:
        cof = tbe.var("cof", dtype=calc_dtype)
        if calc_dtype == "float16":
            tbe.var("cof_empty", dtype=calc_dtype)
        tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", calc_dtype)

    if dtype != calc_dtype:
        data_input_tmp = tbe.cast_to(x, calc_dtype)
    else:
        data_input_tmp = x

    data_input_tmp = tbe.vmuls(data_input_tmp, cof)
    res = tbe.reduce_sum(data_input_tmp, axis=axes, keepdims=keepdims)

    if dtype != calc_dtype:
        if dtype in ("int8", "uint8"):
            res = tbe.cast_to(res, dtype, False)
        else:
            res = tbe.cast_to(res, dtype)

    return res


@register_operator("ReduceMeanD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_mean_d(input_x, output_y, axes,
                  keepdims=None, noop_with_empty_axes=False, kernel_name="reduce_mean_d",
                  impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    Reduce a tensor on a certa in axes based on mean.

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input
    output_y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keepdims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    noop_with_empty_axes : bool, NoneType
        useless attr to avoid dynamic reduce_mean_d compile error.
    kernel_name : str
        cce kernel name, default value is reduce_mean_d

    Returns
    -------
    None
    """
    dtype = input_x["dtype"]
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32", "int8", "uint8")
    para_check.check_dtype(dtype_lower, check_list)
    input_x["rel_pos_to_reduce"] = "before"

    shape = input_x["shape"]
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)
    axes = shape_util.axis_check(shape_len, axes)
    input_axis = {"shape": [len(axes), ], "value": axes, "rel_pos_to_reduce": "axis"}

    schedules = []
    tensors = []
    ins = classify([input_x, input_axis], OpPatternMode.REDUCE,
                   {"keepdims": keepdims is True})
    for (_input_x, _axes) in ins:
        with tbe.compute():
            # not support 5HD
            is_5hdc = False
            shape_var_new = shape_util.variable_shape([_input_x, _axes], op_mode="reduce")[0]
            data_input = tvm.placeholder(shape_var_new, name="data_input",
                                         dtype=dtype_lower)
            res = reduce_mean_d_compute(data_input, output_y, _axes.get("value"),
                                        keepdims, impl_mode=impl_mode,
                                        is_5hdc=is_5hdc)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
