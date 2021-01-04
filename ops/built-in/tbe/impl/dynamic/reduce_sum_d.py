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
dynamic reduce sum
"""
import te.lang.cce as tbe
from te import platform as tbe_platform
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from te.lang.base.operation import add_compile_info

NONETYPE = type(None)


# 'pylint: disable=unused-argument,invalid-name
def reduce_sum_d_compute(x,
                         y,
                         axis=None,
                         keepdims=None,
                         kernel_name="reduce_sum_d"):
    """redusce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    dtype = x.dtype
    cce_product = tbe_platform.get_soc_spec("SOC_VERSION")

    if cce_product not in ("Ascend310",) and dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        x = tbe.cast_to(x, "float32")
    res_sum = tbe.sum(x, axis=axis, keepdims=keepdims)
    res = tbe.cast_to(res_sum, dtype)

    return res


# 'pylint: disable=too-many-locals,invalid-name
@tbe_base.register_operator("ReduceSumD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_sum_d(x, y, axis=None, keepdims=None, kernel_name="reduce_sum_d"):
    """reduce a tensor on a certain axis based on sum.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    None
    """

    dtype = x["dtype"]
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_lower, check_list, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    add_compile_info("_ori_axis", axis)

    shape = x["shape"]
    shape_len = len(shape)
    if not axis:
        axis = range(shape_len)
    if hasattr(axis, 'index'):
        axis = list(axis)
    axis = shape_util.axis_check(shape_len, axis)
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}

    schedules = []
    tensors = []
    ins = tbe_base.shape_classifier.classify([x, input_axis], tbe_base.shape_classifier.Mode.REDUCE)

    for (x, axis) in ins:
        with tbe_base.compute():
            shape_var_new = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
            data_input = tvm.placeholder(shape_var_new, name="data_input",
                                         dtype=dtype_lower)
            res = reduce_sum_d_compute(data_input, y, axis.get("value"), keepdims)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
