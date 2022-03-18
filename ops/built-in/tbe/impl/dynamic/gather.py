# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
gather
"""

from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.dynamic.gather_v2 import GatherV2


# 'pylint: disable=locally-disabled,invalid-name,unused-argument,too-many-arguments
def get_op_support_info(x, indices, y, validate_indices=True, batch_dims=0, kernel_name="Gather"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    format_indices = indices.get("format").upper()
    shape_indices_len = len(indices.get("shape"))
    if format_x == "ND" and format_indices == "ND":
        axis_split_matrix = []
        for j in range(shape_indices_len):
            split_0 = [SplitInput([1, [j], [-1], [-1]]), SplitOutput([0, [j]])]
            axis_split_matrix.append(split_0)
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def gather_tik(x, indices, y, validate_indices=True, batch_dims=0, kernel_name="Gather"):
    """
    gather interface for tik
    """
    axis_dict = {"dtype": "int32"}
    obj = GatherV2(x, indices, axis_dict, y, batch_dims, kernel_name)
    return obj.gather_compute()


def gather_compute(x, indices, y, validate_indices=True, batch_dims=0, kernel_name="gather"):
    """
    gather compute

    Parameters
    ----------
    x: input params shape, dtype and range
    indices: input indices shape, dtype and range
    y: output shape, dtype and range
    validate_indices: Whether to verify the values of indices, not currently enabled
    batch_dims: the number of batch dimensions
    kernel_name: kernel name of gather op

    Returns
    -------
    res: TVM tensor
        the result of gather
    """
    res = tbe.gather(x, indices, batch_dims + 1, batch_dims)

    return res


def gather_dsl(x, indices, y, validate_indices=True, batch_dims=0, kernel_name="gather"):
    """
    gather interface for dsl
    """
    check_list_x = (
        "float16", "float32", "int8", "uint8", "int32", "uint32", "int16", "uint16", "int64", "uint64", "bool")
    check_list_indices = ("int32", "int64")
    dtype_x = x.get("dtype").lower()
    dtype_indices = indices.get("dtype").lower()
    para_check.check_dtype(dtype_x, check_list_x, param_name="x")
    para_check.check_dtype(dtype_indices, check_list_indices, param_name="indices")

    # In the gather scenario, when batch_dims is not 0, set axis and batch_dims to the same value.
    batch_dims = "unknown" if batch_dims is None else batch_dims
    tbe_context.get_context().add_compile_info("attr_name", "batch_dims")
    ins = classify([x, indices, None, batch_dims], OpPatternMode.GATHER)
    schedules, tensors = [], []
    for shape_x, shape_indices, axis_input, batch_dims_input in ins:
        with tbe.compute():
            x_var, indices_var, axis_dim, batch_dims = \
                shape_util.variable_shape([shape_x, shape_indices, axis_input, batch_dims_input], "gather")
            x_tensor = tvm.placeholder(x_var, name="x", dtype=dtype_x)
            indices_tensor = tvm.placeholder(indices_var, name="indices", dtype=dtype_indices)

            res = gather_compute(x_tensor, indices_tensor, y, False, batch_dims, kernel_name)
            tensors.append([x_tensor, indices_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


@register_operator("Gather")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def gather(x, indices, y, validate_indices=True, batch_dims=0, kernel_name="gather"):
    """
    gather interface

    Parameters
    ----------
    x: input params shape, dtype and range
    indices: input indices shape, dtype and range
    y: output shape, dtype and range
    validate_indices: Whether to verify the values of indices, not currently enabled
    batch_dims: the number of batch dimensions
    kernel_name: kernel name of gather op

    Returns
    -------
    None
    """
    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        gather_dsl(x, indices, y, validate_indices, batch_dims, kernel_name)
    else:
        gather_tik(x, indices, y, validate_indices, batch_dims, kernel_name)
