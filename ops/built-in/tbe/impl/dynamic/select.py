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
select
"""
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.select import op_select_format as static_op_select_format
from impl.util.util_compute import check_support_fusion


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# 'pylint: disable=locally-disabled,too-many-statements,too-many-branches
# 'pylint: disable=get-dict-value-exception
def op_select_format(condition, x1, x2, y, kernel_name="select"):
    """1.when all input(condition, x1, x2) have the same ori_shape, ori_format,
       and the format is in ["NCHW", "NHWC", "HWCN"] or ["NDHWC", "DHWCN", "NCDHW"],
       the Op Select can support ND, FRACTAL_NZ, NC1HWC0 and FRACTAL_Z.

        for example:
        conditon : Tensor (shape=(16, 16, 16, 16), "NCHW")
        x1 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        x2 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        the Op Select can process with NC1HWC0:
        conditon : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        x1 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        x2 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")

    2.when all input(x1, x2) have the same ori_shape, ori_format, and the
       format is in ["NCHW", "NHWC", "HWCN"] or ["NDHWC", "DHWCN", "NCDHW"],
       and conditon is a scaler. The Op Select can support ND, FRACTAL_NZ,
       NC1HWC0 and FRACTAL_Z.

        for example:
        conditon : Tensor of (shape=(2), "NCHW")
        x1 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        x2 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        the Op Select can process with NC1HWC0:
        conditon : Tensor of (shape=(2), "NCHW")
        x1 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        x2 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    """
    return static_op_select_format(condition, x1, x2, y, kernel_name="select")


# 'pylint: disable=too-many-locals,invalid-name,unused-argument,too-many-statements
@register_operator_compute("Select", op_mode="dynamic", support_fusion=check_support_fusion)
def select_compute(condition, x1, x2, y, kernel_name="select"):
    """
    compute for select

    Parameters
    ----------
    condition: TVM tensor
        the placeholder of input condition
    x1: TVM tensor
        the placeholder of input x1
    x2: TVM tensor
        the placeholder of input x2
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    num_dtype = x1.dtype

    if num_dtype in ("int8", "uint8"):
        condition = tbe.cast_to(condition, "float16")
    elif num_dtype == "int32":
        if not tbe_platform.api_check_support("te.lang.cce.ceil", num_dtype):
            condition = tbe.cast_to(condition, "float16")
        condition = tbe.ceil(condition)
    else:
        condition = tbe.cast_to(condition, num_dtype)

    _, _, shape_max = shape_util.broadcast_shapes(shape_util.shape_to_list(condition.shape),
                                                  shape_util.shape_to_list(x1.shape),
                                                  param_name_input1="condition",
                                                  param_name_input2="x1")

    if num_dtype in ("int8", "uint8"):
        x1 = tbe.cast_to(x1, "float16")
        x2 = tbe.cast_to(x2, "float16")
        ones = tbe.broadcast(tvm.const(1, dtype="float16"), shape_max, output_dtype="float16")
    else:
        ones = tbe.broadcast(tvm.const(1, dtype=num_dtype), shape_max, output_dtype=num_dtype)

    condition = tbe.broadcast(condition, shape_max)
    x1 = tbe.broadcast(x1, shape_max)
    x2 = tbe.broadcast(x2, shape_max)

    condition_opp = tbe.vsub(ones, condition)

    temp_x = tbe.vmul(x1, condition)
    temp_y = tbe.vmul(x2, condition_opp)
    res = tbe.vadd(temp_x, temp_y)
    if num_dtype in ("int8", "uint8"):
        res = tbe.cast_to(res, num_dtype)
    return res


def x_target_compute(x_target, x2):
    """
    x_target_compute
    """
    x_target["shape"] = list(x_target["shape"])
    x_target["range"] = list(x_target["range"])
    for i, (_shape, _range) in enumerate(zip(x2["shape"], x2["range"])):
        x_target["shape"][i] = _shape if x_target["shape"][i] == -1 else x_target["shape"][i]
        _range_second = x_target["range"][i][1] if x_target["range"][i][1] is not None else _range[1]
        if x_target["range"][i][1] is not None and _range[1] is not None:
            _range_second = min(x_target["range"][i][1], _range[1])

        x_target["range"][i] = (max(x_target["range"][i][0], _range[0]), _range_second)
    x_target["shape"] = tuple(x_target["shape"])
    x_target["range"] = tuple(x_target["range"])
    return x_target


def condition_compute(shape_x1, con_shape, condition, kernel_name):
    """
    condition_compute
    """

    x_len = len(shape_x1)
    con_len = len(con_shape)

    if con_len == 1 and x_len != 1:
        if -1 != con_shape[0] != shape_x1[0] != -1 and con_shape[0] != 1:
            error_detail = "Shape of tensor condition and x1 dim[0] must be equal!"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "condition", "x1", error_detail)
        fill_lens = x_len - con_len
        fill_shape = [1] * fill_lens
        condition["ori_shape"] += tuple(fill_shape)
        condition["range"] += tuple([(1, 1)] * fill_lens)
        condition["shape"] += tuple(fill_shape)

    else:
        if con_len != x_len:
            error_detail = "dims of tensor condition and x1 must be equal!"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "condition", "x1", error_detail)
    return condition


@register_operator("Select")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def select(condition, x1, x2, y, kernel_name="select"):
    """
    Selects elements from `x1` or `x2`, depending on `condition`.
    And the x1 and x2 shape must be same.
    Parameters
    ----------
    condition: dict
        dict of condition, include keys(shape and dtype),
        only support int8,int32
    x1: dict
        dict of x1, only support float16, float32, int32, int8, uint8
    x2: dict
        dict of x2, only support float16, float32, int32, int8, uint8
    y: dict
        dict of output
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    None
    """
    shape_x1 = x1.get("shape")
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    con_shape = condition.get("shape")
    dtype = condition.get("dtype").lower()
    bool_dtype = "int8" if dtype == "bool" else dtype

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(dtype_x1, check_list, param_name="x1")

    if dtype_x1 != dtype_x2:
        error_detail = "Dtype of tensor x1 and x2 must be equal!"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x1", "x2", error_detail)

    x_target = x2.copy() if is_unknown_rank_input(x2) else x1.copy()
    if is_unknown_rank_input([x_target, condition]):
        x_target, condition = [x_target, x_target] if is_unknown_rank_input(x_target) else [condition, condition]
    else:
        x_target = x_target_compute(x1, x2)
        condition = condition_compute(shape_x1, con_shape, condition, kernel_name)

    ins = classify([condition, x_target], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_condition, _x1) in ins:
        with tbe.compute():
            shape_con, shape_x = shape_util.variable_shape([_condition, _x1])
            flag_cloud = tbe_platform.api_check_support("te.lang.cce.vsel", "float32")
            flag_dtype = dtype_x1 in ("float32", "int32")
            if (list(con_shape) != list(shape_x1)) or ((not flag_cloud) and flag_dtype):
                tensor_condition = tvm.placeholder(shape_con, name="condition", dtype=bool_dtype)
            else:
                tensor_condition = tvm.placeholder(shape_con, name="condition", dtype="int8")

            tensor_x = tvm.placeholder(shape_x, dtype_x1, "tensor_x")
            tensor_y = tvm.placeholder(shape_x, dtype_x1, "tensor_y")

            res = select_compute(tensor_condition, tensor_x, tensor_y, y, kernel_name)

            tensors.append([tensor_condition, tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    if list(con_shape) == list(shape_x1):
        config = {"name": kernel_name,
                  "tensor_list": tensors,
                  "bool_storage_as_1bit": False}
    else:
        config = {"name": kernel_name,
                  "tensor_list": tensors}

    tbe.build(schedules, config)
