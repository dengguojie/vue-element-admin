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
dynamic clip_by_norm_no_div_sum
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode

SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
def greater_select_compute(x, y, z, out, operation, kernel_name="greater_select"):
    """
    if x is greater than y, then return 1, else return 0.
    Parameters
    ----------
    x: TVM tensor
        the placeholder of input x
    y: TVM tensor
        the placeholder of input y
    z: TVM tensor
        the placeholder of input z
    out: dict
           dict of out
    operation: str
        function tbe.vcmpsel(lhs, rhs, operation, slhs, srhs)
        lt: lhs < rhs  is ture return slhs ,is false return srhs
        gt: lhs > rhs  is ture return slhs ,is false return srhs
        le: lhs <= rhs  is ture return slhs ,is false return srhs
        ge: lhs >= rhs  is ture return slhs ,is false return srhs
        eq: lhs == rhs  is ture return slhs ,is false return srhs
        ne: lhs != rhs  is ture return slhs ,is false return srhs
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_x = shape_util.shape_to_list(x.shape)
    shape_y = shape_util.shape_to_list(y.shape)
    shape_z = shape_util.shape_to_list(z.shape)
    _, _, broadcast_shape_ = shape_util.broadcast_shapes(shape_x, shape_y,
                                                         param_name_input1="x", param_name_input2="y")

    _, _, broadcast_shape = shape_util.broadcast_shapes(broadcast_shape_, shape_z,
                                                        param_name_input1="x", param_name_input2="z")
    data_x = tbe.broadcast(x, broadcast_shape)
    data_y = tbe.broadcast(y, broadcast_shape)
    data_z = tbe.broadcast(z, broadcast_shape)

    res = tbe.vcmpsel(data_x, data_y, operation, data_x, data_z)
    return res


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
    shape1 = shape_util.shape_to_list(x1.shape)
    shape2 = shape_util.shape_to_list(x2.shape)
    shape1 = shape_util.scalar2tensor_one(shape1)

    shape2 = shape_util.scalar2tensor_one(shape2)

    shape1, shape2, shape_max = shape_util.broadcast_shapes(shape1, shape2, param_name_input1="x1",
                                                            param_name_input2="x2")

    x1 = tbe.broadcast(x1, shape_max)
    x2 = tbe.broadcast(x2, shape_max)

    res = tbe.vsel(condition, x1, x2)
    return res


def greater_compute(x, y, z, kernel_name="greater"):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    x : Tensor
        input data_x
    y : Tensor
        input data_y
    z : dict
        shape and dtype of output data_z
    kernel_name : str
        cce kernel name, default value is "greater"

    Returns
    -------
    the result
    """
    shape_x = shape_util.shape_to_list(x.shape)
    shape_y = shape_util.shape_to_list(y.shape)
    dtype = x.dtype.lower()
    shape_x, shape_y, shape = shape_util.broadcast_shapes(shape_x, shape_y,
                                                          param_name_input1="x", param_name_input2="y")
    data_x = tbe.broadcast(x, shape)
    data_y = tbe.broadcast(y, shape)

    res = tbe.vcmp(data_x, data_y, 'gt', 'bool')

    return res


def maximum_compute(input_x, input_y, output_z, kernel_name="maximum"):
    """
    calculating data maximum

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of input data
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    result: TVM tensor
        the result of sqrt
    """
    shape1 = shape_util.shape_to_list(input_x.shape)
    shape2 = shape_util.shape_to_list(input_y.shape)
    shape1 = shape_util.scalar2tensor_one(shape1)

    shape2 = shape_util.scalar2tensor_one(shape2)

    shape1, shape2, shape_max = shape_util.broadcast_shapes(shape1, shape2, param_name_input1="select1_result",
                                                            param_name_input2="maximum_ones")

    data1_tmp1 = tbe.broadcast(input_x, shape_max)
    data2_tmp1 = tbe.broadcast(input_y, shape_max)
    res = tbe.vmax(data1_tmp1, data2_tmp1)
    return res


@register_operator_compute("ClipByNormNoDivSum", op_mode="dynamic", support_fusion=True)
def clip_by_norm_no_div_sum_compute(data_input_x,
                                    data_greater_zeros,
                                    data_select_ones,
                                    data_maximum_ones,
                                    y,
                                    kernel_name="clip_by_norm_no_div_sum"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """

    # greater_select1
    select_result = greater_select_compute(data_input_x, data_greater_zeros, data_select_ones, {}, "gt", kernel_name)

    # sqrt
    sqrt_result = tbe.vsqrt(select_result)

    # greater_select2
    select1_result = greater_select_compute(data_input_x, data_greater_zeros, sqrt_result, {}, "le", kernel_name)

    res = maximum_compute(select1_result, data_maximum_ones, {}, kernel_name)

    return res


@register_operator("ClipByNormNoDivSum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def clip_by_norm_no_div_sum(x, greater_zeros, select_ones, maximum_ones, y,
                            kernel_name="clip_by_norm_no_div_sum"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    schedules, tensors = [], []
    ins = classify([x, greater_zeros, select_ones, maximum_ones], OpPatternMode.ELEWISE_WITH_BROADCAST)

    for (_x, _greater, _zeros, _maximum) in ins:
        with tbe.compute():
            shape_x, shape_greater, shape_zeros, shape_maximum = shape_util.variable_shape([_x,
                                                                                            _greater,
                                                                                            _zeros,
                                                                                            _maximum])

            data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
            data_greater = tvm.placeholder(shape_greater, dtype=dtype_x, name="data_greater")
            data_zeros = tvm.placeholder(shape_zeros, dtype=dtype_x, name="data_zeros")
            data_maximum = tvm.placeholder(shape_maximum, dtype=dtype_x, name="data_maximum")
            res = clip_by_norm_no_div_sum_compute(data_x, data_greater, data_zeros, data_maximum, y, kernel_name)
            tensors.append([data_x, data_greater, data_zeros, data_maximum, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              }
    tbe.build(schedules, config)
