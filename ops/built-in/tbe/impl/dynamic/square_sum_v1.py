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
square_sum_v1
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode

MIN_FP32 = 2 ** (-126)
# min float16 value
MIN_FP16 = 2 ** (-24)
VALUE_ONE = 1

SHAPE_SIZE_LIMIT = 200000000


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
def reduce_sum_d_compute(x, y, axis=None, keepdims=None, kernel_name="reduce_sum_d"):
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

    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32"):
        x = tbe.cast_to(x, "float32")
    res_sum = tbe.reduce_sum(x, axis=axis, keepdims=keepdims)
    res = tbe.cast_to(res_sum, dtype)

    return res


def square_compute(input_x, output_y, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is square

    Returns
    -------
    res : tvm.tensor
        the result of square
    """
    res = tbe.vmul(input_x, input_x)
    return res


@register_operator_compute("SquareSumV1", op_mode="dynamic", support_fusion=True)
def square_sum_v1_compute(input_x, output1, attr1, attr2, kernel_name="square_sum_v1"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """
    shape = shape_util.shape_to_list(input_x.shape)
    axis_d = []
    if not attr1:
        for i, _ in enumerate(shape):
            axis_d.append(i)
    else:
        axis_d = attr1
    square = square_compute(input_x, {}, kernel_name)

    sum0 = reduce_sum_d_compute(square, {}, axis_d, keepdims=attr2, kernel_name=kernel_name)

    return sum0


@register_operator("SquareSumV1")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def square_sum_v1(input_x, output1, attr1, attr2=True, kernel_name="square_sum_v1"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()

    input_axis = {"shape": [len(attr1),], "value": attr1, "rel_pos_to_reduce": "axis"}
    input_x["rel_pos_to_reduce"] = "before"
    ins = classify([input_x, input_axis], OpPatternMode.REDUCE,
                   {"keepdims": attr2 is True})
    schedules, tensors = [], []

    for (_input_x, _axis) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_input_x, _axis], op_mode="reduce")[0]
            data_input = tvm.placeholder(shape_var_new, name="data_input", dtype=input_dtype)

            res = square_sum_v1_compute(data_input, output1, _axis.get("value"),
                                        attr2, kernel_name)
            tensors.append([data_input, res])


        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
