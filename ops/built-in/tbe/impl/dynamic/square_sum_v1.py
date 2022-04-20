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
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output1, attr1, attr2=True, kernel_name="square_sum_v1"):
    """
    get_op_support_info
    """
    shape_x = shape_util.shape_to_list(input_x.get("shape"))
    axis_d = []
    axis_split = []
    for i, _ in enumerate(shape_x):
        axis_d.append(i)
    format_x = input_x.get("format").upper()
    if attr1 is None:
        attr1 = []
    for i in axis_d:
        if i not in attr1:
            axis_split.append(i)
    if format_x == "ND":
        if attr2:
            axis_split_matrix = []
            for i in axis_split:
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)
            axis_reduce_list = None
        else:
            axis_split_matrix = None
            axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def op_select_format(input_x, output1, attr1, attr2, kernel_name="square_sum_v1"):
    """
    select format dynamically
    op_select_format support desc:
    1. input_format always support 'ND'
    2. when ori_format is 'HWCN', input_format support 'FRACTAL_Z' or 'FRACTAL_NZ' in compile_static process
        for example:
            ori:
                input_x              shape = [5,5,16,16]           format = 'HWCN'
                output1              shape = []                    format = 'ND'
            format transformer:
                input_x              shape = [25,1,16,16]          format = 'FRACTAL_Z'
                output1              shape = []                    format = 'ND'
            ---------------------------------------------------------------------------
            ori:
                input_x              shape = [16,16]               format = 'ND'
                output1              shape = []                    format = 'ND'
            format transformer:
                input_x              shape = [1,1,16,16]          format = 'FRACTAL_NZ'
                output1              shape = []                    format = 'ND'

    """
    dtype = "float16, float"
    input_format = "ND, ND"
    output_format = "ND, ND"
    ori_shape = input_x.get("ori_shape")
    ori_format = input_x.get("ori_format")
    if attr1 is None:
        attr1 = [i for i in range(len(ori_shape))]
    if ori_format in ("HWCN",) and len(ori_shape) == 4 and ori_shape[-1] % 16 == 0 and ori_shape[-2] % 16 == 0 and list(
            attr1) == [0, 1, 2, 3]:
        dtype = "float16, float, float16, float"
        input_format = "ND, ND, FRACTAL_Z, FRACTAL_Z"
        output_format = "ND, ND, ND, ND"
    if len(ori_shape) >= 2 and ori_shape[-1] % 16 == 0 and ori_shape[-2] % 16 == 0 and \
       list(attr1) == [i for i in range(len(ori_shape))]:
        dtype = dtype + ", float16, float"
        input_format = input_format + ", FRACTAL_NZ, FRACTAL_NZ"
        output_format = output_format + ", ND, ND"
    input0 = gen_param(classify="input0", name="input_x", datatype=dtype, format=input_format,
                       unknownshape_format=input_format)
    output0 = gen_param(classify="output0", name="output1", datatype=dtype, format=output_format,
                        unknownshape_format=output_format)

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


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
    x_format = "ND"
    if input_x.op.attrs:
        if "format" in input_x.op.attrs:
            x_format = input_x.op.attrs["format"]

    if not attr1:
        for i, _ in enumerate(shape):
            axis_d.append(i)
    else:
        axis_d = attr1

    # when the input format is FRACTAL_NZ or FRACTAL_Z, only support reduce all the axis(except axis0)
    if x_format in ["FRACTAL_NZ", "FRACTAL_Z"]:
        axis_d = []
        for i, _ in enumerate(shape):
            axis_d.append(i)
        axis_d.remove(0)
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
    x_format = input_x.get("format")

    input_axis = {"shape": [len(attr1)], "value": attr1, "rel_pos_to_reduce": "axis"}
    input_x["rel_pos_to_reduce"] = "before"
    ins = classify([input_x, input_axis], OpPatternMode.REDUCE,
                   {"keepdims": attr2 is True})
    schedules, tensors = [], []

    for (_input_x, _axis) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_input_x, _axis], op_mode="reduce")[0]
            data_input = tvm.placeholder(shape_var_new, name="data_input", dtype=input_dtype)

            data_input.op.attrs["format"] = x_format
            res = square_sum_v1_compute(data_input, output1, _axis.get("value"),
                                        attr2, kernel_name)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
