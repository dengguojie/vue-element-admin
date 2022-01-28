"""
Copyright (C) 2020-2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

concat_d
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util import util_common
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator_compute
from impl.dynamic.concat_d_tik import concat_d_tik
from impl.concat_v2_d import get_op_support_info as concat_v2_get_op_support_info
from impl.concat_v2_d import op_select_format as concat_v2_op_select_format


# 'pylint: disable = unused-argument
# 'pylint: disable=consider-using-in
def get_op_support_info(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    get_op_support_info
    """
    return concat_v2_get_op_support_info(input_values, output_data, concat_dim, kernel_name)


# 'pylint: disable=locally-disabled,unused-argument,too-many-branches
# 'pylint: disable=too-many-locals,too-many-statements,unused-variable
def op_select_format(input_values, output_data, concat_dim,
                     kernel_name="concat"):
    """
    1. When input ori_format is in ["NDCHW", "HWCN", "NCHW"], and
       ori_format indexed by concat_dim is not C or N. When all
       of input's shape is same, and C axis is in [2, 4, 8]. Or
       all of input's shape is not same, C axis of output is
       greater then or equal to 16. The Op ConcatD can support
       NC1HWC0 and NDC1HWC0.
    > for example:
    > x : Tensor of (shape=(16, 16, 16, 16, 16), "NC1HWC0")

    2. When input ori_format is in ["NDCHW", "HWCN", "NCHW"], and
       ori_format indexed by concat_dim is not C. The Op
       ConcatD can support HWCN, NCHW and NDCHW.
    > for example:
    > x : Tensor of (shape=(16, 16, 16, 16), "NCHW")

    3. When length of input is greater then or equal to 2,
    concat_dim is the last dimension or second-to-last index.
    The Op ConcatD can support ND.
    > for example:
    > x : Tensor of (shape=(16, 16, 16, 16), "ND")
    """
    return concat_v2_op_select_format(input_values, output_data, concat_dim, kernel_name)


# 'pylint: disable=unused-argument
@register_operator_compute("ConcatD", op_mode="dynamic", support_fusion=False)
def concat_d_compute(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    algorithm: concat
    Concatenates tensors along one dimension.

    Parameters
    ----------
    datas : list of placeholders, all input data
    output : dict, dict of output
    concat_dim : scalar, in the range [-rank(values), rank(values))]
    kernel_name : string
        cce kernel name, default value is concat

    Returns
    -------
    res : placeholder and res
    """
    res = tbe.concat(input_values, concat_dim)

    return res


def concat_d_dsl(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    algorithm: concat
    Concatenates tensors along one dimension.

    Parameters
    ----------
    input_values : A list of `dict`.dict include keys shape and dtype
    output_data: dict of output_data, dict include keys shape and dtype
    concat_dim : scalar, in the range [-rank(values), rank(values))]
    kernel_name : cce kernel name, default value is "concat"
    Returns
    -------
    None
    """
    dtype_x = input_values[0].get("dtype")

    # update shape based on input format
    new_input_values = []
    for _, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get("ori_shape")
        if not (0 in shape_input or -1 in shape_input or -2 in shape_input):
            tensor_dict = util_common.update_shape_base_other_format(tensor_dict)
            new_input_values.append(tensor_dict)
    if len(input_values) == len(new_input_values):
        input_values = new_input_values

    # update axis base on input format
    for _idx, _input_dict in enumerate(input_values):
        ori_shape = _input_dict.get("ori_shape")
        if -2 not in ori_shape:
            # can not use unkownrank shape to update the axis
            input_format = _input_dict.get("format")
            ori_format = _input_dict.get("ori_format")
            concat_dim = util_common.update_axis_for_other_format(ori_shape, concat_dim, input_format, ori_format)
            break

    extra_params = {"axis": concat_dim}
    ins = classify([input_values], "concat", extra_params)
    schedules, tensors = [], []
    for (input_x_, axis_) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([input_x_], "concat")
            input_tensors = []
            for index, shape in enumerate(shape_x):
                data = tvm.placeholder(shape, dtype=dtype_x, name=f"data_{index}")
                input_tensors.append(data)
            res = concat_d_compute(input_tensors, output_data, axis_, kernel_name)

            tensors.append([*input_tensors, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


@register_operator("ConcatD")
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def concat_d(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    algorithm: concat
    Concatenates tensors along one dimension.

    Parameters
    ----------
    input_values : A list of `dict`.dict include keys shape and dtype
    output_data: dict of output_data, dict include keys shape and dtype
    concat_dim : scalar, in the range [-rank(values), rank(values))]
    kernel_name : cce kernel name, default value is "concat"
    Returns
    -------
    None
    """
    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        concat_d_dsl(input_values, output_data, concat_dim, kernel_name)
    else:
        concat_d_tik(input_values, output_data, concat_dim, kernel_name)
