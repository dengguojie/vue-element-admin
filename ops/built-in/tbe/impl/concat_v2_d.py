# /usr/bin/env python
# -*- coding:utf-8 -*-
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
concat_v2_d: Concatenates tensors along one dimension.
            The number of dimensions of input tensors must match,
            and all dimensions except 'axis' must be equal.
            tf ConcactV2 op

"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.concat_last_dim import ConcatWithVnchw
from impl.concat_last_dim import ConcatWith5HD
from impl.concat_tik import ConcatSchedule
from impl.util import util_select_op_base
from impl.util import util_common
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# pylint: disable = unused-argument
def get_op_support_info(input_values,
                        output_data,
                        axis,
                        kernel_name="concat_v2_d"):
    """
    get_op_support_info
    """
    value_len = len(input_values)
    shape_value_len = len(input_values[0].get("shape"))
    format_value = input_values[0].get("format").upper()
    ori_shape = input_values[0].get("ori_shape")
    if axis < 0:
        axis += len(ori_shape)
    ori_format = input_values[0].get("ori_format")
    axis = util_common.update_axis_for_other_format(ori_shape, axis, format_value, ori_format, True)
    if isinstance(axis, int):
        axis = [axis]
    if format_value in ("ND", "NC1HWC0", "NCHW", "NHWC"):
        axis_split_matrix = []
        for i in range(0, shape_value_len - 1):
            if i not in axis:
                input_list = []
                for j in range(0, value_len):
                    input_0 = [j, [i], [-1], [-1]]
                    input_list.append(input_0)
                split_0 = [SplitInput(*input_list), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# pylint: disable=locally-disabled,unused-argument,too-many-branches
# pylint: disable=too-many-locals,too-many-statements,unused-variable
# pylint: disable=too-many-boolean-expressions
def op_select_format(input_values,
                     output_data,
                     axis,
                     kernel_name="concat_v2_d"):
    """
    1. When input ori_format is in ["NDCHW", "HWCN", "NCHW"], and
       ori_format indexed by concat_dim is not C or N. When all
       of input's shape is same, and C axis is in [2, 4, 8]. Or
       all of input's shape is not same, C axis of output is
       greater then or equal to 16. The Op ConcatV2D can support
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
    shape_len = 1
    data_list = []
    ori_format = input_values[0].get("ori_format").upper()
    for i, input_dict in enumerate(input_values):
        shape_input = input_dict.get("ori_shape")
        shape_input = shape_util.scalar2tensor_one(shape_input)
        data_list.append(shape_input)
        if -2 not in shape_input:
            shape_len = len(shape_input)
    concat_dim = axis % shape_len

    # add op_select_format for not align input with 5HD start
    # like: m.2 + m,2 + m,2 = m,6
    concat_with_5hd_not_align = \
        ConcatWith5HD(input_values, output_data, axis, kernel_name)
    is_support_other_5hd = concat_with_5hd_not_align.check_op_select()
    # add op_select_format for not align input with 5HD end

    # charge the concat_dim whether align
    align_len = 16
    is_concat_dim_align = True
    for i, input_shape in enumerate(data_list[0:len(data_list) - 1]):
        if -2 in input_shape:
            is_concat_dim_align = False
            break
        if input_shape[concat_dim] % align_len != 0:
            is_concat_dim_align = False
            break

    # charge whether support 5HD 6HD
    hd_support_format = \
        util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) \
        + util_common.get_fused_format_str(["N", "H", "W", "C"])
    is_support_hd = False
    is_support_fz = False
    if ori_format in hd_support_format and len(ori_format) == shape_len:
        is_concat_with_c = ori_format[concat_dim] == "C"
        is_concat_with_n = ori_format[concat_dim] == "N"
        # hd condition:
        # 1. do not concat the tensor with c dim
        # 2. concat the tensor with c, and the C dim size align C0 for all input
        if not is_concat_with_c or (is_concat_with_c and is_concat_dim_align):
            is_support_hd = True
        # fz condition:
        # 1. do not concat the tensor with nc dim
        # 2. concat the tensor with nc, and the NC dim size align C0 for all input
        if (not is_concat_with_c and not is_concat_with_n) or is_concat_dim_align:
            is_support_fz = True

    # charge whether support FRACTAL_NZ
    is_support_nz = False
    if shape_len >= 2:
        is_concat_last_one_dim = concat_dim == shape_len - 1
        is_concat_last_second_dim = concat_dim == shape_len - 2
        # condition
        # 1. do not concat the tensor with the -1 or -2 dim
        # 2. concat the tensor with the -1 or -2 dim, and the concat dim size align C0 for all input
        if is_concat_dim_align or not (is_concat_last_one_dim or is_concat_last_second_dim):
            is_support_nz = True

    base_data_type = \
        ["float", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]
    other_data_type = ["float", "float16", "int16", "int32", "uint16", "uint32"]
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        base_data_type.remove("float")
        other_data_type.remove("float")

    dtype_base_out = base_data_type.copy()
    format_base_out = ["ND"] * len(dtype_base_out)
    if is_support_hd:
        other_format = "NC1HWC0" if shape_len == 4 else "NDC1HWC0"
        dtype_base_out = dtype_base_out + other_data_type
        format_base_out = format_base_out + [other_format] * len(other_data_type)
    if is_support_fz and not util_common.is_dynamic_input(input_values):
        other_format = "FRACTAL_Z" if shape_len == 4 else "FRACTAL_Z_3D"
        dtype_base_out = dtype_base_out + other_data_type
        format_base_out = format_base_out + [other_format] * len(other_data_type)
    if is_support_nz and not util_common.is_dynamic_input(input_values):
        other_format = "FRACTAL_NZ"
        dtype_base_out = dtype_base_out + other_data_type
        format_base_out = format_base_out + [other_format] * len(other_data_type)
    if is_support_other_5hd and not util_common.is_dynamic_input(input_values):
        other_data_type = ["float16", "int16", "uint16"]
        other_format = "NC1HWC0"
        dtype_base_out = dtype_base_out + other_data_type
        format_base_out = format_base_out + [other_format] * len(other_data_type)

    dtype_str = ",".join(dtype_base_out)
    format_str = ",".join(format_base_out)
    input0 = util_select_op_base.gen_param(
        classify="input0", name="input_values", datatype=dtype_str, format=format_str, unknownshape_format=format_str)
    output0 = util_select_op_base.gen_param(
        classify="output0", name="output_data", datatype=dtype_str, format=format_str, unknownshape_format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def concat_v2_d_compute(input_values,
                        output_data,
                        axis,
                        kernel_name="concat_v2_d"):
    """how to make concat_v2_d compute these tensors.
    -----------
    Parameters
    ----------
    input_values : A list of tensor objects .
    axis : scalar,in the range [-rank(values), rank(values))
    output_data : A dict resulting from concatenation of the input tensors
    kernel_name : cce kernel name, default value is "concat_v2_d"

    Returns
    -------
    res : the result of concat_v2_d_compute
    """
    res = tbe.concat(input_values, axis)

    return res


@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def concat_v2_d(input_values, output_data, axis, kernel_name="concat_v2_d"):
    """
    algorithm: concat_v2_d

    Parameters
    ----------
    input_values : A list of dict objects.
                 dict with keys(shape and dtype) of tensor
                 dtype only support float32, int8, int16, int32, int64, uint8,
                 uint16, uint32, uint64, float16
    output_data : A dict resulting from concatenation of the input tensors
    axis : scalar,in the range [-rank(values), rank(values))
    kernel_name : cce kernel name, default value is "concat_v2_d"

    Returns
    -------
    None
    """
    new_input_values = []
    for _, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get("ori_shape")
        if 0 not in shape_input:
            tensor_dict = util_common.update_shape_base_other_format(tensor_dict)
            new_input_values.append(tensor_dict)
    input_values = new_input_values

    shape_value = []
    for _, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get("ori_shape")
        shape_value.append(shape_input)
    first_input_shape = input_values[0].get("ori_shape")
    if axis < 0:
        axis_new = len(first_input_shape) + axis
    else:
        axis_new = axis
    for _, element_shape in enumerate(shape_value):
        for j, _ in enumerate(first_input_shape):
            if element_shape[j] != first_input_shape[j] and j != axis_new:
                raise RuntimeError("Axes must equal except merge axis")

    # when format is 5HD check whether concat by C and redefine the axis
    input_format = input_values[0].get("format")
    ori_format = input_values[0].get("ori_format")

    # update axis base on input format
    axis = util_common.update_axis_for_other_format(shape_value[0], axis, input_format, ori_format)

    if input_format == "NC1HWC0":
        # check whether use 5HD when input is not align
        concat_with_5hd_not_align = \
            ConcatWith5HD(input_values, output_data, axis, kernel_name)
        is_support_other_5hd = concat_with_5hd_not_align.check_5hd_vnchw()
        if is_support_other_5hd:
            concat_with_5hd_not_align.do_5hd_concat_cut_by_batch()
            return

    # do check for input
    dim_num = len(input_values[0].get("shape"))

    # check the length of input shape must be equal
    for _, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get("shape")
        if len(shape_input) != dim_num:
            rule_desc = "The length of each shape must be equal"
            error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "shape_input",
                                                              len(shape_input))
    if axis < -dim_num or axis >= dim_num:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "axis", -len(dim_num),
                                                                len(dim_num), axis)

    # begin to check where user branch concat_last_dim with command nchwvconv
    concat_l = ConcatWithVnchw(input_values, output_data, kernel_name)
    if_vnchw_support = concat_l.check_vnchw_supported()
    if if_vnchw_support:
        concat_l.do_concat_vnchw()
        return
    # end to check where user branch concat_last_dim with command nchwvconv

    # begin to check where user branch concat tik
    concat_s = ConcatSchedule(input_values, output_data, axis, kernel_name)
    if_tik_support = concat_s.check_tik_supported()

    if if_tik_support:
        concat_s.concat_compute()
        return
    # end to check where user branch concat tik

    check_list = ("float32", "int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16")
    data = []
    for i, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get("shape")
        para_check.check_shape(shape_input, param_name="input_values")
        inp_dtype = tensor_dict.get("dtype").lower()
        para_check.check_dtype(inp_dtype, check_list, param_name="input_values")
        data.append(tvm.placeholder(shape_input, name="data_%d" % i,
                                    dtype=inp_dtype))

    res = concat_v2_d_compute(data, output_data, axis, kernel_name)
    data.append(res)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": data}

    tbe.cce_build_code(sch, config)
