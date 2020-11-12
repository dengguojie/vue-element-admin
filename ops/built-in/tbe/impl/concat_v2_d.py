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
from impl.concat_last_dim import ConcatWithVnchw
from impl.concat_last_dim import ConcatWith5HD
from impl.concat_tik import ConcatSchedule
from impl.concat_l1fusion import ConcatL1Fusion
from impl.util import util_select_op_base
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


def is_dynamic_shape(input_values):
    for input_value in input_values:
        if -1 in input_value.get("shape"):
            return True

    return False


# pylint: disable = unused-argument
def get_op_support_info(input_values,
                        output_data,
                        axis,
                        kernel_name="concat_v2_d"):
    value_len = len(input_values)
    shape_value_len = len(input_values[0].get("shape"))
    format_value = input_values[0].get("format").upper()
    if axis < 0:
        axis += shape_value_len
    if format_value == "ND" or format_value == "NC1HWC0":
        axis_split_matrix=[]
        for i in range(0, shape_value_len):
            if i != axis:
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
    select format dynamically
    """
    data_list = []
    datatype_5d_xhs = "float16,int32,int8,int16,int64,uint8,uint16,uint32," \
                      "uint64,bool,float16,int32,int8,int16,int64," \
                      "uint8,uint16,uint32,uint64,bool"
    format_5d_xhs = "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0," \
                    "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,ND,ND,ND,ND,ND," \
                    "ND,ND,ND,ND,ND"
    datatype_4d_xhs = "float16,int32,int8,int16,int64,uint8,uint16,uint32," \
                      "uint64,bool"
    format_4d_xhs = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
    datatype_5d = "float16,float,int32,int8,int16,int64,uint8,uint16,uint32," \
                  "uint64,bool,float16,float,int32,int8,int16,int64,uint8," \
                  "uint16,uint32,uint64,bool"
    format_5d = "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0," \
                "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,ND,ND,ND,ND,ND,ND,ND,ND," \
                "ND,ND,ND"
    datatype_4d = "float16,float,int32,int8,int16,int64,uint8,uint16," \
                  "uint32,uint64,bool"
    format_4d = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
    ori_format = input_values[0].get("ori_format").upper()
    for i, input_dict in enumerate(input_values):
        shape_input = input_dict.get("ori_shape")
        data_list.append(shape_input)
    divisible = 16
    nchw_len_axis = 0
    nhwc_len_axis = 0
    if len(data_list[0]) == 4:
        for list_element in data_list:
            if list_element[3] % divisible == 0:
                nhwc_len_axis += 1
            if list_element[1] % divisible == 0:
                nchw_len_axis += 1

    # add op_select_format for not align input with 5HD start
    concat_with_5hd_not_align = \
        ConcatWith5HD(input_values, output_data, axis, kernel_name)
    is_support_other_5hd = concat_with_5hd_not_align.check_op_select()
    if is_support_other_5hd:
        datatype_4d = datatype_4d + ",float16,int16,uint16"
        format_4d = format_4d + ",NC1HWC0,NC1HWC0,NC1HWC0"
        datatype_4d_xhs = datatype_4d_xhs + ",float16,int16,uint16"
        format_4d_xhs = format_4d_xhs + ",NC1HWC0,NC1HWC0,NC1HWC0"
    # add op_select_format for not align input with 5HD end

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        if ori_format == "NHWC" and len(data_list[0]) == 4:
            if _can_process_by_5hd(ori_format, axis, len(data_list), nhwc_len_axis):
                # NC1HWC0+ND
                input0 = util_select_op_base.gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_5d_xhs,
                    format=format_5d_xhs)
                output0 = util_select_op_base.gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_5d_xhs,
                    format=format_5d_xhs)
            else:
                # ND+
                input0 = util_select_op_base.gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_4d_xhs,
                    format=format_4d_xhs)
                output0 = util_select_op_base.gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_4d_xhs,
                    format=format_4d_xhs)
        elif ori_format == "NCHW" and len(data_list[0]) == 4:
            if _can_process_by_5hd(ori_format, axis, len(data_list), nchw_len_axis):
                # NC1HWC0+ND
                input0 = util_select_op_base.gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_5d_xhs,
                    format=format_5d_xhs)
                output0 = util_select_op_base.gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_5d_xhs,
                    format=format_5d_xhs)
            else:
                # ND+
                input0 = util_select_op_base.gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_4d_xhs,
                    format=format_4d_xhs)
                output0 = util_select_op_base.gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_4d_xhs,
                    format=format_4d_xhs)
        else:
            # ND
            input0 = util_select_op_base.gen_param(
                classify="input0",
                name="input_values",
                datatype=datatype_4d_xhs,
                format=format_4d_xhs)
            output0 = util_select_op_base.gen_param(
                classify="output0",
                name="output_data",
                datatype=datatype_4d_xhs,
                format=format_4d_xhs)
    else:
        if ori_format == "NHWC" and len(data_list[0]) == 4:
            if _can_process_by_5hd(ori_format, axis, len(data_list), nhwc_len_axis):
                # NC1HWC0+ND
                input0 = util_select_op_base.gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_5d,
                    format=format_5d)
                output0 = util_select_op_base.gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_5d,
                    format=format_5d)
            else:
                # ND+
                input0 = util_select_op_base.gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_4d,
                    format=format_4d)
                output0 = util_select_op_base.gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_4d,
                    format=format_4d)
        elif ori_format == "NCHW" and len(data_list[0]) == 4:
            if _can_process_by_5hd(ori_format, axis, len(data_list), nchw_len_axis):
                # NC1HWC0+ND
                input0 = util_select_op_base.gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_5d,
                    format=format_5d)
                output0 = util_select_op_base.gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_5d,
                    format=format_5d)
            else:
                # ND+
                input0 = util_select_op_base.gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_4d,
                    format=format_4d)
                output0 = util_select_op_base.gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_4d,
                    format=format_4d)
        else:
            # ND
            input0 = util_select_op_base.gen_param(
                classify="input0",
                name="input_values",
                datatype=datatype_4d,
                format=format_4d)
            output0 = util_select_op_base.gen_param(
                classify="output0",
                name="output_data",
                datatype=datatype_4d,
                format=format_4d)

    if is_dynamic_shape(input_values):
        input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                               datatype=datatype_4d, format=format_4d,
                                               unknownshape_format=format_4d)
        output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                datatype=datatype_4d, format=format_4d,
                                                unknownshape_format=format_4d)

    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _can_process_by_5hd(origin_format, axis, data_len, len_axis):
    if len_axis == data_len and origin_format[axis].upper() == 'C':
        return True
    return origin_format[axis].upper() in ('H', 'W')


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
    input_addr_type = input_values[0].get("addr_type")
    output_addr_type = output_data.get("addr_type")
    if output_addr_type == 1 or input_addr_type == 1:
        _concat_l1fusion = ConcatL1Fusion(input_values, output_data, axis, kernel_name)
        if_l1_support = _concat_l1fusion.check_support_l1_fusion()
        if if_l1_support:
            _concat_l1fusion.do_concat_l1fusion()
    if input_format == "NC1HWC0":
        axis = shape_util.axis_transform_5d(axis, ori_format)
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
            raise RuntimeError("The length of each shape must be equal")
    if axis < -dim_num or axis >= dim_num:
        raise RuntimeError("Axis value out of range")

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

