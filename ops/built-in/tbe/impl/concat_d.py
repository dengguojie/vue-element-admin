# /usr/bin/env python
# -*- coding:utf-8 -*-
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
concat_d
"""
import te.platform as tbe_platform
from te.utils import para_check
from impl import concat_last_dim
from impl import concat_v2_d
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
def get_op_support_info(input_values, output_data, concat_dim, kernel_name="concat"):
    value_len = len(input_values)
    shape_value_len = len(input_values[0].get("shape"))
    format_value = input_values[0].get("format").upper()
    if concat_dim < 0:
        concat_dim += shape_value_len
    if format_value == "ND" or format_value == "NC1HWC0":
        axis_split_matrix=[]
        for i in range(0, shape_value_len):
            if i != concat_dim:
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
def op_select_format(input_values, output_data, concat_dim,
                     kernel_name="concat"):
    """
    select format dynamically
    """
    data_list = []
    datatype_5d_xhs = "float16,int32,int8,int16,int64,uint8,uint16,uint32," \
                      "uint64,bool,float16,int32,int8,int16,int64," \
                      "uint8,uint16,uint32,uint64,bool"
    format_5d_xhs = "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0," \
                    "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,ND,ND,ND,ND,ND,ND," \
                    "ND,ND,ND,ND"
    datatype_4d_xhs = \
        "float16,int32,int8,int16,int64,uint8,uint16,uint32,uint64,bool"
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
        concat_last_dim.ConcatWith5HD(input_values, output_data, concat_dim, kernel_name)
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
            if _can_process_by_5hd(ori_format, concat_dim, len(data_list), nhwc_len_axis):
                # NC1HWC0+ND
                input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                       datatype=datatype_5d_xhs,
                                                       format=format_5d_xhs)
                output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                        datatype=datatype_5d_xhs,
                                                        format=format_5d_xhs)
            else:
                # ND+
                input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                       datatype=datatype_4d_xhs,
                                                       format=format_4d_xhs)
                output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                        datatype=datatype_4d_xhs,
                                                        format=format_4d_xhs)
        elif ori_format == "NCHW" and len(data_list[0]) == 4:
            if _can_process_by_5hd(ori_format, concat_dim, len(data_list), nchw_len_axis):
                # NC1HWC0+ND
                input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                       datatype=datatype_5d_xhs,
                                                       format=format_5d_xhs)
                output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                        datatype=datatype_5d_xhs,
                                                        format=format_5d_xhs)
            else:
                # ND+
                input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                       datatype=datatype_4d_xhs,
                                                       format=format_4d_xhs)
                output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                        datatype=datatype_4d_xhs,
                                                        format=format_4d_xhs)
        else:
            # ND
            input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                   datatype=datatype_4d_xhs,
                                                   format=format_4d_xhs)
            output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                    datatype=datatype_4d_xhs,
                                                    format=format_4d_xhs)
    else:
        if ori_format == "NHWC" and len(data_list[0]) == 4:
            if _can_process_by_5hd(ori_format, concat_dim, len(data_list), nhwc_len_axis):
                # NC1HWC0+ND
                input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                       datatype=datatype_5d, format=format_5d)
                output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                        datatype=datatype_5d, format=format_5d)
            else:
                # ND+
                input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                       datatype=datatype_4d, format=format_4d)
                output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                        datatype=datatype_4d, format=format_4d)
        elif ori_format == "NCHW" and len(data_list[0]) == 4:
            if _can_process_by_5hd(ori_format, concat_dim, len(data_list), nchw_len_axis):
                # NC1HWC0+ND
                input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                       datatype=datatype_5d, format=format_5d)
                output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                        datatype=datatype_5d, format=format_5d)
            else:
                # ND+
                input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                       datatype=datatype_4d, format=format_4d)
                output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                        datatype=datatype_4d, format=format_4d)
        else:
            # ND
            input0 = util_select_op_base.gen_param(classify="input0", name="input_values",
                                                   datatype=datatype_4d, format=format_4d)
            output0 = util_select_op_base.gen_param(classify="output0", name="output_data",
                                                    datatype=datatype_4d, format=format_4d)

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


@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
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
    # concat_d is the same as concat_v2_d
    # use concat_v2_d to replace
    concat_v2_d.concat_v2_d(input_values, output_data, concat_dim, kernel_name)
