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
from te.utils import para_check
from impl import concat_v2_d
from impl.concat_v2_d import op_select_format as concat_v2_op_select_format
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# pylint: disable = unused-argument
def get_op_support_info(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    get_op_support_info
    """
    value_len = len(input_values)
    shape_value_len = len(input_values[0].get("shape"))
    format_value = input_values[0].get("format").upper()
    if concat_dim < 0:
        concat_dim += shape_value_len
    if format_value == "ND" or format_value == "NC1HWC0":
        axis_split_matrix=[]
        for i in range(0, shape_value_len-1):
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
    return concat_v2_op_select_format(input_values, output_data, concat_dim, kernel_name)


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
