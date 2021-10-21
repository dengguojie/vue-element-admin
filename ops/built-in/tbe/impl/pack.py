#!/usr/bin/env python
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
pack
"""

from te import tvm
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.concat_v2_d import concat_v2_d
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable = unused-argument
def get_op_support_info(x, y, axis, kernel_name="pack"):
    """
    get_op_support_info
    """
    x_len = len(x)
    shape_x_len = len(x[0].get("shape"))
    format_x = x[0].get("format").upper()
    if axis < -1:
        axis = axis + 1
    if axis < 0:
        axis += shape_x_len
    if format_x in ("ND", "NC1HWC0"):
        axis_split_matrix = []
        for i in range(0, shape_x_len-1):
            if i != axis:
                input_list = []
                for j in range(0, x_len):
                    input_0 = [j, [i], [-1], [-1]]
                    input_list.append(input_0)
                split_0 = [SplitInput(*input_list), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _pad2concat_params(x, y, axis):
    if axis < -1:
        return x, y, axis + 1

    ori_shape = x[0].get("ori_shape")
    shape = x[0].get("shape")
    if len(shape) == len(ori_shape) and axis in (-1, len(shape)):
        for item in x:
            item["shape"] = list(item["shape"])
            item["ori_shape"] = list(item["ori_shape"])
            item["shape"].append(1)
            item["ori_shape"].append(1)

    return x, y, axis


@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def pack(x, y, axis, kernel_name="pack"):
    """
    algorithm: pack
    Concatenates tensors along one dimension.
    Parameters
    ----------
    x : A list of `dict`.dict include keys shape and dtype
    y: dict of output_data, dict include keys shape and dtype
    axis : int, in the range [-rank(values), rank(values)
    kernel_name : cce kernel name, default value is "pack"
    Returns
    -------
    None
    """
    check_list = ("int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16", "float32")
    data = []
    for i, input_dict in enumerate(x):
        shape_input = input_dict.get("shape")
        para_check.check_shape(shape_input, param_name="x")
        para_check.check_dtype(input_dict.get("dtype").lower(), check_list, param_name="x")
        input_dtype = (input_dict.get("dtype")).lower()
        data.append(tvm.placeholder(shape_input, name="data_%d" % i,
                                    dtype=input_dtype))

    left_value = -len((x[0].get("shape")))-1
    right_value = len((x[0].get("shape")))
    if axis < left_value or axis > right_value:
        expect_value = "[%s, %s]".format(str(left_value), str(right_value))
        error_manager_vector.raise_err_input_value_invalid("pack",
                                                           "axis",
                                                           expect_value, str(axis))

    x, y, axis = _pad2concat_params(x, y, axis)
    concat_v2_d(x, y, axis, kernel_name)
