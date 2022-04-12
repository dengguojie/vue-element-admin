#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
split variable shape
"""
from tbe.common.utils import shape_util
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation


@register_variable("split")
def variable_shape(inputs):
    # type: (list) -> list
    def check_params():
        if len(inputs) != 2:
            dict_args = {"errCode": "E90001", "detailed_cause": "split variable shape requires two input parameters:"
                         "input tensor and split numbers"}
            raise RuntimeError(dict_args, get_error_message(dict_args))
        split_num = len(inputs[1])
        split_input_number_limit = 63
        if split_num > split_input_number_limit or split_num <= 0:
            dict_args = {
                "errCode": "E90001",
                "detailed_cause": f"split numbers error, split numbers must be "
                                  f"greater 0 and less equal {split_input_number_limit}, now, it is {split_num}"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def shape_variable():
        input_vars = []
        for index, (shape_value, range_value) in enumerate(zip(shape_x, range_x)):
            if shape_value == -1:
                _var = operation.var_inner(f"_dim_{index}", range_value)
                input_vars.append(_var)
            else:
                input_vars.append(shape_value)
            if index == 0:
                current_compute.add("_ori_var", input_vars[-1])
                input_vars[-1] = (input_vars[-1] + split_factor - 1) // split_factor * split_factor
        return input_vars

    def get_split_range(r_value, split_num):
        if r_value is None:
            new_r_value = None
        elif r_value < split_num:
            new_r_value = 1
        else:
            new_r_value = r_value // split_num
        return new_r_value

    def split_value_variable():
        avg_split = operation.get_context().get("_avg_split")
        split_num = len(inputs[1])
        if avg_split:
            if shape_x[1] > -1:
                _var = shape_x[1] // split_num
                current_compute.add("_split_is_const", True)
            else:
                low_r = get_split_range(range_x[1][0], split_num)
                high_r = get_split_range(range_x[1][1], split_num)
                _var = operation.var_inner("_split_0", (low_r, high_r))
            split_vars = [_var for _ in range(split_num)]
        else:
            if split_num == 0 and shape_x[1] > -1:
                split_vars = [shape_x[1]]
            elif min(inputs[1]) > -1:
                return list(inputs[1])
            else:
                split_vars = []
                for index in range(split_num):
                    _var = operation.var_inner(f"_split_{index}", (min(1, range_x[1][0]), range_x[1][1]))
                    split_vars.append(_var)
        return split_vars

    check_params()

    input_x = inputs[0]
    shape_x = shape_util.shape_to_list(input_x.get("shape"))
    if len(shape_x) != 2:
        dict_args = {"errCode": "E90001", "detailed_cause": "split variable shape must be two dims"}
        raise RuntimeError(dict_args, get_error_message(dict_args))
    range_x = input_x.get("range")
    mode = input_x.get("mode")
    split_factor = input_x.get("split_factor")
    current_compute = operation.get_context().get_current_compute()
    current_compute.add("_mode", mode)

    return shape_variable(), split_value_variable()
