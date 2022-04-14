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
slice variable shape
"""
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation


@register_variable("slice")
def variable_shape(inputs):
    # type: (list) -> list
    x_info = inputs[0]
    begin_info = inputs[1]
    end_info = inputs[2]

    current_compute = operation.get_context().get_current_compute()
    if 0 in x_info["shape"]:
        current_compute.add("_zero_shape", True)
    else:
        current_compute.add("_zero_shape", False)

    # new x var
    x_shape = []
    for index, value in enumerate(x_info["shape"]):
        _var = None
        if value == -1:
            _var = operation.var_inner("_x_dim_{}".format(index), x_info["range"][index])
            x_shape.append(_var)
        else:
            x_shape.append(value)

    dim_len = len(x_info["shape"])
    begin_list = []
    is_begin_list = isinstance(begin_info, list)
    end_list = []
    is_end_list = isinstance(end_info, list)
    for _idx in range(dim_len):
        if is_begin_list:
            begin_value = begin_info[_idx]
        else:
            if "lr_depad" in begin_info and _idx == 0:
                current_compute.add("_lr_depad", True)
                begin_value = operation.var_inner("_begin_dim_{}".format(_idx), (0, 0))
            else:
                begin_value = operation.var_inner("_begin_dim_{}".format(_idx), (0, None))
        begin_list.append(begin_value)

        if is_end_list:
            end_value = end_info[_idx]
        else:
            end_value = begin_value + operation.var_inner("_size_dim_{}".format(_idx), (1, None))
        end_list.append(end_value)

    return x_shape, begin_list, end_list
