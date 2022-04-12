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
concat variable shape
"""
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation


@register_variable("concat")
def variable_shape(inputs):
    # type: (list) -> list
    def add_zero_axis_var(_is_first_in, _first_var):
        if shape_x[0] == -1:
            if _is_first_in:
                _first_var = operation.var_inner(f"_dim_{index}_0", range_x[0])
                _is_first_in = False
            cur_shape.append(_first_var)
        else:
            cur_shape.append(shape_x[0])
        return _is_first_in, _first_var

    def add_one_axis_var():
        if shape_x[1] == -1:
            _var = operation.var_inner(f"_dim_{index}_1", range_x[1])
            cur_shape.append(_var)
        else:
            cur_shape.append(shape_x[1])

    if len(inputs) != 1:
        dict_args = {"errCode": "E90001", "detailed_cause": "concat input numbers error"}
        raise RuntimeError(dict_args, get_error_message(dict_args))
    is_first_in = True
    shape_out = []
    first_var = -1
    for index, x in enumerate(inputs[0]):
        shape_x = x.get("shape")
        range_x = x.get("range")
        cur_shape = []
        is_first_in, first_var = add_zero_axis_var(is_first_in, first_var)
        if len(shape_x) > 1:
            add_one_axis_var()
        shape_out.append(cur_shape)
    current_compute = operation.get_context().get_current_compute()
    mode = inputs[0][0].get("mode")
    current_compute.add("_mode", mode)
    return shape_out
