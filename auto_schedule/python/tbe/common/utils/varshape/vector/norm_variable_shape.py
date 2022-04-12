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
norm variable shape
"""
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation


@register_variable("norm")
def variable_shape(inputs):
    # type: (list) -> list
    """
    variable shape for norm ops
    """
    def _gen_variable_inputs():
        _variable_outputs = []
        _output_non_duplicate_list = []
        # all inputs are unknown broadcast, vars according to shape_to_variable do not need to generate
        # 'unknown_broadcast: dim + input_type + index
        # 'others:  dim + index
        if mode_non_duplicate_list == ["broadcast_unknown"] * len(mode_non_duplicate_list):
            for _idx, _input_type in enumerate(input_type_non_duplicate_list):
                _shape = input_non_duplicate_list[_idx].get("shape")
                _range = input_non_duplicate_list[_idx].get("range")
                _single_output = []
                for _index, _dim_i in enumerate(_shape):
                    if _dim_i == -1:
                        _var = operation.var_inner(f"_dim_{_input_type}_{_index}", _range[_index])
                        _single_output.append(_var)
                        continue
                    _single_output.append(_dim_i)
                _output_non_duplicate_list.append(_single_output)
        else:
            _shape_after_variable = []
            for _index, _sv_i in enumerate(shape_to_variable):
                if _sv_i == -1:
                    _var = operation.var_inner(f"_dim_{_index}", range_to_variable[_index])
                    _shape_after_variable.append(_var)
                    continue
                _shape_after_variable.append(_sv_i)
            for _idx, _input_type in enumerate(input_type_non_duplicate_list):
                _shape = input_non_duplicate_list[_idx].get("shape")
                _single_output = []
                if mode_non_duplicate_list[_idx] == "broadcast_unknown":
                    _range = input_non_duplicate_list[_idx].get("range")
                    for _index, _dim_i in enumerate(_shape):
                        if _dim_i == -1:
                            _var = operation.var_inner(f"_dim_{_input_type}_{_index}", _range[_index])
                            _single_output.append(_var)
                            continue
                        _single_output.append(_dim_i)
                    continue
                for _index, _dim_i in enumerate(_shape):
                    if _dim_i == -1:
                        _single_output.append(_shape_after_variable[_index])
                        continue
                    _single_output.append(_dim_i)

                _output_non_duplicate_list.append(_single_output)

        for _, _input_type in enumerate(input_type_list):
            _variable_outputs.append(_output_non_duplicate_list[input_type_non_duplicate_list.index(_input_type)])

        return _variable_outputs

    shape_len = len(inputs[0].get("shape"))
    shape_to_variable = [-1] * shape_len
    range_to_variable = [(1, None) for _ in range(shape_len)]

    broadcast_axis = None
    exist_after_broadcast_input = False

    input_type_list = []
    # input type are not duplicated
    input_type_non_duplicate_list = []
    input_non_duplicate_list = []
    mode_non_duplicate_list = []

    norm_pattern = inputs[0].get("norm_pattern")

    for single_input in inputs:
        input_type = single_input.get("input_type")
        input_type_list.append(input_type)
        input_shape = single_input.get("shape")
        mode_str = single_input.get("mode")

        if mode_str == "broadcast_axis_known":
            broadcast_axis = single_input.get("broadcast_axis")

        if input_type not in input_type_non_duplicate_list:
            input_type_non_duplicate_list.append(input_type)
            mode_non_duplicate_list.append(mode_str)
            input_non_duplicate_list.append(single_input)

        if input_type == 0:
            # normalize shape_to_variable according to after broadcast inputs
            shape_to_variable = input_shape
            range_to_variable = single_input.get("range")
            exist_after_broadcast_input = True

    # normalize shape_to_variable according to before broadcast inputs when there is no after broadcast input
    if not exist_after_broadcast_input:
        for idx in range(shape_len):
            input_dim = [x.get("shape")[idx] for x in inputs]
            max_dim = max(input_dim)
            min_dim = min(input_dim)
            if max_dim > 1 or (max_dim == min_dim == 1):
                shape_to_variable[idx] = max_dim
                range_to_variable[idx] = (max_dim, max_dim)

    variable_inputs = _gen_variable_inputs()

    current_operator = operation.get_context()
    if current_operator:
        current_compute = current_operator.get_current_compute()
        if current_compute:
            # ori inputs of ops
            current_compute.add("_input_shapes", variable_inputs)
            if broadcast_axis is not None:
                current_compute.add("_broadcast_axis", broadcast_axis)
            current_compute.add("_norm_pattern", norm_pattern)

    return variable_inputs
