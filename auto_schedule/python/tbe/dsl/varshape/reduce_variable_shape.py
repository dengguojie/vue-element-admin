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
reduce variable shape
"""
from tbe.common.utils import para_check
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation

from tbe.dsl.base import var_api
from tbe.dsl.base import d_format_util
from tbe.dsl.base.operation import add_compile_info_inner


@register_variable("reduce")
def variable_shape(inputs):
    # type: (list) -> list
    """
    variable shape for reduce ops
    """
    if len(inputs) < 1:
        return []

    inputs_before_reduce, inputs_after_reduce, input_axis = [], [], []
    for single_input in inputs:
        input_type = single_input.get("rel_pos_to_reduce")
        if input_type == "axis":
            input_axis.append(single_input)
        elif input_type == "after":
            inputs_after_reduce.append(single_input)
        else:
            inputs_before_reduce.append(single_input)

    axis = input_axis[0].get("value")
    mode = inputs_before_reduce[0].get("mode")
    if mode is None:
        mode = para_check.ORIGINAL
    operation.get_context().add("_mode", mode)
    current_compute = operation.get_context().get_current_compute()
    if current_compute:
        current_compute.add("_mode", mode)
        current_compute.add("_shape", inputs_before_reduce[0]["shape"])
        ori_axis = input_axis[0].get("ori_axis")
        if ori_axis is not None:
            current_compute.add("_ori_axis", ori_axis)
        axis_dtype = input_axis[0].get("axis_dtype")
        if axis_dtype is not None:
            current_compute.add("_axis_dtype", axis_dtype)

    shape_local = [x["shape"] for x in inputs_before_reduce]
    range_local = [x.get("range") if x.get("range") else [(1, None)]*len(shape_local[0]) for x in inputs_before_reduce]
    shape_before_reduce, shape_after_reduce = [], []

    shape_format = inputs_before_reduce[0].get("format")
    if shape_format == "NC1HWC0":
        ori_shape = inputs_before_reduce[0].get("ori_shape")
        np_mapping = inputs_before_reduce[0].get("np_mapping")
        s_format = inputs_before_reduce[0].get("s_format")
        pad_axes = inputs_before_reduce[0].get("pad_axes")
        c_index = pad_axes.get("C")
        add_compile_info_inner("_ori_dim_index", c_index)
        current_compute.add("is_5hd_pattern", True)

        if ori_shape[c_index] == -1:
            c = operation.var_inner("_ori_dim_{}".format(str(c_index)), None, "int32",
                                    addition={"annotation": {"axis_type": "C"}})
        else:
            c = var_api.const(ori_shape[c_index], "int32", annotation={"axis_type": "C"})

        for index, (shape_i, format_i) in enumerate(zip(shape_local[0], s_format)):
            if format_i == 1:
                shape_before_reduce.append(shape_i)
                continue

            if shape_i == -1:
                _var = operation.var_inner("_dim_{}".format(str(index)), range_local[0][index], "int32",
                                           addition={"annotation": {"axis_type": format_i}})
            else:
                _var = var_api.const(shape_i, "int32", annotation={"axis_type": format_i})

            if isinstance(format_i, str) and format_i in np_mapping.keys():
                d_format_util.set_original(_var, c)

            shape_before_reduce.append(_var)
    else:
        for index in range(len(shape_local[0])):
            _var = None
            if shape_local[0][index] == -1:
                _var = operation.var_inner("_dim_{}".format(str(index)), range_local[0][index])
                shape_before_reduce.append(_var)
            else:
                shape_before_reduce.append(shape_local[0][index])

    def _gen_shape_after_reduce():
        for idx, dim_i in enumerate(shape_before_reduce):
            if idx in axis:
                if not len(inputs_after_reduce[0]["shape"]) == len(inputs_before_reduce[0]["shape"]):
                    continue
                shape_after_reduce.append(1)
            else:
                shape_after_reduce.append(dim_i)

    if inputs_after_reduce:
        _gen_shape_after_reduce()

    shape_out = []
    for single_input in inputs:
        input_type = single_input.get("rel_pos_to_reduce")
        if input_type == "axis":
            shape_out.append(input_axis[0].get("shape")[:])
        elif input_type == "after":
            shape_out.append(shape_after_reduce[:])
        else:
            shape_out.append(shape_before_reduce[:])

    return shape_out
