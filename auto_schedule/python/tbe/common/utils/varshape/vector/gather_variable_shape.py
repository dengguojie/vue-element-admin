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
gather variable shape
"""
from itertools import chain

from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation


@register_variable("gather")
def variable_shape(inputs):
    # type: (list) -> list
    params_info = inputs[0]
    indices_info = inputs[1]

    gather_mode = operation.get_context().get("_gather_mode")

    if gather_mode == "gather":
        axis_info = inputs[2]
        batch_dims_info = inputs[3]
        rank_info = 1
    else:
        axis_info = 0
        batch_dims_info = inputs[2]
        rank_info = indices_info["shape"][-1]

    current_compute = operation.get_context().get_current_compute()
    current_compute.add("_axis", axis_info)
    current_compute.add("_rank", rank_info)
    current_compute.add("_params_shape", params_info["shape"])
    current_compute.add("_indices_shape", indices_info["shape"])

    # zeros shape condition
    if 0 in chain(list(params_info["shape"]) + list(indices_info["shape"][:-1])):
        if gather_mode == "gather":
            return params_info["shape"], indices_info["shape"], axis_info, batch_dims_info
        return params_info["shape"], indices_info["shape"], batch_dims_info

    # new params var
    params_shape = []
    for index, value in enumerate(params_info["shape"]):
        _var = None
        if value == -1:
            _var = operation.var_inner("_params_dim_{}".format(index), params_info["range"][index])
            params_shape.append(_var)
        else:
            params_shape.append(value)

    # new indices var
    indices_shape = []
    # batch dims must same
    indices_shape.extend(params_shape[:batch_dims_info])
    for index, value in enumerate(indices_info["shape"][batch_dims_info:]):
        _var = None
        if value == -1:
            _var = operation.var_inner("_indices_dim_{}".format(index + batch_dims_info),
                                       indices_info["range"][index + batch_dims_info])
            indices_shape.append(_var)
        else:
            indices_shape.append(value)

    if gather_mode == "gather":
        return params_shape, indices_shape, axis_info, batch_dims_info
    return params_shape, indices_shape, batch_dims_info
