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
cross_entropy variable shape
"""
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation


class Constant:
    """
    The class for constant
    """
    MAX_INT32_VALUE = 2147483647


# 'pylint: disable=too-many-locals,too-many-statements
@register_variable("softmax_norm")
def variable_shape(inputs: list):
    """
    :param inputs: all inputs
    :return:
    """
    def _fill(_inputs):
        x_0, x_1 = _inputs
        shape0, range0 = list(x_0["shape"]), list(x_0["range"])
        shape1, range1 = list(x_1["shape"]), list(x_1["range"])

        swapped = False
        if len(shape0) < len(shape1):
            shape0, range0, shape1, range1 = shape1, range1, shape0, range0
            swapped = True
        d_v = len(shape0) - len(shape1)
        shape1 = [1] * d_v + shape1
        range1 = [(1, 1)] * d_v + range1
        if swapped:
            shape0, range0, shape1, range1 = shape1, range1, shape0, range0
        return [shape0, shape1], [range0, range1]

    def _maybe_broadcast():
        for _r in ranges:
            if _r[i][0] <= 1:
                return True
        return False

    mode = inputs[0].get("mode")
    if mode is None:
        mode = para_check.ORIGINAL
    operation.get_context().add("mode", mode)
    current_compute = operation.get_context().get_current_compute()
    if current_compute:
        current_compute.add("_mode", mode)
        ori_axis = inputs[0].get("ori_axis")
        if ori_axis is not None:
            current_compute.add("ori_axis", ori_axis)
        axis_dtype = inputs[0].get("axis_dtype")
        if axis_dtype is not None:
            current_compute.add("axis_dtype", axis_dtype)

    shapes, ranges = _fill(inputs)

    d_shapes = [[] for _ in shapes]
    for i in range(len(shapes[0])):
        _var = None
        need_two_vars = _maybe_broadcast() and "copy" not in mode
        _suffix = 0
        for d_shape, shape, _range in zip(d_shapes, shapes, ranges):
            if shape[i] == -1 and _range[i][0] == _range[i][1]:
                operation.var("dim_" + str(_suffix) + "_" + str(i), (1, Constant.MAX_INT32_VALUE))
                d_shape.append(_range[i][0])
            elif shape[i] == -1:
                if _var is None or need_two_vars:
                    _var = operation.var("dim_" + str(_suffix) + "_" + str(i), _range[i])
                else:
                    operation.var("dim_" + str(_suffix) + "_" + str(i), _range[i])
                d_shape.append(_var)
            else:
                operation.var("dim_" + str(_suffix) + "_" + str(i), (1, Constant.MAX_INT32_VALUE))
                d_shape.append(shape[i])
            _suffix += 1

    return d_shapes
