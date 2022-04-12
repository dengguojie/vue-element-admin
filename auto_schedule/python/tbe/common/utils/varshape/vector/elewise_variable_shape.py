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
elewise variable shape
"""
from functools import reduce

from tbe.common.utils import para_check
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation


@register_variable("elewise")
def variable_shape(inputs):
    # type: (list) -> list
    def _get_range_intersection(ranges):
        def _range_intersection(range_a, range_b):
            if range_a is None or range_b is None:
                return None
            a_lower, a_upper = range_a
            b_lower, b_upper = range_b
            if max(a_lower, b_lower) > min(a_upper, b_upper):
                return None
            return max(a_lower, b_lower), min(a_upper, b_upper)

        return reduce(_range_intersection, ranges)

    def _update_range(shapes, ranges):
        def _fixed_shape_range(shapes, ranges):
            for _range in ranges:
                for i, (r0, r1) in enumerate(_range):
                    if r0 is None and r1 is None:
                        _range[i] = (para_check.MAX_UNKNOWN_SHAPE_NUM, para_check.MAX_UNKNOWN_SHAPE_NUM)
                    elif r0 is None:
                        _range[i] = (para_check.MAX_UNKNOWN_SHAPE_NUM, r1)
                    elif r1 is None:
                        _range[i] = (r0, para_check.MAX_UNKNOWN_SHAPE_NUM)
            for _shape, _range in zip(shapes, ranges):
                for i, (s, (r0, r1)) in enumerate(zip(_shape, _range)):
                    if s != -1:
                        _range[i] = (s, s)
                    elif r0 == r1:
                        _shape[i] = r0

        _fixed_shape_range(shapes, ranges)
        t_shapes = list(map(list, zip(*shapes)))
        t_ranges = list(map(list, zip(*ranges)))
        for _shape, _range in zip(t_shapes, t_ranges):
            no_one_range = [r for r in _range if r[0] > 1]
            if len(no_one_range) > 0:
                mied_range = _get_range_intersection(no_one_range)
                if mied_range is None:
                    dict_args = {}
                    dict_args["errCode"] = "E90001"
                    dict_args["detailed_cause"] = "input shape error, shape range no intersection"
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                for i, r in enumerate(_range):
                    if 1 in r:
                        if r[1] < mied_range[0]:
                            _range[i] = (1, 1)
                        elif r[1] > mied_range[1]:
                            _range[i] = (1, mied_range[1])
                    else:
                        _range[i] = mied_range
        shapes = list(map(list, zip(*t_shapes)))
        ranges = list(map(list, zip(*t_ranges)))
        _fixed_shape_range(shapes, ranges)
        return shapes, ranges

    def _get_dim(_i, _shapes):
        return max(s[_i] for s in _shapes)

    def _extract(_inputs):
        def _complete(_in):
            shapes, ranges = [], []
            for x in _in:
                _shape, _range = list(x["shape"]), x.get("range")
                d_v = dim_length - len(_shape)
                x_shape = [1] * d_v + _shape
                x_range = [(1, 1)] * d_v + list(_range)
                shapes.append(x_shape)
                ranges.append(x_range)
            return shapes, ranges

        if support_broadcast:
            dim_length = max(len(s["shape"]) for s in _inputs)
            shapes, ranges = _complete(_inputs)
            shapes, ranges = _update_range(shapes, ranges)
            return shapes, ranges

        _shapes, _ranges = [], []
        for _input in inputs:
            _shapes.append(_input["shape"])
            _ranges.append(_input["range"])
        _shape = [_get_dim(_i, _shapes) for _i in range(len(_shapes[0]))]
        _shapes = [_shape.copy() for _ in range(len(_shapes))]

        return _shapes, _ranges

    def _maybe_broadcast():
        if support_broadcast:
            for _r in ranges:
                if _r[i][0] <= 1:
                    return True
        return False

    def _mode_process():
        if mode == para_check.CONST:
            if support_broadcast:
                input1 = inputs[0]["const_shape"]
                input2 = inputs[1]["const_shape"]
                const_shape = [a & b for a, b in zip(input1, input2)]
            else:
                const_shape = inputs[0]["shape"]
            operation.get_context().get_current_compute().add("_const_shape", const_shape)
        elif mode == para_check.SPECIAL and inputs[0].get("pattern"):
            pattern = inputs[0].get("pattern")
            operation.get_context().get_current_compute().add("_pattern", pattern)
            for i, _pattern in enumerate(pattern):
                if _pattern == para_check.COMMON:
                    for _, shape_j in enumerate(shapes):
                        if shape_j[i] == -1:
                            # mark this dimension dose not exist broadcast
                            shape_j[i] = -77
        elif mode == para_check.SPECIAL_SCALAR:
            pattern = inputs[0].get("pattern")
            operation.get_context().get_current_compute().add("_pattern", pattern)

    if len(inputs) < 1:
        return []
    mode = inputs[0].get("mode") or para_check.ORIGINAL
    current_compute = operation.get_context().get_current_compute()
    current_compute.add("_mode", mode)
    support_broadcast = operation.get_context().get("_support_broadcast") or False

    shapes, ranges = _extract(inputs)
    _mode_process()

    d_shapes = [[] for _ in shapes]
    for i in range(len(shapes[0])):
        _var = None
        need_two_vars = _maybe_broadcast()
        _suffix = 0
        for d_shape, shape, _range in zip(d_shapes, shapes, ranges):
            if shape[i] == -1 and _range[i][0] == _range[i][1]:
                d_shape.append(_range[i][0])
            elif shape[i] == -1:
                if _var is None or need_two_vars:
                    _var = operation.var_inner("_dim_{}_{}".format(str(i), str(_suffix)), _range[i])
                d_shape.append(_var)
            elif shape[i] == -77:
                # no broadcast
                if _var is None:
                    _var = operation.var_inner("_dim_{}_{}".format(str(i), str(_suffix)), _range[i])
                d_shape.append(_var)
            else:
                d_shape.append(shape[i])
            _suffix += 1

    return d_shapes
