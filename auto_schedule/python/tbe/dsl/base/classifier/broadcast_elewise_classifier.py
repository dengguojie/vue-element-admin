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
classifier of shape in broadcast elewise
"""
from typing import Any
from typing import Dict
from typing import Optional
import copy
from functools import reduce
from enum import Enum
from enum import auto

from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation

from . import util

COMMON = "common"
BROADCAST = "broadcast"
UNKNOWN = "UNKNOWN"
SCALAR = "scalar"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
CONST = "const"
ORIGINAL = "original"
EMPTY = "empty"
VAR_BOUND_LIMIT = 2147483647
MAX_BROADCAST_INPUT = 70
UNKNOWN_RANK = -2
MAX_RANK = 8


class BroadcastElewiseClassifier:
    """
    Elewise with broadcast classifier
    """

    def __init__(self, ins: list, extra_params: Optional[Dict[str, Any]]):
        """
        init
        :param ins:
        """
        self.ins = ins
        extra_params = {} if extra_params is None else extra_params
        self.disable_optimization = extra_params.get("disable_optimization", False)
        self.is_unknown_rank = False
        self.maybe_empty_tensor = False

    def _init(self):
        shapes = [x["shape"] for x in self.ins]
        self.dim_length = max(len(s) for s in shapes)
        operation.get_context().add("_unknown_rank", self.is_unknown_rank)

        self.completed_ins = self._complete()
        self.completed_shapes = [x["shape"] for x in self.completed_ins]
        self.completed_ranges = [x["range"] for x in self.completed_ins]

        self._update_shape_range()
        self.f_shapes, self.f_ranges, fusion_index = _simplify_shape(self.completed_shapes,
                                                                     self.completed_ranges, self.disable_optimization)
        operation.add_compile_info_inner("_fusion_index", fusion_index)

        self.normalize_shapes = self._normalize()

    def classify(self):
        """
        classify
        :return:
        """
        self._init()
        if len(self.completed_shapes) > MAX_BROADCAST_INPUT:
            dict_args = {}
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "more than 70 input are not supported"
            raise RuntimeError(dict_args, get_error_message(dict_args))
        return self._classify_const() if self._is_const() else self._classify_var()

    def _complete(self):
        def clone_complete(_in):
            _shape, _range = list(_in["shape"]), _in.get("range")
            d_v = self.dim_length - len(_shape)

            in_x = _in.copy()
            in_x["shape"] = [1] * d_v + _shape
            in_x["range"] = util.generate_range(_shape) if _range is None else \
                [(1, 1)] * d_v + list(_range)
            return in_x

        return [clone_complete(x) for x in self.ins]

    def check_update_unknown_rank(self):
        is_unknown_rank = False
        for _in in self.ins:
            shapes = list(_in["shape"])
            if UNKNOWN_RANK in shapes:
                if len(shapes) != 1:
                    dict_args = {}
                    dict_args["errCode"] = "E90001"
                    dict_args["detailed_cause"] = "if the shape contains -2, it must be [-2] or (-2,)"
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                _in["shape"] = [-1] * MAX_RANK
                _in["range"] = [(1, None)] * MAX_RANK
                is_unknown_rank = True
        self.is_unknown_rank = is_unknown_rank

    def check_update_empty_shape(self):
        is_empty_shape = False
        for _in in self.ins:
            shapes, ranges = list(_in["shape"]), list(_in.get("range"))
            for index, (shape, (r0, r1)) in enumerate(zip(shapes, ranges)):
                if shape == 0:
                    ranges[index] = (0, 0)
                    is_empty_shape = True
                if r0 == 0:
                    ranges[index] = (1, r1)
                    is_empty_shape = True
                if r1 == 0:
                    ranges[index] = (0, 0)
                    is_empty_shape = True
            _in["range"] = ranges
        self.maybe_empty_tensor = is_empty_shape

    def _normalize(self):
        normalize_shapes = copy.deepcopy(self.f_shapes)
        for i in range(len(normalize_shapes)):
            for j in range(len(normalize_shapes[i])):
                if normalize_shapes[i][j] > 1 or \
                        (normalize_shapes[i][j] == -1 and self.f_ranges[i][j][0] > 1):
                    normalize_shapes[i][j] = ShapeValueType.COMMON
                elif normalize_shapes[i][j] == -1:
                    normalize_shapes[i][j] = ShapeValueType.UNKNOWN
                else:
                    normalize_shapes[i][j] = ShapeValueType.ONE
        return normalize_shapes

    def _update_shape_range(self):
        def get_range_intersection(ranges):
            def range_intersection(range_a, range_b):
                if range_a is None or range_b is None:
                    return None
                a_lower, a_upper = range_a
                b_lower, b_upper = range_b
                if max(a_lower, b_lower) > min(a_upper, b_upper):
                    return None
                return max(a_lower, b_lower), min(a_upper, b_upper)

            return reduce(range_intersection, ranges)

        def fixed_shape_range():
            for _range in self.completed_ranges:
                for i, (r0, r1) in enumerate(_range):
                    if r0 is None and r1 is None:
                        _range[i] = (VAR_BOUND_LIMIT, VAR_BOUND_LIMIT)
                    elif r0 is None:
                        _range[i] = (VAR_BOUND_LIMIT, r1)
                    elif r1 is None:
                        _range[i] = (r0, VAR_BOUND_LIMIT)
            for _shape, _range in zip(self.completed_shapes, self.completed_ranges):
                for i, (s, (r0, r1)) in enumerate(zip(_shape, _range)):
                    if s != -1:
                        _range[i] = (s, s)
                    elif r0 == r1:
                        _shape[i] = r0

        fixed_shape_range()
        t_shapes = list(map(list, zip(*self.completed_shapes)))
        t_ranges = list(map(list, zip(*self.completed_ranges)))
        for _shape, _range in zip(t_shapes, t_ranges):
            no_one_range = [r for r in _range if r[0] > 1]
            if len(no_one_range) > 0:
                mied_range = get_range_intersection(no_one_range)
                if mied_range is None:
                    dict_args = dict()
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
        self.completed_shapes = list(map(list, zip(*t_shapes)))
        self.completed_ranges = list(map(list, zip(*t_ranges)))
        fixed_shape_range()

    def _is_const(self):
        if len(self.completed_shapes) > 2:
            return False
        for i in range(self.dim_length):
            dims_i = [s[i] for s in self.completed_shapes]
            min_dim_v, max_dim_v = min(dims_i), max(dims_i)
            if min_dim_v == -1 and max_dim_v in (-1, 1):
                return False

        return True

    def _classify_const(self):
        def divide(i, _shapes):
            if i == self.dim_length:
                return [_shapes]

            dims_i = [s[i] for s in self.completed_shapes]
            min_dim_v, max_dim_v = min(dims_i), max(dims_i)

            # 1. don't need divide, all const value in the current axis
            if min_dim_v != -1:
                append_i(_shapes, i)
                return divide(i + 1, _shapes)

            # 2. -1 divide to 1 and the dim value, need cover broadcast
            ret_shapes = []

            # 2.1. -1 -> 1, x -> x
            _shapes_copy = copy(_shapes)
            append_b(_shapes_copy, i, 1)
            ret_shapes.extend(divide(i + 1, _shapes_copy))

            # 2.2. -1 -> x, x -> x
            _shapes_copy = copy(_shapes)
            append_b(_shapes_copy, i, max_dim_v)
            ret_shapes.extend(divide(i + 1, _shapes_copy))

            return ret_shapes

        def append_i(_shapes, dim_i):
            for i, _shape in enumerate(_shapes):
                _shape.append(self.completed_shapes[i][dim_i])

        def append_b(_shapes, dim_i, dim_v):
            for i, _shape in enumerate(_shapes):
                _shape.append(max(self.completed_shapes[i][dim_i], dim_v))

        def copy(_shapes):
            return [_shape.copy() for _shape in _shapes]

        def gen_const_range(_shapes):
            ranges = []
            for shape in _shapes:
                ranges.append([(s, s) for s in shape])
            return ranges

        ret = []
        for shapes in divide(0, [[] for _ in self.completed_ins]):
            const_range = gen_const_range(shapes)
            fused_shape, _, _ = _simplify_shape(shapes, const_range, self.disable_optimization)
            fused_shape = list(map(list, zip(*fused_shape)))
            ret.append([ConstMode.gen_in(fused_shape[0], shapes[0]), ConstMode.gen_in(fused_shape[1], shapes[1])])

        return ret

    def _classify_var(self):
        def merge_shape_range(left, right, common_need_update):
            shape = [1] * input_length
            _range = [(1, 1)] * input_length
            for i in range(left, right + 1):
                shape = ShapeSimplifier.combine_dim(self.f_shapes[i], shape)
                _range = ShapeSimplifier.combine_range(self.f_ranges[i], _range)
            if common_need_update:
                for i, (s, (_, r1)) in enumerate(zip(shape, _range)):
                    if s != 1:
                        shape[i] = -1
                        _range[i] = (1, r1)
            return shape, _range

        def adapter_broadcast_pattern(common_need_update, b_index, _b_shape, _b_range):
            b_pattern = []
            if not common_need_update:
                b_pattern = self.f_shapes[b_index[0]]
            if len(known_broadcast_index) > 0:
                b_pattern = known_broadcast_pattern[0]
            if len(b_pattern) > 0:
                for index, pattern in enumerate(b_pattern):
                    if pattern == ShapeValueType.ONE:
                        _b_shape[index] = 1
                        _b_range[index] = (1, 1)

        def update_common(_a_shape, _a_range):
            max_shape = max(_a_shape)
            if max_shape != -1:
                return [max_shape] * len(_a_shape), [(max_shape, max_shape)] * len(_a_shape)
            return _a_shape, _a_range

        def gen_common():
            if len(known_broadcast_pattern) > 0:
                return [], []
            if left_no_one > right_no_one:
                a_shape = [1] * input_length
                a_range = [(1, 1)] * input_length
            else:
                a_shape, a_range = merge_shape_range(left_no_one, right_no_one, False)
                a_shape, a_range = update_common(a_shape, a_range)
            return [a_shape], [a_range]

        def find_broadcast(location_b):
            def find_common_broadcast():
                a_index = [left_no_one, dim_length - 2]
                b_index = [left_no_one + 1, dim_length - 1]
                if len(known_broadcast_index) > 0:
                    a_index = [left_no_one, known_broadcast_index[0] - 1]
                if len(known_const_index) > 0:
                    b_index = [known_const_index[-1] + 1, dim_length - 1]
                return a_index, b_index

            def find_broadcast_common():
                b_index = [0, right_no_one - 1]
                a_index = [1, right_no_one]
                if len(known_broadcast_index) > 0:
                    a_index = [known_broadcast_index[-1] + 1, right_no_one]
                if len(known_const_index) > 0:
                    b_index = [0, known_const_index[0] - 1]
                return a_index, b_index

            def find_common_broadcast_common():
                a_index = [[left_no_one, right_no_one - 2],
                           [left_no_one + 2, right_no_one]]
                b_index = [left_no_one + 1, right_no_one - 1]
                if len(known_broadcast_index) > 0:
                    a_index = [[left_no_one, known_broadcast_index[0] - 1],
                               [known_broadcast_index[-1] + 1, right_no_one]]
                    if len(known_const_index) > 0:
                        for kci in known_const_index:
                            if kci < known_broadcast_index[0]:
                                b_index[0] = kci + 1
                            if kci > known_broadcast_index[-1]:
                                b_index[1] = kci - 1
                                break
                return a_index, b_index

            if location_b == SpecialMode.RIGHT:
                return find_common_broadcast()
            if location_b == SpecialMode.LEFT:
                return find_broadcast_common()
            if location_b == SpecialMode.MIDDLE:
                return find_common_broadcast_common()

        def gen_common_broadcast():
            after_known_broadcast_has_const = len(known_const_index) > 0 and len(known_broadcast_index) > 0 \
                                              and known_const_index[-1] > known_broadcast_index[0]
            befor_known_broadcast_no_common = len(known_broadcast_index) > 0 and \
                                              left_no_one >= known_broadcast_index[0]
            last_no_broadcast_first_no_common = dim_length - 1 in known_const_index or left_no_one >= dim_length - 1
            no_common_broadcast = after_known_broadcast_has_const or \
                                  befor_known_broadcast_no_common or last_no_broadcast_first_no_common
            if no_common_broadcast:
                return [], []
            a_index, b_index = find_broadcast(SpecialMode.RIGHT)
            common_need_update = b_index[0] != b_index[1]
            a_shape, a_range = merge_shape_range(a_index[0], a_index[1], common_need_update)
            a_shape, a_range = update_common(a_shape, a_range)
            b_shape, b_range = merge_shape_range(b_index[0], b_index[1], common_need_update)
            adapter_broadcast_pattern(common_need_update, b_index, b_shape, b_range)
            return [a_shape, b_shape], [a_range, b_range]

        def gen_broadcast_common():
            befer_known_broadcast_has_const = len(known_const_index) > 0 and len(known_broadcast_index) > 0 \
                                              and known_const_index[0] < known_broadcast_index[0]
            after_known_broadcast_no_common = len(known_broadcast_index) > 0 and \
                                              right_no_one <= known_broadcast_index[-1]
            first_no_broadcast_last_no_common = 0 in known_const_index or right_no_one <= 0
            no_broadcast_common = befer_known_broadcast_has_const or \
                                  after_known_broadcast_no_common or first_no_broadcast_last_no_common
            if no_broadcast_common:
                return [], []
            a_index, b_index = find_broadcast(SpecialMode.LEFT)
            common_need_update = b_index[0] != b_index[1]
            a_shape, a_range = merge_shape_range(a_index[0], a_index[1], common_need_update)
            a_shape, a_range = update_common(a_shape, a_range)
            b_shape, b_range = merge_shape_range(b_index[0], b_index[1], common_need_update)
            adapter_broadcast_pattern(common_need_update, b_index, b_shape, b_range)
            return [b_shape, a_shape], [b_range, a_range]

        def gen_common_broadcast_common():
            def all_const():
                return all(i in known_const_index for i in range(left_no_one + 1, right_no_one))

            right_left_no_common = len(known_broadcast_index) > 0 and \
                    (right_no_one <= known_broadcast_index[-1] or left_no_one >= known_broadcast_index[0])
            no_broadcast = right_no_one <= left_no_one or all_const()
            if right_left_no_common or no_broadcast:
                return [], []
            a_index, b_index = find_broadcast(SpecialMode.MIDDLE)
            common_need_update = b_index[0] != b_index[1]
            a1_shape, a1_range = merge_shape_range(a_index[0][0], a_index[0][1], common_need_update)
            a1_shape, a1_range = update_common(a1_shape, a1_range)
            a2_shape, a2_range = merge_shape_range(a_index[1][0], a_index[1][1], common_need_update)
            a2_shape, a2_range = update_common(a2_shape, a2_range)
            b_shape, b_range = merge_shape_range(b_index[0], b_index[1], common_need_update)
            adapter_broadcast_pattern(common_need_update, b_index, b_shape, b_range)
            return [a1_shape, b_shape, a2_shape], [a1_range, b_range, a2_range]

        def gen_broadcast():
            if len(known_const_index) > 0:
                return [], []
            b_shape, b_range = merge_shape_range(0, dim_length - 1, False)
            common_need_update = True
            b_index = []
            if dim_length == 1:
                common_need_update = False
                b_index = [0]
            adapter_broadcast_pattern(common_need_update, b_index, b_shape, b_range)
            return [b_shape], [b_range]

        def check_pattern(pattern_key, ranges):
            is_legal_pattern = True
            for pattern, _range in zip(pattern_key, ranges):
                if pattern == 'B' and is_legal_pattern:
                    is_legal_pattern = any(r0 <= 1 for (r0, _) in _range)
            return is_legal_pattern

        def add_special():
            ins_list = []
            special_pattern = {
                SpecialMode.COMMON: gen_common(),
                SpecialMode.COMMON_BROADCAST: gen_common_broadcast(),
                SpecialMode.COMMON_BROADCAST_COMMON: gen_common_broadcast_common(),
                SpecialMode.BROADCAST_COMMON: gen_broadcast_common(),
            }
            if len(self.completed_shapes) > 2:
                special_pattern[SpecialMode.BROADCAST] = gen_broadcast()
            for key, value in special_pattern.items():
                if len(value[0]) > 0 and check_pattern(key, value[1]):
                    ins_list.append(SpecialMode.gen_ins(list(zip(*value[0])), list(zip(*value[1])), key))
            return ins_list

        def add_special_scalar():
            ins_list = []
            shapes = list(zip(*self.f_shapes))
            ranges = list(zip(*self.f_ranges))
            if len(shapes) != 2:
                return ins_list
            # SA
            is_sa = SpecialScalarMode.maybe_all_one(shapes[0], ranges[0]) and \
                    not SpecialScalarMode.must_all_one(shapes[1])

            # AS
            is_as = not SpecialScalarMode.must_all_one(shapes[0]) and \
                    SpecialScalarMode.maybe_all_one(shapes[1], ranges[1])

            match_list = [is_sa, is_as]
            for match, pattern, shape_list in zip(match_list, SpecialScalarMode.PATTERNS,
                                                  SpecialScalarMode.SHAPES_LIST):
                if match:
                    x_0 = SpecialScalarMode.gen_in(shape_list[0], pattern)
                    x_1 = SpecialScalarMode.gen_in(shape_list[1], pattern)
                    ins_list.append([x_0, x_1])

            return ins_list

        def add_original():
            if _get_broadcast_axis_size(self.normalize_shapes) <= 1 and not self.disable_optimization:
                return []

            t_shapes = list(map(list, zip(*self.f_shapes)))
            t_ranges = list(map(list, zip(*self.f_ranges)))
            ins = []
            for shape, _range in zip(t_shapes, t_ranges):
                in_x = OriginalMode.gen_in(shape, _range)
                ins.append(in_x)

            return [ins]

        def add_empty():
            if not (self.maybe_empty_tensor or self.is_unknown_rank):
                return []
            input_length = len(self.completed_shapes)
            ins = [EmptyMode.gen_in()] * input_length
            return [ins]

        def get_known_broadcast_and_const(n_shapes):
            def _all_const(shape):
                return all([s == ShapeValueType.COMMON for s in shape])

            def _pattern_equal(last_pattern, cur_pattern):
                for last, current in zip(last_pattern, cur_pattern):
                    if last != current and ((last == ShapeValueType.ONE and current == ShapeValueType.COMMON) or
                                            (last == ShapeValueType.COMMON and current == ShapeValueType.ONE)):
                        return False
                return True

            def _update_pattern(last_pattern, cur_pattern):
                new_pattern = []
                for last, current in zip(last_pattern, cur_pattern):
                    if last != current and ShapeValueType.ONE in (last, current):
                        new_pattern.append(ShapeValueType.ONE)
                    elif last != current and ShapeValueType.COMMON in (last, current):
                        new_pattern.append(ShapeValueType.COMMON)
                    else:
                        new_pattern.append(last)
                return new_pattern

            known_broadcast_pattern = []
            known_broadcast_index = []
            known_const_index = []
            new_broadcast = True
            for i, n_s in enumerate(n_shapes):
                if not new_broadcast and _is_known_broadcast(n_s):
                    if not _pattern_equal(known_broadcast_pattern[-1], n_s):
                        known_broadcast_pattern.append(n_s)
                    else:
                        known_broadcast_pattern[-1] = _update_pattern(known_broadcast_pattern[-1], n_s)
                    known_broadcast_index.append(i)
                elif new_broadcast and _is_known_broadcast(n_s):
                    known_broadcast_pattern.append(n_s)
                    known_broadcast_index.append(i)
                    new_broadcast = False
                elif _all_const(n_s):
                    known_const_index.append(i)
                    new_broadcast = True
            return known_broadcast_pattern, known_broadcast_index, known_const_index

        def get_no_one_index():
            left_no_one = 0
            right_no_one = len(self.normalize_shapes) - 1
            for i, ns in enumerate(self.normalize_shapes):
                if ShapeValueType.ONE not in ns or ShapeValueType.COMMON in ns:
                    break
                left_no_one += 1
            for i in range(right_no_one, -1, -1):
                if ShapeValueType.ONE not in self.normalize_shapes[i] or \
                        ShapeValueType.COMMON in self.normalize_shapes[i]:
                    break
                right_no_one -= 1
            return left_no_one, right_no_one

        def has_unknown_broadcast():
            for n_shapes in self.normalize_shapes:
                if ShapeValueType.UNKNOWN in n_shapes:
                    return True
            return False

        known_broadcast_pattern, known_broadcast_index, known_const_index = \
            get_known_broadcast_and_const(self.normalize_shapes)
        left_no_one, right_no_one = get_no_one_index()
        ret = []
        input_length = len(self.completed_shapes)
        dim_length = len(self.f_shapes)
        if len(known_broadcast_pattern) <= 1 and not self.disable_optimization:
            ret.extend(add_special())
            ret.extend(add_special_scalar())
        ret.extend(add_original())
        ret.extend(add_empty())
        if not self.disable_optimization and dim_length > 2 and has_unknown_broadcast():
            unknown_len = dim_length - 1
            shapes = [[-1] * unknown_len] * input_length
            ranges = [[(1, None)] * unknown_len] * input_length
            pattern = SpecialMode.All_UNKNOWN
            ret.append(SpecialMode.gen_ins(shapes, ranges, pattern))
            operation.get_context().add("_has_all_unknown", True)
        return ret


def _simplify_shape(completed_shapes, completed_ranges, disable_optimization):
    input_length = len(completed_shapes)
    transpose_shapes = list(map(list, zip(*completed_shapes)))
    transpose_ranges = list(map(list, zip(*completed_ranges)))

    if disable_optimization:
        fusion_index = [[i] for i in range(len(transpose_shapes))]
        return transpose_shapes, transpose_ranges, fusion_index

    f_shapes = [[1] * input_length]
    f_ranges = [[(1, 1)] * input_length]

    fusion_index = []
    current_index = []

    all_one = str(ShapeValueType.ONE) * input_length
    state = all_one
    for index, (s, r) in enumerate(zip(transpose_shapes, transpose_ranges)):
        status = ShapeSimplifier.get_state(s, r)
        state_i = ''.join(list(map(str, status)))
        operator = ShapeSimplifier.get_operator(state, state_i)

        if operator == ShapeSimplifier.Operator.FUSED:
            f_shapes[-1] = ShapeSimplifier.combine_dim(f_shapes[-1], s)
            f_ranges[-1] = ShapeSimplifier.combine_range(f_ranges[-1], r)
            current_index.append(index)
        else:
            f_shapes.append(list(s))
            f_ranges.append(list(r))
            fusion_index.append(current_index)
            current_index = [index]

        if state_i != all_one:
            state = state_i

    fusion_index.append(current_index)
    return f_shapes, f_ranges, fusion_index


def _is_known_broadcast(shape):
    return ShapeValueType.COMMON in shape and ShapeValueType.ONE in shape


def _get_broadcast_axis_size(shapes):
    broadcast_axis_size = 0
    for s in shapes:
        if ShapeValueType.UNKNOWN in s or ShapeValueType.ONE in s:
            broadcast_axis_size += 1
    return broadcast_axis_size


class ShapeValueType:
    """
    there are three shape value type
    """
    # 1
    ONE = 1
    # const or (-1 and range not contains 1)
    COMMON = 2
    # -1 and the range contains 1
    UNKNOWN = 3


class SpecialMode:
    """
    SpecialMode const
    """
    # The (BROADCAST,) pattern covered by special scalar mode
    COMMON = 'A'
    COMMON_BROADCAST = 'AB'
    COMMON_BROADCAST_COMMON = 'ABA'
    BROADCAST_COMMON = 'BA'
    BROADCAST = 'B'

    All_UNKNOWN = ('U', 'U', 'U')

    LEFT = 'left'
    RIGHT = 'right'
    MIDDLE = 'middle'

    @classmethod
    def gen_ins(cls, shape, _range, pattern):
        """
        generate inputs
        :param pattern_status:
        :param pattern: pattern of special
        :return:
        """

        def gen_in(s, r):
            def _get_pattern():
                pattern_list = []
                for p in pattern:
                    if p == 'A':
                        pattern_list.append(COMMON)
                    elif p == 'B':
                        pattern_list.append(BROADCAST)
                    elif p == 'U':
                        pattern_list.append(UNKNOWN)
                return pattern_list

            return {"shape": s,
                    "range": r,
                    "support_broadcast": True,
                    "mode": SPECIAL,
                    "pattern": _get_pattern()
                    }

        shapes = []
        for s, r in zip(shape, _range):
            shapes.append(gen_in(s, r))
        return shapes


class SpecialScalarMode:
    """
    SpecialScalarMode
    """
    PATTERNS = [
        (SCALAR, BROADCAST),
        (BROADCAST, SCALAR),
    ]

    SHAPES_LIST = [
        [[1], [-1]],
        [[-1], [1]],
    ]

    @classmethod
    def maybe_all_one(cls, shape, _range):
        """
        Is it possible to take 1 for all inputs
        :param shape: input shape
        :return:
        """
        return all((s == 1 or (s == -1 and r0 <= 1)) for s, (r0, _) in zip(shape, _range))

    @classmethod
    def must_all_one(cls, shape):
        """
        all input values 1
        :param shape: input shape
        :return:
        """
        return all(s == ShapeValueType.ONE for s in shape)

    @classmethod
    def gen_in(cls, shape, pattern):
        """
        generate input
        :param shape:
        :param pattern:
        :return:
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "support_broadcast": True,
                "mode": SPECIAL_SCALAR,
                "pattern": pattern
                }


class OriginalMode:
    """
    Original Mode
    """

    @classmethod
    def gen_in(cls, shape, _range):
        """
        generate input
        :param shape:
        :return:
        """
        return {"shape": shape,
                "range": _range,
                "support_broadcast": True,
                "mode": ORIGINAL,
                }


class EmptyMode:
    """
    Empty Mode
    """

    @classmethod
    def gen_in(cls):
        """
        generate input
        :return:
        """
        return {"shape": (0, ),
                "range": [(0, 0)],
                "support_broadcast": True,
                "mode": EMPTY,
                }


class ConstMode:
    """
    Const Mode
    """

    @classmethod
    def gen_in(cls, shape, const_shape):
        """
        generate input
        :param shape:
        :return:
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "const_shape": const_shape,
                "mode": CONST,
                "support_broadcast": True,
                }


class ShapeSimplifier:
    """
    ShapeSimplifier
    """

    class Operator(Enum):
        """
        the fusion behavior of two contiguous axis
        """
        # can fuse axis
        FUSED = auto()
        # can not fuse axis
        ALONE = auto()

    @classmethod
    def get_state(cls, shape, _range):
        """
        get_state
        :param shape:
        :param range:
        :return:
        """
        status = []
        for s, (r0, _) in zip(shape, _range):
            if s > 1 or s == -1 and r0 > 1:
                status.append(ShapeValueType.COMMON)
            elif s == -1:
                status.append(ShapeValueType.UNKNOWN)
            elif s == 1:
                status.append(ShapeValueType.ONE)
        return status

    @classmethod
    def get_operator(cls, state1, state2):
        """
        get_operator
        :param state1:
        :param state2:
        :return:
        """
        state1_all_one = all(s == str(ShapeValueType.ONE) for s in state1)
        state2_all_one = all(s == str(ShapeValueType.ONE) for s in state2)
        if state1_all_one or state2_all_one:
            return cls.Operator.FUSED
        dim_diff = 0
        for s1, s2 in zip(state1, state2):
            if s1 != s2:
                dim_diff += 1
            elif s1 == s2 and s1 != str(ShapeValueType.ONE):
                dim_diff += 1
        if dim_diff <= 1:
            return cls.Operator.FUSED
        if str(ShapeValueType.UNKNOWN) in state1 or str(ShapeValueType.UNKNOWN) in state2:
            return cls.Operator.ALONE
        if state1 == state2:
            return cls.Operator.FUSED

        return cls.Operator.ALONE

    @classmethod
    def combine_dim(cls, dim1, dim2):
        """
        combine_dim
        :param dim1:
        :param dim2:
        :return:
        """
        dims = []
        for d1, d2 in zip(dim1, dim2):
            if 0 in (d1, d2):
                dims.append(0)
            elif -1 in (d1, d2):
                dims.append(-1)
            else:
                dims.append(d1 * d2)
        return dims

    @classmethod
    def combine_range(cls, range1, range2):
        """
        combine_range
        :param range1:
        :param range2:
        :return:
        """
        return [util.combine_range([r1, r2]) for r1, r2 in zip(range1, range2)]
