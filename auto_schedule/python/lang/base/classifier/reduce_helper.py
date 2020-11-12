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
helper for reduce classifier
"""
from enum import Enum, auto

from . import util

COMMON = "common"
REDUCE = "reduce"
SPECIAL = "special"


def is_const(shape):
    """
    check const shape
    :param shape:
    :return:
    """
    return all(x > 0 for x in shape)


def generate_ins(reduce_axis_size, dim_len):
    """
    generate all possible reduce ins, with assigned reduce axis size and dim length
    :param reduce_axis_size:
    :param dim_len:
    :return:
    """
    ret = []
    reduce_axis_size = min(reduce_axis_size, (dim_len + 1) // 2)
    for size_i in range(reduce_axis_size):
        for p in generate_patterns(size_i):
            ret.append(generate_in(p))

    for p in generate_patterns(reduce_axis_size):
        if len(p) <= dim_len + 1:
            ret.append(generate_in(p))

    return ret


def generate_patterns(reduce_axis_size):
    """
    generate last nlast axis reduce pattern.
    such as: one reduce axis: (a, r), (a, r, a)
             three reduce axis: (a, r, a, r, a, r), (a, r, a, r, a, r, a)
             'a mean common axis, r means reduce axis'
    :param reduce_axis_size:
    :return:
    """
    if reduce_axis_size == 0:
        return [[COMMON]]

    ptl = [COMMON, REDUCE]
    last_pattern = [ptl[i % 2] for i in range(2 * reduce_axis_size)]
    nlast_pattern = [ptl[i % 2] for i in range(2 * reduce_axis_size + 1)]

    return [last_pattern, nlast_pattern]


def generate_in(pattern):
    """
    generate reduce input by pattern, contains data dict and reduce axes
    :param pattern:
    :return:
    """
    input_x = {
        "shape": [-1] * len(pattern),
        "range": [(1, None)] * len(pattern),
        "mode": SPECIAL
    }

    reduce_axes = []
    for i, p in enumerate(pattern):
        if p == REDUCE:
            reduce_axes.append(i)

    return [input_x, reduce_axes]


def simplify(shape, ranges, reduce_axes):
    """
    simplify shape, range, reduce axis.
    fuse continuous reduce axis or non-reduce axis.
    :param shape:
    :param ranges:
    :param reduce_axes:
    :return:
    """
    f_shape, f_ranges = [1], [(1, 1)]
    f_reduce_axes = []

    state = ShapeSimplifier.State.ONE
    for i, (d, r) in enumerate(zip(shape, ranges)):
        if d == 1:
            continue

        is_reduce_axis = i in reduce_axes
        state_i = ShapeSimplifier.get_state(d, is_reduce_axis)
        operator = ShapeSimplifier.get_operator(state, state_i)

        if operator == ShapeSimplifier.Operator.FUSED:
            f_shape[-1] = util.combine_dim([f_shape[-1], d])
            f_ranges[-1] = util.combine_range([f_ranges[-1], r])
        else:
            f_shape.append(d)
            f_ranges.append(r)

        if is_reduce_axis:
            reduce_axis = len(f_shape) - 1
            if not f_reduce_axes or f_reduce_axes[-1] != reduce_axis:
                f_reduce_axes.append(reduce_axis)

        if state_i != ShapeSimplifier.State.ONE:
            state = state_i

    return f_shape, f_ranges, f_reduce_axes


class ShapeSimplifier:
    """
    ShapeSimplifier
    """

    class State(Enum):
        """
        the axis type
        """
        # dim is one
        ONE = auto()
        # not reduce axis
        COMMON = auto()
        # reduce axis
        REDUCE = auto()

    class Operator(Enum):
        """
        the fusion behavior of two contiguous axis
        """
        # can fuse axis
        FUSED = auto()
        # can not fuse axis
        ALONE = auto()

    @classmethod
    def get_state(cls, dim: int, is_reduce_axis: bool):
        """
        get_state
        :param dim:
        :param is_reduce_axis:
        :return:
        """
        if dim == 1:
            return cls.State.ONE

        return cls.State.REDUCE if is_reduce_axis else cls.State.COMMON

    @classmethod
    def get_operator(cls, state1, state2):
        """
        get_operator
        :param state1:
        :param state2:
        :return:
        """
        if state1 == cls.State.ONE or state2 == cls.State.ONE:
            return cls.Operator.FUSED
        if state1 == state2:
            return cls.Operator.FUSED

        return cls.Operator.ALONE
