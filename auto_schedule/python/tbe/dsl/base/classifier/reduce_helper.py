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
helper for reduce classifier
"""
from copy import deepcopy
from enum import Enum
from enum import auto

from . import util

COMMON = "common"
REDUCE = "reduce"
SPECIAL = "special"
BEFORE = "before"
AFTER = "after"
AXIS = "axis"
ZERO = "zero"


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
        "mode": SPECIAL,
        "rel_pos_to_reduce": BEFORE
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


def inputs_classify(inputs):
    """
    classify inputs_before_reduce and inputs_after_reduce
    """
    inputs_before_reduce, inputs_after_reduce, input_axis, inputs_classification = [], [], [], []
    for single_input in inputs:
        input_type = single_input.get("rel_pos_to_reduce")
        if input_type == AXIS:
            input_axis.append(deepcopy(single_input))
            inputs_classification.append(AXIS)
        elif input_type == AFTER:
            inputs_after_reduce.append(deepcopy(single_input))
            inputs_classification.append(AFTER)
        else:
            inputs_before_reduce.append(deepcopy(single_input))
            inputs_classification.append(BEFORE)

    return inputs_before_reduce, inputs_after_reduce, input_axis, inputs_classification


def _process_all_unknown_shape(shape_list, range_list):
    """
    process input include shape -2
    """
    all_unknown_shape_len = 8
    for single_shape in shape_list:
        if tuple(single_shape) != (-2, ):
            all_unknown_shape_len = len(single_shape)
            break

    for idx, single_shape in enumerate(shape_list):
        if tuple(single_shape) == (-2, ):
            shape_list[idx] = [-1] * all_unknown_shape_len
            range_list[idx] = [(0, None)] * all_unknown_shape_len


def generate_reduce_input(inputs_before_reduce, inputs_after_reduce=None, reduce_axis=None, keep_dims=None):
    """
    obtain the shape and range to classify
    """
    if inputs_after_reduce:
        for single_input in inputs_after_reduce:
            ori_shape = list(single_input["shape"])
            ori_range = list(single_input.get("range") if single_input.get("range") else [(1, None)] * len(ori_shape))
            for axis in reduce_axis:
                # the dim corresponding to reduce_axis of input_after_reduce is not working in judging const and
                # should be set to -1
                if not keep_dims:
                    ori_shape.insert(axis, -1)
                    ori_range.insert(axis, (0, None))
                else:
                    ori_shape[axis] = -1
                    ori_range[axis] = (0, None)
            single_input["shape"] = ori_shape
            single_input["range"] = ori_range
        inputs_before_reduce.extend(inputs_after_reduce)

    shape_local = [x["shape"] for x in inputs_before_reduce]
    range_local = [x.get("range") if x.get("range") else [(1, None)] * len(shape_local[0]) for x in
                   inputs_before_reduce]

    _process_all_unknown_shape(shape_local, range_local)

    def _get_dim(i):
        return max([s[i] for s in shape_local])

    shape_out = [_get_dim(i) for i in range(len(shape_local[0]))]

    def _select_min_upper_bound(input_list):
        min_ele = util.VAR_BOUND_LIMIT + 1
        for ele in input_list:
            if ele is None:
                continue
            if ele < min_ele:
                min_ele = ele
        return min_ele if min_ele != util.VAR_BOUND_LIMIT + 1 else None

    def _get_range(i):
        if shape_out[i] != -1:
            return shape_out[i], shape_out[i]
        else:
            return max([r[i][0] for r in range_local]), _select_min_upper_bound([r[i][1] for r in range_local])

    range_out = [_get_range(i) for i in range(len(range_local[0]))]
    for index in range(len(shape_out)):
        if range_out[index][0] == range_out[index][1]:
            shape_out[index] = range_out[index][0]

    return {"shape": shape_out, "range": range_out}


def generate_ins_of_after_reduce(input_x, input_axis, keep_dims):
    """
    generate ins of inputs after reduce
    """
    if isinstance(input_axis, dict):
        reduce_axis = input_axis.get("value")
    else:
        reduce_axis = input_axis
    out_shape, out_range = [], []
    for i in range(len(input_x["shape"])):
        if i in reduce_axis:
            if not keep_dims:
                continue
            else:
                out_shape.append(1)
                out_range.append((1, 1))
        else:
            out_shape.append(input_x["shape"][i])
            out_range.append(input_x["range"][i])

    out_ins = {"shape": out_shape, "range": out_range,
               "mode": input_x["mode"], "rel_pos_to_reduce": AFTER}

    return out_ins


def generate_ins_of_all(ins_before_reduce, ins_after_reduce, reduce_axis, inputs_classification):
    """
    generate ins of all inputs
    """
    # const shape passes reduce_axis as a dict with the key "ori_axis" and var shape passes a list
    if isinstance(reduce_axis, dict):
        ins_axis = reduce_axis
    else:
        ins_axis = {"shape": [len(reduce_axis), ], "value": reduce_axis, "rel_pos_to_reduce": AXIS}
    ins = []
    for symbol in inputs_classification:
        if symbol == AXIS:
            ins.append(deepcopy(ins_axis))
        elif symbol == AFTER:
            ins.append(deepcopy(ins_after_reduce))
        else:
            ins.append(deepcopy(ins_before_reduce))

    return ins


def refine_ins(ins_before_reduce, reduce_axis):
    """
    if reduce axis is None, should refine ins
    """
    if not reduce_axis:
        ins_before_reduce["shape"] = [1] + ins_before_reduce["shape"]
        ins_before_reduce["range"] = [(1, 1)] + ins_before_reduce["range"]
        reduce_axis.append(0)


def ins_of_prebuild(ins, reduce_axis):
    """
    generate ins when build_config is "disable"
    """
    out_ins = []
    for single in ins:
        input_type = single.get("rel_pos_to_reduce")
        if input_type == AXIS:
            out_ins.append({"shape": [len(reduce_axis), ], "value": reduce_axis, "rel_pos_to_reduce": AXIS})
        else:
            out_ins.append(single)

    return out_ins


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


def generate_zero_ins():
    """

    :return:
    """
    ins_x_0 = {
        "shape": (1, -1, 0),
        "range": [(1, 1), (1, None), (0, 0)],
        "mode": ZERO,
        "rel_pos_to_reduce": BEFORE
    }
    ins_axis_0 = {
        "shape": [1],
        "value": [2],
        "rel_pos_to_reduce": AXIS,
        "ori_axis": [2]
    }

    ins_x_1 = {
        "shape": (1, 0, -1),
        "range": [(1, 1), (0, 0), (1, None)],
        "mode": ZERO,
        "rel_pos_to_reduce": BEFORE
    }
    ins_axis_1 = {
        "shape": [1],
        "value": [2],
        "rel_pos_to_reduce": AXIS,
        "ori_axis": [2]
    }

    return [[ins_x_0, ins_axis_0], [ins_x_1, ins_axis_1]]


class ZeroAxisStatus(Enum):
    """
    shape have zero axis status
    """

    EXIST = auto()
    MAYBE = auto()
    NON_EXIST = auto()
