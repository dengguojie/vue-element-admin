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
helper for 5HD
"""
from copy import deepcopy
from enum import Enum
from enum import auto
from . import util

COMMON = "common"
REDUCE = "reduce"

FORMAT_ARRAY_MAP = {
    "NC1HWC0": ["N", "C1", "H", "W", "C0"]
}


def get_reduce_axis_info(input_axis):
    know_reduce_axis = False
    reduce_axis_value = None
    if input_axis.get("value"):
        know_reduce_axis = True
        reduce_axis_value = input_axis.get("value")
    return know_reduce_axis, reduce_axis_value, input_axis.get("shape")[0]


def get_shape_info(inputs_before_reduce):
    return inputs_before_reduce.get("format"), inputs_before_reduce.get("ori_shape"), \
           inputs_before_reduce.get("ori_format")


def simplify(shape, ranges, reduce_axes, disable_fuse_axes=None, shape_format=None):
    """
    simplify shape, range, reduce axis.
    fuse continuous reduce axis or non-reduce axis.
    :param shape:
    :param ranges:
    :param reduce_axes:
    :param disable_fuse_axes:
    :param format:
    :return:
    """
    f_shape, f_ranges = [], []
    f_reduce_axes = []
    f_fused_format = []

    state = ShapeSimplifier.State.INIT
    state_i = ShapeSimplifier.State.INIT
    format_array = FORMAT_ARRAY_MAP.get(shape_format)
    for i, (d, r, f) in enumerate(zip(shape, ranges, format_array)):
        is_pad_axis = False
        is_reduce_axis = i in reduce_axes

        # for disable fuse axes
        if disable_fuse_axes is not None:
            for _, disable_fuse_axes_element in enumerate(disable_fuse_axes):
                if i in disable_fuse_axes_element:
                    state_i = "disable_fuse_axes" + str(i)
                    is_pad_axis = True
                    break
        if not is_pad_axis:
            state_i = ShapeSimplifier.get_state(d, is_reduce_axis)
        operator = ShapeSimplifier.get_operator(state, state_i)

        if operator == ShapeSimplifier.Operator.FUSED:
            f_shape[-1] = util.combine_dim([f_shape[-1], d])
            f_ranges[-1] = util.combine_range([f_ranges[-1], r])
            f_fused_format[-1] = combine_format(f_fused_format[-1], f)
        else:
            f_shape.append(d)
            f_ranges.append(r)
            f_fused_format.append(f)

        if is_reduce_axis:
            reduce_axis = len(f_shape) - 1
            if not f_reduce_axes or f_reduce_axes[-1] != reduce_axis:
                f_reduce_axes.append(reduce_axis)

        state = state_i

    return [f_shape, f_ranges, f_reduce_axes, f_fused_format]


def combine_format(format1, format2):
    if isinstance(format1, list):
        return format1.append(format2)
    else:
        return [format1, format2]


def get_pad_axes(ori_shape, ori_format, symbol):
    ori_format_list = list(ori_format)
    pad_axes_index = ori_format_list.index(symbol)
    pad_axes_value = ori_shape[pad_axes_index]
    return {symbol: pad_axes_index}, pad_axes_value


def refine_ins(ins_before_reduce, reduce_axis):
    """
    if reduce axis is None, should refine ins
    """
    if not reduce_axis:
        ins_before_reduce["shape"] = [1] + ins_before_reduce["shape"]
        ins_before_reduce["range"] = [(1, 1)] + ins_before_reduce["range"]
        reduce_axis.append(0)
        ins_before_reduce["s_format"] = [1] + ins_before_reduce["s_format"]
    elif reduce_axis[0] == 0:
        ins_before_reduce["shape"] = [1] + ins_before_reduce["shape"]
        ins_before_reduce["range"] = [(1, 1)] + ins_before_reduce["range"]
        reduce_axis_tmp = [x + 1 for x in reduce_axis]
        reduce_axis[:] = reduce_axis_tmp
        ins_before_reduce["s_format"] = [1] + ins_before_reduce["s_format"]


def eliminate_same_pattern(ins_list):
    """
    if shape length  and reduce axis are the same,
    only left the element without one axis.
    """
    common_ins_list = []
    one_ins_list = []
    left_ins_list = common_ins_list
    for ins in ins_list:
        if ins[0].get("shape")[0] == 1:
            one_ins_list.append(ins)
        else:
            common_ins_list.append(ins)

    for one_ins in one_ins_list:
        same_pattern = False
        for common_ins in common_ins_list:
            if len(one_ins[0].get("shape")) == len(common_ins[0].get("shape")) \
                    and one_ins[1].get("value") == common_ins[1].get("value"):
                same_pattern = True
                break
        if not same_pattern:
            left_ins_list.append(one_ins)
    return left_ins_list


class ShapeSimplifier:
    """
    ShapeSimplifier
    """

    class State(Enum):
        """
        the axis type
        """
        # init
        INIT = auto()
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
        return cls.State.REDUCE if is_reduce_axis else cls.State.COMMON

    @classmethod
    def get_operator(cls, state1, state2):
        """
        get_operator
        :param state1:
        :param state2:
        :return:
        """
        if state1 == state2:
            return cls.Operator.FUSED

        return cls.Operator.ALONE