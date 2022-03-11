#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021-2021 Huawei Technologies Co., Ltd
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
classifier of shape in concat
"""
from functools import reduce
from operator import mul
from typing import List

from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation
from . import util


INPUT_NUMBER_LIMIT = 63
UNKNOWN_RANK = -2


def classify(ins: list, extra_params: dict):
    """
    classify
    :param ins: input list
    :param extra_params: include concat axis
    :return:
    """
    if extra_params is None or "axis" not in extra_params:
        dict_args = {
            "errCode": "E90001",
            "detailed_cause": "inputs of classify must include the dict extra_params with the key axis when mode is "
                              "concat "
        }
        raise RuntimeError(dict_args, get_error_message(dict_args))
    classifier = ConcatClassifier(ins, extra_params.get("axis"))
    return classifier.classify()


class ConcatClassifier:
    """
    Concat classifier
    """

    def __init__(self, ins: list, axis: int):
        self.ins = ins
        self.axis = axis
        self.maybe_empty = False
        self.only_empty = False
        self.shapes: List = []
        self.ranges: List = []
        self.merged_shapes: List = []
        self.merged_ranges: List = []

    def _check_inputs(self):
        if len(self.ins) != 1:
            dict_args = {"errCode": "E90001", "detailed_cause": "input numbers error"}
            raise RuntimeError(dict_args, get_error_message(dict_args))
        input_nums = len(self.ins[0])
        if input_nums > INPUT_NUMBER_LIMIT or input_nums <= 0:
            dict_args = {
                "errCode": "E90001",
                "detailed_cause": f"input numbers error, input numbers must be "
                                  f"greater 0 and less equal {INPUT_NUMBER_LIMIT} , now, it is {input_nums}"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def _get_shape_range(self):
        input_x = self.ins[0]
        for x in input_x:
            shapes = x.get("shape")
            if UNKNOWN_RANK in shapes:
                ranges = x.get("range", (0, None))
            else:
                ranges = x.get("range")
            self.shapes.append(shapes)
            self.ranges.append(ranges)

    def _process_unknown_rank(self):
        is_unknown_rank = any(UNKNOWN_RANK in _shape for _shape in self.shapes)
        if not is_unknown_rank:
            return

        is_all_unknown_rank = all(UNKNOWN_RANK in _shape for _shape in self.shapes)
        max_dim_len = max(len(_shape) for _shape in self.shapes)
        for index, _shape in enumerate(self.shapes):
            if UNKNOWN_RANK not in _shape:
                continue
            if len(_shape) != 1:
                dict_args = {"errCode": "E90001",
                             "detailed_cause": "if the shape contains -2, it must be [-2] or (-2,)"}
                raise RuntimeError(dict_args, get_error_message(dict_args))
            if is_all_unknown_rank:
                self.shapes[index] = [-1, -1]
                self.ranges[index] = [(0, None), (0, None)]
            else:
                self.shapes[index] = [-1 for _ in range(max_dim_len)]
                self.ranges[index] = [(0, None) for _ in range(max_dim_len)]

        if is_all_unknown_rank:
            self.axis = 1
        if len(self.shapes) == 1:
            self.axis = 0

    def _process_single_input(self):
        if len(self.shapes) == 1:
            self.axis = 0

    @staticmethod
    def shape_is_const(_shape, start, end):
        return -1 not in _shape[start:end]

    @staticmethod
    def merge_shape(_shape):
        return reduce(mul, _shape or [1])

    def _merge_shape_range(self):
        for shape, _range in zip(self.shapes, self.ranges):
            cur_shape = []
            cur_range = []
            if self.shape_is_const(shape, 0, self.axis):
                cur_shape.append(self.merge_shape(shape[0:self.axis]))
            else:
                value = 1 if self.axis == 0 else -1
                cur_shape.append(value)
            new_range = util.combine_range(_range[0:self.axis] or [(1, 1)])
            cur_range.append(new_range)
            if self.shape_is_const(shape, self.axis, len(shape)):
                cur_shape.append(self.merge_shape(shape[self.axis:]))
            else:
                cur_shape.append(-1)
            new_range = util.combine_range(_range[self.axis:])
            cur_range.append(new_range)
            self.merged_shapes.append(cur_shape)
            self.merged_ranges.append(cur_range)

    def _update_shape_range(self):
        shape_x = [x[0] for x in self.merged_shapes]
        max_x = max(shape_x)
        if max_x != -1:
            for index, value in enumerate(shape_x):
                if value != -1 and value != max_x:
                    dict_args = {"errCode": "E90001", "detailed_cause": "concat input shape error"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                if value == -1:
                    self.merged_shapes[index][0] = max_x
                    self.merged_ranges[index][0] = (max_x, max_x)

    def _process_empty_shape(self):
        zero_axis_shape = [_shape[0] for _shape in self.merged_shapes]
        one_axis_shape = [_shape[1] for _shape in self.merged_shapes]
        zero_axis_min_range = [_range[0][0] for _range in self.merged_ranges]
        one_axis_min_range = [_range[1][0] for _range in self.merged_ranges]
        self.only_empty = 0 in zero_axis_shape or all(s == 0 for s in one_axis_shape)
        self.maybe_empty = 0 in zero_axis_min_range or all(0 == s for s in one_axis_min_range)
        if self.only_empty or not self.maybe_empty:
            return
        for index, _range in enumerate(self.merged_ranges):
            if _range[0][0] == 0:
                self.merged_ranges[index][0] = (1, _range[0][1])

    def classify(self):
        self._check_inputs()
        self._get_shape_range()
        self._process_single_input()
        operation.add_compile_info_inner("_ori_axis", self.axis)
        self._process_unknown_rank()
        if self.axis < 0:
            self.axis = self.axis + len(self.shapes[0])
        self._merge_shape_range()
        self._update_shape_range()
        self._process_empty_shape()
        input_nums = len(self.merged_shapes)
        if self.only_empty:
            return [EmptyMode.gen_in(input_nums)]
        if self.maybe_empty:
            return [OriginalMode.gen_in(self.merged_shapes, self.merged_ranges), EmptyMode.gen_in(input_nums)]
        else:
            return [OriginalMode.gen_in(self.merged_shapes, self.merged_ranges)]


class OriginalMode:
    """
    Original Mode
    """

    @classmethod
    def gen_in(cls, shapes, ranges):
        """
        generate input
        :param shapes:
        :param ranges:
        :return:
        """
        new_input = []
        for shape, _range in zip(shapes, ranges):
            new_input_x = {
                "shape": shape,
                "range": _range,
                "mode": "concat"
            }
            new_input.append(new_input_x)
        axis = 1
        return [new_input, axis]


class EmptyMode:
    """
    Empty Mode
    """

    @classmethod
    def gen_in(cls, input_nums):
        """
        generate input
        :param input_nums: input tensor numbers
        :return:
        """
        new_input = {"shape": (0,),
                     "range": [(0, 0)],
                     "mode": "concat_empty"
                     }
        axis = 0
        return [[new_input for _ in range(input_nums)], axis]
