#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2022-2022 Huawei Technologies Co., Ltd
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
classifier of shape in split
"""
from functools import reduce
from operator import mul
from typing import List

from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation
from . import util


INPUT_NUMBER_LIMIT = 63
UNKNOWN_RANK = -2
BLOCK_SIZE_BYTE = 32
TRANSPOSE_FACTOR = 16
SPLIT = "split"
SPLIT_GENERAL = "split_general"
SPLIT_EMPTY = "split_empty"

BLOCK_NUM_MAPPING = {
    "uint1": 256,
    "bool": 32,
    "int8": 32,
    "uint8": 32,
    "float16": 16,
    "int16": 16,
    "uint16": 16,
    "float32": 8,
    "int32": 8,
    "uint32": 8,
    "int64": 4,
    "uint64": 4,
    "bfloat16": 16,
}


def classify(ins: list, extra_params: dict):
    """
    classify
    :param ins: inputs, [split_dim], [split_size]
    :param extra_params: include split_num and avg_split
    :return: inputs, axis, split_size
    """
    if extra_params is None or "num_split" not in extra_params:
        dict_args = {
            "errCode": "E90001",
            "detailed_cause": "inputs of classify must include the dict extra_params with the key num_split when "
                              "mode is split"
        }
        raise RuntimeError(dict_args, get_error_message(dict_args))
    avg_split = False
    if "avg_split" in extra_params:
        avg_split = extra_params.get("avg_split")
    num_split = extra_params.get("num_split")
    classifier = SplitClassifier(ins, num_split)
    operation.get_context().add("_avg_split", avg_split or num_split == 1)
    return classifier.classify()


class SplitClassifier:
    """
    Concat classifier
    """

    def __init__(self, ins: list, num_split: bool):
        self.ins = ins
        self.ori_axis = None
        self.axis = None
        self.num_split = num_split
        self.is_single_output = num_split == 1
        self.maybe_empty = False
        self.only_empty = False
        self.split_size = [-1] * num_split
        self.shapes: List = []
        self.ranges: List = []
        self.dtype: str = ""
        self.merged_shapes: List = []
        self.merged_ranges: List = []

    def _check_params(self):
        if len(self.ins) <= 1:
            dict_args = {"errCode": "E90001", "detailed_cause": "input numbers error"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def _get_axis(self, split_dim):
        if isinstance(split_dim, int):
            return split_dim
        if isinstance(split_dim, dict):
            axis_value = split_dim.get("value")
            if isinstance(axis_value, int):
                return axis_value
            elif isinstance(axis_value, (tuple, list)):
                return axis_value[0]
        return self.axis

    def _get_split_size(self, split_size):
        if isinstance(split_size, (tuple, list)) and min(split_size) >= 0 and len(split_size) == self.num_split:
            return list(split_size)
        if isinstance(split_size, dict):
            split_size_value = split_size.get("value")
            if isinstance(split_size_value, (tuple, list)) and \
               min(split_size_value) >= 0 and len(split_size_value) == self.num_split:
                return list(split_size_value)
        return self.split_size

    def _get_input_info(self):
        input_x = self.ins[0]
        self.shapes = input_x.get("shape") or (1,)
        if UNKNOWN_RANK in self.shapes:
            self.ranges = input_x.get("range", [(0, None)])
        else:
            self.ranges = input_x.get("range") or ((1, 1),)
        self.dtype = input_x.get("dtype")
        if len(self.ins) > 1:
            self.ori_axis = self._get_axis(self.ins[1])
            self.axis = self.ori_axis
        if len(self.ins) > 2:
            self.split_size = self._get_split_size(self.ins[2])

    def _process_unknown_rank(self):
        if UNKNOWN_RANK in self.shapes:
            if len(self.shapes) != 1:
                dict_args = {"errCode": "E90001",
                             "detailed_cause": "if the shape contains -2, it must be [-2] or (-2,)"}
                raise RuntimeError(dict_args, get_error_message(dict_args))
            self.shapes = [-1, -1]
            self.ranges = [(0, None), (0, None)]
            self.axis = 0 if self.is_single_output else 1

    def _process_single_output(self):
        if self.is_single_output:
            self.axis = 0

    def _process_unknown_axis(self):
        if self.axis is None:
            self.shapes = [-1, -1]
            min_range = [r_min for r_min, _ in self.ranges]
            if 0 in min_range:
                self.ranges = [(0, None), (0, None)]
            else:
                self.ranges = [(1, None), (1, None)]
            self.axis = 0 if self.is_single_output else 1

    @staticmethod
    def shape_is_const(_shape, start, end):
        return -1 not in _shape[start:end]

    @staticmethod
    def merge_shape(_shape):
        return reduce(mul, _shape, 1)

    def _merge_shape_range(self):
        if self.shape_is_const(self.shapes, 0, self.axis):
            self.merged_shapes.append(self.merge_shape(self.shapes[0:self.axis]))
        else:
            value = 1 if self.axis == 0 else -1
            self.merged_shapes.append(value)
        new_range = util.combine_range(self.ranges[0:self.axis] or [(1, 1)])
        self.merged_ranges.append(new_range)
        if self.shape_is_const(self.shapes, self.axis, len(self.shapes)):
            self.merged_shapes.append(self.merge_shape(self.shapes[self.axis:]))
        else:
            self.merged_shapes.append(-1)
        new_range = util.combine_range(self.ranges[self.axis:])
        self.merged_ranges.append(new_range)

    def _process_empty_shape(self):
        self.only_empty = 0 in self.shapes or all(lower == 0 and upper == 0 for lower, upper in self.ranges)
        self.maybe_empty = any(lower == 0 for lower, _ in self.ranges)

    def _update_shape_range(self):
        for index, (shape_value, (lower, upper)) in enumerate(zip(self.merged_shapes, self.merged_ranges)):
            if shape_value != -1:
                self.merged_ranges[index] = (shape_value, shape_value)
            if lower is not None and upper is not None and lower == upper:
                self.merged_shapes[index] = lower

    def _process_const_split_size(self):
        if min(self.split_size) > -1 and min(self.merged_shapes) > -1 and self.ori_axis is not None:
            split_output = self.merged_shapes[1] // self.shapes[self.ori_axis]
            for index, value in enumerate(self.split_size):
                self.split_size[index] = value * split_output

    def _generate_ins(self):
        if self.only_empty:
            return [EmptyMode.gen_in(self.split_size)]
        general_factor = BLOCK_NUM_MAPPING.get(self.dtype) * TRANSPOSE_FACTOR
        res = []
        if self.is_single_output:
            res.append(OriginalMode.gen_in(self.merged_shapes, self.merged_ranges, SPLIT, self.split_size))
        else:
            res.append(OriginalMode.gen_in(self.merged_shapes, self.merged_ranges, SPLIT, self.split_size))
            res.append(OriginalMode.gen_in(self.merged_shapes, self.merged_ranges,
                                           SPLIT_GENERAL, self.split_size, general_factor))
        if self.maybe_empty:
            res.append(EmptyMode.gen_in(self.split_size))
        return res

    def classify(self):
        self._check_params()
        self._get_input_info()
        self._process_single_output()
        self._process_unknown_axis()
        operation.add_compile_info_inner("_ori_axis", self.axis)
        if self.axis < 0:
            self.axis = self.axis + len(self.shapes)
        self._process_unknown_rank()
        self._merge_shape_range()
        self._update_shape_range()
        self._process_empty_shape()
        self._process_const_split_size()
        res = self._generate_ins()
        return res


class OriginalMode:
    """
    Original Mode
    """

    @classmethod
    def gen_in(cls, shapes, ranges, split_mode, split_size, split_factor=1):
        """
        generate input
        :param shapes:
        :param ranges:
        :param split_mode:
        :param split_size:
        :param split_factor:
        :return:
        """
        new_input = {
                "shape": shapes,
                "range": ranges,
                "mode": split_mode,
                "split_factor": split_factor
            }
        axis = 1
        return [new_input, axis, split_size]


class EmptyMode:
    """
    Empty Mode
    """

    @classmethod
    def gen_in(cls, split_size):
        """
        generate input
        :param split_size:
        :return:
        """
        new_input = {"shape": (0, 0),
                     "range": [(0, 0), (0, 0)],
                     "mode": SPLIT_EMPTY,
                     "split_factor": 1
                     }
        axis = 0
        return [new_input, axis, split_size]
