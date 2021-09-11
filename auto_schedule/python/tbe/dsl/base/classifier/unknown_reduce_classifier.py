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
classifier of shape in unknown reduce axis
"""
from itertools import combinations

from . import reduce_helper as helper
from . import util
from .reduce_helper import ZeroAxisStatus

CONST = "const"
BEFORE = "before"
AXIS = "axis"
ZERO = "zero"


class UnknownReduceClassifier:
    """
    classifier for unknown reduce
    """

    def __init__(self, ins, keepdims):
        self.ins = ins
        inputs_before_reduce, inputs_after_reduce, inputs_axis, self.inputs_classification = \
            helper.inputs_classify(ins)
        self.input_x = helper.generate_reduce_input(inputs_before_reduce)
        self.keepdims = keepdims

        self.zero_axis_status = self._handle_zero_axis()

        self.n_shape, self.n_ranges = self._normalize()
        self.dim_len = len(self.n_shape)

        size0 = inputs_axis[0]["shape"][0]
        size1 = (len(self.n_shape) + 1) // 2
        self.axis_type = inputs_axis[0]["dtype"]
        self.reduce_axis_size = size1 if size0 < 0 else min(size0, size1)
        self.const_reduce_axis_size = self.dim_len if size0 < 0 else min(size0, self.dim_len)

    def classify(self):
        """
        classify function
        """
        from tbe.common.buildcfg import get_current_build_config
        if get_current_build_config("enable_op_prebuild"):
            return [helper.ins_of_prebuild(self.ins, list(range(0, self.reduce_axis_size)))]

        if self.zero_axis_status == ZeroAxisStatus.EXIST:
            return helper.generate_zero_ins()

        return self._classify_const() if self._is_const() else self._classify_var()

    def _handle_zero_axis(self):
        shape_x = self.input_x["shape"]
        range_x = self.input_x["range"]

        exist, maybe = False, False
        for i, dim_i in enumerate(shape_x):
            if dim_i == 0:
                exist = True
                break
            elif range_x[i][0] == 0:
                maybe = True

        if exist:
            return ZeroAxisStatus.EXIST
        elif maybe:
            for i, r in enumerate(range_x):
                if range_x[i][0] == 0:
                    range_x[i] = (1, range_x[i][1])
            return ZeroAxisStatus.MAYBE
        else:
            return ZeroAxisStatus.NON_EXIST

    def _normalize(self):
        shape0, ranges0 = list(self.input_x["shape"]), self.input_x.get("range")
        ranges0 = list(ranges0) if ranges0 else util.generate_range(shape0)

        shape, ranges = [], []
        for d, r in zip(shape0, ranges0):
            if d == 1:
                continue
            shape.append(d)
            ranges.append(r)

        if not shape:
            shape.append(1)
            ranges.append((1, 1))

        return shape, ranges

    def _is_const(self):
        return helper.is_const(self.input_x["shape"])

    def _classify_const(self):
        ret = []
        for i in range(self.const_reduce_axis_size):
            for reduce_axes in combinations(range(self.dim_len), i + 1):
                f_shape, f_ranges, f_reduce_axes = helper.simplify(self.n_shape, self.n_ranges, reduce_axes)

                if not f_reduce_axes:
                    f_shape = [1] + f_shape
                    f_ranges = [(1, 1)] + f_ranges
                    f_reduce_axes = [0, ]
                    reduce_axes = []

                def _normalize_const(_shape, _range, _axes):
                    # make const'inputs as same as dynamic
                    length = len(_shape)
                    is_last_reduce = _axes[-1] == length - 1
                    last_pad_one = is_last_reduce and length % 2 != 0
                    nlast_pad_one = not is_last_reduce and length % 2 == 0
                    if last_pad_one or nlast_pad_one:
                        _shape.insert(0, 1)
                        _range.insert(0, (1, 1))
                        _axes = [x + 1 for x in _axes]
                    return _shape, _range, _axes

                f_shape, f_ranges, f_reduce_axes = _normalize_const(f_shape, f_ranges, f_reduce_axes)

                input_x = {
                    "shape": f_shape,
                    "range": f_ranges,
                    "mode": CONST,
                    "rel_pos_to_reduce": BEFORE
                }
                input_axis = {
                    "shape": [len(f_reduce_axes), ],
                    "value": f_reduce_axes,
                    "rel_pos_to_reduce": AXIS,
                    "ori_axis": reduce_axes,
                    "axis_dtype": self.axis_type
                }
                ret.append([input_x, input_axis])
        # if ret is none or reduce dim has 1, should append pure move case
        if not ret or (any(x == 1 for x in self.input_x["shape"]) and list(self.n_shape) != [1, ]):
            input_x = {
                "shape": [1] + [util.combine_dim(self.n_shape)],
                "range": [(1, 1)] + [util.combine_range(self.n_ranges)],
                "mode": CONST,
                "rel_pos_to_reduce": BEFORE
            }
            input_axis = {
                "shape": [1, ],
                "value": [0, ],
                "rel_pos_to_reduce": AXIS,
                "ori_axis": [],
                "axis_dtype": self.axis_type
            }
            ret.append([input_x, input_axis])
        out_ins = []
        for ins in ret:
            ins_after_reduce = helper.generate_ins_of_after_reduce(ins[0], ins[1], self.keepdims)
            out_ins.append(helper.generate_ins_of_all(ins[0], ins_after_reduce, ins[1], self.inputs_classification))
        return out_ins

    def _classify_var(self):
        out_ins = []
        for ins in helper.generate_ins(self.reduce_axis_size, self.dim_len):
            helper.refine_ins(ins[0], ins[1])
            ins_after_reduce = helper.generate_ins_of_after_reduce(ins[0], ins[1], self.keepdims)
            out_ins.append(helper.generate_ins_of_all(ins[0], ins_after_reduce, ins[1], self.inputs_classification))

        if self.zero_axis_status == ZeroAxisStatus.MAYBE:
            out_ins.extend(helper.generate_zero_ins())

        return out_ins
