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
from tbe.common.utils import shape_util
from . import reduce_helper as helper
from . import util

BEFORE = "before"
AFTER = "after"
AXIS = "axis"


class MixedReduceClassifier:
    """
    classifier for mixed reduce
    """

    def __init__(self, ins, keepdims, _known_axis):
        # not support const, zero
        self.ins = ins
        self.keepdims = keepdims
        self.known_axis = True if _known_axis is not None else False
        self.shape_before_reduce = None
        self.range_before_reduce = None
        self.shape_after_reduce = None
        self.range_after_reduce = None
        self.reduce_axes = None

        self.dim_len = None
        self.reduce_axis_size = None
        _, _, self.inputs_axis, self.inputs_classification = helper.inputs_classify(ins)

        self.collect_info()
        self.infer_shape()
        self.infer_range()
        self.update_ins()

    def collect_info(self):
        """
        Func: The whole graph only has three different var, include shape_before_reduce,
        shape_after_reduce, reduce_axes
        """
        # collect ori info
        for _item in self.ins:
            if _item.get("rel_pos_to_reduce") == AXIS:
                self.reduce_axes = _item.get("value")
                if isinstance(self.reduce_axes, int):
                    self.reduce_axes = [self.reduce_axes, ]
                continue

            if -2 not in _item.get("shape"):
                if _item.get("rel_pos_to_reduce") == BEFORE:
                    self.shape_before_reduce = _item.get("shape")
                else:
                    self.shape_after_reduce = _item.get("shape")

        def _normalization(_shape):
            if _shape == []:
                _shape = [1, ]
            elif _shape:
                _shape = [-1] * len(_shape)
            return _shape

        # pre_process
        self.shape_before_reduce = _normalization(self.shape_before_reduce)
        self.shape_after_reduce = _normalization(self.shape_after_reduce)
        if self.known_axis:
            if self.shape_before_reduce:
                shape_len = len(self.shape_before_reduce)
                if self.reduce_axes == []:
                    self.reduce_axes = range(shape_len)
                if hasattr(self.reduce_axes, "index"):
                    self.reduce_axes = list(self.reduce_axes)
                self.reduce_axes = shape_util.axis_check(shape_len, self.reduce_axes)
            else:
                self.reduce_axes = list(set(self.reduce_axes))

    def infer_shape(self):
        """
        infer bound of shape
        0. shape_before_reduce, reduce_axes >>> shape_after_reduce(Precise)
        1. shape_before_reduce, >>> shape_after_reduce(Mini)
        2. shape_after_reduce, reduce_axes >>> shape_before_reduce(Maxi)
        3. shape_after_reduce, >>> shape_before_reduce(Maxi)
        """

        def _infer_main(before_shape, after_shape, reduce_axes):
            # deal reduce_axes while it is []
            if before_shape and reduce_axes == []:
                reduce_axes = list(range(len(before_shape)))
            elif after_shape and reduce_axes == []:
                if self.keepdims:
                    reduce_axes = list(range(len(after_shape)))
                else:
                    reduce_axes = list(range(8))

            if self.keepdims:
                before_shape = before_shape if before_shape else after_shape.copy()
                after_shape = after_shape if after_shape else before_shape.copy()
            else:
                if before_shape and self.known_axis:
                    # precise after_shape
                    length = len(before_shape) - len(reduce_axes)
                    after_shape = [-1] * length if length > 0 else [1]
                elif before_shape and not self.known_axis:
                    # mini after_shape
                    after_shape = [-1] if len(before_shape) > 1 else [1]
                elif after_shape and self.known_axis:
                    # maxi before_shape
                    maxi_length = len(reduce_axes) + len(after_shape)
                    maxi_length = 8 if maxi_length > 8 else maxi_length
                    before_shape = [-1] * maxi_length
                    self.known_axis = False
                else:
                    # maxi before_shape
                    before_shape = [-1] * 8
            return before_shape, after_shape, reduce_axes

        if [self.shape_before_reduce, self.shape_after_reduce] == [None, None]:
            self.shape_before_reduce = [-1] * 8

        self.shape_before_reduce, \
        self.shape_after_reduce, self.reduce_axes = _infer_main(self.shape_before_reduce,
                                                                self.shape_after_reduce,
                                                                self.reduce_axes)

    def infer_range(self):
        """
        infer range for reduce case
        """
        self.range_before_reduce = util.generate_range(self.shape_before_reduce)
        self.range_after_reduce = util.generate_range(self.shape_after_reduce)

    def update_ins(self):
        """
        update ins by shape info and range info
        """
        for _item in self.ins:
            _mode = _item.get("rel_pos_to_reduce")
            if _mode == AXIS:
                if self.known_axis:
                    _item["value"] = self.reduce_axes
            elif _mode == BEFORE:
                _item["shape"] = self.shape_before_reduce
                _item["range"] = self.range_before_reduce
            else:
                _item["shape"] = self.shape_after_reduce
                _item["range"] = self.range_after_reduce

    def _classify_var(self):
        if self.known_axis:
            f_shape, f_ranges, f_reduce_axes = helper.simplify(self.shape_before_reduce,
                                                               self.range_before_reduce,
                                                               self.reduce_axes)
            self.dim_len, self.reduce_axis_size = len(f_shape), len(f_reduce_axes)
        else:
            self.dim_len = len(self.shape_before_reduce)
            maxi_size = (self.dim_len + 1) // 2
            if self.keepdims:
                if self.reduce_axes:
                    self.reduce_axis_size = min(maxi_size, len(self.reduce_axes))
                elif self.reduce_axes == []:
                    self.reduce_axis_size = maxi_size
                else:
                    # need algorithm deliver true
                    current_size = self.inputs_axis[0]["shape"][0]
                    self.reduce_axis_size = maxi_size if current_size < 0 else min(maxi_size, current_size)
            else:
                current_size = self.dim_len - len(self.shape_after_reduce)
                current_size = 1 if current_size == 0 else current_size
                self.reduce_axis_size = min(maxi_size, current_size)

        out_ins = []
        for ins in helper.generate_ins(self.reduce_axis_size, self.dim_len):
            helper.refine_ins(ins[0], ins[1])
            ins_after_reduce = helper.generate_ins_of_after_reduce(ins[0], ins[1], self.keepdims)
            out_ins.append(helper.generate_ins_of_all(ins[0], ins_after_reduce, ins[1], self.inputs_classification))

        if len(self.ins) == 2:
            # special branch: SingleReduce + ZeroMode
            out_ins.extend(helper.generate_zero_ins())
        return out_ins

    def classify(self):
        """
        do classify
        """
        from tbe.common.buildcfg import get_current_build_config
        if get_current_build_config("enable_op_prebuild"):
            return [helper.ins_of_prebuild(self.ins, self.reduce_axes)]

        return self._classify_var()
