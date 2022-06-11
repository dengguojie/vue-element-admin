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
classifier of shape in reduce
"""
from itertools import combinations
from tbe.dsl.base.operation import add_compile_info_inner
from . import reduce_helper as helper
from . import reduce_helper_5hd as helper_5hd
from . import util
from ..expr_compare import is_true
from .reduce_helper import ZeroAxisStatus


CONST = "const"
SPECIAL = "special"
BEFORE = "before"
REDUCE = "reduce"
AXIS = "axis"

PURE_MOVE = "pure move"
ALL_REDUCE = "all reduce"


DIM_DTYPE_ALIGNED = {
    "float32": 16,
    "float16": 16,
    "int8": 32,
    "int32": 16
}


class ReduceClassifier5HD:
    """
    classifier for 5hd reduce
    """

    def __init__(self, ins: list, keepdims: bool, disable_fuse_axes):
        self.ins = ins
        inputs_before_reduce, inputs_after_reduce, inputs_axis, self.inputs_classification = \
            helper.inputs_classify(ins)
        self.format, self.ori_shape, self.ori_format = helper_5hd.get_shape_info(
            inputs_before_reduce[0])
        self.input_type = inputs_before_reduce[0].get("dtype")
        self.pad_axes, self.pad_axes_value = helper_5hd.get_pad_axes(self.ori_shape, self.ori_format, "C")
        self.np_mapping = {"C1": "C", "C0": "C"}
        self.know_reduce_axis, self.reduce_axis_value, self.reduce_axis_shape = \
            helper_5hd.get_reduce_axis_info(inputs_axis[0])
        self.reduce_axis_type = inputs_axis[0].get("dtype")
        self.keepdims = keepdims
        self.disable_fuse_axes = disable_fuse_axes
        if self.know_reduce_axis:
            self.input_x = helper.generate_reduce_input(inputs_before_reduce, inputs_after_reduce,
                                                        self.reduce_axis_value, self.keepdims)
        else:
            self.input_x = helper.generate_reduce_input(inputs_before_reduce)

        self.last_axis_value = self.input_x.get("shape")[-1]
        self.zero_axis_status = helper.handle_zero_axis(self.input_x)
        self.n_shape, self.n_ranges, self.n_reduce_axes = self._normalize()
        self.dim_len = len(self.n_shape)

    def classify(self):
        """
        do classify
        """
        if self.zero_axis_status == ZeroAxisStatus.EXIST:
            return helper.generate_zero_ins()

        if self.know_reduce_axis:
            is_true(self._same_pad_axes_type(self.n_reduce_axes),
                    {"errCode": "E90001",
                     "detailed_cause": "Pad axis of 5HD should be the same reduce type."})

            out_ins = self._classify_all_known_reduce_axis()
        elif self.reduce_axis_shape != -1:
            out_ins = self._classify_known_reduce_shape(self.reduce_axis_shape)
            out_ins.append(self._classify_special_case(PURE_MOVE, SPECIAL))
            out_ins.append(self._classify_special_case(ALL_REDUCE, SPECIAL))
        else:
            out_ins = self._classify_unknown_reduce_shape()
            out_ins.append(self._classify_special_case(PURE_MOVE, SPECIAL))
            out_ins.append(self._classify_special_case(ALL_REDUCE, SPECIAL))

        return out_ins

    def _classify_all_known_reduce_axis(self):
        out_ins = []
        add_compile_info_inner("_ori_axis", self.n_reduce_axes)
        mode = CONST if self._is_const() else SPECIAL
        dim_type_aligned = DIM_DTYPE_ALIGNED.get(self.input_type)
        if self._is_const():
            # for pad axis is aligned,and is const, only one case is need
            #  1.pure move
            #  2.all reduce
            #  3.normal 5HD case
            if self.pad_axes_value % dim_type_aligned == 0:
                if not self.reduce_axis_value:
                    out_ins.append(self._classify_special_case(PURE_MOVE, mode))
                elif len(self.reduce_axis_value) == self.dim_len:
                    out_ins.append(self._classify_special_case(ALL_REDUCE, mode))
                else:
                    out_ins = self._classify_known_reduce_axis(self.n_reduce_axes, mode)
            else:
                out_ins = self._classify_known_reduce_axis(self.n_reduce_axes, mode)
        else:
            # for not const case, two case is need at most.
            out_ins = self._classify_known_reduce_axis(self.n_reduce_axes, mode)
            if not self.reduce_axis_value:
                out_ins.append(self._classify_special_case(PURE_MOVE, mode))
            elif len(self.reduce_axis_value) == self.dim_len:
                out_ins.append(self._classify_special_case(ALL_REDUCE, mode))
        return out_ins

    def _normalize(self):
        """
        1. no need delete one axis when input shape contains -1.
        2. should normalize range and reduce_axis
        """
        shape0, ranges0 = list(self.input_x.get("shape")), self.input_x.get("range")
        if not self._is_const():
            shape0 = [-1] * len(shape0)
        ranges0 = list(ranges0) if ranges0 else util.generate_range(shape0)

        reduce_axes0 = None
        if self.reduce_axis_value:
            reduce_axes0 = [x + len(shape0) if x < 0 else x for x in self.reduce_axis_value]

        return shape0, ranges0, reduce_axes0

    def _is_const(self):
        return helper.is_const(self.input_x.get("shape"))

    def _classify_known_reduce_axis(self, reduce_axes, mode=CONST):
        # 1. fuse axis
        simplify_result = \
            helper_5hd.simplify(self.n_shape, self.n_ranges, reduce_axes, self.disable_fuse_axes, self.format)

        f_shape, f_ranges, f_reduce_axes, s_format = \
            simplify_result[0], simplify_result[1], simplify_result[2], simplify_result[3]

        # 2. replace last value to constant value
        f_shape[-1] = self.last_axis_value

        # 3. generate ins
        ins_before_reduce = {
            "shape": f_shape,
            "range": f_ranges,
            "mode": mode,
            "rel_pos_to_reduce": BEFORE,
            "format": self.format,
            "pad_axes": self.pad_axes,
            "np_mapping": self.np_mapping,
            "ori_shape": self.ori_shape,
            "s_format": s_format
        }

        ins_axis = {
            "shape": [len(f_reduce_axes), ],
            "value": f_reduce_axes,
            "rel_pos_to_reduce": AXIS,
            "ori_axis": reduce_axes,
            "axis_dtype": self.reduce_axis_type
        }

        # 4.refine ins
        helper_5hd.refine_ins(ins_before_reduce, ins_axis.get("value"))

        # 5.generate ins after reduce
        ins_after_reduce = helper.generate_ins_of_after_reduce(ins_before_reduce, ins_axis, self.keepdims)

        # 6. generate out_ins
        return [helper.generate_ins_of_all(ins_before_reduce, ins_after_reduce, ins_axis, self.inputs_classification)]

    def _classify_known_reduce_shape(self, reduce_axis_size):
        out_ins = []
        reduce_axis_size = self.dim_len if reduce_axis_size < 0 else min(reduce_axis_size, self.dim_len)
        for reduce_axes in combinations(range(self.dim_len), reduce_axis_size):
            # check in disable_fuse_axes, all axis should be the same pattern.
            same_pad_axes_type = self._same_pad_axes_type(reduce_axes)
            if not same_pad_axes_type:
                continue

            result = self._classify_known_reduce_axis(reduce_axes, SPECIAL)
            for x in result:
                out_ins.append(x)
        return out_ins

    def _classify_unknown_reduce_shape(self):
        out_ins = []
        max_reduce_axis_size = self.dim_len + 1
        for reduce_axis_size in range(max_reduce_axis_size):
            result = self._classify_known_reduce_shape(reduce_axis_size)
            for x in result:
                out_ins.append(x)

        # eliminate same pattern
        out_ins = helper_5hd.eliminate_same_pattern(out_ins)
        return out_ins

    def _classify_special_case(self, reduce_type=PURE_MOVE, mode=CONST):
        input_x = {
            "shape": [1] + [util.combine_dim(self.n_shape)],
            "range": [(1, 1)] + [util.combine_range(self.n_ranges)],
            "mode": mode,
            "rel_pos_to_reduce": BEFORE
        }
        input_axis = {
            "shape": [1, ],
            "value": [0, ] if reduce_type == PURE_MOVE else [1, ],
            "rel_pos_to_reduce": AXIS,
            "ori_axis": [] if reduce_type == PURE_MOVE else [0, 1, 2, 3, 4],
            "axis_dtype": self.reduce_axis_type
        }

        ins_after_reduce = helper.generate_ins_of_after_reduce(input_x, input_axis, self.keepdims)
        return helper.generate_ins_of_all(input_x, ins_after_reduce, input_axis, self.inputs_classification)

    def _same_pad_axes_type(self, reduce_axes):
        # check in disable_fuse_axes, all axis should be the same pattern
        the_same_flag = True
        for i in range(len(self.disable_fuse_axes)):
            reduce_axis_flag = False
            for j in range(len(self.disable_fuse_axes[i])):
                if self.disable_fuse_axes[i][j] in reduce_axes:
                    if not reduce_axis_flag and j == len(self.disable_fuse_axes[i]) - 1:
                        the_same_flag = False
                        continue
                    reduce_axis_flag = True
                else:
                    if reduce_axis_flag is True:
                        the_same_flag = False
        return the_same_flag
