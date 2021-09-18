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
classifier of shape in norm
"""
import copy

from tbe.common.buildcfg import get_current_build_config
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base.operation import add_compile_info_inner
from tbe.dsl.base.operation import get_context

from . import reduce_helper as helper
from . import util

CONST = "const"
SPECIAL = "special"
BEFORE = "before"
AXIS = "axis"
NORM = "Norm"


def _check(ins, dim_len, min_dim_len):
    is_illegal_case = False
    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") != AXIS:
            if len(single_input.get("shape")) != dim_len:
                is_illegal_case = True

    if is_illegal_case:
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "dim len of input in norm classifier must be the same except -2"
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if min_dim_len > dim_len:
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "min_dim_len in norm classifier should not be larger than dim_len"
        raise RuntimeError(dict_args, get_error_message(dict_args))


def _infer_negative_two(ins, axis):
    dim_len = 8
    is_all_unknown_dim_len = True
    ins_list = []

    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") != AXIS:
            single_shape = single_input.get("shape")
            if tuple(single_shape) != (-2, ):
                is_all_unknown_dim_len = False
                dim_len = len(single_shape)
                break

    local_ins = copy.deepcopy(ins)
    for single_input in local_ins:
        if single_input.get("rel_pos_to_reduce") != AXIS:
            if tuple(single_input.get("shape")) == (-2, ):
                single_input["shape"] = [-1] * dim_len
                single_input["range"] = [(1, None)] * dim_len
    ins_list.append(local_ins)

    min_dim_len = max(x + dim_len if x < 0 else x for x in axis) + 1
    _check(local_ins, dim_len, min_dim_len)

    # special case, input are all -2
    # example: axis is 2, the options of dim_len are as follows:
    # 4, 5, 6, 7, 8 is the same pattern
    # 3 is another pattern
    # 1, 2 is impossible
    if is_all_unknown_dim_len:
        if min_dim_len != dim_len:
            local_ins = copy.deepcopy(ins)
            for single_input in local_ins:
                if single_input.get("rel_pos_to_reduce") != AXIS:
                    if tuple(single_input.get("shape")) == (-2, ):
                        single_input["shape"] = [-1] * min_dim_len
                        single_input["range"] = [(1, None)] * min_dim_len
            ins_list.append(local_ins)

    return ins_list


def classify(ins: list, extra_params: dict):
    """
    classify
    :param ins:
    :return:
    """
    fuse_axis = True
    if extra_params is not None and isinstance(extra_params, dict) and "disable_optimization" in extra_params:
        fuse_axis = not extra_params.get("disable_optimization")

    get_context().set_pattern(NORM)
    add_compile_info_inner("_fuse_axis", fuse_axis)
    axis = None
    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") == AXIS:
            axis = single_input.get("value")
            add_compile_info_inner("_ori_axis", axis)

    ins_list = _infer_negative_two(ins, axis)
    norm_classify_out = []

    for single_ins in ins_list:
        for single_item in NormClassifier(single_ins, fuse_axis).classify():
            norm_classify_out.append(single_item)

    return norm_classify_out


class NormClassifier:

    def __init__(self, ins: list, fuse_axis: bool):
        self.ins = ins
        self.fuse_axis = fuse_axis
        inputs_before_reduce, inputs_after_reduce, inputs_axis, self.inputs_classification = \
            helper.inputs_classify(ins)
        self.reduce_axes = inputs_axis[0].get("value") if inputs_axis[0].get("value") else \
            range(len(inputs_before_reduce[0]["shape"]))
        self.keepdims = True
        self.input_x = helper.generate_reduce_input(inputs_before_reduce, inputs_after_reduce,
                                                    self.reduce_axes, self.keepdims)

        self.n_shape, self.n_ranges, self.n_reduce_axes = self._normalize()
        self.f_shape, self.f_ranges, self.f_reduce_axes = self._simplify()

    def classify(self):
        if get_current_build_config("enable_op_prebuild"):
            return [helper.ins_of_prebuild(self.ins, self.reduce_axes)]

        return self._classify_const() if self._is_const() else self._classify_var()

    def _is_const(self):
        return helper.is_const(self.input_x["shape"])

    def _normalize(self):
        shape, ranges = list(self.input_x["shape"]), self.input_x.get("range")
        ranges = list(ranges) if ranges else util.generate_range(shape)
        reduce_axes = list(set([x + len(shape) if x < 0 else x for x in self.reduce_axes]))
        reduce_axes.sort()

        return shape, ranges, reduce_axes

    def _simplify(self):
        """
        simplify shape, range, reduce axis.
        fuse continuous reduce axis or non-reduce axis.
        """
        if not self.fuse_axis:
            return self.n_shape, self.n_ranges, self.n_reduce_axes

        f_shape, f_ranges, f_reduce_axes = [], [], []
        state = "init"
        for i, (d, r) in enumerate(zip(self.n_shape, self.n_ranges)):
            is_reduce_axis = i in self.n_reduce_axes
            state_i = "reduce" if is_reduce_axis else "common"

            if state == state_i:
                f_shape[-1] = util.combine_dim([f_shape[-1], d])
                f_ranges[-1] = util.combine_range([f_ranges[-1], r])
            else:
                f_shape.append(d)
                f_ranges.append(r)

            if is_reduce_axis:
                reduce_axis = len(f_shape) - 1
                if not f_reduce_axes or f_reduce_axes[-1] != reduce_axis:
                    f_reduce_axes.append(reduce_axis)
            state = state_i

        return f_shape, f_ranges, f_reduce_axes

    def _classify_const(self):
        # R -> 1,R
        if len(self.f_reduce_axes) == len(self.f_shape) and self.fuse_axis:
            out_axes = [i + 1 for i in self.f_reduce_axes]
            out_shape = [1] + self.f_shape
            out_range = [(1, 1)] + self.f_ranges
        else:
            out_axes = self.f_reduce_axes[:]
            out_shape = self.f_shape[:]
            out_range = self.f_ranges[:]

        ins_before_reduce = {
            "shape": out_shape,
            "range": out_range,
            "mode": CONST,
            "rel_pos_to_reduce": BEFORE
        }
        ins_axis = {
            "shape": [len(out_axes), ],
            "value": out_axes,
            "rel_pos_to_reduce": AXIS
        }
        ins_after_reduce = helper.generate_ins_of_after_reduce(ins_before_reduce, ins_axis, self.keepdims)
        ins = [helper.generate_ins_of_all(ins_before_reduce, ins_after_reduce, ins_axis, self.inputs_classification)]

        return ins

    def _classify_var(self):
        # R -> 1,R
        if len(self.f_reduce_axes) == len(self.f_shape) and self.fuse_axis:
            out_axes = [i + 1 for i in self.f_reduce_axes]
            out_shape = [1] + self.f_shape
            out_range = [(1, 1)] + self.f_ranges
        else:
            out_axes = self.f_reduce_axes[:]
            out_shape = self.f_shape[:]
            out_range = self.f_ranges[:]

        ins_before_reduce = {
            "shape": out_shape,
            "range": out_range,
            "mode": SPECIAL,
            "rel_pos_to_reduce": BEFORE
        }
        ins_axis = {
            "shape": [len(out_axes), ],
            "value": out_axes,
            "rel_pos_to_reduce": AXIS
        }
        ins_after_reduce = helper.generate_ins_of_after_reduce(ins_before_reduce, ins_axis, self.keepdims)
        ins = [helper.generate_ins_of_all(ins_before_reduce, ins_after_reduce, ins_axis, self.inputs_classification)]

        return ins
