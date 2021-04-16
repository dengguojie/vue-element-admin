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
from tbe.dsl.base.operation import add_compile_info_inner

from . import reduce_helper as helper
from . import util

CONST = "const"
SPECIAL = "special"
BEFORE = "before"
AXIS = "axis"


def classify(ins: list):
    """
    classify
    :param ins:
    :return:
    """
    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") == AXIS:
            add_compile_info_inner("_ori_axis", single_input.get("value"))

    return NormClassifier(ins).classify()


class NormClassifier:

    def __init__(self, ins: list):
        self.ins = ins
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
        from te import platform as cce
        if cce.fusion_manager.fusion_manager.get_build_cfg() == "disable":
            return [helper.ins_of_prebuild(self.ins, self.reduce_axes)]

        return self._classify_const() if self._is_const() else self._classify_var()

    def _is_const(self):
        return helper.is_const(self.input_x["shape"])

    def _normalize(self):
        shape, ranges = list(self.input_x["shape"]), self.input_x.get("range")
        ranges = list(ranges) if ranges else util.generate_range(shape)
        reduce_axes = [x + len(shape) if x < 0 else x for x in self.reduce_axes]

        return shape, ranges, reduce_axes

    def _simplify(self):
        """
        simplify shape, range, reduce axis.
        fuse continuous reduce axis or non-reduce axis.
        """
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
        if len(self.f_reduce_axes) == len(self.f_shape):
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
        if len(self.f_reduce_axes) == len(self.f_shape):
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
