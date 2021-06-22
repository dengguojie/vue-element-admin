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
classifier of shape in known reduce axis
"""
from . import reduce_helper as helper
from . import util
from .reduce_helper import ZeroAxisStatus

CONST = "const"
BEFORE = "before"
AXIS = "axis"
ZERO = "zero"


class KnownReduceClassifier:

    def __init__(self, ins: list, keepdims: bool):
        self.ins = ins
        inputs_before_reduce, inputs_after_reduce, inputs_axis, self.inputs_classification = \
            helper.inputs_classify(ins)
        self.reduce_axes = inputs_axis[0].get("value")
        self.keepdims = keepdims
        self.input_x = helper.generate_reduce_input(inputs_before_reduce, inputs_after_reduce,
                                                    self.reduce_axes, self.keepdims)

        self.zero_axis_status = self._handle_zero_axis()

        self.n_shape, self.n_ranges, self.n_reduce_axes = self._normalize()
        self.f_shape, self.f_ranges, self.f_reduce_axes = helper.simplify(self.n_shape,
                                                                          self.n_ranges,
                                                                          self.n_reduce_axes)
        self.dim_len, self.reduce_axis_size = len(self.f_shape), len(self.f_reduce_axes)

    def classify(self):
        from tbe.common.buildcfg import get_current_build_config
        if get_current_build_config("enable_op_prebuild"):
            return [helper.ins_of_prebuild(self.ins, self.reduce_axes)]

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
        reduce_axes0 = [x + len(shape0) if x < 0 else x for x in self.reduce_axes]

        shape, ranges, reduce_axes = [], [], []
        skip_count = 0
        for i, (d, r) in enumerate(zip(shape0, ranges0)):
            if d == 1:
                skip_count += 1
                continue
            shape.append(d)
            ranges.append(r)
            if i in reduce_axes0:
                reduce_axes.append(i - skip_count)

        if not shape:
            shape.append(1)
            ranges.append((1, 1))

        return shape, ranges, reduce_axes

    def _is_const(self):
        return helper.is_const(self.input_x["shape"])

    def _classify_const(self):

        def _normalize_const():
            # make const's inputs as same as dynamic
            if self.f_reduce_axes:
                length = len(self.f_shape)
                is_last_reduce = self.f_reduce_axes[-1] == length - 1
                last_pad_one = is_last_reduce and length % 2 != 0
                nlast_pad_one = not is_last_reduce and length % 2 == 0
                if last_pad_one or nlast_pad_one:
                    self.f_shape.insert(0, 1)
                    self.f_ranges.insert(0, (1, 1))
                    self.f_reduce_axes = [x + 1 for x in self.f_reduce_axes]

        _normalize_const()

        ins_before_reduce = {
            "shape": self.f_shape,
            "range": self.f_ranges,
            "mode": CONST,
            "rel_pos_to_reduce": BEFORE
        }
        helper.refine_ins(ins_before_reduce, self.f_reduce_axes)
        ins_axis = {
            "shape": [len(self.f_reduce_axes), ],
            "value": self.f_reduce_axes,
            "rel_pos_to_reduce": AXIS,
            "ori_axis": self.n_reduce_axes
        }
        ins_after_reduce = helper.generate_ins_of_after_reduce(ins_before_reduce, ins_axis, self.keepdims)
        out_ins = [helper.generate_ins_of_all(ins_before_reduce, ins_after_reduce, ins_axis,
                                              self.inputs_classification)]

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
