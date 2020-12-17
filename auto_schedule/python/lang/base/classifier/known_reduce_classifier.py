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
from te import platform as cce

from . import reduce_helper as helper
from . import util

CONST = "const"


class KnownReduceClassifier:

    def __init__(self, ins: list):
        self.ins = ins

        self.input_x, self.reduce_axes = ins[0], ins[1]
        self.n_shape, self.n_ranges, self.n_reduce_axes = self._normalize()
        self.f_shape, self.f_ranges, self.f_reduce_axes = helper.simplify(self.n_shape,
                                                                          self.n_ranges,
                                                                          self.n_reduce_axes)
        self.dim_len, self.reduce_axis_size = len(self.f_shape), len(self.f_reduce_axes)

    def classify(self):
        if cce.fusion_manager.fusion_manager.get_build_cfg() == "disable":
            return [self.ins]

        return self._classify_const() if self._is_const() else self._classify_var()

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

        input_x = {
            "shape": self.f_shape,
            "range": self.f_ranges,
            "mode": CONST,
            "ori_axis": self.n_reduce_axes
        }
        ins = [[input_x, self.f_reduce_axes]]

        if not ins[0][1]:
            ins[0][0]["shape"] = [1] + ins[0][0]["shape"]
            ins[0][0]["range"] = [(1, 1)] + ins[0][0]["range"]
            ins[0][1] = [0, ]

        return ins

    def _classify_var(self):
        out_ins = []
        for ins in helper.generate_ins(self.reduce_axis_size, self.dim_len):
            if ins[1]:
                out_ins.append(ins)
            else:
                ins[0]["shape"] = [1] + ins[0]["shape"]
                ins[0]["range"] = [(1, 1)] + ins[0]["range"]
                ins[1] = [0, ]
                out_ins.append(ins)

        return out_ins
