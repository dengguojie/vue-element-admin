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
from te import platform as cce
from itertools import combinations

from . import reduce_helper as helper
from . import util

CONST = "const"


class UnknownReduceClassifier:

    def __init__(self, ins):
        self.input_x = ins[0]
        self.n_shape, self.n_ranges = self._normalize()
        self.dim_len = len(self.n_shape)

        size0 = ins[1]["shape"][0]
        size1 = (len(self.n_shape) + 1) // 2
        self.axis_type = ins[1]["dtype"]
        self.reduce_axis_size = size1 if size0 < 0 else min(size0, size1)
        self.const_reduce_axis_size = self.dim_len if size0 < 0 else min(size0, self.dim_len)

    def classify(self):
        if cce.fusion_manager.fusion_manager.get_build_cfg() == "disable":
            return [[self.input_x, list(range(0, self.reduce_axis_size))]]

        return self._classify_const() if self._is_const() else self._classify_var()

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
                f_shape, f_ranges, f_reduce_axes = helper.simplify(self.n_shape,
                                                                   self.n_ranges,
                                                                   reduce_axes)
                if not f_reduce_axes:
                    f_shape = [1] + f_shape
                    f_ranges = [(1, 1)] + f_ranges
                    f_reduce_axes = [0, ]

                input_x = {
                    "shape": f_shape,
                    "range": f_ranges,
                    "mode": CONST,
                    "ori_axis": reduce_axes,
                    "axis_dtype": self.axis_type
                }
                ret.append([input_x, f_reduce_axes])
        # if reduce dim has 1, should append pure move case
        if any(x == 1 for x in self.input_x["shape"]):
            input_x = {
                "shape": [1] + [util.combine_dim(self.n_shape)],
                "range": [(1, 1)] + [util.combine_range(self.n_ranges)],
                "mode": CONST,
                "ori_axis": [],
                "axis_dtype": self.axis_type
            }
            ret.append([input_x, [0, ]])

        return ret

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
