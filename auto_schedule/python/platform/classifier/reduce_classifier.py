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
import math

COMMON = "common"
REDUCE = "reduce"
SPECIAL = "special"


class ReduceClassifier:
    """
    ReduceClassifier
    """

    def __init__(self, ins: list):
        self.ins = ins

    def classify(self):
        """
        :return:
        """
        # data
        input_x_0 = self.ins[0]
        dim_len = len(input_x_0["shape"])
        # reduce
        input_x_1 = self.ins[1]
        reduce_axis_size = input_x_1["shape"][0]

        max_reduce_axis_size = math.ceil(dim_len / 2)
        if reduce_axis_size < 0:
            reduce_axis_size = max_reduce_axis_size
        else:
            reduce_axis_size = min(reduce_axis_size, max_reduce_axis_size)

        ret = []
        upper_bound = min(reduce_axis_size * 2, dim_len)
        for i in range(1, upper_bound):
            pattern0, reduce_axis0 = self._generate_pattern(i)
            tx0 = {
                "shape": [-1] * len(pattern0),
                "range": [(1, None)] * len(pattern0),
                "mode": SPECIAL,
                "pattern": pattern0
            }
            ret.append([tx0, reduce_axis0])

        pattern1, reduce_axis1 = self._generate_pattern(upper_bound)
        if upper_bound >= dim_len:
            pattern1 = pattern1[1:]
            reduce_axis1 = [i - 1 for i in reduce_axis1]
        tx1 = {
            "shape": [-1] * len(pattern1),
            "range": [(1, None)] * len(pattern1),
            "mode": SPECIAL,
            "pattern": pattern1
        }
        ret.append([tx1, reduce_axis1])

        return ret

    # noinspection PyMethodMayBeStatic
    def _generate_pattern(self, size):
        pat = [COMMON, REDUCE]
        pattern = [pat[i % 2] for i in range(size + 1)]
        reduce_axis = list(range(1, size + 1, 2))
        return pattern, reduce_axis
