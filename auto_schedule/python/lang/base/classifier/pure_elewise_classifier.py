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
classifier of shape in pure elewise
"""

from . import util

COMMON = "common"
SPECIAL = "special"
CONST = "const"


class PureElewiseClassifier:
    """
    Pure elewise classifier
    """

    def __init__(self, ins: list):
        """
        init
        :param ins:
        """
        self.ins = ins
        self.shapes = [x["shape"] for x in self.ins]
        self.dim_length = len(self.shapes[0])

    def classify(self):
        """
        classify
        :return:
        """
        return self._classify_const() if self._is_const() else self._classify_var()

    def _is_const(self):
        for i in range(self.dim_length):
            if max([s[i] for s in self.shapes]) == -1:
                return False
        return True

    def _classify_const(self):
        def get_dim(i):
            return max([s[i] for s in self.shapes])

        shape = [get_dim(i) for i in range(self.dim_length)]
        return [[ConstMode.gen_in(shape) for _ in self.ins]]

    def _classify_var(self):
        ins = []
        for x in self.ins:
            in_x = SpecialMode.gen_in([-1])
            if "range" in x:
                in_x["range"] = [util.combine_range(x["range"])]
            ins.append(in_x)

        return [ins]


class ConstMode:
    """
    ConstMode
    """

    @classmethod
    def gen_in(cls, shape):
        """
        gen_in
        :param shape:
        :return:
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "mode": CONST,
                "support_broadcast": False,
                }


class SpecialMode:
    """
    SpecialMode
    """

    @classmethod
    def gen_in(cls, shape):
        """
        gen_in
        :param shape:
        :return:
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "mode": SPECIAL,
                "support_broadcast": False,
                "pattern": (COMMON,)
                }
