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
classifier of shape in pure elewise
"""
from functools import reduce as reduceIns
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation

from . import util

COMMON = "common"
SPECIAL = "special"
CONST = "const"
EMPTY = "empty"
UNKNOWN_RANK = -2


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
        self.dim_length = max(len(_ins["shape"]) for _ins in self.ins)
        self.is_unknown_rank = self._check_update_unknown_rank()
        operation.get_context().add("_unknown_rank", self.is_unknown_rank)
        self.shapes = [x["shape"] for x in self.ins]

    def classify(self):
        """
        classify
        :return:
        """
        return self._classify_const() if self._is_const() else self._classify_var()

    def _check_update_unknown_rank(self):
        is_unknown_rank = False
        for _in in self.ins:
            shapes = list(_in["shape"])
            if UNKNOWN_RANK in shapes:
                if len(shapes) != 1:
                    dict_args = {}
                    dict_args["errCode"] = "E90001"
                    dict_args["detailed_cause"] = "if the shape contains -2, it must be [-2] or (-2,)"
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                _in["shape"] = [-1] * self.dim_length
                _in["range"] = [(1, None)] * self.dim_length
                is_unknown_rank = True
        return is_unknown_rank

    def _is_const(self):
        for i in range(self.dim_length):
            if max(s[i] for s in self.shapes) == -1:
                return False
        return True

    def _classify_const(self):
        def get_dim(i):
            return max(s[i] for s in self.shapes)

        const_shape = [get_dim(i) for i in range(self.dim_length)]
        shape = [reduceIns(lambda x, y: x * y, const_shape)]

        inputs = [ConstMode.gen_in(shape) for _ in self.ins]
        for input_x in inputs:
            input_x["const_shape"] = const_shape

        return [inputs]

    def _classify_var(self):
        maybe_empty_tensor = False
        must_empty_tensor = False
        ins = []
        for x in self.ins:
            in_x = SpecialMode.gen_in([-1])
            in_x["range"] = [util.combine_range(x["range"])]
            maybe_empty_tensor = maybe_empty_tensor or 0 in in_x["range"][0]
            if 0 in x["shape"] or (0, 0) in x["range"] or [0, 0] in x["range"]:
                must_empty_tensor = True
                break
            ins.append(in_x)

        ret = []
        if not must_empty_tensor:
            ret.append(ins)
        if maybe_empty_tensor or self.is_unknown_rank:
            input_length = len(self.ins)
            ins = [EmptyMode.gen_in()] * input_length
            ret.append(ins)

        return ret


class EmptyMode:
    """
    Empty Mode
    """

    @classmethod
    def gen_in(cls):
        """
        generate input
        :return:
        """
        return {"shape": (0,),
                "range": [(0, 0)],
                "support_broadcast": True,
                "mode": EMPTY,
                }


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
