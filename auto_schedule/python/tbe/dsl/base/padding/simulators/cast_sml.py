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
cast simulator
"""
import abc

import tbe.dsl.base.padding.graph as m_graph
import tbe.dsl.base.padding.simulator as m_simulator
import tbe.dsl.base.padding.smath as smath
import tbe.dsl.base.padding.util as util
from tbe.dsl.base.padding.value import (PaddingValue, PaddingValueType,
                                        SettingValue)


class BaseCastSimulator(m_simulator.Simulator, abc.ABC):
    def __init__(self, node):
        # type: (m_graph.Node) -> None
        super().__init__(node)
        self._node0 = self._node.get_input_nodes()[0]
        self._target_dtype = util.get_target_dtype(self._node)

    def adjust_calc(self):
        # type: () -> None
        r = self._adjust()
        tensor = self._node.get_tensor()
        if isinstance(r, SettingValue):
            r.add_target(tensor)
        elif r is None:
            pvalue0 = self._node0.get_pvalue()
            pvalue0.add_target(tensor)

        pvalue0 = util.get_pvalue(self._node0, self._node.get_tensor())
        pvalue = self._do_calc(pvalue0)
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        if not util.exist_pad(self._node0):
            return None

        svalues = self._node0.get_svalues()
        for sv in svalues:
            return sv

        return None

    def _do_calc(self, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type0 = pvalue0.type
        if pv_type0 == PaddingValueType.EXACT:
            new_value = self._get_cast_func()(pvalue0.value, self._target_dtype)
            return util.new_pvalue_x(new_value, self._target_dtype)

        if pv_type0 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(self._target_dtype)

        if pv_type0 == PaddingValueType.ANY:
            return util.new_pvalue_any(self._target_dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type0}] in {self.get_type()}")
        return None

    @abc.abstractclassmethod
    def _get_cast_func(cls):
        """"""


class CastSimulator(BaseCastSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_cast"

    @classmethod
    def _get_cast_func(cls):
        return smath.cast_


class CeilSimulator(BaseCastSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_ceil"

    @classmethod
    def _get_cast_func(cls):
        return smath.ceil_


class FloorSimulator(BaseCastSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_floor"

    @classmethod
    def _get_cast_func(cls):
        return smath.floor_


class TruncSimulator(BaseCastSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_trunc"

    @classmethod
    def _get_cast_func(cls):
        return smath.trunc_


class RoundSimulator(BaseCastSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_round"

    @classmethod
    def _get_cast_func(cls):
        return smath.round_


class RounddSimulator(BaseCastSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_round_d"

    @classmethod
    def _get_cast_func(cls):
        return smath.round_d_
