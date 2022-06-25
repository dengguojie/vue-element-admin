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
bitwise simulator
"""
import abc
from typing import List
from typing import Tuple
from typing import Union

import tbe.dsl.padding.graph as m_graph
import tbe.dsl.padding.simulator as m_simulator
import tbe.dsl.padding.smath as smath
import tbe.dsl.padding.util as util
from tbe.dsl.padding.value import PaddingValue
from tbe.dsl.padding.value import SettingValue


class BaseBitwiseSimulator(m_simulator.Simulator, abc.ABC):
    def __init__(self, node):
        # type: (m_graph.Node) -> bool
        super().__init__(node)
        self._dtype = self._node.get_dtype()
        self._nodes = self._extract_inputs()

    def adjust_calc(self):
        # type: () -> None
        tensor = self._node.get_tensor()
        for r in self._adjust():
            r.add_target(tensor)

        pvalue = self._do_calc(*self._get_pvalues())
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        # type: () -> Tuple[Union[PaddingValue, SettingValue], Union[PaddingValue, SettingValue]]
        if self._nodes[0] == self._nodes[1]:
            return self._adjust_in_eq_ins()

        return self._adjust_in_ne_ins()

    def _adjust_in_eq_ins(self):
        # type: () -> Tuple[Union[PaddingValue, SettingValue]]
        svalues = self._nodes[0].get_svalues()
        pvalue = self._nodes[0].get_pvalue()

        return (next(iter(svalues), pvalue),)

    def _adjust_in_ne_ins(self):
        # type: () -> Tuple[Union[PaddingValue, SettingValue], Union[PaddingValue, SettingValue]]
        svalues0 = self._nodes[0].get_svalues()
        svalues1 = self._nodes[1].get_svalues()
        pvalue0 = self._nodes[0].get_pvalue()
        pvalue1 = self._nodes[1].get_pvalue()

        return next(iter(svalues0), pvalue0), next(iter(svalues1), pvalue1)

    def _get_pvalues(self):
        # type: () -> List[PaddingValue]
        def get_pvalue(_node_x):
            return util.get_pvalue(_node_x, self._node.get_tensor())

        return [get_pvalue(node_x) for node_x in self._nodes]

    @abc.abstractmethod
    def _do_calc(self, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        pass

    def _extract_inputs(self):
        # type: () -> List[m_graph.Node]
        nodes = self._node.get_input_nodes()
        if len(nodes) == 1:
            nodes.append(nodes[0])
        return nodes


class BitwiseAndSimulator(BaseBitwiseSimulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_binary_and"

    def _do_calc(self, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        return util.new_pvalue_x(smath.bitwise_and_(pvalue0.value, pvalue1.value), self._dtype)


class BitwiseOrSimulator(BaseBitwiseSimulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_binary_or"

    def _do_calc(self, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        return util.new_pvalue_x(smath.bitwise_or_(pvalue0.value, pvalue1.value), self._dtype)


class BitwiseNotSimulator(m_simulator.Simulator):
    def __init__(self, node):
        # type: (m_graph.Node) -> bool
        super().__init__(node)
        self._dtype = self._node.get_dtype()
        self._node0 = self._node.get_input_nodes()[0]

    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_single_not"

    def adjust_calc(self):
        # type: () -> None
        tensor = self._node.get_tensor()
        self._adjust().add_target(tensor)

        pvalue0 = util.get_pvalue(self._node0, self._node.get_tensor())
        pvalue = self._do_calc(pvalue0)
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        # type: () -> Tuple[Union[PaddingValue, SettingValue]]
        svalues = self._node0.get_svalues()
        pvalue = self._node0.get_pvalue()

        return next(iter(svalues), pvalue)

    def _do_calc(self, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        return util.new_pvalue_x(smath.bitwise_not_(pvalue0.value), self._dtype)
