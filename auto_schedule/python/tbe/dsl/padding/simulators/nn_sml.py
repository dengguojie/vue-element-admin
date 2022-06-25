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
relu simulator
"""
from abc import abstractmethod
from typing import Union

import tbe.dsl.padding.graph as m_graph
import tbe.dsl.padding.simulator as m_simulator
import tbe.dsl.padding.util as util
from tbe.dsl.padding.simulators import spvalue
from tbe.dsl.padding.value import PaddingValue
from tbe.dsl.padding.value import SettingValue


class ReluBaseSimulator(m_simulator.Simulator):
    def __init__(self, node):
        # type: (m_graph.Node) -> None
        super().__init__(node)
        self._node0 = self._node.get_input_nodes()[0]
        self._dtype = self._node.get_dtype()

    def adjust_calc(self):
        # type: () -> None
        r = self._adjust()
        tensor = self._node.get_tensor()
        r.add_target(tensor)

        pvalue = self._do_calc()
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        # type: () -> Union[PaddingValue, SettingValue]
        svalues = self._node0.get_svalues()
        pvalue = self._node0.get_pvalue()

        return next(iter(svalues), pvalue)

    @abstractmethod
    def _do_calc(self):
        # type: () -> PaddingValue
        pass


class ReluSimulator(ReluBaseSimulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_single_relu"

    def _do_calc(self):
        # type: () -> PaddingValue
        pvalue0 = util.get_pvalue(self._node0, self._node.get_tensor())
        return spvalue.relu(pvalue0)


class LeakyReluSimulator(ReluBaseSimulator):
    def __init__(self, node):
        # type: (m_graph.Node) -> bool
        super().__init__(node)
        self._scalar1 = self._node.get_tensor().op.body[0].args[1]

    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_single_lrelu"

    def _do_calc(self):
        # type: () -> PaddingValue
        pvalue0 = util.get_pvalue(self._node0, self._node.get_tensor())
        return spvalue.lrelu(pvalue0, self._scalar1)
