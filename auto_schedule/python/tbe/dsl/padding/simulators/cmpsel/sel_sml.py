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
Select simulator
"""

import tbe.dsl.padding.graph as m_graph
import tbe.dsl.padding.simulator as m_simulator
import tbe.dsl.padding.simulators.cmpsel.scmp as scmp
import tbe.dsl.padding.util as util
from tbe.dsl.padding.value import PaddingValue
from tbe.dsl.padding.value import PaddingValueType
from tbe.tvm.expr import Call


class SelSimulator(m_simulator.Simulator):
    def __init__(self, node):
        # type: (m_graph.Node) -> None
        super().__init__(node)
        self._dtype = self._node.get_dtype()
        self._condition, self._slhs, self._srhs = self._extract_inputs()

    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_multiple_sel"

    def adjust_calc(self):
        # type: () -> None
        rs = self._adjust()
        tensor = self._node.get_tensor()
        for r in rs:
            r.add_target(tensor)

        pvalues = self._get_pvalues()
        pvalue = self._do_calc(*pvalues)
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        # type: () -> tuple
        return self._adjust_in_tensor()

    def _adjust_in_tensor(self):
        # type: () -> tuple
        svalues = self._condition.get_svalues()
        pvalue = self._condition.get_pvalue()

        return (next(iter(svalues), pvalue),)

    def _get_pvalues(self):
        # type: () -> tuple
        def get_pvalue(_node_x):
            return util.get_pvalue(_node_x, self._node.get_tensor())

        pvalues = [get_pvalue(self._condition)]
        return pvalues

    def _do_calc(self, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type0 = pvalue0.type
        dtype = self._dtype
        if pv_type0 == PaddingValueType.EXACT:
            if util.is_0_pvalue(pvalue0):
                rpv = util.new_pvalue_0(dtype)
            else:
                rpv = util.new_pvalue_1(dtype)
        elif pv_type0 == PaddingValueType.TENSOR:
            rpv = util.new_pvalue_tensor(dtype)
        elif pv_type0 == PaddingValueType.ANY:
            rpv = util.new_pvalue_tensor(dtype)
        else:
            util.raise_error(f"Unsupported padding value type[{pv_type0}] in SelSimulator calc.")

        tensor = self._node.get_tensor()

        if util.is_1_pvalue(rpv):
            return scmp.deal_selected_hs(self._slhs, tensor)

        if util.is_0_pvalue(rpv):
            return scmp.deal_selected_hs(self._srhs, tensor)

        return rpv

    def _extract_inputs(self):
        # type: () -> tuple
        def get_node(call_x):
            return self._node.get_node(call_x.func.output(call_x.value_index))

        body = self._node.get_tensor().op.body[0]
        condition = self._node.get_input_nodes()[0]
        slhs, srhs = body.true_value, body.false_value

        if isinstance(slhs, Call):
            slhs = get_node(slhs)

        if isinstance(srhs, Call):
            srhs = get_node(srhs)

        return condition, slhs, srhs
