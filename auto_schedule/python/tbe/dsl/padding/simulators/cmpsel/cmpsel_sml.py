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
cmpsel simulator
"""
import abc

import tbe.dsl.padding.graph as m_graph
import tbe.dsl.padding.simulator as m_simulator
import tbe.dsl.padding.simulators.cmpsel.scmp as scmp
import tbe.dsl.padding.util as util
from tbe.dsl.padding.value import PaddingValue
from tbe.tvm.expr import Call
from tbe.tvm.expr import ConstExpr


class BaseCmpselSimulator(m_simulator.Simulator, abc.ABC):
    def __init__(self, node):
        # type: (m_graph.Node) -> None
        super().__init__(node)
        self._dtype = self._node.get_dtype()
        self._lhs, self._rhs, self._slhs, self._srhs = self._extract_inputs()

    @abc.abstractclassmethod
    def get_cmp_mode(cls):
        # type: () -> scmp.CmpMode
        pass

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
        if isinstance(self._rhs, m_graph.Node):
            if self._lhs != self._rhs:
                return self._adjust_in_ne_tensors()

        return self._adjust_in_tensor()

    def _adjust_in_tensor(self):
        # type: () -> tuple
        svalues = self._lhs.get_svalues()
        pvalue = self._lhs.get_pvalue()

        return (next(iter(svalues), pvalue),)

    def _adjust_in_ne_tensors(self):
        # type: () -> tuple
        svalues0 = self._lhs.get_svalues()
        svalues1 = self._rhs.get_svalues()
        pvalue0 = self._lhs.get_pvalue()
        pvalue1 = self._rhs.get_pvalue()

        return (next(iter(svalues0), pvalue0), next(iter(svalues1), pvalue1))

    def _get_pvalues(self):
        # type: () -> tuple
        def get_pvalue(_node_x):
            return util.get_pvalue(_node_x, self._node.get_tensor())

        pvalues = []
        pvalues.append(get_pvalue(self._lhs))

        if isinstance(self._rhs, m_graph.Node):
            pvalues.append(get_pvalue(self._rhs))
        elif isinstance(self._rhs, ConstExpr):
            pvalues.append(util.new_pvalue_x(util.tvm_const_to_np(self._rhs), self._rhs.dtype))
        else:
            pvalues.append(util.new_pvalue_tensor(self._rhs.dtype))

        return pvalues

    def _do_calc(self, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        rpv = scmp.cmp(pvalue0, pvalue1, self._dtype, self.get_cmp_mode())

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
        condition = body.condition
        lhs, rhs = condition.a, condition.b
        slhs, srhs = body.true_value, body.false_value

        lhs = get_node(lhs)

        if isinstance(rhs, Call):
            rhs = get_node(rhs)

        if isinstance(slhs, Call):
            slhs = get_node(slhs)

        if isinstance(srhs, Call):
            srhs = get_node(srhs)

        return [lhs, rhs, slhs, srhs]


class CmpselGtSimulator(BaseCmpselSimulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_binary_cmpsel_gt"

    @classmethod
    def get_cmp_mode(cls):
        # type: () -> scmp.CmpMode
        return scmp.CmpMode.GT


class CmpselGeSimulator(BaseCmpselSimulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_binary_cmpsel_ge"

    @classmethod
    def get_cmp_mode(cls):
        # type: () -> scmp.CmpMode
        return scmp.CmpMode.GE


class CmpselLtSimulator(BaseCmpselSimulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_binary_cmpsel_lt"

    @classmethod
    def get_cmp_mode(cls):
        # type: () -> scmp.CmpMode
        return scmp.CmpMode.LT


class CmpselLeSimulator(BaseCmpselSimulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_binary_cmpsel_le"

    @classmethod
    def get_cmp_mode(cls):
        # type: () -> scmp.CmpMode
        return scmp.CmpMode.LE


class CmpselEqSimulator(BaseCmpselSimulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_binary_cmpsel_eq"

    @classmethod
    def get_cmp_mode(cls):
        # type: () -> scmp.CmpMode
        return scmp.CmpMode.EQ


class CmpselNeSimulator(BaseCmpselSimulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_binary_cmpsel_ne"

    @classmethod
    def get_cmp_mode(cls):
        # type: () -> scmp.CmpMode
        return scmp.CmpMode.NE
