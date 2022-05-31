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
from typing import Union

import tbe.dsl.base.padding.graph as m_graph
import tbe.dsl.base.padding.simulator as m_simulator
import tbe.dsl.base.padding.smath as smath
import tbe.dsl.base.padding.util as util
from tbe.dsl.base.padding.value import (PaddingValue, PaddingValueType,
                                        SettingValue)
from tbe.tvm.expr import ConstExpr, Expr


class ReluSimulator(m_simulator.Simulator):
    def __init__(self, node):
        # type: (m_graph.Node) -> None
        super().__init__(node)
        self._node0 = self._node.get_input_nodes()[0]
        self._dtype = self._node.get_dtype()

    @classmethod
    def get_type(cls):
        return "elewise_single_relu"

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            return util.new_pvalue_x(smath.relu_(pvalue0.value), dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] in {cls.__name__}.")
        return None

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


class LeakyReluSimulator(m_simulator.Simulator):
    def __init__(self, node):
        # type: (m_graph.Node) -> bool
        super().__init__(node)
        self._node0 = self._node.get_input_nodes()[0]
        self._scalar1 = util.get_hs_b(self._node)

    @classmethod
    def get_type(cls):
        return "elewise_single_lrelu"

    @classmethod
    def _do_calc(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, Expr]) -> PaddingValue
        pv_type = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            if util.equal_0(pvalue0.value):
                return util.new_pvalue_0(dtype)
            if isinstance(scalar1, ConstExpr):
                new_value = smath.lrelu_(pvalue0.value, util.tvm_const_to_np(scalar1))
                return util.new_pvalue_x(new_value, dtype)
            return util.new_pvalue_any(dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] in {cls.__name__}.")
        return None

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
        pvalue = self._do_calc(pvalue0, self._scalar1)
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        if not util.exist_pad(self._node0):
            return None

        svalues = self._node0.get_svalues()
        for sv in svalues:
            return sv

        return None
