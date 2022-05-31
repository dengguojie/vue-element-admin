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
broadcast simulator
"""
import abc
from typing import Callable, Tuple, Union

import tbe.dsl.base.padding.graph as m_graph
import tbe.dsl.base.padding.simulator as m_simulator
import tbe.dsl.base.padding.util as util
from tbe.dsl.base import d_format_util as dfu
from tbe.dsl.base.padding.value import (PaddingValue, SettingValue,
                                        SettingValueType)


class ScalarBroadcastSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "broadcast"

    def adjust_calc(self):
        # type: () -> None
        dtype = self._node.get_dtype()
        self._node.set_pvalue(util.new_pvalue_tensor(dtype))


class TensorBroadcastSimulator(m_simulator.Simulator, abc.ABC):
    def __init__(self, node):
        # type: (m_graph.Node) -> bool
        super().__init__(node)
        self._node0 = self._node.get_input_nodes()[0]
        self._dtype = self._node.get_dtype()

    def adjust_calc(self):
        r = self._adjust()
        tensor = self._node.get_tensor()
        if isinstance(r, SettingValue):
            r.add_target(tensor)
        elif r is None:
            self._node0.get_pvalue().add_target(tensor)
        else:
            svalue = SettingValue(SettingValueType.BROADCAST, self._dtype)
            svalue.condition, svalue.value = r
            svalue.add_target(tensor)
            self._node0.add_svalue(svalue)

        pvalue0 = util.get_pvalue(self._node0, self._node.get_tensor())
        pvalue = util.new_pvalue(pvalue0)
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        svalues = self._node0.get_svalues()
        for sv in svalues:
            if self._do_adjust(sv) is None:
                return sv

        pvalue = self._node0.get_pvalue()
        return self._do_adjust(pvalue)

    def _do_adjust(self, psvalue0):
        # type: (Union[PaddingValue, SettingValue]) -> Tuple(Callable, Callable)
        x_shape = self._node0.get_shape()
        y_shape = self._node.get_shape()
        x_c1, x_c0, x_c = dfu.get_c1(x_shape), dfu.get_c0(x_shape), dfu.get_c(x_shape)
        y_c1, y_c0, y_c = dfu.get_c1(y_shape), dfu.get_c0(y_shape), dfu.get_c(y_shape)

        if util.eq_expr((x_c1, x_c0, x_c), (y_c1, y_c0, y_c)):
            # no pad axis brc: (n, c1, h, w, 16) -> (n', c1, h', w', 16)
            return None

        if util.eq_expr((x_c1, x_c0), (y_c1, y_c0)):
            # src c eq 1, dst c le 16: (n, 1, h, w, 16) -> (n, 1, h, w, 16)
            if psvalue0.type != SettingValueType.BROADCAST:
                return util.get_brc_condition_value(self._node0)

        if util.eq_expr(x_c0, 1) and util.eq_expr(y_c0, 16):
            # (n, 1, h, w, 1) -> (n, 1, h, w, 16) or (n, c1, h, w, 16)
            return None

        if util.eq_expr(x_c0, y_c0) and not util.eq_expr(x_c1, y_c1):
            # (n, 1, h, w, 16) -> (n, c1, h, w, 16)
            if not self._is_valid_pad():
                if psvalue0.type != SettingValueType.BROADCAST:
                    return util.get_brc_condition_value(self._node0)

        return None

    def _is_valid_pad(self):
        return util.is_brc_node(self._node0) and util.is_tensor_pvalue(self._node0.get_pvalue())


class ExplicitBroadcastSimulator(TensorBroadcastSimulator):
    @classmethod
    def get_type(cls):
        return "unified_broadcast"


class ImplicitBroadcastSimulator(TensorBroadcastSimulator):
    @classmethod
    def get_type(cls):
        return "unknown_broadcast"
