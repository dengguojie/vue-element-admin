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
single simulator
"""
import abc
from typing import Union

import tbe.dsl.base.padding.graph as m_graph
import tbe.dsl.base.padding.simulator as m_simulator
import tbe.dsl.base.padding.smath as smath
import tbe.dsl.base.padding.util as util
from tbe.dsl.base.padding.value import (PaddingValue, PaddingValueType,
                                        SettingValue, SettingValueType)


class SingleSimulator(m_simulator.Simulator, abc.ABC):
    def __init__(self, node):
        # type: (m_graph.Node) -> bool
        super().__init__(node)
        self._dtype = self._node.get_dtype()
        self._node0 = self._node.get_input_nodes()[0]

    def adjust_calc(self):
        r = self._adjust()
        tensor = self._node.get_tensor()
        if isinstance(r, SettingValue):
            r.add_target(tensor)
        elif r is None:
            pvalue0 = self._node0.get_pvalue()
            pvalue0.add_target(tensor)
        else:
            dtype = self._node0.get_dtype()
            svalue = SettingValue(SettingValueType.NORMAL, self._dtype)
            svalue.condition = util.get_normal_condition(self._node0)
            svalue.value = util.new_np_num_x(r, dtype)
            svalue.add_target(tensor)
            self._node0.add_svalue(svalue)

        pvalue0 = util.get_pvalue(self._node0, self._node.get_tensor())
        pvalue = self._do_calc(pvalue0)
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        if not util.exist_pad(self._node0):
            return None

        svalues = self._node0.get_svalues()
        stp = util.svalue_to_pvalue
        for sv in svalues:
            if self._do_adjust(stp(sv)) is None:
                return sv

        pvalue = self._node0.get_pvalue()
        return self._do_adjust(pvalue)

    @abc.abstractclassmethod
    def _do_adjust(cls, pvalue0):
        # type: (PaddingValue) -> Union[int, float]
        """"""

    @abc.abstractclassmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        """"""


class AbsSimulator(SingleSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_abs"

    @classmethod
    def _do_adjust(cls, pvalue0):
        # type: (PaddingValue) -> Union[int, float]
        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            return util.new_pvalue_x(smath.abs_(pvalue0.value), dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] in AbsSimulator")
        return None


class ExpSimulator(SingleSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_exp"

    @classmethod
    def _do_adjust(cls, pvalue0):
        # type: (PaddingValue) -> Union[int, float]
        pv_type0 = pvalue0.type
        if pv_type0 == PaddingValueType.EXACT:
            if not util.check_valid(lambda: smath.exp_(pvalue0.value)):
                return 0
        if pv_type0 == PaddingValueType.ANY:
            return 0

        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            return util.new_pvalue_x(smath.exp_(pvalue0.value), dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] in ExpSimulator")
        return None


class LogSimulator(SingleSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_log"

    @classmethod
    def _do_adjust(cls, pvalue0):
        # type: (PaddingValue) -> Union[int, float]
        pv_type0 = pvalue0.type
        if pv_type0 == PaddingValueType.EXACT:
            if not util.check_valid(lambda: smath.log_(pvalue0.value)):
                return 1
        if pv_type0 == PaddingValueType.ANY:
            return 1

        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            return util.new_pvalue_x(smath.log_(pvalue0.value), dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] in LogSimulator")
        return None


class RecSimulator(SingleSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_rec"

    @classmethod
    def _do_adjust(cls, pvalue0):
        # type: (PaddingValue) -> Union[int, float]
        pv_type0 = pvalue0.type
        if pv_type0 == PaddingValueType.EXACT:
            if not util.check_valid(lambda: smath.rec_(pvalue0.value)):
                return 1
        if pv_type0 == PaddingValueType.ANY:
            return 1

        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            return util.new_pvalue_x(smath.rec_(pvalue0.value), dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] in RecSimulator")
        return None


class SqrtSimulator(SingleSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_sqrt"

    @classmethod
    def _do_adjust(cls, pvalue0):
        # type: (PaddingValue) -> Union[int, float]
        pv_type0 = pvalue0.type
        if pv_type0 == PaddingValueType.EXACT:
            if not util.check_valid(lambda: smath.sqrt_(pvalue0.value)):
                return 0
        if pv_type0 == PaddingValueType.ANY:
            return 0

        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            return util.new_pvalue_x(smath.sqrt_(pvalue0.value), dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] in SqrtSimulator")
        return None


class RsqrtSimulator(SingleSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_rsqrt"

    @classmethod
    def _do_adjust(cls, pvalue0):
        # type: (PaddingValue) -> Union[int, float]
        pv_type0 = pvalue0.type
        if pv_type0 == PaddingValueType.EXACT:
            if not util.check_valid(lambda: smath.rsqrt_(pvalue0.value)):
                return 1
        if pv_type0 == PaddingValueType.ANY:
            return 1

        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            return util.new_pvalue_x(smath.rsqrt_(pvalue0.value), dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] in RsqrtSimulator")
        return None
