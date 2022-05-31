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
reduce simulator
"""
import abc
from typing import Union

import tbe.dsl.base.padding.graph as m_graph
import tbe.dsl.base.padding.simulator as m_simulator
import tbe.dsl.base.padding.util as util
from tbe.dsl.base.padding.value import (PaddingValue, PaddingValueType,
                                        SettingValue, SettingValueType)


class ReduceSimulator(m_simulator.Simulator, abc.ABC):
    def __init__(self, node):
        # type: (m_graph.Node) -> bool
        super().__init__(node)
        self._dtype = self._node.get_dtype()
        self._node0 = self._node.get_input_nodes()[0]
        self._reduce_with_pad = not util.exist_pad(self._node)

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

        if util.exist_pad(self._node):
            pvalue0 = util.get_pvalue(self._node0, self._node.get_tensor())
            pvalue = self._do_calc(pvalue0)
            self._node.set_pvalue(pvalue)
        else:
            self._node.set_pvalue(util.new_pvalue_tensor(self._dtype))

    def _adjust(self):
        if not util.exist_pad(self._node0):
            return None

        svalues = self._node0.get_svalues()
        stp = util.svalue_to_pvalue
        for sv in svalues:
            if self._do_adjust(self._reduce_with_pad, stp(sv)) is None:
                return sv

        pvalue = self._node0.get_pvalue()
        return self._do_adjust(self._reduce_with_pad, pvalue)

    @abc.abstractclassmethod
    def _do_adjust(cls, reduce_with_pad, pvalue0):
        # type: (bool, PaddingValue) -> Union[int, float]
        """"""

    @abc.abstractclassmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        """"""


class ReduceSumSimulator(ReduceSimulator):
    @classmethod
    def get_type(cls):
        return "reduce_sum"

    @classmethod
    def _do_adjust(cls, reduce_with_pad, pvalue0):
        # type: (bool, PaddingValue) -> Union[int, float]
        pv_type0 = pvalue0.type
        if reduce_with_pad:
            if pv_type0 == PaddingValueType.EXACT:
                if not util.equal_0(pvalue0.value):
                    return 0
            if pv_type0 == PaddingValueType.TENSOR:
                return 0
            if pv_type0 == PaddingValueType.ANY:
                return 0
        else:
            if pv_type0 == PaddingValueType.EXACT:
                if not util.equal_0(pvalue0.value):
                    return 0
            if pv_type0 == PaddingValueType.ANY:
                return 0

        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type0 == PaddingValueType.EXACT:
            if util.equal_0(pvalue0.value):
                return util.new_pvalue_x(pvalue0.value, dtype)
            return util.new_pvalue_any(dtype)

        if pv_type0 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type0 == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type0}] in ReduceSumSimulator")
        return None


class ReduceMaxSimulator(ReduceSimulator):
    @classmethod
    def get_type(cls):
        return "reduce_max"

    @classmethod
    def _do_adjust(cls, reduce_with_pad, pvalue0):
        # type: (bool, PaddingValue) -> Union[int, float]
        pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        min_value = util.get_min(dtype)
        if reduce_with_pad:
            if pv_type0 == PaddingValueType.EXACT:
                if not util.equal_min(pvalue0.value):
                    return min_value
            if pv_type0 == PaddingValueType.TENSOR:
                return min_value
            if pv_type0 == PaddingValueType.ANY:
                return min_value

        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type0 == PaddingValueType.EXACT:
            return util.new_pvalue_x(pvalue0.value, dtype)

        if pv_type0 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type0 == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type0}] in ReduceMaxSimulator")
        return None


class ReduceMinSimulator(ReduceSimulator):
    @classmethod
    def get_type(cls):
        return "reduce_min"

    @classmethod
    def _do_adjust(cls, reduce_with_pad, pvalue0):
        # type: (bool, PaddingValue) -> Union[int, float]
        pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        max_value = util.get_max(dtype)
        if reduce_with_pad:
            if pv_type0 == PaddingValueType.EXACT:
                if not util.equal_max(pvalue0.value):
                    return max_value
            if pv_type0 == PaddingValueType.TENSOR:
                return max_value
            if pv_type0 == PaddingValueType.ANY:
                return max_value

        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type0 == PaddingValueType.EXACT:
            return util.new_pvalue_x(pvalue0.value, dtype)

        if pv_type0 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type0 == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type0}] in ReduceMinSimulator")
        return None


class ReduceProdSimulator(ReduceSimulator):
    @classmethod
    def get_type(cls):
        return "reduce_prod"

    @classmethod
    def _do_adjust(cls, reduce_with_pad, pvalue0):
        # type: (bool, PaddingValue) -> Union[int, float]
        pv_type0 = pvalue0.type
        if reduce_with_pad:
            if pv_type0 == PaddingValueType.EXACT:
                if not util.equal_1(pvalue0.value):
                    return 1
            if pv_type0 == PaddingValueType.TENSOR:
                return 1
            if pv_type0 == PaddingValueType.ANY:
                return 1
        else:
            if pv_type0 == PaddingValueType.EXACT:
                if not util.equal_0(pvalue0.value) and not util.equal_1(pvalue0.value):
                    return 0
            if pv_type0 == PaddingValueType.ANY:
                return 0

        return None

    @classmethod
    def _do_calc(cls, pvalue0):
        # type: (PaddingValue) -> PaddingValue
        pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        if pv_type0 == PaddingValueType.EXACT:
            if util.equal_1(pvalue0.value) or util.equal_0(pvalue0.value):
                return util.new_pvalue_x(pvalue0.value, dtype)
            return util.new_pvalue_any(dtype)

        if pv_type0 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type0 == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type0}] in ReduceProdSimulator")
        return None


class TupleReduceSumSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "tuple_reduce_sum"

    def adjust_calc(self):
        util.raise_error("Unsupported.")
