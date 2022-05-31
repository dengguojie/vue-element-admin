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
binary simulator
"""
import abc
from typing import Tuple, Union

import tbe.dsl.base.padding.graph as m_graph
import tbe.dsl.base.padding.simulator as m_simulator
import tbe.dsl.base.padding.smath as smath
import tbe.dsl.base.padding.util as util
from tbe.dsl.base.padding.value import (PaddingValue, PaddingValueType,
                                        SettingValue, SettingValueType)


class BinarySimulator(m_simulator.Simulator, abc.ABC):
    def __init__(self, node):
        # type: (m_graph.Node) -> bool
        super().__init__(node)
        self._dtype = self._node.get_dtype()
        self._nodes = self._extract_inputs()

    def adjust_calc(self):
        rs = self._adjust()
        ts = [(SettingValue, PaddingValue) for _ in rs]
        tensor = self._node.get_tensor()
        if util.match_instance(rs, ts):
            for r in rs:
                r.add_target(tensor)
        else:
            for i, r in enumerate(rs):
                node_x = self._nodes[i]
                pvalue = node_x.get_pvalue()
                if r is None:
                    pvalue.add_target(tensor)
                else:
                    dtype = node_x.get_dtype()
                    svalue = SettingValue(SettingValueType.NORMAL, dtype)
                    svalue.condition = util.get_normal_condition(node_x)
                    svalue.value = util.new_np_num_x(r, dtype)
                    svalue.add_target(tensor)
                    node_x.add_svalue(svalue)

        pvalue = self._do_calc(*self._get_pvalues())
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        if not util.exist_pad(self._node):
            return None, None

        if self._nodes[0] == self._nodes[1]:
            return self._adjust_in_eq_ins()

        return self._adjust_in_ne_ins()

    def _adjust_in_eq_ins(self):
        svalues = self._nodes[0].get_svalues()
        for sv in svalues:
            pv = util.svalue_to_pvalue(sv)
            if self._do_adjust(pv, pv) == (None, None):
                return (sv,)

        pvalue = self._nodes[0].get_pvalue()
        return (self._do_adjust(pvalue, pvalue)[0],)

    def _adjust_in_ne_ins(self):
        svalues0 = self._nodes[0].get_svalues()
        svalues1 = self._nodes[1].get_svalues()
        pvalue0 = self._nodes[0].get_pvalue()
        pvalue1 = self._nodes[1].get_pvalue()
        stp = util.svalue_to_pvalue

        for sv0 in svalues0:
            for sv1 in svalues1:
                pv0, pv1 = stp(sv0), stp(sv1)
                if self._do_adjust(pv0, pv1) == (None, None):
                    return sv0, sv1

        for sv1 in svalues1:
            pv1 = stp(sv1)
            if self._do_adjust(pvalue0, pv1) == (None, None):
                return pvalue0, sv1

        for sv0 in svalues0:
            pv0 = stp(sv0)
            if self._do_adjust(pv0, pvalue1) == (None, None):
                return sv0, pvalue1

        return self._do_adjust(pvalue0, pvalue1)

    def _get_pvalues(self):
        def get_pvalue(_node_x):
            return util.get_pvalue(_node_x, self._node.get_tensor())

        return [get_pvalue(node_x) for node_x in self._nodes]

    @abc.abstractclassmethod
    def _do_adjust(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> Tuple(int, Union[int, float])
        """"""

    @abc.abstractclassmethod
    def _do_calc(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        """"""

    def _extract_inputs(self):
        nodes = self._node.get_input_nodes()
        if len(nodes) == 1:
            nodes.append(nodes[0])
        return nodes


class AddSimulator(BinarySimulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_add"

    @classmethod
    def _do_adjust(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> Tuple(int, Union[int, float])
        add_pv_types = (pvalue0.type, pvalue1.type)
        ne_ins = id(pvalue0) != id(pvalue1)
        if add_pv_types == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            if not util.check_valid(lambda: smath.add_(pvalue0.value, pvalue1.value)):
                return (0, None) if ne_ins else (0, 0)

        if add_pv_types == (PaddingValueType.EXACT, PaddingValueType.TENSOR):
            if not util.is_0_pvalue(pvalue0):
                return 0, None

        if add_pv_types == (PaddingValueType.EXACT, PaddingValueType.ANY):
            if not util.is_0_pvalue(pvalue0):
                return 0, None

        if add_pv_types == (PaddingValueType.TENSOR, PaddingValueType.EXACT):
            if not util.is_0_pvalue(pvalue1):
                return None, 0

        if add_pv_types == (PaddingValueType.TENSOR, PaddingValueType.ANY):
            return None, 0

        if add_pv_types == (PaddingValueType.ANY, PaddingValueType.EXACT):
            if not util.is_0_pvalue(pvalue1):
                return 0, None

        if add_pv_types == (PaddingValueType.ANY, PaddingValueType.TENSOR):
            return 0, None

        if add_pv_types == (PaddingValueType.ANY, PaddingValueType.ANY):
            return (0, None) if ne_ins else (0, 0)

        return None, None

    @classmethod
    def _do_calc(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        add_pv_types = (pvalue0.type, pvalue1.type)
        dtype = pvalue0.dtype
        if add_pv_types == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            return util.new_pvalue_x(smath.add_(pvalue0.value, pvalue1.value), dtype)

        if add_pv_types == (PaddingValueType.EXACT, PaddingValueType.TENSOR):
            return util.new_pvalue_tensor(dtype)

        if add_pv_types == (PaddingValueType.EXACT, PaddingValueType.ANY):
            return util.new_pvalue_any(dtype)

        if add_pv_types == (PaddingValueType.TENSOR, PaddingValueType.EXACT):
            return util.new_pvalue_tensor(dtype)

        if add_pv_types == (PaddingValueType.TENSOR, PaddingValueType.TENSOR):
            return util.new_pvalue_tensor(dtype)

        if add_pv_types == (PaddingValueType.ANY, PaddingValueType.EXACT):
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{add_pv_types}] in AddSimulator")
        return None


class SubSimulator(BinarySimulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_sub"

    @classmethod
    def _do_adjust(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> Tuple(int, Union[int, float])
        sub_pv_types = (pvalue0.type, pvalue1.type)
        ne_ins = id(pvalue0) != id(pvalue1)
        if sub_pv_types == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            if not util.check_valid(lambda: smath.sub_(pvalue0.value, pvalue1.value)):
                return (None, 0) if ne_ins else (0, 0)

        if sub_pv_types == (PaddingValueType.EXACT, PaddingValueType.TENSOR):
            if not util.is_0_pvalue(pvalue0):
                return 0, None

        if sub_pv_types == (PaddingValueType.EXACT, PaddingValueType.ANY):
            if not util.is_0_pvalue(pvalue0):
                return 0, None

        if sub_pv_types == (PaddingValueType.TENSOR, PaddingValueType.EXACT):
            if not util.is_0_pvalue(pvalue1):
                return None, 0

        if sub_pv_types == (PaddingValueType.TENSOR, PaddingValueType.ANY):
            return None, 0

        if sub_pv_types == (PaddingValueType.ANY, PaddingValueType.EXACT):
            if not util.is_0_pvalue(pvalue1):
                return 0, None

        if sub_pv_types == (PaddingValueType.ANY, PaddingValueType.TENSOR):
            return 0, None

        if sub_pv_types == (PaddingValueType.ANY, PaddingValueType.ANY):
            if id(pvalue0) != id(pvalue1):
                return None, 0

        return None, None

    @classmethod
    def _do_calc(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        sub_pv_types = (pvalue0.type, pvalue1.type)
        dtype = pvalue0.dtype
        if sub_pv_types == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            return util.new_pvalue_x(smath.sub_(pvalue0.value, pvalue1.value), dtype)

        if sub_pv_types == (PaddingValueType.EXACT, PaddingValueType.TENSOR):
            return util.new_pvalue_tensor(dtype)

        if sub_pv_types == (PaddingValueType.EXACT, PaddingValueType.ANY):
            return util.new_pvalue_any(dtype)

        if sub_pv_types == (PaddingValueType.TENSOR, PaddingValueType.EXACT):
            return util.new_pvalue_tensor(dtype)

        if sub_pv_types == (PaddingValueType.TENSOR, PaddingValueType.TENSOR):
            return util.new_pvalue_tensor(dtype)

        if sub_pv_types == (PaddingValueType.ANY, PaddingValueType.EXACT):
            return util.new_pvalue_any(dtype)

        if sub_pv_types == (PaddingValueType.ANY, PaddingValueType.ANY):
            return util.new_pvalue_0(dtype)

        util.raise_error(f"Unsupported padding value type[{sub_pv_types}] in SubSimulator")
        return None


class MulSimulator(BinarySimulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_mul"

    @classmethod
    def _do_adjust(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> Tuple(int, Union[int, float])
        mul_pv_types = (pvalue0.type, pvalue1.type)
        ne_ins = id(pvalue0) != id(pvalue1)
        if mul_pv_types == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            if not util.check_valid(lambda: smath.mul_(pvalue0.value, pvalue1.value)):
                return (0, None) if ne_ins else (0, 0)

        if mul_pv_types == (PaddingValueType.EXACT, PaddingValueType.TENSOR):
            if not util.is_0_pvalue(pvalue0) and not util.is_1_pvalue(pvalue0):
                return 0, None

        if mul_pv_types == (PaddingValueType.EXACT, PaddingValueType.ANY):
            if not util.is_0_pvalue(pvalue0) and not util.is_1_pvalue(pvalue0):
                return 0, None

        if mul_pv_types == (PaddingValueType.TENSOR, PaddingValueType.EXACT):
            if not util.is_0_pvalue(pvalue1) and not util.is_1_pvalue(pvalue1):
                return None, 0

        if mul_pv_types == (PaddingValueType.TENSOR, PaddingValueType.ANY):
            return None, 0

        if mul_pv_types == (PaddingValueType.ANY, PaddingValueType.EXACT):
            if not util.is_0_pvalue(pvalue1) and not util.is_1_pvalue(pvalue1):
                return 0, None

        if mul_pv_types == (PaddingValueType.ANY, PaddingValueType.TENSOR):
            return 0, None

        if mul_pv_types == (PaddingValueType.ANY, PaddingValueType.ANY):
            return (0, None) if ne_ins else (0, 0)

        return None, None

    @classmethod
    def _do_calc(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        mul_pv_types = (pvalue0.type, pvalue1.type)
        dtype = pvalue0.dtype
        if mul_pv_types == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            return util.new_pvalue_x(smath.mul_(pvalue0.value, pvalue1.value), dtype)

        if mul_pv_types == (PaddingValueType.EXACT, PaddingValueType.TENSOR):
            if util.is_0_pvalue(pvalue0):
                return util.new_pvalue_0(dtype)
            return util.new_pvalue_tensor(dtype)

        if mul_pv_types == (PaddingValueType.EXACT, PaddingValueType.ANY):
            if util.is_0_pvalue(pvalue0):
                return util.new_pvalue_0(dtype)
            return util.new_pvalue_any(dtype)

        if mul_pv_types == (PaddingValueType.TENSOR, PaddingValueType.EXACT):
            if util.is_0_pvalue(pvalue1):
                return util.new_pvalue_0(dtype)
            return util.new_pvalue_tensor(dtype)

        if mul_pv_types == (PaddingValueType.TENSOR, PaddingValueType.TENSOR):
            return util.new_pvalue_tensor(dtype)

        if mul_pv_types == (PaddingValueType.ANY, PaddingValueType.EXACT):
            if util.is_0_pvalue(pvalue1):
                return util.new_pvalue_0(dtype)
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{mul_pv_types}] in MulSimulator")
        return None


class DivSimulator(BinarySimulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_div"

    @classmethod
    def _do_adjust(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> Tuple(int, Union[int, float])
        div_pv_types = (pvalue0.type, pvalue1.type)
        ne_ins = id(pvalue0) != id(pvalue1)
        if div_pv_types == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            if not util.check_valid(lambda: smath.div_(pvalue0.value, pvalue1.value)):
                return (None, 1) if ne_ins else (1, 1)

        if div_pv_types == (PaddingValueType.EXACT, PaddingValueType.TENSOR):
            if not util.is_0_pvalue(pvalue0):
                return 0, None

        if div_pv_types == (PaddingValueType.EXACT, PaddingValueType.ANY):
            if not util.is_0_pvalue(pvalue0):
                return 0, None

        if div_pv_types == (PaddingValueType.TENSOR, PaddingValueType.EXACT):
            if util.is_0_pvalue(pvalue1):
                return None, 1

        if div_pv_types == (PaddingValueType.TENSOR, PaddingValueType.ANY):
            return None, 1

        if div_pv_types == (PaddingValueType.ANY, PaddingValueType.EXACT):
            if not util.is_1_pvalue(pvalue1):
                return None, 1

        if div_pv_types == (PaddingValueType.ANY, PaddingValueType.TENSOR):
            return 0, None

        if div_pv_types == (PaddingValueType.ANY, PaddingValueType.ANY):
            return (None, 1) if ne_ins else (1, 1)

        return None, None

    @classmethod
    def _do_calc(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        div_pv_types = (pvalue0.type, pvalue1.type)
        dtype = pvalue0.dtype
        if div_pv_types == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            return util.new_pvalue_x(smath.div_(pvalue0.value, pvalue1.value), dtype)

        if div_pv_types == (PaddingValueType.EXACT, PaddingValueType.TENSOR):
            if util.is_0_pvalue(pvalue0):
                return util.new_pvalue_0(dtype)
            return util.new_pvalue_tensor(dtype)

        if div_pv_types == (PaddingValueType.EXACT, PaddingValueType.ANY):
            if util.is_0_pvalue(pvalue0):
                return util.new_pvalue_0(dtype)
            return util.new_pvalue_any(dtype)

        if div_pv_types == (PaddingValueType.TENSOR, PaddingValueType.EXACT):
            return util.new_pvalue_tensor(dtype)

        if div_pv_types == (PaddingValueType.TENSOR, PaddingValueType.TENSOR):
            return util.new_pvalue_tensor(dtype)

        if div_pv_types == (PaddingValueType.ANY, PaddingValueType.EXACT):
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{div_pv_types}] in DivSimulator")
        return None


class MaxSimulator(BinarySimulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_max"

    @classmethod
    def _do_adjust(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> Tuple(int, Union[int, float])
        return None, None

    @classmethod
    def _do_calc(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        pv_type0, pv_type1 = pvalue0.type, pvalue1.type
        dtype = pvalue0.dtype
        if util.is_max_pvalue(pvalue0) or util.is_max_pvalue(pvalue1):
            return util.new_pvalue_max(dtype)

        if util.is_min_pvalue(pvalue0) and pv_type1 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type0 == PaddingValueType.TENSOR and util.is_min_pvalue(pvalue1):
            return util.new_pvalue_tensor(dtype)

        if (pv_type0, pv_type1) == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            return util.new_pvalue_x(smath.max_(pvalue0.value, pvalue1.value), dtype)

        if (pv_type0, pv_type1) == (PaddingValueType.TENSOR, PaddingValueType.TENSOR):
            return util.new_pvalue_tensor(dtype)

        return util.new_pvalue_any(dtype)


class MinSimulator(BinarySimulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_min"

    @classmethod
    def _do_adjust(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> Tuple(int, Union[int, float])
        return None, None

    @classmethod
    def _do_calc(cls, pvalue0, pvalue1):
        # type: (PaddingValue, PaddingValue) -> PaddingValue
        pv_type0, pv_type1 = pvalue0.type, pvalue1.type
        dtype = pvalue0.dtype

        if util.is_min_pvalue(pvalue0) or util.is_min_pvalue(pvalue1):
            return util.new_pvalue_min(dtype)

        if util.is_max_pvalue(pvalue0) and pv_type1 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type0 == PaddingValueType.TENSOR and util.is_max_pvalue(pvalue1):
            return util.new_pvalue_tensor(dtype)

        if (pv_type0, pv_type1) == (PaddingValueType.EXACT, PaddingValueType.EXACT):
            return util.new_pvalue_x(smath.min_(pvalue0.value, pvalue1.value), dtype)

        if (pv_type0, pv_type1) == (PaddingValueType.TENSOR, PaddingValueType.TENSOR):
            return util.new_pvalue_tensor(dtype)

        return util.new_pvalue_any(dtype)
