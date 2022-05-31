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
tensor with scalar simulator
"""
import abc
from typing import Union

import tbe.dsl.base.padding.graph as m_graph
import tbe.dsl.base.padding.simulator as m_simulator
import tbe.dsl.base.padding.smath as smath
import tbe.dsl.base.padding.util as util
from tbe.dsl.base.padding.value import (PaddingValue, PaddingValueType,
                                        SettingValue, SettingValueType)
from tbe.tvm.expr import ConstExpr, Expr


class TensorScalarSimulator(m_simulator.Simulator, abc.ABC):
    def __init__(self, node):
        # type: (m_graph.Node) -> bool
        super().__init__(node)
        self._node0 = self._node.get_input_nodes()[0]
        self._scalar1 = util.get_hs_b(self._node)

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
        pvalue = self._do_calc(pvalue0, self._scalar1)
        self._node.set_pvalue(pvalue)

    def _adjust(self):
        if not util.exist_pad(self._node0):
            return None

        svalues = self._node0.get_svalues()
        stp = util.svalue_to_pvalue
        for sv in svalues:
            if self._do_adjust(stp(sv), self._scalar1) is None:
                return sv

        pvalue0 = self._node0.get_pvalue()
        return self._do_adjust(pvalue0, self._scalar1)

    @abc.abstractclassmethod
    def _do_adjust(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, Expr]) -> Union[int, float]
        """"""

    @abc.abstractclassmethod
    def _do_calc(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, Expr]) -> PaddingValue
        """"""


class AddsSimulator(TensorScalarSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_VS_add"

    @classmethod
    def _do_adjust(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, ConstExpr]) -> Union[int, float]
        adds_pv_type0 = pvalue0.type

        if adds_pv_type0 == PaddingValueType.EXACT:
            if isinstance(scalar1, ConstExpr):
                if not util.check_valid(lambda: smath.add_(pvalue0.value, util.tvm_const_to_np(scalar1))):
                    return 0
                return None

            if util.is_max_pvalue(pvalue0) or util.is_min_pvalue(pvalue0):
                return 0

        if adds_pv_type0 == PaddingValueType.ANY:
            if not util.equal_0(scalar1):
                return 0

        return None

    @classmethod
    def _do_calc(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, Expr]) -> PaddingValue
        adds_pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        if adds_pv_type0 == PaddingValueType.EXACT:
            if isinstance(scalar1, ConstExpr):
                new_value = smath.add_(pvalue0.value, util.tvm_const_to_np(scalar1))
                return util.new_pvalue_x(new_value, dtype)
            return util.new_pvalue_any(dtype)

        if adds_pv_type0 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if adds_pv_type0 == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{adds_pv_type0}] in AddsSimulator")
        return None


class MulsSimulator(TensorScalarSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_VS_mul"

    @classmethod
    def _do_adjust(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, ConstExpr]) -> Union[int, float]
        muls_pv_type0 = pvalue0.type

        if muls_pv_type0 == PaddingValueType.EXACT:
            if isinstance(scalar1, ConstExpr):
                if not util.check_valid(lambda: smath.mul_(pvalue0.value, util.tvm_const_to_np(scalar1))):
                    return 0
                return None

            if util.is_max_pvalue(pvalue0) or util.is_min_pvalue(pvalue0):
                return 0

        if muls_pv_type0 == PaddingValueType.ANY:
            if not util.equal_0(scalar1):
                return 0

        return None

    @classmethod
    def _do_calc(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, Expr]) -> PaddingValue
        muls_pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        if muls_pv_type0 == PaddingValueType.EXACT:
            if isinstance(scalar1, ConstExpr):
                new_value = smath.mul_(pvalue0.value, util.tvm_const_to_np(scalar1))
                return util.new_pvalue_x(new_value, dtype)
            if util.is_0_pvalue(pvalue0):
                return util.new_pvalue_0(dtype)
            return util.new_pvalue_any(dtype)

        if muls_pv_type0 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if muls_pv_type0 == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{muls_pv_type0}] in MulsSimulator")
        return None


class MaxsSimulator(TensorScalarSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_VS_max"

    @classmethod
    def _do_adjust(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, ConstExpr]) -> Union[int, float]
        return None

    @classmethod
    def _do_calc(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, Expr]) -> PaddingValue
        maxs_pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        if maxs_pv_type0 == PaddingValueType.EXACT:
            if isinstance(scalar1, ConstExpr):
                new_value = smath.max_(pvalue0.value, util.tvm_const_to_np(scalar1))
                return util.new_pvalue_x(new_value, dtype)
            if util.is_max_pvalue(pvalue0):
                return util.new_pvalue_max(dtype)
            return util.new_pvalue_any(dtype)

        if maxs_pv_type0 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if maxs_pv_type0 == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{maxs_pv_type0}] in MaxsSimulator")
        return None


class MinsSimulator(TensorScalarSimulator):
    @classmethod
    def get_type(cls):
        return "elewise_single_VS_min"

    @classmethod
    def _do_adjust(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, ConstExpr]) -> Union[int, float]
        return None

    @classmethod
    def _do_calc(cls, pvalue0, scalar1):
        # type: (PaddingValue, Union[int, float, Expr]) -> PaddingValue
        mins_pv_type0 = pvalue0.type
        dtype = pvalue0.dtype
        if mins_pv_type0 == PaddingValueType.EXACT:
            if isinstance(scalar1, ConstExpr):
                new_value = smath.min_(pvalue0.value, util.tvm_const_to_np(scalar1))
                return util.new_pvalue_x(new_value, dtype)
            if util.is_min_pvalue(pvalue0):
                return util.new_pvalue_min(dtype)
            return util.new_pvalue_any(dtype)

        if mins_pv_type0 == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if mins_pv_type0 == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{mins_pv_type0}] in MinsSimulator")
        return None
