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
Compare for PaddingValue
"""
from enum import Enum
from enum import auto
from typing import Callable
from typing import Tuple
from typing import Union

import tbe.dsl.padding.graph as m_graph
import tbe.dsl.padding.util as util
from tbe.dsl.padding.value import PaddingValue
from tbe.dsl.padding.value import PaddingValueType
from tbe.tvm.expr import ConstExpr
from tbe.tvm.expr import Expr
from tbe.tvm.tensor import Tensor


class CmpMode(Enum):
    GT = auto()
    GE = auto()
    LT = auto()
    LE = auto()
    EQ = auto()
    NE = auto()


def cmp(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    pv_types = (pvalue0.type, pvalue1.type)
    return _get_cmp_func(pv_types)(pvalue0, pvalue1, dtype, mode)


def _get_cmp_func(pv_types):
    # type: (Tuple[PaddingValue, PaddingValue]) -> Callable
    func_map = {
        (PaddingValueType.EXACT, PaddingValueType.EXACT): _cmp_exact_exact,
        (PaddingValueType.EXACT, PaddingValueType.TENSOR): _cmp_exact_tensor,
        (PaddingValueType.EXACT, PaddingValueType.ANY): _cmp_exact_any,
        (PaddingValueType.TENSOR, PaddingValueType.EXACT): _cmp_tensor_exact,
        (PaddingValueType.TENSOR, PaddingValueType.TENSOR): _cmp_tensor_tensor,
        (PaddingValueType.TENSOR, PaddingValueType.ANY): _cmp_tensor_any,
        (PaddingValueType.ANY, PaddingValueType.EXACT): _cmp_any_exact,
        (PaddingValueType.ANY, PaddingValueType.TENSOR): _cmp_any_tensor,
        (PaddingValueType.ANY, PaddingValueType.ANY): _cmp_any_any,
    }

    return func_map.get(pv_types)


def deal_selected_hs(sxls, target_tensor):
    # type: (Union[m_graph.Node, Expr], Tensor) -> PaddingValue
    dtype = target_tensor.dtype
    if isinstance(sxls, m_graph.Node):
        for sv in sxls.get_svalues():
            sv.add_target(target_tensor)
            return util.svalue_to_pvalue(sv)

        pvalue = sxls.get_pvalue()
        pvalue.add_target(target_tensor)
        return util.new_pvalue(pvalue)

    if isinstance(sxls, ConstExpr):
        return util.new_pvalue_x(util.tvm_const_to_np(sxls), dtype)

    return util.new_pvalue_tensor(dtype)


def _cmp_exact_exact(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    if pvalue0.value > pvalue1.value:
        if mode in (CmpMode.GT, CmpMode.GE, CmpMode.NE):
            return util.new_pvalue_1(dtype)

    if pvalue0.value < pvalue1.value:
        if mode in (CmpMode.LT, CmpMode.LE, CmpMode.NE):
            return util.new_pvalue_1(dtype)

    if pvalue0.value == pvalue1.value:
        if mode in (CmpMode.LE, CmpMode.GE, CmpMode.EQ):
            return util.new_pvalue_1(dtype)

    return util.new_pvalue_0(dtype)


def _cmp_exact_tensor(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    if util.is_max_pvalue(pvalue0):
        if mode == CmpMode.GE:
            return util.new_pvalue_1(dtype)
        if mode == CmpMode.LT:
            return util.new_pvalue_0(dtype)
    if util.is_min_pvalue(pvalue0):
        if mode == CmpMode.LE:
            return util.new_pvalue_1(dtype)
        if mode == CmpMode.GT:
            return util.new_pvalue_0(dtype)
    return util.new_pvalue_any(dtype)


def _cmp_exact_any(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    return _cmp_exact_tensor(pvalue0, pvalue1, dtype, mode)


def _cmp_tensor_exact(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    if util.is_max_pvalue(pvalue1):
        if mode == CmpMode.LE:
            return util.new_pvalue_1(dtype)
        if mode == CmpMode.GT:
            return util.new_pvalue_0(dtype)
    if util.is_min_pvalue(pvalue1):
        if mode == CmpMode.GE:
            return util.new_pvalue_1(dtype)
        if mode == CmpMode.LT:
            return util.new_pvalue_0(dtype)
    return util.new_pvalue_any(dtype)


def _cmp_tensor_tensor(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    return util.new_pvalue_tensor(dtype)


def _cmp_tensor_any(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    return util.new_pvalue_any(dtype)


def _cmp_any_exact(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    return _cmp_tensor_exact(pvalue0, pvalue1, dtype, mode)


def _cmp_any_tensor(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    return util.new_pvalue_any(dtype)


def _cmp_any_any(pvalue0, pvalue1, dtype, mode):
    # type: (PaddingValue, PaddingValue, str, CmpMode) -> PaddingValue
    if id(pvalue0) == id(pvalue1):
        if mode in (CmpMode.LE, CmpMode.GE, CmpMode.EQ):
            return util.new_pvalue_1(dtype)
        else:
            return util.new_pvalue_0(dtype)
    return util.new_pvalue_any(dtype)
