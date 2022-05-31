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
Util for padding
"""
import warnings
from typing import Callable, List, Union

import numpy as np
import tbe.dsl.base.padding.graph as m_graph
import tbe.dsl.base.padding.value as m_value
from tbe import tvm
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import d_format_util
from tbe.dsl.base.expr_compare import expr_equal
from tbe.dsl.base.padding.value import (PaddingValue, PaddingValueType,
                                        SettingValue, SettingValueType)
from tbe.tvm.expr import ConstExpr, Expr
from tbe.tvm.tensor import PlaceholderOp, Tensor

D_BOUNDS = {
    "int8": (-128, 127),
    "uint8": (0, 255),
    "int16": (-32768, 32767),
    "uint16": (0, 65535),
    "int32": (-2147483648, 2147483647),
    "uint32": (0, 4294967295),
    "int64": (-9223372036854775808, 9223372036854775807),
    "uint64": (0, 18446744073709551615),
    "float16": (-65504.0, 65504.0),
    "float32": (-3.4028234663852886e+38, 3.4028234663852886e+38),
}

D_CLS = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
}


PyFiT = Union[int, float]
NpFiT = Union[np.integer, np.floating]


def is_placeholder(tensor):
    # type: (Tensor) -> bool
    return isinstance(tensor.op, PlaceholderOp)


def get_insn(node):
    # type: (m_graph.Node) -> str
    tag = node.get_tensor().op.tag
    if tag.find("|") != -1:
        insn = tag.split("|")[0]
    else:
        insn = tag
    return insn


def np_num_to_tvm(x):
    # type: (NpFiT) -> ConstExpr
    return tvm.const(x.item(), str(x.dtype))


def tvm_const_to_np(x):
    # type: (ConstExpr) -> NpFiT
    return D_CLS.get(x.dtype)(x.value)


def get_normal_condition(node):
    # type: (m_graph.Node) -> Callable
    var_c = None
    idx_c1, idx_c0 = None, None
    shape = node.get_tensor().shape
    c0 = shape[-1].value
    for i, var_x in enumerate(shape):
        axis_type = d_format_util.get_axis_type(var_x)
        if d_format_util.eq_axis_type(axis_type, "C1"):
            idx_c1 = i
            var_c = d_format_util.get_original(var_x)
        if d_format_util.eq_axis_type(axis_type, "C0"):
            idx_c0 = i

    condition = lambda *i: tvm.all(i[idx_c1] >= var_c // c0, i[idx_c0] >= var_c % c0)
    return condition


def get_brc_condition_value(node):
    # type: (m_graph.Node) -> Callable
    var_c = None
    idx_c0 = None
    tensor = node.get_tensor()
    shape = tensor.shape
    for i, var_x in enumerate(shape):
        axis_type = d_format_util.get_axis_type(var_x)
        if d_format_util.eq_axis_type(axis_type, "C1"):
            var_c = d_format_util.get_original(var_x)
        if d_format_util.eq_axis_type(axis_type, "C0"):
            idx_c0 = i

    def get_value_idx(v_idx):
        _idx = []
        for i, x in enumerate(v_idx):
            if i == idx_c0:
                _idx.append(0)
                continue
            _idx.append(x)
        return tuple(_idx)

    condition = lambda *i: tvm.all(var_c == 1, i[idx_c0] > 0)
    # t means tensor, for the tensor do cache_read scene
    value = lambda t: lambda *i: t[get_value_idx(i)]
    return condition, value


def is_d_format(node):
    # type: (m_graph.Node) -> bool
    shape = node.get_tensor().shape
    return d_format_util.is_5hd_format(shape, ignore_none=True)


def exist_pad(node):
    # type: (m_graph.Node) -> bool
    shape = node.get_tensor().shape
    dim_c1, dim_c0, ori_c = None, None, None
    for dim in shape:
        axis_type = d_format_util.get_axis_type(dim)
        if d_format_util.eq_axis_type(axis_type, "C1"):
            dim_c1 = dim
            ori_c = d_format_util.get_original(dim_c1)
        elif d_format_util.eq_axis_type(axis_type, "C0"):
            dim_c0 = dim
            ori_c = d_format_util.get_original(dim_c0)

    if dim_c1 is None and dim_c0 is None:
        return False

    if expr_equal(dim_c1 * dim_c0, ori_c):
        return False

    return True


def check_valid(call):
    # type: (Callable) -> bool
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            call()
        except Warning as w:
            ws = str(w)
            invalid_msgs = [
                "overflow encountered",
                "divide by zero encountered",
                "invalid value encountered"
            ]
            for msg in invalid_msgs:
                if msg in ws:
                    return False
        return True


def raise_error(message, error_code="E90001"):
    # type: (str, str) -> None
    dict_args = {}
    dict_args["errCode"] = error_code
    dict_args["detailed_cause"] = message
    raise RuntimeError(dict_args, get_error_message(dict_args))


def get_hs_b(node):
    # type: (m_graph.Node) -> Expr
    return node.get_tensor().op.body[0].b


def get_target_dtype(node):
    # type: (m_graph.Node) -> str
    return node.get_tensor().op.body[0].dtype


def equal_0(a):
    # type: (Union[NpFiT, ConstExpr]) -> bool
    return _get_value(a) == 0


def equal_1(a):
    # type: (Union[NpFiT, ConstExpr]) -> bool
    return _get_value(a) == 1


def equal_max(a):
    # type: (Union[NpFiT, ConstExpr]) -> bool
    return _get_value(a) == D_BOUNDS.get(get_dtype(a))[1]


def equal_min(a):
    # type: (Union[NpFiT, ConstExpr]) -> bool
    return _get_value(a) == D_BOUNDS.get(get_dtype(a))[0]


def _get_value(a):
    # type: (Union[NpFiT, ConstExpr]) -> PyFiT
    if isinstance(a, (np.floating, np.integer)):
        return a.item()
    if isinstance(a, ConstExpr):
        return a.value
    return None


def get_dtype(a):
    # type: (Union[NpFiT, ConstExpr]) -> PyFiT
    if isinstance(a, (np.floating, np.integer)):
        return str(a.dtype)
    if isinstance(a, ConstExpr):
        return a.dtype
    return None


def is_max_pvalue(pvalue):
    # type: (m_value.PaddingValue) -> bool
    if pvalue.type == m_value.PaddingValueType.EXACT:
        return pvalue.value.item() == D_BOUNDS.get(str(pvalue.value.dtype))[1]
    return False


def is_min_pvalue(pvalue):
    # type: (m_value.PaddingValue) -> bool
    if pvalue.type == m_value.PaddingValueType.EXACT:
        return pvalue.value.item() == D_BOUNDS.get(str(pvalue.value.dtype))[0]
    return False


def is_0_pvalue(pvalue):
    # type: (m_value.PaddingValue) -> bool
    if pvalue.type == m_value.PaddingValueType.EXACT:
        return pvalue.value.item() == 0
    return False


def is_1_pvalue(pvalue):
    # type: (m_value.PaddingValue) -> bool
    if pvalue.type == m_value.PaddingValueType.EXACT:
        return pvalue.value.item() == 1
    return False


def is_tensor_pvalue(pvalue):
    # type: (m_value.PaddingValue) -> bool
    return pvalue is not None and pvalue.type == m_value.PaddingValueType.TENSOR


def new_np_num_max(dtype):
    # type: (str) -> NpFiT
    return new_np_num_x(D_BOUNDS.get(dtype)[1], dtype)


def new_np_num_min(dtype):
    # type: (str) -> NpFiT
    return new_np_num_x(D_BOUNDS.get(dtype)[0], dtype)


def new_np_num_0(dtype):
    # type: (str) -> NpFiT
    return new_np_num_x(0, dtype)


def new_np_num_1(dtype):
    # type: (str) -> NpFiT
    return new_np_num_x(1, dtype)


def new_np_num_x(x, dtype):
    # type: (PyFiT, str) -> NpFiT
    return D_CLS.get(dtype)(x)


def new_pvalue_max(dtype):
    # type: (str) -> m_value.PaddingValue
    return new_pvalue_x(new_np_num_max(dtype), dtype)


def new_pvalue_min(dtype):
    # type: (str) -> m_value.PaddingValue
    return new_pvalue_x(new_np_num_min(dtype), dtype)


def new_pvalue_0(dtype):
    # type: (str) -> m_value.PaddingValue
    return new_pvalue_x(new_np_num_0(dtype), dtype)


def new_pvalue_1(dtype):
    # type: (str) -> m_value.PaddingValue
    return new_pvalue_x(new_np_num_1(dtype), dtype)


def new_pvalue_x(x, dtype):
    # type: (Union[PyFiT, NpFiT], str) -> m_value.PaddingValue
    if isinstance(x, (int, float)):
        return m_value.PaddingValue(m_value.PaddingValueType.EXACT, dtype, value=new_np_num_x(x, dtype))
    if isinstance(x, (np.integer, np.floating)):
        return m_value.PaddingValue(m_value.PaddingValueType.EXACT, dtype, value=x)

    raise_error(f"Unsupported value type: {type(x)}.")
    return None


def new_pvalue_tensor(dtype):
    # type: (str) -> m_value.PaddingValue
    return m_value.PaddingValue(m_value.PaddingValueType.TENSOR, dtype)


def new_pvalue_any(dtype):
    # type: (str) -> m_value.PaddingValue
    return m_value.PaddingValue(m_value.PaddingValueType.ANY, dtype)


def new_pvalue(pvalue):
    # type: (m_value.PaddingValue) -> m_value.PaddingValue
    if pvalue is None:
        return None

    pv_type, dtype = pvalue.type, pvalue.dtype
    if pv_type == PaddingValueType.TENSOR:
        return new_pvalue_tensor(dtype)

    if pv_type == PaddingValueType.ANY:
        return new_pvalue_any(dtype)

    if pv_type == PaddingValueType.EXACT:
        return new_pvalue_x(pvalue.value, dtype)

    return None


def get_min(dtype):
    # type: (str) -> PyFiT
    return D_BOUNDS.get(dtype)[0]


def get_max(dtype):
    # type: (str) -> PyFiT
    return D_BOUNDS.get(dtype)[1]


def get_pvalue(node, target_tensor):
    # type: (m_graph.Node, Tensor) -> m_value.PaddingValue
    dtype = node.get_dtype()
    for svalue in node.get_svalues():
        if target_tensor not in svalue.target:
            continue
        _value = svalue.value
        if isinstance(_value, Callable):
            return new_pvalue_tensor(dtype)
        return new_pvalue_x(_value, dtype)

    return node.get_pvalue()


def is_brc_node(node):
    # type: (m_graph.Node) -> bool
    return get_insn(node) in ("broadcast", "unified_broadcast", "unknown_broadcast")


def eq_expr(a, b):
    # type: (Expr, Expr) -> bool
    def eq(a_, b_):
        if a_ is None and b_ is None:
            return True
        if a_ is None or b_ is None:
            return False
        return expr_equal(a_, b_)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        for a_x, b_x in zip(a, b):
            if not eq(a_x, b_x):
                return False
        return True

    return eq(a, b)


def svalue_to_pvalue(svalue):
    # type: (SettingValue) -> PaddingValue
    if svalue.type == SettingValueType.NORMAL:
        return new_pvalue_x(svalue.value, svalue.dtype)

    if svalue.type == SettingValueType.BROADCAST:
        return new_pvalue_tensor(svalue.dtype)

    return None


def match_instance(vars_, types):
    for v, t in zip(vars_, types):
        if not isinstance(v, t):
            return False
    return True
