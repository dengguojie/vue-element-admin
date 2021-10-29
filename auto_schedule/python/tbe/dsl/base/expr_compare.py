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
compare expr
"""
import sympy
from tbe.common.utils.errormgr import get_error_message
from tbe.tvm import expr


def is_true(expr, dict_args):
    if expr:
        raise RuntimeError(dict_args, get_error_message(dict_args))


def _te_expr2sympy_expr(te_expr):
    if isinstance(te_expr, (int, float)):
        return te_expr
    if isinstance(te_expr, expr.ConstExpr):
        return te_expr.value
    if isinstance(te_expr, expr.Var):
        return sympy.symbols(te_expr.name)
    if isinstance(te_expr, expr.Max):
        return sympy.Max(_te_expr2sympy_expr(te_expr.a), _te_expr2sympy_expr(te_expr.b))
    if isinstance(te_expr, expr.Mul):
        return sympy.Mul(_te_expr2sympy_expr(te_expr.a), _te_expr2sympy_expr(te_expr.b))
    dict_args = {}
    dict_args["errCode"] = "E90001"
    dict_args["detailed_cause"] = "Only accecpt (int, float, ConstExpr, Var, Mul, Max), " \
                                  "but now is [%s]" % type(te_expr)
    raise RuntimeError(dict_args, get_error_message(dict_args))


def expr_equal(expr_a, expr_b, condition=None):
    is_true(condition is not None,
            {"errCode": "E90001",
            "detailed_cause": "Now, not support condition"})
    sympy_a = _te_expr2sympy_expr(expr_a)
    sympy_b = _te_expr2sympy_expr(expr_b)
    return sympy_a == sympy_b
