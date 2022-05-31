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
var api
"""
from tbe import tvm


ORIGINAL = "original"


def _set_var_attr(target_var, src_vars=None, original_op=None, annotation=None):
    def merge(_vars):
        _annotation, _originals = {}, []
        exist_original = False
        for x_var in _vars:
            if not isinstance(x_var, tvm.expr.Expr):
                _originals.append(x_var)
                continue
            _originals.append(x_var)
            for k, v in get_annotation(x_var).items():
                if k == ORIGINAL:
                    _originals[-1] = v
                    exist_original = True
                    continue
                if k not in _annotation or v not in _annotation.get(k):
                    _annotation.setdefault(k, []).append(v)

        for k in list(_annotation.keys()):
            v = _annotation.get(k)
            if len(v) == 1:
                _annotation[k] = v[0]
            else:
                _annotation.pop(k)

        if exist_original is False:
            _originals = []

        return _annotation, _originals

    src_vars = [] if src_vars is None else src_vars
    annotation = {} if annotation is None else annotation
    merged_annotation, originals = merge(src_vars)
    merged_annotation.update(annotation)

    if ORIGINAL not in merged_annotation and originals and original_op:
        original_expr = original_op(*originals)
        ori_merged_annotation, _ = merge(originals)
        set_annotation(original_expr, ori_merged_annotation)
        merged_annotation[ORIGINAL] = original_expr

    set_annotation(target_var, merged_annotation)


def get_annotation(var_):
    return tvm.get_expr_annotation(var_)


def set_annotation(var_, annotation):
    for k, v in annotation.items():
        set_attr(var_, k, v)


def get_attr_keys(var_):
    return tvm.expr_attr_keys(var_)


def get_attr(var_, key):
    return tvm.get_expr_attr(var_, key)


def set_attr(var_, key, value):
    tvm.set_expr_attr(var_, key, value)


def const(value, dtype=None, annotation=None):
    """construct a constant

    Parameters
    ----------
    value : number
        The content of the constant number.

    dtype : str or None, optional
        The data type.

    annotation: dict
        annotation info

    Returns
    -------
    const_val: tvm.Expr
        The result expression.
    """
    var_ = tvm.const(value, dtype)
    _set_var_attr(var_, annotation=annotation)

    return var_


def var(name="tindex", dtype=None, annotation=None):
    """Create a new variable with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : int
        The data type

    annotation: dict
        annotation info

    Returns
    -------
    var : Var
        The result symbolic variable.
    """
    var_ = tvm.var(name, dtype)
    _set_var_attr(var_, annotation=annotation)

    return var_


def div(a, b, annotation=None):
    """Compute a / b as in C/C++ semantics.

    Parameters
    ----------
    a : Expr
        The left hand operand, known to be non-negative.

    b : Expr
        The right hand operand, known to be non-negative.

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.
    Note
    ----
    When operands are integers, returns truncdiv(a, b).
    """
    var_ = tvm.div(a, b)
    _set_var_attr(var_, src_vars=[a, b], original_op=tvm.div, annotation=annotation)

    return var_


def indexdiv(a, b, annotation=None):
    """Compute floor(a / b) where a and b are non-negative.

    Parameters
    ----------
    a : Expr
        The left hand operand, known to be non-negative.

    b : Expr
        The right hand operand, known to be non-negative.

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    var_ = tvm.indexdiv(a, b)
    _set_var_attr(var_, src_vars=[a, b], original_op=tvm.indexdiv, annotation=annotation)

    return var_


def indexmod(a, b, annotation=None):
    """Compute the remainder of indexdiv. a and b are non-negative.

    Parameters
    ----------
    a : Expr
        The left hand operand, known to be non-negative.

    b : Expr
        The right hand operand, known to be non-negative.

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    var_ = tvm.indexmod(a, b)
    _set_var_attr(var_, src_vars=[a, b], original_op=tvm.indexmod, annotation=annotation)

    return var_


def truncdiv(a, b, annotation=None):
    """Compute the truncdiv of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    var_ = tvm.truncdiv(a, b)
    _set_var_attr(var_, src_vars=[a, b], original_op=tvm.truncdiv, annotation=annotation)

    return var_


def truncmod(a, b, annotation=None):
    """Compute the truncmod of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    var_ = tvm.truncmod(a, b)
    _set_var_attr(var_, src_vars=[a, b], original_op=tvm.truncmod, annotation=annotation)

    return var_


def floordiv(a, b, annotation=None):
    """Compute the floordiv of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.
    """
    var_ = tvm.floordiv(a, b)
    _set_var_attr(var_, src_vars=[a, b], original_op=tvm.floordiv, annotation=annotation)

    return var_


def floormod(a, b, annotation=None):
    """Compute the floormod of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.
    """
    var_ = tvm.floormod(a, b)
    _set_var_attr(var_, src_vars=[a, b], original_op=tvm.floormod, annotation=annotation)

    return var_


def sum(*args, annotation=None):
    """Compute the sum of expressions.

    Parameters
    ----------
    args : Expr
        The operands

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.
    """
    var_ = tvm.sum(*args)
    _set_var_attr(var_, src_vars=args, original_op=tvm.sum, annotation=annotation)

    return var_


def min(*args, annotation=None):
    """Compute the min of expressions.

    Parameters
    ----------
    args : Expr
        The operands

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.
    """
    var_ = tvm.min(*args)
    _set_var_attr(var_, src_vars=args, original_op=tvm.min, annotation=annotation)

    return var_


def max(*args, annotation=None):
    """Compute the max of expressions.

    Parameters
    ----------
    args : Expr
        The operands

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.
    """
    var_ = tvm.max(*args)
    _set_var_attr(var_, src_vars=args, original_op=tvm.max, annotation=annotation)

    return var_


def prod(*args, annotation=None):
    """Compute the prod of expressions.

    Parameters
    ----------
    args : Expr
        The operands

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.
    """
    var_ = tvm.prod(*args)
    _set_var_attr(var_, src_vars=args, original_op=tvm.prod, annotation=annotation)

    return var_


def bit(*args, annotation=None):
    """Compute the bit of expressions.

    Parameters
    ----------
    args : Expr
        The operands

    annotation: dict
        annotation info

    Returns
    -------
    res : Expr
        The result expression.
    """
    var_ = tvm.bit(*args)
    _set_var_attr(var_, src_vars=args, original_op=tvm.bit, annotation=annotation)

    return var_
