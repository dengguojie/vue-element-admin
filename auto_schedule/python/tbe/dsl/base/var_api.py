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

    return var_
