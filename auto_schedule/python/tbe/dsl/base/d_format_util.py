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
D-Format util
"""
from typing import Union

from tbe.dsl.base import var_api
from tbe.tvm.expr import Expr

_AXIS_TYPE = "axis_type"
_ORIGINAL = "original"


def get_format(shape, ignore_none=True):
    # type: (Union[list, tuple], bool) -> list
    shape_format = []
    for axis in shape:
        axis_type = get_axis_type(axis)
        if isinstance(axis_type, (list, tuple)):
            shape_format.extend(axis_type)
        else:
            shape_format.append(axis_type)

    if ignore_none:
        return [x for x in shape_format if x is not None]

    return shape_format


def is_5hd_format(shape, ignore_none=True):
    # type: (Union[list, tuple], bool) -> bool
    shape_format = get_format(shape, ignore_none=ignore_none)
    return shape_format == ["N", "C1", "H", "W", "C0"]


def set_axis_type(var_, axis_type):
    # type: (Expr, str) -> None
    var_api.set_attr(var_, _AXIS_TYPE, axis_type)


def get_axis_type(var_):
    # type: (Expr) -> str
    return var_api.get_attr(var_, _AXIS_TYPE)


def set_original(var_, original):
    # type: (Expr, Expr) -> None
    var_api.set_attr(var_, _ORIGINAL, original)


def get_original(var_):
    # type: (Expr) -> Expr
    return var_api.get_attr(var_, _ORIGINAL)


def eq_axis_type(axis_type1, axis_type2):
    # type: (Union[str, list, tuple], Union[str, list, tuple]) -> bool
    if axis_type1 == axis_type2:
        return True

    if axis_type1 is None or axis_type2 is None:
        return False

    if isinstance(axis_type1, str):
        axis_type1 = [axis_type1]
    if isinstance(axis_type2, str):
        axis_type2 = [axis_type2]
    return list(axis_type1) == list(axis_type2)


def in_axis_type(axis, axis_types):
    # type: (Expr, Union[list, tuple]) -> bool
    axis_type = get_axis_type(axis)
    return any(eq_axis_type(axis_type, x) for x in axis_types)


def get_axis(shape, axis_type):
    # type: (Union[list, tuple], str) -> Expr
    is_target_axis = lambda x: eq_axis_type(get_axis_type(x), axis_type)
    return next((x for x in shape if is_target_axis(x)), None)


def get_c0(shape):
    # type: (Union[list, tuple]) -> Expr
    return get_axis(shape, "C0")


def get_c1(shape):
    # type: (Union[list, tuple]) -> Expr
    return get_axis(shape, "C1")


def get_c(shape):
    # type: (Union[list, tuple]) -> Expr
    is_padding_axis = lambda x: in_axis_type(x, ["C1", "C0"])
    return next((get_original(x) for x in shape if is_padding_axis(x)), None)
