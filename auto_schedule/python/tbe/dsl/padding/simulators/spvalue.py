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
Calc for PaddingValue
"""
from typing import Union

import tbe.dsl.padding.smath as smath
import tbe.dsl.padding.util as util
from tbe.dsl.padding.value import PaddingValue
from tbe.dsl.padding.value import PaddingValueType
from tbe.tvm.expr import ConstExpr
from tbe.tvm.expr import Expr


def relu(pvalue0):
    # type: (PaddingValue) -> PaddingValue
        pv_type, dtype = pvalue0.type, pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            return util.new_pvalue_x(smath.relu_(pvalue0.value), dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] for relu.")
        return None


def lrelu(pvalue0, scalar1):
    # type: (PaddingValue, Union[int, float, Expr]) -> PaddingValue
        pv_type, dtype = pvalue0.type, pvalue0.dtype
        if pv_type == PaddingValueType.EXACT:
            if util.equal_0(pvalue0.value):
                return util.new_pvalue_0(dtype)
            if isinstance(scalar1, ConstExpr):
                new_value = smath.lrelu_(pvalue0.value, util.tvm_const_to_np(scalar1))
                return util.new_pvalue_x(new_value, dtype)
            return util.new_pvalue_any(dtype)

        if pv_type == PaddingValueType.TENSOR:
            return util.new_pvalue_tensor(dtype)

        if pv_type == PaddingValueType.ANY:
            return util.new_pvalue_any(dtype)

        util.raise_error(f"Unsupported padding value type[{pv_type}] for leaky relu.")
        return None
