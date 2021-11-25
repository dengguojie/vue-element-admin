#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
layer_norm_x_backprop_v2 tiling case
"""

import copy
from enum import Enum, auto
from . import util

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_tiling_case
from .constants import Pattern

COMMON = "common"
BROADCAST = "broadcast"
SCALAR = "scalar"
REDUCE = "reduce"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
CONST = "const"
ORIGINAL = "original"
DB_KEY = 10000

class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    NONE_CUT = auto()


@register_tiling_case(pattern=Pattern.LAYER_NORM_X_BACKPROP_V2)
def calc_layer_norm_x_backprop_v2(outs, option=None):
    """
    tiling_case func for layer_norm_x_backprop_v2 dynamic shape

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    shape = util.shape_to_list(out.shape)
    cases = []

    base_key = 10000
    cases.append({
        "key": base_key,
        "block_tiling_axis": 0,
        "ub_tiling_axis": 1,
        "tiling_strategy": TilingStrategy.NONE_CUT})

    return cases
