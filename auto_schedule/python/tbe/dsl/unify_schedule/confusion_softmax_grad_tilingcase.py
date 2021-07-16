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
elewise tiling case
"""
from enum import Enum  # pylint: disable=E0611
from enum import auto
from typing import Any

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import register_tiling_case
import tbe.common.platform as tbe_platform
from tbe.common.utils import shape_to_list

from .computation import Computation
from .constants import CompileInfo
from .constants import DTYPE_BYTE_MAPPING
from .constants import ElewisePattern
from .constants import Pattern

DEFAULT = "default"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
COMMON = "common"
CONST = "const"
EMPTY = "empty"
STATIC = "static"
ORIGINAL = "original"
DB_KEY = 10000
EMPTY_KEY = -2 ** 31


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    NONE_CUT = auto()
    ONE_CUT = auto()

@register_tiling_case("ConfusionSoftmaxGrad")
def calc(outs, option=None):
    len([option])
    mode = operation.get_context().get("mode")

    def calc_base_key():
        if mode == SPECIAL:
            pattern = operation.get_context().get_current_compute().get("pattern")
            _base_key = 00000000
            if REDUCE in pattern:
                _base_key = 10000000
        elif mode == SPECIAL_SCALAR:
            _base_key = 20000000
        else:
            _base_key = 0
        
        return _base_key

    _base_key = 10000000

    # if mode == SPECIAL:
    #     pattern = operation.get_context().get_current_compute().get("pattern")
    
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    shape = shape_to_list(out.shape)
    dim_len = len(shape)

    is_db = False
    if dim_len == 1:
        return _calc_one_dim(outs, base_key, is_db)
    return _calc_general(outs, _base_key)
    
def _calc_one_dim(outs, base_key, is_db=False):
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    dtype = out.dtype

    c_bounds = {
        1: (1, 32767),
        2: (1, 32767),
        4: (1, 16383),
        8: (1, 8191)
    }

    cases = [{"key": base_key,
              "block_tiling_axis": 0,
              "ub_tiling_axis": 0,
              "ub_factor_bound": c_bounds[DTYPE_BYTE_MAPPING[dtype]],
              "tiling_strategy": TilingStrategy.ONE_CUT,
              "is_need_db": False    
            }]
    
    if is_db:
        cases = [{"key": base_key,
              "block_tiling_axis": 0,
              "ub_tiling_axis": 0,
              "ub_factor_bound": c_bounds[DTYPE_BYTE_MAPPING[dtype]],
              "tiling_strategy": TilingStrategy.ONE_CUT,
              "is_need_db": True
            }]
    return cases

def _calc_general(outs, base_key):
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    cases = []
    out = outs[0]
    shape = shape_to_list(out.shape)
    dim_len = len(shape)

    cases.append({"key": base_key, "tiling_strategy": TilingStrategy.NONE_CUT})

    base = base_key + 1
    cases.append({
        "key": base+0*dim_len+1,
        "block_tiling_axis": 0,
        "ub_tiling_axis": 1,
        "tiling_strategy": TilingStrategy.ONE_CUT
    })

    return cases