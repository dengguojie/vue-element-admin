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
softmax tiling case
"""
from enum import Enum, auto  # pylint: disable=E0611

from .constants import CompileInfo
from .constants import Pattern
from . import util
from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import register_tiling_case
from tbe.dsl.base.operation import add_compile_info
from te.platform.cce_conf import get_soc_spec

class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    NONE_CUT = auto()
    BEGIN_REDUCE = auto()
    END_REDUCE = auto()
    MID_REDUCE = auto()

def _get_block_size(dtype):
    if dtype in ["float32", "fp32", "int32"]:
        block_size = 8
    elif dtype in ["bool", "int8", "uint8"]:
        block_size = 32
    elif dtype in ["float16", "fp16"]:
        block_size = 16
    elif dtype in ["int64"]:
        block_size = 4
    else:
        raise RuntimeError("[%s] is not support type" % dtype)
    return block_size

# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.SOFTMAX)
def calc(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    # ###############TILING KEY RULE################
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    shape = util.shape_to_list(out.shape)
    dim_len = len(shape)
    cases = []
    if dim_len == 1:
        cases = [{"key": 200000000, "tiling_strategy": TilingStrategy.NONE_CUT}]
    elif dim_len == 2:
        if shape[0].name == 'b':
            cases = [{"key": 300000000,
                      "block_tiling_axis": 1,
                      "ub_tiling_axis": 1,
                      "tiling_strategy": TilingStrategy.BEGIN_REDUCE,
                      }]
        elif shape[1].name == 'b':
            cases = [{"key": 400000000,
                      "block_tiling_axis": 0,
                      "ub_tiling_axis": 0,
                      "tiling_strategy": TilingStrategy.END_REDUCE,
                      }]
    elif dim_len == 3:
        cases.append({
            "key": 500000010,
            "block_tiling_axis": 0,
            "ub_tiling_axis": 0,
            "tiling_strategy": TilingStrategy.MID_REDUCE,
        })
        cases.append({
            "key": 500000020,
            "block_tiling_axis": 0,
            "ub_tiling_axis": 2,
            "tiling_strategy": TilingStrategy.MID_REDUCE,
        })
        cases.append({
            "key": 500000030,
            "block_tiling_axis": 2,
            "ub_tiling_axis": 2,
            "tiling_strategy": TilingStrategy.MID_REDUCE,
        })

    max_ub_count = get_soc_spec("UB_SIZE") // 128 * 128
    core_num = get_soc_spec("CORE_NUM")
    keep_dims = 1
    reduce_block_size = _get_block_size(out.dtype)
    common_info = [max_ub_count, core_num, keep_dims, reduce_block_size]
    add_compile_info("common_info", common_info)

    return cases
