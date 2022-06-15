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
dynamic softmax_cross_entropy_with_logits tiling case
"""
from enum import Enum  # pylint: disable=E0611
from enum import auto

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import register_tiling_case

from . import util
from .constants import CompileInfo
from .constants import Pattern


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    ALL_CUT = auto()
    NONE_CUT = auto()
    ONE_CUT = auto()
    STATIC = auto()
    CONST = auto()


# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS)
def calc(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    # ###############TILING KEY RULE################
    # use int32, max value 2147483647
    # 0~1: dim len
    mode = operation.get_context().get("mode")

    def calc_base_key():
        # first 1 is used to fill int32 length,
        # second 1 means the case is reduce mode.
        if "copy" in mode:
            _base_key = 10
        elif "vec1" in mode:
            _base_key = 1
        elif "vec4" in mode:
            _base_key = 4
        elif "vec2" in mode:
            _base_key = 2
        elif "vec8" in mode:
            _base_key = 8
        elif "vec9" in mode:
            _base_key = 9
        elif "vec6" in mode:
            _base_key = 6
        else:
            _base_key = 0
        if "cut" in mode:
            _base_key += 100

        return _base_key

    base_key = calc_base_key()

    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

    return _calc_general(outs, base_key, mode)


def _calc_general(outs, base_key, mode):
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    if len(outs) > 1:
        out = outs[1]
    else:
        out = outs[0]
    shape = util.shape_to_list(out.shape)
    dim_len = len(shape)

    cases = []
    # methodology 1:
    # general, no split: no block tiling, no ub tiling, no db
    # cases.append({"key": base_key, "tiling_strategy": TilingStrategy.NONE_CUT})

    # methodology 2:
    # split special axis for block tiling and ub tiling
    # block tiling: fused the axis(which before multi-core axis)
    # and multi-core axis
    # ub tiling axis >= block tiling axis
    base = base_key
    # is_align: case last dim is aligned each block size 
    align_base_key = 10000
    for i in range(dim_len):
        for j in range(i, dim_len):
            if i < 1 and j < 1:
                if not "cut" in mode:
                    cases.append({
                        "key": base + i * 10 + j,
                        "block_tiling_axis": i,
                        "ub_tiling_axis": j,
                        "is_align": True,
                        "tiling_strategy": TilingStrategy.ONE_CUT,
                    })
                    cases.append({
                        "key": base + i * 10 + j + align_base_key,
                        "block_tiling_axis": i,
                        "ub_tiling_axis": j,
                        "is_align": False,
                        "tiling_strategy": TilingStrategy.ONE_CUT,
                    })
                else:
                    cases.append({
                        "key": base + i * 10 + j,
                        "block_tiling_axis": i,
                        "ub_tiling_axis": j,
                        "is_align": False,
                        "tiling_strategy": TilingStrategy.ONE_CUT,
                    })

    return cases
