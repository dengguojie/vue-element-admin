#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

elewise tiling case
"""
from enum import Enum, auto  # pylint: disable=E0611

from . import DTYPE_BYTE_MAPPING
from . import Pattern
from . import util
from te.platform.operation import register_tiling_case


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    ALL_CUT = auto()
    NONE_CUT = auto()
    ONE_CUT = auto()
    STATIC = auto()


# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.ELEMWISE)
def calc(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    # avoid pylint
    len([option])
    # ###############TILING KEY RULE################
    # use int32, max value 2147483647
    # 0~1: dim len

    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    shape = util.shape_to_list(out.shape)
    dim_len = len(shape)

    if dim_len == 1:
        return _calc_one_dim(outs)

    return _calc_general(outs)


def _calc_one_dim(outs):
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    dtype = out.dtype

    cases = []

    # schedule in var bound
    seed = 100000000
    c_bounds = {
        1: [(1, 32767)],
        2: [(1, 32767)],
        4: [(1, 16383)],
        8: [(1, 8191)],
    }

    for i, bound_i in enumerate(c_bounds[DTYPE_BYTE_MAPPING[dtype]]):
        cases.append({"key": seed + i,
                      "block_tiling_axis": 0,
                      "ub_tiling_axis": 0,
                      "ub_factor_bound": bound_i,
                      "tiling_strategy": TilingStrategy.ONE_CUT,
                      })
    return cases


def _calc_general(outs):
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    cases = []
    out = outs[0]
    shape = util.shape_to_list(out.shape)
    dim_len = len(shape)

    seed = 0
    # methodology 1:
    # general, no split: no block tiling, no ub tiling, no db
    cases.append({"key": seed + 2, "tiling_strategy": TilingStrategy.NONE_CUT})

    # methodology 2:
    # split special axis for block tiling and ub tiling
    # block tiling: fused the axis(which before multi-core axis)
    # and multi-core axis
    # ub tiling axis >= block tiling axis
    seed = seed + 10000000
    for i in range(dim_len):
        for j in range(i, dim_len):
            cases.append({
                "key": seed + i * dim_len + j,
                "block_tiling_axis": i,
                "ub_tiling_axis": j,
                "tiling_strategy": TilingStrategy.ONE_CUT,
            })

    return cases
