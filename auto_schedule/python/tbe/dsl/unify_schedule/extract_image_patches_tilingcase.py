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
extract_image_patches tiling
"""
from enum import Enum, auto
from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_tiling_case
from .constants import Pattern


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    AXIS_ALIGN = auto()
    AXIS_NOT_ALIGN = auto()


@register_tiling_case(pattern=Pattern.EXTRACT_IMAGE_PATCHES)
def calc_extract_image_patches(outs, option=None):
    """
    tiling_case func for extract_image_patches dynamic shape

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    cases = []
    axis_align_key = 10000
    axis_not_align_key = 20000
    block_size = 16

    origin_c_in = operation.get_context().get("origin_c_in")
    if origin_c_in % block_size == 0:
        cases.append({
            "key": axis_align_key,
            "block_tiling_axis": 0,
            "ub_tiling_axis": 1,
            "tiling_strategy": TilingStrategy.AXIS_ALIGN
        })
    else:
        cases.append({
            "key": axis_not_align_key,
            "block_tiling_axis": 0,
            "ub_tiling_axis": 1,
            "tiling_strategy": TilingStrategy.AXIS_NOT_ALIGN
        })

    return cases
