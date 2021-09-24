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
gemm tiling case
"""
import warnings


def calc_matmul(outs, option=None):
    """
    tiling_case func for dynamic shape matmul

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    warnings.warn("te.lang.dynamic.schedule.gemm_tilingcase is expired, "
        "please replace it with the func tbe.dsl.unify_schedule.gemm_tilingcase",
        DeprecationWarning)
    from tbe.dsl.unify_schedule.gemm_tilingcase import calc_matmul
    if option:
        return calc_matmul(outs, option)
    else:
        return calc_matmul(outs)
