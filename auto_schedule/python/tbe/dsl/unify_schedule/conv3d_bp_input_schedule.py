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
conv3d_backprop_input schedule
"""
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.static_schedule.conv3d_backprop_input_schedule import CceConv3dBackpropInputOp
from tbe.dsl.static_schedule.cce_schedule import get_op_info
from tbe.tvm import schedule as tvm
from .constants import Pattern


@register_schedule(pattern=Pattern.CONV3D_BACKPROP_INPUT)
def schedule(outs, tiling_case):
    """
    schedule for conv3d backprop input dynamic shape
    """
    outs_list = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    op_info = get_op_info(outs_list)

    sch = tvm.create_schedule(
        [res.op for res in outs_list if res not in op_info['tensor_map']])
    sch.tiling_key = tiling_case['key']
    dynamic_para = {
        "var_range": tiling_case['var_range'],
        "tiling": tiling_case['tiling_strategy']
    }
    scope = 'local.UB'
    cce_op = CceConv3dBackpropInputOp(scope, need_tensorize=True, need_pragma=True)
    cce_op.schedule(outs_list[0], outs_list, [sch], dynamic_para=dynamic_para)

    return sch
