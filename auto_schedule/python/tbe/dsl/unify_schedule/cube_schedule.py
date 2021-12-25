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
cube schedule
"""

from tbe import tvm
from tbe.common.platform import intrinsic_check_support
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.static_schedule.conv_schedule import CceConvOp
from tbe.dsl.static_schedule.conv2d_backprop_filter_schedule import \
    CceConv2dBackpropFilterOp
from tbe.dsl.static_schedule.conv2d_backprop_input_schedule import \
    CceConv2dBackpropInputOp
from tbe.dsl.static_schedule.conv3d_backprop_filter_schedule import \
    CceConv3dBackpropFilterOp
from tbe.dsl.static_schedule.gemm_schedule import gemm_schedule as gemm_schedule1
from tbe.dsl.static_schedule.gemm_integrated_schedule import gemm_schedule as gemm_schedule2
from .constants import Pattern


@register_schedule(pattern=Pattern.CONV3D_BACKPROP_FILTER)
def schedule_conv3d_bp_filter(outs, tiling_case):
    """
    schedule for conv3d_backprop_filter op
    """

    return ConvSchedule(outs, tiling_case).do_conv3dbp_filter_schedule()


@register_schedule(pattern=Pattern.CONV2D_BACKPROP_FILTER)
def schedule_conv2d_bp_filter(outs, tiling_case):
    """
    schedule for conv2d_backprop_filter op
    """

    return ConvSchedule(outs, tiling_case).do_conv2dbp_filter_schedule()


@register_schedule(pattern=Pattern.MAT_MUL)
@register_schedule(pattern=Pattern.BATCH_MATMUL)
def schedule_matmul(outs, tiling_case):
    """
    schedule for matmul op
    """

    return ConvSchedule(outs, tiling_case).do_mat_mul_schedule()


class ConvSchedule:
    """
    conv-category op schedule
    """

    def __init__(self, outs, tiling_case):
        self._outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

        self._tiling_case = tiling_case
        self._scope = "local.UB"
        self._schedule_op = None
        self._get_schedule_info()

    def _get_schedule_info(self):
        """
        get scheule info
        """
        self._var_range = self._tiling_case['var_range']

        self._schedule = tvm.create_schedule([res.op for res in self._outs])
        self._schedule.tiling_key = self._tiling_case['key']
        self._tiling_strategy = self._tiling_case['tiling_strategy']
        self._m_k_n_shape = self._tiling_case.get('m_k_n_shape')

    def do_conv2d_schedule(self):
        """
        do schedule for conv2d
        """
        self._schedule_op = CceConvOp()
        self._schedule_op.schedule(
            self._outs[0], self._outs, [self._schedule], convbn1_flag=False,
            tiling_case=self._tiling_strategy, var_range=self._var_range)

        return self._schedule

    def do_conv2dbp_input_schedule(self):
        """
        do schedule for conv2dbp_input
        """
        self._schedule_op = CceConv2dBackpropInputOp(
            self._scope, need_tensorize=True, need_pragma=True)
        self._schedule_op.schedule(
            self._outs[0], self._outs, [self._schedule],
            tiling_case=self._tiling_strategy, var_range=self._var_range)

        return self._schedule

    def do_conv2dbp_filter_schedule(self):
        """
        do schedule for conv2dbp_filter
        """
        attach_flags = ("dynamic_l0a_attach", "dynamic_l0b_attach",
                        "dynamic_al1_attach", "dynamic_bl1_attach",
                        "bl1_hw_allin_flag")
        dynamic_para = {f: self._tiling_strategy.pop(f) for f in attach_flags}
        dynamic_para.update({
            "var_range": self._tiling_case['var_range'],
            "tiling": self._tiling_strategy
        })

        self._schedule_op = CceConv2dBackpropFilterOp(
            self._scope, need_tensorize=True, need_pragma=True)
        self._schedule_op.schedule(
            self._outs[0], self._outs, [self._schedule],
            dynamic_para=dynamic_para)

        return self._schedule

    def do_conv3dbp_filter_schedule(self):
        """
        do schedule for conv3dbp_filter
        """
        self._schedule_op = CceConv3dBackpropFilterOp(
            self._scope, need_tensorize=True, need_pragma=True)
        self._schedule_op.schedule(
            self._outs[0], self._outs, [self._schedule],
            dynamic_para=self._tiling_case)

        return self._schedule

    def do_mat_mul_schedule(self):
        """
        do schedule for gemm
        """
        gemm_schedule = gemm_schedule2
        if intrinsic_check_support("Intrinsic_fix_pipe_l0c2out"):
            gemm_schedule = gemm_schedule1

        gemm_schedule(self._outs[0], [self._schedule],
                      {"tiling_strategy": self._tiling_strategy,
                       "m_k_n_shape": self._m_k_n_shape,
                       "var_range": self._var_range})

        return self._schedule
