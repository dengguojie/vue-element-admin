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

from te import tvm

from te.lang.cce.te_schedule.conv_schedule import CceConvOp
from te.lang.cce.te_schedule.conv2d_backprop_input_schedule import \
    CceConv2dBackpropInputOp
from te.lang.cce.te_schedule.conv2d_backprop_filter_schedule import \
    CceConv2dBackpropFilterOp

from te.lang.base.operation import register_schedule

from . import Pattern


@register_schedule(pattern=Pattern.CONV2D_BACKPROP_FILTER)
def schedule(outs, tiling_case):
    """
    schedule for conv2d_backprop_filter op
    """

    return ConvSchedule(outs, tiling_case).do_conv2dbp_filter_schedule()


class ConvSchedule:
    """
    conv-category op schedule
    """

    def __init__(self, outs, tiling_case):
        self._outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

        self._tiling_case = tiling_case
        self._scope = "local.UB"
        self._get_schedule_info()

    def _get_schedule_info(self):
        self._var_range = self._tiling_case['var_range']

        self._schedule = tvm.create_schedule([res.op for res in self._outs])
        self._schedule.tiling_key = self._tiling_case['key']
        self._tiling_strategy = self._tiling_case['tiling_strategy']

    def do_conv2d_schedule(self):
        self._schedule_op = CceConvOp()
        self._schedule_op.schedule(
            self._outs[0], self._outs, [self._schedule], convbn1_flag=False,
            tiling_case=self._tiling_strategy, var_range=self._var_range)

        return self._schedule

    def do_conv2dbp_input_schedule(self):
        self._schedule_op = CceConv2dBackpropInputOp(
            self._scope, need_tensorize=True, need_pragma=True)
        self._schedule_op.schedule(
            self._outs[0], self._outs, [self._schedule],
            tiling_case=self._tiling_strategy, var_range=self._var_range)

        return self._schedule

    def do_conv2dbp_filter_schedule(self):
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
