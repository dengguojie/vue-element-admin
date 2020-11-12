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
conv2d schedule
"""
from te.lang.dynamic.schedule.constants import Pattern
from te.lang.cce.te_schedule.conv2d_backprop_input_schedule import CceConv2dBackpropInputOp
from te.lang.cce.te_schedule.cce_schedule import get_op_info

from te.tvm import schedule as tvm
from te.lang.base.operation import register_schedule


@register_schedule(pattern=Pattern.CONV2D_BACKPROP_INPUT)
def schedule(outs, tiling_case):
    """
    schedule for conv2d backprop input dynamic shape
    """

    return Conv2dBackpropInputSchedule(outs, tiling_case).do_schedule()


class Conv2dBackpropInputSchedule:
    """
    Conv2dBackpropInputSchedule
    """

    def __init__(self, outs, tiling_case):
        self._outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

        self._schedule = None
        self._tiling_case = tiling_case

        self._scope = "local.UB"
        self._cce_op = CceConv2dBackpropInputOp(self._scope, need_tensorize=True, need_pragma=True)

    def do_schedule(self):
        """
        do schedule
        """

        op_info = get_op_info(self._outs)
        self._var_range = self._tiling_case['var_range']

        self._schedule = tvm.create_schedule(
            [res.op for res in self._outs if res not in op_info['tensor_map']])
        self._schedule.tiling_key = self._tiling_case['key']
        self._tiling_strategy = self._tiling_case['tiling_strategy']

        self._cce_op.schedule(self._outs[0], self._outs, [self._schedule],
                              tiling_case=self._tiling_strategy, var_range=self._var_range)

        return self._schedule
