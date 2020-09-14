#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

conv2d schedule
"""

import te
import te.lang.cce
from te import tvm
from te.platform.operation import register_schedule

from . import Pattern


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
        self._cce_op = te.lang.cce.CceConv2dBackpropInputOp(self._scope,
            need_tensorize=True, need_pragma=True)

    def do_schedule(self):
        """
        do schedule
        """

        op_info = te.lang.cce.get_op_info(self._outs)
        self._var_range = self._tiling_case['var_range']

        self._schedule = tvm.create_schedule(
            [res.op for res in self._outs if res not in op_info['tensor_map']])
        self._schedule.tiling_key = self._tiling_case['key']
        self._tiling_strategy = self._tiling_case['tiling_strategy']

        self._cce_op.schedule(self._outs[0], self._outs, [self._schedule],
            tiling_case=self._tiling_strategy, var_range=self._var_range)

        return self._schedule
