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

conv2d schedule
"""

import te
import te.lang.cce
from te import tvm
from te.platform.operation import register_schedule

from . import Pattern


@register_schedule(pattern=Pattern.CONV2D)
def schedule(outs, tiling_case):
    """
    schedule for conv2d dynamic shape
    """

    return Conv2dSchedule(outs, tiling_case).do_schedule()


class Conv2dSchedule:
    """
    Conv2dSchedule
    """

    def __init__(self, outs, tiling_case):
        self._outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

        self._schedule = None
        self._tiling_case = tiling_case

        self._scope = "local.UB"
        self._cce_conv_op = te.lang.cce.CceConvOp()

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

        self._cce_conv_op.schedule(self._outs[0], self._outs, [self._schedule],
            convbn1_flag=False, tiling_case=self._tiling_strategy, var_range=self._var_range)

        return self._schedule
