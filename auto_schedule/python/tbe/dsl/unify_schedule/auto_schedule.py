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
auto_schedule
"""
from __future__ import absolute_import as _abs

import tbe.dsl.static_schedule as static
import tbe.dsl.unify_schedule as dynamic
from tbe import tvm
from tbe.dsl.base import operation
from topi import generic


@tvm.target.generic_func
def auto_schedule(outs, option=None):
    """Entry of auto-Schedule.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.
    option:
    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    pass


@auto_schedule.register("cce")
@generic.auto_schedule.register("cce")
def _schedule_cce(outs, option=None):
    context = operation.get_context()
    if context is not None and context.get_mode() in ("dynamic", "static"):
        return dynamic.schedule_cce(outs, option)

    return static.schedule_cce(outs, option)
