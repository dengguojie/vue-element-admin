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
build
"""
from __future__ import absolute_import as _abs

from tbe.dsl.static_schedule.cce_schedule import cce_build_code as static_build
from tbe.dsl.base import operation

from .unify_auto_schedule import build as dynamic_build


def build(sch, config_map=None):
    """
    :param sch:
    :param config_map:
    :return:
    """
    context = operation.get_context()
    if context is not None and context.get_mode() in ("dynamic", "static"):
        return dynamic_build(sch, config_map)

    sch = sch[0] if isinstance(sch, list) else sch
    tensors = config_map.get("tensor_list", [])
    if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
        config_map["tensor_list"] = tensors[0]
    return static_build(sch, config_map)
