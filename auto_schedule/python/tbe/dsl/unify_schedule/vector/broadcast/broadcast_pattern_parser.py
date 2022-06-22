#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
broadcast pattern parser
"""
from typing import Dict
from typing import List
from typing import Union

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.pattern_manager import PatternParser
from tbe.tvm.tensor import Tensor


class BroadcastPatternParser(PatternParser):
    def __init__(self, outs, compute_type_size_map, compute_type_tensor_map):
        # type: (Union[Tensor, List[Tensor]], Dict[ComputeType, int], Dict[ComputeType, List[Tensor]]) -> None
        super().__init__(outs, compute_type_size_map, compute_type_tensor_map)

    def match(self):
        """
        check whether compute graph matches the current pattern
        """
        ph_size = self.compute_type_size_map.get(ComputeType.PLACEHOLDER, 0)
        elewise_size = self.compute_type_size_map.get(ComputeType.ELEWISE, 0)
        broadcast_size = self.compute_type_size_map.get(ComputeType.BROADCAST, 0)
        cast_size = self.compute_type_size_map.get(ComputeType.CAST, 0)
        total = self.compute_type_size_map.get(ComputeType.ANY, 0)

        if broadcast_size == 0:
            return False

        return ph_size + elewise_size + broadcast_size + cast_size == total

    def get_pattern(self):
        """
        return the current pattern
        """
        return Pattern.BROADCAST
