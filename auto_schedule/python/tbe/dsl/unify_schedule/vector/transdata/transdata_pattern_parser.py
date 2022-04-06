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
transdata pattern parser
"""
from typing import Dict
from typing import List
from typing import Union

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.pattern_manager import PatternParser
from tbe.tvm.tensor import Tensor


class TransdataPatternParser(PatternParser):
    def __init__(self, outs, compute_type_size_map, compute_type_tensor_map):
        # type: (Union[Tensor, List[Tensor]], Dict[ComputeType, int], Dict[ComputeType, List[Tensor]]) -> None
        super().__init__(outs, compute_type_size_map, compute_type_tensor_map)

    def match(self):
        """
        check whether compute graph matches the current pattern
        """
        ph_size = self.compute_type_size_map.get(ComputeType.PLACEHOLDER, 0)
        transdata_size = self.compute_type_size_map.get(ComputeType.TRANSDATA, 0)
        total = self.compute_type_size_map.get(ComputeType.ANY, 0)
        return ph_size + transdata_size == total

    def get_pattern(self):
        """
        return the current pattern
        """
        return Pattern.TRANSDATA
