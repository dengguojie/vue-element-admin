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
tuple reduce tilingcase
"""
# Standard Packages
from typing import List
from enum import Enum
from copy import deepcopy
# Local Packages
from tbe import tvm
from ... import util
from ...constants import Pattern
from ...computation import Computation
from ...constants import TupleReducePattern
# Tuple Reduce
from . import tuple_reduce_tilingcase_info

DEFAULT = "default"
CONST = "const"
DYNAMIC = "dynamic"


def _raise_error(message):
    dict_args = {"errCode": "E90003", "detailed_cause": message}
    raise RuntimeError(dict_args, get_error_message(dict_args))


class CalcTupleReduceTilingCase(Computation):
    """
    calculate tuple reduce tiling case
    """

    def __init__(self, outs, option):
        self.outs = outs
        self.option = option
        self.tiling_shape = None
    
    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)
    
    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.TUPLE_REDUCE]
    
    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]
    
    def get_sub_pattern(self):
        return TupleReducePattern.TR_0

    def do_tiling_case(self):
        """
        do tuple reduce tiling case
        @return:
        """
        return self.calc_tiling_case()
    
    def calc_tiling_case(self):
        """
        Generate tiling cases for all possible situations
        @return:
        """
        tiling_case_list: List[TupleReduceTilingCase] = []
        outs = list(self.outs) if isinstance(self.outs, (list, tuple)) else [self.outs]
        info = tuple_reduce_tilingcase_info.Info(outs)
        self.tiling_shape = info.max_shape
        for block_idx in range(len(self.tiling_shape)):
            for ub_idx in range(len(self.tiling_shape)):
                case = TupleReduceTilingCase(info, block_idx, ub_idx)
                if case.check_validity:
                    tiling_case_list.append(case)
        
        return tiling_case_list


class TupleReduceTilingCase:
    """
    tiling case data struct for tuple-reduce
    """

    class ScheduleType(Enum):
        """
        Tuple Reduce Schedule Type Enum
        """
        SPATIAL_TILING = "spatial-dependent schedule"
        TIME_TILING = "time-dependent schedule"
    
    def __init__(self, info: tuple_reduce_tilingcase_info.Info, block_axis: int, ub_axis: int):
        self.info = info
        self.block_axis = block_axis
        self.ub_axis = ub_axis
        self.block_factor = None
        self.ub_factor = None

        self.dynamic_mode = CONST if self.info.is_const else DYNAMIC
        if self.block_axis in self.info.reduce_axis:
            self.schedule_type = self.ScheduleType.TIME_TILING
        else:
            self.schedule_type = self.ScheduleType.SPATIAL_TILING
        self.tiling_key = self.calc_tiling_key()
    
    @property
    def check_validity(self):
        if self.info.is_const and self.tiling_key != self.info.tiling_key:
            return False
        if not self.info.atomic_support and self.block_axis in self.info.reduce_axis:
            return False
        if self.block_axis in self.info.reduce_axis \
                and self.ub_axis in self.info.reduce_axis \
                and self.block_axis  > self.ub_axis:
            return False
        if self.block_axis not in self.info.reduce_axis \
                and self.ub_axis not in self.info.reduce_axis \
                and self.block_axis > self.ub_axis:
            return False
        return True
    
    @staticmethod
    def _calc_reduce_axis(reduce_axis):
        one_hot = [0 for _ in range(8)]
        for i in reduce_axis:
            one_hot[i] = 1
        pattern = 0
        for v in one_hot:
            pattern = 2 * pattern + v
        return pattern
    
    def calc_tiling_key(self):
        block_axis, ub_axis = self.block_axis, self.ub_axis
        schedule_type = 0 if self.schedule_type == self.ScheduleType.SPATIAL_TILING else 1
        reduce_pattern = self._calc_reduce_axis(self.info.reduce_axis)
        key_values = (reduce_pattern, schedule_type, block_axis, ub_axis)
        key_weights = (10**3, 10**2, 10**1, 10**0)
        tiling_key = [value * weight for value, weight in zip(key_values, key_weights)]
        return sum(tiling_key)
