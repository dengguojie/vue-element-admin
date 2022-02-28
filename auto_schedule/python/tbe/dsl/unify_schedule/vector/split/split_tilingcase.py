#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2022-2022 Huawei Technologies Co., Ltd
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
split tiling case
"""
from enum import Enum
from enum import auto
from typing import Optional

from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import add_build_arg
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.unify_schedule.computation import Computation
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.constants import SplitPattern
from tbe.dsl.unify_schedule import util


DEFAULT = "default"
SPLIT = "split"
SPLIT_GENERAL = "split_general"
SPLIT_EMPTY = "split_empty"

EMPTY_KEY = 2 ** 31 - 1


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    NONE_CUT = auto()
    CONST = auto()
    GENERAL = auto()
    CUT_M = auto()
    BASE = auto()
    ALL_ALIGN = auto()
    ALL_ALIGN_COPY = auto()
    EMPTY = auto()


class SplitComputation(Computation):
    """
    Split Tilingcase Computation
    """

    def __init__(self, outs, option):
        self.out = outs if isinstance(outs, (list, tuple)) else [outs]
        self.option = option

    def get_sub_pattern(self):
        return SplitPattern.S_0

    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.SPLIT]

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    def do_tiling_case(self):  # type: () -> list[Any]
        def is_const():
            for _out in self.out:
                shapes = util.shape_to_list(_out.shape)
                if not all(isinstance(s, int) for s in shapes):
                    return False
            return True

        def add_base_split_tiling_case(base_key, tiling_strategy):
            case = SplitTilingCase()
            case.tiling_key = base_key
            case.tiling_strategy = tiling_strategy
            case.block_split_axis = 0
            tiling_case.append(case)

        def add_double_split_tiling_case(base_key, tiling_strategy):
            for i in range(dim_len):
                case = SplitTilingCase()
                case.tiling_key = base_key + i
                case.tiling_strategy = tiling_strategy
                case.block_split_axis = i
                tiling_case.append(case)

        def add_single_split_tiling_case(tiling_key, tiling_strategy):
            case = SplitTilingCase()
            case.tiling_key = tiling_key
            case.tiling_strategy = tiling_strategy
            case.block_split_axis = 0
            tiling_case.append(case)

        shape = util.shape_to_list(self.out[0].shape)
        dtype = self.out[0].dtype
        dim_len = len(shape)

        tiling_case = []

        mode = get_context().get_current_compute().get("_mode")
        avg_split = get_context().get("_avg_split")
        if mode == SPLIT_EMPTY:
            add_single_split_tiling_case(EMPTY_KEY, TilingStrategy.EMPTY)
            return tiling_case

        if is_const():
            add_single_split_tiling_case(1000000, TilingStrategy.CONST)
            return tiling_case

        if mode == SPLIT:
            add_base_split_tiling_case(3000000, TilingStrategy.BASE)
            if len(self.out) == 1:
                return tiling_case
            add_double_split_tiling_case(4000000, TilingStrategy.ALL_ALIGN)
            add_single_split_tiling_case(4100000, TilingStrategy.ALL_ALIGN_COPY)

        if mode == SPLIT_GENERAL and dtype in ["float16", "int16", "uint16"]:
            add_single_split_tiling_case(0, TilingStrategy.NONE_CUT)
            add_single_split_tiling_case(5000000, TilingStrategy.CUT_M)
            if avg_split:
                add_double_split_tiling_case(2000000, TilingStrategy.GENERAL)

        return tiling_case


class SplitTilingCase:
    """
    Split Tiling Case
    """
    _enable_db: bool

    def __init__(self):
        self._tiling_key = 0
        self._tiling_strategy: Optional[Enum] = None
        self._block_split_axis = 0
        self._enable_db = False

    @property
    def tiling_key(self):
        """
        :return: tiling_key
        """
        return self._tiling_key

    @property
    def tiling_strategy(self):
        """
        :return: tiling_strategy
        """
        return self._tiling_strategy

    @property
    def block_split_axis(self):
        """
        :return: block_split_axis
        """
        return self._block_split_axis

    @property
    def enable_db(self):
        """
        enable_db
        """
        return self._enable_db

    @tiling_key.setter
    def tiling_key(self, value):
        """
        set tiling_key
        :param value:
        :return:
        """
        self._tiling_key = value

    @tiling_strategy.setter
    def tiling_strategy(self, value):
        """
        set tiling_strategy
        :param value:
        :return:
        """
        self._tiling_strategy = value

    @block_split_axis.setter
    def block_split_axis(self, value):
        """
        set block_split_axis
        :param value:
        :return:
        """
        self._block_split_axis = value

    @enable_db.setter
    def enable_db(self, value):
        """
        set enable_db
        :param value:
        :return:
        """
        self._enable_db = value


def _pre_build(schedules_list):
    def flatten_sch():
        for sub_schs in schedules_list:
            if isinstance(sub_schs, list):
                schedules.extend(sub_schs)
            else:
                schedules.append(sub_schs)

    def get_vars():
        cpt_computes = get_context().get_computes()
        for cpt in cpt_computes:
            cpt_vars = cpt.get_vars()
            for cpt_sch in cpt.get_schedules():
                split_nums.append(cpt_sch.get("_split_num"))
                compute_vars.append(cpt_vars)

    avg_split = get_context().get("_avg_split")
    schedules = []
    flatten_sch()
    compute_vars = []
    split_nums = []
    get_vars()
    for sch, c_vars, split_num in zip(schedules, compute_vars, split_nums):
        if sch is None:
            continue
        var_names = [x.get_name() for x in c_vars]
        if avg_split:
            if "_dim_1" in var_names and "_split_0" in var_names:
                tvm_vas = [x.get_tvm_var() for x in c_vars]
                dim_index = var_names.index("_dim_1")
                split_index = var_names.index("_split_0")
                sch.set_var_value(tvm_vas[dim_index], tvm_vas[split_index] * split_num)
        else:
            if "_dim_1" in var_names:
                dim_index = var_names.index("_dim_1")
                tvm_vas = [x.get_tvm_var() for x in c_vars]
                var_value = 0
                for name, t_var in zip(var_names, tvm_vas):
                    if name.startswith("_split"):
                        var_value += t_var
                sch.set_var_value(tvm_vas[dim_index], var_value)


@register_build_pointcut(pattern=Pattern.SPLIT)
def build_pointcut(func, *args, **kwargs):
    """
    build pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    add_build_arg("enable_branch_eliminator_else_case", False)
    _pre_build(args[0])
    func(*args, **kwargs)
