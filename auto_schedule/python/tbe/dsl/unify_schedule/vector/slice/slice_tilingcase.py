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
slice tiling case
"""
from enum import Enum
from enum import auto
from typing import Any

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut

from ... import util
from ...computation import Computation
from ...constants import CompileInfo
from ...constants import SlicePattern
from ...constants import Pattern

DEFAULT = "default"
# slice mode
DATA_MOV = "data_mov"
DEPAD = "depad"
LR_DEPAD = "lr_depad"
UNALIGED_STRIDE = "unaliged_stride"
BOTH_ALIGN = "both_align"
ONE_DIM = "one_dim"
SCALAR = "scalar"


MODE_VALUE_MAP = {
    DATA_MOV: 1000000,
    DEPAD: 2000000,
    UNALIGED_STRIDE: 3000000,
    BOTH_ALIGN: 4000000,
    ONE_DIM: 5000000,
    SCALAR: 6000000,
    LR_DEPAD: 7000000
}


class SliceCompileInfo:
    """
    Built-in Compile info keys
    """
    X_DTYPE_SIZE = "_x_dtype_size"
    IS_STATIC = "_is_static"
    IS_CONST = "_is_const"
    CONST_INFO = "_const_info"
    ZEROS_SHAPE = "_zero_shape"
    CONST_BEGINS = "_const_begins"
    CONST_SIZES = "_const_sizes"
    CONST_ENDS = "_const_ends"
    END_MODE = "_end_mode"


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    DYNAMIC = auto()
    STATIC = auto()
    ZEROS = auto()


class SliceComputation(Computation):
    """
    SliceComputation
    """

    def __init__(self, outs, option):
        self.outs = outs
        self.option = option

    @staticmethod
    def _calc_base_key(dim_len):
        return 500000000 + dim_len * 10000000

    @staticmethod
    def _calc_mode(ub_axis, dim_len):
        if ub_axis == dim_len - 1:
            return [DATA_MOV]
        return [DATA_MOV, DEPAD, UNALIGED_STRIDE, BOTH_ALIGN, SCALAR]

    @staticmethod
    def _calc_zero_slice():
        cases = []
        cases.append({
            "key": 500000000,
            "block_tiling_axis": 0,
            "ub_tiling_axis": 0,
            "tiling_strategy": TilingStrategy.ZEROS
        })

        return cases

    @classmethod
    def get_instance(cls, outs, option):  # type: (list[Any], dict[str, Any]) -> "Computation"
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):  # type: () -> list[str]
        return [Pattern.SLICE]

    @classmethod
    def get_supported_soc(cls):  # type: () -> list[str]
        return [DEFAULT]

    def do_tiling_case(self):
        is_static = operation.get_op_mode() == "static"
        is_const = operation.get_context().get(SliceCompileInfo.IS_CONST)
        is_zero_shape = operation.get_context().get_current_compute().get(SliceCompileInfo.ZEROS_SHAPE)

        # zero shape
        # const/dynamic
        if is_zero_shape:
            return self._calc_zero_slice()
        return self._calc_static_slice() if is_static or is_const else self._calc_dynamic_slice()

    def get_sub_pattern(self):  # type: () -> str
        return SlicePattern.NORMAL_SCHEDULE

    def _calc_dynamic_slice(self):
        out = self.outs[0] if isinstance(self.outs, (list, tuple)) else self.outs
        shape = util.shape_to_list(out.shape)
        dim_len = len(shape)

        base_key = self._calc_base_key(dim_len)
        cases = []

        if dim_len == 1:
            # one dim
            cases.append({
                "key": base_key + MODE_VALUE_MAP.get(ONE_DIM),
                "block_tiling_axis": 0,
                "ub_tiling_axis": 0,
                "tiling_strategy": TilingStrategy.DYNAMIC,
                "mode": ONE_DIM
            })

        # base tiling case, cover all condition
        for i in range(dim_len):
            for j in range(i, dim_len):
                mode = self._calc_mode(j , dim_len)
                for _mode in mode:
                    cases.append({
                        "key": base_key + MODE_VALUE_MAP.get(_mode) + i * dim_len + j,
                        "block_tiling_axis": i,
                        "ub_tiling_axis": j,
                        "tiling_strategy": TilingStrategy.DYNAMIC,
                        "mode": _mode
                    })

        # lr depad
        if dim_len == 2:
            cases.append({
                "key": base_key + MODE_VALUE_MAP.get(LR_DEPAD),
                "block_tiling_axis": 0,
                "ub_tiling_axis": 0,
                "tiling_strategy": TilingStrategy.DYNAMIC,
                "mode": LR_DEPAD
            })

        return cases

    def _calc_static_slice(self):
        out = self.outs[0] if isinstance(self.outs, (list, tuple)) else self.outs
        shape = util.shape_to_list(out.shape)
        dim_len = len(shape)

        base_key = self._calc_base_key(dim_len)
        cases = []
        cases.append({
            "key": base_key + 999,
            "tiling_strategy": TilingStrategy.STATIC
        })

        return cases


def _pre_build(schedules_list):
    def _flatten_sch(_schedules: list):
        for sub_schs in schedules_list:
            if isinstance(sub_schs, list):
                _schedules.extend(sub_schs)
            else:
                _schedules.append(sub_schs)

    # set special build cfg
    operation.add_build_arg("double_buffer_non_reuse", True)

    cpt_computes = operation.get_context().get_computes()

    schedules = []
    _flatten_sch(schedules)

    cpt_cores, cpt_ub_size, cpt_x_dtype_size = [], [], []

    for cpt in cpt_computes:
        for sch_context in cpt.get_schedules():
            cpt_cores.append(sch_context.get(CompileInfo.CORE_NUM))
            cpt_ub_size.append(sch_context.get(CompileInfo.UB_SIZE))
            cpt_x_dtype_size.append(sch_context.get(SliceCompileInfo.X_DTYPE_SIZE))

        base_info = [max(cpt_cores), min(cpt_ub_size), min(cpt_x_dtype_size)]
        operation.add_compile_info_inner(CompileInfo.BASE_INFO, base_info)

        cpt_end_mode = operation.get_context().get(SliceCompileInfo.END_MODE)
        operation.add_compile_info_inner(SliceCompileInfo.END_MODE, cpt_end_mode)

        cpt_is_const = operation.get_context().get(SliceCompileInfo.IS_CONST)
        operation.add_compile_info_inner(SliceCompileInfo.IS_CONST, cpt_is_const)
        if cpt_is_const:
            cpt_const_info = operation.get_context().get(SliceCompileInfo.CONST_INFO)
            operation.add_compile_info_inner(SliceCompileInfo.CONST_INFO, cpt_const_info)

        cpt_const_begins = operation.get_context().get(SliceCompileInfo.CONST_BEGINS)
        if cpt_const_begins:
            operation.add_compile_info_inner(SliceCompileInfo.CONST_BEGINS, cpt_const_begins)

        cpt_const_ends = operation.get_context().get(SliceCompileInfo.CONST_ENDS)
        if cpt_const_ends:
            operation.add_compile_info_inner(SliceCompileInfo.CONST_ENDS, cpt_const_ends)


def _post_build():
    """
    encode normal vars in norm sch
    """
    def _name_to_int(_var_names):
        new_var_names = []
        for name in _var_names:
            if name[0] != "_":
                continue
            names = name[1:].split("_")
            if names[0] == "x":
                new_var_names.append(10000 + int(names[2]))
            elif names[0] == "begin":
                new_var_names.append(20000 + int(names[2]))
            elif names[0] == "size":
                new_var_names.append(30000 + int(names[2]))
            elif names[0] == "block":
                new_var_names.append(40000 + int(names[2]))
            elif names[0] == "ub":
                new_var_names.append(50000 + int(names[2]))

        return new_var_names


    normal_vars = operation.get_compile_info().get(CompileInfo.NORMAL_VARS)
    slice_vars = {}
    for tiling_key, var_names in normal_vars.items():
        slice_vars[tiling_key] = _name_to_int(var_names)
    operation.add_compile_info_inner("_slice_vars", slice_vars)


@register_build_pointcut(pattern=Pattern.SLICE)
def build_pointcut(func, *args, **kwargs):
    """
    build_pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    _pre_build(args[0])
    func(*args, **kwargs)
    _post_build()

