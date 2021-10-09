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
elewise tiling case
"""
from enum import Enum  # pylint: disable=E0611
from enum import auto
from typing import Any

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut

from ... import util
from ...computation import Computation
from ...constants import CompileInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import ElewisePattern
from ...constants import Pattern

DEFAULT = "default"
SPECIAL = "special"
COMMON = "common"
CONST = "const"
EMPTY = "empty"
STATIC = "static"
ORIGINAL = "original"
DB_KEY = 10000
INT32_MAX = 2147483647
BLOCK_SIZE_BYTE = 32


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    ONE_CUT = auto()
    STATIC = auto()
    CONST = auto()
    EMPTY = auto()


# noinspection PyMethodMayBeStatic
class ElewiseComputation(Computation):
    """
    ElewiseComputation
    """

    def __init__(self, outs, option):
        self.outs = outs
        self.option = option

    def do_tiling_case(self):
        mode = operation.get_context().get_current_compute().get("_mode")

        def calc_base_key():
            if mode == SPECIAL:
                _base_key = 210000000
            elif mode == CONST:
                _base_key = operation.get_context().get("_const_base_key") or 100000000
                operation.get_context().add("_const_base_key", _base_key + 1)
            elif mode == EMPTY:
                _base_key = INT32_MAX
            else:
                _base_key = 0
            return _base_key

        base_key = calc_base_key()
        if mode in (CONST, EMPTY) or operation.get_context().get_mode() == STATIC:
            cases = self._const_tiling(base_key)
            return cases

        # db handle
        enable_db_func = self._default_db_func
        if mode == SPECIAL:
            enable_db_func = self._pure_eletwise_db_func

        outs = list(self.outs) if isinstance(self.outs, (list, tuple)) else [self.outs]
        cases = self._calc_elewise(outs, base_key, enable_db_func)

        return cases

    def get_sub_pattern(self):
        return ElewisePattern.E_0

    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.ELEMWISE]

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    def _default_db_func(self, db_params=None):
        return False

    def _pure_eletwise_db_func(self):
        return True

    def _const_tiling(self, base_key):
        cases = [{"key": base_key, "tiling_strategy": TilingStrategy.CONST}]
        return cases

    def _calc_elewise(self, outs, base_key, enable_db_func=_default_db_func):
        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        out = outs[0]
        dtype = out.dtype

        # schedule in var bound
        c_bounds = {
            0.125: (1, 32767),
            1: (1, 32767),
            2: (1, 32767),
            4: (1, 16383),
            8: (1, 8191),
        }

        cases = [{"key": base_key,
                  "block_tiling_axis": 0,
                  "ub_tiling_axis": 0,
                  "ub_factor_bound": c_bounds[DTYPE_BYTE_MAPPING[dtype]],
                  "tiling_strategy": TilingStrategy.ONE_CUT,
                  "is_need_db": False,
                  "is_one_dim": True
                  }]

        if enable_db_func():
            cases.append({"key": base_key + DB_KEY,
                          "block_tiling_axis": 0,
                          "ub_tiling_axis": 0,
                          "ub_factor_bound": c_bounds[DTYPE_BYTE_MAPPING[dtype]],
                          "tiling_strategy": TilingStrategy.ONE_CUT,
                          "is_need_db": True,
                          "is_one_dim": True
                          })

        return cases


def _pre_build(schedules_list):
    def _flatten_sch(_schedules: list):
        for sub_schs in schedules_list:
            if isinstance(sub_schs, list):
                _schedules.extend(sub_schs)
            else:
                _schedules.append(sub_schs)

    def _name_to_int(_var_names):
        new_var_names = []
        for name in _var_names:
            if name[0] != '_':
                continue
            names = name[1:].split('_')
            if names[0] == 'dim':
                new_var_names.append(10000 + int(names[1]) * 100 + int(names[2]))
            elif names[0] == 'block':
                new_var_names.append(20000 + int(names[2]))
            elif names[0] == 'ub':
                new_var_names.append(30000 + int(names[2]))
        return new_var_names

    def _get_pattern_key(_mode, _pattern):
        _pattern_key = 0
        if _mode == SPECIAL:
            base = 1
            for axis in _pattern:
                _pattern_key *= 10
                if axis == COMMON:
                    _pattern_key += base
        return str(_pattern_key).ljust(3, '0')

    def _set_special_build_config():
        # add build config
        operation.add_build_arg("double_buffer_non_reuse", True)
        # close double calculation switch
        operation.add_build_arg("enable_vector_2x", False)
        # ascend 920 eletwise schedule use mask counter mode
        if util.is_v220():
            operation.add_build_arg("enable_mask_counter_mode", "default_counter")

    _set_special_build_config()
    only_const_tiling = False
    support_broadcast = operation.get_context().get("_support_broadcast")
    unknown_rank = operation.get_context().get("_unknown_rank")
    unknown_rank = False if unknown_rank is None else unknown_rank
    cpt_computes = operation.get_context().get_computes()
    use_special_pattern = False
    support_absorbable_broadcast = False
    is_const = False
    for compute in cpt_computes:
        if compute.get("_mode") != ORIGINAL:
            use_special_pattern = True
        if compute.get("_mode") == CONST:
            is_const = True
    if is_const:
        const_shapes, const_block_dims = [], []
        for compute in cpt_computes:
            const_shapes.append(compute.get("_const_shape"))
            const_block_dims.append(compute.get("_const_block_dim"))
        is_const_shapes = True
        operation.add_compile_info_inner(CompileInfo.CONST_SHAPES, const_shapes)
        operation.add_compile_info_inner(CompileInfo.CONST_BLOCK_DIMS, const_block_dims)
        flag_info = [only_const_tiling, is_const_shapes, support_broadcast,
                     use_special_pattern, support_absorbable_broadcast, unknown_rank]
        operation.add_compile_info_inner(CompileInfo.FLAG_INFO, flag_info)
        return
    else:
        is_const_shapes = False
    flag_info = [only_const_tiling, is_const_shapes, support_broadcast,
                 use_special_pattern, support_absorbable_broadcast, unknown_rank]
    operation.add_compile_info_inner(CompileInfo.FLAG_INFO, flag_info)

    schedules = []
    _flatten_sch(schedules)

    te_vars_list = []
    op_vars = operation.get_context().get_vars()
    base_info = {}
    for cpt in cpt_computes:
        if cpt.get("_mode") == EMPTY or len(cpt.get_schedules()) == 0:
            continue
        cpt_vars = cpt.get_vars()
        cpt_ub_sizes, cpt_max_dtypes, cpt_coexisting_quantitys, cores = [], [], [], []
        for sch_context in cpt.get_schedules():
            sch_vars = sch_context.get_vars()
            cpt_ub_sizes.append(sch_context.get(CompileInfo.UB_SIZE))
            cpt_max_dtypes.append(sch_context.get(CompileInfo.MAX_DTYPE))
            cpt_coexisting_quantitys.append(sch_context.get(CompileInfo.COEXISTING_QUANTITY))
            cores.append(sch_context.get(CompileInfo.CORE_NUM))
            te_vars_list.append(op_vars + cpt_vars + sch_vars)
        pattern_key = _get_pattern_key(cpt.get("_mode"), cpt.get("_pattern"))
        max_available_ub = (((min(cpt_ub_sizes) // max(cpt_coexisting_quantitys)) // BLOCK_SIZE_BYTE) *
                            BLOCK_SIZE_BYTE) // max(cpt_max_dtypes)
        max_available_ub_db = (((min(cpt_ub_sizes) // 2 // max(cpt_coexisting_quantitys)) // BLOCK_SIZE_BYTE) *
                               BLOCK_SIZE_BYTE) // max(cpt_max_dtypes)
        base_info[pattern_key] = [max(cores), max(cpt_max_dtypes), max_available_ub, max_available_ub_db]
    operation.add_compile_info_inner(CompileInfo.BASE_INFO, base_info)

    compile_vars = {}
    for sch, te_vars in zip(schedules, te_vars_list):
        if sch is None:
            continue
        var_names = [x.get_name() for x in te_vars]
        compile_vars[sch.tiling_key] = _name_to_int(var_names)
    operation.add_compile_info_inner(CompileInfo.ELEWISE_VARS, compile_vars)


@register_build_pointcut(pattern=Pattern.ELEMWISE)
def build_pointcut(func, *args, **kwargs):
    """
    build pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    _pre_build(args[0])
    func(*args, **kwargs)
