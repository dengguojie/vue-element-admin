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
dynamic softmax_cross_entropy_with_logits tiling case
"""
from enum import Enum  # pylint: disable=E0611
from enum import auto

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import register_tiling_case

from . import util
from .constants import CompileInfo
from .constants import Pattern

COMMON = "common"
BROADCAST = "broadcast"
SCALAR = "scalar"
REDUCE = "common_reduce"
BROADCAST_REDUCE = "broadcast_reduce"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
CONST = "const"
ORIGINAL = "original"
DB_KEY = 10000


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    ALL_CUT = auto()
    NONE_CUT = auto()
    ONE_CUT = auto()
    STATIC = auto()
    CONST = auto()


# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS)
def calc(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    # avoid pylint
    len([option])
    # ###############TILING KEY RULE################
    # use int32, max value 2147483647
    # 0~1: dim len
    mode = operation.get_context().get("mode")

    def calc_base_key():
        # first 1 is used to fill int32 length,
        # second 1 means the case is reduce mode.
        if mode == "copy":
            _base_key = 10
        elif mode == "vec1":
            _base_key = 1
        elif mode == "vec4":
            _base_key = 4
        elif mode == "vec2":
            _base_key = 2
        elif mode == "vec8":
            _base_key = 8
        elif mode == "vec9":
            _base_key = 9
        elif mode == "vec6":
            _base_key = 6
        else:
            _base_key = 0
        return _base_key

    base_key = calc_base_key()

    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

    a = _calc_general(outs, base_key)
    return _calc_general(outs, base_key)


def _const_tiling(base_key):
    cases = [{"key": base_key, "tiling_strategy": TilingStrategy.CONST}]
    return cases


def _calc_general(outs, base_key):
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    if len(outs) > 1:
        out = outs[1]
    else:
        out = outs[0]
    shape = util.shape_to_list(out.shape)
    dim_len = len(shape)

    cases = []
    # methodology 1:
    # general, no split: no block tiling, no ub tiling, no db
    # cases.append({"key": base_key, "tiling_strategy": TilingStrategy.NONE_CUT})

    # methodology 2:
    # split special axis for block tiling and ub tiling
    # block tiling: fused the axis(which before multi-core axis)
    # and multi-core axis
    # ub tiling axis >= block tiling axis
    base = base_key
    for i in range(dim_len):
        for j in range(i, dim_len):
            if i < 1 and j < 1:
                cases.append({
                    "key": base + i * 10 + j,
                    "block_tiling_axis": i,
                    "ub_tiling_axis": j,
                    "tiling_strategy": TilingStrategy.ONE_CUT,
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
                if axis == BROADCAST:
                    _pattern_key += (base * 2)
                if axis == REDUCE:
                    _pattern_key += (base * 3)
                if axis == BROADCAST_REDUCE:
                    _pattern_key += (base * 4)
        return str(_pattern_key).ljust(3, '0')

    only_const_tiling = False
    # support_broadcast = operation.get_context().get("support_broadcast")
    support_broadcast = True
    cpt_computes = operation.get_context().get_computes()
    use_special_pattern = False
    support_absorbable_broadcast = False
    if operation.get_context().get("_mode") == CONST:
        const_shapes, const_block_dims = [], []
        for compute in cpt_computes:
            const_shapes.append(compute.get("const_shape"))
            const_block_dims.append(compute.get("const_block_dim"))
        is_const_shapes = True

        operation.add_compile_info(CompileInfo.CONST_SHAPES, const_shapes)
        operation.add_compile_info(CompileInfo.CONST_BLOCK_DIMS, const_block_dims)
        flag_info = [only_const_tiling, is_const_shapes, support_broadcast,
                     use_special_pattern, support_absorbable_broadcast]
        operation.add_compile_info("flag_info", flag_info)
        return
    else:
        is_const_shapes = False
    flag_info = [only_const_tiling, is_const_shapes, support_broadcast,
                 use_special_pattern, support_absorbable_broadcast]
    operation.add_compile_info("flag_info", flag_info)

    schedules = []
    _flatten_sch(schedules)

    te_vars_list = []
    op_vars = operation.get_context().get_vars()
    base_info = {}
    for i, cpt in enumerate(cpt_computes):
        cpt_vars = cpt.get_vars()
        cpt_ub_sizes, cpt_max_dtypes, cpt_coexisting_quantitys, cores = [], [], [], []
        for sch_context in cpt.get_schedules():
            sch_vars = sch_context.get_vars()
            cpt_ub_sizes.append(sch_context.get(CompileInfo.UB_SIZE))
            cpt_max_dtypes.append(sch_context.get(CompileInfo.MAX_DTYPE))
            cpt_coexisting_quantitys.append(sch_context.get(CompileInfo.COEXISTING_QUANTITY))
            cores.append(sch_context.get(CompileInfo.CORE_NUM))
            te_vars_list.append(op_vars + cpt_vars + sch_vars)
        pattern_key = _get_pattern_key(cpt.get("mode"), cpt.get("pattern"))
        base_info[pattern_key] = [min(cpt_ub_sizes), max(cpt_max_dtypes), max(cpt_coexisting_quantitys), max(cores)]
    operation.add_compile_info("base_info", base_info)

    compile_vars = {}
    for sch, te_vars in zip(schedules, te_vars_list):
        if sch is None:
            continue
        var_names = [x.get_name() for x in te_vars]
        compile_vars[sch.tiling_key] = _name_to_int(var_names)
    operation.add_compile_info("elewise_vars", compile_vars)

    # add build config
    operation.add_build_arg("double_buffer_non_reuse", True)


@register_build_pointcut(pattern=Pattern.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS)
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