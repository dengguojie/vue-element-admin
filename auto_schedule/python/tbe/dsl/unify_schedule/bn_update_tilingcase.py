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
bn_training_update tiling case
"""
from enum import Enum, auto

from . import CompileInfo
from . import DTYPE_BYTE_MAPPING
from tbe.dsl.unify_schedule.constants import Pattern
from . import util

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import register_tiling_case
import tbe.common.platform as tbe_platform

COMMON = "common"
BROADCAST = "broadcast"
SCALAR = "scalar"
REDUCE = "reduce"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
CONST = "const"
STATIC = "static"
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


@register_tiling_case(pattern=Pattern.BN_UPDATE)
def calc(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    len([option])
    mode = operation.get_context().get("mode")

    base_key = 0
    if mode == CONST or operation.get_context().get_mode() == STATIC:
        return _const_tiling(base_key)

    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

    return _calc_general(outs, base_key)


def _const_tiling(base_key):
    cases = [{"key": base_key, "tiling_strategy": TilingStrategy.CONST}]
    return cases


def _calc_storage_bound(_is_db):
    _ub_size = util.get_ub_size()
    _coexisting_quantity = 7

    tensor_space = _ub_size // _coexisting_quantity
    if _is_db:
        tensor_space = tensor_space // 2
    _tensor_space = tensor_space // tbe_platform.BLOCK_REDUCE_INT8 * tbe_platform.BLOCK_REDUCE_INT8

    return _coexisting_quantity, _tensor_space


def _calc_general(outs, base_key):
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    cases = []
    dim_len = 4
    base = base_key + 1

    _dtypes = set()
    visited_tensors = set()

    def _dfs_sub_graph(out):
        _dtypes.add(out.dtype)
        for tensor_i in out.op.input_tensors:
            if tensor_i in visited_tensors:
                continue
            visited_tensors.add(tensor_i)
            _dfs_sub_graph(tensor_i)

    for out in outs:
        _dfs_sub_graph(out)

    max_dtype_bytes = max([DTYPE_BYTE_MAPPING[dtype] for dtype in _dtypes])
    coexisting_quantity, tensor_space = _calc_storage_bound(True)
    max_ub_count = tensor_space // max_dtype_bytes

    def add_case(block_tiling_axis, ub_tiling_axis, h_range, w_range, sub_key):
        cases.append({
            "key": (base + block_tiling_axis * dim_len + ub_tiling_axis) * 10 + sub_key,
            "block_tiling_axis": block_tiling_axis,
            "ub_tiling_axis": ub_tiling_axis,
            "tiling_strategy": TilingStrategy.ONE_CUT,
            "is_need_db": True,
            "coexisting_quantity": coexisting_quantity,
            "tensor_space": tensor_space,
            "max_dtype_bytes": max_dtype_bytes,
            "h_range": h_range,
            "w_range": w_range,
        })

    max_hw_count = max_ub_count // tbe_platform.C0_SIZE

    add_case(0, 1, h_range=[1, max_hw_count], w_range=[1, max_hw_count], sub_key=0)
    add_case(0, 2, h_range=[1, max_hw_count], w_range=[1, max_hw_count], sub_key=0)
    add_case(0, 2, h_range=[max_hw_count, None], w_range=[1, max_hw_count], sub_key=1)
    add_case(0, 3, h_range=[1, max_hw_count], w_range=[max_hw_count, None], sub_key=0)
    add_case(0, 3, h_range=[max_hw_count, None], w_range=[max_hw_count, None], sub_key=1)

    return cases


def _pre_build(schedules_list):
    def _flatten_sch(_schedules):
        for sub_schs in schedules_list:
            if isinstance(sub_schs, list):
                _schedules.extend(sub_schs)
            else:
                _schedules.append(sub_schs)

    def _name_to_int(_var_names):
        for index, name in enumerate(_var_names):
            names = name.split("_")
            if names[0] == "dim":
                _var_names[index] = 10000 + int(names[1]) * 100 + int(names[2])
            elif names[0] == "block":
                _var_names[index] = 20000 + int(names[2])
            elif names[0] == "ub":
                _var_names[index] = 30000 + int(names[2])
        return _var_names

    def _get_pattern_key(_mode, _pattern):
        _pattern_key = 0
        if _mode == SPECIAL or _mode == SPECIAL_SCALAR:
            base = 1
            for axis in _pattern:
                _pattern_key *= 10
                if axis == COMMON:
                    _pattern_key += base
                elif axis == BROADCAST:
                    _pattern_key += (base * 2)
                else:
                    _pattern_key += (base * 3)
        return str(_pattern_key).ljust(3, '0')

    only_const_tiling = False
    support_broadcast = operation.get_context().get("support_broadcast")
    cpt_computes = operation.get_context().get_computes()
    use_special_pattern = False
    support_absorbable_broadcast = False
    for compute in cpt_computes:
        if compute.get("mode") != ORIGINAL:
            use_special_pattern = True
        if compute.get("mode") == SPECIAL_SCALAR:
            support_absorbable_broadcast = True
    if operation.get_context().get("mode") == CONST:
        const_shapes, const_block_dims = [], []
        for compute in cpt_computes:
            const_shapes.append(compute.get("const_shape"))
            const_block_dims.append(compute.get("const_block_dim"))
        is_const_shapes = True
        operation.add_compile_info(CompileInfo.CONST_SHAPES, const_shapes)
        operation.add_compile_info(CompileInfo.CONST_BLOCK_DIMS, const_block_dims)
        flag_info = [only_const_tiling, is_const_shapes, support_broadcast, \
                     use_special_pattern, support_absorbable_broadcast]
        operation.add_compile_info(CompileInfo.FLAG_INFO, flag_info)
        return
    else:
        is_const_shapes = False
    flag_info = [only_const_tiling, is_const_shapes, support_broadcast, \
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
            te_vars_list.append(op_vars + cpt_vars)
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

    operation.add_build_arg("double_buffer_non_reuse", True)


@register_build_pointcut(pattern=Pattern.BN_UPDATE)
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
