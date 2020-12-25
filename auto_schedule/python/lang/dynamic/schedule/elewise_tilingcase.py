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
from enum import Enum, auto  # pylint: disable=E0611

from . import CompileInfo
from . import DTYPE_BYTE_MAPPING
from . import Pattern
from . import util

from te.lang.base import operation_impl as operation
from te.lang.base.operation_impl import register_build_pointcut
from te.lang.base.operation_impl import register_tiling_case

COMMON = "common"
BROADCAST = "broadcast"
SCALAR = "scalar"
REDUCE = "reduce"
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
@register_tiling_case(pattern=(Pattern.ELEMWISE, Pattern.BROADCAST))
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
        if mode == SPECIAL:
            pattern = operation.get_context().get_current_compute().get("pattern")
            _base_key = 200000000
            base = 10000000
            for axis in pattern:
                if axis == COMMON:
                    _base_key += base
                if axis == BROADCAST:
                    _base_key += (base * 2)
                base //= 10
        elif mode == CONST:
            _base_key = operation.get_context().get("const_base_key")
            if _base_key is None:
                _base_key = 100000000
            else:
                _base_key += 1
            operation.get_context().add("const_base_key", _base_key)
        elif mode == SPECIAL_SCALAR:
            pattern = operation.get_context().get_current_compute().get("pattern")
            _base_key = 200000000
            base = 10000000
            for axis in pattern:
                if axis == SCALAR:
                    _base_key += (base*3)
                if axis == BROADCAST:
                    _base_key += (base * 2)
                base //= 10
        else:
            _base_key = 0
        return _base_key

    base_key = calc_base_key()
    if mode == CONST:
        return _const_tiling(base_key)

    # db handle
    enable_db_func = _default_db_func
    if mode == SPECIAL:
        pattern = operation.get_context().get_current_compute().get("pattern")

        pattern = tuple(pattern) if pattern else None

        if pattern == (COMMON,):
            enable_db_func = _pure_eletwise_db_func
        elif pattern == (COMMON, BROADCAST):
            enable_db_func = _special_last_broadcast_db_func

    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    shape = util.shape_to_list(out.shape)
    dim_len = len(shape)

    if dim_len == 1:
        return _calc_one_dim(outs, base_key, enable_db_func)

    return _calc_general(outs, base_key, enable_db_func)


def _default_db_func(db_params={}):
    return False


def _pure_eletwise_db_func(db_params={}):
    return True


def _special_last_broadcast_db_func(db_params={}):

    if db_params.get("ub_tiling_axis") == (db_params.get("dim_len") - 1):
        return True
    
    return False


def _const_tiling(base_key):
    cases = [{"key": base_key, "tiling_strategy": TilingStrategy.CONST}]
    return cases


def _calc_one_dim(outs, base_key, enable_db_func=_default_db_func):
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    dtype = out.dtype

    # schedule in var bound
    c_bounds = {
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
              "is_need_db": False
              }]

    if enable_db_func():
        cases.append({"key": base_key + DB_KEY,
              "block_tiling_axis": 0,
              "ub_tiling_axis": 0,
              "ub_factor_bound": c_bounds[DTYPE_BYTE_MAPPING[dtype]],
              "tiling_strategy": TilingStrategy.ONE_CUT,
              "is_need_db": True,
              "is_pure_eletwise": True
              })

    return cases


def _calc_general(outs, base_key, enable_db_func=_default_db_func):
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    cases = []
    out = outs[0]
    shape = util.shape_to_list(out.shape)
    dim_len = len(shape)

    # methodology 1:
    # general, no split: no block tiling, no ub tiling, no db
    cases.append({"key": base_key, "tiling_strategy": TilingStrategy.NONE_CUT})

    # methodology 2:
    # split special axis for block tiling and ub tiling
    # block tiling: fused the axis(which before multi-core axis)
    # and multi-core axis
    # ub tiling axis >= block tiling axis
    base = base_key + 1
    for i in range(dim_len):
        for j in range(i, dim_len):
            cases.append({
                "key": base + i * dim_len + j,
                "block_tiling_axis": i,
                "ub_tiling_axis": j,
                "tiling_strategy": TilingStrategy.ONE_CUT,
            })

            db_params = {"ub_tiling_axis": j, "dim_len": dim_len}
            if enable_db_func(db_params):
                cases.append({
                    "key": base + i * dim_len + j + DB_KEY,
                    "block_tiling_axis": i,
                    "ub_tiling_axis": j,
                    "tiling_strategy": TilingStrategy.ONE_CUT,
                    "is_need_db": True,
                    "is_pure_eletwise": False
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
                    _pattern_key += (base*2)
                else:
                    _pattern_key += (base*3)
        return str(_pattern_key).ljust(3, '0')

    operation.add_compile_info(CompileInfo.ONLY_CONST_TILING, False)
    support_broadcast = operation.get_context().get("support_broadcast")
    fusion_flag = util.get_build_cfg()
    if fusion_flag == "disable":
        _fusion = 0
    else:
        _fusion = operation.get_context().get(CompileInfo.FUSION)
        if _fusion is None:
            _fusion = 1
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
        flag_info = [support_broadcast, use_special_pattern, support_absorbable_broadcast, is_const_shapes, _fusion]
        operation.add_compile_info(CompileInfo.FLAG_INFO, flag_info)
        return
    else:
        is_const_shapes = False
    flag_info = [support_broadcast, use_special_pattern, support_absorbable_broadcast, is_const_shapes, _fusion]
    operation.add_compile_info(CompileInfo.FLAG_INFO, flag_info)

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
    operation.add_compile_info(CompileInfo.BASE_INFO, base_info)

    compile_vars = {}
    for sch, te_vars in zip(schedules, te_vars_list):
        if sch is None:
            continue
        var_names = [x.get_name() for x in te_vars]
        compile_vars[sch.tiling_key] = _name_to_int(var_names)
    operation.add_compile_info(CompileInfo.ELEWISE_VARS, compile_vars)

    # add build config
    operation.add_build_arg("double_buffer_non_reuse", True)


@register_build_pointcut(pattern=(Pattern.ELEMWISE, Pattern.BROADCAST))
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
