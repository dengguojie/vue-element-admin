#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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
broadcast tiling case
"""
from enum import Enum  # 'pylint: disable=E0611
from enum import auto
from typing import Optional

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut
from tbe.common.platform import SHORT_SOC_VERSION
from tbe.common.platform import ASCEND_910B
from tbe.common.platform.platform_info import get_soc_spec

from ... import util
from ...computation import Computation
from ...constants import BroadcastPattern
from ...constants import CompileInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import Pattern

DEFAULT = "default"
COMMON = "common"
BROADCAST = "broadcast"
SCALAR = "scalar"
UNKNOWN = "unknown"
ALIGN = "align"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
CONST = "const"
EMPTY = "empty"
PURE_BRC = "pure_brc"
STORE_ALIGN = "store_align"
STATIC = "static"
ORIGINAL = "original"
DB_KEY = 10000
EMPTY_KEY = 2 ** 31 - 1
BLOCK_SIZE_BYTE = 32
DISABLE_BRANCH_NODE_LIMIT = 60
DIGIST_BASE = 10

# TYPE DOUNDS
TYPE_DOUNDS = {
    0.125: (1, 32767),
    1: (1, 32767),
    2: (1, 32767),
    4: (1, 16383),
    8: (1, 8191),
}


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    ALL_CUT = auto()
    NONE_CUT = auto()
    ONE_CUT = auto()
    STATIC = auto()
    CONST = auto()
    EMPTY = auto()


class TilingKey:
    CONST_BASE_KEY = 100000000
    SPECIAL_BASE_KEY = 200000000
    PURE_BRC_BASE_KEY = 300000000
    STORE_ALIGN_BASE_KEY = 400000000
    PATTERN_KEY = 10000000
    PATTERN_OFFSET = 30000000
    DB_KEY = 10000
    FEATURE_KEY = 1000000
    COMMON_KEY = 1
    BROADCAST_KEY = 2
    SCALAR_KEY = 3
    ALIGN_KEY = 8
    UNKNOWN_KEY = 9


# noinspection PyMethodMayBeStatic
class BroadcastComputation(Computation):
    """
    Broadcast Tilingcase Computation
    """

    def __init__(self, outs, option):
        self.outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        self.option = option

    def do_tiling_case(self):
        # ###############TILING KEY RULE################
        # use int32, max value 2147483647, currently using 9 bits, the description from high to low is as follows:
        # 0: model, 0 is ORIGINAL, 1 is CONST, 2 is SPECIAL or SPECIAL_SCALAR, 3 is PURE_BRC
        # 1-3: pattern, 1 is COMMON, 2 is BROADCAST, 3 is SCALAR, only valid by SPECIAL and SPECIAL_SCALAR
        # 4: double buffer, 0 is disable, 1 is enable
        # 5-8: block_split_axis * dim_length + ub_split_axis

        dim_len = 0
        dtype = self.outs[0].dtype
        for out in self.outs:
            shape = util.shape_to_list(out.shape)
            if len(shape) > dim_len:
                dim_len = len(shape)
            if DTYPE_BYTE_MAPPING[out.dtype] > DTYPE_BYTE_MAPPING[dtype]:
                dtype = out.dtype

        mode = operation.get_context().get_current_compute().get("_mode")
        if mode in (SPECIAL, SPECIAL_SCALAR):
            tiling_case = self._calc_special_tiling_case(dim_len, dtype)
        elif mode == CONST or operation.get_context().get_mode() == STATIC:
            tiling_case = self._calc_const_tiling_case(dim_len)
        elif mode == EMPTY:
            tiling_case = self._calc_empty_tiling_case()
        elif mode == STORE_ALIGN:
            tiling_case = self._calc_store_align_tiling_case(dim_len)
        elif mode == PURE_BRC:
            tiling_case = self._calc_pure_brc_tiling_case(dim_len, dtype)
        else:
            tiling_case = self._calc_original_tiling_case(dim_len, dtype)

        return tiling_case

    def get_sub_pattern(self):
        return BroadcastPattern.B_0

    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.BROADCAST]

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    def _default_db_func(self, db_params=None):
        return False

    def _pure_eletwise_db_func(self):
        return True

    def _calc_pattern_key(self, pattern):
        pattern_key = 0
        base = TilingKey.PATTERN_KEY
        for axis in pattern:
            if axis == COMMON:
                pattern_key += (base * TilingKey.COMMON_KEY)
            elif axis == BROADCAST:
                pattern_key += (base * TilingKey.BROADCAST_KEY)
            elif axis == SCALAR:
                pattern_key += (base * TilingKey.SCALAR_KEY)
            elif axis == ALIGN:
                pattern_key += (base * TilingKey.ALIGN_KEY)
            elif axis == UNKNOWN:
                pattern_key += (base * TilingKey.UNKNOWN_KEY)
            base //= DIGIST_BASE
        return pattern_key

    def _calc_special_tiling_case(self, dim_len, dtype):
        pattern = operation.get_context().get_current_compute().get(CompileInfo.PATTERN)
        base_key = TilingKey.SPECIAL_BASE_KEY + self._calc_pattern_key(pattern)

        t_pattern = tuple(pattern)
        enable_db_func = self._default_db_func
        if t_pattern == (COMMON,):
            enable_db_func = self._pure_eletwise_db_func
        elif t_pattern == (COMMON, BROADCAST):
            enable_db_func = self._special_last_broadcast_db_func

        if dim_len == 1:
            tiling_case = self._calc_one_dim(dtype, base_key, enable_db_func)
        else:
            tiling_case = self._calc_general(dim_len, base_key, False, enable_db_func)
        return tiling_case

    def _calc_store_align_tiling_case(self, dim_len):
        pattern = operation.get_context().get_current_compute().get(CompileInfo.PATTERN)
        base_key = TilingKey.STORE_ALIGN_BASE_KEY + self._calc_pattern_key(pattern)
        enable_db_func = self._default_db_func
        tiling_case = self._calc_general(dim_len, base_key, True, enable_db_func)
        return tiling_case

    def _calc_pure_brc_tiling_case(self, dim_len, dtype):
        pattern = operation.get_context().get_current_compute().get(CompileInfo.PATTERN)
        base_key = TilingKey.PURE_BRC_BASE_KEY + self._calc_pattern_key([pattern[-1]]) + TilingKey.PATTERN_OFFSET
        last_common = pattern[-1] == COMMON
        operation.get_context().add("_pure_brc_common", last_common)
        enable_db_func = self._default_db_func
        if dim_len == 1:
            tiling_case = self._calc_one_dim(dtype, base_key, enable_db_func)
        else:
            tiling_case = self._calc_general(dim_len, base_key, False, enable_db_func)
            if util.is_v220():
                return tiling_case
            base_key += TilingKey.FEATURE_KEY
            if last_common:
                base_key += DB_KEY
                operation.get_context().add("_pure_brc_common_db", True)
                store_align_case = self._calc_pure_brc_store_align(dim_len, base_key, True)
            else:
                store_align_case = self._calc_pure_brc_store_align(dim_len, base_key)
            tiling_case.extend(store_align_case)
        return tiling_case

    def _calc_const_tiling_case(self, dim_len):
        base_key = operation.get_context().get("_const_base_key") or TilingKey.CONST_BASE_KEY
        const_tiling_case = BroadcastTilingCase()
        const_tiling_case.set_const_tiling_case(base_key, dim_len == 1)
        tiling_case = [const_tiling_case]
        operation.get_context().add("_const_base_key", base_key + 1)
        return tiling_case

    def _calc_empty_tiling_case(self):
        empty_tiling_case = BroadcastTilingCase()
        empty_tiling_case.set_const_tiling_case(EMPTY_KEY)
        tiling_case = [empty_tiling_case]
        return tiling_case

    def _calc_original_tiling_case(self, dim_len, dtype):
        base_key = 0
        enable_db_func = self._default_db_func
        if dim_len == 1:
            tiling_case = self._calc_one_dim(dtype, base_key, enable_db_func)
        else:
            tiling_case = self._calc_general(dim_len, base_key, False, enable_db_func)
        return tiling_case

    def _special_last_broadcast_db_func(self, db_params):
        if db_params.get("ub_tiling_axis") == (db_params.get("dim_len") - 1):
            return True
        return False

    def _calc_one_dim(self, dtype, base_key, enable_db_func):
        one_dim_tiling_case = BroadcastTilingCase()
        one_dim_tiling_case.set_one_dim_tiling_case(base_key, TYPE_DOUNDS[DTYPE_BYTE_MAPPING[dtype]], is_one_dim=True)
        tiling_case = [one_dim_tiling_case]

        if enable_db_func():
            one_dim_db_tiling_case = BroadcastTilingCase()
            one_dim_db_tiling_case.set_one_dim_tiling_case(base_key + DB_KEY, TYPE_DOUNDS[DTYPE_BYTE_MAPPING[dtype]],
                                                           enable_db=True, is_one_dim=True)
            tiling_case.append(one_dim_db_tiling_case)

        return tiling_case

    def _calc_pure_brc_store_align(self, dim_len, base_key, enable_db=False):
        tiling_case = []

        # methodology 1:
        # general, no split: no block tiling, no ub tiling, no db
        none_cut_tiling_case = BroadcastTilingCase()
        none_cut_tiling_case.set_none_cut_tiling_case(base_key, True)
        tiling_case.append(none_cut_tiling_case)

        # methodology 2:
        block_axis = 0
        base_key += 1
        for i in range(dim_len):
            tiling_key = base_key + block_axis * dim_len + i
            one_cut_tiling_case = BroadcastTilingCase()
            one_cut_tiling_case.set_pure_brc_store_align_tiling_case(tiling_key, block_axis, i, enable_db=enable_db)
            tiling_case.append(one_cut_tiling_case)

        return tiling_case

    def _calc_general(self, dim_len, base_key, is_storage_align, enable_db_func):
        tiling_case = []

        # methodology 1:
        # general, no split: no block tiling, no ub tiling, no db
        if not is_storage_align:
            none_cut_tiling_case = BroadcastTilingCase()
            none_cut_tiling_case.set_none_cut_tiling_case(base_key)
            tiling_case.append(none_cut_tiling_case)

        # methodology 2:
        # split special axis for block tiling and ub tiling
        # block tiling: fused the axis(which before multi-core axis)
        # and multi-core axis
        # ub tiling axis >= block tiling axis
        # block_axis == 0. don't mean real axis
        block_axis = 0
        base_key += 1
        for i in range(dim_len):
            tiling_key = base_key + block_axis * dim_len + i
            one_cut_tiling_case = BroadcastTilingCase()
            one_cut_tiling_case.set_general_tiling_case(tiling_key, block_axis, i, is_storage_align)
            tiling_case.append(one_cut_tiling_case)

            db_params = {"ub_tiling_axis": i, "dim_len": dim_len}
            if enable_db_func(db_params):
                tiling_key = base_key + block_axis * dim_len + i + DB_KEY
                one_cut_db_tiling_case = BroadcastTilingCase()
                one_cut_db_tiling_case.set_general_tiling_case(
                    tiling_key, block_axis, i, is_storage_align, enable_db=True)
                tiling_case.append(one_cut_db_tiling_case)

        return tiling_case


def _pre_build():
    def _get_pattern_key(_tiling_key):
        # tiling key: use int32, max value 2147483647, currently using 9 bits
        # 1-3: pattern
        _pattern_key = _tiling_key // 100000 % 1000
        return str(_pattern_key).ljust(3, '0')

    def _add_const_compile_info():
        is_const_shapes = True
        flag_info = [only_const_tiling, is_const_shapes, support_broadcast,
                     use_special_pattern, support_absorbable_broadcast, unknown_rank, has_all_unknown]
        operation.add_compile_info_inner(CompileInfo.FLAG_INFO, flag_info)

        const_shapes, const_block_dims = [], []
        for compute in cpt_computes:
            const_shapes.append(compute.get("_const_shape"))
            const_block_dims.append(compute.get("_const_block_dim"))
        operation.add_compile_info_inner(CompileInfo.CONST_SHAPES, const_shapes)
        operation.add_compile_info_inner(CompileInfo.CONST_BLOCK_DIMS, const_block_dims)

    def _add_schedule_compile_info():
        is_const_shapes = False
        flag_info = [only_const_tiling, is_const_shapes, support_broadcast,
                     use_special_pattern, support_absorbable_broadcast, unknown_rank, has_all_unknown]
        operation.add_compile_info_inner(CompileInfo.FLAG_INFO, flag_info)

        base_info = {}
        for cpt in cpt_computes:
            if cpt.get("_mode") == EMPTY or len(cpt.get_schedules()) == 0:
                continue
            tiling_key = -1
            for sch_context in cpt.get_schedules():
                if sch_context.get("sch_pattern") == "rl_sch":
                    continue
                cur_ub_size = sch_context.get(CompileInfo.UB_SIZE)
                cur_max_dtype = sch_context.get(CompileInfo.MAX_DTYPE)
                cur_coexisting_quantity = sch_context.get(CompileInfo.COEXISTING_QUANTITY)
                cur_core = sch_context.get(CompileInfo.CORE_NUM)
                max_brc_type = sch_context.get(CompileInfo.MAX_BRC_TYPE) or 1
                tiling_key = sch_context.get("_tiling_key")
                pattern_key = _get_pattern_key(tiling_key)
                max_available_ub = (((cur_ub_size // cur_coexisting_quantity) // BLOCK_SIZE_BYTE) *
                                    BLOCK_SIZE_BYTE) // cur_max_dtype
                max_available_ub_db = (((cur_ub_size // 2 // cur_coexisting_quantity) // BLOCK_SIZE_BYTE) *
                                       BLOCK_SIZE_BYTE) // cur_max_dtype
                default_base_info = [cur_core, cur_max_dtype, max_available_ub, max_available_ub_db, max_brc_type]
                last_base_info = base_info.get(pattern_key, default_base_info)
                base_info[pattern_key] = [max(cur_core, last_base_info[0]),
                                          max(cur_max_dtype, last_base_info[1]),
                                          min(max_available_ub, last_base_info[2]),
                                          min(max_available_ub_db, last_base_info[3]),
                                          max(max_brc_type, last_base_info[4])]
        operation.add_compile_info_inner(CompileInfo.BASE_INFO, base_info)

    # add build config
    operation.add_build_arg("double_buffer_non_reuse", True)
    node_nums = operation.get_context().get("_node_nums") or 0
    if node_nums > DISABLE_BRANCH_NODE_LIMIT:
        operation.add_build_arg("enable_branch_eliminator_else_case", False)

    only_const_tiling = False
    use_special_pattern = False
    support_absorbable_broadcast = False
    support_broadcast = operation.get_context().get("_support_broadcast")
    unknown_rank = operation.get_context().get("_unknown_rank") or False
    has_all_unknown = operation.get_context().get("_has_all_unknown") or False
    all_unknown_last_const = operation.get_context().get("_all_unknown_last_const") or False
    cpt_computes = operation.get_context().get_computes()
    is_const = False
    is_pure_brc = False
    has_store_align = False
    compute_patterns = set()
    for compute in cpt_computes:
        mode = compute.get("_mode")
        if mode in [SPECIAL, SPECIAL_SCALAR]:
            use_special_pattern = True
        if mode == SPECIAL_SCALAR:
            support_absorbable_broadcast = True
        if mode == CONST:
            is_const = True
        if mode == PURE_BRC:
            is_pure_brc = True
        if mode == STORE_ALIGN:
            has_store_align = True

        compute_patterns.add(compute.get_pattern())

    operation.add_compile_info_inner(CompileInfo.CLASSIFY_INPUTS_NUM,
                                     operation.get_context().get("_classify_inputs_num"))
    operation.add_compile_info_inner(CompileInfo.CONTAINS_ELEWISE_SCH, False)
    if {Pattern.ELEMWISE, Pattern.BROADCAST} == compute_patterns:
        operation.add_compile_info_inner(CompileInfo.CONTAINS_ELEWISE_SCH, True)

    operation.add_compile_info_inner(CompileInfo.SOC_VERSION, get_soc_spec(SHORT_SOC_VERSION))
    operation.add_compile_info_inner("_all_unknown_last_const", all_unknown_last_const)
    operation.add_compile_info_inner("_is_pure_brc", is_pure_brc)
    operation.add_compile_info_inner("_has_store_align", has_store_align)
    if is_const:
        _add_const_compile_info()
        return
    else:
        _add_schedule_compile_info()


def _after_build():
    def _name_str_to_int(_var_names):
        new_var_names = []
        for name in _var_names:
            names = name[1:].split('_')
            if names[0] == 'dim':
                new_var_names.append(10000 + int(names[1]) * 100 + int(names[2]))
            elif names[0] == 'block':
                new_var_names.append(20000 + int(names[2]))
            elif names[0] == 'ub':
                new_var_names.append(30000 + int(names[2]))
            if names[0] == 'ori':
                new_var_names.append(40000 + int(names[2]) * 100 + int(names[3]))
        return new_var_names

    normal_vars = operation.get_compile_info().get(CompileInfo.NORMAL_VARS)
    broadcast_vars = {}
    for tiling_key, var_names in normal_vars.items():
        broadcast_vars[tiling_key] = _name_str_to_int(var_names)
    operation.add_compile_info_inner(CompileInfo.ELEWISE_VARS, broadcast_vars)


@register_build_pointcut(pattern=Pattern.BROADCAST)
def build_pointcut(func, *args, **kwargs):
    """
    build pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    _pre_build()
    func(*args, **kwargs)
    _after_build()


class BroadcastTilingCase:
    """
    Broadcast Tiling Case
    """

    def __init__(self):
        self._soc = get_soc_spec(SHORT_SOC_VERSION)
        self._tiling_key = EMPTY_KEY
        self._tiling_strategy: Optional[Enum] = None
        self._block_split_axis = None
        self._ub_split_axis = None
        self._ub_factor_bound = None
        self._enable_db = False
        self._is_one_dim = False
        self._is_storage_align = False

    def set_none_cut_tiling_case(self, tiling_key, is_storage_align=False):
        """
        set_none_cut_tiling_case
        """
        self._tiling_strategy = TilingStrategy.NONE_CUT
        self._tiling_key = tiling_key
        self._is_storage_align = is_storage_align

    def set_const_tiling_case(self, tiling_key, is_one_dim=False):
        """
        set_const_tiling_case
        """
        self._tiling_strategy = TilingStrategy.CONST
        self._tiling_key = tiling_key
        self._is_one_dim = is_one_dim

    def set_one_cut_tiling_case(self, tiling_key, block_split_axis,
                                ub_split_axis, enable_db=False, is_one_dim=False):
        """
        set_one_cut_tiling_case
        """
        self._tiling_strategy = TilingStrategy.ONE_CUT
        self._tiling_key = tiling_key
        self._block_split_axis = block_split_axis
        self._ub_split_axis = ub_split_axis
        self._enable_db = enable_db
        self._is_one_dim = is_one_dim

    def set_general_tiling_case(self, tiling_key, block_split_axis,
                                ub_split_axis, is_storage_align=False,
                                enable_db=False, is_one_dim=False):
        """
        set_general_tiling_case
        """
        self._tiling_strategy = TilingStrategy.ONE_CUT
        self._tiling_key = tiling_key
        self._block_split_axis = block_split_axis
        self._ub_split_axis = ub_split_axis
        self._enable_db = enable_db
        self._is_one_dim = is_one_dim
        self._is_storage_align = is_storage_align

    def set_pure_brc_store_align_tiling_case(self, tiling_key, block_split_axis,
                                             ub_split_axis, enable_db=False, is_one_dim=False):
        """
        set_pure_brc_store_align_tiling_case
        """
        self._tiling_strategy = TilingStrategy.ONE_CUT
        self._tiling_key = tiling_key
        self._block_split_axis = block_split_axis
        self._ub_split_axis = ub_split_axis
        self._enable_db = enable_db
        self._is_one_dim = is_one_dim
        self._is_storage_align = True

    def set_one_dim_tiling_case(self, tiling_key, ub_factor_bound, enable_db=False, is_one_dim=False):
        """
        set_one_dim_tiling_case
        """
        self._tiling_strategy = TilingStrategy.ONE_CUT
        self._tiling_key = tiling_key
        self._block_split_axis = 0
        self._ub_split_axis = 0
        if self._soc != ASCEND_910B:
            self._ub_factor_bound = ub_factor_bound
        self._enable_db = enable_db
        self._is_one_dim = is_one_dim

    @property
    def tiling_strategy(self):
        """
        return tiling_strategy
        """
        return self._tiling_strategy

    @property
    def enable_db(self):
        """
        enable_db
        """
        return self._enable_db

    @property
    def is_one_dim(self):
        """
        return is_one_dim
        """
        return self._is_one_dim

    @property
    def block_split_axis(self):
        """
        return block_split_axis
        """
        return self._block_split_axis

    @property
    def ub_split_axis(self):
        """
        return ub_split_axis
        """
        return self._ub_split_axis

    @property
    def tiling_key(self):
        """
        return tiling_key
        """
        return self._tiling_key

    @property
    def ub_factor_bound(self):
        """
        return ub_factor_bound
        """
        return self._ub_factor_bound

    @property
    def is_storage_align(self):
        """
        return is_storage_align
        """
        return self._is_storage_align
