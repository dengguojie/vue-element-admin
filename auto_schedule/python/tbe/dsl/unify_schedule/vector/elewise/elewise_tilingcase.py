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
from enum import Enum  # 'pylint: disable=E0611
from enum import auto
from typing import Optional

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
SPECIAL_SCALAR = "special_scalar"
COMMON = "common"
CONST = "const"
EMPTY = "empty"
STATIC = "static"
ORIGINAL = "original"

# TYPE DOUNDS
TYPE_DOUNDS = {
    1: (1, 32767),
    2: (1, 32767),
    4: (1, 16383),
    8: (1, 8191),
}

INT32_MAX = 2147483647
BLOCK_SIZE_BYTE = 32
DB_KEY = 10000
CONST_BASE_KEY = 100000000
COMMON_BASE_KEY = 210000000
NOT_ALL_FUSE_BASE_KEY = 211100000
BROADCAST_BASE_KEY = 220000000
BROADCAST_SCALAR_BASE_KEY = 223000000
SCALAR_BROADCAST_BASE_KEY = 232000000
TILING_DIV_VALUE = 100000
PATTERN_DIV_VALUE = 1000
CLASSIFY_PATTERN_LEN = 3
INDEX_MULTI_VALUE = 100
SHAPE_VAR_NUM = 10000
SHAPE_DIM_INDEX = 1
SHAPE_INPUT_INDEX = 2
BLOCK_VAR_NUM = 20000
BLOCK_TILING_AXIS_INDEX = 2
UB_VAR_NUM = 30000
UB_TILING_AXIS_INDEX = 2
ORI_DIM_VAR_NUM = 40000
ORI_C_INDEX = 2
ORI_C_INPUT_INDEX = 3


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    ALL_FUSE = auto()
    NOT_ALL_FUSE = auto()
    STATIC = auto()
    CONST = auto()
    EMPTY = auto()


# noinspection PyMethodMayBeStatic
class ElewiseComputation(Computation):
    """
    ElewiseComputation
    """

    def __init__(self, outs, option):
        self.outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        self.option = option

    def do_tiling_case(self):
        # get the max output dtype
        max_len = 0
        max_dtype = self.outs[0].dtype
        for out in self.outs:
            if len(out.shape) > max_len:
                max_len = len(out.shape)
            if DTYPE_BYTE_MAPPING.get(out.dtype) > DTYPE_BYTE_MAPPING.get(max_dtype):
                max_dtype = out.dtype

        # calculate the tiling_key
        pattern = operation.get_context().get_current_compute().get(CompileInfo.PATTERN)
        if pattern is not None:
            pattern = list(pattern)

        base_key = self.calc_base_key(pattern)

        mode = operation.get_context().get_current_compute().get("_mode")

        tiling_case = None
        if mode == CONST or operation.get_context().get_mode() == STATIC:
            tiling_case = self._calc_const_tiling_case(max_len)
        elif mode == EMPTY:
            tiling_case = self._calc_empty_tiling_case()
        elif mode in (SPECIAL, SPECIAL_SCALAR, ORIGINAL):
            if pattern != ['not_all_fuse']:
                tiling_case = self._calc_special_tiling_case(base_key, max_dtype)
            else:
                tiling_case = self._calc_not_all_fuse_tiling_case(max_len, base_key, max_dtype, self._default_db_func)

        return tiling_case

    def calc_base_key(self, pattern):
        """
        get tiling case base key
        """
        # common includes old pure elewise, scalar broadcast, distribute of broadcast common pattern
        if pattern == ['common']:
            base_key = COMMON_BASE_KEY
        elif pattern == ['not_all_fuse']:
            base_key = NOT_ALL_FUSE_BASE_KEY
        elif pattern == ['broadcast']:
            base_key = BROADCAST_BASE_KEY
        elif pattern == ['broadcast', 'scalar']:
            base_key = BROADCAST_SCALAR_BASE_KEY
        elif pattern == ['scalar', 'broadcast']:
            base_key = SCALAR_BROADCAST_BASE_KEY
        else:
            base_key = CONST_BASE_KEY

        return base_key

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
        return True

    def _calc_const_tiling_case(self, dim_len):
        base_key = operation.get_context().get("_const_base_key") or CONST_BASE_KEY
        const_tiling_case = ElewiseTilingCase()
        const_tiling_case.set_const_tiling_case(base_key, dim_len == 1)
        operation.get_context().add("_const_base_key", base_key + 1)
        return [const_tiling_case]

    def _calc_empty_tiling_case(self):
        empty_tiling_case = ElewiseTilingCase()
        empty_tiling_case.set_const_tiling_case(INT32_MAX)
        return [empty_tiling_case]

    def _calc_special_tiling_case(self, key, out_dtype):
        enable_db_func = self._default_db_func
        tiling_case = self._calc_all_fuse_tiling_case(key, out_dtype, enable_db_func)
        return tiling_case

    def _calc_all_fuse_tiling_case(self, base_key, dtype, enable_db_func):
        tiling_case = []
        all_fuse_tiling_case = ElewiseTilingCase()
        all_fuse_tiling_case.set_all_fuse_tiling_case(base_key, TYPE_DOUNDS.get(DTYPE_BYTE_MAPPING.get(dtype)))
        tiling_case.append(all_fuse_tiling_case)

        if enable_db_func():
            all_fuse_db_tiling_case = ElewiseTilingCase()
            all_fuse_db_tiling_case.set_all_fuse_tiling_case(base_key + DB_KEY,
                                                             TYPE_DOUNDS.get(DTYPE_BYTE_MAPPING.get(dtype)),
                                                             enable_db=True)
            tiling_case.append(all_fuse_db_tiling_case)
        return tiling_case

    def _calc_not_all_fuse_tiling_case(self, dim_len, base_key, dtype, enable_db_func):
        if dim_len == 1:
            return self._calc_all_fuse_tiling_case(base_key,
                                                   TYPE_DOUNDS.get(DTYPE_BYTE_MAPPING.get(dtype)),
                                                   self._default_db_func)
        tiling_case = []
        # do split
        block_axis = 0
        for i in range(dim_len):
            tiling_key = base_key + block_axis * dim_len + i
            not_all_fuse_tiling_case = ElewiseTilingCase()
            not_all_fuse_tiling_case.set_not_all_fuse_tiling_case(tiling_key, block_axis, i)
            tiling_case.append(not_all_fuse_tiling_case)

            if enable_db_func():
                tiling_key = base_key + block_axis * dim_len + i + DB_KEY
                not_all_fuse_db_tiling_case = ElewiseTilingCase()
                not_all_fuse_db_tiling_case.set_not_all_fuse_tiling_case(tiling_key, block_axis, i, enable_db=True)
                tiling_case.append(not_all_fuse_db_tiling_case)
        return tiling_case


def _pre_build(schedules_list):
    def _flatten_sch(_schedules: list):
        for sub_schs in schedules_list:
            if isinstance(sub_schs, list):
                _schedules.extend(sub_schs)
            else:
                _schedules.append(sub_schs)

    def _get_pattern_key(_tiling_key):
        _pattern_key = _tiling_key // TILING_DIV_VALUE % PATTERN_DIV_VALUE
        return str(_pattern_key).ljust(CLASSIFY_PATTERN_LEN, '0')

    def _name_to_int(_var_names):
        new_var_names = []
        for name in _var_names:
            if name[0] != '_':
                continue
            names = name[1:].split('_')
            if names[0] == 'dim':
                new_var_names.append(
                    SHAPE_VAR_NUM + int(names[SHAPE_DIM_INDEX]) * INDEX_MULTI_VALUE + int(names[SHAPE_INPUT_INDEX]))
            elif names[0] == 'block':
                new_var_names.append(BLOCK_VAR_NUM + int(names[BLOCK_TILING_AXIS_INDEX]))
            elif names[0] == 'ub':
                new_var_names.append(UB_VAR_NUM + int(names[UB_TILING_AXIS_INDEX]))
            elif names[0] == 'ori':
                new_var_names.append(
                    ORI_DIM_VAR_NUM + int(names[ORI_C_INDEX]) * INDEX_MULTI_VALUE + int(names[ORI_C_INPUT_INDEX]))
        return new_var_names

    def _add_const_compile_info():
        const_shapes, const_block_dims = [], []
        for const_compute in cpt_computes:
            const_shapes.append(const_compute.get("_const_shape"))
            const_block_dims.append(const_compute.get("_const_block_dim"))
        operation.add_compile_info_inner(CompileInfo.CONST_SHAPES, const_shapes)
        operation.add_compile_info_inner(CompileInfo.CONST_BLOCK_DIMS, const_block_dims)

    def _add_dynamic_compile_info():
        schedules = []
        _flatten_sch(schedules)

        op_vars = operation.get_context().get_vars()
        te_vars_list = []
        base_info = {}
        for cpt in cpt_computes:
            if cpt.get("_mode") == EMPTY or len(cpt.get_schedules()) == 0:
                continue
            cpt_vars = cpt.get_vars()
            cpt_ub_sizes, cpt_max_dtypes, cpt_coexisting_quantitys, cores = [], [], [], []
            for sch_context in cpt.get_schedules():
                sch_vars = sch_context.get_vars()
                te_vars_list.append(op_vars + cpt_vars + sch_vars)
                if sch_context.get("sch_pattern") == "rl_sch":
                    continue
                cpt_ub_sizes.append(sch_context.get(CompileInfo.UB_SIZE))
                cpt_max_dtypes.append(sch_context.get(CompileInfo.MAX_DTYPE))
                cpt_coexisting_quantitys.append(sch_context.get(CompileInfo.COEXISTING_QUANTITY))
                cores.append(sch_context.get(CompileInfo.CORE_NUM))
                tiling_key = sch_context.get("_tiling_key")

            pattern_key = _get_pattern_key(tiling_key)
            max_available_ub = (((min(cpt_ub_sizes) // max(cpt_coexisting_quantitys)) // BLOCK_SIZE_BYTE) *
                                BLOCK_SIZE_BYTE) // max(cpt_max_dtypes)
            max_available_ub_db = (((min(cpt_ub_sizes) // 2 // max(cpt_coexisting_quantitys)) // BLOCK_SIZE_BYTE) *
                                   BLOCK_SIZE_BYTE) // max(cpt_max_dtypes)
            base_info[pattern_key] = [max(cores), max(cpt_max_dtypes), max_available_ub, max_available_ub_db]

        if base_info:
            operation.add_compile_info_inner(CompileInfo.BASE_INFO, base_info)
        compile_vars = {}
        for sch, te_vars in zip(schedules, te_vars_list):
            if sch is None:
                continue
            var_names = [x.get_name() for x in te_vars]
            compile_vars[sch.tiling_key] = _name_to_int(var_names)
        if compile_vars:
            operation.add_compile_info_inner(CompileInfo.ELEWISE_VARS, compile_vars)

    def _set_special_build_config():
        # add build config
        operation.add_build_arg("double_buffer_non_reuse", True)
        # close double calculation switch
        operation.add_build_arg("enable_vector_2x", False)
        # if enable mask count, schedule use mask counter mode
        if util.is_v220():
            operation.add_build_arg("enable_mask_counter_mode", "default_counter")

    _set_special_build_config()

    only_const_tiling = False
    is_const_shapes = False
    support_broadcast = operation.get_context().get("_support_broadcast")
    use_special_pattern = False
    support_absorbable_broadcast = False
    unknown_rank = operation.get_context().get("_unknown_rank") or False
    cpt_computes = operation.get_context().get_computes()

    for compute in cpt_computes:
        classify_mode = compute.get("_mode")
        if classify_mode == CONST:
            is_const_shapes = True
        if classify_mode in [SPECIAL, SPECIAL_SCALAR]:
            use_special_pattern = True
        if classify_mode == SPECIAL_SCALAR:
            support_absorbable_broadcast = True

    flag_info = [only_const_tiling, is_const_shapes, support_broadcast,
                 use_special_pattern, support_absorbable_broadcast, unknown_rank]

    operation.add_compile_info_inner(CompileInfo.FLAG_INFO, flag_info)
    operation.add_compile_info_inner(CompileInfo.CLASSIFY_INPUTS_NUM,
                                     operation.get_context().get("_classify_inputs_num"))
    if is_const_shapes:
        _add_const_compile_info()
    else:
        _add_dynamic_compile_info()


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


class ElewiseTilingCase:
    """
    Elewise Tiling Case
    """

    def __init__(self):
        self._tiling_key = INT32_MAX
        self._tiling_strategy: Optional[Enum] = None
        self._block_split_axis = None
        self._ub_split_axis = None
        self._ub_factor_bound = None
        self._enable_db = False
        self._is_one_dim = False

    @property
    def tiling_key(self):
        """
        elewise tiling_key for schedule
        """
        return self._tiling_key

    @property
    def tiling_strategy(self):
        """
        elewise tiling_strategy for schedule
        """
        return self._tiling_strategy

    @property
    def block_split_axis(self):
        """
        elewise block_split_axis for schedule
        """
        return self._block_split_axis

    @property
    def ub_split_axis(self):
        """
        elewise ub_split_axis for schedule
        """
        return self._ub_split_axis

    @property
    def ub_factor_bound(self):
        """
        elewise ub_factor_bound for schedule
        """
        return self._ub_factor_bound

    @property
    def enable_db(self):
        """
        elewise enable_db for schedule
        """
        return self._enable_db

    @property
    def is_one_dim(self):
        """
        elewise is_one_dim for schedule
        """
        return self._is_one_dim

    def set_const_tiling_case(self, key, is_one_dim=False):
        """
        set const tiling case
        """
        self._tiling_strategy = TilingStrategy.CONST
        self._tiling_key = key
        self._is_one_dim = is_one_dim

    def set_all_fuse_tiling_case(self, key, ub_factor_bound, enable_db=False, is_one_dim=True):
        """
        set all fuse tiling case
        """
        self._tiling_strategy = TilingStrategy.ALL_FUSE
        self._tiling_key = key
        self._block_split_axis = 0
        self._ub_split_axis = 0
        self._ub_factor_bound = ub_factor_bound
        self._enable_db = enable_db
        self._is_one_dim = is_one_dim

    def set_not_all_fuse_tiling_case(self, key, block_split_axis, ub_split_axis, enable_db=False, is_one_dim=False):
        """
        set not all fuse tiling case
        """
        self._tiling_strategy = TilingStrategy.NOT_ALL_FUSE
        self._tiling_key = key
        self._block_split_axis = block_split_axis
        self._ub_split_axis = ub_split_axis
        self._enable_db = enable_db
        self._is_one_dim = is_one_dim
