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
broadcast tiling case
"""
from enum import Enum  # pylint: disable=E0611
from enum import auto
from typing import Optional

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_build_pointcut
from tbe.common.platform import SOC_VERSION
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
UNKNOWN = "UNKNOWN"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
CONST = "const"
EMPTY = "empty"
STATIC = "static"
ORIGINAL = "original"
ASCEND920 = "Ascend920"
DB_KEY = 10000
EMPTY_KEY = -2 ** 31
BLOCK_SIZE_BYTE = 32

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
        # 0: model, 0 is ORIGINAL, 1 is CONST, 2 is SPECIAL or SPECIAL_SCALAR
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

    def _calc_special_tiling_case(self, dim_len, dtype):
        pattern = operation.get_context().get_current_compute().get(CompileInfo.PATTERN)
        base_key = 200000000
        base = 10000000
        for axis in pattern:
            if axis == COMMON:
                base_key += base
            elif axis == BROADCAST:
                base_key += (base * 2)
            elif axis == SCALAR:
                base_key += (base * 3)
            elif axis == UNKNOWN:
                base_key += (base * 9)
            base //= 10

        t_pattern = tuple(pattern)
        enable_db_func = self._default_db_func
        if t_pattern == (COMMON,):
            enable_db_func = self._pure_eletwise_db_func
        elif t_pattern == (COMMON, BROADCAST):
            enable_db_func = self._special_last_broadcast_db_func

        if dim_len == 1:
            tiling_case = self._calc_one_dim(dtype, base_key, enable_db_func)
        else:
            tiling_case = self._calc_general(dim_len, base_key, enable_db_func)
        return tiling_case

    def _calc_const_tiling_case(self, dim_len):
        base_key = operation.get_context().get("_const_base_key") or 100000000
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
            tiling_case = self._calc_general(dim_len, base_key, enable_db_func)
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

    def _calc_general(self, dim_len, base_key, enable_db_func):
        tiling_case = []

        # methodology 1:
        # general, no split: no block tiling, no ub tiling, no db
        none_cut_tiling_case = BroadcastTilingCase()
        none_cut_tiling_case.set_none_cut_tiling_case(base_key)
        tiling_case.append(none_cut_tiling_case)

        # methodology 2:
        # split special axis for block tiling and ub tiling
        # block tiling: fused the axis(which before multi-core axis)
        # and multi-core axis
        # ub tiling axis >= block tiling axis
        base_key += 1
        for i in range(dim_len):
            for j in range(i, dim_len):
                tiling_key = base_key + i * dim_len + j
                one_cut_tiling_case = BroadcastTilingCase()
                one_cut_tiling_case.set_one_cut_tiling_case(tiling_key, i, j)
                tiling_case.append(one_cut_tiling_case)

                db_params = {"ub_tiling_axis": j, "dim_len": dim_len}
                if enable_db_func(db_params):
                    tiling_key = base_key + i * dim_len + j + DB_KEY
                    one_cut_db_tiling_case = BroadcastTilingCase()
                    one_cut_db_tiling_case.set_one_cut_tiling_case(tiling_key, i, j, enable_db=True)
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
            cpt_ub_sizes, cpt_max_dtypes, cpt_coexisting_quantitys, cores = [], [], [], []
            tiling_key = -1
            for sch_context in cpt.get_schedules():
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
        operation.add_compile_info_inner(CompileInfo.BASE_INFO, base_info)

    # add build config
    operation.add_build_arg("double_buffer_non_reuse", True)

    only_const_tiling = False
    use_special_pattern = False
    support_absorbable_broadcast = False
    support_broadcast = operation.get_context().get("_support_broadcast")
    unknown_rank = operation.get_context().get("_unknown_rank") or False
    has_all_unknown = operation.get_context().get("_has_all_unknown") or False
    cpt_computes = operation.get_context().get_computes()
    is_const = False
    for compute in cpt_computes:
        if compute.get("_mode") in [SPECIAL, SPECIAL_SCALAR]:
            use_special_pattern = True
        if compute.get("_mode") == SPECIAL_SCALAR:
            support_absorbable_broadcast = True
        if compute.get("_mode") == CONST:
            is_const = True
    operation.add_compile_info_inner(CompileInfo.SOC_VERSION, get_soc_spec(SOC_VERSION))
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
        self._soc = get_soc_spec(SOC_VERSION)
        self._tiling_key = EMPTY_KEY
        self._tiling_strategy: Optional[Enum] = None
        self._block_split_axis = None
        self._ub_split_axis = None
        self._ub_factor_bound = None
        self._enable_db = False
        self._is_one_dim = False

    def set_none_cut_tiling_case(self, tiling_key):
        """
        set_none_cut_tiling_case
        """
        self._tiling_strategy = TilingStrategy.NONE_CUT
        self._tiling_key = tiling_key

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

    def set_one_dim_tiling_case(self, tiling_key, ub_factor_bound, enable_db=False, is_one_dim=False):
        """
        set_one_dim_tiling_case
        """
        self._tiling_strategy = TilingStrategy.ONE_CUT
        self._tiling_key = tiling_key
        self._block_split_axis = 0
        self._ub_split_axis = 0
        if self._soc != ASCEND920:
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
