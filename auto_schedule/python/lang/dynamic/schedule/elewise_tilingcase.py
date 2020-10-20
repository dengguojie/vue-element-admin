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

from . import DTYPE_BYTE_MAPPING
from . import Pattern
from . import CompileInfo
from . import util
from te.platform.operation import register_tiling_case
from te.platform.operation import register_build
from te.platform import operation


COMMON = "common"
BROADCAST = "broadcast"
SCALAR = "scalar"
REDUCE = "reduce"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
CONST = "const"
ORIGINAL = "original"


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
@register_tiling_case(pattern=Pattern.ELEMWISE)
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
                    _base_key += (base*2)
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
            _base_key = 300000000
            base = 10000000
            for axis in pattern:
                if axis == SCALAR:
                    _base_key += base
                if axis == BROADCAST:
                    _base_key += (base*2)
                base //= 10
        else:
            _base_key = 0
        return _base_key

    base_key = calc_base_key()
    if mode == CONST:
        return _const_tiling(base_key)

    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    shape = util.shape_to_list(out.shape)
    dim_len = len(shape)

    if dim_len == 1:
        return _calc_one_dim(outs, base_key)

    return _calc_general(outs, base_key)


def _const_tiling(base_key):
    cases = [{"key": base_key, "tiling_strategy": TilingStrategy.CONST}]
    return cases


def _calc_one_dim(outs, base_key):
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
              }]

    return cases


def _calc_general(outs, base_key):
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

    return cases


# noinspection PyUnusedLocal
@register_build(pattern=Pattern.ELEMWISE)
def build(schedules_list, config_map=None):
    """
    :param schedules_list:
    :param config_map:
    :return:
    """
    support_broadcast = operation.get_context().get("support_broadcast")
    operation.add_compile_info(CompileInfo.IS_SUPPORT_BROADCAST,
                               support_broadcast)
    fusion_flag = util.get_build_cfg()
    if fusion_flag == "disable":
        operation.add_compile_info(CompileInfo.FUSION, 0)
    else:
        _fusion = operation.get_compile_info().get(CompileInfo.FUSION)
        if _fusion is None:
            operation.add_compile_info(CompileInfo.FUSION, 1)
    if operation.get_context().get("mode") == CONST:
        const_shapes = operation.get_context().get("const_shapes")
        operation.add_compile_info(CompileInfo.IS_CONST_SHAPES, True)
        operation.add_compile_info(CompileInfo.CONST_SHAPES, const_shapes)
        const_block_dims = operation.get_context().get("const_block_dims")
        operation.add_compile_info(CompileInfo.CONST_BLOCK_DIMS,
                                   const_block_dims)
    else:
        operation.add_compile_info(CompileInfo.IS_CONST_SHAPES, False)

    support_absorbable_broadcast = operation.get_context().get("support_absorbable_broadcast")
    operation.add_compile_info(CompileInfo.IS_SUPPORT_ABSORBABLE_BROADCAST,
                               support_absorbable_broadcast is True)

    cpt_computes = operation.get_context().get_computes()
    use_special_pattern = False
    for compute in cpt_computes:
        if compute.get("mode") != ORIGINAL:
            use_special_pattern = True
    operation.add_compile_info(CompileInfo.USE_SPECIAL_PATTERN, use_special_pattern)
