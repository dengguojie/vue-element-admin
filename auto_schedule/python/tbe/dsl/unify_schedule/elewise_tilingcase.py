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
from typing import Dict
from typing import Optional

from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_tiling_case

from . import util
from .constants import CompileInfo
from .constants import DTYPE_BYTE_MAPPING
from .constants import Pattern

SPECIAL = "special"
CONST = "const"
EMPTY = "empty"
STATIC = "static"
DB_KEY = 10000
EMPTY_KEY = -2 ** 31


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    ONE_CUT = auto()
    STATIC = auto()
    CONST = auto()
    EMPTY = auto()


# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.ELEMWISE)
def calc(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    # ###############TILING KEY RULE################
    # use int32, max value 2147483647
    # 0~1: dim len

    mode = operation.get_context().get("_mode")
    redundant_coe = _get_option_v(option, "redundant_coe", 0)

    def calc_base_key():
        if mode == SPECIAL:
            _base_key = 210000000
        elif mode == CONST:
            _base_key = operation.get_context().get("_const_base_key")
            if _base_key is None:
                _base_key = 100000000
            else:
                _base_key += 1
            operation.get_context().add("_const_base_key", _base_key)
        elif mode == EMPTY:
            _base_key = EMPTY_KEY
        else:
            _base_key = 0
        return _base_key

    base_key = calc_base_key()
    if mode in (CONST, EMPTY) or operation.get_context().get_mode() == STATIC:
        cases =  _const_tiling(base_key)
        for case in cases:
            case["redundant_coe"] = redundant_coe
        return cases

    # db handle
    enable_db_func = _default_db_func
    if mode == SPECIAL:
        enable_db_func = _pure_eletwise_db_func

    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    cases = _calc_elewise(outs, base_key, enable_db_func)

    for case in cases:
        case["redundant_coe"] = redundant_coe

    return cases


def _get_option_v(option, k, default_v=None):
    # type: (Optional[Dict[str, Any]], str, Any) -> Any

    return option.get(k, default_v) if option else default_v


def _default_db_func(db_params=None):
    return False


def _pure_eletwise_db_func():
    return True


def _const_tiling(base_key):
    cases = [{"key": base_key, "tiling_strategy": TilingStrategy.CONST}]
    return cases


def _calc_elewise(outs, base_key, enable_db_func=_default_db_func):
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
