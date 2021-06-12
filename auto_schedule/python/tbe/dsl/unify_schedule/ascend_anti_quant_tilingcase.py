# Copyright 2021 Huawei Technologies Co., Ltd
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
ascend_anti_quant tiling case
"""
from .constants import Pattern
from . import util
from tbe.dsl.base.operation import register_tiling_case
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_compile_info
from te.platform.cce_conf import get_soc_spec
from .vector_tilingcase import TilingCaseBase
from ...common.utils.errormgr import get_error_message


class AntiQuantTilingCase(TilingCaseBase):
    def __init__(self):
        self.block_tiling_axis = None
        self.ub_tiling_axis = None
        self.multi_core = True
        self.is_split_ub = None
        self.tiling_key = None
        self.block_factor = None
        self.ub_factor = None
        self.is_fuse_block = None

    def __repr__(self):
        return "ANTIQUANT:" + "b_axis=" + str(self.block_tiling_axis) + ",u_axis=" + str(self.ub_tiling_axis) + \
               ",is_split_ub=" + str(self.is_split_ub) + ",tiling_key=" + str(self.tiling_key) + \
               ",b_factor=" + str(self.block_factor) + ",u_factor=" + str(self.ub_factor) + \
               ",is_fuse_block=" + str(self.is_fuse_block)

    def __hash__(self):
        return hash((self.block_tiling_axis, self.block_factor, self.ub_tiling_axis, self.ub_factor, self.multi_core))

    def __eq__(self, other):
        condition1 = other.block_tiling_axis == self.block_tiling_axis
        condition2 = other.block_factor == self.block_factor
        condition3 = other.ub_tiling_axis == self.ub_tiling_axis
        condition4 = other.ub_factor == self.ub_factor
        condition5 = other.multi_core == self.multi_core
        return type(other) == type(self) and condition1 and condition2 and condition3 and condition4 and condition5

    def __ne__(self, other):
        condition1 = other.block_tiling_axis != self.block_tiling_axis
        condition2 = other.block_factor != self.block_factor
        condition3 = other.ub_tiling_axis != self.ub_tiling_axis
        condition4 = other.ub_factor != self.ub_factor
        condition5 = other.multi_core != self.multi_core
        return type(other) != type(self) or condition1 or condition2 or condition3 or condition4 or condition5


def _gen_tiling_case(b_start, b_end, u_end, is_fuse_block):
    """
    get all tiling cases
    """
    tiling_case_list = []
    for i in range(b_start, b_end):
        block_tiling_axis = i
        for j in range(i, u_end):
            ub_tiling_axis = j
            tiling_case = AntiQuantTilingCase()
            tiling_case.block_tiling_axis = block_tiling_axis
            tiling_case.ub_tiling_axis = ub_tiling_axis
            tiling_case.multi_core = True
            tiling_case.is_fuse_block = is_fuse_block
            if ub_tiling_axis == block_tiling_axis:
                tiling_case.is_split_ub = False
            else:
                tiling_case.is_split_ub = True
            tiling_case_list.append(tiling_case)
    return tiling_case_list


def _calc_tiling_key(shape, tiling):
    block_tiling_axis = tiling.block_tiling_axis
    ub_tiling_axis = tiling.ub_tiling_axis
    is_fuse_block = tiling.is_fuse_block
    shape_type, double_buf = 0, 0
    tiling_key = _get_tiling_key(double_buf, is_fuse_block, shape_type, block_tiling_axis, ub_tiling_axis, shape)
    tiling.tiling_key = tiling_key


def _get_tiling_key(double_buf, is_fuse_block, shape_type, block_tiling_axis, ub_tiling_axis, shape):
    """
    get tiling key
    """
    def _check(idx, value):
        rule = [range(2), range(2), range(100), range(9), range(9), range(1000)]
        name = ["double_buf", "is_fuse_block", "shape_type", "block_tiling_axis", "ub_tiling_axis", "pattern"]
        if value not in rule[idx]:
            dict_args = dict()
            dict_args["errCode"] = "E90003"
            dict_args["detailed_cause"] = "%s should in %s, but is %d" % (name[idx], str(rule[idx]), value)
            raise RuntimeError(dict_args, get_error_message(dict_args))

    pattern = _get_pattern_key(shape, block_tiling_axis, ub_tiling_axis)
    pos = (double_buf, is_fuse_block, shape_type, block_tiling_axis, ub_tiling_axis, pattern)
    val = (10 ** 9, 10 ** 7, 10 ** 6, 10 ** 5, 10 ** 4, 10 ** 3)
    key = 0
    for item, value in enumerate(pos):
        _check(item, value)
        key += value * val[item]
    return key


def _get_pattern_key(shape, block_tiling_axis=0, ub_tiling_axis=0):
    pattern_key = 0
    length = len(shape)
    for i in range(length):
        pattern_key += 2 ** (length - i - 1)
    pattern_key += block_tiling_axis * 100 + ub_tiling_axis * 10
    return pattern_key


def _init_max_ub_count():
    soc_ub_size = get_soc_spec("UB_SIZE") // 2
    total_width = 2
    max_bound = total_width * 128
    max_ub_count = int(soc_ub_size // max_bound * 128)
    return max_ub_count


# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.ASCEND_ANTI_QUANT)
def calc_tiling_case(outs, option=None):
    """
    antiquant tiling case
    """
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    out = outs[0]
    # dtype = out.dtype
    shape = util.shape_to_list(out.shape)
    shape[1] = shape[1] // 2
    shape[-1] = shape[-1] * 2

    tiling_case_list = []
    tiling_case_list += _gen_tiling_case(0, len(shape) - 1, len(shape) - 1, True)
    tiling_case_list += _gen_tiling_case(1, len(shape) - 2, len(shape) - 1, False)

    max_ub_count = _init_max_ub_count()
    core_num = get_soc_spec("CORE_NUM")
    common_info = [max_ub_count, core_num]

    pre_compile_info = get_compile_info()
    if pre_compile_info:
        info_map = {"common_info": common_info}
        for key in info_map.keys():
            if key not in pre_compile_info.keys():
                add_compile_info(key, info_map.get(key))
            else:
                if key != "common_info":
                    key_info = pre_compile_info.get(key)
                    key_info += info_map.get(key)
                    add_compile_info(key, key_info)
    else:
        raise RuntimeError("pre_compile_info is Null")

    for tiling_case in tiling_case_list:
        _calc_tiling_key(shape, tiling_case)

    return tiling_case_list