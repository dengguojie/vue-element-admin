#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2021-2021 Huawei Technologies Co., Ltd
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
transpose tiling case
"""
from abc import ABC
from enum import Enum
from enum import auto
from typing import Optional

from tbe.dsl.unify_schedule.computation import Computation
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.constants import TransposePattern
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from tbe.dsl.unify_schedule import util

DEFAULT = "default"
ALIGN_THRESHOLD = 128

TYPE_IN_BLOCK = {
    1: 32,
    2: 16,
    4: 8,
    8: 4,
}


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    NONE_CUT = auto()
    STORE_ALIGN_NONE_CUT = auto()
    READ_ALIGN = auto()
    READ_ALIGN_NONE_CUT = auto()
    CONST = auto()
    GENERAL = auto()
    STORE_ALIGN_SINGLE = auto()
    PURE_COPY = auto()


class TransposeComputation(Computation, ABC):
    """
    Transpose Tilingcase Computation
    """

    def __init__(self, outs, option):
        self.out = outs[0] if isinstance(outs, (list, tuple)) else outs
        self.option = option

    def get_sub_pattern(self):
        return TransposePattern.T_0

    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.TRANSPOSE]

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    def do_tiling_case(self):  # type: () -> list[Any]
        def is_const(shapes):
            return all(isinstance(s, int) for s in shapes)

        def add_general_case():
            none_cut_tiling_case = TransposeTilingCase()
            none_cut_tiling_case.tiling_key = 0
            none_cut_tiling_case.tiling_strategy = TilingStrategy.NONE_CUT
            tiling_case.append(none_cut_tiling_case)
            base_key = 2000000
            for i in range(dim_len):
                for j in range(i, dim_len):
                    for k in range(j + 1):
                        case = TransposeTilingCase()
                        case._tiling_key = base_key + i * 100 + k * 10 + j
                        case.tiling_strategy = TilingStrategy.GENERAL
                        case.block_split_axis = i
                        case.low_ub_split_axis = k
                        case.high_ub_split_axis = j
                        tiling_case.append(case)

        def add_store_align_case():
            none_cut_tiling_case = TransposeTilingCase()
            none_cut_tiling_case.tiling_key = 10000
            none_cut_tiling_case.tiling_strategy = TilingStrategy.STORE_ALIGN_NONE_CUT
            tiling_case.append(none_cut_tiling_case)

            base_key = 2030000
            for i in range(dim_len):
                for j in range(i, dim_len):
                    case = TransposeTilingCase()
                    case._tiling_key = base_key + i * 100 + j
                    case.tiling_strategy = TilingStrategy.STORE_ALIGN_SINGLE
                    case.block_split_axis = i
                    case.high_ub_split_axis = j
                    tiling_case.append(case)

        def add_read_align_case():
            base_key = 2020000
            none_cut_tiling_case = TransposeTilingCase()
            none_cut_tiling_case.tiling_key = 20000
            none_cut_tiling_case.tiling_strategy = TilingStrategy.READ_ALIGN_NONE_CUT
            tiling_case.append(none_cut_tiling_case)
            for i in range(dim_len):
                for j in range(i, dim_len):
                    for k in range(j + 1):
                        if k == dim_len - 1 and k == j:
                            continue
                        case = TransposeTilingCase()
                        case._tiling_key = base_key + i * 100 + k * 10 + j
                        case.tiling_strategy = TilingStrategy.READ_ALIGN
                        case.block_split_axis = i
                        case.low_ub_split_axis = k
                        case.high_ub_split_axis = j
                        tiling_case.append(case)

        transpose_index = [int(i) for i in self.out.op.attrs["permute"]]
        perm_len = len(transpose_index) - 1
        is_nlast_transpose = transpose_index[perm_len] == perm_len
        shape = util.shape_to_list(self.out.shape)
        dtype = self.out.dtype
        ele_in_block = TYPE_IN_BLOCK[DTYPE_BYTE_MAPPING[dtype]]
        nlast_align = is_nlast_transpose and (not isinstance(shape[-1], int)
                                              or shape[-1] > ALIGN_THRESHOLD or shape[-1] % ele_in_block == 0)
        is_read_align = is_nlast_transpose and (not isinstance(shape[-1], int) or shape[-1] % ele_in_block == 0)

        dim_len = len(shape)

        tiling_case = []

        if is_const(shape):
            const_tiling_case = TransposeTilingCase()
            const_tiling_case.tiling_key = 1000000
            const_tiling_case.tiling_strategy = TilingStrategy.CONST
            return [const_tiling_case]

        # pure copy
        if dim_len == 1:
            pure_copy_tiling_case = TransposeTilingCase()
            pure_copy_tiling_case.tiling_key = 3000000
            pure_copy_tiling_case.tiling_strategy = TilingStrategy.STORE_ALIGN_SINGLE
            return [pure_copy_tiling_case]

        add_general_case()
        if nlast_align:
            add_store_align_case()
        if is_read_align:
            add_read_align_case()

        return tiling_case


class TransposeTilingCase:
    """
    Transpose Tiling Case
    """

    def __init__(self):
        self._tiling_key = 0
        self._tiling_strategy: Optional[Enum] = None
        self._block_split_axis = 0
        # input ub split axis index by output
        self._low_ub_split_axis = 0
        # output ub split axis index by output
        self._high_ub_split_axis = 0
        self._enable_db = False

    @property
    def tiling_strategy(self):
        """
        :return: tiling_strategy
        """
        return self._tiling_strategy

    @property
    def enable_db(self):
        """
        enable_db
        """
        return self._enable_db

    @property
    def block_split_axis(self):
        """
        :return: block_split_axis
        """
        return self._block_split_axis

    @property
    def low_ub_split_axis(self):
        """
        :return: low_ub_split_axis
        """
        return self._low_ub_split_axis

    @property
    def high_ub_split_axis(self):
        """
        :return: high_ub_split_axis
        """
        return self._high_ub_split_axis

    @property
    def tiling_key(self):
        """
        :return: tiling_key
        """
        return self._tiling_key

    @tiling_key.setter
    def tiling_key(self, value):
        """
        set tiling_key
        :param value:
        :return:
        """
        self._tiling_key = value

    @tiling_strategy.setter
    def tiling_strategy(self, value):
        """
        set tiling_strategy
        :param value:
        :return:
        """
        self._tiling_strategy = value

    @block_split_axis.setter
    def block_split_axis(self, value):
        """
        set block_split_axis
        :param value:
        :return:
        """
        self._block_split_axis = value

    @low_ub_split_axis.setter
    def low_ub_split_axis(self, value):
        """
        set low_ub_split_axis
        :param value:
        :return:
        """
        self._low_ub_split_axis = value

    @high_ub_split_axis.setter
    def high_ub_split_axis(self, value):
        """
        set high_ub_split_axis
        :param value:
        :return:
        """
        self._high_ub_split_axis = value

    @enable_db.setter
    def enable_db(self, value):
        """
        set enable_db
        :param value:
        :return:
        """
        self._enable_db = value
