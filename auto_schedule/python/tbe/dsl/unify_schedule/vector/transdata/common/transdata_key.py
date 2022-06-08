#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
Transdata Schedule Key
"""

from tbe.common.utils.errormgr import get_error_message
from .transdata_graph_info import ComputeGraphInfo
from .constants import FORWARD, BACKWARD, CONST_KEY
from .constants import STORAGE_ALIGN, COMMON_ALIGN
from .constants import TRANSPOSE_WORK, TRANSPOSE_NOT_WORK
from .constants import AVOID_CONFLICT_WORK, AVOID_CONFLICT_NOT_WORK
from .constants import CATEGORY_MAP_UB, RULE


class TransdataCase:
    """
    Obj-TransdataCase
    """

    def __init__(self, graph: ComputeGraphInfo):
        self.tiling_key = CONST_KEY

        # decide kinds of cases
        self.db = 0
        self.sch_type = FORWARD if graph.is_forward else BACKWARD
        self.ub_category = CATEGORY_MAP_UB.get(graph.category, 0)
        self.shape_type = 0
        self.block_split_idx = None
        self.ub_split_first_idx = None
        self.ub_split_second_idx = None
        self.transpose_work = 1
        self.avoid_bank_conflict = 1

        # update value
        self.block_factor = None
        self.ub_first_factor = None
        self.ub_second_factor = None
        self.tensor_ub_size_list = []
        self.transdata_category = graph.category


class TransdataSplit:
    """
    Obj-TransdataSplit: base + borrow
    """

    def __init__(self, outs, option):
        self.outs = outs
        self.option = option

        self.is_const = None
        self.bit = None
        self.ori_bit = None
        self.graph_info = None
        self.align_size = None

    @staticmethod
    def split(_length, _perm):
        """
        Return all split cases no matter sch support or not.
        Eg: Input:[A,B,C,D,E], output:[E,D,C,B,A], perm: [4,3,2,1,0]
                           |->split_i: B = B.outer * B.inner
                 Input: [A, B, C, D, E]
                 Output:[E, D, C, B, A]
                                     |->split_o: A = A.outer * A.inner
                 AxisInUB: [E, D, C, B.inner, A.inner]
                 AxisOutUB: [B.outer, A.outer]
                                |->split_b
        split_i : ub split in input
        split_o : ub split in output
        split_b : block split in output
        """
        _out = []
        for i in range(_length - 1, -1, -1):
            # axes in ub by split input that base on output
            input_axis_inner = {_perm.index(x) for x in range(i + 1, _length, 1)}
            for o in range(_length - 1, -1, -1):
                # axes in ub by split output
                output_axis_inner = set(range(o + 1, _length, 1))
                if o in input_axis_inner or _perm.index(i) in output_axis_inner:
                    continue
                output_axis_inner = output_axis_inner.union(input_axis_inner)
                axis_outer = set(range(_length)).difference(output_axis_inner)
                for b in axis_outer:
                    _out.append([b, _perm.index(i), o])
        return _out

    def base_filter(self, inputs):
        """
        Return legal cases from all split cases
        1. not support to split last c0
        2. maybe not existed c1 and c0
        3. not split c1 and c0 together
        """

        def split_c1c0_together(_input):
            c1 = self.graph_info.c1c0[0]
            c0 = self.graph_info.c1c0[1]
            return True if c1 in _input and c0 in _input else False

        def split_c0(_input):
            return True if self.graph_info.c1c0[1] in _input else False

        if not self.graph_info.c1c0:
            return inputs

        out = []
        for i in inputs:
            if not split_c1c0_together(i) and not split_c0(i):
                out.append(i)
        return out

    def base_generation(self, inputs):
        """
        For BaseSch, create cases as soon as possible.
        ShapeType has two mode: StorageAlign, CommonAlign.
        TransposeWork has two mode: work, not-work.
        AvoidBankConflict has two mode: work, not-work.
        """
        out = []
        shape_type = [STORAGE_ALIGN, COMMON_ALIGN]
        avoid_conflict_work = [AVOID_CONFLICT_WORK, AVOID_CONFLICT_NOT_WORK]
        transpose_work = [TRANSPOSE_WORK, ]
        if not self.graph_info.is_last_transpose:
            transpose_work.append(TRANSPOSE_NOT_WORK)

        for i in inputs:
            for j in shape_type:
                for k in transpose_work:
                    for l in avoid_conflict_work:
                        if l == AVOID_CONFLICT_WORK and k == TRANSPOSE_NOT_WORK:
                            continue
                        if l == AVOID_CONFLICT_WORK and j == COMMON_ALIGN and self.graph_info.is_forward:
                            continue
                        if j == COMMON_ALIGN and not self._check_common_align(i):
                            continue
                        case = TransdataCase(self.graph_info)
                        case.block_split_idx = i[0]
                        case.ub_split_first_idx = i[1]
                        case.ub_split_second_idx = i[2]
                        case.shape_type = j
                        case.transpose_work = k
                        case.avoid_bank_conflict = l
                        calc_key(case)
                        ComputeGraphInfo.update_tensor_ub_sizes(self.graph_info, case)
                        out.append(case)
        return out

    def borrow_filter(self, inputs):
        """
        Return legal cases from all split cases
        1. not split C
        2. not split X0(N0 H0)
        3. if src-tensor is fp32, it would be interpret as fp16 that don't split last-dim
        """

        def split_c(input_):
            for c in self.graph_info.c1c0:
                if c in input_:
                    return True
            return False

        out = []
        length = len(self.graph_info.transpose_2_tensor.shape)
        for i in inputs:
            not_split_c = not split_c(i)
            not_split_x0 = self.graph_info.x1x0[-1] not in i
            if not_split_c and not_split_x0:
                if length - 1 in i and self.ori_bit != self.bit:
                    continue
                out.append(i)
        return out

    def borrow_generation(self, inputs):
        result = []
        for i in inputs:
            for transpose_work in [TRANSPOSE_WORK, TRANSPOSE_NOT_WORK]:
                case = TransdataCase(self.graph_info)
                case.block_split_idx = i[0]
                case.ub_split_first_idx = i[1]
                case.ub_split_second_idx = i[2]
                case.transpose_work = transpose_work
                calc_key(case)
                ComputeGraphInfo.update_tensor_ub_sizes(self.graph_info, case)
                result.append(case)
        return result

    def _check_common_align(self, _input):
        """
        If forward: (N,H,C) -> (N,C1,H,C0), don't split C.
        If backward: (N,C1,H,C0) -> (N,H,C), don't split C.
        Attention backward split on transpose-tensor(N,H,C1,C0)
        """
        last_dim = self.graph_info.reshape[-1]
        if not isinstance(self.graph_info.reshape[-1], (list, tuple)):
            last_dim = [self.graph_info.reshape[-1]]

        new_input = [self.graph_info.permute[x] for x in _input] if self.graph_info.is_forward else _input
        for v in last_dim:
            if v in new_input:
                return False
        return True


def calc_key(case: TransdataCase):
    """
    :param case: TilingCase
    :return: case.tiling_key
    """

    value = [case.db, case.sch_type, case.ub_category, case.shape_type,
             case.block_split_idx, case.ub_split_first_idx, case.ub_split_second_idx,
             case.transpose_work, case.avoid_bank_conflict]

    def check(k, v):
        if k not in v[1]:
            dict_args = {"errCode": "E90003",
                         "detailed_cause": "%s should in %s, but is %d" % (v[0], str(v[1]), k)}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    key = 0
    for k, v in zip(value, RULE):
        check(k, v)
        key += k * v[-1]
    case.tiling_key = key
