#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
transdata forward schedule
"""
from tbe.dsl.base.operation import var_inner
from tbe.dsl.unify_schedule.constants import TransdataCategory
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from tbe.common.utils.shape_util import shape_to_list
from tbe.common.utils.errormgr import get_error_message
from ..common.transdata_base_sch import TransdataBaseSch
from ..common.transdata_graph_info import math_prod, set_align
from ..common.constants import STORAGE_ALIGN, COMMON_ALIGN
from ..common.constants import BLOCK, NO_OVERLAP, B8, B16


class TransForwardSchedule(TransdataBaseSch):
    """
    TransForwardSchedule: base + forward schedule
    """

    def __init__(self, outs, tiling_case):
        TransdataBaseSch.__init__(self, outs, tiling_case)
        self.mapping_indexes = []
        self.c0_index = None
        self.c1_index = None

    @classmethod
    def get_supported_sub_pattern(cls):
        return TransdataCategory.GENERAL_FORWARD

    def do_schedule(self, ):
        """
        Process of schedule
        """
        self._analysis_case()
        self._create_schedule()
        self._do_cache_read()
        self._do_set_scope()
        self._do_cache_write()
        self._init_tensors()

        self._calc_tiling()
        self._do_tiling()

        self._analysis_transpose()

        self._do_storage_bound()
        self._do_set_constraint()
        self._do_storage_align()
        self._do_mem_reused()

        self._calc_multi_core()
        self._do_multi_core()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_emit_insn()
        self._do_emit_insn()

        self.schedule.tiling_key = self.tiling_case.tiling_key
        return self.schedule

    def _analysis_case(self, ):
        if self.tiling_case.shape_type not in [COMMON_ALIGN, STORAGE_ALIGN]:
            dict_args = {"errCode": "E90003", "detailed_cause": "branch is error."}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def _init_tensors(self):
        """
        StorageAlign Process:
        input -> mte2_tensor -> pad_tensor -> reshape_tensor -> transpose_tensor -> output
        CommonAlign Process:
        input -> mte2_tensor -> align_tensor -> pad_tensor -> reshape_tensor -> transpose_tensor -> out
        """
        self.pad_tensors = list(self.graph_info.pad_tensor_set)
        self.pad_tensors.sort(key=lambda x: int(x.op.attrs["axes"]), reverse=True)
        self.pad_axis_list = [int(x.op.attrs["axes"]) for x in self.pad_tensors]

        # remove tiling_tensor from transpose_tensors
        self.tiling_tensor = list(self.graph_info.output_tensor_set)[0]
        self.transpose_tensors = list(self.graph_info.transpose_tensor_set)
        self.transpose_tensors.sort(key=lambda x: int(x.op.attrs["order"]))
        self.transpose_tensors[-1] = self.parent(self.tiling_tensor)

        self.reshape_tensors = list(self.graph_info.s_reshape_tensor_set)
        self.reshape_tensors.sort(key=lambda x: int(x.op.attrs["order"]))
        self.mte2_tensor = self.child(list(self.graph_info.input_tensor_set)[0])

        if self.tiling_case.shape_type == COMMON_ALIGN:
            self.align_pad_tensor = self.single_cache_read(self.mte2_tensor)

    def _calc_tiling(self):
        case = self.tiling_case
        self.tiling_axes = [x for x in self.tiling_tensor.op.axis]
        self.split_once = case.ub_split_second_idx == case.ub_split_first_idx
        self.parses_axis_type(self.graph_info.permute)
        if not case.ub_first_factor:
            case.ub_first_factor = var_inner("_ub_first_factor", (1, None))
        if not self.split_once and not case.ub_second_factor:
            case.ub_second_factor = var_inner("_ub_second_factor", (1, None))
        if not case.block_factor:
            case.block_factor = var_inner("_block_factor", (1, None))

    def _do_tiling(self):
        self._do_ub_tiling()
        self._do_block_tiling()
        self._do_reorder()
        self._do_fragments()

    def _do_set_constraint(self):
        case = self.tiling_case
        blk_idx = case.block_split_idx
        ub_first_idx = case.ub_split_first_idx
        ub_second_idx = case.ub_split_second_idx
        tiling_shape = [x.dom.extent for x in self.tiling_axes]

        constraint = []
        constraint.append(case.block_factor <= tiling_shape[blk_idx])
        constraint.append(case.ub_first_factor <= tiling_shape[ub_first_idx])
        if not self.split_once:
            constraint.append(case.ub_second_factor <= tiling_shape[ub_second_idx])

        for item in constraint:
            if isinstance(item, bool):
                continue
            self.schedule.set_constraint(item)

    def _align_mte2_tensors(self, factor):
        # In common-align, mte2_tensor(NHC) would be align_pad_tensor([NH,Cx]).
        # In storage-align, gm(NHC) would be mte2_tensor([NH,Cx]).
        # Cx = align(C, factor).
        for tensor in self.forward_stage_graph_map.keys():
            if tensor in self.graph_info.input_tensor_set:
                continue
            if tensor in self.graph_info.output_tensor_set:
                continue
            if tensor == self.mte2_tensor and self.tiling_case.shape_type == COMMON_ALIGN:
                continue
            if len(tensor.op.axis) >= 2:
                self.schedule[tensor].storage_align(tensor.op.axis[-2], factor, 0)

    def _align_pad_tensors(self, factor):
        # Padding doesn't malloc tensor, but occur on src_tensor.
        # Src would malloc enough space by storage_align.
        # Don't align first and last dim in [mte2-tensor, align-pad-tensor].
        # Existed align-pad-tensor, don't align mte-tensor.
        tensor = self.mte2_tensor if self.tiling_case.shape_type == STORAGE_ALIGN else self.align_pad_tensor
        shape = shape_to_list(tensor.shape)
        shape[-1] = set_align(shape[-1], factor)

        for j, i in enumerate(self.pad_axis_list):
            if i not in [0, len(shape) - 1]:
                _factor = math_prod(shape[i + 1:]) * self.pad_factor
                shape[i] = set_align(shape[i], self.pad_factor)
                self.schedule[tensor].storage_align(tensor.op.axis[i - 1], _factor, 0)
                for pad_tensor in self.pad_tensors[:j]:
                    self.schedule[pad_tensor].storage_align(pad_tensor.op.axis[i - 1], _factor, 0)

    def _align_transpose_tensors(self, ):
        # n-last-transpose, c0 is last dim, don't need storage align
        # last-transpose, (m,n) -> (n,m), C0 must be m or n
        # last-transpose, (m,n) -> (n,m), storage align dst that make dst had space to store
        # In 5HD(NC1C0H->NC1HC0), fp16, fp8 don't need storage align
        # In 5HD(NC1C0H->NC1HC0), fp32, int32 need C0 storage align 12
        for tensor, perm in zip(self.transpose_tensors, self.permutes):
            if perm and perm[-1] != len(perm) - 1:
                self.do_vnchwconv_align(tensor, self.align_factor)

    def _ub_internal_length(self, index, length):
        if index == self.tiling_case.ub_split_first_idx:
            return self.tiling_case.ub_first_factor
        if index == self.tiling_case.ub_split_second_idx:
            return self.tiling_case.ub_second_factor
        return length

    def _avoid_conflict_nc1hwc0(self, tensor, optimize, a_idx, b_idx):
        """
        Transpose [a,b] to [b,a]
        :param optimize: The func that avoid conflict in transpose instructions.
        :param a_idx: a's index in [b,a]
        :param b_idx: b's index in [b,a]
        :constraint: Support 5HD and NZ
        """
        # a: Left var in transpose-mode [a,b] -> [b,a]
        # b: Right var in transpose-mode [a,b] -> [b,a]
        # c0: align value
        shape = shape_to_list(tensor.shape)
        a = self._ub_internal_length(a_idx, shape[a_idx])
        b = self._ub_internal_length(b_idx, shape[b_idx])
        c0 = shape[self.graph_info.c1c0[1]]

        # align src-tensor to avoid conflict
        # pad occur before transpose need align.
        src, dst = optimize(a, b, c0, tensor)
        perm = [int(x) for x in list(tensor.op.attrs["permute"])]
        for _tensor in self.get_all_producers_stages(tensor):
            if _tensor in self.graph_info.input_tensor_set:
                continue
            if _tensor in self.reshape_tensors:
                self.schedule[_tensor].storage_align(_tensor.op.axis[perm[a_idx]], src, 0)
                continue
            # pad need compute-align (wait new sch api).
            self.schedule[_tensor].storage_align(_tensor.op.axis[-2], src, 0)

        # update pad operation while align src-tensor
        self._align_pad_tensors(src)
        # align dst-tensor to avoid conflict
        self.schedule[tensor].storage_align(tensor.op.axis[b_idx], dst, 0)

        # while last-transpose need update vnchwconv-align
        if perm[-1] != len(perm) - 1:
            self.do_vnchwconv_align(tensor, dst)

    def _vor_nc1hwc0_mode(self, tensor):
        """
        Eg: (H,C1,C0)->(C1,H,C0) has two ways:
        1. Liner Read and Interval Write
        2. Interval Read and Liner Write
        Func: Only support 5HD&&NZ
        """
        idx_c1, idx_h = self.graph_info.c1c0[0], self.graph_info.c1c0[0] + 1
        self._avoid_conflict_nc1hwc0(tensor, self.optimization_vor, idx_h, idx_c1)

    def _vnc_nc1hwc0_mode(self, tensor):
        """
        Eg: (C1,C0,H)->(C1,H,C0) has two ways:
        1. Liner Read and Interval Write
        2. Interval Read and Liner Write
        Func: Only support 5HD&&NZ
        """
        idx_c0, idx_h = self.graph_info.c1c0[1], self.graph_info.c1c0[1] - 1
        self._avoid_conflict_nc1hwc0(tensor, self.optimization_vnc, idx_c0, idx_h)

    def _avoid_bank_conflict(self):
        """
        Func: avoid bank-conflict in n-last-transpose that used VOR.
              avoid bank-conflict in last-transpose that used VNCHWCONV.
        """
        if len(self.transpose_tensors) != 1:
            return
        if self.tiling_case.transpose_work == 0:
            return
        if self.tiling_case.shape_type == COMMON_ALIGN:
            return
        if self.tiling_case.avoid_bank_conflict == 0:
            return

        for tensor, perm in zip(self.transpose_tensors, self.permutes):
            _, insn, _ = self.choose_transpose_insn(tensor.dtype, perm)
            if insn == "vector_or":
                self._vor_nc1hwc0_mode(tensor)
            elif insn == "vector_transpose" and DTYPE_BYTE_MAPPING.get(tensor.dtype, 1) in [B8, B16]:
                self._vnc_nc1hwc0_mode(tensor)

    def _do_storage_align(self):
        self.dtype = self.tiling_tensor.dtype
        self.align_factor = BLOCK // DTYPE_BYTE_MAPPING.get(self.dtype, 1)
        # Last-dim need support align and pad, choose bigger
        last_dim_is_pad_axis = self.pad_axis_list[0] == len(self.mte2_tensor.shape) - 1
        factor = self.align_factor
        if last_dim_is_pad_axis and factor < self.pad_factor:
            factor = self.pad_factor

        self._align_mte2_tensors(factor)
        self._align_pad_tensors(factor)
        self._align_transpose_tensors()
        self._avoid_bank_conflict()

    def _do_mem_reused(self):
        def reused_by(src, dst):
            src_tensor = src
            while src_tensor != dst:
                dst_tensor = list(self.forward_stage_graph_map.get(src_tensor))[0]
                self.schedule[src_tensor].reused_by(dst_tensor)
                if src_tensor in self.graph_info.pad_tensor_set:
                    self.schedule[src_tensor].set_store_predicate(src_tensor.op.body[0].condition)
                src_tensor = dst_tensor

        target = self.reshape_tensors[-1]
        source = self.align_pad_tensor if self.tiling_case.shape_type == COMMON_ALIGN else self.mte2_tensor
        reused_by(source, target)

    def _transpose_emit_insn(self, tensor, perm):
        emit_idx, insn, perm = self.choose_transpose_insn(tensor.dtype, perm)
        if insn in ["vector_transpose", "phony_insn"]:
            self.vnchwconv_insn_map(tensor, insn, tensor.op.axis[emit_idx], perm)
        else:
            self.emit_insn_map[tensor] = {"scope": tensor.op.axis[emit_idx], "instruction": insn}
            if insn in ["vector_or", ]:
                self.emit_insn_map[tensor].update(dict(attrs={"transpose_opt": 1}))

    def _calc_emit_insn(self):
        self.emit_insn_map.clear()
        self.emit_insn_map[self.mte2_tensor] = {"scope": self.mte2_tensor.op.axis[0], "instruction": "dma_copy",
                                                "attrs": {"gm_to_ub_gap_opt": 1}}

        if self.tiling_case.shape_type == COMMON_ALIGN:
            self.common_align_insn_map(self.align_pad_tensor, "align_pad", self.align_pad_tensor.op.axis[0])

        for tensor in self.pad_tensors:
            self.emit_insn_map.update({tensor: {"scope": tensor.op.axis[0], "instruction": "vector_dup"}})

        for tensor in self.reshape_tensors:
            self.emit_insn_map.update({tensor: {"scope": tensor.op.axis[0], "instruction": "phony_insn"}})

        for tensor, perm in zip(self.transpose_tensors, self.permutes):
            self._transpose_emit_insn(tensor, perm)

        self.emit_insn_map[self.tiling_tensor] = {"scope": self.ub_inner, "instruction": "dma_copy",
                                                  "attrs": {NO_OVERLAP: 2, "no_overlap_malloc_buf_for_tail": 0}}

    def _analysis_transpose(self, ):
        work = self.tiling_case.transpose_work
        self.permutes = [self.update_ub_perm(x, transpose_work=work) for x in self.transpose_tensors]
