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
transdata forward schedule
"""
import operator
from functools import reduce

from tbe import tvm
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import var_inner
from tbe.dsl.unify_schedule.constants import TransdataCategory
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from .transdata_base_schedule import TransdataBaseSchedule

STORAGE_ALIGN = "storage_align"
COMMON_ALIGN = "common_align"
NO_OVERLAP = "no_overlap"
BLOCK = 32
FP32_ALIGN_SIZE = 128
FP16_BLOCK = 16
FP32_BLOCK = 8
INT8_BLOCK = 32


class TransForwardSchedule(TransdataBaseSchedule):
    """
    TransForwardSchedule: base + forward schedule
    """

    def __init__(self, outs, tiling_case):
        TransdataBaseSchedule.__init__(self, outs, tiling_case)
        self.iter_block_outer = None
        self.iter_block_inner = None
        self.iter_ub_first_outer = None
        self.iter_ub_first_inner = None
        self.iter_ub_second_outer = None
        self.iter_ub_second_inner = None
        self.ub_outer = None
        self.ub_inner = None
        self.split_once = False

        self.tiling_tensor = None
        self.tiling_axes = []
        self.reorder_list = []

        # different branch
        self.branch = None
        self.mte2_tensor = None
        self.transpose_tensor = None
        self.reshape_tensor = None
        self.align_tensor = None
        self.pad_axis_list = []
        self.permute = []
        self.is_last_transpose = False
        self.is_const = False

        self.align_factor = None
        self.pad_factor = None
        self.dtype = None

    @classmethod
    def get_supported_sub_pattern(cls):
        return TransdataCategory.GENERAL_FORWARD

    def _analysis_case(self, ):
        if self.tiling_case.shape_type == 0:
            self.branch = STORAGE_ALIGN
        elif self.tiling_case.shape_type == 1:
            self.branch = COMMON_ALIGN

    def _init_tensors(self):
        """
        StorageAlign Process:
        input -> mte2_tensor -> pad_tensor -> reshape_tensor -> transpose_tensor -> output
        CommonAlign Process:
        input -> mte2_tensor -> align_tensor -> pad_tensor -> reshape_tensor -> transpose_tensor -> out
        """
        self.tiling_tensor = list(self.graph_info.output_tensor_set)[0]
        self.mte2_tensor = self.cache_read_tensors_and_buffer_map.get(list(self.graph_info.input_tensor_set)[0])
        self.transpose_tensor = self.cache_write_tensors_and_buffer_map.get(self.tiling_tensor)
        self.reshape_tensor = list(self.graph_info.s_reshape_tensor_set)[0]

        if self.branch == COMMON_ALIGN:
            readers = list(self.forward_stage_graph_map[self.mte2_tensor])
            self.align_tensor = self.schedule.cache_read(self.mte2_tensor, "local.UB", readers)
            self.cache_read_tensors_and_buffer_map[self.mte2_tensor] = self.align_tensor
            for item in readers:
                self.update_stage(self.align_tensor, item, True)

        # get pad axis
        for tensor in self.graph_info.pad_tensor_set:
            self.pad_axis_list.extend([int(x) for x in tensor.op.attrs["axes"]])
        self.pad_axis_list.sort(reverse=True)

        self.is_last_transpose = self.graph_info.is_last_transpose

    def _analysis_transpose_operator(self):
        # 1. calc permute in ub
        # 2. deal axis that value is 1
        idx = self.reorder_list.index(self.ub_inner)
        num = len(self.reorder_list) - idx
        ori_permute = self.graph_info.permute[len(self.graph_info.permute) - num:]
        back = sorted(ori_permute.copy())
        for i in ori_permute:
            self.permute.append(back.index(i))

        if back == self.permute:
            # pure_data_move
            self.is_last_transpose = False

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

        self._analysis_transpose_operator()

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

    def _calc_tiling(self):
        case = self.tiling_case
        self.tiling_axes = [x for x in self.tiling_tensor.op.axis]
        self.split_once = case.ub_split_second_idx == case.ub_split_first_idx

        case.block_factor = case.block_factor if case.block_factor else var_inner("_block_factor", (1, None))
        case.ub_first_factor = \
            case.ub_first_factor if case.ub_first_factor else var_inner("_ub_first_factor", (1, None))
        if not self.split_once:
            case.ub_second_factor = \
                case.ub_second_factor if case.ub_second_factor else var_inner("_ub_second_factor", (1, None))

    def _do_tiling(self):
        self._do_ub_tiling()
        self._do_block_tiling()
        self._do_reorder()
        self.schedule[self.tiling_tensor].pragma(self.iter_block_inner, "local.UB_fragments_memory_size", 256)

    def _do_ub_tiling(self):
        first_factor = self.tiling_case.ub_first_factor
        second_factor = self.tiling_case.ub_second_factor

        # first ub tiling
        first_axis_var = self.tiling_axes[self.tiling_case.ub_split_first_idx]
        self.iter_ub_first_outer, self.iter_ub_first_inner = \
            self.schedule[self.tiling_tensor].split(first_axis_var, factor=first_factor)

        # second ub tiling
        if not self.split_once:
            sec_axis_var = self.tiling_axes[self.tiling_case.ub_split_second_idx]
            self.iter_ub_second_outer, self.iter_ub_second_inner = \
                self.schedule[self.tiling_tensor].split(sec_axis_var, factor=second_factor)

    def _do_block_tiling(self):
        """
        block tiling only split axis which belong to outsiders of ub
        """
        case = self.tiling_case
        if case.block_split_idx == case.ub_split_first_idx:
            tiling_axis_var = self.iter_ub_first_outer
        elif not self.split_once and case.block_split_idx == case.ub_split_second_idx:
            tiling_axis_var = self.iter_ub_second_outer
        else:
            tiling_axis_var = self.tiling_axes[case.block_split_idx]

        self.iter_block_outer, self.iter_block_inner = \
            self.schedule[self.tiling_tensor].split(tiling_axis_var, factor=case.block_factor)

    def _do_reorder(self):
        """
        input_inner: axis of input_tensor that belong to internal of UB
        output_inner: axis of output_tensor that belong to internal of UB
        outer: axis of output_tensor that belong to External of UB
        """
        case = self.tiling_case
        perm = self.graph_info.permute
        length = len(self.tiling_axes)
        split_i = case.ub_split_first_idx
        split_o = case.ub_split_second_idx
        split_b = case.block_split_idx

        input_inner = {perm.index(x) for x in range(perm[split_i], length, 1)}
        output_inner = set(range(split_o, length, 1))
        output_inner = output_inner.union(input_inner)
        outer = set(range(length)).difference(output_inner)

        # axis in ub
        output_inner = list(output_inner)
        output_inner.sort()
        for idx in output_inner:
            if idx == split_i:
                self.reorder_list.append(self.iter_ub_first_inner)
            elif idx == split_o:
                self.reorder_list.append(self.iter_ub_second_inner)
            else:
                self.reorder_list.append(self.tiling_axes[idx])

        # axis outside ub
        outer.add(split_i)
        outer.add(split_o)
        outer = list(outer)
        outer.sort()
        outside = []
        for idx in outer:
            if idx == split_b:
                outside.extend([self.iter_block_outer, self.iter_block_inner])
            elif idx == split_i:
                outside.append(self.iter_ub_first_outer)
            elif idx == split_o:
                outside.append(self.iter_ub_second_outer)
            else:
                outside.append(self.tiling_axes[idx])

        self.ub_outer = outside[-1]
        self.ub_inner = self.reorder_list[0]
        self.reorder_list = outside + self.reorder_list
        self.schedule[self.tiling_tensor].reorder(*self.reorder_list)

        # While n_last_transpose and forward
        # NHC1C0 -> NC1HC0, H more likely than C1,
        # swaps(C1,H) help vector_or more effective.
        if not self.graph_info.is_last_transpose:
            _reorder = [self.transpose_tensor.op.axis[x] for x in self.graph_info.permute]
            self.schedule[self.transpose_tensor].reorder(*_reorder)

    def _do_storage_bound(self):
        for stage_tensor in self.forward_stage_graph_map:
            ub_count = self.tiling_case.tensor_ub_size_list[self.tiling_case.shape_type]
            self.schedule[stage_tensor].set_buffer_size(ub_count)

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
        """
        Pad will make these models while in common_align
        1. NHC: (NH,C)
        2. NCH: (N,CH)
        3. NhC: (N,hC)
        """
        tensor = self.mte2_tensor if self.branch == STORAGE_ALIGN else self.align_tensor
        if len(tensor.op.axis) >= 2:
            self.schedule[tensor].storage_align(tensor.op.axis[-2], factor, 0)

    def _align_pad_tensors(self, factor):
        for tensor in self.graph_info.pad_tensor_set:
            if len(tensor.op.axis) >= 2:
                self.schedule[tensor].storage_align(tensor.op.axis[-2], factor, 0)

        # Operation of padding not work in pad_tensors
        tensors = [self.mte2_tensor, ] if self.branch == STORAGE_ALIGN else [self.mte2_tensor, self.align_tensor]
        shape = list(tensors[0].shape)
        shape[-1] = _set_align(shape[-1], factor)

        for i in self.pad_axis_list:
            if i == 0 or i == len(tensors[0].shape) - 1:
                # index is 0: can not storage align
                # index is -1: had been storage align and mte2_tensor
                # not align in common_align
                continue

            _factor = _math_prod(shape[i + 1:]) * self.pad_factor
            shape[i] = _set_align(shape[i], self.pad_factor)
            for tensor in tensors:
                self.schedule[tensor].storage_align(tensor.op.axis[i - 1], _factor, 0)

    def _align_transpose_tensors(self, factor):
        # A <Transpose> B, align for A and B
        # 1. align for mte
        # 2. align for vector_transpose
        idx = self.graph_info.permute.index(len(self.reshape_tensor.shape) - 1)
        if idx != 0:
            shape = list(self.transpose_tensor.shape)
            _factor = _math_prod(shape[idx + 1:]) * factor
            self.schedule[self.transpose_tensor].storage_align(self.transpose_tensor.op.axis[idx - 1], _factor, 0)
        if len(self.reshape_tensor.op.axis) >= 2:
            self.schedule[self.reshape_tensor].storage_align(self.reshape_tensor.op.axis[-2], factor, 0)

        # n-last-transpose, c0 is last dim, don't need storage align
        # last-transpose, (m,n) -> (n,m), C0 must be m or n
        # last-transpose, (m,n) -> (n,m), storage align dst that make dst had space to store
        # In 5HD(NC1C0H->NC1HC0), fp16, fp8 don't need storage align
        # In 5HD(NC1C0H->NC1HC0), fp32, int32 need C0 storage align 128
        if self.is_last_transpose:
            block_size = BLOCK // DTYPE_BYTE_MAPPING.get(self.dtype, 1)
            # avoid H is 1 while const
            self.schedule[self.transpose_tensor].compute_align(self.transpose_tensor.op.axis[-2], block_size)
            if block_size in [FP16_BLOCK, INT8_BLOCK]:
                return
            self.schedule[self.transpose_tensor].storage_align(self.transpose_tensor.op.axis[-2], FP32_ALIGN_SIZE, 0)

    def _do_storage_align(self):
        self.dtype = self.tiling_tensor.dtype
        self.align_factor = BLOCK // DTYPE_BYTE_MAPPING.get(self.dtype, 1)
        self.pad_factor = get_context().get_current_compute().get("_pad_factor")

        last_dim_is_pad_axis = self.pad_axis_list[0] == len(self.mte2_tensor.shape) - 1
        factor = self.align_factor
        if last_dim_is_pad_axis and factor < self.pad_factor:
            factor = self.pad_factor

        self._align_mte2_tensors(factor)
        self._align_pad_tensors(factor)
        self._align_transpose_tensors(factor)

    def _do_mem_reused(self):
        def reused_by(_begin, _end):
            src_tensor = _begin
            while src_tensor != _end:
                dst_tensor = list(self.forward_stage_graph_map.get(src_tensor))[0]
                self.schedule[src_tensor].reused_by(dst_tensor)
                if src_tensor in self.graph_info.pad_tensor_set:
                    self.schedule[src_tensor].set_store_predicate(src_tensor.op.body[0].condition)
                src_tensor = dst_tensor

        if self.branch == COMMON_ALIGN:
            reused_by(self.align_tensor, self.reshape_tensor)
        else:
            reused_by(self.mte2_tensor, self.reshape_tensor)

    def _calc_multi_core(self):
        if self.need_multi_core:
            idx = self.reorder_list.index(self.iter_block_outer)
            fused_list = self.reorder_list[:idx + 1]
            fused_axis = self.schedule[self.tiling_tensor].fuse(*fused_list)
            self.multi_core_fused_axis = fused_axis
            self.multi_core_bind_tensor = self.tiling_tensor

    def _calc_compute_at(self):
        self.compute_at_map.clear()
        for tensor in self.get_all_producers_stages(self.tiling_tensor):
            if tensor not in self.graph_info.input_tensor_set:
                self.compute_at_map[tensor] = {"parent": self.schedule[self.tiling_tensor], "scope": self.ub_outer}

    def _transpose_emit_insn(self):
        emit_idx, insn = self._calc_permute_in_ub()
        if insn == "vector_transpose":
            iter = self.transpose_tensor.op.axis[emit_idx]
            src_in_dst_order = tvm.expr.Call('handle', 'tvm_tuple', self.permute, tvm.expr.Call.PureIntrinsic, None, 0)
            self.emit_insn_map[self.transpose_tensor] = {"scope": iter, "instruction": "vector_transpose",
                                                         "src_in_dst_order": src_in_dst_order}
        elif insn in ["vector_or", "dma_copy"]:
            self.emit_insn_map[self.transpose_tensor] = {"scope": self.transpose_tensor.op.axis[emit_idx],
                                                         "instruction": insn}

    def _calc_emit_insn(self):
        self.emit_insn_map.clear()
        pad_tensors_list = list(self.graph_info.pad_tensor_set)

        if self.branch == COMMON_ALIGN:
            self.emit_insn_map[self.mte2_tensor] = {"scope": self.mte2_tensor.op.axis[0], "instruction": "dma_copy"}
            self.emit_insn_map[self.align_tensor] = {"scope": self.align_tensor.op.axis[0],
                                                     "instruction": "align_pad"}

            for tensor in pad_tensors_list:
                self.emit_insn_map.update({tensor: {"scope": tensor.op.axis[0], "instruction": "vector_dup"}})

            self.emit_insn_map[self.reshape_tensor] = {"scope": self.reshape_tensor.op.axis[0],
                                                       "instruction": "phony_insn"}

            self._transpose_emit_insn()

            self.emit_insn_map[self.tiling_tensor] = {"scope": self.ub_inner, "instruction": "dma_copy",
                                                      NO_OVERLAP: {NO_OVERLAP: 3,
                                                                   "no_overlap_malloc_buf_for_tail": 0}}

        elif self.branch == STORAGE_ALIGN:
            self.emit_insn_map[self.mte2_tensor] = {"scope": self.mte2_tensor.op.axis[0], "instruction": "dma_copy"}

            for tensor in pad_tensors_list:
                self.emit_insn_map.update({tensor: {"scope": tensor.op.axis[0], "instruction": "vector_dup"}})

            self.emit_insn_map[self.reshape_tensor] = {"scope": self.reshape_tensor.op.axis[0],
                                                       "instruction": "phony_insn"}

            self._transpose_emit_insn()

            self.emit_insn_map[self.tiling_tensor] = {"scope": self.ub_inner, "instruction": "dma_copy",
                                                      NO_OVERLAP: {NO_OVERLAP: 3,
                                                                   "no_overlap_malloc_buf_for_tail": 0}}

    def _calc_permute_in_ub(self):
        if self.is_last_transpose:
            insn = "vector_transpose"
            if BLOCK // DTYPE_BYTE_MAPPING.get(self.dtype, 1) in [FP16_BLOCK, INT8_BLOCK]:
                emit_idx = 0
            else:
                emit_idx = -2
                back = self.permute[-2:].copy()
                min_value = min(self.permute)
                self.permute = [x - min_value for x in back]
        else:
            emit_idx = 0
            if BLOCK // DTYPE_BYTE_MAPPING.get(self.dtype, 1) in [INT8_BLOCK, ]:
                insn = "dma_copy"
            else:
                insn = "vector_or"

        return emit_idx, insn


def _set_align(dim, factor):
    return tvm.floordiv(dim + factor - 1, factor) * factor


def _math_prod(iterable):
    return reduce(operator.mul, iterable, 1)
