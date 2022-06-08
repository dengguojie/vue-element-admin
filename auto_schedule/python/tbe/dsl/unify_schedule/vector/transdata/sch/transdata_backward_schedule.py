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
transdata backward schedule
"""
import copy
from tbe.dsl.base.operation import var_inner
from tbe.dsl.unify_schedule.constants import TransdataCategory
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils.shape_util import shape_to_list
from ..common.transdata_base_sch import TransdataBaseSch
from ..common.transdata_base_sch import get_block_size
from ..common.transdata_base_sch import INT8_BLOCK, FP16_BLOCK, FP32_ALIGN_SIZE
from ..common.constants import STORAGE_ALIGN, COMMON_ALIGN
from ..common.constants import NO_OVERLAP, STRIDE_2, B8, B16


class TransBackwardSchedule(TransdataBaseSch):
    """Perform rank conversion in the most direct way.
    It decomposes complex rank conversion operations through
    the simply rank conversion of adjacent axes. In the schedule,
    complex-transpose = last-transpose + n_last_transpose
    """

    def __init__(self, outs, tiling_case):
        TransdataBaseSch.__init__(self, outs, tiling_case)
        self.mapping_indexes = []
        self.handle_tensors = []
        self.c0_index = None
        self.c1_index = None
        self.ori_tiling_case = None

    @classmethod
    def get_supported_sub_pattern(cls):
        return TransdataCategory.GENERAL_BACKWARD

    def do_schedule(self):
        """
        Process of schedule
        """
        self._analysis_case()
        self._create_schedule()
        self._do_cache_read()
        self._do_set_scope()
        self._init_tensors()

        self._calc_tiling()
        self._do_tiling()

        self._analysis_transpose()

        self._do_buffer_align()
        self._do_storage_align()
        self._do_storage_bound()
        self._do_mem_reused()

        self._calc_multi_core()
        self._do_multi_core()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._do_pragma()
        self.schedule.tiling_key = self.tiling_case.tiling_key
        return self.schedule

    def _analysis_case(self, ):
        if self.tiling_case.shape_type not in [COMMON_ALIGN, STORAGE_ALIGN]:
            dict_args = {"errCode": "E90003", "detailed_cause": "branch is error."}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def _init_tensors(self):
        """
        StorageAlign Process:
        input -> mte2_tensor -> transpose_tensor -> reshape_tensor -> depad_tensor -> output(depad)
        CommonAlign Process:
        input -> mte2_tensor -> transpose_tensor -> reshape_tensor -> depad_tensor -> remove_pad -> output(depad)
        """
        self.depad_tensors = list(self.graph_info.de_pad_tensor_set)
        self.depad_tensors.sort(key=lambda x: int(x.op.attrs["axes"]))
        self.depad_axis_list = [int(x.op.attrs["axes"]) for x in self.depad_tensors]

        self.transpose_tensors = list(self.graph_info.transpose_tensor_set)
        self.transpose_tensors.sort(key=lambda x: int(x.op.attrs["order"]))

        self.reshape_tensors = list(self.graph_info.f_reshape_tensor_set)
        self.reshape_tensors.sort(key=lambda x: int(x.op.attrs["order"]))

        self.tiling_tensor = self.depad_tensors[-1]
        self.mte2_tensor = self.child(list(self.graph_info.input_tensor_set)[0])

        # eg: [N,C,H], remove_pad work in H, ub_2_ub work in C
        if self.tiling_case.shape_type == COMMON_ALIGN:
            if self.depad_axis_list[-1] != len(self.tiling_tensor.shape) - 1:
                self._do_cache_write()
                self.depad_tensors[-1] = self.parent(self.tiling_tensor)
                self.remove_pad_tensor = self.single_cache_read(self.depad_tensors[-1])
            else:
                self._do_cache_write()
                self.remove_pad_tensor = self.parent(self.tiling_tensor)
        # remove tiling_tensor from depad_tensors
        if self.depad_tensors[-1] == self.tiling_tensor:
            self.depad_tensors = self.depad_tensors[:-1]
            self.depad_axis_list = self.depad_axis_list[:-1]

    def _calc_tiling(self):
        # define factor
        self.dtype = self.tiling_tensor.dtype
        self.align_factor = get_block_size(self.dtype)
        self.split_once = self.tiling_case.ub_split_second_idx == self.tiling_case.ub_split_first_idx
        if not self.tiling_case.ub_first_factor:
            self.tiling_case.ub_first_factor = var_inner("_ub_first_factor", (1, None))
        if not self.split_once and not self.tiling_case.ub_second_factor:
            self.tiling_case.ub_second_factor = var_inner("_ub_second_factor", (1, None))
        if not self.tiling_case.block_factor:
            self.tiling_case.block_factor = var_inner("_block_factor", (1, None))
        self.ori_tiling_case = copy.copy(self.tiling_case)

        # regulation
        self.parses_axis_type(self.graph_info.permute)
        self.mapping_indexes, self.c1_index, self.c0_index = self.parses_f_reshape(self.reshape_tensors[-1])
        self.parses_tiling_info(self.mapping_indexes, self.c1_index, self.c0_index, self.pad_factor)

    def _do_tiling(self):
        self._do_ub_tiling()
        self._do_block_tiling()
        self._do_reorder()
        self._do_fragments()

    def _do_buffer_align(self):
        """
        Input is [X0,X1,X2,C0], buffer_align reshape to assure mte2.burst_len is x*C0
        """

        def align(tensor, _axis_list, _factor):
            align_list = [[1, 1] for x in range(len(tensor.shape))]
            for i in _axis_list:
                align_list[i] = [1, _factor]
            self.schedule[tensor].buffer_align(*align_list)

        # deal reshape
        if self.c1_index is not None:
            c_idx = self.mapping_indexes[self.c1_index]
            align(self.reshape_tensors[-1], [c_idx, ], self.pad_factor)
        else:
            if not self.depad_axis_list or self.depad_axis_list[-1] == len(self.reshape_tensors[-1].shape) - 1:
                align(self.reshape_tensors[-1], [-1, ], self.pad_factor)

    def _ub_internal_length(self, index, length):
        if index == self.ori_tiling_case.ub_split_first_idx:
            return self.ori_tiling_case.ub_first_factor
        if index == self.ori_tiling_case.ub_split_second_idx:
            return self.ori_tiling_case.ub_second_factor
        return length

    def _avoid_conflict_nc1hwc0(self, tensor, optimize, a_idx, b_idx):
        """
        Transpose [a,b] to [b,a]
        :param optimize: The func that avoid conflict in transpose instructions.
        :param a_idx: a's index in [b,a]
        :param b_idx: b's index in [b,a]
        :constraint: Support 5HD and NZ.
        """
        # a: Left var in transpose-mode [a,b] -> [b,a]
        # b: Right var in transpose-mode [a,b] -> [b,a]
        # c0: align value
        shape = shape_to_list(tensor.shape)
        a = self._ub_internal_length(a_idx, shape[a_idx])
        b = self._ub_internal_length(b_idx, shape[b_idx])
        c0 = shape[self.graph_info.c1c0[1]]

        # handle tensors which to solve conflict
        handle_after_tensor = self.single_cache_read(tensor)
        handle_before_tensor = self.single_cache_read(self.mte2_tensor)
        self.handle_tensors.extend([handle_before_tensor, handle_after_tensor])
        src, dst = optimize(a, b, c0, tensor)
        perm = [int(x) for x in list(tensor.op.attrs["permute"])]

        # align src-tensor to avoid conflict
        self.schedule[handle_before_tensor].storage_align(handle_before_tensor.op.axis[perm[a_idx]], src, 0)
        # align dst-tensor to avoid conflict
        self.schedule[tensor].storage_align(tensor.op.axis[b_idx], dst, 0)
        # while last-transpose need update vnchwconv-align
        if perm[-1] != len(perm) - 1:
            self.do_vnchwconv_align(tensor, dst)
        # backward need deal with gap that create in _avoid_conflict_nc1hwc0
        self.schedule[handle_after_tensor].storage_align(handle_after_tensor.op.axis[-2], c0, 0)

    def _vor_nc1hwc0_mode(self, tensor):
        """
        Eg: (C1,H,C0)->(H,C1,C0) has two ways:
            1. Liner Read and Interval Write
            2. Interval Read and Liner Write
        """
        idx_c1, idx_h = self.graph_info.c1c0[0], self.graph_info.c1c0[0] - 1
        self._avoid_conflict_nc1hwc0(tensor, self.optimization_vor, idx_c1, idx_h)

    def _vnc_nc1hwc0_mode(self, tensor):
        """
        Eg: (C1,H,C0)->(C1,C0,H) has two ways:
            1. Liner Read and Interval Write
            2. Interval Read and Liner Write
        """
        idx_h, idx_c0 = self.graph_info.c1c0[1] + 1, self.graph_info.c1c0[1]
        self._avoid_conflict_nc1hwc0(tensor, self.optimization_vnc, idx_h, idx_c0)

    def _avoid_bank_conflict(self):
        """
        Func: avoid bank-conflict in n-last-transpose that used VOR.
              avoid bank-conflict in last-transpose that used VNCHWCONV.
        """
        if len(self.transpose_tensors) != 1:
            return
        if self.tiling_case.transpose_work == 0:
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
        for tensor, perm in zip(self.transpose_tensors, self.permutes):
            if perm and perm[-1] != len(perm) - 1:
                self.do_vnchwconv_align(tensor, self.align_factor)

        for tensor in self.get_all_producers_stages(self.tiling_tensor):
            if tensor in self.transpose_tensors:
                continue
            if tensor in self.graph_info.input_tensor_set:
                continue
            if tensor in [self.mte2_tensor, self.remove_pad_tensor]:
                continue
            # tensors after transpose
            if len(tensor.shape) >= STRIDE_2:
                align_factor = self.align_factor
                if self.graph_info.is_last_transpose:
                    if get_block_size(tensor.dtype) not in [INT8_BLOCK, FP16_BLOCK]:
                        align_factor = FP32_ALIGN_SIZE
                self.schedule[tensor].storage_align(tensor.op.axis[-2], align_factor, 0)

        self._avoid_bank_conflict()

    def _do_mem_reused(self):
        source_tensor = self.handle_tensors[-1] if self.handle_tensors else self.transpose_tensors[-1]
        for tensor in self.reshape_tensors:
            self.schedule[source_tensor].reused_by(tensor)

    def _do_pragma(self):
        # occur in last-transpose, depad work in dma: need depad work in ub (wait)
        axes = int(self.tiling_tensor.op.attrs["axes"])
        if axes == len(self.tiling_tensor.shape) - 1:
            idx = self.reorder_list.index(self.ub_inner)
            axes = self.reorder_list[idx:][:-1]
            if axes:
                self._do_group(self.tiling_tensor, axes)

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
        self.emit_insn_map[self.mte2_tensor] = {"scope": self.mte2_tensor.op.axis[0], "instruction": "dma_copy"}
        for tensor in self.reshape_tensors:
            self.emit_insn_map[tensor] = {"scope": tensor.op.axis[0], "instruction": "phony_insn"}
        for tensor, perm in zip(self.transpose_tensors, self.permutes):
            self._transpose_emit_insn(tensor, perm)
        for tensor in self.depad_tensors:
            self.emit_insn_map[tensor] = {"scope": tensor.op.axis[0], "instruction": "dma_copy"}

        model = 3 if self.tiling_case.shape_type != COMMON_ALIGN else 2
        self.emit_insn_map[self.tiling_tensor] = {"scope": self.ub_inner, "instruction": "dma_copy",
                                                  "attrs": {NO_OVERLAP: model, "no_overlap_malloc_buf_for_tail": 0}}

        if self.tiling_case.shape_type == COMMON_ALIGN:
            self.common_align_insn_map(self.remove_pad_tensor, "remove_pad", self.remove_pad_tensor.op.axis[0])

        # extra
        for tensor in self.handle_tensors:
            self.emit_insn_map[tensor] = {"scope": tensor.op.axis[0], "instruction": "dma_copy"}

    def _analysis_transpose(self):
        work = self.tiling_case.transpose_work
        self.permutes = [self.update_ub_perm(x, transpose_work=work) for x in self.transpose_tensors]
