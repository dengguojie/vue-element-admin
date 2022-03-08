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
transdata backward schedule
"""
from tbe import tvm
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import var_inner
from tbe.dsl.unify_schedule.constants import TransdataCategory
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from .transdata_base_schedule import TransdataBaseSchedule

NO_OVERLAP = "no_overlap"
STORAGE_ALIGN = "storage_align"
COMMON_ALIGN = "common_align"
BLOCK = 32
FP32_ALIGN_SIZE = 128
FP16_BLOCK = 16
FP32_BLOCK = 8
INT8_BLOCK = 32


class TransBackwardSchedule(TransdataBaseSchedule):
    """
    TransBackwardSchedule: base + bcakward schedule
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
        self.align_factor = None
        self.pad_factor = None
        self.dtype = None

        self.split_once = False
        self.is_do_extra_split_c = False

        self.tiling_tensor = None
        self.mte2_tensor = None
        self.transpose_tensor = None
        self.reshape_tensor = None
        self.depad_tensors = []
        self.tiling_axes = []
        self.reorder_list = []

        self.mapping_indexes = []
        self.c0_index = None
        self.c1_index = None
        self.c0_axis = None
        self.c1_axis = None
        self.axis_in_ub = []
        self.axis_not_in_ub = []
        self.permute = []

        # different branch
        self.branch = None
        self.remove_pad_tensor = None
        self.is_last_transpose = None
        self.depad_axis_list = []

    @classmethod
    def get_supported_sub_pattern(cls):
        return TransdataCategory.GENERAL_BACKWARD

    def _analysis_case(self, ):
        if self.tiling_case.shape_type == 0:
            self.branch = STORAGE_ALIGN
        elif self.tiling_case.shape_type == 1:
            self.branch = COMMON_ALIGN

    def _init_tensors(self):
        """
        StorageAlign Process:
        input -> mte2_tensor -> transpose_tensor -> reshape_tensor -> depad_tensor -> output(depad)
        CommonAlign Process:
        input -> mte2_tensor -> transpose_tensor -> reshape_tensor -> depad_tensor -> remove_pad -> output(depad)
        """
        self.tiling_tensor = list(self.graph_info.output_tensor_set)[0]
        self.mte2_tensor = self.cache_read_tensors_and_buffer_map.get(list(self.graph_info.input_tensor_set)[0])
        self.transpose_tensor = list(self.graph_info.transpose_tensor_set)[0]
        self.reshape_tensor = list(self.graph_info.f_reshape_tensor_set)[0]
        self.is_last_transpose = self.graph_info.is_last_transpose

        # get pad axis
        self.depad_tensors = list(self.graph_info.de_pad_tensor_set)
        self.depad_tensors.sort(key=lambda x: int(x.op.attrs["axes"]))
        self.depad_axis_list = [int(x.op.attrs["axes"]) for x in self.depad_tensors]

        if self.branch == COMMON_ALIGN:
            if self.depad_axis_list[-1] != len(self.tiling_tensor.shape) - 1:
                # eg: [N,C,H], remove_pad work in H, MTE3 work in C
                self._do_cache_write()
                last_depad_tensor = list(self.backward_stage_graph_map[self.tiling_tensor])[0]
                self.depad_tensors.append(last_depad_tensor)
                self.depad_axis_list.append(self.depad_axis_list[-1])

                readers = list(self.forward_stage_graph_map[last_depad_tensor])
                self.remove_pad_tensor = self.schedule.cache_read(last_depad_tensor, "local.UB", readers)
                self.cache_read_tensors_and_buffer_map[last_depad_tensor] = self.remove_pad_tensor
                for item in readers:
                    self.update_stage(self.remove_pad_tensor, item, True)
            else:
                self._do_cache_write()
                self.remove_pad_tensor = list(self.backward_stage_graph_map[self.tiling_tensor])[0]

    def _analysis_transpose_operator(self):
        # 1. calc permute in ub
        # 2. deal axis that value is 1
        idx = self.reorder_list.index(self.ub_inner)
        num = len(self.reorder_list) - idx
        # output mapping input
        length = len(self.graph_info.reshape)
        for index in range(length - 1, length - num - 1, -1):
            value = self.graph_info.reshape[index]
            if isinstance(value, (list, tuple)):
                num += len(value) - 1

        ori_permute = self.graph_info.permute[len(self.graph_info.permute) - num:]
        back = sorted(ori_permute.copy())
        for i in ori_permute:
            self.permute.append(back.index(i))

        if back == self.permute:
            # pure_data_move
            self.is_last_transpose = False

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
        self._do_reorder()

        self._analysis_transpose_operator()

        self._do_buffer_align()
        self._do_storage_bound()
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
        self.dtype = self.tiling_tensor.dtype
        self.align_factor = BLOCK // DTYPE_BYTE_MAPPING.get(self.dtype, 1)
        self.pad_factor = get_context().get_current_compute().get("_pad_factor")
        self.split_once = self.tiling_case.ub_split_second_idx == self.tiling_case.ub_split_first_idx

        case = self.tiling_case
        perm = self.graph_info.permute
        reshape = self.graph_info.reshape
        length = len(perm)
        split_b = case.block_split_idx
        split_i = case.ub_split_first_idx
        split_o = case.ub_split_second_idx

        def parses_reshape():
            """
            Func:
            1. Find indexes of c1 and c0 in transpose_tensor
            2. Create mapping between transpose_tensor and output_tensor by indexes
            """
            for index, value in enumerate(reshape):
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    self.mapping_indexes.extend([index, ] * 2)
                    self.c1_index = value[0]
                    self.c0_index = value[1]
                else:
                    self.mapping_indexes.append(index)

        def parses_factor(_case):
            # define factor
            _case.block_factor = _case.block_factor if _case.block_factor else var_inner("_block_factor", (1, None))
            _case.ub_first_factor = \
                _case.ub_first_factor if _case.ub_first_factor else var_inner("_ub_first_factor", (1, None))
            if not self.split_once:
                _case.ub_second_factor = \
                    _case.ub_second_factor if _case.ub_second_factor else var_inner("_ub_second_factor", (1, None))

        def parses_split_one():
            """
            Func: Classified transpose_tensor's axis into UB Internal and UB External
            """
            ub_internal_input = {perm.index(x) for x in range(perm[split_i], length, 1)}
            ub_internal_output = set(range(split_o, length, 1))
            self.axis_in_ub = ub_internal_output.union(ub_internal_input)
            self.axis_not_in_ub = set(range(length)).difference(self.axis_in_ub)
            self.axis_not_in_ub = self.axis_not_in_ub.union({split_i, split_o})
            # for reorder
            self.axis_in_ub = list(self.axis_in_ub)
            self.axis_not_in_ub = list(self.axis_not_in_ub)
            self.axis_in_ub.sort()
            self.axis_not_in_ub.sort()

        def parses_split_two(_case):
            """
            Func:
            1. Do extra split for tiling_tensor(split C as c1 and c0)
            2. Collect tiling_axes
            3. Update axis_in_ub and axis_not_in_ub base on tiling_tensor
            """
            is_c1_out_ub = self.c1_index in self.axis_not_in_ub
            is_c0_in_ub = self.c0_index in self.axis_in_ub
            is_split_c1 = self.c1_index in [split_i, split_o]
            self.is_do_extra_split_c = is_c1_out_ub and is_c0_in_ub and not is_split_c1

            # update factor while split c1:
            # 1. split c1 mean that do not extra split c as c1 and c0
            # 2. ub split firstly, if ub split c1, factor is factor * 16(float16)
            # 3. block split secondly, if block split c1 and ub not split c1, factor is factor * 16(float16)
            if split_i == self.c1_index:
                _case.ub_first_factor *= self.pad_factor
            if not self.split_once and split_o == self.c1_index:
                _case.ub_second_factor *= self.pad_factor
            if split_b == self.c1_index and split_i != self.c1_index \
                    and split_o != self.c1_index and not self.is_do_extra_split_c:
                _case.block_factor *= self.pad_factor

            if self.is_do_extra_split_c:
                # do extra split
                root_var = self.tiling_tensor.op.axis[self.mapping_indexes[self.c1_index]]
                self.c1_axis, self.c0_axis = self.schedule[self.tiling_tensor].split(root_var,
                                                                                     factor=self.pad_factor)
                # collect tiling_axes
                for idx, value in enumerate(self.tiling_tensor.op.axis):
                    if idx == self.mapping_indexes[self.c1_index]:
                        self.tiling_axes.extend([self.c1_axis, self.c0_axis])
                    else:
                        self.tiling_axes.append(value)
            else:
                for value in self.tiling_tensor.op.axis:
                    self.tiling_axes.append(value)

                # update axis_in_ub and axis_not_in_ub that make them based on tiling_tensor
                _case.block_split_idx = self.mapping_indexes[split_b]
                _case.ub_split_first_idx = self.mapping_indexes[split_i]
                _case.ub_split_second_idx = self.mapping_indexes[split_o]
                if self.c1_index in self.axis_in_ub and self.c0_index in self.axis_in_ub:
                    self.axis_in_ub.remove(self.c1_index)
                if self.c1_index in self.axis_not_in_ub and self.c0_index in self.axis_not_in_ub:
                    self.axis_not_in_ub.remove(self.c0_index)
                self.axis_in_ub = [self.mapping_indexes[x] for x in self.axis_in_ub]
                self.axis_not_in_ub = [self.mapping_indexes[x] for x in self.axis_not_in_ub]

        parses_reshape()
        parses_factor(case)
        parses_split_one()
        parses_split_two(case)

    def _do_tiling(self):
        self._do_ub_tiling()
        self._do_block_tiling()
        self.schedule[self.tiling_tensor].pragma(self.iter_block_inner, "local.UB_fragments_memory_size", 256)

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

    def _do_ub_tiling(self):
        first_factor = self.tiling_case.ub_first_factor
        second_factor = self.tiling_case.ub_second_factor

        # first ub tiling
        first_axis_var = self.tiling_axes[self.tiling_case.ub_split_first_idx]
        self.iter_ub_first_outer, self.iter_ub_first_inner = \
            self.schedule[self.tiling_tensor].split(first_axis_var, factor=first_factor)

        # second ub tiling
        if not self.split_once:
            second_axis_var = self.tiling_axes[self.tiling_case.ub_split_second_idx]
            self.iter_ub_second_outer, self.iter_ub_second_inner = \
                self.schedule[self.tiling_tensor].split(second_axis_var, factor=second_factor)

    def _do_reorder(self):
        """
        Regulation: [D,E,C,B,A] ~ [D,E,C.outer,C.inner,B,A.outer,A.inner]
        if [D,E,B] belong to ub_outer, reorder is [D,E,C.outer,B,A.outer,C.inner,A.inner]
        """
        case = self.tiling_case
        split_i = case.ub_split_first_idx
        split_o = case.ub_split_second_idx
        split_b = case.block_split_idx

        for idx in self.axis_in_ub:
            if idx == split_i:
                self.reorder_list.append(self.iter_ub_first_inner)
            elif not self.split_once and idx == split_o:
                self.reorder_list.append(self.iter_ub_second_inner)
            else:
                self.reorder_list.append(self.tiling_axes[idx])

        outside = []
        for idx in self.axis_not_in_ub:
            if idx == split_b:
                outside.extend([self.iter_block_outer, self.iter_block_inner])
            elif idx == split_i:
                outside.append(self.iter_ub_first_outer)
            elif not self.split_once and idx == split_o:
                outside.append(self.iter_ub_second_outer)
            else:
                outside.append(self.tiling_axes[idx])

        self.ub_outer = outside[-1]
        self.ub_inner = self.reorder_list[0]
        self.reorder_list = outside + self.reorder_list
        self.schedule[self.tiling_tensor].reorder(*self.reorder_list)

    def _do_storage_bound(self):
        for stage_tensor in self.forward_stage_graph_map:
            ub_count = self.tiling_case.tensor_ub_size_list[self.tiling_case.shape_type]
            self.schedule[stage_tensor].set_buffer_size(ub_count)

    def _last_transpose_align(self):
        # C can't be last_dim in ouotput (NC1HC0 -> NC1C0H)
        if BLOCK // DTYPE_BYTE_MAPPING.get(self.dtype, 1) in [FP16_BLOCK, INT8_BLOCK]:
            align_factor = self.pad_factor
        else:
            align_factor = FP32_ALIGN_SIZE

        # (H,C0) -> (C0,H)
        for tensor in self.get_all_producers_stages(self.tiling_tensor):
            if tensor in self.graph_info.input_tensor_set:
                continue
            if tensor in [self.mte2_tensor, self.remove_pad_tensor]:
                continue
            self.schedule[tensor].storage_align(tensor.op.axis[-2], align_factor, 0)

    def _n_last_transpose_align(self):
        # C must be last dim in output (NC1HC0->NHC1C0)
        for tensor in self.depad_tensors:
            if tensor == self.tiling_tensor:
                continue
            if len(tensor.shape) >= 2:
                self.schedule[tensor].storage_align(tensor.op.axis[-2], self.pad_factor, 0)

    def _do_storage_align(self):
        if self.is_last_transpose:
            self._last_transpose_align()
        else:
            self._n_last_transpose_align()

    def _do_mem_reused(self):
        self.schedule[self.reshape_tensor].reused_by(self.transpose_tensor)

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
            align(self.reshape_tensor, [c_idx, ], self.pad_factor)
        else:
            if self.depad_axis_list[-1] == len(self.reshape_tensor.shape) - 1:
                align(self.reshape_tensor, [-1, ], self.pad_factor)

    def _calc_multi_core(self):
        if self.need_multi_core:
            idx = self.reorder_list.index(self.iter_block_outer)
            backward_fused_list = self.reorder_list[:idx + 1]
            self.multi_core_fused_axis = self.schedule[self.tiling_tensor].fuse(*backward_fused_list)
            self.multi_core_bind_tensor = self.tiling_tensor

    def _calc_compute_at(self):
        self.compute_at_map.clear()
        for tensor in self.get_all_producers_stages(self.tiling_tensor):
            if tensor not in self.graph_info.input_tensor_set:
                self.compute_at_map[tensor] = {"parent": self.schedule[self.tiling_tensor], "scope": self.ub_outer}

    def _transpose_emit_insn(self, tensor):
        emit_idx, insn = self._calc_permute_in_ub()
        if insn == "vector_transpose":
            iter = tensor.op.axis[emit_idx]
            src_in_dst_order = tvm.expr.Call('handle', 'tvm_tuple', self.permute, tvm.expr.Call.PureIntrinsic, None, 0)
            self.emit_insn_map[tensor] = {"scope": iter, "instruction": "vector_transpose",
                                          "src_in_dst_order": src_in_dst_order}
        elif insn in ["vector_or", "dma_copy"]:
            self.emit_insn_map[tensor] = {"scope": tensor.op.axis[0], "instruction": insn}

    def _calc_emit_insn(self):
        self.emit_insn_map.clear()
        self._transpose_emit_insn(self.transpose_tensor)
        self.emit_insn_map[self.mte2_tensor] = {"scope": self.mte2_tensor.op.axis[0], "instruction": "dma_copy"}
        self.emit_insn_map[self.reshape_tensor] = {"scope": self.reshape_tensor.op.axis[0], "instruction": "phony_insn"}

        for tensor in self.depad_tensors:
            if tensor == self.tiling_tensor:
                self.emit_insn_map[tensor] = {"scope": self.ub_inner, "instruction": "dma_copy",
                                              NO_OVERLAP: {NO_OVERLAP: 3,
                                                           "no_overlap_malloc_buf_for_tail": 0}}
            else:
                self.emit_insn_map[tensor] = {"scope": tensor.op.axis[0], "instruction": "dma_copy"}

        if self.branch == COMMON_ALIGN:
            self.emit_insn_map[self.remove_pad_tensor] = {"scope": self.remove_pad_tensor.op.axis[0],
                                                          "instruction": "remove_pad"}

    def _calc_permute_in_ub(self):
        if self.is_last_transpose:
            instruction = "vector_transpose"
            if BLOCK // DTYPE_BYTE_MAPPING.get(self.dtype, 1) in [FP16_BLOCK, INT8_BLOCK]:
                idx = 0
            else:
                idx = -2
                back = self.permute[-2:].copy()
                min_value = min(self.permute)
                self.permute = [x - min_value for x in back]
        else:
            idx = 0
            if BLOCK // DTYPE_BYTE_MAPPING.get(self.dtype, 1) in [INT8_BLOCK, ]:
                instruction = "dma_copy"
            else:
                instruction = "vector_or"

        return idx, instruction
