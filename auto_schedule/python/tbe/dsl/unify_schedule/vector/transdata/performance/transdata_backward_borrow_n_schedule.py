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
transdata backward borrow n
"""
from tbe import tvm
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import var_inner
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.unify_schedule.constants import TransdataCategory
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from tbe.dsl.unify_schedule.vector.transdata.transdata_base_schedule import TransdataBaseSchedule
from tbe.dsl.unify_schedule.vector.transdata.transdata_graph_info import get_reshape
from tbe.dsl.unify_schedule.vector.transdata.transdata_graph_info import choose_transpose_insn

BLOCK = 32
STRIDE_2 = 2
STRIDE_3 = 3
NO_OVERLAP = "no_overlap"


class TransBackwardBorrowNSchedule(TransdataBaseSchedule):
    """
    TransBackwardBorrowNSchedule: backward + n-last-transpose + borrow n
    """

    def __init__(self, outs, tiling_case):
        TransdataBaseSchedule.__init__(self, outs, tiling_case)
        self.tiling_case = tiling_case
        self.pad_factor = None
        self.align_factor = None
        self.split_once = None

        self.n1_index = None
        self.n0_index = None
        self.c1_index = None
        self.c0_index = None
        self.n1_axis = None
        self.n0_axis = None
        self.axis_in_ub = []
        self.axis_not_in_ub = []
        self.n_mapping_indexes = []
        self.c_mapping_indexes = []
        self.tiling_axes = []
        self.is_do_extra_split_n = False

        self.iter_ub_outer = None
        self.iter_ub_inner = None
        self.iter_block_outer = None
        self.iter_block_inner = None
        self.reorder_list = []
        self.ub_outer = None
        self.ub_inner = None

        self.mte2_tensor = None
        self.crd_tensor = None
        self.pad_tensor = None
        self.s_reshape_tensor = None
        self.transpose_0_tensor = None

        self.transpose_1_tensor = None
        self.f_reshape_0_tensor = None
        self.depad_tensors = None
        self.depad_axis_list = []

        self.transpose_2_tensor = None
        self.f_reshape_1_tensor = None
        self.tiling_tensor = None

        self.perm_2 = []
        self.perm_1 = []
        self.perm_0 = []

    @classmethod
    def get_supported_sub_pattern(cls):
        return TransdataCategory.BORROW_N_B8B16_BACKWARD

    def _bn_backward_init_tensors(self):
        """
        DataStream: pre-stream + main-stream + post-stream
            pre-stream:
                gm -> mte2_tensor -> crd_tensor -> pad_tensor -> s_reshape_tensor -> transpose_0
            main-stream:
                transpose_0 -> transpose_1 -> f_reshape_0 -> depad_tensor
            post-stream:
                depad_tensor -> transpose_2 -> f_reshape_1 -> res
        """
        self.mte2_tensor = self.cache_read_tensors_and_buffer_map[list(self.graph_info.input_tensor_set)[0]]
        readers = self.forward_stage_graph_map[self.mte2_tensor]
        self.crd_tensor = self.schedule.cache_read(self.mte2_tensor, "local.UB", readers)
        self.cache_read_tensors_and_buffer_map[self.mte2_tensor] = self.crd_tensor
        for item in readers:
            self.update_stage(self.crd_tensor, item, True)

        self.pad_tensor = list(self.graph_info.pad_tensor_set)[0]
        self.s_reshape_tensor = list(self.graph_info.s_reshape_tensor_set)[0]
        self.transpose_0_tensor = list(self.forward_stage_graph_map[self.s_reshape_tensor])[0]

        self.transpose_1_tensor = list(self.forward_stage_graph_map[self.transpose_0_tensor])[0]
        self.f_reshape_0_tensor = list(self.forward_stage_graph_map[self.transpose_1_tensor])[0]
        self.depad_tensors = list(self.graph_info.de_pad_tensor_set)
        self.depad_tensors.sort(key=lambda x: int(x.op.attrs["axes"]))
        self.depad_axis_list = [int(x.op.attrs["axes"]) for x in self.depad_tensors]

        self.tiling_tensor = list(self.graph_info.output_tensor_set)[0]
        self.transpose_2_tensor = list(self.forward_stage_graph_map[self.depad_tensors[-1]])[0]
        self.f_reshape_1_tensor = list(self.forward_stage_graph_map[self.transpose_2_tensor])[0]

    def do_schedule(self):
        """
        Process of schedule
        """
        self._create_schedule()
        self._do_cache_read()
        self._do_set_scope()
        self._bn_backward_init_tensors()

        self._bn_backward_calc_tiling()
        self._bn_backward_do_tiling()

        self._analysis_transpose_operator()

        self._bn_backward_do_mem_reused()
        self._bn_backward_do_storage_bound()
        self._bn_backward_do_storage_align()
        self._bn_backward_do_buffer_align()

        self._bn_backward_calc_multi_core()
        self._do_multi_core()

        self._bn_backward_calc_compute_at()
        self._do_compute_at()

        self._bn_backward_calc_emit_insn()
        self._do_emit_insn()

        self._bn_backward_do_pragma()
        self.schedule.tiling_key = self.tiling_case.tiling_key
        return self.schedule

    def _bn_backward_calc_tiling(self):
        self.pad_factor = get_context().get_current_compute().get("_pad_factor")
        self.align_factor = BLOCK // DTYPE_BYTE_MAPPING.get(self.tiling_tensor.dtype, 1)
        self.split_once = self.tiling_case.ub_split_second_idx == self.tiling_case.ub_split_first_idx
        if not self.split_once:
            dict_args = {"errCode": "E90001", "detailed_cause": "only support ub split once"}
            raise RuntimeError(dict_args, get_error_message(dict_args))
        # split_i and split_o are based on transpose_2_tensor
        split_i = self.tiling_case.ub_split_first_idx
        split_o = self.tiling_case.ub_split_second_idx
        split_b = self.tiling_case.block_split_idx
        perm = [int(x) for x in list(self.transpose_2_tensor.op.attrs["permute"])]
        length = len(perm)

        def parses_reshape_n():
            """
            Func:
            1. Find indexes of n1 and n0 in f_reshape_1_tensor.
            2. Create mapping between f_reshape_1_tensor and tiling_tensor by indexes
            """
            reshape = get_reshape(self.f_reshape_1_tensor)
            for index, value in enumerate(reshape):
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    self.n_mapping_indexes.extend([index, ] * 2)
                    self.n1_index = value[0]
                    self.n0_index = value[1]
                else:
                    self.n_mapping_indexes.append(index)

        def parses_reshape_c():
            """
            Func:
            1. Find indexes of c1 and c0 in f_reshape_0_tensor.
            2. Create mapping between f_reshape_0_tensor and depad_tensor by indexes
            """
            reshape = get_reshape(self.f_reshape_0_tensor)
            for index, value in enumerate(reshape):
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    self.c_mapping_indexes.extend([index, ] * 2)
                    self.c1_index = value[0]
                    self.c0_index = value[1]
                else:
                    self.c_mapping_indexes.append(index)

        def parses_factor():
            # define factor
            self.tiling_case.block_factor = self.tiling_case.block_factor if self.tiling_case.block_factor \
                else var_inner("_block_factor", (1, None))
            self.tiling_case.ub_first_factor = self.tiling_case.ub_first_factor if self.tiling_case.ub_first_factor \
                else var_inner("_ub_first_factor", (1, None))

        def parses_split_one():
            """
            Func: Classified transpose_2_tensor's axis into UB Internal and UB External
            """
            input_internal_ub = {perm.index(x) for x in range(perm[split_i], length, 1)}
            output_internal_ub = set(range(split_o, length, 1))
            self.axis_in_ub = output_internal_ub.union(input_internal_ub)
            self.axis_not_in_ub = set(range(length)).difference(self.axis_in_ub)
            self.axis_not_in_ub = self.axis_not_in_ub.union({split_o, split_i})
            # for reorder
            self.axis_not_in_ub = list(self.axis_not_in_ub)
            self.axis_not_in_ub.sort()
            self.axis_in_ub = list(self.axis_in_ub)
            self.axis_in_ub.sort()

        def parses_split_two():
            """
            Func:
            1. Do extra split for tiling_tensor(split N as N1 and N0)
            2. Collect tiling_axes
            3. Update axis_in_ub and axis_not_in_ub base on tiling_tensor
            """
            is_n0_in_ub = self.n0_index in self.axis_in_ub
            is_n1_out_ub = self.n1_index in self.axis_not_in_ub
            is_split_n1 = self.n1_index in [split_i, split_o]
            self.is_do_extra_split_n = is_n1_out_ub and is_n0_in_ub and not is_split_n1

            # update factor while split n1:
            # 1. split n1 mean that do not extra split n as n1 and n0
            # 2. ub split firstly, if ub split n1, factor is factor * 16(float16)
            # 3. block split secondly, if block split n1 and ub not split n1, factor is factor * 16(float16)
            if split_i == self.n1_index:
                self.tiling_case.ub_first_factor *= self.pad_factor
            if split_b == self.n1_index and split_i != self.n1_index and not self.is_do_extra_split_n:
                self.tiling_case.block_factor *= self.pad_factor

            if self.is_do_extra_split_n:
                # do extra split
                _var = self.tiling_tensor.op.axis[self.n_mapping_indexes[self.n1_index]]
                self.n1_axis, self.n0_axis = self.schedule[self.tiling_tensor].split(_var, factor=self.pad_factor)
                # collect tiling_axes
                for _idx, _value in enumerate(self.tiling_tensor.op.axis):
                    if _idx == self.n_mapping_indexes[self.n1_index]:
                        self.tiling_axes.extend([self.n1_axis, self.n0_axis])
                    else:
                        self.tiling_axes.append(_value)
            else:
                for _value in self.tiling_tensor.op.axis:
                    self.tiling_axes.append(_value)

                # update axis_in_ub and axis_not_in_ub that make them based on tiling_tensor
                self.tiling_case.block_split_idx = self.n_mapping_indexes[split_b]
                self.tiling_case.ub_split_first_idx = self.n_mapping_indexes[split_i]
                self.tiling_case.ub_split_second_idx = self.n_mapping_indexes[split_o]
                if self.n1_index in self.axis_in_ub and self.n0_index in self.axis_in_ub:
                    self.axis_in_ub.remove(self.n1_index)
                if self.n1_index in self.axis_not_in_ub and self.n0_index in self.axis_not_in_ub:
                    self.axis_not_in_ub.remove(self.n0_index)
                self.axis_in_ub = [self.n_mapping_indexes[x] for x in self.axis_in_ub]
                self.axis_not_in_ub = [self.n_mapping_indexes[x] for x in self.axis_not_in_ub]

        parses_reshape_n()
        parses_reshape_c()
        parses_factor()
        parses_split_one()
        parses_split_two()

    def _bn_backward_do_tiling(self):
        self._do_ub_tiling()
        self._do_block_tiling()
        self._do_reorder()
        self._do_fragments()

    def _do_ub_tiling(self):
        _var = self.tiling_axes[self.tiling_case.ub_split_first_idx]
        self.iter_ub_outer, self.iter_ub_inner = \
            self.schedule[self.tiling_tensor].split(_var, factor=self.tiling_case.ub_first_factor)

    def _do_block_tiling(self):
        _tensor = self.tiling_tensor
        _factor = self.tiling_case.block_factor
        _ub_split_idx = self.tiling_case.ub_split_first_idx
        _blk_split_idx = self.tiling_case.block_split_idx

        _var = self.iter_ub_outer if _blk_split_idx == _ub_split_idx else self.tiling_axes[_blk_split_idx]
        self.iter_block_outer, self.iter_block_inner = self.schedule[_tensor].split(_var, factor=_factor)

    def _do_reorder(self):
        _ub_split_idx = self.tiling_case.ub_split_first_idx
        _blk_split_idx = self.tiling_case.block_split_idx

        for idx in self.axis_in_ub:
            self.reorder_list.append(self.iter_ub_inner if idx == _ub_split_idx else self.tiling_axes[idx])

        outside = []
        for i in self.axis_not_in_ub:
            if i == _blk_split_idx:
                outside.extend([self.iter_block_outer, self.iter_block_inner])
            elif i == _ub_split_idx:
                outside.append(self.iter_ub_outer)
            else:
                outside.append(self.tiling_axes[i])

        self.ub_outer, self.ub_inner = outside[-1], self.reorder_list[0]
        self.reorder_list = outside + self.reorder_list
        self.schedule[self.tiling_tensor].reorder(*self.reorder_list)

    def _do_fragments(self):
        self.schedule[self.tiling_tensor].pragma(self.iter_block_inner, "local.UB_fragments_memory_size", 256)

    def _bn_backward_do_storage_bound(self):
        for stage_tensor in self.forward_stage_graph_map:
            ub_count = self.tiling_case.tensor_ub_size_list[self.tiling_case.shape_type]
            self.schedule[stage_tensor].set_buffer_size(ub_count)

    def _bn_backward_do_buffer_align(self):
        def align(tensor, _axis_list, _factor):
            align_list = [[1, 1] for x in tensor.shape]
            for i in _axis_list:
                align_list[i] = [1, _factor]
            self.schedule[tensor].buffer_align(*align_list)

        if self.c1_index is not None:
            c_idx = self.c_mapping_indexes[self.c1_index]
            align(self.f_reshape_0_tensor, [c_idx, ], self.pad_factor)

        if self.n1_index is not None:
            n_idx = self.n_mapping_indexes[self.n1_index]
            align(self.f_reshape_1_tensor, [n_idx, ], self.pad_factor)

    def _bn_backward_do_storage_align(self):
        align_var = self.tiling_case.tensor_ub_size_list[self.tiling_case.shape_type]
        if self.tiling_case.ub_split_first_idx == 0:
            align_var = STRIDE_3 * self.align_factor
        else:
            align_var = align_var // self.align_factor // self.align_factor
            align_var = align_var if align_var % 2 != 0 else align_var - 1
            align_var *= self.align_factor

        sch = self.schedule
        # pre-process
        sch[self.crd_tensor].storage_align(self.crd_tensor.op.axis[0], align_var, 0)
        sch[self.pad_tensor].storage_align(self.pad_tensor.op.axis[0], align_var, 0)
        sch[self.s_reshape_tensor].storage_align(self.s_reshape_tensor.op.axis[1], align_var, 0)

        # main-process(None) post-process
        sch[self.transpose_2_tensor].storage_align(self.transpose_2_tensor.op.axis[1], align_var, 0)
        sch[self.f_reshape_1_tensor].storage_align(self.f_reshape_1_tensor.op.axis[0], align_var, 0)

    def _bn_backward_do_mem_reused(self):
        sch = self.schedule
        sch[self.crd_tensor].reused_by(self.pad_tensor)
        sch[self.pad_tensor].reused_by(self.s_reshape_tensor)
        sch[self.transpose_1_tensor].reused_by(self.f_reshape_0_tensor)
        sch[self.transpose_2_tensor].reused_by(self.f_reshape_1_tensor)

    def _bn_backward_calc_multi_core(self):
        if self.need_multi_core:
            idx = self.reorder_list.index(self.iter_block_outer)
            backward_bn_fused_list = self.reorder_list[:idx + 1]
            self.multi_core_fused_axis = self.schedule[self.tiling_tensor].fuse(*backward_bn_fused_list)
            self.multi_core_bind_tensor = self.tiling_tensor

    def _bn_backward_calc_compute_at(self):
        self.compute_at_map.clear()
        for _tensor in self.get_all_producers_stages(self.tiling_tensor):
            if _tensor not in self.graph_info.input_tensor_set:
                self.compute_at_map[_tensor] = {"parent": self.schedule[self.tiling_tensor], "scope": self.ub_outer}

    def _transpose_emit_insn(self, tensor, perm):
        emit_idx, insn = choose_transpose_insn(perm)
        if insn == "vector_transpose":
            src_in_dst_order = tvm.expr.Call('handle', 'tvm_tuple', perm, tvm.expr.Call.PureIntrinsic, None, 0)
            self.emit_insn_map[tensor] = {"scope": tensor.op.axis[emit_idx], "instruction": "vector_transpose",
                                          "src_in_dst_order": src_in_dst_order}
        elif insn in ["dma_copy"]:
            self.emit_insn_map[tensor] = {"scope": tensor.op.axis[0], "instruction": insn}

    def _bn_backward_calc_emit_insn(self):
        _map = self.emit_insn_map
        _map.clear()
        _map[self.mte2_tensor] = {"scope": self.mte2_tensor.op.axis[0], "instruction": "dma_copy"}
        _map[self.crd_tensor] = {"scope": self.crd_tensor.op.axis[0], "instruction": "dma_copy"}
        _map[self.pad_tensor] = {"scope": self.pad_tensor.op.axis[0], "instruction": "phony_insn"}
        _map[self.s_reshape_tensor] = {"scope": self.s_reshape_tensor.op.axis[0], "instruction": "phony_insn"}

        # transpose
        self._transpose_emit_insn(self.transpose_0_tensor, self.perm_0)
        self._transpose_emit_insn(self.transpose_1_tensor, self.perm_1)
        _map[self.f_reshape_0_tensor] = {"scope": self.f_reshape_0_tensor.op.axis[0], "instruction": "phony_insn"}

        for _tensor in self.depad_tensors:
            _map[_tensor] = {"scope": _tensor.op.axis[0], "instruction": "dma_copy"}

        self._transpose_emit_insn(self.transpose_2_tensor, self.perm_2)
        _map[self.f_reshape_1_tensor] = {"scope": self.f_reshape_1_tensor.op.axis[0], "instruction": "phony_insn"}

        _map[self.tiling_tensor] = {"scope": self.ub_inner, "instruction": "dma_copy",
                                    NO_OVERLAP: {NO_OVERLAP: 3, "no_overlap_malloc_buf_for_tail": 0}}

    def _bn_backward_do_pragma(self):
        """
        src-shape >>> (N, C1, H, C0)
        pad-shape >>> (Nx, C1, H ,C0)
        split >>> (Nx.o, 16, C1, H, C0)
        trans_0 >>> (Nx.o, C1, H, C0, 16)
        trans_1 >>> (Nx.o, H, C1, C0, 16)
        fuse_0 >>> (Nx.o, H, C1*C0, 16)
        depad >>> (Nx.o, H, C, 16)
        trans_2 >>> (Nx.o, 16, H, C)
        fuse_1 >>> (Nx, H, C)
        dst >>> (N,H,C)
        """
        sch = self.schedule
        group_id = tvm.make.Call("int32", "axis_group", [0, "overwrite"], tvm.expr.Call.Extern, None, 0)

        def _pragma(tensor, begin, end):
            for i in range(begin, end):
                sch[tensor].pragma(tensor.op.axis[i], "axis_group", group_id)

        _pragma(self.transpose_2_tensor, STRIDE_2, len(self.transpose_2_tensor.shape))
        _pragma(self.transpose_0_tensor, 1, len(self.transpose_0_tensor.shape) - 1)
        _pragma(self.crd_tensor, 1, len(self.crd_tensor.shape))

    def _analysis_transpose_operator(self):
        # Func: according to axes that belong to ub-internal, to infer new permute
        root_axis_num = 0
        for i in self.axis_not_in_ub:
            if i not in self.axis_in_ub:
                root_axis_num += 1

        def _get_perm(t_tensor):
            perm = [int(x) for x in list(t_tensor.op.attrs["permute"])]
            ori_permute = perm[root_axis_num:]
            back = sorted(ori_permute.copy())
            return [back.index(j) for j in ori_permute]

        self.perm_2 = _get_perm(self.transpose_2_tensor)
        self.perm_1 = _get_perm(self.transpose_1_tensor)
        self.perm_0 = _get_perm(self.transpose_0_tensor)
