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
transdata forward borrow h
"""
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import var_inner
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from tbe.dsl.unify_schedule.constants import TransdataCategory
from ..common.transdata_base_sch import TransdataBaseSch
from ..common.transdata_graph_info import choose_transpose_insn, get_reshape
from ..common.transdata_graph_info import math_prod, set_align
from ..common.constants import BLOCK


class TransFBHSchedule(TransdataBaseSch):
    """
    TransForwardBorrowHSchedule: forward + n_last_transpose + borrow H
    """

    def __init__(self, outs, tiling_case):
        TransdataBaseSch.__init__(self, outs, tiling_case)

        self.h1_index = None
        self.h0_index = None
        self.h_mapping_indexes = []

        # tensors
        self.transpose_0_tensor = None
        self.transpose_1_tensor = None
        self.transpose_2_tensor = None

        self.shadow_pad_tensor = None
        self.s_reshape_tensor = None
        self.f_reshape_tensor = None
        self.h_pad_tensor = None

        # tensors' attrs
        self.hi = None
        self.perm_2 = []
        self.perm_1 = []
        self.perm_0 = []

    @classmethod
    def get_supported_sub_pattern(cls):
        return TransdataCategory.BORROW_H_B8B16_FORWARD

    def do_schedule(self):
        """
        Process of schedule
        """
        self._create_schedule()
        self._do_set_scope()
        self._init_tensors()

        self._calc_tiling()
        self._do_tiling()

        self._analysis_transpose()

        self._do_mem_reused()
        self._do_storage_bound()
        self._do_storage_align()
        self._do_buffer_align()
        self._do_set_predicate()

        self._calc_multi_core()
        self._do_multi_core()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._do_pragma()
        self.schedule.tiling_key = self.tiling_case.tiling_key
        return self.schedule

    def _init_tensors(self):
        """
        DataStream: pre-stream + main-stream + post-stream
            pre-stream:
                gm -> mte2_tensor(pad+s_reshape) -> transpose_0 -> shadow_pad
            main-stream:
                shadow_pad -> pad_tensor(C...) -> s_reshape -> transpose_1
            post-stream:
                transpose_1 -> transpose_2 -> f_reshape -> pad_tensor(H) -> res
        """
        self.mte2_tensor = self.child(list(self.graph_info.input_tensor_set)[0])
        self.transpose_0_tensor = self.child(self.mte2_tensor)
        # shadow-pad: do pad in shadow-pad
        self.shadow_pad_tensor = self.single_cache_read(self.transpose_0_tensor)

        # backward-infer
        self.tiling_tensor = list(self.graph_info.output_tensor_set)[0]
        self.h_pad_tensor = self.parent(self.tiling_tensor)
        self.f_reshape_tensor = self.parent(self.h_pad_tensor)
        self.transpose_2_tensor = self.parent(self.f_reshape_tensor)
        self.transpose_1_tensor = self.parent(self.transpose_2_tensor)
        self.s_reshape_tensor = self.parent(self.transpose_1_tensor)
        # pad-tensors
        self.pad_tensors = self.graph_info.pad_tensor_set.copy()
        self.pad_tensors.remove(self.h_pad_tensor)
        self.pad_tensors = list(self.pad_tensors)
        self.pad_tensors.sort(key=lambda x: int(x.op.attrs["axes"]), reverse=True)
        self.pad_axis_list = [int(x.op.attrs["axes"]) for x in self.pad_tensors]
        # make db for mte2 mte3
        if self.tiling_case.transpose_work:
            self._do_cache_write()

    def _define_factor(self):
        self.hi = list(self.transpose_2_tensor.shape)[self.h0_index]
        if self.tiling_case.ub_first_factor:
            # hi is VarExpr from compute that stored in ub_first_factor in const-model.
            # While const, hi is assigned a constant value.
            compute = get_context().get_current_compute().get_operator_context().get_current_compute()
            compute.get_var("_hi").set_bound([self.tiling_case.ub_first_factor, self.tiling_case.ub_first_factor])
        # ub_factor based on transpose_2_tensor is constant in compilation
        if not self.tiling_case.block_factor:
            self.tiling_case.block_factor = var_inner("_block_factor", (1, None))
        self.tiling_case.ub_first_factor = self.align_factor

    def _calc_tiling(self):
        """
        The schedule has four regulations:
        1. only support split once.
        2. ub_factor based on transpose_2_tensor is constant, just like 16(fp16), 32(int8).
        3. ub_factor based on tiling_tensor is 16 * hi (fp16).
        4. hi is VarExpr that from compute, it should be assigned in const-model.
        """
        self.align_factor = BLOCK // DTYPE_BYTE_MAPPING.get(self.tiling_tensor.dtype, 1)
        self.split_once = self.tiling_case.ub_split_second_idx == self.tiling_case.ub_split_first_idx
        if not self.split_once:
            dict_args = {"errCode": "E90003", "detailed_cause": "In the schedule, only support ub split once"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        # regulation
        perm = [int(x) for x in list(self.transpose_2_tensor.op.attrs["permute"])]
        self.parses_axis_type(perm, ub_internal_c=True)
        self.h_mapping_indexes, self.h1_index, self.h0_index = self.parses_f_reshape(self.f_reshape_tensor)
        self._define_factor()
        self.parses_tiling_info(self.h_mapping_indexes, self.h1_index, self.h0_index, self.hi)

    def _do_tiling(self):
        self._do_ub_tiling()
        self._do_block_tiling()
        self._do_reorder()
        self._do_fragments()

    def _do_buffer_align(self):
        def align(tensor, _axis_list, _factor):
            align_list = [[1, 1] for x in tensor.shape]
            for i in _axis_list:
                align_list[i] = [1, _factor]
            self.schedule[tensor].buffer_align(*align_list)

        if self.h1_index is not None:
            h_idx = self.h_mapping_indexes[self.h1_index]
            align(self.h_pad_tensor, [h_idx, ], self.align_factor * self.hi)

    def _do_set_predicate(self):
        self.schedule[self.mte2_tensor].set_store_predicate(self.mte2_tensor.op.body[0].condition)
        for tensor in self.graph_info.pad_tensor_set:
            self.schedule[tensor].set_store_predicate(tensor.op.body[0].condition)

    def _do_storage_align(self):
        align_var = self.tiling_case.tensor_ub_size_list[self.tiling_case.shape_type]
        align_var = align_var // self.align_factor // self.align_factor
        align_var = align_var if align_var % 2 != 0 else align_var - 1
        align_var *= self.align_factor

        for v in get_reshape(self.mte2_tensor):
            if isinstance(v, (list, tuple)):
                self.schedule[self.mte2_tensor].storage_align(self.mte2_tensor.op.axis[v[0]], align_var, 0)

        shape = list(self.shadow_pad_tensor.shape)
        for i in self.pad_axis_list:
            # Attention shape[-1] would be split by align_factor
            _factor = math_prod(shape[i + 1:-1]) * self.align_factor * self.pad_factor
            shape[i] = set_align(shape[i], self.pad_factor)
            self.schedule[self.shadow_pad_tensor].storage_align(self.shadow_pad_tensor.op.axis[i - 1], _factor, 0)

    def _do_mem_reused(self):
        self.schedule[self.shadow_pad_tensor].reused_by(*self.pad_tensors)
        self.schedule[self.shadow_pad_tensor].reused_by(self.s_reshape_tensor)
        self.schedule[self.transpose_2_tensor].reused_by(*[self.f_reshape_tensor, self.h_pad_tensor])

    def _transpose_emit_insn(self, tensor, perm):
        emit_idx, insn = choose_transpose_insn(perm)
        if insn in ["vector_transpose", "phony_insn"]:
            self.vnchwconv_insn_map(tensor, insn, tensor.op.axis[emit_idx], perm)
        else:
            self.emit_insn_map[tensor] = {"scope": tensor.op.axis[0], "instruction": insn}

    def _calc_emit_insn(self):
        _map = self.emit_insn_map
        _map.clear()

        # pre-process
        _map[self.mte2_tensor] = {"scope": self.mte2_tensor.op.axis[0], "instruction": "dma_copy"}
        self._transpose_emit_insn(self.transpose_0_tensor, self.perm_0)
        _map[self.shadow_pad_tensor] = {"scope": self.shadow_pad_tensor.op.axis[0], "instruction": "dma_copy"}
        # main-process
        for i in self.pad_tensors:
            _map[i] = {"scope": i.op.axis[0], "instruction": "vector_dup"}
        _map[self.s_reshape_tensor] = {"scope": self.s_reshape_tensor.op.axis[0], "instruction": "phony_insn"}

        self._transpose_emit_insn(self.transpose_1_tensor, self.perm_1)
        self._transpose_emit_insn(self.transpose_2_tensor, self.perm_2)
        _map[self.f_reshape_tensor] = {"scope": self.f_reshape_tensor.op.axis[0], "instruction": "phony_insn"}
        _map[self.h_pad_tensor] = {"scope": self.h_pad_tensor.op.axis[0], "instruction": "vector_dup"}
        _map[self.tiling_tensor] = {"scope": self.ub_inner, "instruction": "dma_copy"}

        # make db for mte2 mte3
        if self.tiling_case.transpose_work:
            tensor = self.parent(self.tiling_tensor)
            _map[tensor] = {"scope": tensor.op.axis[0], "instruction": "dma_copy"}

    def _do_pragma(self):

        def _pragma(tensor, begin, end):
            axes = [tensor.op.axis[i] for i in range(begin, end)]
            self._do_group(tensor, axes)

        _pragma(self.shadow_pad_tensor, self.pad_axis_list[0], len(self.shadow_pad_tensor.shape))
        _pragma(self.transpose_1_tensor, len(self.transpose_1_tensor.shape) - 2, len(self.transpose_1_tensor.shape))

        for _tensor, _axis in zip(self.pad_tensors, self.pad_axis_list):
            _pragma(_tensor, _axis, len(_tensor.shape))

    def _analysis_transpose(self):
        # Func: judge transpose work or not
        self.perm_2 = self.update_ub_perm(self.transpose_2_tensor)
        self.perm_1 = self.update_ub_perm(self.transpose_1_tensor, transpose_work=self.tiling_case.transpose_work)
        self.perm_0 = self.update_ub_perm(self.transpose_0_tensor)
