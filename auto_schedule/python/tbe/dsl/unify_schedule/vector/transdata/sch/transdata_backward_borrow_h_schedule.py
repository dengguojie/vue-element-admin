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
transdata backward borrow h
"""
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils.shape_util import shape_to_list
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import var_inner
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from tbe.dsl.unify_schedule.constants import TransdataCategory
from ..common.transdata_base_sch import TransdataBaseSch
from ..common.transdata_graph_info import choose_transpose_insn, get_reshape
from ..common.transdata_graph_info import math_prod
from ..common.constants import BLOCK, NO_OVERLAP


class TransBBHSchedule(TransdataBaseSch):
    """
    TransBackwardBorrowHSchedule: backward + n_last_transpose + borrow H
    """

    def __init__(self, outs, tiling_case):
        TransdataBaseSch.__init__(self, outs, tiling_case)

        self.h1_index = None
        self.h0_index = None
        self.h_mapping_indexes = []
        self.c1_index = None
        self.c0_index = None
        self.c_mapping_indexes = []

        # tensors
        self.h_pad_tensor = None
        self.f0_reshape_tensor = None
        self.f1_reshape_tensor = None
        self.f2_reshape_tensor = None

        self.transpose_0_tensor = None
        self.transpose_1_tensor = None
        self.transpose_2_tensor = None

        # tensors' attrs
        self.hi = None
        self.perm_2 = []
        self.perm_1 = []
        self.perm_0 = []

    @classmethod
    def get_supported_sub_pattern(cls):
        return TransdataCategory.BORROW_H_B8B16_BACKWARD

    def do_schedule(self):
        """
        Process of schedule
        """
        self._create_schedule()
        self._do_cache_read()
        self._do_set_scope()
        self._init_tensors()

        self._calc_tiling()
        self._do_tiling()

        self._analysis_transpose()

        self._do_mem_reused()
        self._do_storage_bound()
        self._do_storage_align()
        self._do_buffer_align()

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
                gm -> mte2_tensor -> pad_tensor(H) -> fake_s_reshape -> f_reshape -> transpose_0
            main-stream:
                transpose_0 -> transpose_1 -> f_reshape -> depad
            post-stream:
                depad -> transpose_2 -> f_reshape -> res
        """
        self.mte2_tensor = self.child(list(self.graph_info.input_tensor_set)[0])
        self.h_pad_tensor = self.child(self.mte2_tensor)

        for i in self.graph_info.f_reshape_tensor_set:
            if self.parent(i) in self.graph_info.s_reshape_tensor_set:
                self.f0_reshape_tensor = i
                self.transpose_0_tensor = self.child(self.f0_reshape_tensor)
            elif self.child(i) in self.graph_info.output_tensor_set:
                self.f2_reshape_tensor = i
                self.transpose_2_tensor = self.parent(self.f2_reshape_tensor)
            else:
                self.f1_reshape_tensor = i
                self.transpose_1_tensor = self.parent(self.f1_reshape_tensor)

        # depad tensors
        self.depad_tensors = list(self.graph_info.de_pad_tensor_set)
        self.depad_tensors.sort(key=lambda x: int(x.op.attrs["axes"]))
        self.depad_axis_list = [int(x.op.attrs["axes"]) for x in self.depad_tensors]
        self.tiling_tensor = list(self.graph_info.output_tensor_set)[0]

    def _define_factor(self):
        self.hi = list(self.transpose_2_tensor.shape)[self.h0_index]
        if self.tiling_case.ub_first_factor:
            # hi is VarExpr from computTransdataTilingCasee that stored in ub_first_factor in const-model.
            # While const, hi is assigned a constant value.
            compute = get_context().get_current_compute().get_operator_context().get_current_compute()
            compute.get_var("_hi").set_bound([self.tiling_case.ub_first_factor, self.tiling_case.ub_first_factor])
        # ub_factor based on transpose_2_tensor is constant in compilation.
        if not self.tiling_case.block_factor:
            self.tiling_case.block_factor = var_inner("_block_factor", (1, None))
        self.tiling_case.ub_first_factor = self.align_factor

    def _calc_tiling(self):
        """
        The schedule has three regulations:
        1. only support split once.
        2. shape of res is as same as transpose-2 that make ub_factor is constant, just like 16(fp16), 32(int8)
        3. hi is VarExpr that from compute, it should be assigned in const-model.
        """
        self.align_factor = BLOCK // DTYPE_BYTE_MAPPING.get(self.tiling_tensor.dtype, 1)
        self.split_once = self.tiling_case.ub_split_second_idx == self.tiling_case.ub_split_first_idx
        if not self.split_once:
            dict_args = {"errCode": "E90003", "detailed_cause": "In the schedule, only support ub split once"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        # regulation
        perm = [int(x) for x in list(self.transpose_2_tensor.op.attrs["permute"])]
        self.parses_axis_type(perm)
        self.h_mapping_indexes, self.h1_index, self.h0_index = self.parses_f_reshape(self.f2_reshape_tensor)
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

        self.c_mapping_indexes, self.c1_index, self.c0_index = self.parses_f_reshape(self.f1_reshape_tensor)

        if self.c1_index is not None:
            c_idx = self.c_mapping_indexes[self.c1_index]
            align(self.f1_reshape_tensor, [c_idx, ], self.pad_factor)

        if self.h1_index is not None:
            h_idx = self.h_mapping_indexes[self.h1_index]
            align(self.f2_reshape_tensor, [h_idx, ], self.align_factor * self.hi)

    def _do_storage_align(self):
        """
        Func:
        N C1 H C0
        N C1 Hx C0: Hx = SetAlign(H, 256)
        // handle reshape
        N C1 ho 256*hi C0 -> s0
        N C1 ho 16 16*hi C0 -> s1
        // handle reshape
        N C1 ho*16 16*hi C0 -> f0: dma and storage-align
        N C1 16*hi C0 ho*16 -> v0
        N 16*hi C1 C0 ho*16 -> v1
        N 16*hi Cx ho*16 -> f1
        N 16*hi C ho*16 -> depad
        N ho*16 16*hi C -> v2
        N (ho*16)*(16*hi) C -> f2
        N H C --> res
        1. make mte2_tensor copy data from gm to ub by shape of h_pad_tensor.2
        2. storage-align f0_reshape_tensor to avoid bank-conflict.
        3. storage-align transpose_2_tensor to help simply.
        Wait:
        1. dynamic storage-align
        """
        sch = self.schedule
        idx = self._seek_h_idx(self.mte2_tensor)
        if idx - 1 >= 0:
            # align for fake-pad-h
            shape = shape_to_list(self.mte2_tensor.shape)
            factor = self.align_factor * self.hi * math_prod(shape[idx + 1:])
            sch[self.mte2_tensor].storage_align(self.mte2_tensor.op.axis[idx - 1], factor, 0)

        idx = self._seek_h_idx(self.f0_reshape_tensor)
        if idx - 1 >= 0:
            # align for avoid bank-conflict
            var = self.hi + 1
            shape = shape_to_list(self.f0_reshape_tensor.shape)
            # last-dim is C0, hi must be 16*x
            factor = math_prod(shape[idx + 1:]) * var
            sch[self.f0_reshape_tensor].bind_buffer(self.f0_reshape_tensor.op.axis[idx - 1], factor, 0)

        idx = self._seek_h_idx(self.transpose_2_tensor)
        if idx - 1 >= 0:
            # align for avoid simply
            sch[self.transpose_2_tensor].storage_align(self.transpose_2_tensor.op.axis[idx - 1],
                                                       self.align_factor, 0)

    def _do_mem_reused(self):
        sch = self.schedule
        sch[self.mte2_tensor].reused_by(self.h_pad_tensor)
        sch[self.mte2_tensor].reused_by(*self.graph_info.s_reshape_tensor_set)
        sch[self.transpose_1_tensor].reused_by(self.f1_reshape_tensor)
        sch[self.transpose_2_tensor].reused_by(self.f2_reshape_tensor)

    def _transpose_emit_insn(self, tensor, perm):
        emit_idx, insn = choose_transpose_insn(perm)
        if insn in ["vector_transpose", "phony_insn"]:
            self.vnchwconv_insn_map(tensor, insn, tensor.op.axis[emit_idx], perm)
        else:
            self.emit_insn_map[tensor] = {"scope": tensor.op.axis[0], "instruction": insn}

    def _calc_emit_insn(self):
        _map = self.emit_insn_map
        _map.clear()

        _map[self.mte2_tensor] = {"scope": self.mte2_tensor.op.axis[0], "instruction": "dma_copy"}
        _map[self.h_pad_tensor] = {"scope": self.h_pad_tensor.op.axis[0], "instruction": "phony_insn"}

        for tensor in self.graph_info.s_reshape_tensor_set:
            _map[tensor] = {"scope": tensor.op.axis[0], "instruction": "phony_insn"}
        _map[self.f0_reshape_tensor] = {"scope": self.f0_reshape_tensor.op.axis[0], "instruction": "dma_copy"}

        self._transpose_emit_insn(self.transpose_0_tensor, self.perm_0)
        self._transpose_emit_insn(self.transpose_1_tensor, self.perm_1)
        _map[self.f1_reshape_tensor] = {"scope": self.f1_reshape_tensor.op.axis[0], "instruction": "phony_insn"}

        for tensor in self.depad_tensors:
            _map[tensor] = {"scope": tensor.op.axis[0], "instruction": "dma_copy"}
        self._transpose_emit_insn(self.transpose_2_tensor, self.perm_2)
        _map[self.f2_reshape_tensor] = {"scope": self.f2_reshape_tensor.op.axis[0], "instruction": "phony_insn"}

        _map[self.tiling_tensor] = {"scope": self.ub_inner, "instruction": "dma_copy",
                                    "attrs": {NO_OVERLAP: 2, "no_overlap_malloc_buf_for_tail": 0}}

    def _do_pragma(self):
        def _pragma(tensor, begin, end):
            axes = [tensor.op.axis[i] for i in range(begin, end)]
            self._do_group(tensor, axes)

        num = 0
        for k in self.axis_not_in_ub:
            if k not in self.axis_in_ub:
                num += 1
        idx = self._seek_h_idx(self.f0_reshape_tensor)
        _pragma(self.f0_reshape_tensor, num, idx)
        _pragma(self.transpose_1_tensor, len(self.transpose_1_tensor.shape) - 2, len(self.transpose_1_tensor.shape))
        for tensor in self.depad_tensors:
            _pragma(tensor, len(tensor.shape) - 2, len(tensor.shape))

    def _analysis_transpose(self):
        # Func: judge transpose work or not
        self.perm_2 = self.update_ub_perm(self.transpose_2_tensor)
        self.perm_1 = self.update_ub_perm(self.transpose_1_tensor, transpose_work=self.tiling_case.transpose_work)
        self.perm_0 = self.update_ub_perm(self.transpose_0_tensor)

    def _seek_h_idx(self, tensor):
        h = int(self.h_pad_tensor.op.attrs["axes"])
        if tensor in self.get_all_producers_stages(self.h_pad_tensor) or tensor == self.h_pad_tensor:
            return h

        src_tensor = self.h_pad_tensor
        while src_tensor != tensor:
            src_tensor = self.child(src_tensor)
            tag = src_tensor.op.tag
            if tag.find("transdata|transpose") != -1:
                perm = [int(x) for x in list(src_tensor.op.attrs["permute"])]
                h = perm.index(h)
            elif tag.find("transdata|f_reshape") != -1:
                axes = get_reshape(src_tensor)
                h = h + 1 if h not in axes else axes.index(h)
            elif tag.find("transdata|s_reshape") != -1:
                axes = get_reshape(src_tensor)
                h = axes[h] if isinstance(axes[h], int) else axes[h][1]
        return h
