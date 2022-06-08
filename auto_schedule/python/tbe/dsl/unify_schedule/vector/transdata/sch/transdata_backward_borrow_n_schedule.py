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
transdata backward borrow n
"""
from tbe.dsl.base.operation import var_inner
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.unify_schedule.constants import TransdataCategory
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from ..common.transdata_base_sch import TransdataBaseSch
from ..common.transdata_graph_info import choose_transpose_insn
from ..common.constants import BLOCK, STRIDE_2, STRIDE_3, NO_OVERLAP


class TransBackwardBorrowNSchedule(TransdataBaseSch):
    """
    TransBackwardBorrowNSchedule: backward + n-last-transpose + borrow n
    """

    def __init__(self, outs, tiling_case):
        TransdataBaseSch.__init__(self, outs, tiling_case)
        self.n1_index = None
        self.n0_index = None
        self.c1_index = None
        self.c0_index = None
        self.n_mapping_indexes = []
        self.c_mapping_indexes = []

        self.crd_tensor = None
        self.pad_tensor = None
        self.s_reshape_tensor = None
        self.f_reshape_0_tensor = None
        self.f_reshape_1_tensor = None

        self.transpose_0_tensor = None
        self.transpose_1_tensor = None
        self.transpose_2_tensor = None

        self.perm_2 = []
        self.perm_1 = []
        self.perm_0 = []

    @classmethod
    def get_supported_sub_pattern(cls):
        return TransdataCategory.BORROW_N_B8B16_BACKWARD

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
                gm -> mte2_tensor -> crd_tensor -> pad_tensor -> s_reshape_tensor -> transpose_0
            main-stream:
                transpose_0 -> transpose_1 -> f_reshape_0 -> depad_tensor
            post-stream:
                depad_tensor -> transpose_2 -> f_reshape_1 -> res
        """
        self.mte2_tensor = self.child(list(self.graph_info.input_tensor_set)[0])
        self.crd_tensor = self.single_cache_read(self.mte2_tensor)

        self.pad_tensor = list(self.graph_info.pad_tensor_set)[0]
        self.s_reshape_tensor = list(self.graph_info.s_reshape_tensor_set)[0]
        self.transpose_0_tensor = self.child(self.s_reshape_tensor)
        self.transpose_1_tensor = self.child(self.transpose_0_tensor)
        self.f_reshape_0_tensor = self.child(self.transpose_1_tensor)

        self.depad_tensors = list(self.graph_info.de_pad_tensor_set)
        self.depad_tensors.sort(key=lambda x: int(x.op.attrs["axes"]))
        self.depad_axis_list = [int(x.op.attrs["axes"]) for x in self.depad_tensors]

        self.transpose_2_tensor = self.child(self.depad_tensors[-1])
        self.f_reshape_1_tensor = self.child(self.transpose_2_tensor)
        self.tiling_tensor = list(self.graph_info.output_tensor_set)[0]

    def _calc_tiling(self):
        self.align_factor = BLOCK // DTYPE_BYTE_MAPPING.get(self.tiling_tensor.dtype, 1)
        self.split_once = self.tiling_case.ub_split_second_idx == self.tiling_case.ub_split_first_idx
        if not self.split_once:
            dict_args = {"errCode": "E90003", "detailed_cause": "only support ub split once"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        # define factor
        if not self.tiling_case.ub_first_factor:
            self.tiling_case.ub_first_factor = var_inner("_ub_first_factor", (1, None))
        if not self.tiling_case.block_factor:
            self.tiling_case.block_factor = var_inner("_block_factor", (1, None))

        # regulation
        perm = [int(x) for x in list(self.transpose_2_tensor.op.attrs["permute"])]
        self.parses_axis_type(perm)
        self.n_mapping_indexes, self.n1_index, self.n0_index = self.parses_f_reshape(self.f_reshape_1_tensor)
        self.parses_tiling_info(self.n_mapping_indexes, self.n1_index, self.n0_index, self.pad_factor)

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

        self.c_mapping_indexes, self.c1_index, self.c0_index = self.parses_f_reshape(self.f_reshape_0_tensor)

        if self.c1_index is not None:
            c_idx = self.c_mapping_indexes[self.c1_index]
            align(self.f_reshape_0_tensor, [c_idx, ], self.pad_factor)

        if self.n1_index is not None:
            n_idx = self.n_mapping_indexes[self.n1_index]
            align(self.f_reshape_1_tensor, [n_idx, ], self.pad_factor)

    def _do_storage_align(self):
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

    def _do_mem_reused(self):
        sch = self.schedule
        sch[self.crd_tensor].reused_by(self.pad_tensor)
        sch[self.pad_tensor].reused_by(self.s_reshape_tensor)
        sch[self.transpose_1_tensor].reused_by(self.f_reshape_0_tensor)
        sch[self.transpose_2_tensor].reused_by(self.f_reshape_1_tensor)

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
                                    "attrs": {NO_OVERLAP: 3, "no_overlap_malloc_buf_for_tail": 0}}

    def _do_pragma(self):
        def _pragma(tensor, begin, end):
            axes = [tensor.op.axis[i] for i in range(begin, end)]
            self._do_group(tensor, axes)

        _pragma(self.transpose_2_tensor, STRIDE_2, len(self.transpose_2_tensor.shape))
        _pragma(self.transpose_0_tensor, 1, len(self.transpose_0_tensor.shape) - 1)
        _pragma(self.crd_tensor, 1, len(self.crd_tensor.shape))

    def _analysis_transpose(self):
        # Func: judge transpose work or not
        self.perm_2 = self.update_ub_perm(self.transpose_2_tensor)
        self.perm_1 = self.update_ub_perm(self.transpose_1_tensor, transpose_work=self.tiling_case.transpose_work)
        self.perm_0 = self.update_ub_perm(self.transpose_0_tensor)
