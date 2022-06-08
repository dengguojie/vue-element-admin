#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
reduce group schedule
"""
import copy

from tbe import tvm
from tbe.common.platform import ASCEND_910B
from tbe.common.platform.platform_info import get_soc_spec
from tbe.dsl.base.operation import var_inner
from tbe.dsl.base.operation import get_context

from .reduce_tilingcase import ReduceTilingCase
from .reduce_tilingcase import SingleReduceInfo
from .vector_info import ComputeGraphInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import FAKE_NODE_TAG
from ...constants import INSN_MAPPING

BLOCK_IDX = "blockIdx.x"
BLOCK_SIZE_BYTE = 32
CONST = "const"
LOCAL_UB = "local.UB"


def _get_insn(tensor):
    """
    get insn of tensor
    """
    tag = tensor.op.tag
    insn = tag.split("|")[0] if tensor.op.tag.find("|") != -1 else tag

    return INSN_MAPPING.get(insn, insn)


class ReduceGroupSchedule:
    """
    Schedule for Group Reduce
    """

    def __init__(self, graph_info: ComputeGraphInfo, reduce_info: SingleReduceInfo):
        self._outs = None
        self._tiling_case = None
        self._sch = None

        self._graph_info = graph_info
        self._reduce_info = reduce_info

        self._scope = LOCAL_UB

        self._forward_graph_map = graph_info.tensor_consumers_map
        self._backward_graph_map = graph_info.tensor_producers_map

        self._after_reduce_tensor_set = set()
        self._before_reduce_tensor_set = set()

        # mid tensor include pure mid tensor and non gm input tensor
        self._mid_tensor_set = graph_info.mid_tensor_set | graph_info.non_gm_input_tensor_set
        # real output except mid output tensor
        self._real_pure_output = graph_info.real_output_tensor_set - graph_info.mid_output_tensor_set
        # get last endpoint output tensor
        self._res_tensor = list(graph_info.endpoint_output_tensor_set)[0]

        self._reduce_tensor = None
        self._reduce_rf_tensor = None
        self._reduce_rf_ub_tensor = None
        self._reduce_rf_reread_ub_tensor = None

        self._cache_read_ori_and_cur_tensor_map = {}
        self._cache_read_cur_and_ori_tensor_map = {}

        self._cache_read_and_align_pad_tensor_map = {}
        self._align_pad_and_cache_read_tensor_map = {}

        self._cache_write_ori_and_cur_tensor_map = {}
        self._cache_write_cur_and_ori_tensor_map = {}

        self._block_split_result = {}
        self._ub_split_result = {}

        self._reorder_map = {}

        self._multi_core_bind_axis = None
        self._block_num = 1

        self._storage_align_map = {}
        self._bind_buffer_map = {}
        self._compute_align_map = {}

        self._compute_at_map = {}
        self._compute_root_tensors = set()

        self._emit_insn_map = {}

    def do_schedule(self, outs, tiling_case: ReduceTilingCase):
        """
        do group reduce schedule
        """
        self._outs = outs
        self._tiling_case = tiling_case

        self._create_schedule()

        self._do_cache_read()

        self._do_align_pad()

        self._do_cache_write()

        self._set_scope()

        self._do_mid_output_tensor_process()

        self._do_tiling_and_rfactor()

        self._calc_reorder()
        self._do_reorder()

        self._do_common_tensor_categorization()

        self._calc_storage_align()
        self._do_storage_align()

        self._calc_bind_buffer()
        self._do_bind_buffer()

        self._do_set_buffer_size()

        self._do_set_constraint()

        self._do_multi_core_and_set_store_predicate()

        self._do_block_sync()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._calc_compute_align()
        self._do_compute_align()

        self._do_pragma()

        return self._sch

    def _create_schedule(self):
        self._sch = tvm.create_schedule([self._res_tensor.op])

    def _do_cache_read(self):
        for cache_read_tensor in self._graph_info.input_tensor_set:
            read_buffer = self._sch.cache_read(
                cache_read_tensor, self._scope, self._forward_graph_map.get(cache_read_tensor))
            self._cache_read_cur_and_ori_tensor_map[read_buffer] = cache_read_tensor
            self._cache_read_ori_and_cur_tensor_map[cache_read_tensor] = read_buffer

    def _do_align_pad(self):
        if not self._tiling_case.is_reduce_pad_case:
            return

        for cache_read_tensor in self._graph_info.input_tensor_set:
            # after last reduce tensor can not align pad
            if cache_read_tensor in self._graph_info.tensors_after_reduce and self._reduce_info.is_reduce_last_axis():
                continue
            cache_read_buffer = self._cache_read_ori_and_cur_tensor_map.get(cache_read_tensor)
            align_pad_buffer = self._sch.cache_read(cache_read_buffer, self._scope,
                                                    self._forward_graph_map.get(cache_read_tensor))
            self._cache_read_and_align_pad_tensor_map[cache_read_buffer] = align_pad_buffer
            self._align_pad_and_cache_read_tensor_map[align_pad_buffer] = cache_read_buffer

    def _do_cache_write(self):
        for cache_write_tensor in self._real_pure_output:
            write_buffer = self._sch.cache_write(cache_write_tensor, self._scope)
            self._cache_write_cur_and_ori_tensor_map[write_buffer] = cache_write_tensor
            self._cache_write_ori_and_cur_tensor_map[cache_write_tensor] = write_buffer

    def _set_scope(self):
        for pure_mid_tensor in self._mid_tensor_set - self._graph_info.real_output_tensor_set:
            self._sch[pure_mid_tensor].set_scope(self._scope)

    def _do_mid_output_tensor_process(self):
        for single_mid_output_tensor in self._graph_info.mid_output_tensor_set:
            write_buffer = self._sch.cache_write(single_mid_output_tensor, self._scope)
            self._cache_write_cur_and_ori_tensor_map[write_buffer] = single_mid_output_tensor
            self._cache_write_ori_and_cur_tensor_map[single_mid_output_tensor] = write_buffer
            read_buffer = self._sch.cache_read(
                single_mid_output_tensor, self._scope, self._forward_graph_map.get(single_mid_output_tensor))
            self._cache_read_cur_and_ori_tensor_map[read_buffer] = single_mid_output_tensor
            self._cache_read_ori_and_cur_tensor_map[single_mid_output_tensor] = read_buffer
            self._sch[write_buffer].reused_by(read_buffer)
            self._sch[read_buffer].reused_by(reuse_data=True)

    def _do_tiling_and_rfactor(self):
        self._reduce_tensor = self._cache_write_ori_and_cur_tensor_map.get(self._reduce_info.reduce_tensor,
                                                                           self._reduce_info.reduce_tensor)
        # block split R axis on reduce_tensor
        block_factor = self._tiling_case.block_factor
        block_split_factor = block_factor if block_factor is not None else var_inner("_block_factor", (1, None))
        self._block_split_result["factor"] = block_split_factor
        block_split_axis_index = self._tiling_case.block_split_axis_index
        block_split_reduce_axis_index = self._reduce_info.reduce_axis_indexes.index(block_split_axis_index)
        block_axis_var = self._reduce_tensor.op.reduce_axis[block_split_reduce_axis_index]
        block_outer, block_inner = self._sch[self._reduce_tensor].split(block_axis_var, factor=block_split_factor)
        self._block_split_result["outer_itervar"] = block_outer
        self._block_split_result["inner_itervar"] = block_inner

        # reduce_rf_tensor is workspace tensor
        self._reduce_rf_tensor = self._sch.rfactor(self._reduce_tensor, block_outer, block_split_axis_index)
        self._reduce_rf_ub_tensor = self._sch.cache_write(self._reduce_rf_tensor, self._scope)
        self._reduce_rf_reread_ub_tensor = \
            self._sch.cache_read(self._reduce_rf_tensor, self._scope, [self._reduce_tensor])
        self._sch[self._reduce_rf_tensor].set_scope("")
        self._outs.append(self._reduce_rf_tensor)

        # ub split R or A axis that after block_split_axis on reduce_rf_ub_tensor
        ub_factor = self._tiling_case.ub_factor
        ub_split_factor = ub_factor if ub_factor is not None else var_inner("_ub_factor", (1, None))
        self._ub_split_result["factor"] = ub_split_factor
        ub_split_axis_index = self._tiling_case.ub_split_axis_index
        # split R
        if ub_split_axis_index in self._reduce_info.reduce_axis_indexes:
            # some reduce axes have been fused to split block
            ub_tiling_reduce_axis_index = \
                self._reduce_info.reduce_axis_indexes.index(ub_split_axis_index) - block_split_reduce_axis_index
            ub_axis_var = self._sch[self._reduce_rf_ub_tensor].op.reduce_axis[ub_tiling_reduce_axis_index]
        else:
            ub_tiling_axis_index = ub_split_axis_index
            ub_axis_var = self._sch[self._reduce_rf_ub_tensor].op.axis[ub_tiling_axis_index]
        ub_outer, ub_inner = self._sch[self._reduce_rf_ub_tensor].split(ub_axis_var, factor=ub_split_factor)
        self._ub_split_result["outer_itervar"] = ub_outer
        self._ub_split_result["inner_itervar"] = ub_inner

    def _calc_reorder(self):
        def _calc_reduce_rf_reorder_axis():
            count_common_axis = 0
            count_reduce_axis = 0
            reduce_rf_reorder_axis = []

            for index, _ in enumerate(self._reduce_info.shape_before_reduce):
                if index == self._tiling_case.block_split_axis_index:
                    cur_common_axis = self._sch[self._reduce_rf_ub_tensor].op.axis[count_common_axis]
                    reduce_rf_reorder_axis.append(cur_common_axis)
                    count_common_axis += 1

                if index in self._reduce_info.reduce_axis_indexes:
                    if index == self._tiling_case.ub_split_axis_index:
                        reduce_rf_reorder_axis.append(self._ub_split_result.get("outer_itervar"))
                        reduce_rf_reorder_axis.append(self._ub_split_result.get("inner_itervar"))
                    else:
                        cur_reduce_axis = self._sch[self._reduce_rf_ub_tensor].op.reduce_axis[count_reduce_axis]
                        reduce_rf_reorder_axis.append(cur_reduce_axis)
                    count_reduce_axis += 1

                    if self._reduce_info.keepdims:
                        count_common_axis += 1
                else:
                    if index == self._tiling_case.ub_split_axis_index:
                        reduce_rf_reorder_axis.append(self._ub_split_result.get("outer_itervar"))
                        reduce_rf_reorder_axis.append(self._ub_split_result.get("inner_itervar"))
                    else:
                        cur_common_axis = self._sch[self._reduce_rf_ub_tensor].op.axis[count_common_axis]
                        reduce_rf_reorder_axis.append(cur_common_axis)
                    count_common_axis += 1

            self._reorder_map[self._reduce_rf_ub_tensor] = reduce_rf_reorder_axis

        def _calc_reduce_reorder_axis():
            count_common_axis = 0
            count_reduce_axis = 0
            reduce_reorder_axis = []

            for index, _ in enumerate(self._reduce_info.shape_before_reduce):
                if index in self._reduce_info.reduce_axis_indexes:
                    if index <= self._tiling_case.block_split_axis_index:
                        cur_reduce_axis = self._sch[self._reduce_tensor].op.reduce_axis[count_reduce_axis]
                        reduce_reorder_axis.append(cur_reduce_axis)
                        count_reduce_axis += 1

                    if self._reduce_info.keepdims:
                        count_common_axis += 1
                else:
                    cur_common_axis = self._sch[self._reduce_tensor].op.axis[count_common_axis]
                    reduce_reorder_axis.append(cur_common_axis)
                    count_common_axis += 1

            self._reorder_map[self._reduce_tensor] = reduce_reorder_axis

        _calc_reduce_rf_reorder_axis()
        _calc_reduce_reorder_axis()

    def _do_reorder(self):
        for single_tensor, param in self._reorder_map.items():
            self._sch[single_tensor].reorder(*param)

    def _do_common_tensor_categorization(self):
        def _categorize(_ori_tensor, _cur_tensor):
            if _ori_tensor in self._graph_info.tensors_before_reduce:
                self._before_reduce_tensor_set.add(_cur_tensor)
            else:
                self._after_reduce_tensor_set.add(_cur_tensor)

        for single_tensor in self._mid_tensor_set:
            if single_tensor == self._reduce_tensor:
                continue
            _categorize(single_tensor, single_tensor)

        for ori_tensor, cur_tensor in self._cache_read_ori_and_cur_tensor_map.items():
            _categorize(ori_tensor, cur_tensor)

        for cache_read_tensor, align_pad_tensor in self._cache_read_and_align_pad_tensor_map.items():
            ori_tensor = self._cache_read_cur_and_ori_tensor_map.get(cache_read_tensor)
            _categorize(ori_tensor, align_pad_tensor)

        for ori_tensor, cur_tensor in self._cache_write_ori_and_cur_tensor_map.items():
            if cur_tensor == self._reduce_tensor:
                continue
            _categorize(ori_tensor, cur_tensor)

    def _calc_storage_align(self):
        align_index = -2

        # before reduce stage
        for before_reduce_tensor in self._before_reduce_tensor_set:
            align_factor = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(before_reduce_tensor.dtype))
            self._storage_align_map[before_reduce_tensor] = \
                [before_reduce_tensor.op.axis[align_index], align_factor, 0]

        # if is reduce last axis, after reduce tensor don't need to storage align
        if not self._reduce_info.is_reduce_last_axis():
            for after_reduce_tensor in self._after_reduce_tensor_set:
                align_factor = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(after_reduce_tensor.dtype))
                self._storage_align_map[after_reduce_tensor] = \
                    [after_reduce_tensor.op.axis[align_index], align_factor, 0]

        reduce_align_factor = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self._reduce_tensor.dtype))
        self._storage_align_map[self._reduce_rf_ub_tensor] = \
            [self._sch[self._reduce_rf_ub_tensor].op.axis[align_index], reduce_align_factor, 0]
        self._storage_align_map[self._reduce_rf_reread_ub_tensor] = \
            [self._sch[self._reduce_rf_reread_ub_tensor].op.axis[align_index], reduce_align_factor, 0]
        if not self._reduce_info.is_reduce_last_axis():
            self._storage_align_map[self._reduce_tensor] = \
                [self._sch[self._reduce_tensor].op.axis[align_index], reduce_align_factor, 0]

        # if cache read tensor has been aligned pad in ub, it cannot do storage align
        for single_tensor in self._cache_read_and_align_pad_tensor_map:
            self._storage_align_map.pop(single_tensor)

    def _do_storage_align(self):
        for single_tensor, param in self._storage_align_map.items():
            self._sch[single_tensor].storage_align(param[0], param[1], param[2])

    def _calc_bind_buffer(self):
        align_factor = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self._reduce_tensor.dtype))
        bind_axis = len(self._reduce_rf_tensor.shape) - 2
        bind_factor = \
            tvm.div((self._reduce_rf_tensor.shape[bind_axis + 1] + align_factor - 1), align_factor) * align_factor
        self._bind_buffer_map[self._reduce_rf_tensor] = \
            [self._sch[self._reduce_rf_tensor].op.axis[bind_axis], bind_factor, 0]

    def _do_bind_buffer(self):
        for single_tensor, param in self._bind_buffer_map.items():
            self._sch[single_tensor].bind_buffer(param[0], param[1], param[2])

    def _do_set_buffer_size(self):
        for before_reduce_tensor in self._before_reduce_tensor_set:
            self._sch[before_reduce_tensor].set_buffer_size(self._tiling_case.tensor_ub_size_before_reduce)
        for after_reduce_tensor in self._after_reduce_tensor_set:
            self._sch[after_reduce_tensor].set_buffer_size(self._tiling_case.tensor_ub_size_after_reduce)
        self._sch[self._res_tensor].set_buffer_size(self._tiling_case.tensor_ub_size_after_reduce)

        self._sch[self._reduce_rf_ub_tensor].set_buffer_size(self._tiling_case.tensor_ub_size_before_reduce)
        self._sch[self._reduce_rf_reread_ub_tensor].set_buffer_size(self._tiling_case.tensor_ub_size_before_reduce)
        self._sch[self._reduce_tensor].set_buffer_size(self._tiling_case.tensor_ub_size_after_reduce)

    def _do_set_constraint(self):
        if get_context().get_current_compute().get("_mode") == CONST:
            return

        ub_size = self._tiling_case.tensor_ub_size_before_reduce
        # first stage constraint
        first_stage_shape_in_ub = 1
        for index, value in enumerate(self._reduce_info.shape_before_reduce):
            if index < self._tiling_case.ub_split_axis_index:
                continue
            if index == self._tiling_case.ub_split_axis_index:
                self._sch.set_constraint(self._ub_split_result.get("factor") <= ub_size)
                first_stage_shape_in_ub *= self._ub_split_result.get("factor")
            else:
                self._sch.set_constraint(value <= ub_size)
                first_stage_shape_in_ub *= value

        self._sch.set_constraint(first_stage_shape_in_ub <= ub_size)

        # core num constraint
        for index, value in enumerate(self._reduce_info.shape_before_reduce):
            if index == self._tiling_case.block_split_axis_index:
                cur_block_num =\
                    (value + self._block_split_result.get("factor") - 1) // self._block_split_result.get("factor")
                self._sch.set_constraint(cur_block_num <= get_soc_spec("CORE_NUM"))
                self._block_num *= cur_block_num
                break
            self._sch.set_constraint(value <= get_soc_spec("CORE_NUM"))
            self._block_num *= value

        # second stage constraint
        second_stage_shape_in_ub = self._block_num
        for index, value in enumerate(self._reduce_info.shape_before_reduce):
            if index <= self._tiling_case.block_split_axis_index:
                continue
            if index in self._reduce_info.reduce_axis_indexes:
                continue
            second_stage_shape_in_ub *= value

        self._sch.set_constraint(second_stage_shape_in_ub <= ub_size)

    def _do_multi_core_and_set_store_predicate(self):
        fuse_axis_list = self._sch[self._reduce_rf_tensor].op.axis[:self._tiling_case.block_split_axis_index + 1]
        self._multi_core_bind_axis = self._sch[self._reduce_rf_tensor].fuse(*fuse_axis_list)
        block = tvm.thread_axis(BLOCK_IDX)
        self._sch[self._reduce_rf_tensor].bind(self._multi_core_bind_axis, block)

        set_store_predicate_tensor_set = \
            self._after_reduce_tensor_set | {self._reduce_rf_reread_ub_tensor, self._reduce_tensor, self._res_tensor}
        for after_reduce_tensor in set_store_predicate_tensor_set:
            self._sch[after_reduce_tensor].set_store_predicate(block.var.equal(self._block_num - 1),
                                                               partition=False, rebase_root=True)

    def _do_block_sync(self):
        sync_tensor = self._sch.create_block_sync()
        # sync axis is the axis after self._multi_core_bind_axis
        sync_axis = self._sch[self._reduce_rf_tensor].leaf_iter_vars[1]
        self._sch[self._reduce_rf_tensor].wait_block_sync(sync_axis, tensor=sync_tensor, bottom=True)
        self._sch[self._reduce_rf_tensor].set_block_sync(sync_axis, tensor=sync_tensor, bottom=True)

    def _calc_compute_at(self):
        for before_reduce_tensor in self._before_reduce_tensor_set:
            self._compute_at_map[before_reduce_tensor] = \
                [self._reduce_rf_ub_tensor, self._ub_split_result.get("outer_itervar")]
        self._compute_at_map[self._reduce_rf_ub_tensor] = [self._reduce_rf_tensor, self._multi_core_bind_axis]

        for after_reduce_tensor in self._after_reduce_tensor_set:
            self._compute_root_tensors.add(after_reduce_tensor)
        self._compute_root_tensors.add(self._reduce_rf_reread_ub_tensor)
        self._compute_root_tensors.add(self._reduce_tensor)

    def _do_compute_at(self):
        for single_tensor, param in self._compute_at_map.items():
            self._sch[single_tensor].compute_at(self._sch[param[0]], param[1])
        for compute_root_tensor in self._compute_root_tensors:
            self._sch[compute_root_tensor].compute_root()

    def _calc_emit_insn(self):
        def __handle_special_tensors():
            reduce_attrs = {"storage_bound": self._tiling_case.tensor_ub_size_before_reduce}
            self._emit_insn_map[self._reduce_rf_ub_tensor] = \
                [self._ub_split_result.get("inner_itervar"), _get_insn(self._reduce_tensor), reduce_attrs]

            if self._reduce_info.is_reduce_last_axis():
                reduce_tensor_emit_axis = self._sch[self._reduce_tensor].op.axis[0]
            else:
                reduce_tensor_emit_axis = self._sch[self._reduce_tensor].op.reduce_axis[0]
            self._emit_insn_map[self._reduce_tensor] = \
                [reduce_tensor_emit_axis, _get_insn(self._reduce_tensor), reduce_attrs]

            attrs = {"no_overlap": 1}
            if self._tiling_case.block_split_axis_index != len(self._reduce_info.shape_before_reduce) - 1 and \
                    self._tiling_case.ub_split_axis_index != len(self._reduce_info.shape_before_reduce) - 1:
                attrs = {"no_overlap": 0}
            self._emit_insn_map[self._reduce_rf_tensor] = \
                [self._sch[self._reduce_rf_tensor].leaf_iter_vars[1], "dma_copy", attrs]

            self._emit_insn_map[self._reduce_rf_reread_ub_tensor] = \
                [self._reduce_rf_reread_ub_tensor.op.axis[0], "dma_copy", attrs]

        for ori_tensor, cur_tensor in self._cache_read_ori_and_cur_tensor_map.items():
            # recache read buffer of mid output tensor does not do dma_copy
            insn = "phony_insn" if ori_tensor in self._graph_info.mid_output_tensor_set else "dma_copy"
            self._emit_insn_map[cur_tensor] = [cur_tensor.op.axis[0], insn]

        for single_tensor in self._mid_tensor_set - self._graph_info.real_output_tensor_set:
            if single_tensor == self._reduce_tensor:
                continue
            self._emit_insn_map[single_tensor] = [single_tensor.op.axis[0], _get_insn(single_tensor)]

        for ori_tensor, cur_tensor in self._cache_write_ori_and_cur_tensor_map.items():
            if cur_tensor == self._reduce_tensor:
                continue
            self._emit_insn_map[cur_tensor] = [cur_tensor.op.axis[0], _get_insn(ori_tensor)]

        for single_tensor in self._align_pad_and_cache_read_tensor_map:
            self._emit_insn_map[single_tensor] = [single_tensor.op.axis[0], "align_pad"]

        for out_tensor in self._graph_info.real_output_tensor_set:
            self._emit_insn_map[out_tensor] = [out_tensor.op.axis[0], "dma_copy"]

        if self._res_tensor.op.tag == FAKE_NODE_TAG:
            self._emit_insn_map[self._res_tensor] = [self._res_tensor.op.axis[0], "phony_insn"]

        __handle_special_tensors()

    def _do_emit_insn(self):
        for single_tensor, param in self._emit_insn_map.items():
            if len(param) <= 2:
                self._sch[single_tensor].emit_insn(param[0], param[1])
            else:
                self._sch[single_tensor].emit_insn(param[0], param[1], attrs=param[2])

    def _calc_compute_align(self):
        if get_soc_spec("SHORT_SOC_VERSION") != ASCEND_910B:
            return

        for single_tensor, param in self._storage_align_map.items():
            insn = self._emit_insn_map.get(single_tensor)[1]
            if insn == "dma_copy":
                continue
            if self._reduce_info.is_reduce_last_axis() and insn in ["reduce_prod", "reduce_sum"]:
                continue
            factor = param[1]
            # last reduce tensor
            if self._sch[single_tensor].op.reduce_axis is not None and self._reduce_info.is_reduce_last_axis():
                axis = self._sch[single_tensor].op.reduce_axis[-1]
            else:
                axis = self._sch[single_tensor].op.axis[-1]

            self._compute_align_map[single_tensor] = [axis, factor, None]

    def _do_compute_align(self):
        for single_tensor, param in self._compute_align_map.items():
            self._sch[single_tensor].compute_align(param[0], param[1], param[2])

    def _do_pragma(self):
        def __mark_group_axis_on_common_tensor(_single_tensor, _tensor_type="before_reduce"):
            # axis_group = 0 means original no_fuse branch will be overwrited by fuse branch
            # axis_group = 1 means fuse branch will be appended after original no_fuse branch
            append_id = tvm.make.Call("int32", "axis_group", [1, "append"], tvm.expr.Call.Extern, None, 0)
            overwrite_and_append_id = tvm.make.Call("int32", "axis_group", [0, "overwrite", 1, "append"],
                                                    tvm.expr.Call.Extern, None, 0)

            start_index = self._tiling_case.ub_split_axis_index if _tensor_type == "before_reduce" else 0
            for index, value in enumerate(self._sch[_single_tensor].leaf_iter_vars):
                if index < start_index:
                    continue
                group_id = overwrite_and_append_id
                if index == len(self._sch[_single_tensor].leaf_iter_vars) - 1:
                    group_id = append_id
                self._sch[_single_tensor].pragma(value, "axis_group", group_id)

        disable_group_axis_tensor_set = set(self._align_pad_and_cache_read_tensor_map.keys())

        for before_reduce_tensor in self._before_reduce_tensor_set | {self._reduce_rf_tensor}:
            if before_reduce_tensor in disable_group_axis_tensor_set:
                continue
            __mark_group_axis_on_common_tensor(before_reduce_tensor, "before_reduce")

        for after_reduce_tensor in \
                self._after_reduce_tensor_set | {self._reduce_rf_reread_ub_tensor, self._res_tensor}:
            if after_reduce_tensor in disable_group_axis_tensor_set:
                continue
            __mark_group_axis_on_common_tensor(after_reduce_tensor, "after_reduce")
