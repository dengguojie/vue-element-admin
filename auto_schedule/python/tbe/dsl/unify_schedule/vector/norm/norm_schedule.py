#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
norm schedule
"""
from tbe import tvm
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import var_inner

from ...constants import INSN_MAPPING
from ...constants import NormPattern
from ...constants import Pattern
from ...schedule import Schedule

from .norm_tilingcase import get_block_size as get_align_factor
from .norm_tilingcase import reorder_reduce_shape

BLOCK_IDX = "blockIdx.x"
LOCAL_UB = "local.UB"
NO_OVERLAP = "no_overlap"
STORAGE_BOUND = "storage_bound"


def _get_insn(tensor):
    """
    get insn
    """
    tag = tensor.op.tag
    if tensor.op.tag.find("|") != -1:
        insn = tag.split("|")[0]
    else:
        insn = tag
    return INSN_MAPPING.get(insn, insn)


def _add_sub_dict_to_dict(_map, _key, _sub_key, _sub_value):
    """
    add sub dict to dict
    """
    if _key not in _map:
        _map[_key] = {}
    _map[_key][_sub_key] = _sub_value


class EntryNormSchedule(Schedule):
    """
    entrance to norm schedule
    """
    def __init__(self, outs, tiling_case):
        self.outs = outs
        self.tiling_case = tiling_case

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return ["default"]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.NORM]

    @classmethod
    def get_supported_sub_pattern(cls):
        return [NormPattern.N_0]

    def do_schedule(self):
        # Get Compute Graph Info
        current_compute = get_context().get_current_compute()
        norm_compute_graph_info = current_compute.get("_compute_graph_info")
        norm_info = current_compute.get("_norm_info")

        if self.tiling_case.ub_split_axis_index not in norm_info.reduce_axis_indices:
            norm_sch = NormNormalSchedule(norm_compute_graph_info, norm_info, self.tiling_case, self.outs)
        else:
            norm_sch = NormWorkspaceSchedule(norm_compute_graph_info, norm_info, self.tiling_case, self.outs)
        real_schedule = norm_sch.do_schedule()

        return real_schedule


class NormNormalSchedule:
    """
    norm schedule when ub split common axis
    """
    def __init__(self, graph_info, norm_info, tiling_case, outs):
        self._outs = outs
        self._sch = None
        self._scope = LOCAL_UB

        self._graph_info = graph_info
        self._forward_compute_graph_map = graph_info.tensor_consumers_map
        self._backward_compute_graph_map = graph_info.tensor_producers_map

        # get last endpoint output tensor
        self._res_tensor = tuple(graph_info.endpoint_output_tensor_set)[0]
        self._norm_info = norm_info
        self._tiling_case = tiling_case

        self._cache_read_tensors = set()
        self._cache_read_buffer_and_tensor_map = {}
        self._cache_read_tensor_and_buffer_map = {}

        self._cache_write_tensors = set()
        self._cache_write_buffer_and_tensor_map = {}
        self._cache_write_tensor_and_buffer_map = {}

        self._compute_inline_tensors = set()
        self._compute_inlined_tensors = set()

        self._block_split_result = {}
        self._ub_split_result = {}

        self._reorder_map = {}
        self._multi_core_bind_axis = None
        self._storage_align_map = {}
        self._compute_at_map = {}
        self._emit_insn_map = {}

        self._is_last_common_axis_split_block = False
        self._is_last_common_axis_split_ub = False
        self._is_split_block = not norm_info.is_all_reduce
        self._is_const = get_context().get_current_compute().get("_mode") == "const"

    def do_schedule(self):
        """
        normal norm schedule process
        """
        # to ensure that the number of input parameters in the normal sch is the same as that in the workspace sch.
        fake_workspace_count = 0
        # all reduce tensors are workspace tensors in workspace_partial_reorder sch
        # while some reduce tensors may not workspace tensors in workspace sch
        local_set = self._graph_info.workspace_and_reduce_tensor_set \
            if self._norm_info.exist_partial_reorder else self._graph_info.workspace_tensor_set
        for _ in local_set:
            fake_workspace = tvm.placeholder([], dtype="uint8", name="fake_workspace_" + str(fake_workspace_count))
            fake_workspace_count += 1
            self._outs.append(fake_workspace)

        self._sch = tvm.create_schedule([tensor.op for tensor in self._graph_info.endpoint_output_tensor_set])
        self._sch.tiling_key = self._tiling_case.tiling_key

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        self._calc_compute_inline()
        self._do_compute_inline()

        self._do_tiling()

        self._calc_reorder()
        self._do_reorder()

        self._calc_storage_align()
        self._do_storage_align()

        self._calc_multi_core()
        self._do_multi_core()

        self._do_storage_bound()
        self._do_set_constraint()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._do_pragma()

        return self._sch

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._graph_info.input_tensor_set)

    def _do_cache_read(self):
        for cache_read_tensor in self._cache_read_tensors:
            read_buffer = self._sch.cache_read(
                cache_read_tensor, self._scope, self._forward_compute_graph_map[cache_read_tensor])
            self._cache_read_buffer_and_tensor_map[read_buffer] = cache_read_tensor
            self._cache_read_tensor_and_buffer_map[cache_read_tensor] = read_buffer

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._graph_info.output_tensor_set)

    def _do_cache_write(self):
        for cache_write_tensor in self._cache_write_tensors:
            buffer_tensor = self._sch.cache_write(cache_write_tensor, self._scope)
            self._cache_write_buffer_and_tensor_map[buffer_tensor] = cache_write_tensor
            self._cache_write_tensor_and_buffer_map[cache_write_tensor] = buffer_tensor

    def _set_scope(self):
        for mid_tensor in self._graph_info.mid_tensor_set:
            self._sch[mid_tensor].set_scope(self._scope)

    def _calc_compute_inline(self):
        # nlast reduce, broadcast could compute_inline
        if not self._norm_info.is_reduce_last_axis:
            for broadcast_tensor in self._graph_info.broadcast_tensor_set:
                self._compute_inline_tensors.add(broadcast_tensor)
                # compute_inlined_tensor may has been cache_write
                for compute_inlined_tensor in self._forward_compute_graph_map[broadcast_tensor]:
                    if compute_inlined_tensor in self._cache_write_tensor_and_buffer_map:
                        self._compute_inlined_tensors.add(self._cache_write_tensor_and_buffer_map
                                                          [compute_inlined_tensor])
                    else:
                        self._compute_inlined_tensors.add(compute_inlined_tensor)

    def _do_compute_inline(self):
        for compute_inline_tensor in self._compute_inline_tensors:
            self._sch[compute_inline_tensor].compute_inline()

    def _do_tiling(self):
        block_split_axis_index = self._tiling_case.block_split_axis_index
        ub_split_axis_index = self._tiling_case.ub_split_axis_index
        if self._is_split_block:
            block_factor = self._tiling_case.block_factor
            block_split_factor = block_factor if self._is_const else var_inner("_block_factor", (1, None))
            block_outer, block_inner = self._sch[self._res_tensor].split(
                self._res_tensor.op.axis[block_split_axis_index], factor=block_split_factor)

            self._block_split_result["axis"] = block_split_axis_index
            self._block_split_result["outer_itervar"] = block_outer
            self._block_split_result["inner_itervar"] = block_inner
            self._block_split_result["factor"] = block_split_factor

            self._is_last_common_axis_split_block = block_split_axis_index > max(self._norm_info.reduce_axis_indices)

        ub_factor = self._tiling_case.ub_factor
        ub_split_factor = ub_factor if self._is_const else var_inner("_ub_factor", (1, None))

        if self._is_split_block and block_split_axis_index == ub_split_axis_index:
            ub_outer, ub_inner = self._sch[self._res_tensor].split(self._block_split_result["inner_itervar"],
                                                                   factor=ub_split_factor)
        else:
            ub_outer, ub_inner = self._sch[self._res_tensor].split(self._res_tensor.op.axis[ub_split_axis_index],
                                                                   factor=ub_split_factor)

        self._ub_split_result["axis"] = ub_split_axis_index
        self._ub_split_result["outer_itervar"] = ub_outer
        self._ub_split_result["inner_itervar"] = ub_inner
        self._ub_split_result["factor"] = ub_split_factor

        self._is_last_common_axis_split_ub = ub_split_axis_index > max(self._norm_info.reduce_axis_indices)

    def _calc_reorder(self):
        def __calc_split_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            ori_blk_axis = self._tiling_case.block_split_axis_index
            ori_ub_axis = self._tiling_case.ub_split_axis_index
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reduce_reorder_axis, _, ori_to_reorder_axis_map = reorder_reduce_shape(ori_axis,
                                                                                   reduce_axis_index,
                                                                                   is_reduce_last_axis)
            reorder_axis = []
            for idx, axis in enumerate(reduce_reorder_axis):
                if idx == ori_to_reorder_axis_map.get(ori_blk_axis) == ori_to_reorder_axis_map.get(ori_ub_axis):
                    reorder_axis.append(self._block_split_result["outer_itervar"])
                    reorder_axis.append(self._ub_split_result["outer_itervar"])
                    reorder_axis.append(self._ub_split_result["inner_itervar"])
                elif idx == ori_to_reorder_axis_map.get(ori_blk_axis):
                    reorder_axis.append(self._block_split_result["outer_itervar"])
                    reorder_axis.append(self._block_split_result["inner_itervar"])
                elif idx == ori_to_reorder_axis_map.get(ori_ub_axis):
                    reorder_axis.append(self._ub_split_result["outer_itervar"])
                    reorder_axis.append(self._ub_split_result["inner_itervar"])
                else:
                    reorder_axis.append(axis)

            reorder_first_r_index = max(reduce_axis_index) + 1 - len(reduce_axis_index)
            # reorder outer before reduce
            if self._is_last_common_axis_split_block:
                start_axis = max(reduce_axis_index) + 1
                end_axis = reorder_axis.index(self._ub_split_result["inner_itervar"])
                local_list = reorder_axis[start_axis:end_axis]
                reorder_axis[start_axis:end_axis] = []
                reorder_axis[reorder_first_r_index:reorder_first_r_index] = local_list
            elif self._is_last_common_axis_split_ub:
                start_axis = max(reduce_axis_index) + 1 + self._is_split_block
                end_axis = reorder_axis.index(self._ub_split_result["inner_itervar"])
                local_list = reorder_axis[start_axis:end_axis]
                reorder_axis[start_axis:end_axis] = []
                reorder_axis[reorder_first_r_index + self._is_split_block:
                             reorder_first_r_index + self._is_split_block] = local_list

            return reorder_axis

        def __calc_reduce_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reorder_axis = []

            reduce_reorder_axis, reorder_to_ori_axis_map, _ = reorder_reduce_shape(ori_axis,
                                                                                   reduce_axis_index,
                                                                                   is_reduce_last_axis)
            reduce_count = 0
            for idx, axis in enumerate(reduce_reorder_axis):
                if reorder_to_ori_axis_map[idx] in reduce_axis_index:
                    reorder_axis.append(tensor.op.reduce_axis[reduce_count])
                    reduce_count += 1
                else:
                    reorder_axis.append(axis)

            return reorder_axis

        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors:
            if single_tensor in self._graph_info.reduce_tensor_set:
                reorder_axis_list = __calc_reduce_tensor_reorder_axis(single_tensor)
                self._reorder_map[single_tensor] = reorder_axis_list

        reorder_axis_list = __calc_split_tensor_reorder_axis(self._res_tensor)
        self._reorder_map[self._res_tensor] = reorder_axis_list

    def _do_reorder(self):
        for single_tensor, param in self._reorder_map.items():
            self._sch[single_tensor].reorder(*param)

    def _calc_storage_align(self):
        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors:
            # after reduce node do not storage align when last axis reduce
            # while there is no broadcast tensor, the after reduce tensors have to do storage_align
            if self._norm_info.is_reduce_last_axis and single_tensor in self._graph_info.tensors_after_reduce and \
                    not self._norm_info.is_none_reduce:
                continue
            align_factor = get_align_factor(single_tensor.dtype)
            storage_axis = len(single_tensor.shape) - 2
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor in self._cache_read_buffer_and_tensor_map:
            align_factor = get_align_factor(single_tensor.dtype)
            storage_axis = len(single_tensor.shape) - 2
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor in self._cache_write_buffer_and_tensor_map:
            align_factor = get_align_factor(single_tensor.dtype)
            storage_axis = len(single_tensor.shape) - 2
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

    def _do_storage_align(self):
        for single_tensor, param in self._storage_align_map.items():
            self._sch[single_tensor].storage_align(param[0], param[1], param[2])

    def _calc_multi_core(self):
        if self._tiling_case.multi_core:
            block_bind_axis = self._block_split_result["outer_itervar"]
            reorder_axis = self._reorder_map[self._res_tensor]
            fuse_axis_list = reorder_axis[:reorder_axis.index(block_bind_axis) + 1]
            self._multi_core_bind_axis = self._sch[self._res_tensor].fuse(*fuse_axis_list)

    def _do_multi_core(self):
        if self._multi_core_bind_axis is not None:
            block = tvm.thread_axis(BLOCK_IDX)
            self._sch[self._res_tensor].bind(self._multi_core_bind_axis, block)

    def _do_storage_bound(self):
        storage_bound_tensors = (self._graph_info.mid_tensor_set - self._compute_inline_tensors)\
            .union(self._cache_read_buffer_and_tensor_map.keys())\
            .union(self._cache_write_buffer_and_tensor_map.keys())

        for single_tensor in storage_bound_tensors:
            storage_bound_value = self._graph_info.available_ub_size
            self._sch[single_tensor].set_storage_bound(storage_bound_value)

    def _do_set_constraint(self):
        if self._is_const:
            return

        ori_shape = self._res_tensor.shape
        ub_split_inner = self._ub_split_result["factor"]
        ori_ub_axis = self._tiling_case.ub_split_axis_index
        reduce_axis_index = self._norm_info.reduce_axis_indices
        is_reduce_last_axis = self._norm_info.is_reduce_last_axis
        reduce_reorder_shape, _, ori_to_reorder_axis_map = reorder_reduce_shape(ori_shape,
                                                                                reduce_axis_index,
                                                                                is_reduce_last_axis)
        reorder_ub_axis = ori_to_reorder_axis_map[ori_ub_axis]
        shape_in_ub = ub_split_inner
        self._sch.set_constraint(ub_split_inner <= self._graph_info.available_ub_size)
        for i in range(reorder_ub_axis + 1, len(reduce_reorder_shape)):
            shape_in_ub *= reduce_reorder_shape[i]

        self._sch.set_constraint(shape_in_ub <= self._graph_info.available_ub_size)

    def _calc_compute_at(self):
        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors:
            self._compute_at_map[single_tensor] = [self._res_tensor, self._ub_split_result["outer_itervar"]]

        for single_tensor in self._cache_read_buffer_and_tensor_map:
            self._compute_at_map[single_tensor] = [self._res_tensor, self._ub_split_result["outer_itervar"]]

        for single_tensor in self._cache_write_buffer_and_tensor_map:
            self._compute_at_map[single_tensor] = [self._res_tensor, self._ub_split_result["outer_itervar"]]

    def _do_compute_at(self):
        for single_tensor, param in self._compute_at_map.items():
            self._sch[single_tensor].compute_at(self._sch[param[0]], param[1])

    def _calc_emit_insn(self):
        emit_insn_axis_index = 0
        self._emit_insn_map[self._res_tensor] = [self._ub_split_result.get("inner_itervar"),
                                                 "dma_copy",
                                                 {NO_OVERLAP: 3}]
        if self._is_last_common_axis_split_block or len(self._norm_info.reduce_axis_indices) > 1:
            # copy ub to gm with stride, the last block need process the overlap too
            self._emit_insn_map.get(self._res_tensor).pop()
            self._emit_insn_map.get(self._res_tensor).append({NO_OVERLAP: 2})

        for source, _ in self._cache_read_buffer_and_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[emit_insn_axis_index], "dma_copy"]

        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors:
            emit_insn_axis = single_tensor.op.axis[emit_insn_axis_index]
            if single_tensor in self._graph_info.reduce_tensor_set:
                if not self._norm_info.is_reduce_last_axis:
                    emit_insn_axis = single_tensor.op.reduce_axis[0]
                else:
                    first_reduce_axis = self._norm_info.reduce_tensor.op.reduce_axis[0]
                    axis_dom = first_reduce_axis.dom
                    if hasattr(axis_dom.min, "value") and hasattr(axis_dom.extent, "value"):
                        if axis_dom.min.value == 0 and axis_dom.extent.value == 1:
                            emit_insn_axis = single_tensor.op.reduce_axis[0]
                self._emit_insn_map[single_tensor] = [emit_insn_axis, _get_insn(single_tensor),
                                                      {STORAGE_BOUND: self._graph_info.available_ub_size}]
            else:
                self._emit_insn_map[single_tensor] = [emit_insn_axis, _get_insn(single_tensor)]

        for source, target in self._cache_write_buffer_and_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[emit_insn_axis_index], _get_insn(target)]

    def _do_emit_insn(self):
        for single_tensor, param in self._emit_insn_map.items():
            if len(param) > 2:
                self._sch[single_tensor].emit_insn(param[0], param[1], attrs=param[2])
            else:
                self._sch[single_tensor].emit_insn(param[0], param[1])

    def _do_pragma(self):
        def __mark_group_axis_on_split_tensor(_single_tensor):
            # axis_group = 1 means fuse branch will be appended after original no_fuse branch
            append_id = tvm.make.Call("int32", "axis_group", [1, "append"], tvm.expr.Call.Extern, None, 0)
            reorder_axis = self._reorder_map.get(_single_tensor)
            for index in range(reorder_axis.index(self._ub_split_result.get("inner_itervar")), len(reorder_axis)):
                pragma_axis = reorder_axis[index]
                # after ub_split_index may has been reorder, cannot overwrite no_fuse branch
                group_id = append_id
                self._sch[_single_tensor].pragma(pragma_axis, "axis_group", group_id)

        def __mark_group_axis_on_common_tensor(_single_tensor, _tensor_type="common"):
            # axis_group = 0 means original no_fuse branch will be overwrited by fuse branch
            # axis_group = 1 means fuse branch will be appended after original no_fuse branch
            overwrite_and_append_id = tvm.make.Call("int32", "axis_group", [0, "overwrite", 1, "append"],
                                                    tvm.expr.Call.Extern, None, 0)
            append_id = tvm.make.Call("int32", "axis_group", [1, "append"], tvm.expr.Call.Extern, None, 0)
            # If the compute at tensor has been reordered, common tensor should be based on the reorder shape
            # marking group axis. Otherwise, mark group axis on outer ub axis will invalidate it.
            reorder_axis, _, ori_to_reorder_axis_map = \
                reorder_reduce_shape(_single_tensor.op.axis, self._norm_info.reduce_axis_indices,
                                     self._norm_info.is_reduce_last_axis)
            reorder_ub_axis_index = ori_to_reorder_axis_map[self._tiling_case.ub_split_axis_index]
            for index in range(reorder_ub_axis_index, len(reorder_axis)):
                pragma_axis = reorder_axis[index]
                # after ub_split_index may has been reorder, cannot overwrite no_fuse branch
                if _tensor_type == "cache_read_tensor":
                    group_id = append_id
                else:
                    group_id = append_id if index == len(_single_tensor.shape) - 1 else overwrite_and_append_id
                self._sch[_single_tensor].pragma(pragma_axis, "axis_group", group_id)

        for single_tensor in (self._graph_info.mid_tensor_set - self._compute_inline_tensors) \
                .union(self._cache_write_buffer_and_tensor_map.keys()):
            # elewise tensor
            # compute_inlined_tensors can not fuse axis due to broadcast logic
            if single_tensor not in (self._graph_info.reduce_tensor_set | self._graph_info.broadcast_tensor_set |
                                     self._compute_inlined_tensors):
                __mark_group_axis_on_common_tensor(single_tensor)

        for single_tensor in self._cache_read_buffer_and_tensor_map:
            __mark_group_axis_on_common_tensor(single_tensor, "cache_read_tensor")

        __mark_group_axis_on_split_tensor(self._res_tensor)


class NormWorkspaceSchedule:
    """
    norm schedule when ub split reduce axis
    """
    def __init__(self, graph_info, norm_info, tiling_case, outs):
        self._outs = outs
        self._sch = None
        self._scope = LOCAL_UB

        self._graph_info = graph_info
        self._forward_compute_graph_map = graph_info.tensor_consumers_map
        self._backward_compute_graph_map = graph_info.tensor_producers_map
        # all reduce tensors are workspace tensor in partial reorder sch
        self._workspace_tensor_set = graph_info.workspace_tensor_set if not tiling_case.is_partial_reorder_case \
            else graph_info.workspace_and_reduce_tensor_set
        self._workspace_and_reduce_tensor_set = graph_info.workspace_and_reduce_tensor_set
        self._split_tensor_and_sub_graph_map = graph_info.split_tensor_and_sub_graph_map

        # get last endpoint output tensor
        self._res_tensor = tuple(graph_info.endpoint_output_tensor_set)[0]
        self._norm_info = norm_info
        self._tiling_case = tiling_case

        self._workspace_map = {}

        self._cache_read_tensors = set()
        self._cache_read_buffer_and_tensor_map = {}
        self._cache_read_tensor_and_buffer_map = {}

        self._cache_write_tensors = set()
        self._cache_write_buffer_and_tensor_map = {}
        self._cache_write_tensor_and_buffer_map = {}

        self._cache_clone_tensors = set()
        self._cache_clone_buffer_and_tensor_map = {}
        self._cache_clone_tensor_and_buffer_map = {}

        self._compute_inline_tensors = set()
        self._compute_inlined_tensors = set()

        self._block_split_result = {}
        self._ub_split_result = {}

        self._reorder_map = {}
        self._multi_core_bind_axis_map = {}
        self._storage_align_map = {}
        self._bind_buffer_map = {}
        self._compute_at_map = {}
        self._compute_root_tensors = set()
        self._emit_insn_map = {}

        self._is_last_common_axis_split_block = False
        self._is_split_block = not norm_info.is_all_reduce
        self._is_const = get_context().get_current_compute().get("_mode") == "const"

    def do_schedule(self):
        """
        workspace norm schedule process
        """
        real_output_tensor_op_list = [tensor.op for tensor in self._graph_info.endpoint_output_tensor_set]
        for workspace_tensor in self._workspace_tensor_set:
            self._outs.append(workspace_tensor)
            real_output_tensor_op_list.append(workspace_tensor.op)

        # all reduce tensors are workspace tensors in partial reorder sch
        # while some reduce tensors may not workspace tensors in workspace sch
        # some fake workspace need to be added
        if self._norm_info.exist_partial_reorder and not self._tiling_case.is_partial_reorder_case:
            fake_workspace_count = 0
            for reduce_tensor in self._graph_info.reduce_tensor_set:
                if reduce_tensor not in self._workspace_tensor_set:
                    fake_workspace = tvm.placeholder([], dtype="uint8",
                                                     name="fake_workspace_" + str(fake_workspace_count))
                    fake_workspace_count += 1
                    self._outs.append(fake_workspace)

        self._sch = tvm.create_schedule(real_output_tensor_op_list)
        self._sch.tiling_key = self._tiling_case.tiling_key

        self._calc_cache_clone()
        self._do_cache_clone()

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        self._do_workspace_process()

        self._calc_compute_inline()
        self._do_compute_inline()

        self._do_tiling()

        self._calc_reorder()
        self._do_reorder()

        self._calc_storage_align()
        self._do_storage_align()

        self._calc_bind_buffer()
        self._do_bind_buffer()

        self._calc_multi_core()
        self._do_multi_core()

        self._do_storage_bound()
        self._do_set_constraint()

        self._calc_compute_at()
        self._do_compute_at()
        self._do_compute_root()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._do_pragma()

        return self._sch

    def _calc_cache_clone(self):
        self._cache_clone_tensors.update(self._graph_info.cache_clone_tensor_set)

    def _do_cache_clone(self):
        for cache_clone_tensor in self._cache_clone_tensors:
            if cache_clone_tensor in self._graph_info.input_tensor_set:
                continue
            for sub_graph_split_tensor in self._split_tensor_and_sub_graph_map:
                sub_tensor_list = self._split_tensor_and_sub_graph_map[sub_graph_split_tensor]["sub_tensor_list"]
                if cache_clone_tensor in sub_tensor_list:
                    sub_tensor_consumers_map = \
                        self._split_tensor_and_sub_graph_map[sub_graph_split_tensor]["sub_tensor_consumers_map"]
                    local_consumers = list(sub_tensor_consumers_map[cache_clone_tensor])[:]
                    for idx, tensor in enumerate(local_consumers):
                        # the readers have cache clone tensor
                        if tensor in self._cache_clone_tensor_and_buffer_map:
                            local_consumers[idx] = list(self._cache_clone_tensor_and_buffer_map[tensor].keys())[0]
                    clone_buffer = self._sch.cache_clone(cache_clone_tensor, self._scope, local_consumers)
                    _add_sub_dict_to_dict(self._cache_clone_buffer_and_tensor_map, clone_buffer,
                                          cache_clone_tensor, sub_graph_split_tensor)
                    _add_sub_dict_to_dict(self._cache_clone_tensor_and_buffer_map, cache_clone_tensor,
                                          clone_buffer, sub_graph_split_tensor)
                    # cant support cache clone twice now
                    break

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._graph_info.input_tensor_set)

    def _do_cache_read(self):
        for cache_read_tensor in self._cache_read_tensors:
            for sub_graph_split_tensor in self._split_tensor_and_sub_graph_map:
                sub_tensor_list = self._split_tensor_and_sub_graph_map[sub_graph_split_tensor]["sub_tensor_list"]
                if cache_read_tensor in sub_tensor_list:
                    sub_tensor_consumers_map = \
                        self._split_tensor_and_sub_graph_map[sub_graph_split_tensor]["sub_tensor_consumers_map"]
                    local_consumers = list(sub_tensor_consumers_map[cache_read_tensor])[:]
                    for idx, tensor in enumerate(local_consumers):
                        # the readers have cache clone tensor
                        if tensor in self._cache_clone_tensor_and_buffer_map and \
                                sub_graph_split_tensor == \
                                list(self._cache_clone_tensor_and_buffer_map.get(tensor).values())[0]:
                            local_consumers[idx] = list(self._cache_clone_tensor_and_buffer_map[tensor].keys())[0]
                    read_buffer = self._sch.cache_read(cache_read_tensor, self._scope, local_consumers)
                    _add_sub_dict_to_dict(self._cache_read_buffer_and_tensor_map, read_buffer,
                                          cache_read_tensor, sub_graph_split_tensor)
                    _add_sub_dict_to_dict(self._cache_read_tensor_and_buffer_map, cache_read_tensor,
                                          read_buffer, sub_graph_split_tensor)

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._graph_info.output_tensor_set)

    def _do_cache_write(self):
        for cache_write_tensor in self._cache_write_tensors:
            buffer_tensor = self._sch.cache_write(cache_write_tensor, self._scope)
            _add_sub_dict_to_dict(self._cache_write_buffer_and_tensor_map, buffer_tensor,
                                  cache_write_tensor, self._res_tensor)
            _add_sub_dict_to_dict(self._cache_write_tensor_and_buffer_map, cache_write_tensor,
                                  buffer_tensor, self._res_tensor)

    def _set_scope(self):
        for mid_tensor in self._graph_info.mid_tensor_set:
            self._sch[mid_tensor].set_scope(self._scope)

    def _do_workspace_process(self):
        for workspace_tensor in self._workspace_tensor_set:
            self._workspace_map[workspace_tensor] = {}
            for sub_graph_split_tensor in self._split_tensor_and_sub_graph_map:
                sub_tensor_list = self._split_tensor_and_sub_graph_map[sub_graph_split_tensor]["sub_tensor_list"]
                if workspace_tensor in sub_tensor_list:
                    sub_tensor_consumers_map = \
                        self._split_tensor_and_sub_graph_map[sub_graph_split_tensor]["sub_tensor_consumers_map"]
                    local_consumers = list(sub_tensor_consumers_map[workspace_tensor])[:]
                    # if workspace_tensor has consumers
                    if local_consumers:
                        # readers may include res tensor, but res tensor has been cache write
                        for idx, single_tensor in enumerate(local_consumers):
                            if single_tensor in self._cache_write_tensor_and_buffer_map:
                                local_consumers[idx] =\
                                    list(self._cache_write_tensor_and_buffer_map[single_tensor].keys())[0]
                        read_buffer = self._sch.cache_read(workspace_tensor, self._scope, local_consumers)
                        _add_sub_dict_to_dict(self._workspace_map[workspace_tensor], "reread_ub_tensor",
                                              read_buffer, sub_graph_split_tensor)

        for workspace_tensor in self._workspace_tensor_set:
            workspace_ub_tensor = self._sch.cache_write(workspace_tensor, self._scope)
            self._workspace_map[workspace_tensor]["ub_tensor"] = workspace_ub_tensor

        for workspace_tensor in self._workspace_tensor_set:
            self._sch[workspace_tensor].set_scope("global")

    def _calc_compute_inline(self):
        # nlast reduce, broadcast could compute_inline
        if not self._norm_info.is_reduce_last_axis:
            for broadcast_tensor in self._graph_info.broadcast_tensor_set:
                # broadcast do not compute_inline to workspace
                if not self._forward_compute_graph_map[broadcast_tensor] & self._workspace_tensor_set:
                    self._compute_inline_tensors.add(broadcast_tensor)
                    # compute_inlined_tensor may has been cache_write
                    for compute_inlined_tensor in self._forward_compute_graph_map[broadcast_tensor]:
                        if compute_inlined_tensor in self._cache_write_tensor_and_buffer_map:
                            self._compute_inlined_tensors.add(list(self._cache_write_tensor_and_buffer_map
                                                                   [compute_inlined_tensor].keys())[0])
                        else:
                            self._compute_inlined_tensors.add(compute_inlined_tensor)

    def _do_compute_inline(self):
        for compute_inline_tensor in self._compute_inline_tensors:
            self._sch[compute_inline_tensor].compute_inline()

    def _do_tiling(self):
        block_split_axis_index = self._tiling_case.block_split_axis_index
        ub_split_axis_index = self._tiling_case.ub_split_axis_index
        if self._is_split_block:
            block_factor = self._tiling_case.block_factor
            block_split_factor = block_factor if self._is_const else var_inner("_block_factor", (1, None))
            for workspace_tensor in self._workspace_tensor_set:
                block_outer, block_inner = self._sch[workspace_tensor].split(
                    workspace_tensor.op.axis[block_split_axis_index], factor=block_split_factor)
                self._block_split_result[workspace_tensor] = {
                    "axis": block_split_axis_index,
                    "outer_itervar": block_outer,
                    "inner_itervar": block_inner,
                    "factor": block_split_factor
                }
            block_outer, block_inner = self._sch[self._res_tensor].split(
                self._res_tensor.op.axis[block_split_axis_index], factor=block_split_factor)
            self._block_split_result[self._res_tensor] = {
                "axis": block_split_axis_index,
                "outer_itervar": block_outer,
                "inner_itervar": block_inner,
                "factor": block_split_factor
            }
            self._is_last_common_axis_split_block = block_split_axis_index > max(self._norm_info.reduce_axis_indices)

        ub_factor = self._tiling_case.ub_factor
        ub_split_factor = ub_factor if self._is_const else var_inner("_ub_factor", (1, None))
        # block tiling on A, ub tiling on R
        # non_reduce workspace node, block and ub split on workspace tensor
        # reduce workspace node, block split on workspace tensor and ub split on workspace ub tensor
        # common reduce node, ub split on reduce tensor
        for split_tensor in self._workspace_and_reduce_tensor_set:
            if split_tensor in self._graph_info.reduce_tensor_set:
                if split_tensor in self._workspace_tensor_set:
                    split_tensor = self._workspace_map[split_tensor]["ub_tensor"]

                ub_split_reduce_axis_index = sorted(self._norm_info.reduce_axis_indices).index(
                    ub_split_axis_index)
                ub_outer, ub_inner = self._sch[split_tensor].split(
                    split_tensor.op.reduce_axis[ub_split_reduce_axis_index], factor=ub_split_factor)
                self._ub_split_result[split_tensor] = {
                    "axis": ub_split_axis_index,
                    "reduce_axis": ub_split_reduce_axis_index,
                    "outer_itervar": ub_outer,
                    "inner_itervar": ub_inner,
                    "factor": ub_split_factor
                }
            else:
                ub_outer, ub_inner = self._sch[split_tensor].split(
                    split_tensor.op.axis[ub_split_axis_index], factor=ub_split_factor)
                self._ub_split_result[split_tensor] = {
                    "axis": ub_split_axis_index,
                    "outer_itervar": ub_outer,
                    "inner_itervar": ub_inner,
                    "factor": ub_split_factor
                }

        ub_outer, ub_inner = self._sch[self._res_tensor].split(self._res_tensor.op.axis[ub_split_axis_index],
                                                               factor=ub_split_factor)
        self._ub_split_result[self._res_tensor] = {
            "axis": ub_split_axis_index,
            "outer_itervar": ub_outer,
            "inner_itervar": ub_inner,
            "factor": ub_split_factor
        }

    def _calc_reorder(self):
        def __calc_split_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            ori_blk_axis = self._tiling_case.block_split_axis_index
            ori_ub_axis = self._tiling_case.ub_split_axis_index
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reduce_reorder_axis, _, ori_to_reorder_axis_map = reorder_reduce_shape(ori_axis,
                                                                                   reduce_axis_index,
                                                                                   is_reduce_last_axis)
            reorder_axis = []
            for idx, axis in enumerate(reduce_reorder_axis):
                if tensor in self._block_split_result and idx == ori_to_reorder_axis_map[ori_blk_axis]:
                    reorder_axis.append(self._block_split_result[tensor]["outer_itervar"])
                    reorder_axis.append(self._block_split_result[tensor]["inner_itervar"])
                elif tensor in self._ub_split_result and idx == ori_to_reorder_axis_map[ori_ub_axis]:
                    reorder_axis.append(self._ub_split_result[tensor]["outer_itervar"])
                    reorder_axis.append(self._ub_split_result[tensor]["inner_itervar"])
                else:
                    reorder_axis.append(axis)

            reorder_first_r_index = max(reduce_axis_index) + 1 - len(reduce_axis_index)
            # reorder outer before reduce
            if self._is_last_common_axis_split_block and tensor in self._block_split_result:
                start_axis = max(reduce_axis_index) + 1
                if tensor in self._ub_split_result:
                    start_axis += 1
                end_axis = reorder_axis.index(self._block_split_result[tensor]["inner_itervar"])
                local_list = reorder_axis[start_axis:end_axis]
                reorder_axis[start_axis:end_axis] = []
                reorder_axis[reorder_first_r_index:reorder_first_r_index] = local_list

            return reorder_axis

        def __calc_reduce_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reorder_axis = []
            reduce_reorder_axis, reorder_to_ori_axis_map, _ = reorder_reduce_shape(ori_axis,
                                                                                   reduce_axis_index,
                                                                                   is_reduce_last_axis)
            reduce_count = 0
            for idx, axis in enumerate(reduce_reorder_axis):
                # ub split R
                if reorder_to_ori_axis_map[idx] in reduce_axis_index:
                    if tensor in self._ub_split_result and \
                            reorder_to_ori_axis_map[idx] == self._ub_split_result[tensor]["axis"]:
                        reorder_axis.append(self._ub_split_result[tensor]["outer_itervar"])
                        reorder_axis.append(self._ub_split_result[tensor]["inner_itervar"])
                    else:
                        reorder_axis.append(tensor.op.reduce_axis[reduce_count])
                    reduce_count += 1
                # block split A
                else:
                    if tensor in self._block_split_result and \
                            reorder_to_ori_axis_map[idx] == self._block_split_result[tensor]["axis"]:
                        reorder_axis.append(self._block_split_result[tensor]["outer_itervar"])
                        reorder_axis.append(self._block_split_result[tensor]["inner_itervar"])
                    else:
                        reorder_axis.append(axis)

            reorder_first_r_index = max(reduce_axis_index) + 1 - len(reduce_axis_index)
            # reorder outer before reduce
            if self._is_last_common_axis_split_block and tensor in self._block_split_result:
                start_axis = max(reduce_axis_index) + 1 + 1
                end_axis = reorder_axis.index(self._block_split_result[tensor]["inner_itervar"])
                local_list = reorder_axis[start_axis:end_axis]
                reorder_axis[start_axis:end_axis] = []
                reorder_axis[reorder_first_r_index:reorder_first_r_index] = local_list

            return reorder_axis

        def __common_reorder_process():
            for split_tensor in self._workspace_and_reduce_tensor_set:
                if split_tensor in self._graph_info.reduce_tensor_set:
                    if split_tensor in self._workspace_tensor_set:
                        workspace_ub_tensor = self._workspace_map[split_tensor]["ub_tensor"]
                        reorder_axis_list = __calc_split_tensor_reorder_axis(split_tensor)
                        self._reorder_map[split_tensor] = reorder_axis_list
                        reorder_axis_list = __calc_reduce_tensor_reorder_axis(workspace_ub_tensor)
                        self._reorder_map[workspace_ub_tensor] = reorder_axis_list
                    else:
                        reorder_axis_list = __calc_reduce_tensor_reorder_axis(split_tensor)
                        self._reorder_map[split_tensor] = reorder_axis_list
                else:
                    reorder_axis_list = __calc_split_tensor_reorder_axis(split_tensor)
                    self._reorder_map[split_tensor] = reorder_axis_list

            reorder_axis_list = __calc_split_tensor_reorder_axis(self._res_tensor)
            self._reorder_map[self._res_tensor] = reorder_axis_list

        def __partial_reorder_process():
            for workspace_tensor in self._workspace_tensor_set:
                workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
                if workspace_tensor in self._graph_info.reduce_tensor_set:
                    reorder_axis_list = __calc_split_tensor_reorder_axis(workspace_tensor)
                    self._reorder_map[workspace_tensor] = reorder_axis_list
                    reorder_axis_list = __calc_reduce_tensor_reorder_axis(workspace_ub_tensor)
                    self._reorder_map[workspace_ub_tensor] = reorder_axis_list

        if self._tiling_case.is_partial_reorder_case:
            __partial_reorder_process()
        else:
            __common_reorder_process()

    def _do_reorder(self):
        for single_tensor, param in self._reorder_map.items():
            self._sch[single_tensor].reorder(*param)

    def _calc_storage_align(self):
        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors -\
                             self._workspace_tensor_set:
            # after reduce node do not storage align when last axis reduce
            if self._norm_info.is_reduce_last_axis and single_tensor in self._graph_info.tensors_after_reduce:
                continue
            align_factor = get_align_factor(single_tensor.dtype)
            storage_axis = len(single_tensor.shape) - 2
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor in self._cache_read_buffer_and_tensor_map:
            align_factor = get_align_factor(single_tensor.dtype)
            storage_axis = len(single_tensor.shape) - 2
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor in self._cache_write_buffer_and_tensor_map:
            align_factor = get_align_factor(single_tensor.dtype)
            storage_axis = len(single_tensor.shape) - 2
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor in self._cache_clone_buffer_and_tensor_map:
            align_factor = get_align_factor(single_tensor.dtype)
            storage_axis = len(single_tensor.shape) - 2
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for workspace_tensor in self._workspace_tensor_set:
            # after reduce node do not storage align when last axis reduce
            if self._norm_info.is_reduce_last_axis and workspace_tensor in self._graph_info.tensors_after_reduce:
                continue
            workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
            reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
            align_factor = get_align_factor(workspace_ub_tensor.dtype)
            storage_axis = len(workspace_ub_tensor.shape) - 2
            self._storage_align_map[workspace_ub_tensor] = [workspace_ub_tensor.op.axis[storage_axis], align_factor, 0]
            for single_tensor in reread_workspace_ub_tensor_map.keys():
                align_factor = get_align_factor(single_tensor.dtype)
                storage_axis = len(single_tensor.shape) - 2
                self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

    def _do_storage_align(self):
        for single_tensor, param in self._storage_align_map.items():
            self._sch[single_tensor].storage_align(param[0], param[1], param[2])

    def _calc_bind_buffer(self):
        for workspace_tensor in self._workspace_tensor_set:
            if not self._norm_info.is_reduce_last_axis or workspace_tensor in self._graph_info.tensors_before_reduce:
                align_factor = get_align_factor(workspace_tensor.dtype)
                bind_axis = len(workspace_tensor.shape) - 2
                bind_factor =\
                    tvm.div((workspace_tensor.shape[bind_axis + 1] + align_factor - 1), align_factor) * align_factor
                self._bind_buffer_map[workspace_tensor] = [workspace_tensor.op.axis[bind_axis], bind_factor, 0]

    def _do_bind_buffer(self):
        for single_tensor, param in self._bind_buffer_map.items():
            self._sch[single_tensor].bind_buffer(param[0], param[1], param[2])

    def _calc_multi_core(self):
        if self._tiling_case.multi_core:
            block_bind_axis = self._block_split_result[self._res_tensor]["outer_itervar"]
            reorder_axis = self._reorder_map[self._res_tensor]
            fuse_axis_list = reorder_axis[:reorder_axis.index(block_bind_axis) + 1]
            self._multi_core_bind_axis_map[self._res_tensor] = self._sch[self._res_tensor].fuse(*fuse_axis_list)

            for workspace_tensor in self._workspace_tensor_set:
                block_bind_axis = self._block_split_result[workspace_tensor]["outer_itervar"]
                reorder_axis = self._reorder_map[workspace_tensor]
                fuse_axis_list = reorder_axis[:reorder_axis.index(block_bind_axis) + 1]
                self._multi_core_bind_axis_map[workspace_tensor] = self._sch[workspace_tensor].fuse(*fuse_axis_list)

    def _do_multi_core(self):
        if self._multi_core_bind_axis_map:
            block = tvm.thread_axis(BLOCK_IDX)
            for single_tensor, param in self._multi_core_bind_axis_map.items():
                self._sch[single_tensor].bind(param, block)

    def _do_storage_bound(self):
        storage_bound_tensors = (self._graph_info.mid_tensor_set - self._compute_inline_tensors -
                                 self._workspace_tensor_set)\
            .union(self._cache_read_buffer_and_tensor_map.keys())\
            .union(self._cache_write_buffer_and_tensor_map.keys())\
            .union(self._cache_clone_buffer_and_tensor_map.keys())

        for workspace_tensor in self._workspace_tensor_set:
            workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
            reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
            storage_bound_tensors.add(workspace_ub_tensor)
            storage_bound_tensors = storage_bound_tensors.union(reread_workspace_ub_tensor_map.keys())

        for single_tensor in storage_bound_tensors:
            storage_bound_value = self._graph_info.workspace_available_min_ub_size
            self._sch[single_tensor].set_storage_bound(storage_bound_value)

    def _do_set_constraint(self):
        if self._is_const:
            return

        ori_shape = self._res_tensor.shape
        ub_split_inner = self._ub_split_result[self._res_tensor]["factor"]
        ori_ub_axis = self._tiling_case.ub_split_axis_index
        reduce_axis_index = self._norm_info.reduce_axis_indices
        is_reduce_last_axis = self._norm_info.is_reduce_last_axis
        reduce_reorder_shape, _, ori_to_reorder_axis_map = reorder_reduce_shape(ori_shape,
                                                                                reduce_axis_index,
                                                                                is_reduce_last_axis)
        reorder_ub_axis = ori_to_reorder_axis_map[ori_ub_axis]
        shape_in_ub = ub_split_inner
        self._sch.set_constraint(ub_split_inner <= self._graph_info.workspace_available_min_ub_size)
        for i in range(reorder_ub_axis + 1, len(reduce_reorder_shape)):
            shape_in_ub *= reduce_reorder_shape[i]
        if self._is_last_common_axis_split_block:
            blk_split_inner = self._block_split_result[self._res_tensor]["factor"]
            shape_in_ub = shape_in_ub // reduce_reorder_shape[-1] * blk_split_inner
            self._sch.set_constraint(blk_split_inner <= self._graph_info.workspace_available_min_ub_size)

        self._sch.set_constraint(shape_in_ub <= self._graph_info.workspace_available_min_ub_size)

    def _calc_compute_at(self):
        def __get_compute_at_workspace_ub_tensor(_ori_compute_at_tensor):
            # reduce workspace tensor compute at to the workspace_ub_tensor
            if _ori_compute_at_tensor not in self._ub_split_result and _ori_compute_at_tensor in self._workspace_map:
                return self._workspace_map[_ori_compute_at_tensor]["ub_tensor"]

            return _ori_compute_at_tensor

        def __handle_common_reduce_tensor(_tensor, _compute_at_tensor):
            if _compute_at_tensor in self._graph_info.reduce_tensor_set:
                self._compute_at_map[_tensor] = \
                    [_compute_at_tensor, self._ub_split_result[_compute_at_tensor]["outer_itervar"]]
                return

            if self._norm_info.is_all_reduce:
                self._compute_root_tensors.add(_tensor)
                return

            # common reduce tensor should compute at previous A axis to avoid repeated calculation
            ori_shape = self._res_tensor.shape
            ori_block_axis = self._tiling_case.block_split_axis_index
            ori_ub_axis = self._tiling_case.ub_split_axis_index
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reduce_reorder_shape, reorder_to_ori_axis_map, ori_to_reorder_axis_map =\
                reorder_reduce_shape(ori_shape, reduce_axis_index, is_reduce_last_axis)
            reorder_block_axis = ori_to_reorder_axis_map[ori_block_axis]
            reorder_ub_axis = ori_to_reorder_axis_map[ori_ub_axis]

            compute_at_axis_index = -1
            for index in range(reorder_ub_axis, -1, -1):
                if reorder_to_ori_axis_map[index] not in reduce_axis_index:
                    compute_at_axis_index = index
                    break

            if compute_at_axis_index > reorder_block_axis:
                self._compute_at_map[_tensor] = \
                    [_compute_at_tensor, _compute_at_tensor.op.axis[reorder_to_ori_axis_map[compute_at_axis_index]]]
            elif compute_at_axis_index == reorder_block_axis:
                self._compute_at_map[_tensor] = \
                    [_compute_at_tensor, self._block_split_result[_compute_at_tensor]["inner_itervar"]]
            else:
                if _compute_at_tensor in self._multi_core_bind_axis_map:
                    self._compute_at_map[_tensor] = \
                        [_compute_at_tensor, self._multi_core_bind_axis_map[_compute_at_tensor]]
                else:
                    self._compute_at_map[_tensor] = \
                        [_compute_at_tensor, self._block_split_result[_compute_at_tensor]["outer_itervar"]]

            return

        def __handle_workspace_tensor():
            for workspace_tensor in self._workspace_tensor_set:
                workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
                reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]

                for reread_workspace_ub_tensor in reread_workspace_ub_tensor_map:
                    compute_at_tensor = __get_compute_at_workspace_ub_tensor(
                        reread_workspace_ub_tensor_map[reread_workspace_ub_tensor])
                    self._compute_at_map[reread_workspace_ub_tensor] =\
                        [compute_at_tensor, self._ub_split_result[compute_at_tensor]["outer_itervar"]]

                # non_reduce workspace node, block and ub split on workspace tensor
                if workspace_tensor not in self._graph_info.reduce_tensor_set:
                    self._compute_at_map[workspace_ub_tensor] =\
                        [workspace_tensor, self._ub_split_result[workspace_tensor]["outer_itervar"]]
                # reduce workspace node, block split on workspace tensor and ub split on workspace ub tensor
                else:
                    if workspace_tensor in self._multi_core_bind_axis_map:
                        # after fuse axis
                        self._compute_at_map[workspace_ub_tensor] =\
                            [workspace_tensor, self._multi_core_bind_axis_map[workspace_tensor]]
                    elif workspace_tensor in self._block_split_result:
                        self._compute_at_map[workspace_ub_tensor] =\
                            [workspace_tensor, self._block_split_result[workspace_tensor]["outer_itervar"]]
                    else:
                        self._compute_root_tensors.add(workspace_ub_tensor)

        for single_tensor in \
                self._graph_info.mid_tensor_set - self._compute_inline_tensors - self._workspace_tensor_set:
            for sub_graph_split_tensor in self._split_tensor_and_sub_graph_map:
                sub_tensor_list = self._split_tensor_and_sub_graph_map[sub_graph_split_tensor]["sub_tensor_list"]
                if single_tensor in sub_tensor_list and single_tensor != sub_graph_split_tensor:
                    compute_at_tensor = __get_compute_at_workspace_ub_tensor(sub_graph_split_tensor)
                    if single_tensor in self._graph_info.reduce_tensor_set:
                        __handle_common_reduce_tensor(single_tensor, compute_at_tensor)
                    else:
                        self._compute_at_map[single_tensor] =\
                            [compute_at_tensor, self._ub_split_result[compute_at_tensor]["outer_itervar"]]

        for single_tensor in self._cache_read_buffer_and_tensor_map:
            compute_at_tensor = __get_compute_at_workspace_ub_tensor(
                list(self._cache_read_buffer_and_tensor_map[single_tensor].values())[0])
            self._compute_at_map[single_tensor] = [compute_at_tensor,
                                                   self._ub_split_result[compute_at_tensor]["outer_itervar"]]

        for single_tensor in self._cache_write_buffer_and_tensor_map:
            compute_at_tensor = __get_compute_at_workspace_ub_tensor(
                list(self._cache_write_buffer_and_tensor_map[single_tensor].values())[0])
            self._compute_at_map[single_tensor] = [compute_at_tensor,
                                                   self._ub_split_result[compute_at_tensor]["outer_itervar"]]

        for single_tensor in self._cache_clone_buffer_and_tensor_map:
            compute_at_tensor = __get_compute_at_workspace_ub_tensor(
                list(self._cache_clone_buffer_and_tensor_map[single_tensor].values())[0])
            self._compute_at_map[single_tensor] = [compute_at_tensor,
                                                   self._ub_split_result[compute_at_tensor]["outer_itervar"]]

        __handle_workspace_tensor()

    def _do_compute_at(self):
        for single_tensor, param in self._compute_at_map.items():
            self._sch[single_tensor].compute_at(self._sch[param[0]], param[1])

    def _do_compute_root(self):
        for single_tensor in self._compute_root_tensors:
            self._sch[single_tensor].compute_root()

    def _calc_emit_insn(self):
        def __handle_workspace_tensor():
            for workspace_tensor in self._workspace_tensor_set:
                workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
                reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
                # src and dst are align
                is_align = workspace_tensor in self._bind_buffer_map and \
                           (self._tiling_case.block_split_axis_index !=
                            len(self._norm_info.shape_before_reduce) - 1 and
                            self._tiling_case.ub_split_axis_index !=
                            len(self._norm_info.shape_before_reduce) - 1)
                # non_reduce workspace node, block and ub split on workspace tensor`
                if workspace_tensor not in self._graph_info.reduce_tensor_set:
                    self._emit_insn_map[workspace_tensor] =\
                            [self._ub_split_result[workspace_tensor]["inner_itervar"], "dma_copy"]

                    if is_align:
                        self._emit_insn_map.get(workspace_tensor).append({NO_OVERLAP: 0})
                    elif need_enable_no_overlap_two:
                        self._emit_insn_map.get(workspace_tensor).append({NO_OVERLAP: 2})

                    if self._tiling_case.is_partial_reorder_case:
                        if self._tiling_case.block_split_axis_index > self._tiling_case.ub_split_axis_index:
                            self._emit_insn_map[workspace_tensor] =\
                                [self._block_split_result[workspace_tensor]["inner_itervar"], "dma_copy"]

                    self._emit_insn_map[workspace_ub_tensor] =\
                        [workspace_ub_tensor.op.axis[0], _get_insn(workspace_tensor)]
                # reduce workspace node, block split on workspace tensor and ub split on workspace ub tensor
                else:
                    emit_axis = self._block_split_result[workspace_tensor]["inner_itervar"] \
                        if workspace_tensor in self._block_split_result else workspace_tensor.op.axis[0]
                    self._emit_insn_map[workspace_tensor] = [emit_axis, "dma_copy"]
                    if is_align:
                        self._emit_insn_map.get(workspace_tensor).append({NO_OVERLAP: 0})

                    self._emit_insn_map[workspace_ub_tensor] =\
                        [self._ub_split_result[workspace_ub_tensor]["inner_itervar"], _get_insn(workspace_tensor),
                        {STORAGE_BOUND: self._graph_info.workspace_available_min_ub_size}]

                for reread_workspace_ub_tensor in reread_workspace_ub_tensor_map.keys():
                    self._emit_insn_map[reread_workspace_ub_tensor] =\
                        [reread_workspace_ub_tensor.op.axis[0], "dma_copy"]

        need_enable_no_overlap_two = self._is_last_common_axis_split_block or \
                                     len(self._norm_info.reduce_axis_indices) > 1 and \
                                     not self._tiling_case.is_partial_reorder_case and \
                                     self._is_split_block
        for source, _ in self._cache_clone_buffer_and_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], _get_insn(source)]

        for source, _ in self._cache_read_buffer_and_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], "dma_copy"]

        for single_tensor in \
                self._graph_info.mid_tensor_set - self._compute_inline_tensors - self._workspace_tensor_set:
            if single_tensor in self._graph_info.reduce_tensor_set:
                self._emit_insn_map[single_tensor] = \
                    [self._ub_split_result[single_tensor]["inner_itervar"], _get_insn(single_tensor),
                     {STORAGE_BOUND: self._graph_info.workspace_available_min_ub_size}]
            else:
                self._emit_insn_map[single_tensor] = [single_tensor.op.axis[0], _get_insn(single_tensor)]

        __handle_workspace_tensor()

        for source, target_map in self._cache_write_buffer_and_tensor_map.items():
            target = list(target_map.values())[0]
            self._emit_insn_map[source] = [source.op.axis[0], _get_insn(target)]

        if need_enable_no_overlap_two:
            self._emit_insn_map[self._res_tensor] = [self._ub_split_result[self._res_tensor]["inner_itervar"],
                                                     "dma_copy",
                                                     {NO_OVERLAP: 2}]
        else:
            self._emit_insn_map[self._res_tensor] =\
                [self._ub_split_result[self._res_tensor]["inner_itervar"], "dma_copy"]

        if self._tiling_case.is_partial_reorder_case:
            if self._tiling_case.block_split_axis_index > self._tiling_case.ub_split_axis_index:
                self._emit_insn_map[self._res_tensor] =\
                    [self._block_split_result[self._res_tensor]["inner_itervar"], "dma_copy"]

    def _do_emit_insn(self):
        for single_tensor, param in self._emit_insn_map.items():
            if len(param) > 2:
                self._sch[single_tensor].emit_insn(param[0], param[1], attrs=param[2])
            else:
                self._sch[single_tensor].emit_insn(param[0], param[1])

    def _do_pragma(self):
        def __mark_group_axis_on_split_tensor(_single_tensor):
            # axis_group = 0 means original no_fuse branch will be overwrited by fuse branch
            # axis_group = 1 means fuse branch will be appended after original no_fuse branch
            overwrite_and_append_id = tvm.make.Call("int32", "axis_group", [0, "overwrite", 1, "append"],
                                                    tvm.expr.Call.Extern, None, 0)
            append_id = tvm.make.Call("int32", "axis_group", [1, "append"], tvm.expr.Call.Extern, None, 0)
            if self._tiling_case.is_partial_reorder_case:
                for index in range(self._tiling_case.ub_split_axis_index, len(_single_tensor.shape)):
                    pragma_axis = self._ub_split_result.get(_single_tensor).get("inner_itervar") \
                        if index == self._tiling_case.ub_split_axis_index and \
                           _single_tensor in self._ub_split_result else _single_tensor.op.axis[index]
                    group_id = append_id if index == len(_single_tensor.shape) - 1 else overwrite_and_append_id
                    # in partial reorder sch, block split first common axis probably after ub split axis
                    if index == self._tiling_case.block_split_axis_index:
                        self._sch[_single_tensor].pragma(
                            self._block_split_result.get(_single_tensor).get("outer_itervar"),
                            "axis_group", group_id)
                        self._sch[_single_tensor].pragma(
                            self._block_split_result.get(_single_tensor).get("inner_itervar"),
                            "axis_group", group_id)
                    else:
                        self._sch[_single_tensor].pragma(pragma_axis, "axis_group", group_id)
                return

            reorder_axis = self._reorder_map.get(_single_tensor)
            for index in range(reorder_axis.index(self._ub_split_result.get(_single_tensor).get("inner_itervar")),
                               len(reorder_axis)):
                pragma_axis = reorder_axis[index]
                # after ub_split_index may has been reorder, cannot overwrite no_fuse branch
                group_id = append_id
                self._sch[_single_tensor].pragma(pragma_axis, "axis_group", group_id)

        def __mark_group_axis_on_common_tensor(_single_tensor, _tensor_type="common"):
            # axis_group = 0 means original no_fuse branch will be overwrited by fuse branch
            # axis_group = 1 means fuse branch will be appended after original no_fuse branch
            overwrite_and_append_id = tvm.make.Call("int32", "axis_group", [0, "overwrite", 1, "append"],
                                                    tvm.expr.Call.Extern, None, 0)
            append_id = tvm.make.Call("int32", "axis_group", [1, "append"], tvm.expr.Call.Extern, None, 0)
            # in partial reorder sch, common tensors only reorder when compute at reduce tensor
            if self._tiling_case.is_partial_reorder_case:
                if self._compute_at_map.get(_single_tensor)[0] not in self._graph_info.reduce_tensor_set:
                    for index in range(self._tiling_case.ub_split_axis_index, len(_single_tensor.shape)):
                        group_id = append_id if index == len(_single_tensor.shape) - 1 else overwrite_and_append_id
                        self._sch[_single_tensor].pragma(_single_tensor.op.axis[index], "axis_group", group_id)
                        return
            # If the compute at tensor has been reordered, common tensor should be based on the reorder shape
            # marking group axis. Otherwise, mark group axis on outer ub axis will invalidate it.
            reorder_axis, _, ori_to_reorder_axis_map = \
                reorder_reduce_shape(_single_tensor.op.axis, self._norm_info.reduce_axis_indices,
                                     self._norm_info.is_reduce_last_axis)
            reorder_ub_axis_index = ori_to_reorder_axis_map[self._tiling_case.ub_split_axis_index]
            for index in range(reorder_ub_axis_index, len(reorder_axis)):
                pragma_axis = reorder_axis[index]
                # after ub_split_index may has been reorder, cannot overwrite no_fuse branch
                if _tensor_type == "cache_read_tensor":
                    group_id = append_id
                else:
                    group_id = append_id if index == len(_single_tensor.shape) - 1 else overwrite_and_append_id
                self._sch[_single_tensor].pragma(pragma_axis, "axis_group", group_id)

        for single_tensor in (self._graph_info.mid_tensor_set - self._compute_inline_tensors -
                              self._workspace_tensor_set) \
                .union(self._cache_write_buffer_and_tensor_map.keys()) \
                .union(self._cache_clone_buffer_and_tensor_map.keys()):
            # elewise tensor
            # compute_inlined_tensors can not fuse axis due to broadcast logic
            if single_tensor not in (self._graph_info.reduce_tensor_set | self._graph_info.broadcast_tensor_set |
                                     self._compute_inlined_tensors):
                __mark_group_axis_on_common_tensor(single_tensor)

        for single_tensor in self._cache_read_buffer_and_tensor_map:
            __mark_group_axis_on_common_tensor(single_tensor, "cache_read_tensor")

        for workspace_tensor in self._workspace_tensor_set:
            __mark_group_axis_on_split_tensor(workspace_tensor)

            workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
            reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
            # elewise tensor
            # compute_inlined_tensors can not fuse axis due to broadcast logic
            if workspace_tensor not in (self._graph_info.reduce_tensor_set | self._graph_info.broadcast_tensor_set |
                                        self._compute_inlined_tensors):
                __mark_group_axis_on_common_tensor(workspace_ub_tensor)
            for reread_workspace_ub_tensor in reread_workspace_ub_tensor_map:
                __mark_group_axis_on_common_tensor(reread_workspace_ub_tensor, "cache_read_tensor")

        __mark_group_axis_on_split_tensor(self._res_tensor)
