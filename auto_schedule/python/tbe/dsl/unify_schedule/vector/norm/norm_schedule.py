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

from .norm_tilingcase import get_broadcast_axis
from .norm_tilingcase import get_block_size as get_align_factor
from .norm_tilingcase import judge_tvm_shape_equal
from ... import util
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import FAKE_NODE_TAG
from ...constants import INSN_MAPPING
from ...constants import NormPattern
from ...constants import Pattern
from ...schedule import Schedule

BLOCK_IDX = "blockIdx.x"
LOCAL_UB = "local.UB"
NO_OVERLAP = "no_overlap"
STORAGE_BOUND = "storage_bound"
ENABLE_VNCHWCONV = "enable_vnchwconv"


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


def _get_sub_dict_first_key(_map, _key):
    """
    get sub dict first key
    """
    return list(_map.get(_key).keys())[0]


def _get_sub_dict_first_value(_map, _key):
    """
    get sub dict first value
    """
    return list(_map.get(_key).values())[0]


def _reorder_reduce_shape(shape_before_reduce, reduce_axis_index, is_reduce_last_axis):
    return _reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index) if is_reduce_last_axis else \
        _reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index)


def _reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index):
    ori_to_reorder_axis_map = {}
    reorder_to_ori_axis_map = {}
    reordered_shape = []
    temp_axis = 0
    for i, ele in enumerate(shape_before_reduce):
        if i not in reduce_axis_index:
            reordered_shape.append(ele)
            reorder_to_ori_axis_map[temp_axis] = i
            ori_to_reorder_axis_map[i] = temp_axis
            temp_axis += 1

    for i, ele in enumerate(shape_before_reduce):
        if i in reduce_axis_index:
            reordered_shape.append(ele)
            reorder_to_ori_axis_map[temp_axis] = i
            ori_to_reorder_axis_map[i] = temp_axis
            temp_axis += 1

    return reordered_shape, reorder_to_ori_axis_map, ori_to_reorder_axis_map


def _reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index):
    last_none_reduce_axis = max(reduce_axis_index) + 1
    ori_to_reorder_axis_map = {}
    reorder_to_ori_axis_map = {}
    reordered_shape = list(shape_before_reduce)
    temp_axis = last_none_reduce_axis - 1
    for i in range(len(reduce_axis_index) - 1, -1, -1):
        reordered_shape[temp_axis] = shape_before_reduce[reduce_axis_index[i]]
        reorder_to_ori_axis_map[temp_axis] = reduce_axis_index[i]
        ori_to_reorder_axis_map[reduce_axis_index[i]] = temp_axis
        temp_axis -= 1

    for i in range(last_none_reduce_axis - 1, -1, -1):
        if i not in reduce_axis_index:
            reordered_shape[temp_axis] = shape_before_reduce[i]
            reorder_to_ori_axis_map[temp_axis] = i
            ori_to_reorder_axis_map[i] = temp_axis
            temp_axis -= 1

    for i in range(last_none_reduce_axis, len(shape_before_reduce)):
        reorder_to_ori_axis_map[i] = i
        ori_to_reorder_axis_map[i] = i

    return reordered_shape, reorder_to_ori_axis_map, ori_to_reorder_axis_map


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
        # ub split on A axis or dont split ub
        if self.tiling_case.ub_split_axis_index not in norm_info.reduce_axis_indices or\
                self.tiling_case.ub_split_axis_index is None:
            norm_sch = NormNormalSchedule(norm_compute_graph_info, norm_info, self.tiling_case, self.outs)
        # ub split on R axis
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
        # mid tensor include pure mid tensor and non gm input tensor
        self._mid_tensor_set = graph_info.mid_tensor_set | graph_info.non_gm_input_tensor_set
        # get last endpoint output tensor
        self._res_tensor = graph_info.endpoint_output_tensor
        self._norm_info = norm_info
        self._tiling_case = tiling_case

        self._cache_read_tensors = set()
        self._cache_read_buffer_and_tensor_map = {}
        self._cache_read_tensor_and_buffer_map = {}

        self._cache_write_tensors = set()
        self._cache_write_buffer_and_tensor_map = {}
        self._cache_write_tensor_and_buffer_map = {}

        self._reduce_and_rfactor_tensor_map = {}
        self._rfactor_and_reduce_tensor_map = {}

        self._compute_inline_tensors = set()
        self._compute_inlined_tensors = set()

        self._block_split_result = {}
        self._ub_split_result = {}

        self._align_pad_and_ori_tensor_map = {}
        self._ori_and_align_pad_tensor_map = {}
        self._remove_pad_and_ori_tensor_map = {}
        self._ori_and_remove_pad_tensor_map = {}

        self._reorder_map = {}
        self._multi_core_bind_axis = None
        self._storage_align_map = {}
        self._compute_at_map = {}
        self._compute_root_tensors = set()
        self._emit_insn_map = {}

        self._is_last_common_axis_split_block = False
        self._is_last_common_axis_split_ub = False
        # has common axis
        self._is_split_block_and_ub = not norm_info.is_all_reduce
        self._is_const = not get_context().get("_const_and_dynamic_mixed") and \
            get_context().get_current_compute().get("_mode") == "const"
        self._available_size = self._graph_info.pad_available_ub_size \
            if tiling_case.is_aligned_in_ub_case else self._graph_info.available_ub_size

    def do_schedule(self):
        """
        normal norm schedule process
        """
        # normal schedule do not need workspace
        workspace_compile_info = {"common_workspace": [], "reduce_workspace": [], "broadcast_fork_workspace": []}
        current_schedule = get_context().get_current_compute().get_current_schedule()
        if current_schedule:
            current_schedule.add("_workspace_info", workspace_compile_info)

        self._sch = tvm.create_schedule([self._graph_info.endpoint_output_tensor.op])
        self._sch.tiling_key = self._tiling_case.tiling_key

        self._calc_cache_read()
        self._do_cache_read()

        self._do_align_pad()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        self._do_mid_output_tensor_process()

        self._calc_compute_inline()
        self._do_compute_inline()

        self._do_remove_pad()

        self._do_tiling()

        self._calc_reorder()
        self._do_reorder()

        self._do_rfactor()

        self._calc_storage_align()
        self._do_storage_align()

        self._calc_multi_core()
        self._do_multi_core()

        self._do_set_buffer_size()
        self._do_set_constraint()

        self._calc_compute_at()
        self._do_compute_at()

        self._do_reused_by()
        self._do_set_store_predicate()

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

    def _do_align_pad(self):
        if self._tiling_case.is_aligned_in_ub_case:
            for cache_read_tensor in self._cache_read_tensors:
                # before reduce tensor can align pad
                if cache_read_tensor in self._graph_info.before_reduce_tensor_set:
                    read_buffer = self._cache_read_tensor_and_buffer_map[cache_read_tensor]
                    align_pad_buffer = self._sch.cache_read(
                        read_buffer, self._scope, self._forward_compute_graph_map[cache_read_tensor])
                    self._align_pad_and_ori_tensor_map[align_pad_buffer] = read_buffer
                    self._ori_and_align_pad_tensor_map[read_buffer] = align_pad_buffer

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._graph_info.real_pure_output_tensor_set)

    def _do_cache_write(self):
        for cache_write_tensor in self._cache_write_tensors:
            buffer_tensor = self._sch.cache_write(cache_write_tensor, self._scope)
            self._cache_write_buffer_and_tensor_map[buffer_tensor] = cache_write_tensor
            self._cache_write_tensor_and_buffer_map[cache_write_tensor] = buffer_tensor

    def _set_scope(self):
        for mid_tensor in self._mid_tensor_set - self._graph_info.real_output_tensor_set:
            self._sch[mid_tensor].set_scope(self._scope)

    def _do_mid_output_tensor_process(self):
        for mid_output_tensor in self._graph_info.mid_output_tensor_set:
            write_buffer = self._sch.cache_write(mid_output_tensor, self._scope)
            self._cache_write_buffer_and_tensor_map[write_buffer] = mid_output_tensor
            self._cache_write_tensor_and_buffer_map[mid_output_tensor] = write_buffer
            read_buffer = self._sch.cache_read(
                mid_output_tensor, self._scope, self._forward_compute_graph_map[mid_output_tensor])
            self._cache_read_buffer_and_tensor_map[read_buffer] = mid_output_tensor
            self._cache_read_tensor_and_buffer_map[mid_output_tensor] = read_buffer
            self._sch[write_buffer].reused_by(read_buffer)

    def _calc_compute_inline(self):
        for broadcast_tensor in self._graph_info.broadcast_tensor_set:
            # nlast broadcast, broadcast can compute_inline, but broadcast can not compute inline to reduce
            if not util.is_unified_broadcast(broadcast_tensor):
                continue
            broadcast_axis = get_broadcast_axis(broadcast_tensor)
            if len(broadcast_tensor.shape) - 1 in broadcast_axis:
                continue
            if self._forward_compute_graph_map[broadcast_tensor] & self._graph_info.reduce_tensor_set:
                continue
            # broadcast can not be output tensor
            if broadcast_tensor in self._graph_info.real_output_tensor_set:
                continue
            self._compute_inline_tensors.add(broadcast_tensor)
            # compute_inlined_tensor may has been cache_write
            for compute_inlined_tensor in self._forward_compute_graph_map[broadcast_tensor]:
                self._compute_inlined_tensors.add(compute_inlined_tensor)

    def _do_compute_inline(self):
        for compute_inline_tensor in self._compute_inline_tensors:
            self._sch[compute_inline_tensor].compute_inline()

    def _do_remove_pad(self):
        if self._tiling_case.is_aligned_in_ub_case:
            for cache_write_tensor in self._cache_write_tensors:
                # before reduce tensor can remove pad
                if cache_write_tensor in self._graph_info.before_reduce_tensor_set:
                    remove_pad_buffer = self._sch.cache_write(cache_write_tensor, self._scope)
                    self._remove_pad_and_ori_tensor_map[remove_pad_buffer] = cache_write_tensor
                    self._ori_and_remove_pad_tensor_map[cache_write_tensor] = remove_pad_buffer

    def _do_tiling(self):
        block_split_axis_index = self._tiling_case.block_split_axis_index
        ub_split_axis_index = self._tiling_case.ub_split_axis_index

        if not self._is_split_block_and_ub:
            return

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

        if block_split_axis_index == ub_split_axis_index:
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
            reduce_reorder_axis, _, ori_to_reorder_axis_map = _reorder_reduce_shape(ori_axis,
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
                start_axis = max(reduce_axis_index) + 1 + self._is_split_block_and_ub
                end_axis = reorder_axis.index(self._ub_split_result["inner_itervar"])
                local_list = reorder_axis[start_axis:end_axis]
                reorder_axis[start_axis:end_axis] = []
                reorder_axis[reorder_first_r_index + self._is_split_block_and_ub:
                             reorder_first_r_index + self._is_split_block_and_ub] = local_list

            return reorder_axis

        def __calc_reduce_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reorder_axis = []

            reduce_reorder_axis, reorder_to_ori_axis_map, _ = _reorder_reduce_shape(ori_axis,
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

        if not self._is_split_block_and_ub:
            return
        # only nlast reduce mid tensor need reorder
        if not self._norm_info.is_reduce_last_axis:
            for single_tensor in self._mid_tensor_set - self._compute_inline_tensors:
                if single_tensor in self._graph_info.reduce_tensor_set:
                    reduce_tensor = self._cache_write_tensor_and_buffer_map[single_tensor]\
                        if single_tensor in self._cache_write_tensor_and_buffer_map else single_tensor
                    reorder_axis_list = __calc_reduce_tensor_reorder_axis(reduce_tensor)
                    self._reorder_map[reduce_tensor] = reorder_axis_list

        reorder_axis_list = __calc_split_tensor_reorder_axis(self._res_tensor)
        self._reorder_map[self._res_tensor] = reorder_axis_list

    def _do_reorder(self):
        for single_tensor, param in self._reorder_map.items():
            self._sch[single_tensor].reorder(*param)

    def _do_rfactor(self):
        # last reduce and discontinuous reduce
        need_rfactor = self._norm_info.is_reduce_last_axis and self._norm_info.is_discontinuous_reduce_axis
        if not need_rfactor:
            return
        # axes in ub are not equal to 1
        if self._is_split_block_and_ub:
            ub_ori_axis = self._norm_info.shape_before_reduce[self._tiling_case.ub_split_axis_index]
            if util.expr_equal(ub_ori_axis, 1):
                return
        ub_factor = self._ub_split_result.get("factor")
        if isinstance(ub_factor, int) and ub_factor == 1:
            return
        reorder_axis = self._reorder_map.get(self._res_tensor)
        ub_inner_index = reorder_axis.index(self._ub_split_result["inner_itervar"])
        for axis_index in range(ub_inner_index + 1, len(reorder_axis)):
            axis_dom = reorder_axis[axis_index].dom
            if hasattr(axis_dom.min, "value") and hasattr(axis_dom.extent, "value"):
                if axis_dom.min.value == 0 and axis_dom.extent.value == 1:
                    return

        for single_tensor in self._graph_info.reduce_tensor_set:
            # reduce tensor has been cache write
            reduce_buffer = self._cache_write_tensor_and_buffer_map[single_tensor] if \
                single_tensor in self._cache_write_tensor_and_buffer_map else single_tensor
            reduce_rf = self._sch.rfactor(reduce_buffer, reduce_buffer.op.reduce_axis[-1], -1)
            self._sch[reduce_rf].set_scope(self._scope)
            self._reduce_and_rfactor_tensor_map[reduce_buffer] = reduce_rf
            self._rfactor_and_reduce_tensor_map[reduce_rf] = reduce_buffer
            # reorder reduce axis according to reduce axis index
            rf_reduce_axis_index = util.get_reduce_axis_indexes(self._sch[reduce_rf])
            rfactor_reorder_list = self._sch[reduce_rf].op.axis[:]
            count_reduce_axis = 0
            for idx in rf_reduce_axis_index:
                rfactor_reorder_list[idx] = self._sch[reduce_rf].op.reduce_axis[count_reduce_axis]
                count_reduce_axis += 1
            self._sch[reduce_rf].reorder(*rfactor_reorder_list)

    def _calc_storage_align(self):
        def __judge_need_storage_align(_ori_tensor):
            # last broadcast tensors in broadcast fork do not need storage align
            if _ori_tensor in self._graph_info.broadcast_fork_tensor_set and \
                    util.expr_equal(_ori_tensor.shape[-1], 1):
                return False
            # after reduce tensors do not storage align when last axis reduce
            # while there is no broadcast tensor, the after reduce tensors have to do storage_align
            if self._norm_info.is_reduce_last_axis and _ori_tensor in self._graph_info.after_reduce_tensor_set and \
                    not self._norm_info.is_none_reduce:
                return False

            return True

        # shape length mush greater than 1
        if len(self._norm_info.shape_before_reduce) == 1:
            return
        storage_axis = -2
        for single_tensor in self._mid_tensor_set - self._compute_inline_tensors:
            if not __judge_need_storage_align(single_tensor):
                continue
            align_factor = get_align_factor(single_tensor.dtype)
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor, ori_tensor in self._cache_read_buffer_and_tensor_map.items():
            if not __judge_need_storage_align(ori_tensor):
                continue
            # buffer that before align buffer does not storage align
            if single_tensor not in self._ori_and_align_pad_tensor_map:
                align_factor = get_align_factor(single_tensor.dtype)
                self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]
            else:
                align_buffer = self._ori_and_align_pad_tensor_map[single_tensor]
                align_factor = get_align_factor(align_buffer.dtype)
                self._storage_align_map[align_buffer] = [align_buffer.op.axis[storage_axis], align_factor, 0]

        for single_tensor, ori_tensor in self._cache_write_buffer_and_tensor_map.items():
            if not __judge_need_storage_align(ori_tensor):
                continue
            align_factor = get_align_factor(single_tensor.dtype)
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor, ori_tensor in self._rfactor_and_reduce_tensor_map.items():
            if not __judge_need_storage_align(ori_tensor):
                continue
            align_factor = get_align_factor(single_tensor.dtype)
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

    def _do_set_buffer_size(self):
        for single_tensor in (self._mid_tensor_set - self._compute_inline_tensors)\
                .union(self._cache_read_buffer_and_tensor_map.keys())\
                .union(self._cache_write_buffer_and_tensor_map.keys())\
                .union(self._align_pad_and_ori_tensor_map.keys())\
                .union(self._remove_pad_and_ori_tensor_map.keys())\
                .union(self._rfactor_and_reduce_tensor_map.keys()):
            storage_bound_value = self._available_size * DTYPE_BYTE_MAPPING[self._graph_info.max_type] // \
                                  DTYPE_BYTE_MAPPING[single_tensor.dtype]
            self._sch[single_tensor].set_buffer_size(storage_bound_value)

        if self._res_tensor.op.tag == FAKE_NODE_TAG:
            storage_bound_value = self._available_size * DTYPE_BYTE_MAPPING[self._graph_info.max_type] // \
                                  DTYPE_BYTE_MAPPING[self._res_tensor.dtype]
            self._sch[self._res_tensor].set_buffer_size(storage_bound_value)

    def _do_set_constraint(self):
        if self._is_const:
            return

        ori_shape = self._res_tensor.shape
        if self._is_split_block_and_ub:
            ub_split_inner = self._ub_split_result["factor"]
            ori_ub_axis = self._tiling_case.ub_split_axis_index
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reduce_reorder_shape, _, ori_to_reorder_axis_map = _reorder_reduce_shape(ori_shape,
                                                                                     reduce_axis_index,
                                                                                     is_reduce_last_axis)
            reorder_ub_axis = ori_to_reorder_axis_map[ori_ub_axis]
            shape_in_ub = ub_split_inner
            self._sch.set_constraint(ub_split_inner <= self._available_size)
            for i in range(reorder_ub_axis + 1, len(reduce_reorder_shape)):
                shape_in_ub *= reduce_reorder_shape[i]
                self._sch.set_constraint(reduce_reorder_shape[i] <= self._available_size)
        else:
            shape_in_ub = 1
            for i in range(len(ori_shape)):
                shape_in_ub *= ori_shape[i]
                self._sch.set_constraint(ori_shape[i] <= self._available_size)

        self._sch.set_constraint(shape_in_ub <= self._available_size)

    def _calc_compute_at(self):
        for single_tensor in (self._mid_tensor_set - self._compute_inline_tensors)\
                .union(self._cache_read_buffer_and_tensor_map.keys())\
                .union(self._cache_write_buffer_and_tensor_map.keys())\
                .union(self._align_pad_and_ori_tensor_map.keys())\
                .union(self._remove_pad_and_ori_tensor_map.keys())\
                .union(self._rfactor_and_reduce_tensor_map.keys()):
            if self._is_split_block_and_ub:
                self._compute_at_map[single_tensor] = [self._res_tensor, self._ub_split_result["outer_itervar"]]
            else:
                self._compute_root_tensors.add(single_tensor)

    def _do_compute_at(self):
        for single_tensor, param in self._compute_at_map.items():
            self._sch[single_tensor].compute_at(self._sch[param[0]], param[1])
        for single_tensor in self._compute_root_tensors:
            self._sch[single_tensor].compute_root()

    def _do_reused_by(self):
        for single_tensor in self._graph_info.set_value_tensor_set:
            reused_tensor = list(self._backward_compute_graph_map[single_tensor])[0]
            if reused_tensor in self._cache_read_tensor_and_buffer_map:
                reused_tensor = self._cache_read_tensor_and_buffer_map[reused_tensor]
            self._sch[reused_tensor].reused_by(single_tensor)

    def _do_set_store_predicate(self):
        for single_tensor in self._graph_info.set_value_tensor_set:
            self._sch[single_tensor].set_store_predicate(single_tensor.op.body[0].condition)

    def _calc_emit_insn(self):
        def __handle_reduce_tensor(_single_tensor, _insn):
            if _single_tensor not in self._reduce_and_rfactor_tensor_map:
                _emit_insn_axis = self._sch[_single_tensor].op.axis[emit_insn_axis_index]\
                    if self._norm_info.is_reduce_last_axis else self._sch[_single_tensor].op.reduce_axis[0]
                self._emit_insn_map[_single_tensor] = [_emit_insn_axis, _insn,
                                                       {STORAGE_BOUND: self._available_size}]
            else:
                _rfactor_tensor = self._reduce_and_rfactor_tensor_map[_single_tensor]
                self._emit_insn_map[_rfactor_tensor] = [self._sch[_rfactor_tensor].op.reduce_axis[0], _insn,
                                                        {STORAGE_BOUND: self._available_size}]
                self._emit_insn_map[_single_tensor] = [self._sch[_single_tensor].op.axis[emit_insn_axis_index], _insn,
                                                       {STORAGE_BOUND: self._available_size}]
        emit_insn_axis_index = 0
        for out_tensor in self._graph_info.real_output_tensor_set:
            if self._is_split_block_and_ub and out_tensor == self._res_tensor:
                emit_insn_axis = self._ub_split_result.get("inner_itervar")
            else:
                emit_insn_axis = out_tensor.op.axis[emit_insn_axis_index]
            # enable nooverlap 2:
            # output is before reduce tensor and last common axis is split or reduce axis is discontinuous
            # copy ub to gm with stride, the last block need process the overlap too
            need_enable_no_overlap_two =  \
                out_tensor in self._graph_info.before_reduce_tensor_set and \
                (self._is_last_common_axis_split_block or self._norm_info.is_discontinuous_reduce_axis)
            attrs = {NO_OVERLAP: 2} if need_enable_no_overlap_two else {NO_OVERLAP: 3}
            self._emit_insn_map[out_tensor] = [emit_insn_axis, "dma_copy", attrs]

        for single_buffer, tensor in self._cache_read_buffer_and_tensor_map.items():
            # recache read buffer of mid output tensor does not do dma_copy
            insn = "phony_insn" if tensor in self._graph_info.mid_output_tensor_set else "dma_copy"
            self._emit_insn_map[single_buffer] = [single_buffer.op.axis[emit_insn_axis_index], insn]

        for single_tensor in self._mid_tensor_set - self._compute_inline_tensors -\
                self._graph_info.real_output_tensor_set:
            if single_tensor in self._graph_info.reduce_tensor_set:
                __handle_reduce_tensor(single_tensor, _get_insn(single_tensor))
            else:
                self._emit_insn_map[single_tensor] = [single_tensor.op.axis[emit_insn_axis_index],
                                                      _get_insn(single_tensor)]

        for single_buffer, tensor in self._cache_write_buffer_and_tensor_map.items():
            if tensor in self._graph_info.reduce_tensor_set:
                __handle_reduce_tensor(single_buffer, _get_insn(tensor))
            else:
                self._emit_insn_map[single_buffer] = [single_buffer.op.axis[emit_insn_axis_index], _get_insn(tensor)]

        if self._res_tensor.op.tag == FAKE_NODE_TAG:
            emit_insn_axis = self._ub_split_result.get("inner_itervar") if self._is_split_block_and_ub \
                else self._res_tensor.op.axis[emit_insn_axis_index]
            self._emit_insn_map[self._res_tensor] = [emit_insn_axis, "phony_insn"]

        for source, _ in self._align_pad_and_ori_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[emit_insn_axis_index], "align_pad"]

        for source, _ in self._remove_pad_and_ori_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[emit_insn_axis_index], "remove_pad"]

        for single_tensor, param in self._emit_insn_map.items():
            is_enable_vnchwconv = param[1] == "vector_broadcast" and single_tensor.dtype == "float16"
            if is_enable_vnchwconv:
                param.append({ENABLE_VNCHWCONV: True})

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
            if self._is_split_block_and_ub:
                reorder_axis = self._reorder_map.get(_single_tensor)
                start_index = reorder_axis.index(self._ub_split_result.get("inner_itervar"))
            else:
                reorder_axis = _single_tensor.op.axis
                start_index = 0

            for index in range(start_index, len(reorder_axis)):
                pragma_axis = reorder_axis[index]
                # axis of dma_tensor that after ub_split_index may has been reorder, cannot overwrite no_fuse branch
                group_id = append_id
                self._sch[_single_tensor].pragma(pragma_axis, "axis_group", group_id)

        def __mark_group_axis_on_common_tensor(_single_tensor, _tensor_type="common"):
            # axis_group = 0 means original no_fuse branch will be overwrited by fuse branch
            # axis_group = 1 means fuse branch will be appended after original no_fuse branch
            overwrite_and_append_id = tvm.make.Call("int32", "axis_group", [0, "overwrite", 1, "append"],
                                                    tvm.expr.Call.Extern, None, 0)
            append_id = tvm.make.Call("int32", "axis_group", [1, "append"], tvm.expr.Call.Extern, None, 0)
            if self._is_split_block_and_ub:
                # If the compute at tensor has been reordered, common tensor should be based on the reorder shape
                # marking group axis. Otherwise, mark group axis on outer ub axis will invalidate it.
                reorder_axis, _, ori_to_reorder_axis_map = \
                    _reorder_reduce_shape(_single_tensor.op.axis, self._norm_info.reduce_axis_indices,
                                          self._norm_info.is_reduce_last_axis)
                start_index = ori_to_reorder_axis_map[self._tiling_case.ub_split_axis_index]
            else:
                reorder_axis = _single_tensor.op.axis
                start_index = 0

            for index in range(start_index, len(reorder_axis)):
                pragma_axis = reorder_axis[index]
                # axis of dma_tensor that after ub_split_index may has been reorder, cannot overwrite no_fuse branch
                if _tensor_type == "dma_tensor":
                    group_id = append_id
                else:
                    group_id = append_id if index == len(_single_tensor.op.axis) - 1 else overwrite_and_append_id
                self._sch[_single_tensor].pragma(pragma_axis, "axis_group", group_id)

        disable_group_axis_tensor_set = \
            self._graph_info.reduce_tensor_set | self._graph_info.broadcast_tensor_set | \
            self._compute_inlined_tensors | self._graph_info.set_value_tensor_set
        # enable tensors:
        # 1. elewise tensors(compute_inlined_tensors can not fuse axis due to broadcast logic)
        # 2. dma tensors
        for single_tensor in self._mid_tensor_set - self._compute_inline_tensors:
            if single_tensor in self._graph_info.real_output_tensor_set:
                continue
            if single_tensor in disable_group_axis_tensor_set:
                continue
            __mark_group_axis_on_common_tensor(single_tensor)

        for single_tensor, ori_tensor in self._cache_write_buffer_and_tensor_map.items():
            if ori_tensor in disable_group_axis_tensor_set:
                continue
            __mark_group_axis_on_common_tensor(single_tensor)

        for single_tensor in self._cache_read_buffer_and_tensor_map:
            __mark_group_axis_on_common_tensor(single_tensor, "dma_tensor")

        for single_tensor in self._graph_info.real_output_tensor_set:
            if single_tensor != self._res_tensor:
                __mark_group_axis_on_common_tensor(single_tensor, "dma_tensor")

        if self._res_tensor.op.tag != FAKE_NODE_TAG:
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
        # mid tensor include pure mid tensor and non gm input tensor
        self._mid_tensor_set = graph_info.mid_tensor_set | graph_info.non_gm_input_tensor_set
        # all reduce tensors are workspace tensor in partial reorder sch
        self._workspace_tensor_set = graph_info.workspace_tensor_set if not tiling_case.is_partial_reorder_case \
            else graph_info.workspace_and_reduce_tensor_set
        # exclude real output tensor
        self._real_workspace_tensor_set = self._workspace_tensor_set - self._graph_info.real_output_tensor_set
        # key is split_tensor
        # value is a dict with key: sub_tensor_list, sub_tensor_consumers_map, sub_tensor_producers_map
        self._split_tensor_and_sub_graph_map = graph_info.split_tensor_and_sub_graph_map
        # get last endpoint output tensor
        self._res_tensor = graph_info.endpoint_output_tensor
        self._norm_info = norm_info
        self._tiling_case = tiling_case
        # actual workspace tensor list
        self._real_workspace_tensor_list = []
        # key is workspace_tensor
        # value is a dict with key: "ub_tensor" and "reread_ub_tensor"({cache_read_tensor: split_tensor})
        self._workspace_map = {}

        self._cache_read_tensors = set()
        self._cache_write_tensors = set()
        self._cache_clone_tensors = set()
        # key is ub_buffer
        # value is a dict with key: ori_tensor, value: split_tensor
        self._cache_read_buffer_and_tensor_dual_map = {}
        self._cache_write_buffer_and_tensor_dual_map = {}
        self._cache_clone_buffer_and_tensor_dual_map = {}
        # key is ori_tensor
        # value is a dict with key: ub_buffer, value: split_tensor
        self._cache_read_tensor_and_buffer_dual_map = {}
        self._cache_write_tensor_and_buffer_dual_map = {}
        self._cache_clone_tensor_and_buffer_dual_map = {}

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
        self._reused_map = {}
        self._emit_insn_map = {}

        self._is_last_common_axis_split_block = False
        self._is_split_block = not norm_info.is_all_reduce
        self._is_const = not get_context().get("_const_and_dynamic_mixed") and \
            get_context().get_current_compute().get("_mode") == "const"

    def do_schedule(self):
        """
        workspace norm schedule process
        """
        self._pre_workspace_process()
        real_output_tensor_op_list = [self._graph_info.endpoint_output_tensor.op]
        for workspace_tensor in self._real_workspace_tensor_list:
            self._outs.append(workspace_tensor)
            real_output_tensor_op_list.append(workspace_tensor.op)

        self._sch = tvm.create_schedule(real_output_tensor_op_list)
        self._sch.tiling_key = self._tiling_case.tiling_key

        self._calc_cache_clone()
        self._do_cache_clone()

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        self._do_mid_output_tensor_process()

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

        self._do_set_buffer_size()
        self._do_set_constraint()

        self._calc_compute_at()
        self._do_compute_at()
        self._do_compute_root()

        self._calc_reused_by()
        self._do_reused_by()
        self._do_set_store_predicate()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._do_pragma()

        return self._sch

    def _pre_workspace_process(self):
        workspace_compile_info_dict = {
            "common_workspace": [],
            "reduce_workspace": [],
            "broadcast_fork_workspace": []
        }
        common_workspace_list = []
        reduce_workspace_list = []
        broadcast_fork_workspace_list = []

        for workspace_tensor in self._real_workspace_tensor_set:
            # single_workspace_info include workspace type and workspace element bytes
            single_workspace_info = []
            # used to calculate the size of workspace, 1 means before reduce, 0 means after reduce
            if workspace_tensor in self._graph_info.before_reduce_tensor_set:
                single_workspace_info.append(1)
            else:
                single_workspace_info.append(0)
            # number of bytes of an element
            single_workspace_info.append(DTYPE_BYTE_MAPPING[workspace_tensor.dtype])

            if workspace_tensor in self._graph_info.reduce_tensor_set:
                workspace_compile_info_dict["reduce_workspace"].append(tuple(single_workspace_info))
                reduce_workspace_list.append(workspace_tensor)
            elif workspace_tensor in self._graph_info.broadcast_fork_tensor_set:
                workspace_compile_info_dict["broadcast_fork_workspace"].append(tuple(single_workspace_info))
                broadcast_fork_workspace_list.append(workspace_tensor)
            else:
                workspace_compile_info_dict["common_workspace"].append(tuple(single_workspace_info))
                common_workspace_list.append(workspace_tensor)

        get_context().get_current_compute().get_current_schedule().add("_workspace_info", workspace_compile_info_dict)

        self._real_workspace_tensor_list = common_workspace_list[:]
        self._real_workspace_tensor_list.extend(reduce_workspace_list)
        self._real_workspace_tensor_list.extend(broadcast_fork_workspace_list)

    def _calc_cache_clone(self):
        self._cache_clone_tensors.update(self._graph_info.cache_clone_tensor_set)

    def _do_cache_clone(self):
        for cache_clone_tensor in self._cache_clone_tensors:
            cache_clone_count = 1
            if cache_clone_tensor in self._graph_info.input_tensor_set:
                continue
            for sub_graph_split_tensor, sub_graph_map in self._split_tensor_and_sub_graph_map.items():
                sub_tensor_list = sub_graph_map["sub_tensor_list"]
                if cache_clone_tensor in sub_tensor_list:
                    sub_tensor_consumers_map = sub_graph_map["sub_tensor_consumers_map"]
                    local_consumers = list(sub_tensor_consumers_map[cache_clone_tensor])
                    for idx, tensor in enumerate(local_consumers):
                        # the readers have cache clone tensor
                        if tensor in self._cache_clone_tensor_and_buffer_dual_map:
                            local_consumers[idx] = \
                                _get_sub_dict_first_key(self._cache_clone_tensor_and_buffer_dual_map, tensor)
                    clone_buffer = self._sch.cache_clone(cache_clone_tensor, self._scope, local_consumers)
                    _add_sub_dict_to_dict(self._cache_clone_buffer_and_tensor_dual_map, clone_buffer,
                                          cache_clone_tensor, sub_graph_split_tensor)
                    _add_sub_dict_to_dict(self._cache_clone_tensor_and_buffer_dual_map, cache_clone_tensor,
                                          clone_buffer, sub_graph_split_tensor)
                    # cache clone count is equal to num_path - 1
                    if cache_clone_count >= \
                            self._graph_info.cache_clone_tensor_and_num_path_map[cache_clone_tensor] - 1:
                        break
                    cache_clone_count += 1

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._graph_info.input_tensor_set)

    def _do_cache_read(self):
        for cache_read_tensor in self._cache_read_tensors:
            for sub_graph_split_tensor, sub_graph_map in self._split_tensor_and_sub_graph_map.items():
                sub_tensor_list = sub_graph_map["sub_tensor_list"]
                if cache_read_tensor in sub_tensor_list:
                    sub_tensor_consumers_map = sub_graph_map["sub_tensor_consumers_map"]
                    local_consumers = list(sub_tensor_consumers_map[cache_read_tensor])
                    for idx, tensor in enumerate(local_consumers):
                        # the readers have cache clone tensor with same sub_graph_split_tensor
                        if tensor in self._cache_clone_tensor_and_buffer_dual_map:
                            for cache_clone_buffer, cur_split_tensor in\
                                    self._cache_clone_tensor_and_buffer_dual_map[tensor].items():
                                if sub_graph_split_tensor == cur_split_tensor:
                                    local_consumers[idx] = cache_clone_buffer
                    read_buffer = self._sch.cache_read(cache_read_tensor, self._scope, local_consumers)
                    _add_sub_dict_to_dict(self._cache_read_buffer_and_tensor_dual_map, read_buffer,
                                          cache_read_tensor, sub_graph_split_tensor)
                    _add_sub_dict_to_dict(self._cache_read_tensor_and_buffer_dual_map, cache_read_tensor,
                                          read_buffer, sub_graph_split_tensor)

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._graph_info.real_pure_output_tensor_set - self._workspace_tensor_set)

    def _do_cache_write(self):
        for cache_write_tensor in self._cache_write_tensors:
            buffer_tensor = self._sch.cache_write(cache_write_tensor, self._scope)
            _add_sub_dict_to_dict(self._cache_write_buffer_and_tensor_dual_map, buffer_tensor,
                                  cache_write_tensor, self._res_tensor)
            _add_sub_dict_to_dict(self._cache_write_tensor_and_buffer_dual_map, cache_write_tensor,
                                  buffer_tensor, self._res_tensor)

    def _get_reduce_tensor_cache_write_buffer(self, _single_tensor):
        if _single_tensor not in self._graph_info.reduce_tensor_set:
            return _single_tensor
        if _single_tensor not in self._cache_write_tensor_and_buffer_dual_map:
            return _single_tensor

        return _get_sub_dict_first_key(self._cache_write_tensor_and_buffer_dual_map, _single_tensor)

    def _set_scope(self):
        for mid_tensor in self._mid_tensor_set - self._graph_info.real_output_tensor_set:
            self._sch[mid_tensor].set_scope(self._scope)

    def _do_mid_output_tensor_process(self):
        for mid_output_tensor in self._graph_info.mid_output_tensor_set - self._workspace_tensor_set:
            for sub_graph_split_tensor, sub_graph_map in self._split_tensor_and_sub_graph_map.items():
                sub_tensor_list = sub_graph_map["sub_tensor_list"]
                if mid_output_tensor in sub_tensor_list and mid_output_tensor != sub_graph_split_tensor:
                    write_buffer = self._sch.cache_write(mid_output_tensor, self._scope)
                    _add_sub_dict_to_dict(self._cache_write_buffer_and_tensor_dual_map, write_buffer,
                                          mid_output_tensor, sub_graph_split_tensor)
                    _add_sub_dict_to_dict(self._cache_write_tensor_and_buffer_dual_map, mid_output_tensor,
                                          write_buffer, sub_graph_split_tensor)
                    sub_tensor_consumers_map = sub_graph_map["sub_tensor_consumers_map"]
                    local_consumers = list(sub_tensor_consumers_map[mid_output_tensor])
                    # readers may have tensors that has been cache write
                    for idx, single_tensor in enumerate(local_consumers):
                        if single_tensor in self._cache_write_tensor_and_buffer_dual_map:
                            local_consumers[idx] =\
                                _get_sub_dict_first_key(self._cache_write_tensor_and_buffer_dual_map, single_tensor)
                    read_buffer = self._sch.cache_read(mid_output_tensor, self._scope, local_consumers)
                    _add_sub_dict_to_dict(self._cache_read_buffer_and_tensor_dual_map, read_buffer,
                                          mid_output_tensor, sub_graph_split_tensor)
                    _add_sub_dict_to_dict(self._cache_read_tensor_and_buffer_dual_map, mid_output_tensor,
                                          read_buffer, sub_graph_split_tensor)
                    self._sch[write_buffer].reused_by(read_buffer)

    def _do_workspace_process(self):
        for workspace_tensor in self._workspace_tensor_set:
            self._workspace_map[workspace_tensor] = {}
            for sub_graph_split_tensor, sub_graph_map in self._split_tensor_and_sub_graph_map.items():
                sub_tensor_list = sub_graph_map["sub_tensor_list"]
                if workspace_tensor in sub_tensor_list and workspace_tensor != sub_graph_split_tensor:
                    sub_tensor_consumers_map = sub_graph_map["sub_tensor_consumers_map"]
                    local_consumers = list(sub_tensor_consumers_map[workspace_tensor])
                    # if workspace_tensor has consumers
                    if local_consumers:
                        # readers may have tensors that has been cache write
                        for idx, single_tensor in enumerate(local_consumers):
                            if single_tensor in self._cache_write_tensor_and_buffer_dual_map:
                                local_consumers[idx] =\
                                    _get_sub_dict_first_key(self._cache_write_tensor_and_buffer_dual_map,
                                                            single_tensor)
                        read_buffer = self._sch.cache_read(workspace_tensor, self._scope, local_consumers)
                        _add_sub_dict_to_dict(self._workspace_map[workspace_tensor], "reread_ub_tensor",
                                              read_buffer, sub_graph_split_tensor)
                    else:
                        self._workspace_map[workspace_tensor]["reread_ub_tensor"] = {}

        for workspace_tensor in self._workspace_tensor_set:
            workspace_ub_tensor = self._sch.cache_write(workspace_tensor, self._scope)
            self._workspace_map[workspace_tensor]["ub_tensor"] = workspace_ub_tensor

        for workspace_tensor in self._workspace_tensor_set:
            self._sch[workspace_tensor].set_scope("global")

    def _calc_compute_inline(self):
        for broadcast_tensor in self._graph_info.broadcast_tensor_set:
            # nlast unified broadcast, broadcast could compute_inline
            if not util.is_unified_broadcast(broadcast_tensor):
                continue
            broadcast_axis = get_broadcast_axis(broadcast_tensor)
            if len(broadcast_tensor.shape) - 1 in broadcast_axis:
                continue
            # broadcast do not compute_inline to workspace
            if self._forward_compute_graph_map[broadcast_tensor] & self._workspace_tensor_set:
                continue
            # broadcast do not compute_inline to reduce
            if self._forward_compute_graph_map[broadcast_tensor] & self._graph_info.reduce_tensor_set:
                continue
            # broadcast can not be workspace tensor
            if broadcast_tensor in self._workspace_tensor_set:
                continue
            # broadcast can not be output tensor
            if broadcast_tensor in self._graph_info.real_output_tensor_set:
                continue

            self._compute_inline_tensors.add(broadcast_tensor)
            # compute_inlined_tensor may has been cache_write
            for compute_inlined_tensor in self._forward_compute_graph_map[broadcast_tensor]:
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
        for split_tensor in self._graph_info.workspace_and_reduce_tensor_set:
            if split_tensor in self._graph_info.reduce_tensor_set:
                if split_tensor in self._workspace_tensor_set:
                    split_tensor = self._workspace_map[split_tensor]["ub_tensor"]
                # reduce tensor is output tensor and has been cache write
                # ub split cache write buffer
                split_tensor = self._get_reduce_tensor_cache_write_buffer(split_tensor)
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
            reduce_reorder_axis, _, ori_to_reorder_axis_map = _reorder_reduce_shape(ori_axis,
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
            reduce_reorder_axis, reorder_to_ori_axis_map, _ = _reorder_reduce_shape(ori_axis,
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
            for split_tensor in self._graph_info.workspace_and_reduce_tensor_set:
                if split_tensor in self._graph_info.reduce_tensor_set:
                    if split_tensor in self._workspace_tensor_set:
                        workspace_ub_tensor = self._workspace_map[split_tensor]["ub_tensor"]
                        reorder_axis_list = __calc_split_tensor_reorder_axis(split_tensor)
                        self._reorder_map[split_tensor] = reorder_axis_list
                        reorder_axis_list = __calc_reduce_tensor_reorder_axis(workspace_ub_tensor)
                        self._reorder_map[workspace_ub_tensor] = reorder_axis_list
                    else:
                        split_tensor = self._get_reduce_tensor_cache_write_buffer(split_tensor)
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

            for out_tensor in self._graph_info.real_output_tensor_set:
                if out_tensor in self._graph_info.reduce_tensor_set:
                    reorder_axis_list = __calc_split_tensor_reorder_axis(out_tensor)
                    self._reorder_map[out_tensor] = reorder_axis_list

        if self._tiling_case.is_partial_reorder_case:
            __partial_reorder_process()
        else:
            __common_reorder_process()

    def _do_reorder(self):
        for single_tensor, param in self._reorder_map.items():
            self._sch[single_tensor].reorder(*param)

    def _calc_storage_align(self):
        def __judge_need_storage_align(_ori_tensor):
            # last broadcast tensors in broadcast fork do not need storage align
            if _ori_tensor in self._graph_info.broadcast_fork_tensor_set and \
                    util.expr_equal(_ori_tensor.shape[-1], 1):
                return False
            # after reduce tensors do not storage align when last axis reduce
            # while there is no broadcast tensor, the after reduce tensors have to do storage_align
            if self._norm_info.is_reduce_last_axis and _ori_tensor in self._graph_info.after_reduce_tensor_set and \
                    not self._norm_info.is_none_reduce:
                return False

            return True

        # shape length mush greater than 1
        if len(self._norm_info.shape_before_reduce) == 1:
            return
        storage_axis = -2
        for single_tensor in self._mid_tensor_set - self._compute_inline_tensors - self._workspace_tensor_set:
            if not __judge_need_storage_align(single_tensor):
                continue
            align_factor = get_align_factor(single_tensor.dtype)
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor, tensor_map in self._cache_read_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            if not __judge_need_storage_align(ori_tensor):
                continue
            align_factor = get_align_factor(single_tensor.dtype)
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor, tensor_map in self._cache_write_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            if not __judge_need_storage_align(ori_tensor):
                continue
            align_factor = get_align_factor(single_tensor.dtype)
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for single_tensor, tensor_map in self._cache_clone_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            if not __judge_need_storage_align(ori_tensor):
                continue
            align_factor = get_align_factor(single_tensor.dtype)
            self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

        for workspace_tensor in self._workspace_tensor_set:
            if not __judge_need_storage_align(workspace_tensor):
                continue
            workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
            reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
            align_factor = get_align_factor(workspace_ub_tensor.dtype)
            storage_axis = len(workspace_ub_tensor.shape) - 2
            self._storage_align_map[workspace_ub_tensor] = [workspace_ub_tensor.op.axis[storage_axis],
                                                            align_factor, 0]
            for single_tensor in reread_workspace_ub_tensor_map:
                align_factor = get_align_factor(single_tensor.dtype)
                storage_axis = len(single_tensor.shape) - 2
                self._storage_align_map[single_tensor] = [single_tensor.op.axis[storage_axis], align_factor, 0]

    def _do_storage_align(self):
        for single_tensor, param in self._storage_align_map.items():
            self._sch[single_tensor].storage_align(param[0], param[1], param[2])

    def _calc_bind_buffer(self):
        # workspce tensor should do bind buffer when
        # nlast reduce or last reduce but workspace tensor is before reduce tensor
        if len(self._norm_info.shape_before_reduce) == 1:
            return
        for workspace_tensor in self._real_workspace_tensor_set:
            if not self._norm_info.is_reduce_last_axis or\
                    workspace_tensor in self._graph_info.before_reduce_tensor_set:
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
            if self._res_tensor in self._reorder_map:
                reorder_axis = self._reorder_map[self._res_tensor]
                fuse_axis_list = reorder_axis[:reorder_axis.index(block_bind_axis) + 1]
            else:
                # partial reorder sch, res do not need reorder except reduce tensor
                fuse_axis_list = [block_bind_axis]
            self._multi_core_bind_axis_map[self._res_tensor] = self._sch[self._res_tensor].fuse(*fuse_axis_list)

            for workspace_tensor in self._workspace_tensor_set:
                block_bind_axis = self._block_split_result[workspace_tensor]["outer_itervar"]
                if workspace_tensor in self._reorder_map:
                    reorder_axis = self._reorder_map[workspace_tensor]
                    fuse_axis_list = reorder_axis[:reorder_axis.index(block_bind_axis) + 1]
                else:
                    fuse_axis_list = [block_bind_axis]
                self._multi_core_bind_axis_map[workspace_tensor] = self._sch[workspace_tensor].fuse(*fuse_axis_list)

    def _do_multi_core(self):
        if self._multi_core_bind_axis_map:
            block = tvm.thread_axis(BLOCK_IDX)
            for single_tensor, param in self._multi_core_bind_axis_map.items():
                self._sch[single_tensor].bind(param, block)

    def _do_set_buffer_size(self):
        storage_bound_tensors = (self._mid_tensor_set - self._compute_inline_tensors - self._workspace_tensor_set)\
                .union(self._cache_read_buffer_and_tensor_dual_map.keys())\
                .union(self._cache_write_buffer_and_tensor_dual_map.keys())\
                .union(self._cache_clone_buffer_and_tensor_dual_map.keys())

        for workspace_tensor in self._workspace_tensor_set:
            workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
            reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
            storage_bound_tensors.add(workspace_ub_tensor)
            storage_bound_tensors = storage_bound_tensors.union(reread_workspace_ub_tensor_map.keys())

        if self._res_tensor.op.tag == FAKE_NODE_TAG:
            storage_bound_tensors.add(self._res_tensor)

        for single_tensor in storage_bound_tensors:
            storage_bound_value = self._graph_info.workspace_available_ub_size * \
                                  DTYPE_BYTE_MAPPING[self._graph_info.max_type] // \
                                  DTYPE_BYTE_MAPPING[single_tensor.dtype]
            self._sch[single_tensor].set_buffer_size(storage_bound_value)

    def _do_set_constraint(self):
        if self._is_const:
            return

        ori_shape = self._res_tensor.shape
        ub_split_inner = self._ub_split_result[self._res_tensor]["factor"]
        ori_ub_axis = self._tiling_case.ub_split_axis_index
        reduce_axis_index = self._norm_info.reduce_axis_indices
        is_reduce_last_axis = self._norm_info.is_reduce_last_axis
        reduce_reorder_shape, _, ori_to_reorder_axis_map = _reorder_reduce_shape(ori_shape,
                                                                                 reduce_axis_index,
                                                                                 is_reduce_last_axis)
        reorder_ub_axis = ori_to_reorder_axis_map[ori_ub_axis]
        shape_in_ub = ub_split_inner
        self._sch.set_constraint(ub_split_inner <= self._graph_info.workspace_available_ub_size)
        for i in range(reorder_ub_axis + 1, len(reduce_reorder_shape)):
            shape_in_ub *= reduce_reorder_shape[i]
            self._sch.set_constraint(reduce_reorder_shape[i] <= self._graph_info.workspace_available_ub_size)
        if self._is_last_common_axis_split_block:
            blk_split_inner = self._block_split_result[self._res_tensor]["factor"]
            shape_in_ub = shape_in_ub // reduce_reorder_shape[-1] * blk_split_inner
            self._sch.set_constraint(blk_split_inner <= self._graph_info.workspace_available_ub_size)

        self._sch.set_constraint(shape_in_ub <= self._graph_info.workspace_available_ub_size)

    def _calc_compute_at(self):
        def __get_compute_at_workspace_ub_tensor(_ori_compute_at_tensor):
            # reduce workspace tensor compute at to the workspace_ub_tensor
            if _ori_compute_at_tensor not in self._ub_split_result and _ori_compute_at_tensor in self._workspace_map:
                return self._workspace_map[_ori_compute_at_tensor]["ub_tensor"]

            return _ori_compute_at_tensor

        def __handle_special_after_reduce_tensor(_single_tensor, _compute_at_tensor):
            # reduce fork tensor and after reduce tensor around mid output tensor should compute at block_outer
            if _compute_at_tensor not in self._block_split_result:
                self._compute_root_tensors.add(_single_tensor)
            else:
                if _compute_at_tensor in self._multi_core_bind_axis_map:
                    self._compute_at_map[_single_tensor] = \
                        [_compute_at_tensor, self._multi_core_bind_axis_map[_compute_at_tensor]]
                else:
                    self._compute_at_map[_single_tensor] = \
                        [_compute_at_tensor, self._block_split_result[_compute_at_tensor]["outer_itervar"]]

        def __handle_common_reduce_tensor(_tensor, _compute_at_tensor):
            if not judge_tvm_shape_equal(_compute_at_tensor.shape, self._norm_info.shape_before_reduce):
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
            _, reorder_to_ori_axis_map, ori_to_reorder_axis_map =\
                _reorder_reduce_shape(ori_shape, reduce_axis_index, is_reduce_last_axis)
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

                for reread_workspace_ub_tensor, local_compute_at_tensor in reread_workspace_ub_tensor_map.items():
                    # compute at tensor is workspace
                    local_compute_at_tensor = __get_compute_at_workspace_ub_tensor(local_compute_at_tensor)
                    # compute at tensor is reduce output tensor
                    local_compute_at_tensor = self._get_reduce_tensor_cache_write_buffer(local_compute_at_tensor)
                    if workspace_tensor in special_after_reduce_tensor_set:
                        __handle_special_after_reduce_tensor(reread_workspace_ub_tensor, local_compute_at_tensor)
                    else:
                        self._compute_at_map[reread_workspace_ub_tensor] =\
                            [local_compute_at_tensor, self._ub_split_result[local_compute_at_tensor]["outer_itervar"]]

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

        # need compute at block outer
        special_after_reduce_tensor_set = (self._graph_info.reduce_fork_tensor_set |
                                           self._graph_info.after_mid_out_and_before_broadcast_tensor_set |
                                           self._graph_info.before_mid_out_and_after_reduce_tensor_set)

        for single_tensor in self._mid_tensor_set - self._compute_inline_tensors - self._workspace_tensor_set:
            for sub_graph_split_tensor, sub_graph_map in self._split_tensor_and_sub_graph_map.items():
                sub_tensor_list = sub_graph_map["sub_tensor_list"]
                if single_tensor in sub_tensor_list and single_tensor != sub_graph_split_tensor:
                    compute_at_tensor = __get_compute_at_workspace_ub_tensor(sub_graph_split_tensor)
                    compute_at_tensor = self._get_reduce_tensor_cache_write_buffer(compute_at_tensor)
                    if single_tensor in special_after_reduce_tensor_set:
                        __handle_special_after_reduce_tensor(single_tensor, compute_at_tensor)
                        continue
                    if single_tensor in self._graph_info.reduce_tensor_set:
                        __handle_common_reduce_tensor(single_tensor, compute_at_tensor)
                    else:
                        self._compute_at_map[single_tensor] =\
                            [compute_at_tensor, self._ub_split_result[compute_at_tensor]["outer_itervar"]]

        for single_tensor, tensor_map in self._cache_read_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            ori_compute_at_tensor = list(tensor_map.values())[0]
            compute_at_tensor = __get_compute_at_workspace_ub_tensor(ori_compute_at_tensor)
            compute_at_tensor = self._get_reduce_tensor_cache_write_buffer(compute_at_tensor)
            if ori_tensor in special_after_reduce_tensor_set:
                __handle_special_after_reduce_tensor(single_tensor, compute_at_tensor)
                continue
            self._compute_at_map[single_tensor] = [compute_at_tensor,
                                                   self._ub_split_result[compute_at_tensor]["outer_itervar"]]

        for single_tensor, tensor_map in self._cache_write_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            ori_compute_at_tensor = list(tensor_map.values())[0]
            compute_at_tensor = __get_compute_at_workspace_ub_tensor(ori_compute_at_tensor)
            compute_at_tensor = self._get_reduce_tensor_cache_write_buffer(compute_at_tensor)
            if ori_tensor in special_after_reduce_tensor_set:
                __handle_special_after_reduce_tensor(single_tensor, compute_at_tensor)
                continue
            if ori_tensor in self._graph_info.reduce_tensor_set:
                __handle_common_reduce_tensor(single_tensor, compute_at_tensor)
                continue

            self._compute_at_map[single_tensor] = [compute_at_tensor,
                                                   self._ub_split_result[compute_at_tensor]["outer_itervar"]]

        for single_tensor, tensor_map in self._cache_clone_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            ori_compute_at_tensor = list(tensor_map.values())[0]
            compute_at_tensor = __get_compute_at_workspace_ub_tensor(ori_compute_at_tensor)
            compute_at_tensor = self._get_reduce_tensor_cache_write_buffer(compute_at_tensor)
            if ori_tensor in special_after_reduce_tensor_set:
                __handle_special_after_reduce_tensor(single_tensor, compute_at_tensor)
                continue
            self._compute_at_map[single_tensor] = [compute_at_tensor,
                                                   self._ub_split_result[compute_at_tensor]["outer_itervar"]]

        __handle_workspace_tensor()

    def _do_compute_at(self):
        for single_tensor, param in self._compute_at_map.items():
            self._sch[single_tensor].compute_at(self._sch[param[0]], param[1])

    def _do_compute_root(self):
        for single_tensor in self._compute_root_tensors:
            self._sch[single_tensor].compute_root()

    def _calc_reused_by(self):
        for single_tensor in self._graph_info.set_value_tensor_set:
            if single_tensor in self._graph_info.workspace_tensor_set:
                reused_tensor = list(
                    self._split_tensor_and_sub_graph_map[single_tensor]["sub_tensor_producers_map"][single_tensor])[0]
                if reused_tensor in self._cache_read_tensor_and_buffer_dual_map:
                    reused_tensor = _get_sub_dict_first_key(self._cache_read_tensor_and_buffer_dual_map,
                                                            reused_tensor)
                if reused_tensor in self._cache_clone_tensor_and_buffer_dual_map:
                    reused_tensor = _get_sub_dict_first_key(self._cache_clone_tensor_and_buffer_dual_map,
                                                            reused_tensor)
                self._reused_map[self._workspace_map[single_tensor]["ub_tensor"]] = reused_tensor
                continue

            for sub_graph_split_tensor, sub_graph_map in self._split_tensor_and_sub_graph_map.items():
                sub_tensor_list = sub_graph_map["sub_tensor_list"]
                if single_tensor in sub_tensor_list and single_tensor != sub_graph_split_tensor:
                    reused_tensor = list(sub_graph_map["sub_tensor_producers_map"][single_tensor])[0]
                    if single_tensor in self._cache_clone_tensor_and_buffer_dual_map:
                        reuse_tensor = _get_sub_dict_first_key(self._cache_clone_tensor_and_buffer_dual_map,
                                                               single_tensor)
                        sub_graph_split_tensor = \
                            _get_sub_dict_first_value(self._cache_clone_tensor_and_buffer_dual_map, single_tensor)
                        if reused_tensor in self._cache_read_tensor_and_buffer_dual_map:
                            for cache_read_buffer in self._cache_read_tensor_and_buffer_dual_map[reused_tensor]:
                                if self._cache_read_tensor_and_buffer_dual_map[reused_tensor][cache_read_buffer] == \
                                        sub_graph_split_tensor:
                                    self._reused_map[reuse_tensor] = cache_read_buffer
                                else:
                                    self._reused_map[single_tensor] = cache_read_buffer
                    elif reused_tensor in self._cache_read_tensor_and_buffer_dual_map:
                        self._reused_map[single_tensor] = \
                            _get_sub_dict_first_key(self._cache_read_tensor_and_buffer_dual_map, single_tensor)
                    else:
                        self._reused_map[single_tensor] = reused_tensor

    def _do_reused_by(self):
        for single_tensor, reuse_tensor in self._reused_map.items():
            self._sch[reuse_tensor].reused_by(single_tensor)

    def _do_set_store_predicate(self):
        for single_tensor in self._reused_map:
            self._sch[single_tensor].set_store_predicate(single_tensor.op.body[0].condition)

    def _calc_emit_insn(self):
        def __handle_workspace_tensor():
            for workspace_tensor in self._workspace_tensor_set:
                workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
                reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
                # src and dst are align
                is_align = workspace_tensor in self._bind_buffer_map and \
                    (self._tiling_case.block_split_axis_index != len(self._norm_info.shape_before_reduce) - 1 and
                     self._tiling_case.ub_split_axis_index != len(self._norm_info.shape_before_reduce) - 1)
                # non_reduce workspace node, block and ub split on workspace tensor
                if workspace_tensor not in self._graph_info.reduce_tensor_set:
                    self._emit_insn_map[workspace_tensor] =\
                            [self._ub_split_result[workspace_tensor]["inner_itervar"], "dma_copy"]

                    if is_align:
                        self._emit_insn_map.get(workspace_tensor).append({NO_OVERLAP: 0})
                    elif need_enable_no_overlap_two and workspace_tensor in self._graph_info.before_reduce_tensor_set:
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
                         {STORAGE_BOUND: self._graph_info.workspace_available_ub_size}]

                for reread_workspace_ub_tensor in reread_workspace_ub_tensor_map:
                    self._emit_insn_map[reread_workspace_ub_tensor] =\
                        [reread_workspace_ub_tensor.op.axis[0], "dma_copy"]

        def __handle_real_output_tensor():
            for out_tensor in self._graph_info.real_output_tensor_set:
                emit_insn_axis = self._ub_split_result[self._res_tensor]["inner_itervar"]\
                    if out_tensor == self._res_tensor else out_tensor.op.axis[0]
                attrs = {NO_OVERLAP: 2} if out_tensor in self._graph_info.before_reduce_tensor_set and \
                    need_enable_no_overlap_two else {NO_OVERLAP: 3}
                self._emit_insn_map[out_tensor] = [emit_insn_axis, "dma_copy", attrs]

                if self._tiling_case.is_partial_reorder_case:
                    if self._tiling_case.block_split_axis_index > self._tiling_case.ub_split_axis_index:
                        if out_tensor == self._res_tensor:
                            self._emit_insn_map[self._res_tensor] =\
                                [self._block_split_result[self._res_tensor]["inner_itervar"], "dma_copy"]

        need_enable_no_overlap_two = \
            self._is_last_common_axis_split_block or self._norm_info.is_discontinuous_reduce_axis

        for single_tensor in self._mid_tensor_set - self._compute_inline_tensors - \
                self._workspace_tensor_set - self._graph_info.real_output_tensor_set:
            if single_tensor in self._graph_info.reduce_tensor_set:
                self._emit_insn_map[single_tensor] = \
                    [self._ub_split_result[single_tensor]["inner_itervar"], _get_insn(single_tensor),
                     {STORAGE_BOUND: self._graph_info.workspace_available_ub_size}]
                continue

            self._emit_insn_map[single_tensor] = [single_tensor.op.axis[0], _get_insn(single_tensor)]

        for single_buffer in self._cache_clone_buffer_and_tensor_dual_map:
            self._emit_insn_map[single_buffer] = [single_buffer.op.axis[0], _get_insn(single_buffer)]

        for single_buffer, tensor_map in self._cache_read_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            if ori_tensor in self._graph_info.mid_output_tensor_set:
                self._emit_insn_map[single_buffer] = [single_buffer.op.axis[0], "phony_insn"]
                continue
            self._emit_insn_map[single_buffer] = [single_buffer.op.axis[0], "dma_copy"]

        for single_buffer, tensor_map in self._cache_write_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            if ori_tensor in self._graph_info.reduce_tensor_set:
                self._emit_insn_map[single_buffer] = \
                    [self._ub_split_result[single_buffer]["inner_itervar"], _get_insn(ori_tensor),
                     {STORAGE_BOUND: self._graph_info.workspace_available_ub_size}]
                continue
            self._emit_insn_map[single_buffer] = [single_buffer.op.axis[0], _get_insn(ori_tensor)]

        __handle_workspace_tensor()
        __handle_real_output_tensor()

        if self._res_tensor.op.tag == FAKE_NODE_TAG:
            self._emit_insn_map[self._res_tensor] = [self._ub_split_result[self._res_tensor]["inner_itervar"],
                                                     "phony_insn"]

        for single_tensor, param in self._emit_insn_map.items():
            if param[1] == "vector_broadcast":
                param.append({ENABLE_VNCHWCONV: True})

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
                        if index == self._tiling_case.ub_split_axis_index and _single_tensor in self._ub_split_result\
                        else _single_tensor.op.axis[index]
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
                _reorder_reduce_shape(_single_tensor.op.axis, self._norm_info.reduce_axis_indices,
                                      self._norm_info.is_reduce_last_axis)
            reorder_ub_axis_index = ori_to_reorder_axis_map[self._tiling_case.ub_split_axis_index]
            for index in range(reorder_ub_axis_index, len(reorder_axis)):
                pragma_axis = reorder_axis[index]
                # after ub_split_index may has been reorder, cannot overwrite no_fuse branch
                if _tensor_type == "dma_tensor":
                    group_id = append_id
                else:
                    group_id = append_id if index == len(_single_tensor.shape) - 1 else overwrite_and_append_id
                self._sch[_single_tensor].pragma(pragma_axis, "axis_group", group_id)

        disable_group_axis_tensor_set = \
            self._graph_info.reduce_tensor_set | self._graph_info.broadcast_tensor_set | \
            self._compute_inlined_tensors | self._graph_info.set_value_tensor_set

        # enable tensors:
        # 1. elewise tensors(compute_inlined_tensors can not fuse axis due to broadcast logic)
        # 2. dma tensors
        for single_tensor in self._mid_tensor_set - self._compute_inline_tensors - self._workspace_tensor_set - \
                self._graph_info.real_output_tensor_set:
            if single_tensor in disable_group_axis_tensor_set:
                continue
            __mark_group_axis_on_common_tensor(single_tensor)

        for single_buffer in self._cache_read_buffer_and_tensor_dual_map:
            __mark_group_axis_on_common_tensor(single_buffer, "dma_tensor")

        for single_buffer, tensor_map in self._cache_write_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            if ori_tensor in disable_group_axis_tensor_set:
                continue
            __mark_group_axis_on_common_tensor(single_buffer)

        for single_buffer, tensor_map in self._cache_clone_buffer_and_tensor_dual_map.items():
            ori_tensor = list(tensor_map.keys())[0]
            if ori_tensor in disable_group_axis_tensor_set:
                continue
            __mark_group_axis_on_common_tensor(single_buffer)

        for workspace_tensor in self._workspace_tensor_set:
            __mark_group_axis_on_split_tensor(workspace_tensor)
            workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
            reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
            if workspace_tensor not in disable_group_axis_tensor_set:
                __mark_group_axis_on_common_tensor(workspace_ub_tensor)
            for reread_workspace_ub_tensor in reread_workspace_ub_tensor_map:
                __mark_group_axis_on_common_tensor(reread_workspace_ub_tensor, "dma_tensor")

        for single_tensor in self._graph_info.real_output_tensor_set:
            if single_tensor != self._res_tensor:
                __mark_group_axis_on_common_tensor(single_tensor, "dma_tensor")

        if self._res_tensor.op.tag != FAKE_NODE_TAG:
            __mark_group_axis_on_split_tensor(self._res_tensor)
