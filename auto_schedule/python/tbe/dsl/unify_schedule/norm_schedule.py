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

from .constants import INSN_MAPPING
from .constants import NormPattern
from .constants import Pattern
from .norm_tilingcase import get_block_size as get_align_factor
from .norm_tilingcase import reorder_reduce_shape
from .schedule import Schedule


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
        norm_compute_graph_info = get_context().get_current_compute().get("_compute_graph_info")
        norm_info = get_context().get_current_compute().get("_norm_info")

        if self.tiling_case.ub_split_axis_index not in norm_info.reduce_axis_indices:
            norm_sch = NormNormalSchedule(norm_compute_graph_info, norm_info, self.tiling_case, self.outs)
        else:
            norm_sch = NormWorkspaceSchedule(norm_compute_graph_info, norm_info, self.tiling_case, self.outs)
        real_schedule = norm_sch.do_schedule()

        return real_schedule


class NormNormalSchedule:
    def __init__(self, graph_info, norm_info, tiling_case, outs):
        self._outs = outs
        self._sch = None
        self._scope = "local.UB"

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

        self._block_split_result = {}
        self._ub_split_result = {}

        self._reorder_map = {}
        self._multi_core_bind_axis = None
        self._storage_align_map = {}
        self._compute_at_map = {}
        self._emit_insn_map = {}

        self._is_nlast_block_split_last_a = False
        self._is_nlast_ub_split_last_a = False

    def do_schedule(self):
        # to ensure that the number of input parameters in the normal sch is the same as that in the workspace sch.
        fake_workspace_count = 0
        for _ in self._graph_info.workspace_tensor_set:
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

    def _do_compute_inline(self):
        for compute_inline_tensor in self._compute_inline_tensors:
            self._sch[compute_inline_tensor].compute_inline()

    def _do_tiling(self):
        # get tiling axis
        block_split_axis_index = self._tiling_case.block_split_axis_index
        ub_split_axis_index = self._tiling_case.ub_split_axis_index
        # get tiling params
        block_factor = self._tiling_case.block_factor
        block_split_factor = block_factor if block_factor is not None else var_inner("_block_factor", (1, None))
        ub_factor = self._tiling_case.ub_factor
        ub_split_factor = ub_factor if ub_factor is not None else var_inner("_ub_factor", (1, None))

        # block tiling
        block_outer, block_inner = self._sch[self._res_tensor].split(self._res_tensor.op.axis[block_split_axis_index],
                                                                     factor=block_split_factor)
        if block_split_axis_index != ub_split_axis_index:
            ub_outer, ub_inner = self._sch[self._res_tensor].split(self._res_tensor.op.axis[ub_split_axis_index],
                                                                   factor=ub_split_factor)
        else:
            ub_outer, ub_inner = self._sch[self._res_tensor].split(block_inner, factor=ub_split_factor)

        self._block_split_result["axis"] = block_split_axis_index
        self._block_split_result["outer_itervar"] = block_outer
        self._block_split_result["inner_itervar"] = block_inner
        self._block_split_result["factor"] = block_split_factor

        self._ub_split_result["axis"] = ub_split_axis_index
        self._ub_split_result["outer_itervar"] = ub_outer
        self._ub_split_result["inner_itervar"] = ub_inner
        self._ub_split_result["factor"] = ub_split_factor

        self._is_nlast_block_split_last_a = (not self._norm_info.is_reduce_last_axis and
                                             block_split_axis_index == len(self._norm_info.shape_before_reduce) - 1)
        self._is_nlast_ub_split_last_a = (not self._norm_info.is_reduce_last_axis and
                                          ub_split_axis_index == len(self._norm_info.shape_before_reduce) - 1)

    def _calc_reorder(self):
        def __calc_split_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            ori_blk_axis = self._tiling_case.block_split_axis_index
            ori_ub_axis = self._tiling_case.ub_split_axis_index
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reorder_first_r_index = len(self._norm_info.shape_before_reduce) - len(reduce_axis_index) -\
                                    (not is_reduce_last_axis)
            reduce_reorder_axis, _, ori_to_reorder_axis_map = reorder_reduce_shape(ori_axis,
                                                                                   reduce_axis_index,
                                                                                   is_reduce_last_axis)
            reorder_axis = []
            for idx, axis in enumerate(reduce_reorder_axis):
                if idx == ori_to_reorder_axis_map[ori_blk_axis] == ori_to_reorder_axis_map[ori_ub_axis]:
                    reorder_axis.append(self._block_split_result["outer_itervar"])
                    reorder_axis.append(self._ub_split_result["outer_itervar"])
                    reorder_axis.append(self._ub_split_result["inner_itervar"])
                elif idx == ori_to_reorder_axis_map[ori_blk_axis]:
                    reorder_axis.append(self._block_split_result["outer_itervar"])
                    reorder_axis.append(self._block_split_result["inner_itervar"])
                elif idx == ori_to_reorder_axis_map[ori_ub_axis]:
                    reorder_axis.append(self._ub_split_result["outer_itervar"])
                    reorder_axis.append(self._ub_split_result["inner_itervar"])
                else:
                    reorder_axis.append(axis)
            # reorder outer before reduce
            if self._is_nlast_block_split_last_a:
                reorder_axis.insert(reorder_first_r_index, self._ub_split_result["outer_itervar"])
                reorder_axis.insert(reorder_first_r_index, self._block_split_result["outer_itervar"])
                reorder_axis.pop(-2)
                reorder_axis.pop(-2)
            elif self._is_nlast_ub_split_last_a:
                reorder_axis.insert(reorder_first_r_index, self._ub_split_result["outer_itervar"])
                reorder_axis.pop(-2)

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

        def __calc_other_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis

            reduce_reorder_axis, _, _ = reorder_reduce_shape(ori_axis,
                                                             reduce_axis_index,
                                                             is_reduce_last_axis)

            return reduce_reorder_axis

        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors:
            if single_tensor in self._graph_info.reduce_tensor_set:
                reorder_axis_list = __calc_reduce_tensor_reorder_axis(single_tensor)
            else:
                reorder_axis_list = __calc_other_tensor_reorder_axis(single_tensor)
            self._reorder_map[single_tensor] = reorder_axis_list

        for single_tensor in self._cache_read_buffer_and_tensor_map:
            reorder_axis_list = __calc_other_tensor_reorder_axis(single_tensor)
            self._reorder_map[single_tensor] = reorder_axis_list

        for single_tensor in self._cache_write_buffer_and_tensor_map:
            reorder_axis_list = __calc_other_tensor_reorder_axis(single_tensor)
            self._reorder_map[single_tensor] = reorder_axis_list

        reorder_axis_list = __calc_split_tensor_reorder_axis(self._res_tensor)
        self._reorder_map[self._res_tensor] = reorder_axis_list

    def _do_reorder(self):
        for single_tensor, param in self._reorder_map.items():
            self._sch[single_tensor].reorder(*param)

    def _calc_storage_align(self):
        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors:
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
            block = tvm.thread_axis("blockIdx.x")
            self._sch[self._res_tensor].bind(self._multi_core_bind_axis, block)

    def _do_storage_bound(self):
        storage_bound_tensors = (self._graph_info.mid_tensor_set - self._compute_inline_tensors)\
            .union(self._cache_read_buffer_and_tensor_map.keys())\
            .union(self._cache_write_buffer_and_tensor_map.keys())

        for single_tensor in storage_bound_tensors:
            storage_bound = self._graph_info.available_ub_size
            self._sch[single_tensor].set_storage_bound(storage_bound)

    def _do_set_constraint(self):
        if get_context().get_current_compute().get("_mode") == "const":
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
        emit_insn_axis_index = self._tiling_case.ub_split_axis_index
        # TODO
        self._emit_insn_map[self._res_tensor] = [self._ub_split_result["inner_itervar"], "dma_copy"]
        if self._is_nlast_block_split_last_a:
            # first r index in reorder shape
            emit_insn_axis_index = len(self._norm_info.shape_before_reduce) - 1 - \
                                   len(self._norm_info.reduce_axis_indices)
            # copy ub to gm with stride, the last block need process the overlap too
            self._emit_insn_map[self._res_tensor].append({"no_overlap": 2})

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
                                                      {"storage_bound": self._graph_info.available_ub_size}]
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


class NormWorkspaceSchedule:
    def __init__(self, graph_info, norm_info, tiling_case, outs):
        self._outs = outs
        self._sch = None
        self._scope = "local.UB"

        self._graph_info = graph_info
        self._forward_compute_graph_map = graph_info.tensor_consumers_map
        self._backward_compute_graph_map = graph_info.tensor_producers_map
        self._workspace_tensor_set = graph_info.workspace_tensor_set
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

        self._block_split_result = {}
        self._ub_split_result = {}

        self._reorder_map = {}
        self._multi_core_bind_axis_map = {}
        self._storage_align_map = {}
        self._bind_buffer_map = {}
        self._compute_at_map = {}
        self._emit_insn_map = {}

        self._is_nlast_block_split_last_a = False

    def do_schedule(self):
        real_output_tensor_op_list = [tensor.op for tensor in self._graph_info.endpoint_output_tensor_set]
        for workspace_tensor in self._workspace_tensor_set:
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

        self._calc_emit_insn()
        self._do_emit_insn()

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
                    local_consumers = list(sub_tensor_consumers_map[cache_clone_tensor]).copy()
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
                    local_consumers = list(sub_tensor_consumers_map[cache_read_tensor]).copy()
                    for idx, tensor in enumerate(local_consumers):
                        # the readers have cache clone tensor
                        if tensor in self._cache_clone_tensor_and_buffer_map and \
                                sub_graph_split_tensor == \
                                    list(self._cache_clone_tensor_and_buffer_map[tensor].values())[0]:
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
                    local_consumers = list(sub_tensor_consumers_map[workspace_tensor]).copy()
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
                if tuple(self._backward_compute_graph_map[broadcast_tensor])[0] not in self._workspace_tensor_set:
                    self._compute_inline_tensors.add(broadcast_tensor)

    def _do_compute_inline(self):
        for compute_inline_tensor in self._compute_inline_tensors:
            self._sch[compute_inline_tensor].compute_inline()

    def _do_tiling(self):
        # get tiling axis
        block_split_axis_index = self._tiling_case.block_split_axis_index
        ub_split_axis_index = self._tiling_case.ub_split_axis_index
        # get tiling params
        block_factor = self._tiling_case.block_factor
        block_split_factor = block_factor if block_factor is not None else var_inner("_block_factor", (1, None))
        ub_factor = self._tiling_case.ub_factor
        ub_split_factor = ub_factor if ub_factor is not None else var_inner("_ub_factor", (1, None))
        # block tiling on A, ub tiling on R
        # non_reduce workspace node, block and ub split on workspace tensor
        # reduce workspace node, block split on workspace tensor and ub split on workspace ub tensor
        for workspace_tensor in self._workspace_tensor_set:
            workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
            if workspace_tensor in self._graph_info.reduce_tensor_set:
                ub_split_reduce_axis_index = sorted(self._norm_info.reduce_axis_indices).index(
                    ub_split_axis_index)
                ub_outer, ub_inner = self._sch[workspace_ub_tensor].split(
                    workspace_ub_tensor.op.reduce_axis[ub_split_reduce_axis_index], factor=ub_split_factor)
                self._ub_split_result[workspace_ub_tensor] = {
                    "axis": ub_split_axis_index,
                    "reduce_axis": ub_split_reduce_axis_index,
                    "outer_itervar": ub_outer,
                    "inner_itervar": ub_inner,
                    "factor": ub_split_factor
                }
            else:
                ub_outer, ub_inner = self._sch[workspace_tensor].split(
                    workspace_tensor.op.axis[ub_split_axis_index], factor=ub_split_factor)
                self._ub_split_result[workspace_tensor] = {
                    "axis": ub_split_axis_index,
                    "outer_itervar": ub_outer,
                    "inner_itervar": ub_inner,
                    "factor": ub_split_factor
                }
            block_outer, block_inner = self._sch[workspace_tensor].split(
                workspace_tensor.op.axis[block_split_axis_index], factor=block_split_factor)
            self._block_split_result[workspace_tensor] = {
                "axis": block_split_axis_index,
                "outer_itervar": block_outer,
                "inner_itervar": block_inner,
                "factor": block_split_factor
            }

        block_outer, block_inner = self._sch[self._res_tensor].split(self._res_tensor.op.axis[block_split_axis_index],
                                                                     factor=block_split_factor)
        ub_outer, ub_inner = self._sch[self._res_tensor].split(self._res_tensor.op.axis[ub_split_axis_index],
                                                               factor=ub_split_factor)

        self._block_split_result[self._res_tensor] = {
            "axis": block_split_axis_index,
            "outer_itervar": block_outer,
            "inner_itervar": block_inner,
            "factor": block_split_factor
        }
        self._ub_split_result[self._res_tensor] = {
            "axis": ub_split_axis_index,
            "outer_itervar": ub_outer,
            "inner_itervar": ub_inner,
            "factor": ub_split_factor
        }

        self._is_nlast_block_split_last_a = (not self._norm_info.is_reduce_last_axis and
                                             block_split_axis_index == len(self._norm_info.shape_before_reduce) - 1)

    def _calc_reorder(self):
        def __calc_split_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            ori_blk_axis = self._tiling_case.block_split_axis_index
            ori_ub_axis = self._tiling_case.ub_split_axis_index
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reorder_first_r_index = len(self._norm_info.shape_before_reduce) - len(reduce_axis_index) -\
                                    (not is_reduce_last_axis)
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
            # reorder outer before reduce
            if self._is_nlast_block_split_last_a and tensor in self._block_split_result:
                reorder_axis.insert(reorder_first_r_index, self._block_split_result[tensor]["outer_itervar"])
                reorder_axis.pop(-2)

            return reorder_axis

        def __calc_reduce_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis
            reorder_first_r_index = len(self._norm_info.shape_before_reduce) - len(reduce_axis_index) -\
                                    (not is_reduce_last_axis)
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
            # reorder outer before reduce
            if self._is_nlast_block_split_last_a and tensor in self._block_split_result:
                reorder_axis.insert(reorder_first_r_index, self._block_split_result[tensor]["outer_itervar"])
                reorder_axis.pop(-2)

            return reorder_axis

        def __calc_other_tensor_reorder_axis(tensor):
            ori_axis = tensor.op.axis
            reduce_axis_index = self._norm_info.reduce_axis_indices
            is_reduce_last_axis = self._norm_info.is_reduce_last_axis

            reduce_reorder_axis, _, _ = reorder_reduce_shape(ori_axis,
                                                             reduce_axis_index,
                                                             is_reduce_last_axis)

            return reduce_reorder_axis

        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors -\
                             self._workspace_tensor_set:
            reorder_axis_list = __calc_other_tensor_reorder_axis(single_tensor)
            self._reorder_map[single_tensor] = reorder_axis_list

        for single_tensor in self._cache_clone_buffer_and_tensor_map:
            reorder_axis_list = __calc_other_tensor_reorder_axis(single_tensor)
            self._reorder_map[single_tensor] = reorder_axis_list

        for single_tensor in self._cache_read_buffer_and_tensor_map:
            reorder_axis_list = __calc_other_tensor_reorder_axis(single_tensor)
            self._reorder_map[single_tensor] = reorder_axis_list

        for single_tensor in self._cache_write_buffer_and_tensor_map:
            reorder_axis_list = __calc_other_tensor_reorder_axis(single_tensor)
            self._reorder_map[single_tensor] = reorder_axis_list

        for workspace_tensor in self._workspace_tensor_set:
            workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
            reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
            if workspace_tensor in self._graph_info.reduce_tensor_set:
                reorder_axis_list = __calc_split_tensor_reorder_axis(workspace_tensor)
                self._reorder_map[workspace_tensor] = reorder_axis_list
                reorder_axis_list = __calc_reduce_tensor_reorder_axis(workspace_ub_tensor)
                self._reorder_map[workspace_ub_tensor] = reorder_axis_list
                # the reread_ub_tensor of workspace tensor is a normal tensor
                for single_tensor in reread_workspace_ub_tensor_map.keys():
                    reorder_axis_list = __calc_other_tensor_reorder_axis(single_tensor)
                    self._reorder_map[single_tensor] = reorder_axis_list
            else:
                reorder_axis_list = __calc_split_tensor_reorder_axis(workspace_tensor)
                self._reorder_map[workspace_tensor] = reorder_axis_list
                reorder_axis_list = __calc_other_tensor_reorder_axis(workspace_ub_tensor)
                self._reorder_map[workspace_ub_tensor] = reorder_axis_list
                # the reread_ub_tensor of workspace tensor is a normal tensor
                for single_tensor in reread_workspace_ub_tensor_map.keys():
                    reorder_axis_list = __calc_other_tensor_reorder_axis(single_tensor)
                    self._reorder_map[single_tensor] = reorder_axis_list

        reorder_axis_list = __calc_split_tensor_reorder_axis(self._res_tensor)
        self._reorder_map[self._res_tensor] = reorder_axis_list

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
                bind_factor = tvm.div((workspace_tensor.shape[bind_axis + 1] + align_factor - 1), align_factor) *\
                              align_factor
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
            block = tvm.thread_axis("blockIdx.x")
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
            storage_bound = self._graph_info.workspace_available_min_ub_size
            self._sch[single_tensor].set_storage_bound(storage_bound)

    def _do_set_constraint(self):
        if get_context().get_current_compute().get("_mode") == "const":
            return

        ori_shape = self._res_tensor.shape
        blk_split_inner = self._block_split_result[self._res_tensor]["factor"]
        ub_split_inner = self._ub_split_result[self._res_tensor]["factor"]
        ori_ub_axis = self._tiling_case.ub_split_axis_index
        reduce_axis_index = self._norm_info.reduce_axis_indices
        is_reduce_last_axis = self._norm_info.is_reduce_last_axis
        reduce_reorder_shape, _, ori_to_reorder_axis_map = reorder_reduce_shape(ori_shape,
                                                                                reduce_axis_index,
                                                                                is_reduce_last_axis)
        reorder_ub_axis = ori_to_reorder_axis_map[ori_ub_axis]
        shape_in_ub = ub_split_inner
        for i in range(reorder_ub_axis + 1, len(reduce_reorder_shape)):
            shape_in_ub *= reduce_reorder_shape[i]
        if self._is_nlast_block_split_last_a:
            shape_in_ub = shape_in_ub // reduce_reorder_shape[-1] * blk_split_inner

        self._sch.set_constraint(shape_in_ub <= self._graph_info.workspace_available_min_ub_size)

    def _calc_compute_at(self):
        def __get_compute_at_workspace_ub_tensor(ori_compute_at_tensor):
            # reduce workspace tensor compute at to the workspace_ub_tensor
            if ori_compute_at_tensor not in self._ub_split_result and ori_compute_at_tensor in self._workspace_map:
                return self._workspace_map[ori_compute_at_tensor]["ub_tensor"]
            else:
                return ori_compute_at_tensor

        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors -\
                             self._workspace_tensor_set:
            for sub_graph_split_tensor in self._split_tensor_and_sub_graph_map:
                sub_tensor_list = self._split_tensor_and_sub_graph_map[sub_graph_split_tensor]["sub_tensor_list"]
                if single_tensor in sub_tensor_list:
                    compute_at_tensor = __get_compute_at_workspace_ub_tensor(sub_graph_split_tensor)
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
                else:
                    self._compute_at_map[workspace_ub_tensor] =\
                        [workspace_tensor, self._block_split_result[workspace_tensor]["outer_itervar"]]

    def _do_compute_at(self):
        for single_tensor, param in self._compute_at_map.items():
            self._sch[single_tensor].compute_at(self._sch[param[0]], param[1])

    def _calc_emit_insn(self):
        ub_split_axis = self._tiling_case.ub_split_axis_index

        for source, _ in self._cache_clone_buffer_and_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[ub_split_axis], _get_insn(source)]

        for source, _ in self._cache_read_buffer_and_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[ub_split_axis], "dma_copy"]

        for single_tensor in self._graph_info.mid_tensor_set - self._compute_inline_tensors -\
                             self._workspace_tensor_set:
            self._emit_insn_map[single_tensor] = [single_tensor.op.axis[ub_split_axis], _get_insn(single_tensor)]

        for workspace_tensor in self._workspace_tensor_set:
            workspace_ub_tensor = self._workspace_map[workspace_tensor]["ub_tensor"]
            reread_workspace_ub_tensor_map = self._workspace_map[workspace_tensor]["reread_ub_tensor"]
            # non_reduce workspace node, block and ub split on workspace tensor
            if workspace_tensor not in self._graph_info.reduce_tensor_set:
                # TODO
                if self._is_nlast_block_split_last_a:
                    self._emit_insn_map[workspace_tensor] =\
                        [self._block_split_result[workspace_tensor]["inner_itervar"],
                         "dma_copy",
                         {"no_overlap": 2}]
                else:
                    self._emit_insn_map[workspace_tensor] =\
                        [self._ub_split_result[workspace_tensor]["inner_itervar"], "dma_copy"]
                # src and dst are align
                is_align = (not self._norm_info.is_reduce_last_axis or
                            workspace_tensor in self._graph_info.tensors_before_reduce) and \
                           self._tiling_case.block_split_axis_index != len(self._norm_info.shape_before_reduce) - 1
                if is_align:
                    self._emit_insn_map[workspace_tensor].append({"no_overlap": 0})

                self._emit_insn_map[workspace_ub_tensor] =\
                    [workspace_ub_tensor.op.axis[ub_split_axis], _get_insn(workspace_tensor)]
            # reduce workspace node, block split on workspace tensor and ub split on workspace ub tensor
            else:
                self._emit_insn_map[workspace_tensor] =\
                    [self._block_split_result[workspace_tensor]["inner_itervar"], "dma_copy"]
                # src and dst are align
                is_align = (not self._norm_info.is_reduce_last_axis or
                            workspace_tensor in self._graph_info.tensors_before_reduce) and \
                           self._tiling_case.block_split_axis_index != len(self._norm_info.shape_before_reduce) - 1
                if is_align:
                    self._emit_insn_map[workspace_tensor].append({"no_overlap": 0})

                self._emit_insn_map[workspace_ub_tensor] =\
                    [self._ub_split_result[workspace_ub_tensor]["inner_itervar"], _get_insn(workspace_tensor),
                     {"storage_bound": self._graph_info.workspace_available_min_ub_size}]

            for reread_workspace_ub_tensor in reread_workspace_ub_tensor_map.keys():
                self._emit_insn_map[reread_workspace_ub_tensor] =\
                    [reread_workspace_ub_tensor.op.axis[ub_split_axis], "dma_copy"]

        for source, target_map in self._cache_write_buffer_and_tensor_map.items():
            target = list(target_map.values())[0]
            self._emit_insn_map[source] = [source.op.axis[ub_split_axis], _get_insn(target)]

        # TODO
        if self._is_nlast_block_split_last_a:
            self._emit_insn_map[self._res_tensor] =\
                [self._block_split_result[self._res_tensor]["inner_itervar"], "dma_copy", {"no_overlap": 2}]
        else:
            self._emit_insn_map[self._res_tensor] =\
                [self._ub_split_result[self._res_tensor]["inner_itervar"], "dma_copy"]

    def _do_emit_insn(self):
        for single_tensor, param in self._emit_insn_map.items():
            if len(param) > 2:
                self._sch[single_tensor].emit_insn(param[0], param[1], attrs=param[2])
            else:
                self._sch[single_tensor].emit_insn(param[0], param[1])
