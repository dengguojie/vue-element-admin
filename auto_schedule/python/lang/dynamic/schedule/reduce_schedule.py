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
Reduce Schedule Remake stage 1
"""

# Standard Package
from typing import List
from typing import Union
from typing import Optional
from typing import NoReturn

# Third-party Packages
from te.tvm.tensor import Tensor
from te.lang.dynamic.schedule import Pattern
from .util import get_dsl_insn
from .util import is_reduce_tensor
from .util import get_reduce_axis_indices
from .util import is_keepdims
from .constants import INSN_MAPPING
from .constants import DTYPE_BYTE_MAPPING
from .vector_info import ComputeGraphInfo
from .vector_schedule import VectorSchedule
from .reduce_tilingcase import ReduceTilingCase
from .reduce_tilingcase import SingleReduceInfo
from .reduce_atomic_schedule import ReduceAtomicSchedule
from ...base.operation_impl import var
from ...base.operation_impl import get_context
from ...base.operation_impl import register_schedule

CONST = "const"


def schedule(outs, tiling_case: ReduceTilingCase):
    [outs].clear()
    # Get Compute Graph Info
    graph_info = get_context().get_current_compute().get("compute_graph_info")
    single_reduce_info: SingleReduceInfo = get_context().get_current_compute().get("single_reduce_info")
    if tiling_case.is_atomic:
        reduce_sch: ReduceAtomicSchedule = ReduceAtomicSchedule()
        reduce_sch.init(outs, [])
        if single_reduce_info.is_reduce_all_axes():
            reduce_sch._reduce_case = 1
        elif single_reduce_info.is_reduce_not_last_axis():
            reduce_sch._reduce_case = 2
        else:
            reduce_sch._reduce_case = 3
        real_schedule = reduce_sch.do_schedule(outs, tiling_case, graph_info, single_reduce_info)
        real_schedule.tiling_key = tiling_case.tiling_key
    else:
        reduce_sch: ReduceSchedule = ReduceSchedule(graph_info, single_reduce_info)
        real_schedule = reduce_sch.do_schedule(tiling_case)
        real_schedule.tiling_key = tiling_case.tiling_key
    return real_schedule


BLOCK_SIZE_BYTE = 32


class ReduceSchedule(VectorSchedule):
    def __init__(self, graph_info: ComputeGraphInfo, single_reduce_info: SingleReduceInfo):
        VectorSchedule.__init__(self, graph_info)
        self.reduce_info = single_reduce_info
        self.block_tiling_result_pair: Optional[List[VectorSchedule.Placeholder, VectorSchedule.Placeholder]] = None
        self.ub_tiling_result_pair: Optional[List[VectorSchedule.Placeholder, VectorSchedule.Placeholder]] = None
        self.block_tiling_info: Optional[VectorSchedule.TilingInfo] = None
        self.ub_tiling_info: Optional[VectorSchedule.TilingInfo] = None

    def _calc_reduced_axis_indices(self):
        last_input_tensor = None
        for input_tensor in self.graph_info.input_tensor_set:
            if last_input_tensor is None:
                last_input_tensor = input_tensor
            if len(last_input_tensor.shape) != len(input_tensor.shape):
                raise RuntimeError("SingleReduceSchedule doesn't support multiple input with different dim_num")
            self.tensor_reduced_axis_indices[input_tensor] = []
            # Check through all consumer
            waiting_tensors: List[Tensor] = [input_tensor]
            while waiting_tensors:
                current_tensor = waiting_tensors.pop()
                current_indices = self.tensor_reduced_axis_indices[current_tensor]
                for consumer in self.forward_compute_graph_map[current_tensor]:
                    if consumer in self.tensor_reduced_axis_indices:
                        continue
                    if is_reduce_tensor(current_tensor) and not is_keepdims(current_tensor):
                        self.tensor_reduced_axis_indices[consumer] = \
                            current_indices + get_reduce_axis_indices(current_tensor)
                    else:
                        self.tensor_reduced_axis_indices[consumer] = current_indices
                    waiting_tensors.append(consumer)

    def _calc_data_flow_control(self) -> NoReturn:
        for tensor in self.graph_info.mid_tensor_set:
            if tensor not in self.graph_info.output_tensor_set:
                self._tensor_to_scope_map[tensor] = "local.UB"
        self.do_auto_data_flow_control()

    def _calc_compute_inline(self) -> NoReturn:
        """
        No inline tensor
        """
        pass

    def _calc_storage_bound(self) -> NoReturn:
        for stage_tensor in self.forward_stage_graph_map:
            if self.forward_stage_graph_map[stage_tensor]:
                self.storage_bound_map[stage_tensor] = self.graph_info.max_single_tensor_ub_size

    def _calc_tiling(self):
        """
        res_tensor: real_tensor
        reduce_tensor: real_tensor
        reduce_buffeer: cache_write_buffer[reduce_tensor]
        return: init tiling_info
        """
        case: ReduceTilingCase = self.tiling_case
        if not isinstance(case, ReduceTilingCase):
            raise RuntimeError("ReduceTilingCase required for ReduceSchedule!")

        # Warning: single output supported only
        if len(self.graph_info.output_tensor_set) != 1:
            raise RuntimeError("SingleReduceSchedule supports single output only")

        # Get tiling tensor
        res_tensor = tuple(self.graph_info.output_tensor_set)[0]
        reduce_ub_buffer = self.get_buffers_of(self.reduce_info.reduce_tensor)[0]

        # Get Tiling axes
        block_split_axis_index = case.block_split_axis_index
        ub_split_axis_index = case.ub_split_axis_index

        # Get tiling params
        block_factor = case.block_factor
        block_inner = block_factor if block_factor is not None else var("block_factor", (1, None))
        ub_factor = case.ub_factor
        ub_inner = ub_factor if ub_factor is not None else var("ub_factor", (1, None))
        self._need_multi_core = case.multi_core

        # block tiling
        self.block_tiling_result_pair = list(self.split(res_tensor, block_split_axis_index, block_inner))
        self.block_tiling_info = self._tiling[-1]

        # ub tiling
        self.ub_tiling_result_pair = list(self.split(reduce_ub_buffer, ub_split_axis_index, ub_inner))
        self.ub_tiling_info = self._tiling[-1]

        # block fuse
        if case.multi_core is None:
            raise RuntimeError("Tilingcase didn't declare multi_core switch")
        if case.multi_core:
            block_tiling_result = self.block_tiling_info
            tensor = block_tiling_result.tiling_tensor
            block_split_axis_index = block_tiling_result.tiling_axis_index
            res_block_outer = self.block_tiling_result_pair[0]

            fuse_axis_list = [res_block_outer]
            fusable_axis_indices = [idx for idx in range(0, block_split_axis_index)
                                    if idx not in self.tensor_reduced_axis_indices[tensor]]
            fuse_axis_list.extend(fusable_axis_indices)

            self.multi_core_bind_tensor = \
                tuple(self.graph_info.output_tensor_set)[0]
            if len(fuse_axis_list) > 1:
                self.multi_core_bind_axis = self.fuse(tensor, fuse_axis_list)
            else:
                self.multi_core_bind_axis = fuse_axis_list[0]

    def _calc_reorder(self):
        is_nlast_reduce = self.reduce_info.is_reduce_not_last_axis()
        # Reduce axes must be placed together, then move last axis of non-reduce axes after reduce axes
        # For Reduce Tensor and Tensors before reduce
        all_axis_indices = list(range(len(self.reduce_info.all_axes)))
        reduce_axis_indices: List[Union[VectorSchedule.Placeholder, int]] = self.reduce_info.reduce_axis_indices[:]
        non_reduce_axis_indices = [axis_idx for axis_idx in all_axis_indices if axis_idx not in reduce_axis_indices]
        reduce_axis_indices_original = reduce_axis_indices[:]
        non_reduce_axis_indices_original = non_reduce_axis_indices[:]
        reduce_ub_stage_tensor = self.get_buffers_of(self.reduce_info.reduce_tensor)[0]
        # Find all split axis
        ub_tiling_axis_index = self.ub_tiling_info.tiling_axis_index
        if ub_tiling_axis_index in reduce_axis_indices:
            idx_of_ub_split_axis_index = reduce_axis_indices.index(ub_tiling_axis_index)
            reduce_axis_indices.remove(ub_tiling_axis_index)
            reduce_axis_indices.insert(idx_of_ub_split_axis_index, self.ub_tiling_result_pair[0])
            reduce_axis_indices.insert(idx_of_ub_split_axis_index + 1, self.ub_tiling_result_pair[1])
        elif ub_tiling_axis_index in non_reduce_axis_indices:
            idx_of_ub_split_axis_index = non_reduce_axis_indices.index(ub_tiling_axis_index)
            non_reduce_axis_indices.remove(ub_tiling_axis_index)
            non_reduce_axis_indices.insert(idx_of_ub_split_axis_index, self.ub_tiling_result_pair[0])
            non_reduce_axis_indices.insert(idx_of_ub_split_axis_index + 1, self.ub_tiling_result_pair[1])
        # Construct reorder target
        if is_nlast_reduce:
            reorder_target = [*non_reduce_axis_indices[:-1], *reduce_axis_indices, non_reduce_axis_indices[-1]]
        else:
            reorder_target = [*non_reduce_axis_indices, *reduce_axis_indices]
        # Add reorder target for reduce_ub_buffer
        self._tensor_to_reorder_map[reduce_ub_stage_tensor] = reorder_target
        all_producers = self.get_all_producer_stages(reduce_ub_stage_tensor)
        # Add reorder target for all tensors before reduce_ub_buffer
        if is_nlast_reduce:
            reorder_target_original = [*non_reduce_axis_indices_original[:-1],
                                       *reduce_axis_indices_original,
                                       non_reduce_axis_indices_original[-1]]
        else:
            reorder_target_original = [*non_reduce_axis_indices_original,
                                       *reduce_axis_indices_original]
        for producer in all_producers:
            if producer not in self.graph_info.input_tensor_set:
                self._tensor_to_reorder_map[producer] = reorder_target_original

    def _calc_double_buffer(self):
        pass

    def _calc_constraint(self):
        ub_tiling_info = self.ub_tiling_info
        ub_tiling_axis_idx = ub_tiling_info.tiling_axis_index
        ub_tiling_tensor = ub_tiling_info.tiling_tensor
        ub_tiling_factor = ub_tiling_info.factor
        max_ub_count = self.graph_info.max_single_tensor_ub_size
        params = (ub_tiling_tensor, list(self.graph_info.input_tensor_set)[0])

        def func(_ub_tiling_tensor: Tensor,
                 _before_reduce_compute_tensor: Tensor) -> list:
            results = []
            all_shape = self.reduce_info.shape_before_reduce
            reduce_source_tensor = _ub_tiling_tensor.op.input_tensors[0]
            reduce_source_tensor_ub = self.get_buffers_of(reduce_source_tensor)[0]
            all_itervars = self.schedule[reduce_source_tensor_ub].all_iter_vars[:len(all_shape)]
            all_reordered_itervars = self.schedule[reduce_source_tensor_ub].leaf_iter_vars[:len(all_shape)]
            reordered_shape_indices = [all_itervars.index(itervar) for itervar in all_reordered_itervars]
            # UB Factor smaller than max_ub_count
            results.append(ub_tiling_factor <= max_ub_count)
            ub_size_product = ub_tiling_factor
            # Each UB var smaller than max_ub_count
            reordered_ub_tiling_axis_idx = reordered_shape_indices.index(ub_tiling_axis_idx)
            for shape_index in reordered_shape_indices[reordered_ub_tiling_axis_idx + 1:]:
                ub_size_product *= all_shape[shape_index]
                results.append(all_shape[shape_index] <= max_ub_count)
            # Total UB Size smaller than max_ub_count
            results.append(ub_size_product <= max_ub_count)
            return results
        self.constraint_func_pair_list.append((params, func))

    def __need_storage_align(self):
        ub_split_axis_index = self.ub_tiling_info.tiling_axis_index
        shape_before_reduce = self.reduce_info.shape_before_reduce
        reduce_axis_indices = self.reduce_info.reduce_axis_indices
        # for shape(r4,a4,r3,a3,r2,a2,r1,a1), if ub split a1, do not need storage align
        if self.reduce_info.is_reduce_not_last_axis():
            a1_start_index, a1_end_index = \
                self.reduce_info.find_last_none_reduce_axis(shape_before_reduce,
                                                            reduce_axis_indices)
            if a1_end_index is None:
                return False
            if a1_start_index <= ub_split_axis_index <= a1_end_index:
                return False

        else:
            r1_start_index, r1_end_index = self.reduce_info.find_last_reduce_axis(
                shape_before_reduce,
                reduce_axis_indices)
            if r1_end_index is None:
                return False
            # for shape(a4,r4,a3,r3,a2,r2,a1,r1), if ub split r1, do not need storage align
            if r1_start_index <= ub_split_axis_index <= r1_end_index:
                return False
        return True

    def _calc_storage_align(self):
        if not self.__need_storage_align():
            return

        shape_before_reduce = self.reduce_info.shape_before_reduce
        reduce_axis_indices = self.reduce_info.reduce_axis_indices

        a1_start_index, a1_end_index = self.reduce_info.find_last_none_reduce_axis(shape_before_reduce,
                                                                                   reduce_axis_indices)
        if self.reduce_info.is_reduce_not_last_axis():
            # None-Last Reduce needs to storage align reduce_tensor and all_other tensors before reduce
            # Align at last reduce axis
            align_axis_index = a1_start_index - 1
            reduce_tensor = self.reduce_info.reduce_tensor
            tensors_before = self.get_all_producer_stages(reduce_tensor)
            tensors_after = self.get_all_consumer_stages(reduce_tensor)
            for tensor in tensors_before:
                if tensor not in self.graph_info.input_tensor_set:
                    storage_align_info = self.StorageAlignInfo(tensor,
                                                               align_axis_index,
                                                               BLOCK_SIZE_BYTE //
                                                               DTYPE_BYTE_MAPPING[tensor.dtype])
                    self.storage_align_list.append(storage_align_info)
            if a1_start_index - len(self.reduce_info.reduce_axis_indices) == 0:
                return
            align_axis_index -= 1
            if reduce_tensor in self._data_flow_control:
                # reduce tensor has performed cache_write()
                reduce_tensor = self.get_buffers_of(reduce_tensor)[0]
            reduce_storage_align_info = self.StorageAlignInfo(reduce_tensor,
                                                              align_axis_index,
                                                              BLOCK_SIZE_BYTE //
                                                              DTYPE_BYTE_MAPPING[reduce_tensor.dtype])
            self.storage_align_list.append(reduce_storage_align_info)
            for tensor in tensors_after:
                if tensor not in self.graph_info.output_tensor_set:
                    storage_align_info = self.StorageAlignInfo(tensor,
                                                               align_axis_index,
                                                               BLOCK_SIZE_BYTE //
                                                               DTYPE_BYTE_MAPPING[tensor.dtype])
                    self.storage_align_list.append(storage_align_info)
        else:
            reduce_tensor = self.reduce_info.reduce_tensor
            if reduce_tensor in self._data_flow_control:
                # reduce tensor has performed cache_write()
                reduce_tensor = self.get_buffers_of(reduce_tensor)[0]
            tensors_before = self.get_all_producer_stages(reduce_tensor)
            align_axis_index = a1_end_index
            for tensor in tensors_before:
                if tensor not in self.graph_info.input_tensor_set:
                    storage_align_info = self.StorageAlignInfo(tensor,
                                                               align_axis_index,
                                                               BLOCK_SIZE_BYTE //
                                                               DTYPE_BYTE_MAPPING[tensor.dtype])
                    self.storage_align_list.append(storage_align_info)

    def _calc_compute_at(self):
        """
        Calculate the tensor that needs compute at

        Parameters:
        -----------
        None

        Return
        ------
        None
        """
        # All tensors before reduce_ub_tensor
        reduce_ub_tiling_tensor = self.ub_tiling_info.tiling_tensor
        reduce_ub_split_axis_outer = self.ub_tiling_result_pair[0]

        if self.reduce_info.is_reduce_not_last_axis():
            a1_start_index, a1_end_index = self.reduce_info.find_last_none_reduce_axis(
                self.reduce_info.shape_before_reduce,
                self.reduce_info.reduce_axis_indices)
            if a1_start_index <= self.ub_tiling_info.tiling_axis_index <= a1_end_index:
                reduce_ub_split_axis_outer = self.reduce_info.reduce_axis_indices[-1]

        # Add anchor point
        self.anchor_point_list.append(reduce_ub_tiling_tensor)
        self.anchor_point_axis_index_list.append(reduce_ub_split_axis_outer)

        block_tiling_tensor = self.block_tiling_info.tiling_tensor
        res_block_outer = self.block_tiling_result_pair[0]

        # Add anchor point
        self.anchor_point_list.append(block_tiling_tensor)
        self.anchor_point_axis_index_list.append(res_block_outer)

    def _calc_emit_insn(self):
        reduce_ub_tensor = self.get_buffers_of(self.reduce_info.reduce_tensor)[0]
        # Reduce-ub
        emit_insn_axis = self.ub_tiling_result_pair[1]
        ub_split_axis_index = self.ub_tiling_info.tiling_axis_index
        if self.reduce_info.is_reduce_not_last_axis():
            a1_start_index, a1_end_index = self.reduce_info.find_last_none_reduce_axis(
                self.reduce_info.shape_before_reduce,
                self.reduce_info.reduce_axis_indices)
            if ub_split_axis_index < a1_start_index and ub_split_axis_index not in self.reduce_info.reduce_axis_indices:
                emit_insn_axis = self.reduce_info.reduce_axis_indices[0]
        elif ub_split_axis_index not in self.reduce_info.reduce_axis_indices:
            first_reduce_axis = self.reduce_info.reduce_tensor.op.reduce_axis[0]
            axis_dom = first_reduce_axis.dom
            if hasattr(axis_dom.min, "value") and hasattr(axis_dom.extent, "value"):
                if first_reduce_axis.dom.min.value == 0 and first_reduce_axis.dom.extent.value == 1:
                    emit_insn_axis = self.reduce_info.reduce_axis_indices[0]
        self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(reduce_ub_tensor, emit_insn_axis,
                                                               INSN_MAPPING[get_dsl_insn(reduce_ub_tensor)]))
        # Before-And-After-reduce
        before_reduce_tensors = self.get_all_producer_stages(reduce_ub_tensor)
        after_reduce_tensors = self.get_all_consumer_stages(reduce_ub_tensor)
        remaining_tensors = before_reduce_tensors | after_reduce_tensors
        for tensor in remaining_tensors:
            if tensor in self.graph_info.input_tensor_set:
                continue
            insn = get_dsl_insn(tensor)
            emit_insn_axis_index = 0
            if tensor in self.graph_info.output_tensor_set:
                insn = "dma_copy"
                emit_insn_axis_index = self.block_tiling_result_pair[1]
            if insn == "":
                insn = "dma_copy"
            self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(tensor, emit_insn_axis_index, INSN_MAPPING[insn]))
