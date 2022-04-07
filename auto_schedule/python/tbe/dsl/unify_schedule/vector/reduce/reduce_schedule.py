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
Reduce Schedule Remake stage 1
"""

# Standard Package
import copy
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Union

from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import var_inner
from tbe.tvm.tensor import Tensor
from tbe.common.platform import ASCEND_920A
from tbe.common.platform import SOC_VERSION
from tbe.common.platform.platform_info import get_soc_spec

from ...constants import DTYPE_BYTE_MAPPING
from ...constants import INSN_MAPPING
from ...constants import ReducePattern
from ...constants import Pattern
from ...constants import ReduceCategory
from .reduce_atomic_schedule import ReduceAtomicSchedule
from .reduce_tilingcase import Dim
from .reduce_tilingcase import R
from .reduce_tilingcase import A
from .reduce_tilingcase import ReduceTilingCase
from .reduce_tilingcase import SingleReduceInfo
from ...util import get_dsl_insn
from ...util import get_reduce_axis_indexes
from ...util import is_keepdims
from ...util import is_reduce_tensor
from .vector_info import ComputeGraphInfo
from .vector_schedule import VectorSchedule
from .empty_schedule import EmptySchedule
from ...schedule import Schedule

CONST = "const"
DEFAULT = "default"
INT32_MAX = 2 ** 31 - 1
BLOCK_SIZE_BYTE = 32


class EntryReduceSchedule(Schedule):
    """
    Entry class for All Reduce Schedule
    """

    def __init__(self, outs, tiling_case):
        self.outs = outs
        self.tiling_case: ReduceTilingCase = tiling_case

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.REDUCE]

    @classmethod
    def get_supported_sub_pattern(cls):
        return [ReducePattern.R_0]

    def do_schedule(self):
        """
        Entry method of reduce schedule
        """
        outs, tiling_case = self.outs, self.tiling_case
        # Get Compute Graph Info
        graph_info = get_context().get_current_compute().get("_compute_graph_info")
        single_reduce_info: SingleReduceInfo = get_context().get_current_compute().get("_single_reduce_info")
        if tiling_case.type == tiling_case.Type.EMPTY:
            reduce_sch: EmptySchedule = EmptySchedule(graph_info)
            real_schedule = reduce_sch.do_schedule(tiling_case)
        elif tiling_case.type == tiling_case.Type.ATOMIC_REDUCE:
            reduce_sch: ReduceAtomicSchedule = ReduceAtomicSchedule(graph_info, single_reduce_info)
            if single_reduce_info.is_reduce_all_axes():
                reduce_sch._reduce_case = ReduceCategory.ALL_REDUCE
            elif single_reduce_info.is_reduce_not_last_axis():
                reduce_sch._reduce_case = ReduceCategory.NOT_LAST_REDUCE
            else:
                reduce_sch._reduce_case = ReduceCategory.LAST_REDUCE
            real_schedule = reduce_sch.do_schedule(outs, tiling_case)
        elif tiling_case.type == tiling_case.Type.NORMAL_REDUCE:
            reduce_sch: ReduceSchedule = ReduceSchedule(graph_info, single_reduce_info)
            real_schedule = reduce_sch.do_schedule(tiling_case)
        else:
            raise NotImplementedError("Reduce schedule received invalid type: %s" % str(tiling_case.type))

        real_schedule.tiling_key = tiling_case.tiling_key
        return real_schedule


class ReduceSchedule(VectorSchedule):
    """
    Schedule for normal reduce
    """

    def __init__(self, graph_info: ComputeGraphInfo, single_reduce_info: SingleReduceInfo):
        VectorSchedule.__init__(self, graph_info)
        self.reduce_info = single_reduce_info
        self.block_tiling_result_pair: Optional[List[VectorSchedule.Placeholder, VectorSchedule.Placeholder]] = None
        self.ub_tiling_result_pair: Optional[List[VectorSchedule.Placeholder, VectorSchedule.Placeholder]] = None
        self.block_tiling_info: Optional[VectorSchedule.TilingInfo] = None
        self.ub_tiling_info: Optional[VectorSchedule.TilingInfo] = None

        self.reduce_rf = None
        self.rf_tiling_result_pair = None
        self._serial_group = None

    def _calc_reduced_axis_indexes(self):
        last_input_tensor = None
        for input_tensor in self.graph_info.input_tensor_set:
            if last_input_tensor is None:
                last_input_tensor = input_tensor
            self.tensor_reduced_axis_indexes[input_tensor] = []
            # Check through all consumer
            waiting_tensors: List[Tensor] = [input_tensor]
            while waiting_tensors:
                current_tensor = waiting_tensors.pop()
                current_indexes = self.tensor_reduced_axis_indexes[current_tensor]
                for consumer in self.forward_compute_graph_map[current_tensor]:
                    if consumer in self.tensor_reduced_axis_indexes:
                        continue
                    if is_reduce_tensor(current_tensor) and not is_keepdims(current_tensor):
                        self.tensor_reduced_axis_indexes[consumer] = \
                            current_indexes + get_reduce_axis_indexes(current_tensor)
                    else:
                        self.tensor_reduced_axis_indexes[consumer] = current_indexes
                    waiting_tensors.append(consumer)

    def _calc_data_flow_control(self) -> NoReturn:
        for tensor in self.graph_info.mid_tensor_set:
            if tensor not in self.graph_info.real_output_tensor_set:
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
                if stage_tensor in self.graph_info.real_output_tensor_set:
                    # don't set bound for real_output_tensors(gm)
                    continue
                if stage_tensor in self.graph_info.input_tensor_set:
                    # don't set bound for input_tensor_set(gm)
                    continue

                if stage_tensor in self.graph_info.tensors_after_reduce:
                    ub_count = self.tiling_case.tensor_ub_size_after_reduce
                elif stage_tensor in self.graph_info.tensors_before_reduce:
                    ub_count = self.tiling_case.tensor_ub_size_before_reduce
                else:
                    # some tensors are Placeholders
                    if isinstance(stage_tensor, VectorSchedule.Placeholder):
                        if stage_tensor is self.reduce_rf:
                            ub_count = self.tiling_case.tensor_ub_size_before_reduce
                        elif stage_tensor.key[0] in self.graph_info.tensors_after_reduce:
                            ub_count = self.tiling_case.tensor_ub_size_after_reduce
                        elif stage_tensor.key[0] in self.graph_info.tensors_before_reduce:
                            ub_count = self.tiling_case.tensor_ub_size_before_reduce
                        else:
                            raise RuntimeError("undefined tensor")
                    else:
                        raise RuntimeError("undefined tensor")
                self.storage_bound_map[stage_tensor] = ub_count

    def _calc_tiling(self):
        """
        res_tensor: real_tensor
        reduce_tensor: real_tensor
        reduce_buffeer: cache_write_buffer[reduce_tensor]
        return: init tiling_info
        """
        # noinspection PyTypeChecker
        case: ReduceTilingCase = self.tiling_case
        if not isinstance(case, ReduceTilingCase):
            raise RuntimeError("ReduceTilingCase required for ReduceSchedule!")

        # Get tiling tensor
        res_tensor = tuple(self.graph_info.output_tensor_set)[0]
        reduce_ub_buffer = self.get_buffers_of(self.reduce_info.reduce_tensor)[0]

        # Get Tiling axes
        block_split_axis_index = case.block_split_axis_index
        ub_split_axis_index = case.ub_split_axis_index

        # Get tiling params
        block_factor = case.block_factor
        block_inner = block_factor if block_factor is not None else var_inner("_block_factor", (1, None))
        ub_factor = case.ub_factor
        ub_inner = ub_factor if ub_factor is not None else var_inner("_ub_factor", (1, None))
        self._need_multi_core = case.multi_core

        def _base_tiling():
            # block tiling
            nonlocal block_split_axis_index
            self.block_tiling_result_pair = list(self.split(res_tensor, block_split_axis_index, block_inner))
            self.block_tiling_info = self._tiling[-1]

            # ub tiling
            self.ub_tiling_result_pair = list(self.split(reduce_ub_buffer, ub_split_axis_index, ub_inner))
            self.ub_tiling_info = self._tiling[-1]

        def _rf_tiling():
            # block_tiling
            nonlocal block_split_axis_index
            self.block_tiling_result_pair = list(self.split(res_tensor, block_split_axis_index, block_inner))
            self.block_tiling_info = self._tiling[-1]

            # rfactor
            if ub_split_axis_index == self.reduce_info.reduce_axis_indexes[-1]:
                # part of last_dim as rfi for rfactor, and [ub_inner, ub_outer] = [rfi, rfo]
                # tiling_rf_pair[0] set as ub_outer if target is reduce_rf
                # tiling_rf_pair[1] set as ub_inner if target is reduce_rf
                tiling_rf_pair = list(self.split(reduce_ub_buffer, ub_split_axis_index, ub_inner))
                self.reduce_rf = self.rfactor(reduce_ub_buffer, tiling_rf_pair[1], -1, scope="local.UB")
                # find rfi
                rf_ub_inner_idx = self.Placeholder(
                    self.Placeholder.PlaceholderType.ITER_VAR, (
                        self.reduce_rf, len(self.reduce_info.reduce_tensor.shape))
                )
                # find rfo
                rf_ub_outer_idx = self.Placeholder(
                    self.Placeholder.PlaceholderType.ITER_VAR, (
                        self.reduce_rf, -1)
                )
                self.ub_tiling_result_pair = [rf_ub_outer_idx, rf_ub_inner_idx]
                self.ub_tiling_info = copy.copy(self._tiling[-2])
                self.ub_tiling_info.factor = self._tiling[-2].factor
                self.ub_tiling_info.tiling_tensor = self.reduce_rf
            else:
                self.reduce_rf = self.rfactor(reduce_ub_buffer, len(self.reduce_info.shape_before_reduce) - 1,
                                              -1, scope="local.UB")
                # ub tiling
                _rf_offset = self.reduce_info.reduce_axis_indexes.index(ub_split_axis_index)
                rf_ub_split_idx = self.Placeholder(
                    self.Placeholder.PlaceholderType.ITER_VAR, (
                        self.reduce_rf, len(self.reduce_info.reduce_tensor.shape) + 1 + _rf_offset)
                )
                self.ub_tiling_result_pair = list(self.split(self.reduce_rf, rf_ub_split_idx, ub_inner))
                self.ub_tiling_info = self._tiling[-1]

        if self._last_reduction_rf_optimization():
            _rf_tiling()
        else:
            _base_tiling()

        # block fuse
        if case.multi_core is None:
            raise RuntimeError("Tilingcase didn't declare multi_core switch")
        if case.multi_core:
            block_tiling_result = self.block_tiling_info
            tensor = block_tiling_result.tiling_tensor
            block_split_axis_index = block_tiling_result.tiling_axis_index
            res_block_outer = self.block_tiling_result_pair[0]

            fuse_axis_list = [res_block_outer]
            fusable_axis_indexes = \
                [idx for idx in range(0, block_split_axis_index) if idx not in self.tensor_reduced_axis_indexes[tensor]]
            fuse_axis_list.extend(fusable_axis_indexes)

            self.multi_core_bind_tensor = \
                tuple(self.graph_info.output_tensor_set)[0]
            if len(fuse_axis_list) > 1:
                self.multi_core_bind_axis = self.fuse(tensor, fuse_axis_list)
            else:
                self.multi_core_bind_axis = fuse_axis_list[0]

    def _calc_reorder(self):
        """
        calc how to reorder
        """
        # For zero compute mode, skip reorder stage
        compute = get_context().get_current_compute()
        if compute.get("_mode") == "zero":
            return

        is_nlast_reduce = self.reduce_info.is_reduce_not_last_axis()
        # Reduce axes must be placed together, then move last axis of non-reduce axes after reduce axes
        # For Reduce Tensor and Tensors before reduce
        all_axis_indexes = list(range(len(self.reduce_info.all_axes)))
        reduce_axis_indexes: List[Union[VectorSchedule.Placeholder, int]] = self.reduce_info.reduce_axis_indexes[:]
        non_reduce_axis_indexes = [axis_idx for axis_idx in all_axis_indexes if axis_idx not in reduce_axis_indexes]
        reduce_axis_indexes_original = reduce_axis_indexes[:]
        non_reduce_axis_indexes_original = non_reduce_axis_indexes[:]
        reduce_ub_stage_tensor = self.get_buffers_of(self.reduce_info.reduce_tensor)[0]
        # Find all split axis
        ub_tiling_axis_index = self.tiling_case.ub_split_axis_index
        if ub_tiling_axis_index in reduce_axis_indexes:
            idx_of_ub_split_axis_index = reduce_axis_indexes.index(ub_tiling_axis_index)
            reduce_axis_indexes.remove(ub_tiling_axis_index)
            reduce_axis_indexes.insert(idx_of_ub_split_axis_index, self.ub_tiling_result_pair[0])
            reduce_axis_indexes.insert(idx_of_ub_split_axis_index + 1, self.ub_tiling_result_pair[1])
        elif ub_tiling_axis_index in non_reduce_axis_indexes:
            idx_of_ub_split_axis_index = non_reduce_axis_indexes.index(ub_tiling_axis_index)
            non_reduce_axis_indexes.remove(ub_tiling_axis_index)
            non_reduce_axis_indexes.insert(idx_of_ub_split_axis_index, self.ub_tiling_result_pair[0])
            non_reduce_axis_indexes.insert(idx_of_ub_split_axis_index + 1, self.ub_tiling_result_pair[1])

        # Construct reorder target
        if is_nlast_reduce:
            reorder_target = [*non_reduce_axis_indexes[:-1], *reduce_axis_indexes, non_reduce_axis_indexes[-1]]
        else:
            if self._last_reduction_rf_optimization():
                # Only reduce_rf needs special reorder,
                # and reordering is no longer required for reduce.
                reduce_ub_stage_tensor = self.reduce_rf
                # [Remove] last_dim for rfactor
                if reduce_axis_indexes_original[-1] in reduce_axis_indexes:
                    reduce_axis_indexes.remove(reduce_axis_indexes_original[-1])

                # [Replace] rf = sch.rfactor(reduce, rfi, -1), "-1" decides position of rfi is
                # before reduce_axis and after non_reduce_axis in sch[rf].all_iter_vars.
                # Replace all non split reduce_axis.
                for key, reduce_idx in enumerate(reduce_axis_indexes):
                    if isinstance(reduce_idx, int):
                        _index = reduce_axis_indexes_original.index(reduce_idx) + \
                                 len(self.reduce_info.reduce_tensor.shape) + 1
                        reduce_axis_indexes[key] = self.Placeholder(
                            self.Placeholder.PlaceholderType.ITER_VAR, (self.reduce_rf, _index))

                # [Add] add rfi while rf = sch.rfactor(reduce, rfi, -1)
                if self.tiling_case.ub_split_axis_index != reduce_axis_indexes_original[-1]:
                    reduce_axis_indexes.append(
                        self.Placeholder(self.Placeholder.PlaceholderType.ITER_VAR,
                                         (self.reduce_rf, len(self.reduce_info.reduce_tensor.shape))))

                reorder_target = [*non_reduce_axis_indexes, *reduce_axis_indexes]
            else:
                reorder_target = [*non_reduce_axis_indexes, *reduce_axis_indexes]

        # Add reorder target for reduce_ub_buffer
        self._tensor_to_reorder_map[reduce_ub_stage_tensor] = reorder_target

    def _calc_double_buffer(self):
        pass

    def _calc_constraint(self):
        ub_info, blk_info = self.ub_tiling_info, self.block_tiling_info
        ub_tensor, blk_tensor = ub_info.tiling_tensor, blk_info.tiling_tensor
        ub_factor, blk_factor = ub_info.factor, blk_info.factor
        ub_idx, blk_idx = self.tiling_case.ub_split_axis_index, blk_info.tiling_axis_index
        params = (ub_tensor,)

        def func(_ub_tiling_tensor: Tensor, ) -> list:
            # shape after reorder must follow:
            # [R,A],[A,R],[A,R,A],[A,R,A,R],...,[A,R,...,A,R]
            in_shape = self.reduce_info.shape_before_reduce
            reduce_indexes = self.reduce_info.reduce_axis_indexes
            i_length, r_length = len(in_shape), len(reduce_indexes)

            def _reorder(_r_shape, _r_indexes, _r_ou_shape):
                # r_shape: in_shape after reorder
                # r_indexes: r_shape[x] == in_shape[r_indexes[x]]
                pos_a = 0
                if reduce_indexes[-1] == i_length - 1:
                    pos_r = i_length - r_length  # last dim is R
                else:
                    pos_r = i_length - r_length - 1  # last dim is A
                for idx, value in enumerate(in_shape):
                    if idx == i_length - 1 and idx not in reduce_indexes:
                        _r_shape[-1], _r_indexes[-1] = value, idx
                        _r_ou_shape[-1] = _r_shape[-1]
                    elif idx in reduce_indexes:
                        _r_shape[pos_r], _r_indexes[pos_r] = value, idx
                        _r_ou_shape[pos_r] = 1
                        pos_r += 1
                    else:
                        _r_shape[pos_a], _r_indexes[pos_a] = value, idx
                        _r_ou_shape[pos_a] = value
                        pos_a += 1

            r_in_shape = [0] * i_length
            r_indexes = [0] * i_length
            r_ou_shape = [0] * i_length
            results = []
            _reorder(r_in_shape, r_indexes, r_ou_shape)

            # do constraint for blk_split and output
            r_blk_idx = r_indexes.index(blk_idx)
            max_ub_count = self.tiling_case.tensor_ub_size_after_reduce
            output_size = blk_factor
            for _dim in r_ou_shape[r_blk_idx + 1:]:
                output_size *= _dim
                if not isinstance(_dim <= max_ub_count, bool):
                    results.append(_dim <= max_ub_count)
            results.append(output_size <= max_ub_count)
            r_in_shape[r_blk_idx] = blk_factor  # for constraint(ub)

            # do constraint for ub_split and input
            if ub_idx not in reduce_indexes and ub_idx < blk_idx:
                raise RuntimeError(
                    "ub_idx must >= blk_idx"
                    "while ub_idx not in reduce_indexes."
                    "ub_idx is %d, blk_idx is %d" % (ub_idx, blk_idx))
            else:
                input_size = ub_factor
                r_ub_idx = r_indexes.index(ub_idx)
                max_ub_count = self.tiling_case.tensor_ub_size_before_reduce
                # [A,R,A,R,A] --> [A,A,R,R,A]
                for _dim in r_in_shape[r_ub_idx + 1:]:
                    input_size *= _dim
                    if not isinstance(_dim <= max_ub_count, bool):
                        results.append(_dim <= max_ub_count)
                results.append(input_size <= max_ub_count)

            # do constraint extra
            results.append(ub_factor <= self.tiling_case.tensor_ub_size_before_reduce)
            results.append(blk_factor <= self.tiling_case.tensor_ub_size_after_reduce)

            return results

        self.constraint_func_pair_list.append((params, func))

    def __need_storage_align(self):
        ub_split_axis_index = self.tiling_case.ub_split_axis_index
        shape_before_reduce = self.reduce_info.shape_before_reduce
        reduce_axis_indexes = self.reduce_info.reduce_axis_indexes
        # for shape(r4,a4,r3,a3,r2,a2,r1,a1), if ub split a1, do not need storage align
        if self.reduce_info.is_reduce_not_last_axis():
            a1_start_index, a1_end_index = \
                self.reduce_info.find_last_none_reduce_axis(shape_before_reduce,
                                                            reduce_axis_indexes)
            if a1_end_index is None:
                return False
            if a1_start_index <= ub_split_axis_index <= a1_end_index:
                return False

        else:
            r1_start_index, r1_end_index = self.reduce_info.find_last_reduce_axis(
                shape_before_reduce,
                reduce_axis_indexes)
            if r1_end_index is None:
                return False
            # for shape(a4,r4,a3,r3,a2,r2,a1,r1), if ub split r1, do not need storage align
            if r1_start_index <= ub_split_axis_index <= r1_end_index:
                return False
        return True

    def _calc_storage_align(self):
        if self.tiling_case.is_reduce_transpose_case:
            return

        if not self.__need_storage_align():
            return

        shape_before_reduce = self.reduce_info.shape_before_reduce
        reduce_axis_indexes = self.reduce_info.reduce_axis_indexes

        a1_start_index, a1_end_index = self.reduce_info.find_last_none_reduce_axis(shape_before_reduce,
                                                                                   reduce_axis_indexes)
        if self.reduce_info.is_reduce_not_last_axis():
            # None-Last Reduce needs to storage align reduce_tensor and all_other tensors before reduce
            # Align at last reduce axis
            align_axis_index = a1_start_index - 1
            reduce_tensor = self.reduce_info.reduce_tensor
            tensors_before = self.get_all_producer_stages(reduce_tensor)
            tensors_after = self.get_all_consumer_stages(reduce_tensor)
            for tensor in tensors_before:
                if tensor not in self.graph_info.input_tensor_set:
                    align_num = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[tensor.dtype])
                    storage_align_info = self.StorageAlignInfo(tensor,
                                                               align_axis_index,
                                                               align_num)
                    self.storage_align_list.append(storage_align_info)
            if a1_start_index - len(self.reduce_info.reduce_axis_indexes) == 0:
                return
            align_axis_index -= 1
            if reduce_tensor in self._data_flow_control:
                # reduce tensor has performed cache_write()
                reduce_tensor = self.get_buffers_of(reduce_tensor)[0]
            align_num = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[reduce_tensor.dtype])
            reduce_storage_align_info = self.StorageAlignInfo(reduce_tensor,
                                                              align_axis_index,
                                                              align_num)
            self.storage_align_list.append(reduce_storage_align_info)
            for tensor in tensors_after:
                if tensor not in self.graph_info.output_tensor_set:
                    align_num = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[tensor.dtype])
                    storage_align_info = self.StorageAlignInfo(tensor,
                                                               align_axis_index,
                                                               align_num)
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
                    align_num = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[tensor.dtype])
                    storage_align_info = self.StorageAlignInfo(tensor,
                                                               align_axis_index,
                                                               align_num)
                    self.storage_align_list.append(storage_align_info)

    def _calc_compute_align(self):
        if self.tiling_case.is_reduce_transpose_case:
            return

        if get_soc_spec(SOC_VERSION) != ASCEND_920A:
            return

        def _set_align(_tensor, _factor):
            return self.ComputeAlignInfo(_tensor, None, _factor)

        def _set_reduce_align(_reduce):
            if _reduce in self._data_flow_control:
                # _reduce has performed cache_write()
                _reduce = self.get_buffers_of(_reduce)[0]
            factor = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[_reduce.dtype])
            self.compute_align_list.append(_set_align(_reduce, factor))

        # No distinction between N_last and last
        # last not storage_align reduce
        # N_last storage_align reduce unless pattern in pure data_move
        for item in self.storage_align_list:
            tensor, factor = item.tensor, item.factor
            if not get_dsl_insn(tensor) in ["dma_copy", ""]:
                self.compute_align_list.append(_set_align(tensor, factor))

        if self.reduce_info.is_reduce_last_axis():
            reduce_tensor = self.reduce_info.reduce_tensor
            if get_dsl_insn(reduce_tensor) in ["reduce_max", "reduce_min"]:
                _set_reduce_align(reduce_tensor)

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
                self.reduce_info.reduce_axis_indexes)
            if a1_start_index <= self.tiling_case.ub_split_axis_index <= a1_end_index:
                reduce_ub_split_axis_outer = self.reduce_info.reduce_axis_indexes[-1]

        # Add anchor point
        self.anchor_point_list.append([reduce_ub_tiling_tensor, "ub_anchor"])
        self.anchor_point_axis_index_list.append(reduce_ub_split_axis_outer)

        # rfactor
        # input -> cast -> reduce_rf -> reduce -> cast -> output
        # reduce_rf compute_at reduce's last A
        if self._last_reduction_rf_optimization():
            reduce_ub_buffer = self.get_buffers_of(self.reduce_info.reduce_tensor)[0]
            _index = -3
            if not self.reduce_info.keepdims:
                _index = -2
            # support R in rfactor mode
            if len(self.reduce_info.shape_before_reduce) == 1:
                _index = -2
            rf_ub_split_idx = self.Placeholder(
                self.Placeholder.PlaceholderType.ITER_VAR, (
                    reduce_ub_buffer, _index)
            )

            self.anchor_point_list.append([reduce_ub_buffer, "ub_anchor"])
            self.anchor_point_axis_index_list.append(rf_ub_split_idx)

        block_tiling_tensor = self.block_tiling_info.tiling_tensor
        res_block_outer = self.block_tiling_result_pair[0]

        # Add anchor point
        self.anchor_point_list.append([block_tiling_tensor, "block_anchor"])
        self.anchor_point_axis_index_list.append(res_block_outer)

    def _calc_emit_insn(self):
        reduce_ub_tensor = self.get_buffers_of(self.reduce_info.reduce_tensor)[0]
        # Reduce-ub
        emit_insn_axis = self.ub_tiling_result_pair[1]
        ub_split_axis_index = self.tiling_case.ub_split_axis_index
        if self.reduce_info.is_reduce_not_last_axis():
            a1_start_index, a1_end_index = self.reduce_info.find_last_none_reduce_axis(
                self.reduce_info.shape_before_reduce,
                self.reduce_info.reduce_axis_indexes)
            if ub_split_axis_index < a1_start_index and ub_split_axis_index not in self.reduce_info.reduce_axis_indexes:
                emit_insn_axis = self.reduce_info.reduce_axis_indexes[0]
        elif ub_split_axis_index not in self.reduce_info.reduce_axis_indexes:
            compute = get_context().get_current_compute()
            if compute.get("_mode") == "zero":
                pass
            else:
                first_reduce_axis = self.reduce_info.reduce_tensor.op.reduce_axis[0]
                axis_dom = first_reduce_axis.dom
                if hasattr(axis_dom.min, "value") and hasattr(axis_dom.extent, "value"):
                    if first_reduce_axis.dom.min.value == 0 and first_reduce_axis.dom.extent.value == 1:
                        emit_insn_axis = self.reduce_info.reduce_axis_indexes[0]
        # Op of reduce need extra_space
        extra_space = self.tiling_case.tensor_ub_size_before_reduce
        if self._contains_zero_axis():
            self._emit_zero_reduce_insn(reduce_ub_tensor, emit_insn_axis)
        else:
            def _emit_reduce_insn():
                if self._last_reduction_rf_optimization():
                    # reduce emit_insn
                    reduce_emit_insn_axis = self.Placeholder(
                        self.Placeholder.PlaceholderType.ITER_VAR, (reduce_ub_tensor, -1))
                    self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(reduce_ub_tensor,
                                                                           reduce_emit_insn_axis,
                                                                           INSN_MAPPING.get(
                                                                               get_dsl_insn(reduce_ub_tensor)),
                                                                           {"extra_space": extra_space}))
                    # reduce_rf emit_insn
                    reduce_rf_emit_insn_axis = emit_insn_axis
                    reduce_rf_ub_tensor = self.get_buffers_of(self.reduce_rf)[0]
                    self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(reduce_rf_ub_tensor,
                                                                           reduce_rf_emit_insn_axis,
                                                                           INSN_MAPPING.get(
                                                                               get_dsl_insn(reduce_ub_tensor)),
                                                                           {"extra_space": extra_space}))

                else:
                    if self.tiling_case.is_reduce_transpose_case:
                        self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(reduce_ub_tensor,
                                                                               emit_insn_axis,
                                                                               INSN_MAPPING.get(
                                                                                   get_dsl_insn(reduce_ub_tensor)),
                                                                               {"extra_space": extra_space,
                                                                                "trans": True}))
                    else:
                        self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(reduce_ub_tensor,
                                                                               emit_insn_axis,
                                                                               INSN_MAPPING.get(
                                                                                   get_dsl_insn(reduce_ub_tensor)),
                                                                               {"extra_space": extra_space}))

            _emit_reduce_insn()

        def _traverse():
            # tensors_before_root: exclude reduce_tensor unless reduce_tensor is output_tensor
            tensors_before_root = self.get_all_producer_stages(list(self.graph_info.output_tensor_set)[0])
            remove_list = []
            for _tensor_i in tensors_before_root:
                if is_reduce_tensor(_tensor_i):
                    remove_list.append(_tensor_i)
                    continue
            for item in remove_list:
                if item not in self.graph_info.real_output_tensor_set:
                    tensors_before_root.remove(item)

            # add fake_node
            tensors_set = tensors_before_root | self.graph_info.output_tensor_set
            return tensors_set

        # Before-And-After-reduce
        before_reduce_tensors = self.get_all_producer_stages(reduce_ub_tensor)
        after_reduce_tensors = self.get_all_consumer_stages(reduce_ub_tensor)
        remaining_tensors = before_reduce_tensors | after_reduce_tensors
        if get_context().get_current_compute().get("_mode") != "zero":
            # traversing all tensors and their buffers
            remaining_tensors = _traverse()

        def _emit_remaining_tensors(remaining_tensor):
            insn = get_dsl_insn(remaining_tensor)
            emit_insn_axis_index = 0

            # if not have fake_node: output_tensor_set will cover real_output_tensor_set
            if remaining_tensor in self.graph_info.real_output_tensor_set:
                insn = "dma_copy"
                emit_insn_axis_index = 0
            if remaining_tensor in self.graph_info.output_tensor_set:
                emit_insn_axis_index = self.block_tiling_result_pair[1]

            if insn == "":
                insn = "dma_copy"

            if 0 in remaining_tensor.shape:
                self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(remaining_tensor, emit_insn_axis_index,
                                                                       "phony_insn"))
            else:
                self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(remaining_tensor, emit_insn_axis_index,
                                                                       INSN_MAPPING[insn]))

        for tensor in remaining_tensors:
            if tensor in self.graph_info.input_tensor_set:
                continue
            _emit_remaining_tensors(tensor)

    def _calc_pragma(self):
        # For zero compute mode, skip pragma stage
        if get_context().get_current_compute().get("_mode") == "zero":
            return
        # For rf compute mode, skip pragma stage
        if self._last_reduction_rf_optimization():
            return

        # create virtual mapping
        _shape = [Dim(R, _idx) if _idx in self.reduce_info.reduce_axis_indexes else Dim(A, _idx)
                  for _idx in range(len(self.reduce_info.shape_before_reduce))]

        # split
        blk_split_idx = self.tiling_case.block_split_axis_index
        ub_split_idx = self.tiling_case.ub_split_axis_index
        if blk_split_idx <= ub_split_idx:
            Dim.split(_shape, blk_split_idx)
            ub_split_idx += 1
            Dim.split(_shape, ub_split_idx, model="UBSplit")
        else:
            Dim.split(_shape, ub_split_idx, model="UBSplit")
            blk_split_idx += 1
            Dim.split(_shape, blk_split_idx, )

        # reorder
        _a_shape, _r_shape = [], []
        for item in _shape:
            if item.axis_type == A:
                _a_shape.append(item)
            else:
                _r_shape.append(item)
        target_shape = _a_shape + _r_shape
        if not self.reduce_info.is_reduce_last_axis():
            _r_shape.append(_a_shape.pop(-1))
            target_shape = _a_shape + _r_shape

        # find serial axis
        idx_ub_outer = target_shape.index(_shape[ub_split_idx])
        axis_in_ub = target_shape[idx_ub_outer + 1:]
        axis_in_ub.sort(key=lambda x: x.idx, reverse=True)
        self._serial_group = Dim.group([x.idx for x in axis_in_ub])
        self._serial_group.sort(key=lambda x: x[1] - x[0], reverse=True)

        # Initialization
        reduce_ub_tensor = self.get_buffers_of(self.reduce_info.reduce_tensor)[0]
        before_reduce_tensors = self.get_all_producer_stages(reduce_ub_tensor)
        ub_tiling_on_reduce_axis = self.ub_tiling_info.tiling_axis_index in self.reduce_info.reduce_axis_indexes
        # For tensor before reduce, try to fuse all continuous reordered axis, Search in reversed order.
        for tensor in before_reduce_tensors:
            # Do not pragma placeholder
            if tensor in self.graph_info.input_tensor_set:
                continue
            # For ub tensor
            if get_dsl_insn(tensor) != "":
                # Iterate all axis after ub_tiling_axis and check if they need to be in the axis_group
                for axis_idx in range(self.ub_tiling_info.tiling_axis_index,
                                      len(self.reduce_info.shape_before_reduce) - 1):
                    if axis_idx in self.reduce_info.reduce_axis_indexes or not ub_tiling_on_reduce_axis:
                        self.pragma(tensor, axis_idx, "axis_group", 0)
                # last axis needs to be in the axis_group
                self.pragma(tensor, len(self.reduce_info.shape_before_reduce) - 1, "axis_group", 0)
            else:
                # For dma tensor
                extend = self._serial_group[0][1] - self._serial_group[0][0] + 1
                length = len(self.reduce_info.shape_before_reduce)
                axis_range = \
                    range(length - 1, length - 1 - extend, -1) if extend != 1 else range(length - 1, length - 3, -1)

                for axis_idx in range(self.ub_tiling_info.tiling_axis_index,
                                      len(self.reduce_info.shape_before_reduce)):
                    if axis_idx in axis_range:
                        self.pragma(tensor, axis_idx, "axis_group", 0)

        # For reduce tensor
        # ub_tiling_axis inner needs to be in the axis_group if it is reduce axis
        if ub_tiling_on_reduce_axis:
            self.pragma(reduce_ub_tensor, self.ub_tiling_result_pair[1], "axis_group", 0)
        for axis_idx in range(self.ub_tiling_info.tiling_axis_index + 1,
                              len(self.reduce_info.shape_before_reduce)):
            # pragma on all reduce axis
            if axis_idx in self.reduce_info.reduce_axis_indexes:
                self.pragma(reduce_ub_tensor, axis_idx, "axis_group", 0)

    @staticmethod
    def _contains_zero_axis():
        compute = get_context().get_current_compute()
        return compute.get("_mode") == "zero"

    def _emit_zero_reduce_insn(self, reduce_ub_tensor, emit_insn_axis):
        compute = get_context().get_current_compute()
        insn = "phony_insn"
        if compute.get("_shape") == (1, -1, 0):
            insn = "vector_dup"

        self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(reduce_ub_tensor, emit_insn_axis, insn))

    def _last_reduction_rf_optimization(self):
        if self.reduce_info.is_reduce_last_axis() \
                and self.reduce_info.all_axes[self.tiling_case.ub_split_axis_index] \
                in self.reduce_info.reduce_axes:
            return True
        return False
