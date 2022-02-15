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
reduce atomic schedule
"""
import abc
import copy

from tvm.tensor import Tensor

from tbe import tvm
from tbe.common.platform import ASCEND_920A
from tbe.common.platform import SOC_VERSION
from tbe.common.platform import scope_ubuf
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation
from tbe.dsl.base.operation import var_inner

from ...constants import DTYPE_BYTE_MAPPING
from ...constants import INSN_MAPPING
from ...constants import ReduceCategory
from ...util import get_dsl_insn
from ...util import is_reduce_tensor

from .reduce_tilingcase import Dim
from .reduce_tilingcase import R
from .reduce_tilingcase import A
from .reduce_tilingcase import ReduceTilingCase
from .vector_info import ComputeGraphInfo

CONST = "const"
BLOCK_SIZE_BYTE = 32


class _VectorSchedule:
    """
    class for Vector Schedule
    """

    class ComputeAlignInfo:
        """
        class for Compute Align Info
        """

        def __init__(self, tensor=None, pad=None, factor=None):
            self.tensor = tensor
            self.pad = pad
            self.factor = factor

    def __init__(self, graph_info):
        self.schedule = None
        self.graph_info = graph_info

        self.forward_compute_graph_map = self.graph_info.tensor_consumers_map
        self.forward_stage_graph_map = ComputeGraphInfo.set_map_deepcopy(self.forward_compute_graph_map)
        self.backward_compute_graph_map = self.graph_info.tensor_producers_map
        self.backward_stage_graph_map = ComputeGraphInfo.set_map_deepcopy(self.backward_compute_graph_map)

        self.cache_read_tensors_and_buffer_map = {}
        self.cache_write_tensors_and_buffer_map = {}
        self.double_buffer_tensors = []

        self.need_multi_core = True
        self.multi_core_bind_tensor = None
        self.multi_core_fused_axis = None

        self.compute_at_map = {}
        self.emit_insn_map = {}

        self._ori_and_align_pad_tensor_map = {}
        self._align_pad_tensor_list = []
        self.remove_pad_tensor = None

    def _create_schedule(self):
        self.schedule = tvm.create_schedule([tensor.op for tensor in self.graph_info.output_tensor_set])

    def _do_cache_read(self):
        for tensor in self.graph_info.input_tensor_set:
            readers = self.graph_info.tensor_consumers_map[tensor]
            read_buffer = self.schedule.cache_read(tensor, scope_ubuf, readers)
            self.cache_read_tensors_and_buffer_map[tensor] = read_buffer
            self.update_stage(read_buffer, tensor, False)

            if self.tiling_case.is_reduce_pad_case:
                align_pad_buffer = self.schedule.cache_read(read_buffer, scope_ubuf, readers)
                self._ori_and_align_pad_tensor_map[read_buffer] = align_pad_buffer
                self._align_pad_tensor_list.append(align_pad_buffer)
                self.cache_read_tensors_and_buffer_map[read_buffer] = align_pad_buffer
                self.update_stage(align_pad_buffer, read_buffer, False)

    def _do_set_scope(self):
        for tensor in self.graph_info.mid_tensor_set:
            if tensor not in self.graph_info.real_output_tensor_set:
                self.schedule[tensor].set_scope(scope_ubuf)

    @abc.abstractmethod
    def _do_tiling(self):
        """
        :return:
        """

    @abc.abstractmethod
    def _do_reorder(self):
        """
        :return:
        """

    @abc.abstractmethod
    def _do_storage_bound(self):
        """
        :return:
        """

    @abc.abstractmethod
    def _do_set_constraint(self):
        """
        :return:
        """

    @abc.abstractmethod
    def _do_storage_align(self):
        """
        :return:
        """

    @abc.abstractmethod
    def _do_compute_align(self):
        """
        :return:
        """

    def _do_multi_core(self):
        if self.need_multi_core:
            res = self.multi_core_bind_tensor
            block = tvm.thread_axis("blockIdx.x")
            self.schedule[res].bind(self.multi_core_fused_axis, block)

    def _do_compute_at(self):
        for stage in self.compute_at_map:
            parent_stage = self.compute_at_map[stage]["parent"]
            scope_iter_var = self.compute_at_map[stage]["scope"]
            self.schedule[stage].compute_at(parent_stage, scope_iter_var)

    def _do_emit_insn(self):
        for stage in self.emit_insn_map:
            scope_iter_var = self.emit_insn_map[stage]["scope"]
            instruction = self.emit_insn_map[stage]["instruction"]
            extra_space = self.emit_insn_map[stage].get("extra_space")
            if extra_space:
                self.schedule[stage].emit_insn(scope_iter_var, instruction,
                                               attrs={'storage_bound': [extra_space]})
            else:
                self.schedule[stage].emit_insn(scope_iter_var, instruction)

    def _do_double_buffer(self):
        for _tensor in self.double_buffer_tensors:
            self.schedule[_tensor].double_buffer()
        operation.add_build_arg("double_buffer_non_reuse", True)

    def update_stage(self, source_tensor, dst_tensor, before):
        """
        update  graph stage map by new tensor
        """
        if before:
            self.forward_stage_graph_map.setdefault(source_tensor, set())
            self.backward_stage_graph_map.setdefault(source_tensor, set())
            for producer in tuple(self.backward_stage_graph_map[dst_tensor]):
                self.forward_stage_graph_map[producer].remove(dst_tensor)
                self.forward_stage_graph_map[producer].add(source_tensor)
                self.backward_stage_graph_map[dst_tensor].remove(producer)
                self.backward_stage_graph_map[source_tensor].add(producer)
            self.forward_stage_graph_map[source_tensor].add(dst_tensor)
            self.backward_stage_graph_map[dst_tensor].add(source_tensor)
        else:
            self.forward_stage_graph_map.setdefault(source_tensor, set())
            self.backward_stage_graph_map.setdefault(source_tensor, set())
            for consumer in tuple(self.forward_stage_graph_map[dst_tensor]):
                self.forward_stage_graph_map[dst_tensor].discard(consumer)
                self.backward_stage_graph_map[consumer].discard(dst_tensor)
                self.backward_stage_graph_map[consumer].add(source_tensor)
                self.forward_stage_graph_map[source_tensor].add(consumer)
            self.forward_stage_graph_map[dst_tensor].add(source_tensor)
            self.backward_stage_graph_map[source_tensor].add(dst_tensor)

    def get_all_producers_stages(self, tensor):
        """
        get all produce stages for current tensor
        """
        producers = set()
        for producer in self.backward_stage_graph_map[tensor]:
            producers.add(producer)
            producers.update(self.get_all_producers_stages(producer))
        return producers


class ReduceAtomicSchedule(_VectorSchedule):
    """
    Schedule for Atomic Reduce
    """

    def __init__(self, graph_info, reduce_info):
        _VectorSchedule.__init__(self, graph_info)
        self.tiling_case = None
        self.reduce_info = reduce_info
        self.block_tiling_result_pair = None
        self.ub_tiling_result_pair = None

        self.iter_block_outer = None
        self.iter_block_inner = None
        self.iter_ub_outer = None
        self.iter_ub_inner = None

        self.reduce_rfs_rfs = None
        self.reduce_rfs = None
        self.reduce_repls = None

        self._storage_align_para = {}
        self._compute_align_list = []
        self._axis_offset = 0
        self._reduce_case = 0
        self._serial_group = None

    def do_schedule(self, outs, tiling_case):
        """
        do atomic schedule
        """
        self.tiling_case = tiling_case
        self._create_schedule()
        self._do_cache_read()
        self._do_set_scope()

        self._calculate_tiling()
        self._do_tiling()
        self._do_reorder()

        self._do_storage_bound()
        self._do_set_constraint()

        self._calculate_storage_align()
        self._do_storage_align()

        self._calculate_compute_align()
        self._do_compute_align()

        self._calculate_multi_core()
        self._do_multi_core()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn()
        self._do_emit_insn()

        self._calculate_pragma()
        self._do_pragma()

        self._calculate_db()
        self._do_double_buffer()

        self._replace_outs(outs)

        return self.schedule

    def _calculate_tiling(self):
        """
        calculate tiling strategy
        """
        case: ReduceTilingCase = self.tiling_case
        res_tensor = list(self.graph_info.real_output_tensor_set)[0]

        block_split_axis_index = case.block_split_axis_index
        ub_split_axis_index = case.ub_split_axis_index

        block_factor = case.block_factor if case.block_factor is not None else var_inner("_block_factor", (1, None))
        ub_factor = case.ub_factor if case.ub_factor is not None else var_inner("_ub_factor", (1, None))

        self.block_tiling_result_pair = [res_tensor, block_split_axis_index, block_factor]
        self.ub_tiling_result_pair = [res_tensor, ub_split_axis_index, ub_factor]

    def _do_tiling(self):
        """
        :return:
        """
        self._do_block_tiling()

        self._atomic_additonal_schedule()

        self._do_ub_tiling()

    def _do_block_tiling(self):
        tiling_tensor = self.block_tiling_result_pair[0]
        tiling_axis = self.block_tiling_result_pair[1]
        tiling_factor = self.block_tiling_result_pair[2]

        if tiling_axis not in self.reduce_info.reduce_axis_indexes:
            dict_args = {"errCode": "E90003", "detailed_cause": "Atomic schedule block tiling can " \
                                                                "only split reduce axis! " \
                                                                "block_split_axis is [%s], " \
                                                                "while reduce_axis is [%s]" \
                                                                % (tiling_axis, self.reduce_info.reduce_axis_indexes)}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        tiling_axis = self.reduce_info.reduce_axis_indexes.index(tiling_axis)
        axis_var = tiling_tensor.op.reduce_axis[tiling_axis]
        self.iter_block_outer, self.iter_block_inner = \
            self.schedule[tiling_tensor].split(axis_var, factor=tiling_factor)

    def _do_ub_tiling(self):
        block_tiling_tensor = self.block_tiling_result_pair[0]
        block_split_axis = self.block_tiling_result_pair[1]

        ub_tiling_tensor = self.ub_tiling_result_pair[0]
        ub_split_axis = self.ub_tiling_result_pair[1]
        ub_split_factor = self.ub_tiling_result_pair[2]

        if ub_split_axis in self.reduce_info.reduce_axis_indexes:
            axis = self.reduce_info.reduce_axis_indexes.index(ub_split_axis)
            axis_var = block_tiling_tensor.op.reduce_axis[axis]
            if block_split_axis == ub_split_axis:
                axis_var = ub_tiling_tensor.op.reduce_axis[-1]

            if not self._last_reduction_rf_optimization():
                self.iter_ub_outer, self.iter_ub_inner = \
                    self.schedule[ub_tiling_tensor].split(axis_var, factor=ub_split_factor)
            else:
                if ub_split_axis == self.reduce_info.reduce_axis_indexes[-1]:
                    # when UB split on last R axis, before do rfactor on last reduce axis
                    # we should do UB split on atomic rfactor
                    self.iter_ub_outer, self.iter_ub_inner = \
                        self.schedule[ub_tiling_tensor].split(axis_var, factor=ub_split_factor)
                    self.reduce_rfs_rfs = self.schedule.rfactor(self.reduce_rfs, self.iter_ub_inner, -1)
                else:
                    # UB not split last axis,rfactor will use last reduce axis
                    # the last element in array reduce_axis is k1.inner,so we will get reduce_axis index of -2
                    # for last reduce axis
                    self.reduce_rfs_rfs = self.schedule.rfactor(self.reduce_rfs, self.reduce_rfs.op.reduce_axis[-2],
                                                                -1)
                    self.ub_tiling_result_pair[0] = self.reduce_rfs_rfs
                    ub_tiling_tensor = self.reduce_rfs_rfs
                    if block_split_axis == ub_split_axis:
                        axis_var = ub_tiling_tensor.op.reduce_axis[-1]
                    self.iter_ub_outer, self.iter_ub_inner = \
                        self.schedule[ub_tiling_tensor].split(axis_var, factor=ub_split_factor)
                self.schedule[self.reduce_rfs_rfs].set_scope(scope_ubuf)
                self.update_stage(self.reduce_rfs_rfs, self.reduce_rfs, True)

        else:
            none_reduce_index_map = self._find_none_reduce_axis_map(
                self.reduce_info.shape_before_reduce,
                self.reduce_info.reduce_axis_indexes,
                self.reduce_info.keepdims)
            axis = none_reduce_index_map[ub_split_axis]
            axis_var = ub_tiling_tensor.op.axis[axis + self._axis_offset]
            self.iter_ub_outer, self.iter_ub_inner = \
                self.schedule[ub_tiling_tensor].split(axis_var, factor=ub_split_factor)

    def _atomic_additonal_schedule(self):
        block_tiling_tensor = self.block_tiling_result_pair[0]
        block_split_axis = self.block_tiling_result_pair[1]
        block_outer_var = self.iter_block_outer

        fused_list = []
        block_split_axis = self.reduce_info.reduce_axis_indexes.index(block_split_axis)
        for i in range(0, block_split_axis):
            fused_list.append(block_tiling_tensor.op.reduce_axis[i])

        fused_list.append(block_outer_var)
        fused = self.schedule[block_tiling_tensor].fuse(*fused_list)

        # rf tensor
        factor_axis, self._axis_offset = 0, 1
        self.reduce_rfs = self.schedule.rfactor(block_tiling_tensor,
                                                fused, factor_axis=factor_axis)

        # cache write
        if len(self.graph_info.real_output_tensor_set) > 1:
            # TupleSum
            self.ub_tiling_result_pair[0] = self.reduce_rfs[0]
            self.schedule[self.reduce_rfs[0]].set_scope(scope_ubuf)
            for _idx, _tensor in enumerate(list(self.graph_info.real_output_tensor_set)):
                self.update_stage(self.reduce_rfs[_idx], _tensor, True)

            self.reduce_repls = self.schedule.cache_write(self.graph_info.real_output_tensor_set, "")
            for _idx, _tensor in enumerate(list(self.graph_info.real_output_tensor_set)):
                self.update_stage(self.reduce_repls[_idx], _tensor, True)
                self.cache_write_tensors_and_buffer_map[_tensor] = self.reduce_repls[_idx]
        else:
            # SingleSum
            self.ub_tiling_result_pair[0] = self.reduce_rfs
            self.schedule[self.reduce_rfs].set_scope(scope_ubuf)
            self.update_stage(self.reduce_rfs, block_tiling_tensor, True)

            self.reduce_repls = self.schedule.cache_write(block_tiling_tensor, "")
            self.update_stage(self.reduce_repls, block_tiling_tensor, True)
            self.cache_write_tensors_and_buffer_map[block_tiling_tensor] = self.reduce_repls

            if self.tiling_case.need_remove_pad:
                self.remove_pad_tensor = self.schedule.cache_read(self.reduce_rfs, scope_ubuf, [self.reduce_repls])
                self._ori_and_align_pad_tensor_map[self.reduce_rfs] = self.remove_pad_tensor
                self.cache_read_tensors_and_buffer_map[self.reduce_rfs] = self.remove_pad_tensor
                self.update_stage(self.remove_pad_tensor, self.reduce_rfs, False)

    def _do_reorder(self):

        final_out_tensor_global = self.reduce_repls if isinstance(self.reduce_repls, Tensor) else self.reduce_repls[0]
        final_out_tensor_ub_rf = self.reduce_rfs if isinstance(self.reduce_rfs, Tensor) else self.reduce_rfs[0]

        if self._reduce_case == ReduceCategory.ALL_REDUCE:
            # don't need reorder
            self._reorder_atomic_reduce_all(final_out_tensor_ub_rf,
                                            final_out_tensor_global)
        if self._reduce_case == ReduceCategory.NOT_LAST_REDUCE:
            # for shape(r4,a4,r3,a3,r2,a2,r1,a1),
            # reorder ir (a1,a2,..ak,rbo,r1,.,rb-1,rb+1,..rn,rbi) to
            # (rbo,a1,a2,..ak,r1,.rb-1,rbi,rb+1,,.rn)
            self._reorder_atomic_reduce_not_last_axis(final_out_tensor_ub_rf,
                                                      final_out_tensor_global)
        if self._reduce_case == ReduceCategory.LAST_REDUCE:
            # for shape (a4,r4,a3,r3,a2,r2,a1,r1),
            # reorder ir (a1,a2,..ak,rbo,r1,.,rb-1,rb+1,..rn,rbi) to
            # (rbo,a1,a2,..ak-1,r1,.rb-1,rbi,rb+1,,.,ak,rn)
            self._reorder_atomic_reduce_last_axis(final_out_tensor_ub_rf,
                                                  final_out_tensor_global)

    def _do_storage_bound(self):
        tensors_before_reduce = self.get_all_producers_stages(self.reduce_rfs)
        for stage_tensor in self.forward_stage_graph_map:
            if stage_tensor is self.reduce_rfs:
                ub_count = self.tiling_case.tensor_ub_size_before_reduce
            elif stage_tensor in tensors_before_reduce:
                ub_count = self.tiling_case.tensor_ub_size_before_reduce
            else:
                ub_count = self.tiling_case.tensor_ub_size_after_reduce
            self.schedule[stage_tensor].set_buffer_size(ub_count)

    def _do_set_constraint(self):
        if operation.get_context().get("_mode") == CONST:
            return

        ub_split_axis = self.ub_tiling_result_pair[1]
        ub_split_inner = self.ub_tiling_result_pair[2]

        shape_before_reduce = self.reduce_info.shape_before_reduce
        reduce_axis_index = self.reduce_info.reduce_axis_indexes
        max_ub_count = self.tiling_case.tensor_ub_size_before_reduce

        if self._reduce_case == ReduceCategory.NOT_LAST_REDUCE:
            reordered_shape, _, orignal_to_reorder_axis_map = \
                self._reorder_reduce_nlast_shape(shape_before_reduce,
                                                 reduce_axis_index)
            axis = orignal_to_reorder_axis_map[ub_split_axis]
            shape_in_ub = ub_split_inner
            if isinstance(shape_in_ub, tvm.expr.Var):
                self.schedule.set_constraint(ub_split_inner <= max_ub_count)
            for i in range(axis, len(reordered_shape)):
                shape_in_ub = shape_in_ub * reordered_shape[i]
                if isinstance(shape_in_ub, tvm.expr.Var):
                    self.schedule.set_constraint(
                        reordered_shape[i] <= max_ub_count)

        else:
            reordered_shape, _, orignal_to_reorder_axis_map = \
                self._reorder_reduce_last_shape(shape_before_reduce,
                                                reduce_axis_index)
            axis = orignal_to_reorder_axis_map[ub_split_axis]
            shape_in_ub = ub_split_inner
            if isinstance(shape_in_ub, tvm.expr.Var):
                self.schedule.set_constraint(ub_split_inner <= max_ub_count)
            for i in range(axis + 1, len(reordered_shape)):
                shape_in_ub = shape_in_ub * reordered_shape[i]
                if isinstance(shape_in_ub, tvm.expr.Var):
                    self.schedule.set_constraint(
                        reordered_shape[i] <= max_ub_count)

        self.schedule.set_constraint(shape_in_ub <= max_ub_count)

    def _calculate_storage_align(self):
        # None-Last Reduce needs to storage align reduce_tensor and all_other tensors before reduce
        # Align at last reduce axis
        self._storage_align_para.clear()
        shape_before_reduce = self.reduce_info.shape_before_reduce
        reduce_axis_index = self.reduce_info.reduce_axis_indexes
        is_keep_dims = self.reduce_info.keepdims

        # condition don't align
        if not self._need_storage_align():
            return
        a1_start_index, a1_end_index = \
            self._find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)
        if a1_end_index is None:
            return

        def _storage_align_tensors_before_reduce(_align_axis, reduce_tensor):
            tensors_before_reduce = self.get_all_producers_stages(reduce_tensor)
            for tensor in tensors_before_reduce:
                if tensor not in self.graph_info.input_tensor_set:
                    if tensor in self._ori_and_align_pad_tensor_map.keys():
                        continue
                    align_factor = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[tensor.dtype])
                    para = {"align_axis_var": tensor.op.axis[_align_axis],
                            "align_factor": align_factor,
                            "offset": 0
                            }
                    self._storage_align_para[tensor] = para

        def _storage_align_reduce_tensor_nlast(is_rf_rf_reduce, reduce_tensor):
            res_a1_start_index = a1_start_index
            if not is_keep_dims:
                if is_rf_rf_reduce:
                    # for rf_rf_tesnor the last R axis has been Rfacor as A axis
                    # so only has len(reduce_axis_index) -1 R aixs before last A axis
                    res_a1_start_index = a1_start_index - len(reduce_axis_index) + 1
                else:
                    res_a1_start_index = a1_start_index - len(reduce_axis_index)

            if res_a1_start_index == 0:
                return
            res_align_axis = res_a1_start_index - 1

            align_factor = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[reduce_tensor.dtype])
            para = {"align_axis_var": reduce_tensor.op.axis[res_align_axis + self._axis_offset],
                    "align_factor": align_factor,
                    "offset": 0}
            self._storage_align_para[reduce_tensor] = para

        if self._reduce_case == ReduceCategory.NOT_LAST_REDUCE:
            align_axis = a1_start_index - 1
            if align_axis < 0:
                align_axis = a1_end_index
            _storage_align_tensors_before_reduce(align_axis, self.reduce_rfs)
            _storage_align_reduce_tensor_nlast(False, self.reduce_rfs)

        else:
            align_axis = a1_end_index
            if self._last_reduction_rf_optimization():
                _storage_align_tensors_before_reduce(align_axis, self.reduce_rfs_rfs)
                _storage_align_reduce_tensor_nlast(True, self.reduce_rfs_rfs)
            else:
                _storage_align_tensors_before_reduce(align_axis, self.reduce_rfs)

    def _do_storage_align(self):
        for stage in self._storage_align_para:
            scope_iter_var = self._storage_align_para[stage]["align_axis_var"]
            align_factor = self._storage_align_para[stage]["align_factor"]
            offset = self._storage_align_para[stage]["offset"]
            self.schedule[stage].storage_align(
                scope_iter_var, align_factor, offset)

    def _calculate_compute_align(self):
        if get_soc_spec(SOC_VERSION) != ASCEND_920A:
            return

        def _set_align(_tensor, _factor):
            return self.ComputeAlignInfo(_tensor, None, _factor)

        def _set_reduce_align(_reduce):
            factor = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[_reduce.dtype])
            self._compute_align_list.append(_set_align(_reduce, factor))

        # No distinction between N_last and last
        # last not storage_align reduce
        # N_last storage_align reduce unless pattern in pure data_move
        # Attention tag of rf is "" while used get_dsl_insn
        for tensor in self._storage_align_para:
            factor = self._storage_align_para[tensor]["align_factor"]
            if hasattr(self.reduce_rfs, "index") and \
                    tensor in self.reduce_rfs or tensor in [self.reduce_rfs, ]:
                self._compute_align_list.append(_set_align(tensor, factor))
            else:
                if not get_dsl_insn(tensor) in ["dma_copy", ""]:
                    self._compute_align_list.append(_set_align(tensor, factor))

        # Deal Reduce
        if self.reduce_info.is_reduce_last_axis():
            if get_dsl_insn(self.reduce_info.reduce_tensor) in ["reduce_max", "reduce_min"]:
                _set_reduce_align(self.reduce_rfs)

    def _do_compute_align(self):
        """
        Reorder of AtomicSch decided distinction between last and N_last in
        reduction of "rf" while "repls" always in last pattern. The pattern
        of "rf" is as same as ori reduce_tensor
        """
        for compute_align in self._compute_align_list:
            pad = compute_align.pad
            factor = compute_align.factor
            tensor = compute_align.tensor
            stage = self.schedule[tensor]
            if is_reduce_tensor(tensor):
                if self.reduce_info.is_reduce_last_axis():
                    axis = tensor.op.reduce_axis[-1] if len(tensor.op.reduce_axis) == 1 else tensor.op.reduce_axis[-2]
                else:
                    axis = tensor.op.axis[-1]
            else:
                axis = tensor.op.axis[-1]
            stage.compute_align(axis, factor, pad)

    def _calculate_multi_core(self):
        if self.need_multi_core:
            self.multi_core_fused_axis = self.reduce_repls.op.reduce_axis[0]
            self.multi_core_bind_tensor = self.reduce_repls

    def _calculate_compute_at(self):
        """
        Calculate the tensor that needs compute at
        """

        self.compute_at_map.clear()

        ub_tiling_tensor = self.ub_tiling_result_pair[0]
        ub_split_axis = self.ub_tiling_result_pair[1]
        block_split_axis = self.block_tiling_result_pair[1]
        res_ub_outer = self.iter_ub_outer
        reduce_axis_index = self.reduce_info.reduce_axis_indexes
        shape_before_reduce = self.reduce_info.shape_before_reduce
        is_keep_dim = self.reduce_info.keepdims

        if self._reduce_case == ReduceCategory.NOT_LAST_REDUCE:
            # for shape (r4,a4,r3,a3,r2,a2,r1,a1), if ub split a1, compute at r1
            # when a1 is continous,
            a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            if a1_end_index is None:
                dict_args = {"errCode": "E90001", "detailed_cause": "a1_end_index can not be none!"}
                raise RuntimeError(dict_args, get_error_message(dict_args))
            if a1_start_index <= ub_split_axis <= a1_end_index:
                if len(ub_tiling_tensor.op.reduce_axis) > 1:
                    res_ub_outer = ub_tiling_tensor.op.reduce_axis[-2]
                else:
                    res_ub_outer = ub_tiling_tensor.op.reduce_axis[-1]

        def _compute_at_tensor_before_reduce(reduce_tensor, scop_axis):
            tensors_before_reduce = self.get_all_producers_stages(reduce_tensor)
            for tensor in tensors_before_reduce:
                if tensor not in self.graph_info.input_tensor_set:
                    para = {"parent": self.schedule[reduce_tensor], "scope": scop_axis}
                    self.compute_at_map[tensor] = para

            if self.is_ARA_1_0_case():
                for input_tensor in self._ori_and_align_pad_tensor_map.keys():
                    align_buffer = self._ori_and_align_pad_tensor_map.get(input_tensor)
                    para = {"parent": self.schedule[align_buffer], "scope": self.schedule[align_buffer].op.axis[0]}
                    self.compute_at_map[input_tensor] = para

        if not self._last_reduction_rf_optimization():
            _compute_at_tensor_before_reduce(self.reduce_rfs, res_ub_outer)
        else:
            if self.ub_tiling_result_pair[1] != self.reduce_info.reduce_axis_indexes[-1]:
                # split on rf_rf tensor,so compute at res_ub_outer
                _compute_at_tensor_before_reduce(self.reduce_rfs_rfs, res_ub_outer)
            else:
                # split on rf tensor,so compute at k2.inner.inner
                scope_axis = self.schedule[self.reduce_rfs_rfs].op.reduce_axis[-1]
                _compute_at_tensor_before_reduce(self.reduce_rfs_rfs, scope_axis)

            a1_start_index, a1_end_index = self._find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)
            none_reduce_index_map = self._find_none_reduce_axis_map(
                shape_before_reduce, reduce_axis_index, is_keep_dim)
            if a1_end_index is None:
                dict_args = {"errCode": "E90001", "detailed_cause": "a1_end_index can not be none!"}
                raise RuntimeError(dict_args, get_error_message(dict_args))

            # scop axis of reduce rfs should be last A axis of rfs tensor
            # if rf tensor left A00A0A1R, A1 will be scop axis
            reduce_rfs_scope_axis = none_reduce_index_map[a1_end_index] + self._axis_offset
            para_reduce_rfs = {"parent": self.schedule[self.reduce_rfs],
                               "scope": self.schedule[self.reduce_rfs].op.axis[reduce_rfs_scope_axis]}
            self.compute_at_map[self.reduce_rfs_rfs] = para_reduce_rfs

        para_reduce_repls = {"parent": self.schedule[self.reduce_repls],
                             "scope": self.reduce_repls.op.reduce_axis[0]}
        self.compute_at_map[self.reduce_rfs] = para_reduce_repls

        if self.remove_pad_tensor is not None:
            self.compute_at_map[self.remove_pad_tensor] = para_reduce_repls

    def _calculate_emit_insn(self):
        self.emit_insn_map.clear()
        ub_tiling_tensor = self.ub_tiling_result_pair[0]
        ub_split_axis = self.ub_tiling_result_pair[1]
        res_ub_inner = self.iter_ub_inner
        reduce_axis_index = self.reduce_info.reduce_axis_indexes
        shape_before_reduce = self.reduce_info.shape_before_reduce

        # TensorsBeforeReduce
        if self._last_reduction_rf_optimization():
            tensors_before_reduce = self.get_all_producers_stages(self.reduce_rfs_rfs)
        else:
            tensors_before_reduce = self.get_all_producers_stages(self.reduce_rfs)

        def emit_reduce_insn_ub_split_on_reduce_axis():
            for tensor in tensors_before_reduce:
                if tensor in self.graph_info.input_tensor_set:
                    continue
                if tensor in self._align_pad_tensor_list:
                    insn = "align_pad"
                    if self.is_ARA_1_0_case():
                        emit_insn_axis_index = -2
                    else:
                        emit_insn_axis_index = 0
                else:
                    insn = INSN_MAPPING.get(get_dsl_insn(tensor), get_dsl_insn(tensor))
                    if insn == "":
                        insn = "dma_copy"
                        emit_insn_axis_index = 0
                    else:
                        emit_insn_axis_index = 0

                para = {"scope": self.schedule[tensor].op.axis[emit_insn_axis_index],
                        "instruction": insn}
                self.emit_insn_map[tensor] = para

        emit_reduce_insn_ub_split_on_reduce_axis()

        # Reduce
        res_tensor = self.reduce_info.reduce_tensor
        insn = INSN_MAPPING.get(get_dsl_insn(res_tensor), get_dsl_insn(res_tensor))
        extra_space = self.tiling_case.tensor_ub_size_before_reduce

        def emit_reduce_insn_ub_split_on_reduce_axis():
            # when do ub split last R axis,rf tensor has been do rfactor(rf.rf) after rf tensor tiling,
            # rf factor will disable tiling variable
            if not self._last_reduction_rf_optimization():
                self.emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                        "instruction": insn,
                                                        "extra_space": extra_space}
            else:
                if self.ub_tiling_result_pair[1] != self.reduce_info.reduce_axis_indexes[-1]:
                    reduce_rfs_rfs_scope = res_ub_inner
                else:
                    reduce_rfs_rfs_scope = self.schedule[self.reduce_rfs_rfs].op.axis[-1]

                self.emit_insn_map[self.reduce_rfs_rfs] = {"scope": reduce_rfs_rfs_scope,
                                                           "instruction": insn}
                self.emit_insn_map[self.reduce_rfs] = {
                    "scope": self.schedule[self.reduce_rfs].op.reduce_axis[-1],
                    "instruction": insn,
                    "extra_space": extra_space}

        if self._reduce_case == ReduceCategory.NOT_LAST_REDUCE:
            # for shape (r4,a4,r3,a3,r2,a2,r1,a1),
            # the ir order (a4,a3,a2,r4,r3,r2,r1,a1)
            # if ub split a2,a3 or a4, emit insn should target at r4
            # when a1 is continuous
            a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            if a1_end_index is None:
                dict_args = {"errCode": "E90001", "detailed_cause": "a1_end_index can not be none!"}
                raise RuntimeError(dict_args, get_error_message(dict_args))
            if ub_split_axis < a1_start_index and \
                    ub_split_axis not in reduce_axis_index:
                res_ub_inner = ub_tiling_tensor.op.reduce_axis[-1]

            self.emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                    "instruction": insn,
                                                    "extra_space": extra_space}
        elif self.reduce_info.is_reduce_last_axis():
            # ub cut ak (none reduce axis),
            if ub_split_axis not in reduce_axis_index:
                self.emit_insn_map[ub_tiling_tensor] = {
                    "scope": ub_tiling_tensor.op.reduce_axis[-1],
                    "instruction": insn,
                    "extra_space": extra_space}
            else:
                emit_reduce_insn_ub_split_on_reduce_axis()
        else:
            self.emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                    "instruction": insn,
                                                    "extra_space": extra_space}

        if self.remove_pad_tensor is not None:
            self.emit_insn_map[self.remove_pad_tensor] = {
            "scope": self.schedule[self.remove_pad_tensor].op.axis[0],
            "instruction": 'remove_pad'}

        self.emit_insn_map[self.reduce_repls] = {
            "scope": self.reduce_repls.op.axis[0],
            "instruction": 'dma_copy'}
        self.emit_insn_map[res_tensor] = {
            "scope": self.schedule[res_tensor].op.axis[0],
            "instruction": 'phony_insn'}

    def _calculate_db(self):
        if not self.tiling_case.db:
            return
        for _tensor in self.cache_read_tensors_and_buffer_map:
            _target = self.cache_read_tensors_and_buffer_map[_tensor]
            self.double_buffer_tensors.append(_target)

        for tensor in self.graph_info.mid_tensor_set:
            if tensor not in self.graph_info.real_output_tensor_set:
                self.double_buffer_tensors.append(tensor)

        if hasattr(self.reduce_rfs, "index"):
            self.double_buffer_tensors.extend(self.reduce_rfs)
        else:
            self.double_buffer_tensors.append(self.reduce_rfs)

    def _replace_outs(self, outs):
        ori_outs = copy.copy(outs)
        outs.clear()
        for ori_tensor in ori_outs:
            outs.append(self.cache_write_tensors_and_buffer_map[ori_tensor])

    def _reorder_reduce_nlast_shape(self, shape_before_reduce,
                                    reduce_axis_index):
        """
        reorder shape (r4,a4,r3,a3,r2,a2,r1,a1) to (a4,a3,a2, r4,r3,r2,,r1,a1)
        :param shape_before_reduce: like (r4,a4,r3,a3,r2,a2,r1,a1)
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce: (ak+1,rk,..,r2,a2,r1,a1)
        # find the last none-reduce axis a1

        a1_start_index, _ = self._find_last_none_reduce_axis(
            shape_before_reduce, reduce_axis_index)

        last_none_reduce_axis = a1_start_index

        orignal_to_reorder_axis_map = {}
        reorder_to_orignal_axis_map = {}
        #  (ak+1,ak,...,a2, rk,..,r2,,r1,a1)
        reordered_shape = list(shape_before_reduce)
        temp_axis = last_none_reduce_axis - 1
        for i in range(len(reduce_axis_index) - 1, -1, -1):
            reordered_shape[temp_axis] = shape_before_reduce[
                reduce_axis_index[i]]
            reorder_to_orignal_axis_map[temp_axis] = reduce_axis_index[i]
            orignal_to_reorder_axis_map[reduce_axis_index[i]] = temp_axis
            temp_axis = temp_axis - 1
        for i in range(last_none_reduce_axis - 1, -1, -1):
            if i not in reduce_axis_index:
                reordered_shape[temp_axis] = shape_before_reduce[i]
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis - 1

        for i in range(last_none_reduce_axis, len(shape_before_reduce)):
            reorder_to_orignal_axis_map[i] = i
            orignal_to_reorder_axis_map[i] = i

        return reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map

    def _calculate_pragma(self):
        """
        Determine continues axises
        """
        # For rf compute mode, skip pragma stage
        if self._last_reduction_rf_optimization():
            return

        # create virtual mapping
        _shape = [Dim(R, _idx) if _idx in self.reduce_info.reduce_axis_indexes else Dim(A, _idx)
                  for _idx in range(len(self.reduce_info.shape_before_reduce))]

        # split
        blk_split_idx, ub_split_idx = self.block_tiling_result_pair[1], self.ub_tiling_result_pair[1]
        if blk_split_idx <= ub_split_idx:
            Dim.split(_shape, blk_split_idx)
            ub_split_idx += 1
            Dim.split(_shape, ub_split_idx, model="UBSplit")
        else:
            Dim.split(_shape, ub_split_idx, model="UBSplit")
            blk_split_idx += 1
            Dim.split(_shape, blk_split_idx, )

        # rfactor
        axis = [item for item in _shape[: blk_split_idx + 1] if item.axis_type == R]
        _a_shape, _r_shape = Dim.rfactor(_shape, axis, factor_axis=0)

        # reorder
        if self._reduce_case == ReduceCategory.NOT_LAST_REDUCE:
            _r_shape.append(_a_shape.pop(-1))
        _rf_shape = _a_shape + _r_shape

        # find serial axis
        idx_ub_outer = _rf_shape.index(_shape[ub_split_idx])
        axis_in_ub = _rf_shape[idx_ub_outer + 1:]
        axis_in_ub.sort(key=lambda x: x.idx, reverse=True)
        self._serial_group = Dim.group([x.idx for x in axis_in_ub])
        self._serial_group.sort(key=lambda x: x[1] - x[0], reverse=True)

    def _do_pragma(self):
        # For rf compute mode, skip pragma stage
        if self._last_reduction_rf_optimization():
            return

        # Initialization
        reduce_ub_tensor = self.reduce_rfs
        before_reduce_tensors = self.get_all_producers_stages(reduce_ub_tensor)
        ub_tiling_on_reduce_axis = self.ub_tiling_result_pair[1] in self.reduce_info.reduce_axis_indexes
        # For tensor before reduce, try to fuse all continuous reordered axis, Search in reversed order.
        for tensor in before_reduce_tensors:
            stage = self.schedule[tensor]
            # Do not pragma placeholder
            if tensor in self.graph_info.input_tensor_set:
                continue

            if tensor in self._align_pad_tensor_list:
                continue

            # For ub tensor
            if get_dsl_insn(tensor) != "":
                # Iterate all axis after ub_tiling_axis and check if they need to be in the axis_group
                for axis_idx in range(self.ub_tiling_result_pair[1],
                                      len(self.reduce_info.shape_before_reduce) - 1):
                    if axis_idx in self.reduce_info.reduce_axis_indexes or not ub_tiling_on_reduce_axis:
                        stage.pragma(stage.op.axis[axis_idx], "axis_group", 0)
                # last axis needs to be in the axis_group
                stage.pragma(stage.op.axis[len(self.reduce_info.shape_before_reduce) - 1], "axis_group", 0)
            else:
                # For dma tensor
                extend = self._serial_group[0][1] - self._serial_group[0][0] + 1
                length = len(self.reduce_info.shape_before_reduce)
                axis_range = \
                    range(length - 1, length - 1 - extend, -1) if extend != 1 else range(length - 1, length - 3, -1)

                for axis_idx in range(self.ub_tiling_result_pair[1],
                                      len(self.reduce_info.shape_before_reduce)):
                    if axis_idx in axis_range:
                        stage.pragma(stage.op.axis[axis_idx], "axis_group", 0)

    def _need_storage_align(self):
        """
        :return:
        """
        ub_split_axis = self.ub_tiling_result_pair[1]
        shape_before_reduce = self.reduce_info.shape_before_reduce
        reduce_axis_index = self.reduce_info.reduce_axis_indexes

        # for shape(r4,a4,r3,a3,r2,a2,r1,a1), if ub split a1, do not need storage_align
        if self._reduce_case == ReduceCategory.NOT_LAST_REDUCE:
            a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            if a1_end_index is None:
                return False
            if a1_start_index <= ub_split_axis <= a1_end_index:
                return False

        elif self._reduce_case == ReduceCategory.LAST_REDUCE:
            r1_start_index, r1_end_index = self._find_last_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            if r1_end_index is None:
                return False
            # for shape(a4,r4,a3,r3,a2,r2,a1,r1), if ub split r1, do not need storage_align
            if r1_start_index <= ub_split_axis <= r1_end_index:
                return False
        else:
            return False

        return True

    def _reorder_atomic_reduce_all(self, out_tensor_ub_rf, out_tensor_global):

        block_split_axis = self.block_tiling_result_pair[1]
        ub_split_axis = self.ub_tiling_result_pair[1]
        res_ub_outer = self.iter_ub_outer
        res_ub_inner = self.iter_ub_inner

        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self.schedule[out_tensor_global].reorder(*global_reordered_axis_list)

        ub_rf_reordered_axis_list = [out_tensor_ub_rf.op.axis[-1 + self._axis_offset]]

        reduce_block_axis = self.reduce_info.reduce_axis_indexes.index(block_split_axis)
        reduce_ub_axis = self.reduce_info.reduce_axis_indexes.index(ub_split_axis)

        if block_split_axis != ub_split_axis:
            # rbi
            ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])
        for i in range(reduce_block_axis, reduce_ub_axis - 1):
            reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
            ub_rf_reordered_axis_list.append(reduce_axis)

        ub_rf_reordered_axis_list.append(res_ub_outer)
        ub_rf_reordered_axis_list.append(res_ub_inner)
        for i in range(reduce_ub_axis,
                       len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
            reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
            ub_rf_reordered_axis_list.append(reduce_axis)

        self.schedule[out_tensor_ub_rf].reorder(*ub_rf_reordered_axis_list)

    def _reorder_atomic_reduce_not_last_axis(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        # reorder (ak+1,ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak+1,ak,..a2,rk,.,rb-1,rbi,rb+1,..r2,r1,a1) or
        # (rbo_fused, ak,..a2,rbi,rb+1,..r2,r1,a1) if need fused
        block_split_axis = self.block_tiling_result_pair[1]
        ub_split_axis = self.ub_tiling_result_pair[1]
        res_ub_outer = self.iter_ub_outer
        res_ub_inner = self.iter_ub_inner

        # rbo axis of out_tensor_global
        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)
        self.schedule[out_tensor_global].reorder(*global_reordered_axis_list)

        shape_before_reduce = self.reduce_info.shape_before_reduce
        reduce_axis_index = self.reduce_info.reduce_axis_indexes
        is_keep_dim = self.reduce_info.keepdims
        a1_start_index, a1_end_index = self._find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)
        none_reduce_index_map = self._find_none_reduce_axis_map(shape_before_reduce, reduce_axis_index, is_keep_dim)

        # for shape (r4,a4,r3,a3,r2,a2,r1,a1),
        # reorder ir (ak+1,ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo_fused, ak,..a2,rbi,rb-1,..r2,r1,a1)
        # append rbo_fused
        ub_rf_reordered_axis_list = [out_tensor_ub_rf.op.axis[-1 + self._axis_offset]]

        def __reorder_case_1(ub_rf_reordered_axis_list):
            # add axis (ak,..a2)
            for i in range(0, a1_start_index):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self.schedule[out_tensor_ub_rf].op.axis[
                            none_reduce_index + self._axis_offset])

            # add a1 outer, a1 may be continuous
            for i in range(a1_start_index, ub_split_axis):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(
                    self.schedule[out_tensor_ub_rf].op.axis[
                        none_reduce_index + self._axis_offset])
            ub_rf_reordered_axis_list.append(res_ub_outer)

            # add rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            for i in range(0, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # add a1 inner, a1 may be continuous
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(ub_split_axis + 1, a1_end_index + 1):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(
                    self.schedule[out_tensor_ub_rf].op.axis[
                        none_reduce_index + self._axis_offset])

            self.schedule[out_tensor_ub_rf].reorder(*ub_rf_reordered_axis_list)

        def __reorder_case_2(ub_rf_reordered_axis_list):
            # add axis (ak,..a2)
            for i in range(0, a1_start_index):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self.schedule[out_tensor_ub_rf].op.axis[
                            none_reduce_index + self._axis_offset])

            # add rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            reduce_ub_axis = reduce_axis_index.index(ub_split_axis)
            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(reduce_ub_axis,
                           len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # add a1
            for i in range(a1_start_index, a1_end_index + 1):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(
                    self.schedule[out_tensor_ub_rf].op.axis[
                        none_reduce_index + self._axis_offset])
            self.schedule[out_tensor_ub_rf].reorder(*ub_rf_reordered_axis_list)

        def __reorder_case_3(ub_rf_reordered_axis_list):
            # add axis (ak,..a2)
            for i in range(0, ub_split_axis - 1):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self.schedule[out_tensor_ub_rf].op.axis[
                            none_reduce_index + self._axis_offset])
            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(ub_split_axis + 1, a1_start_index):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self.schedule[out_tensor_ub_rf].op.axis[
                            none_reduce_index + self._axis_offset])

            # add rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            for i in range(0, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # add a1
            for i in range(a1_start_index, a1_end_index + 1):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(
                    self.schedule[out_tensor_ub_rf].op.axis[
                        none_reduce_index + self._axis_offset])

            self.schedule[out_tensor_ub_rf].reorder(*ub_rf_reordered_axis_list)

        # if ub split axis in(a1)
        reduce_block_axis = reduce_axis_index.index(block_split_axis)
        if a1_start_index <= ub_split_axis <= a1_end_index:
            __reorder_case_1(ub_rf_reordered_axis_list)
            return

        # if ub split axis in (rbi,rb-1,..r2,r1)
        if ub_split_axis in reduce_axis_index:
            __reorder_case_2(ub_rf_reordered_axis_list)
            return

        # if ub split axis in (ak,..a2)
        if ub_split_axis not in reduce_axis_index:
            __reorder_case_3(ub_rf_reordered_axis_list)

    def _reorder_atomic_reduce_last_axis(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        # for shape (a4,r4,a3,r3,a2,r2,a1,r1),
        # reorder (ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak,..a2,rk,.,rb-1,rbi,rb+1,..a1,r1) or
        # (rbo_fused, ak,..a2,rbi,rb+1,..a1,r1) if need fused
        block_split_axis = self.block_tiling_result_pair[1]
        ub_split_axis = self.ub_tiling_result_pair[1]
        res_ub_outer = self.iter_ub_outer
        res_ub_inner = self.iter_ub_inner

        shape_before_reduce = self.reduce_info.shape_before_reduce
        reduce_axis_index = self.reduce_info.reduce_axis_indexes
        is_keep_dim = self.reduce_info.keepdims

        # reorder ir (ak,..a2,a1,rbo) to (rbo,ak,..a2,a1)
        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self.schedule[out_tensor_global].reorder(*global_reordered_axis_list)

        none_reduce_index_map = self._find_none_reduce_axis_map(
            shape_before_reduce, reduce_axis_index, is_keep_dim)

        # 'reorder (ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak,..a2,a1, rk,.,rb-1,rbi,rb+1,..r2,r1) or
        # (rbo_fused, ak,..a2,a1, rbi,rb+1,..r2,r1) if need fused
        # rbo
        ub_rf_reordered_axis_list = [self.schedule[out_tensor_ub_rf].op.axis[-1 + self._axis_offset]]

        def reorder_rf_tensor_ub_split_on_reduce_axis():
            # add axis (ak,..a2,a1)
            for i in range(0, len(self.schedule[out_tensor_ub_rf].op.axis) - 1):
                ub_rf_reordered_axis_list.append(
                    self.schedule[out_tensor_ub_rf].op.axis[i + self._axis_offset])

            if self._last_reduction_rf_optimization():
                # for last reduce axis optimization rf tensor left only one reduce axis after rf.rf tensor's reduce
                ub_rf_reordered_axis_list.append(self.schedule[out_tensor_ub_rf].op.reduce_axis[-1])
            else:
                # 'append rbi
                if block_split_axis != ub_split_axis:
                    ub_rf_reordered_axis_list.append(self.schedule[out_tensor_ub_rf].op.reduce_axis[-1])

                # add axis (rb-1,..r2,r1)
                reduce_block_axis = reduce_axis_index.index(block_split_axis)
                reduce_ub_axis = reduce_axis_index.index(ub_split_axis)
                for i in range(reduce_block_axis, reduce_ub_axis - 1):
                    reduce_axis_before_ub_axis = self.schedule[out_tensor_ub_rf].op.reduce_axis[i - reduce_block_axis]
                    ub_rf_reordered_axis_list.append(reduce_axis_before_ub_axis)

                ub_rf_reordered_axis_list.append(res_ub_outer)
                ub_rf_reordered_axis_list.append(res_ub_inner)

                for i in range(reduce_ub_axis,
                               len(self.schedule[out_tensor_ub_rf].op.reduce_axis) + reduce_block_axis - 1):
                    reduce_axis_after_ub_axis = self.schedule[out_tensor_ub_rf].op.reduce_axis[i - reduce_block_axis]
                    ub_rf_reordered_axis_list.append(reduce_axis_after_ub_axis)

        # 'if ub split axis in (rbi,rb+1,..r2,r1)
        if ub_split_axis in reduce_axis_index:
            reorder_rf_tensor_ub_split_on_reduce_axis()
        # 'if ub split axis in (ak,..a2,a1)
        else:
            # add axis (ak,..a2,a1)
            for i in range(0, ub_split_axis - 1):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self.schedule[out_tensor_ub_rf].op.axis[none_reduce_index + self._axis_offset])

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)

            for i in range(ub_split_axis + 1, len(shape_before_reduce)):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self.schedule[out_tensor_ub_rf].op.axis[none_reduce_index + self._axis_offset])

            # 'rbi
            ub_rf_reordered_axis_list.append(self.schedule[out_tensor_ub_rf].op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            for i in range(0, len(self.schedule[out_tensor_ub_rf].op.reduce_axis) - 1):
                reduce_axis = self.schedule[out_tensor_ub_rf].op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

        self.schedule[out_tensor_ub_rf].reorder(*ub_rf_reordered_axis_list)

        if self._last_reduction_rf_optimization():
            self._do_rf_rf_tensor_reorder()

    def _do_rf_rf_tensor_reorder(self):
        """
        do reorder for rf.rf tensor
        """
        block_split_axis = self.block_tiling_result_pair[1]
        ub_split_axis = self.ub_tiling_result_pair[1]
        res_ub_outer = self.iter_ub_outer
        res_ub_inner = self.iter_ub_inner
        reduce_axis_index = self.reduce_info.reduce_axis_indexes
        reduce_rfs_rfs = self.reduce_rfs_rfs

        ub_rf_reordered_axis_list = []
        # only add all A axis include first axis rbo ,but except last A axis
        for i in range(0, len(self.schedule[reduce_rfs_rfs].op.axis) - 1):
            ub_rf_reordered_axis_list.append(
                self.schedule[reduce_rfs_rfs].op.axis[i])

        reduce_block_axis = reduce_axis_index.index(block_split_axis)
        reduce_ub_axis = reduce_axis_index.index(ub_split_axis)

        # the order will be rbo ak ak-1 a2 a1 rbi rb-1 rb-2 ruo+1 ruo rui ru-1 ru-2 r2 r1 rf_rf
        if self.ub_tiling_result_pair[1] != self.reduce_info.reduce_axis_indexes[-1]:
            # append rbi when split on rf_rf tensor rbi is reduce_axis -1
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(self.schedule[reduce_rfs_rfs].op.reduce_axis[-1])

            # append r axis between rbi and ruo(rbi rb-1 rb-2 ruo+1)
            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = self.schedule[reduce_rfs_rfs].op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # append ruo rui
            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)

            # append r axis after rui (rui-1 rui-2 r2 r1)
            for i in range(reduce_ub_axis,
                           len(self.schedule[reduce_rfs_rfs].op.reduce_axis) + reduce_block_axis - 1):
                reduce_axis = self.schedule[reduce_rfs_rfs].op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)
        else:
            # split on rf tensor,rf.rf tensor not the ub split tensor
            # so add all axis after block split axis
            # as split at last R axis, so k2.outer and k2.inner will be follow the order of last place of tensor.
            for i in range(reduce_block_axis,
                           len(self.schedule[reduce_rfs_rfs].op.reduce_axis) + reduce_block_axis):
                reduce_axis = self.schedule[reduce_rfs_rfs].op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

        # append rf_rf axis
        ub_rf_reordered_axis_list.append(self.schedule[reduce_rfs_rfs].op.axis[-1])

        self.schedule[reduce_rfs_rfs].reorder(*ub_rf_reordered_axis_list)

    @staticmethod
    def _find_none_reduce_axis_map(shape_before_reduce, reduce_axis_index, keep_dims):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        none_reduce_index_map = {}
        if keep_dims:
            for i in range(0, len(shape_before_reduce)):
                if i not in reduce_axis_index:
                    none_reduce_index_map[i] = i
        else:
            count = 0
            for i in range(0, len(shape_before_reduce)):
                if i not in reduce_axis_index:
                    none_reduce_index_map[i] = count
                    count += 1

        return none_reduce_index_map

    @staticmethod
    def _find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce:(ak+1,rk,..,r2,a2,r1,a1) or (ak,rk,..,r2,a1,r1),
        # find a1 position, a1 may contain continues axis
        a1_end_index = None
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if i not in reduce_axis_index:
                a1_end_index = i
                break
        a1_start_index = a1_end_index
        if a1_end_index is None:
            return a1_start_index, a1_end_index
        for i in range(a1_end_index, -1, -1):
            if i in reduce_axis_index:
                a1_start_index = i + 1
                break
            if i == 0:
                a1_start_index = i

        return a1_start_index, a1_end_index

    @staticmethod
    def _find_last_reduce_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce:(ak+1,rk,..,r2,a2,r1,a1) or (ak,rk,..,r2,a1,r1),
        # find r1 position, r1 may contain continues axis
        r1_end_index = None
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if i in reduce_axis_index:
                r1_end_index = i
                break
        r1_start_index = r1_end_index
        if r1_end_index is None:
            return r1_start_index, r1_end_index
        for i in range(r1_end_index, -1, -1):
            if i not in reduce_axis_index:
                r1_start_index = i + 1
                break
            if i == 0:
                r1_start_index = i

        return r1_start_index, r1_end_index

    @staticmethod
    def _reorder_reduce_last_shape(shape_before_reduce,
                                   reduce_axis_index):
        """
        reorder shape (a4,r4,a3,r3,a2,r2,a1,r1) to (a4,a3,a2,a1,r4,r3,r2,,r1)
        :param shape_before_reduce: like (a4,r4,a3,r3,a2,r2,a1,r1)
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce: (a4,r4,a3,r3,a2,r2,a1,r1)
        orignal_to_reorder_axis_map = {}
        reorder_to_orignal_axis_map = {}

        reordered_shape = []
        temp_axis = 0
        for i, ele in enumerate(shape_before_reduce):
            if i not in reduce_axis_index:
                reordered_shape.append(ele)
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis + 1

        for i, ele in enumerate(shape_before_reduce):
            if i in reduce_axis_index:
                reordered_shape.append(ele)
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis + 1

        return reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map

    def _last_reduction_rf_optimization(self):
        if self.reduce_info.is_reduce_last_axis() \
                and self.reduce_info.all_axes[self.tiling_case.ub_split_axis_index] \
                in self.reduce_info.reduce_axes:
            return True
        return False

    def is_ARA_1_0_case(self):
        ub_split_axis = self.ub_tiling_result_pair[1]
        block_split_axis = self.block_tiling_result_pair[1]
        if len(self._align_pad_tensor_list) != 0 and block_split_axis == 1 and ub_split_axis == 0:
            return True
        return False
