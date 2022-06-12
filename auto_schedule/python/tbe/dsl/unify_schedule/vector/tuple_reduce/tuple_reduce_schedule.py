#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
tuple reduce schedule
"""
# Standard Packages
from typing import List
# Third-Party Packages
from tbe.tvm.tensor import Tensor

# Local Packages
from tbe import tvm
from tbe.dsl.base.operation import var_inner
from ...constants import DTYPE_BYTE_MAPPING, Pattern
from ...constants import TupleReducePattern
from ...schedule import Schedule

# Tuple-Reduce Packages
from .tuple_reduce_tilingcase import TupleReduceTilingCase
from .tuple_reduce_tilingcase_info import Info
from . import tuple_reduce_schedule_helper

DEFAULT = "default"
CONST = "const"
DYNAMIC = "dynamic"

BLOCK_IDX = "blockIdx.x"
LOCAL_UB = "local.UB"
NO_OVERLAP = "no_overlap"
STORAGE_BOUND = "storage_bound"
ENABLE_VNCHWCONV = "enable_vnchwconv"


class EntryTupleReduceSchedule(Schedule):
    """
    Entry class for Tuple Reduce Schedule
    """

    def __init__(self, outs, tiling_case):
        self.outs = outs
        self.tiling_case: TupleReduceTilingCase = tiling_case

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.TUPLE_REDUCE]

    @classmethod
    def get_supported_sub_pattern(cls):
        return [TupleReducePattern.TR_0]

    def do_schedule(self):
        """
        Entry method of reduce schedule
        """
        outs, tiling_case = self.outs, self.tiling_case
        if tiling_case.schedule_type == tiling_case.ScheduleType.TIME_TILING:
            schedule = TupleReduceTimeTiling(outs, tiling_case)
        elif tiling_case.schedule_type == tiling_case.ScheduleType.SPATIAL_TILING:
            schedule = TupleReduceSpatialTiling(outs, tiling_case)

        real_schedule = schedule.do_schedule()
        real_schedule.tiling_key = schedule.tiling_key
        return real_schedule


class TupleReduceTimeTiling:
    """
    Tiling data in time dimension
    """

    def __init__(self, outs, tiling_case):
        self.outs: List[Tensor] = outs
        self.tiling_case: TupleReduceTilingCase = tiling_case
        self.info: Info = self.tiling_case.info
        self.tiling_key = self.tiling_case.tiling_key

        # SCHEDULE INFORMATION
        self.sch: tuple_reduce_schedule_helper.Schedule = tuple_reduce_schedule_helper.Schedule(self.outs)
        self.scope = LOCAL_UB
        self.info.atomic = 1

        self.reduce_stage = None
        self.reduce_rf_stage = None
        self.block_split_axis = 0
        self.block_tiling_axis = 0
        self.block_outer = 0
        self.block_inner = 0
        self.ub_split_axis = 0
        self.ub_tiling_axis = 0
        self.ub_outer = 0
        self.ub_inner = 0

        # align pad
        self.align_pad_stages = set()
        self.remove_pad_stages = set()

        # set switches
        self.info.buffer_size.switches.transpose_reduce = self.tiling_case.options == 1
        self.info.buffer_size.switches.align_pad = self.tiling_case.options == 2
        # buffer size options
        self.info.buffer_size.estimate(self.info.max_dtype_size)

        # reduce emit insn
        self.dichotomy_reduce_attr = {"reduce_opt_mode": "dichotomy_reduce",
                                      "storage_bound": self.info.buffer_size.grande_buffer_size // 2,
                                      "reuse_src_tensor": True,
                                      "nlast_reduce_dichotomy": 16}
        self.transpose_reduce_attr = {"trans": True}

    def do_schedule(self):
        self._cache_read()
        self._align_pad()
        self._reorder_reduce_stage()
        self._tiling_block()
        self._reorder_again()
        self._set_scope()
        self._compute_at()
        self._bind_block()
        self._time_tiling_emit_insn()
        self._storage_align()
        self._buffer_size()
        self._compute_inline()
        self._compute_root()
        self._mem_unique()

        return self.sch.sch

    def _cache_read(self):
        sch = self.sch
        for ph in sch.placeholder:
            consumers = sch.consumer(sch[ph])
            readers = [stage.origin_op for stage in consumers]
            sch.cache_read(ph, self.scope, readers)

    def _align_pad(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        if info.switches.align_pad:
            cache_read_stages_zero = list(sch.cache_read_stages)
            for stage in cache_read_stages_zero:
                consumers = sch.consumer(stage)
                readers = [stage.origin_op for stage in consumers]
                t = sch.cache_read(sch.get_tensor(stage), self.scope, readers)
                self.align_pad_stages.add(sch[t])

    def _reorder_reduce_stage(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        # Get reorder stage
        reduce_tensor = info.graph.reduce_tensor[0]
        reduce_stage = sch[reduce_tensor]

        # original order
        reduce_axis_one_hot = [1 if i in info.reduce_axis else 0 for i, _ in enumerate(info.max_shape)]
        origin_order = []
        for i, _ in enumerate(info.max_shape):
            if reduce_axis_one_hot[i]:
                origin_order.append(reduce_stage.op.reduce_axis[info.reduce_axis.index(i)])
            else:
                origin_order.append(reduce_stage.op.axis[i])

        # tiling order [A,...,A,R,...,R,*]
        tiling_order = []
        for i, _ in enumerate(info.max_shape[:-1]):
            if origin_order[i] in reduce_stage.op.axis:
                tiling_order.append(origin_order[i])
        for i, _ in enumerate(info.max_shape[:-1]):
            if origin_order[i] in reduce_stage.op.reduce_axis:
                tiling_order.append(origin_order[i])
        tiling_order.append(origin_order[-1])
        reduce_stage.reorder(*tiling_order)

    def _tiling_block(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        # Get tiling stage
        reduce_tensor = info.graph.reduce_tensor[0]
        reduce_stage = sch[reduce_tensor]

        # Get tiling axes index
        block_split_axis = case.block_axis
        ub_split_axis = case.ub_axis
        # Get tiling factors
        if case.info.is_const:
            block_factor = info.block_factor
            ub_factor = info.ub_factor
        else:
            block_factor = var_inner("_block_factor", (1, None))
            ub_factor = var_inner("_ub_factor", (1, None))

        # Get block tiling axis
        axis_idx = info.reduce_axis.index(block_split_axis)
        block_tiling_axis = reduce_stage.op.reduce_axis[axis_idx]
        if ub_split_axis in info.reduce_axis:
            axis_idx = info.reduce_axis.index(ub_split_axis)
            ub_tiling_axis = reduce_stage.op.reduce_axis[axis_idx]
        else:
            ub_tiling_axis = reduce_stage.op.axis[ub_split_axis]

        # block tiling
        block_outer, block_inner = reduce_stage.split(block_tiling_axis, factor=block_factor)
        # fuse all R before block tiling axis
        to_fuse_block_outer = []
        for axis in reduce_stage.op.reduce_axis:
            if axis == block_tiling_axis:
                break
            to_fuse_block_outer.append(axis)
        to_fuse_block_outer.append(block_outer)
        fused_block_outer = reduce_stage.fuse(*to_fuse_block_outer)

        # rfactor
        reduce_rf = sch.rfactor(reduce_tensor, fused_block_outer, 0)[0]
        reduce_rf_stage = sch[reduce_rf]
        reduce_stage = sch.get_stage(reduce_tensor)

        # ub_split on reduce_rf stage
        # find ub_tiling_axis in reduce_rf_stage
        if ub_split_axis == block_split_axis:
            ub_tiling_axis = block_inner
        for thisaxis in reduce_rf_stage.leaf_iter_vars:
            if thisaxis.var.name == ub_tiling_axis.var.name:
                ub_tiling_axis = thisaxis
                break
        # ub tiling
        ub_outer, ub_inner = reduce_rf_stage.split(ub_tiling_axis, factor=ub_factor)

        # save
        self.reduce_rf_stage, self.reduce_stage = reduce_rf_stage, reduce_stage
        self.block_split_axis, self.block_tiling_axis = block_split_axis, block_tiling_axis
        self.block_outer, self.block_inner = fused_block_outer, block_inner
        self.ub_split_axis, self.ub_tiling_axis = ub_split_axis, ub_tiling_axis
        self.ub_outer, self.ub_inner = ub_outer, ub_inner

    def _reorder_again(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        # Get reorder stages
        reduce_stage = self.reduce_stage
        reduce_rf_stage = self.reduce_rf_stage

        # Pivot block outer in reduce stage
        # by the definition of rfactor
        # there will be only one reduce axis in this stage
        gm_reduce_order = list(reduce_stage.op.reduce_axis) + list(reduce_stage.op.axis)
        reduce_stage.reorder(*gm_reduce_order)

        # Pivot rfactor in a proper order
        data_par_iter = sch.data_parallel_iteration(reduce_rf_stage)
        common_reduce_iter = sch.comm_reduce(reduce_rf_stage)
        common_reduce_iter = sorted(common_reduce_iter, key=lambda thisaxis: thisaxis.var.name.split('.')[0])

        if info.last_reduce:  # last reduce
            tiling_order = data_par_iter + common_reduce_iter
        else:  # nlast reduce
            tiling_order = data_par_iter[:-1] + common_reduce_iter + data_par_iter[-1:]
        reduce_rf_stage.reorder(*tiling_order)

    def _set_scope(self):
        self.sch.stages_not_on_ub.add(self.reduce_stage)
        for stage in self.sch.stages_on_ub:
            stage.set_scope(self.scope)

    def _compute_at(self):
        sch, reduce_stage, reduce_rf_stage = self.sch, self.reduce_stage, self.reduce_rf_stage
        stages_before_reduce_rf = sch.stages_on_ub.intersection(sch.poset(self.reduce_rf_stage))
        for stage in stages_before_reduce_rf:
            stage.compute_at(self.reduce_rf_stage, self.ub_outer)
        reduce_rf_stage.compute_at(reduce_stage, reduce_stage.op.reduce_axis[0])

    def _bind_block(self):
        reduce_stage, sch = self.reduce_stage, self.sch
        block = tvm.thread_axis(BLOCK_IDX)
        reduce_stage.bind(reduce_stage.op.reduce_axis[0], block)

    def _time_tiling_emit_insn(self):
        sch, info = self.sch, self.info

        # emit insn atomic add stage
        self.reduce_stage.emit_insn(self.reduce_stage.op.axis[0], "dma_copy")

        # get reduce mode
        if info.switches.transpose_reduce:
            reduce_attr = self.transpose_reduce_attr
        elif info.is_const:
            reduce_attr = self.dichotomy_reduce_attr
        else:
            reduce_attr = {}

        # get emit insn axis
        emit_axis = self.ub_inner
        if not info.last_reduce and self.ub_split_axis not in info.reduce_axis:
            emit_axis = sch.reduce_emit_axis(self.reduce_rf_stage, self.ub_inner)
        # emit insn reduce node
        self.reduce_rf_stage.emit_insn(emit_axis, "vector_reduce_sum", attrs=reduce_attr)

        # emit insn align pad stages
        for stage in self.align_pad_stages:
            stage.emit_insn(stage.op.axis[0], "align_pad", {"avoid_bank_conflict": True})

        # emit insn others
        for stage in self.sch.stages_on_ub - {self.reduce_rf_stage} - self.align_pad_stages:
            if stage in sch.cache_read_stages:
                stage.emit_insn(stage.op.axis[0], "dma_copy")
            else:
                stage.emit_insn(stage.op.axis[0], info.get_insn(stage))

    def _storage_align(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        if info.switches.transpose_reduce:
            return
        stages_before_reduce_rf = sch.stages_on_ub.intersection(sch.poset(self.reduce_rf_stage))
        storage_align_stages = stages_before_reduce_rf.union({self.reduce_rf_stage})
        for stage in self.align_pad_stages:
            storage_align_stages = storage_align_stages - sch.poset(stage)
        for stage in storage_align_stages:
            dtype_size = DTYPE_BYTE_MAPPING.get(sch.get_tensor(stage).dtype)
            stage.storage_align(stage.op.axis[-2], info.soc.block_size // dtype_size, 0)

    def _buffer_size(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        for stage in sch.stages_on_ub:
            if stage == self.reduce_rf_stage or \
                sch.cache_read_stages.get(stage) in info.buffer_size.short_tensors or \
                sch.get_ori_tensor(stage) in info.buffer_size.short_tensors:
                stage.set_buffer_size(info.buffer_size.short_buffer_size)
            else:
                stage.set_buffer_size(info.buffer_size.grande_buffer_size)

    def _compute_inline(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        for stage in sch.stages_on_ub.intersection(sch.broadcast_stages):
            if set(sch.consumer(stage)).intersection(sch.reduce_stages):
                continue
            if not sch.is_last_broadcast(stage) or sch.is_scalar_broadcast(stage):
                stage.compute_inline()

    def _compute_root(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        if info.switches.compute_root:
            for stage in sch.broadcast_branch.intersection(sch.stages_on_ub):
                stage.compute_root()

    def _mem_unique(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        if info.switches.mem_unique:
            for stage, tensor in sch.cache_read_stages.items():
                if tensor in info.buffer_size.unique_tensors:
                    stage.mem_unique()


class TupleReduceSpatialTiling:
    """
    Tiling data in spatial dimension
    """

    def __init__(self, outs, tiling_case):
        self.outs: List[Tensor] = outs
        self.tiling_case: TupleReduceTilingCase = tiling_case
        self.info: Info = self.tiling_case.info
        self.tiling_key = self.tiling_case.tiling_key

        # SCHEDULE INFORMATION
        self.sch: tuple_reduce_schedule_helper.Schedule = tuple_reduce_schedule_helper.Schedule(self.outs)
        self.scope = LOCAL_UB
        self.info.atomic = 0

        self.res = None
        self.reduce_stage = None
        self.block_split_axis = 0
        self.ub_split_axis = 0
        self.block_tiling_axis = 0
        self.ub_tiling_axis = 0
        self.block_outer = 0
        self.ub_outer = 0
        self.block_inner = 0
        self.ub_inner = 0

        # align pad
        self.align_pad_stages = set()
        self.remove_pad_stages = set()

        # set switches
        self.info.buffer_size.switches.transpose_reduce = self.tiling_case.options == 1
        self.info.buffer_size.switches.align_pad = self.tiling_case.options == 2
        # buffer size options
        self.info.buffer_size.estimate(self.info.max_dtype_size)

        # reduce emit insn
        self.dichotomy_reduce_attr = {"reduce_opt_mode": "dichotomy_reduce",
                                      "storage_bound": self.info.buffer_size.grande_buffer_size // 2,
                                      "reuse_src_tensor": True,
                                      "nlast_reduce_dichotomy": 16}
        self.transpose_reduce_attr = {"trans": True}

    def do_schedule(self):
        self._cache_read()
        self._align_pad()
        self._cache_write()
        self._remove_pad()
        self._set_scope()
        self._tiling()
        self._reorder()
        self._fuse()
        self._buffer_size()
        self._bind_block()
        self._compute_at()
        self._storage_align()
        self._spatial_tiling_emit_insn()
        self._spatial_tiling_compute_inline()
        self._compute_root()
        self._mem_unique()

        return self.sch.sch

    def _cache_read(self):
        sch = self.sch
        for ph in sch.placeholder:
            consumers = sch.consumer(sch[ph])
            readers = [stage.origin_op for stage in consumers]
            sch.cache_read(ph, self.scope, readers)

    def _align_pad(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        if info.switches.align_pad:
            cache_read_stages_zero = list(sch.cache_read_stages)
            for stage in cache_read_stages_zero:
                consumers = sch.consumer(stage)
                readers = [stage.origin_op for stage in consumers]
                t = sch.cache_read(sch.get_tensor(stage), self.scope, readers)
                self.align_pad_stages.add(sch[t])

    def _cache_write(self):
        self.sch.cache_write(self.sch.outs, self.scope)

    def _remove_pad(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        if info.switches.align_pad:
            t = self.sch.cache_write(self.sch.outs, self.scope)
            self.remove_pad_stages.add(sch[t[0]])

    def _set_scope(self):
        for stage in self.sch.stages_on_ub:
            stage.set_scope(self.scope)

    def _tiling(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        # Get tiling stage
        res = sch.stage_map[sch.real_outputs[0]]
        for stage in sch.postorder[::-1]:
            if stage in sch.stages_on_ub and stage.op.tag.find("reduce") != -1:
                reduce_stage = stage

        # Get tiling axes index
        block_split_axis = case.block_axis
        ub_split_axis = case.ub_axis
        # Get tiling factors
        if case.info.is_const:
            block_factor = info.block_factor
            ub_factor = info.ub_factor
        else:
            block_factor = var_inner("_block_factor", (1, None))
            ub_factor = var_inner("_ub_factor", (1, None))

        # Get block tiling axis
        block_tiling_axis = res.op.axis[block_split_axis]
        # block tiling
        block_outer, block_inner = res.split(
            block_tiling_axis, factor=block_factor)

        # Get ub tiling axis
        if ub_split_axis in info.reduce_axis:
            axis_idx = info.reduce_axis.index(ub_split_axis)
            ub_tiling_axis = reduce_stage.op.reduce_axis[axis_idx]
        else:
            ub_tiling_axis = reduce_stage.op.axis[ub_split_axis]
        # ub tiling
        ub_outer, ub_inner = reduce_stage.split(
            ub_tiling_axis, factor=ub_factor)

        # save
        self.res, self.reduce_stage = res, reduce_stage
        self.block_split_axis, self.block_tiling_axis = block_split_axis, block_tiling_axis
        self.ub_split_axis, self.ub_tiling_axis = ub_split_axis, ub_tiling_axis
        self.block_outer, self.block_inner = block_outer, block_inner
        self.ub_outer, self.ub_inner = ub_outer, ub_inner

    def _reorder(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        # Get reorder stage
        res, reduce_stage = self.res, self.reduce_stage

        # original order
        reduce_axis_one_hot = [
            1 if i in info.reduce_axis else 0 for i, _ in enumerate(info.max_shape)]
        origin_order = []
        for i, _ in enumerate(info.max_shape):
            if reduce_axis_one_hot[i]:
                origin_order.append(
                    reduce_stage.op.reduce_axis[info.reduce_axis.index(i)])
            else:
                origin_order.append(reduce_stage.op.axis[i])

        # tiling order [A,...,A,R,...,R,*]
        tiling_order = list(reduce_stage.leaf_iter_vars)
        if origin_order[-1] in tiling_order:
            tiling_order.remove(origin_order[-1])
            tiling_order.append(origin_order[-1])
        else:  # ub split last axis
            tiling_order.remove(self.ub_inner)
            tiling_order.append(self.ub_inner)
        reduce_stage.reorder(*tiling_order)

    def _fuse(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        # Get fuse stage
        res, reduce_stage = self.res, self.reduce_stage
        # Get fuse axes
        common_reduce_iter = sch.comm_reduce(reduce_stage)
        if self.ub_outer not in common_reduce_iter:
            return
        to_fuse_ub_outer = []
        for thisaxis in common_reduce_iter:
            to_fuse_ub_outer.append(thisaxis)
            if thisaxis == self.ub_outer:
                break
        fused_ub_outer = reduce_stage.fuse(*to_fuse_ub_outer)
        self.ub_outer = fused_ub_outer

    def _bind_block(self):
        block = tvm.thread_axis(BLOCK_IDX)
        self.res.bind(self.block_outer, block)

    def _compute_at(self):
        sch = self.sch
        stages_before_reduce = sch.stages_on_ub.intersection(sch.poset(self.reduce_stage))
        stages_between_reduce_and_res = sch.stages_on_ub.intersection(sch.poset(self.res)) - stages_before_reduce
        for stage in stages_between_reduce_and_res:
            stage.compute_at(self.res, self.block_outer)
        for stage in stages_before_reduce:
            stage.compute_at(self.reduce_stage, self.ub_outer)

    def _spatial_tiling_emit_insn(self):
        sch, info = self.sch, self.info
        # emit insn data move to gm
        self.res.emit_insn(self.block_inner, "dma_copy")

        # emit insn stages exclude reduce
        emit_insn_stages = sch.stages_on_ub - \
            {self.res, self.reduce_stage} - \
            self.align_pad_stages - self.remove_pad_stages
        for stage in emit_insn_stages:
            if stage in sch.cache_read_stages:
                stage.emit_insn(stage.op.axis[0], "dma_copy")
            else:
                stage.emit_insn(stage.op.axis[0], info.get_insn(stage))

        # align pad
        for stage in self.align_pad_stages:
            stage.emit_insn(stage.op.axis[0], "align_pad", {"avoid_bank_conflict": True})
        # remove pad
        for stage in self.remove_pad_stages:
            stage.emit_insn(stage.op.axis[0], "remove_pad", {"avoid_bank_conflict": True})

        # get reduce mode
        if info.switches.transpose_reduce:
            reduce_attr = self.transpose_reduce_attr
        elif info.is_const:
            reduce_attr = self.dichotomy_reduce_attr
        else:
            reduce_attr = {}

        # get emit insn axis
        emit_axis = self.ub_inner
        if not info.last_reduce:
            emit_axis = sch.reduce_emit_axis(self.reduce_stage, self.ub_inner)
        # emit insn reduce stage
        self.reduce_stage.emit_insn(emit_axis, "vector_reduce_sum", attrs=reduce_attr)

    def _buffer_size(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        for stage in sch.stages_on_ub:
            if stage == self.reduce_stage \
                    or sch.cache_read_stages.get(stage) in info.buffer_size.short_tensors \
                    or sch.get_ori_tensor(stage) in info.buffer_size.short_tensors:
                stage.set_buffer_size(info.buffer_size.short_buffer_size)
            else:
                stage.set_buffer_size(info.buffer_size.grande_buffer_size)

    def _storage_align(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        if info.switches.transpose_reduce:
            return
        # Get point stage
        res, reduce_stage = self.res, self.reduce_stage
        # Get storage align stage
        storage_align_stages = sch.stages_on_ub.intersection(sch.poset(reduce_stage))
        storage_align_stages = storage_align_stages.union({reduce_stage})
        # align pad do not need align
        for stage in self.align_pad_stages:
            storage_align_stages = storage_align_stages - sch.poset(stage)
        # stages after remove pad stages do not need align
        if self.remove_pad_stages:
            stages_before_remove_pad = set()
            for stage in self.remove_pad_stages:
                stages_before_remove_pad = stages_before_remove_pad.union(sch.poset(stage))
            stages_after_remove_pad = sch.stages_on_ub - stages_before_remove_pad - self.remove_pad_stages
            storage_align_stages = storage_align_stages - stages_after_remove_pad

        for stage in storage_align_stages:
            dtype_size = DTYPE_BYTE_MAPPING.get(sch.get_tensor(stage).dtype)
            stage.storage_align(stage.op.axis[-2], info.soc.block_size // dtype_size, 0)

    def _mem_unique(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        if info.switches.mem_unique:
            for stage, tensor in sch.cache_read_stages.items():
                if tensor in info.buffer_size.unique_tensors:
                    stage.mem_unique()

    def _spatial_tiling_compute_inline(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        for stage in sch.stages_on_ub:
            if stage.op.tag.find("broadcast") != -1 and not info.last_reduce:
                stage.compute_inline()

    def _compute_root(self):
        case, sch, info = self.tiling_case, self.sch, self.info
        if info.switches.compute_root:
            for stage in sch.broadcast_branch.intersection(sch.stages_on_ub):
                stage.compute_root()
