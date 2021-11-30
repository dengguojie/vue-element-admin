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
transdata base schedule
"""

import abc
from tbe import tvm
from tbe.common.platform import scope_ubuf
from tbe.dsl.base import operation
from tbe.dsl.base.operation import get_context
from tbe.dsl.unify_schedule.schedule import Schedule
from tbe.dsl.unify_schedule.constants import Pattern
from .transdata_graph_info import ComputeGraphInfo

NO_OVERLAP = "no_overlap"
DEFAULT = "default"


class TransdataBaseSchedule(Schedule):
    """
    Class for transdata base schedule
    """

    def __init__(self, outs, tiling_case):
        self.outs = outs
        self.tiling_case = tiling_case
        self.graph_info = get_context().get_current_compute().get("_compute_graph_info")

        self.schedule = None
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

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.TRANSDATA]

    def _create_schedule(self):
        self.schedule = tvm.create_schedule([tensor.op for tensor in self.graph_info.output_tensor_set])

    def _do_cache_read(self):
        for tensor in self.graph_info.input_tensor_set:
            readers = self.graph_info.tensor_consumers_map[tensor]
            read_buffer = self.schedule.cache_read(tensor, scope_ubuf, readers)
            self.cache_read_tensors_and_buffer_map[tensor] = read_buffer
            for item in readers:
                self.update_stage(read_buffer, item, True)

    def _do_cache_write(self):
        for tensor in self.graph_info.output_tensor_set:
            writers = self.graph_info.tensor_producers_map[tensor]
            write_buffer = self.schedule.cache_write(tensor, scope_ubuf)
            self.cache_write_tensors_and_buffer_map[tensor] = write_buffer
            for item in writers:
                self.update_stage(write_buffer, item, False)

    def _do_set_scope(self):
        for tensor in self.graph_info.mid_tensor_set:
            if tensor not in self.graph_info.output_tensor_set:
                self.schedule[tensor].set_scope(scope_ubuf)

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
            if instruction in ["vector_transpose", ]:
                src_in_dst_order = self.emit_insn_map[stage].get("src_in_dst_order")
                self.schedule[stage].emit_insn(scope_iter_var, instruction,
                                               attrs=dict(src_in_dst_order=src_in_dst_order))
            elif instruction in ["remove_pad", ]:
                self.schedule[stage].emit_insn(scope_iter_var, instruction,
                                               attrs={"enough_buffer": False})
            elif instruction in ["align_pad", ]:
                attr_list = {"enough_buffer": False}
                pad_value = self.emit_insn_map[stage].get("pad_value")
                if pad_value:
                    attr_list.update(pad_value)
                self.schedule[stage].emit_insn(scope_iter_var, instruction,
                                               attrs=attr_list)
            else:
                self.schedule[stage].emit_insn(scope_iter_var, instruction,
                                               attrs=self.emit_insn_map[stage].get(NO_OVERLAP))

    def _do_double_buffer(self):
        for _tensor in self.double_buffer_tensors:
            self.schedule[_tensor].double_buffer()
        operation.add_build_arg("double_buffer_non_reuse", True)

    def update_stage(self, source_tensor, dst_tensor, before):
        """
        update graph stage map by new tensor
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
