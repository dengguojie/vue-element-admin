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
BN TRAINING UPDATE GRAD   
"""
from __future__ import absolute_import
from __future__ import division
import math
import functools

from tbe.common.utils import shape_to_list
from tbe import tvm
from tbe.common.platform import platform_info as cce
from tbe.common.platform.platform_info import get_soc_spec
from . import util

# Standard Package
from typing import List
from typing import Union
from typing import NoReturn
from typing import Optional

from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base.operation import var 
from .constants import Pattern
from te.tvm.tensor import Tensor

from .bn_training_update_grad_tilingcase import BNTrainingUpdateGradTilingCase
from .bn_training_update_grad_tilingcase import BNTrainingUpdateGradInfo
from .bn_training_update_grad_tilingcase import ComputeGraphInfo
from .vector_schedule import VectorSchedule 
from tbe.dsl.base.expr_compare import expr_equal

CONST = "const"
FAKE_NODE_TAG = "elewise_empty_intrin"

RESNET_50_SHAPE_LIST = [
    [32, 1024 // 16, 7, 7, 16],
    [32, 1024 // 16, 14, 14, 16],
    [32, 128 // 16, 14, 14, 16],
    [32, 128 // 16, 28, 28, 16],
    [32, 128 // 16, 56, 56, 16],
    [32, 2048 // 16, 7, 7, 16],
    [32, 256 // 16, 14, 14, 16],
    [32, 256 // 16, 28, 28, 16],
    [32, 256 // 16, 56, 56, 16],
    [32, 256 // 16, 7, 7, 16],
    [32, 512 // 16, 14, 14, 16],
    [32, 512 // 16, 28, 28, 16],
    [32, 512 // 16, 7, 7, 16],
    [32, 64 // 16, 112, 112, 16],
    [32, 64 // 16, 28, 28, 16],
    [32, 64 // 16, 56, 56, 16],

    [32, 1, 224, 224, 16],
    [32, 4, 57, 57, 16],
    [32, 4, 112, 112, 16],
    [32, 8, 29, 29, 16],
    [32, 8, 57, 57, 16],
    [32, 16, 15, 15, 16],
    [32, 16, 29, 29, 16],
    [32, 16, 57, 57, 16],
    [32, 32, 15, 15, 16],
    [32, 32, 29, 29, 16],
    [32, 32, 8, 8, 16],
    [32, 64, 15, 15, 16],
]

DTYPE_WIDTH_MAP = {"float16": 1,
                   "float32": 2,
                   "int32": 2,
                   "int16": 1,
                   "uint16": 1,
                   "int8": 0.5,
                   "uint8": 0.5,
                   "bool": 0.5}

DTYPE_BYTE_MAPPING = {
    "uint1": 0.125,
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float16": 2,
    "int16": 2,
    "uint16": 2,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8
}


@register_schedule("BNTrainingUpdateGrad")
def schedule(outs, tiling_case: BNTrainingUpdateGradTilingCase):
    [outs].clear()

    graph_info = get_context().get_current_compute().get("compute_graph_info")
    bn_training_update_grad_info: BNTrainingUpdateGradInfo = get_context().get_current_compute().get("bn_training_update_grad_info")

    bn_training_update_grad_sch: BNUpdateGradSchedule = BNUpdateGradSchedule(graph_info, bn_training_update_grad_info)
    real_schedule = bn_training_update_grad_sch.do_schedule(outs, tiling_case)
    real_schedule.tiling_key = tiling_case.tiling_key

    return real_schedule


class BNUpdateGradSchedule():
    def __init__(self, graph_info: ComputeGraphInfo, bn_training_update_grad_info: BNTrainingUpdateGradInfo):
        self.bn_training_update_grad_info = bn_training_update_grad_info
        self.mid_tensor_set = tuple(graph_info.mid_tensor_set)
        self.output_tensor_set = tuple(graph_info.output_tensor_set)
        self.input_tensor_set = tuple(graph_info.input_tensor_set)
        self.broadcast_tensor_set = tuple(graph_info.broadcast_tensor_set)

        self.out = self.output_tensor_set[0]
        self.res = None

        self.schedule = None
        self.block_outer = None
        self.block_inner = None
        self.ub_outer = None
        self.ub_inner = None
        self.sum_x_block_outer = None
        self.sum_x_block_inner = None
        self.sum_x_ub_outer = None
        self.sum_x_ub_inner = None

        self.tensor_list = []
        self.sch_list = []
        self.input2dst_tensor_map = {}
        self.mid_tensor_dst_tensor_map = {}
        self.tensor_list_dst_tensor_map = {}
        self.visited_list = []
        self.input_broadcast_tensors = []
        self.broadcast_tensor_list = []

        self.input_tensor_buffer_map = {}
        self.mid_out_read_buffer_map = {}
        self.mid_tensor_buffer_map = {}
        self.broadcast_tensor_buffers = []
        self.cache_write_exclude_tensor = []
        self.final_out_list = []
        self.final_out_buffer_list = []

        self.shape_x = None
        self.dtype = None
        self.ub_size = get_soc_spec("UB_SIZE")

        self.storage_bound = None
        self.block_axis_is_reduce = False
        self.final_out_tensor_ub_rf = None
        self.final_out_tensor_global = None
        
    def _gen_reversed_subgraph_list(self):
        """traverse tensors by Depth-First-Search

        Parameters
        ----------
        out_tensor : tensor
            traverse tensors from this tensor,
            traversing its input tensors recursively.

        """

        stack = [self.out]

        while stack:
            cur_tensor = stack.pop()
            self.visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor not in self.visited_list:
                    stack.append(in_tensor)
                    self.tensor_list.append(in_tensor)

                    if in_tensor.op.tag.find("broadcast_for_tensor") != -1:
                        self.input_broadcast_tensors.append(cur_tensor)

                self._map_apend(self.tensor_list_dst_tensor_map, in_tensor, cur_tensor)
        
        final_out_tensor_list = list(self.output_tensor_set)

        input_x_tensor = None
        for tensor in self.input_tensor_set:
            if tensor.op.name == "x_input":
                input_x_tensor = tensor
        self.shape_x = shape_to_list(input_x_tensor.shape)

        self.dtype = input_x_tensor.dtype.lower()
        max_ub_count = self._get_max_ub_count(self.dtype, self.shape_x) // 2

        for tensor in self.tensor_list_dst_tensor_map:
            if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                self.input2dst_tensor_map[tensor] = list(set(self.tensor_list_dst_tensor_map[tensor]))
            else:
                self.mid_tensor_dst_tensor_map[tensor] = self.tensor_list_dst_tensor_map[tensor]
        
        self.broadcast_tensor_list = list(self.broadcast_tensor_set)
        self.input_broadcast_tensors = list(set(self.input_broadcast_tensors))

        for tensor in self.broadcast_tensor_list:
            if tensor.op.tag.find("broadcast_for_tensor") != -1:
                self.cache_write_exclude_tensor.append(tensor)

    def _map_apend(self, input_map, key, value):
        if input_map.get(key):
            if isinstance(value, list):
                for sub_v in value:
                    if sub_v not in input_map[key]:
                        input_map[key].append(sub_v)
            else:
                if value not in input_map[key]:
                    input_map[key].append(value)
        else:
            if isinstance(value, list):
                input_map[key] = value
            else:
                input_map[key] = [value]

    def _get_factors_of_positive_integer(self, value):
        factors = []
        if value <= 0:
            return factors
        sqrt_n = int(math.sqrt(value))
        for i in range(1, sqrt_n + 1, 1):
            if value % i == 0:
                tmp = value // i
                factors.append(i)
                if tmp != i:
                    factors.append(tmp)
        factors.sort()
        return factors

    def _find_closest_factor(self, factors, value):
        """
        find closest factor
        """
        if not factors:
            return None
        factors.sort()
        index = 0
        is_find = False
        for i in range(0, len(factors), 1):
            if factors[i] > value:
                index = i
                is_find = True
                break
        if is_find:
            if index > 0:
                index = index - 1
        else:
            index = len(factors) - 1

        closest_factor = factors[index]
        return closest_factor
    
    def _get_emit_insn_map(self, tensor):
        insn_map = {"elewise_single_cast": "vector_conv",
                    "elewise_single_VS_max": "vector_maxs",
                    "elewise_single_VS_min": "vector_mins",
                    "elewise_single_log": "vector_ln",
                    "elewise_single_exp": "vector_exp",
                    "elewise_single_relu": "vector_relu",
                    "elewise_single_abs": "vector_abs",
                    "elewise_single_not": "vector_not",
                    "elewise_single_sqrt": "vector_sqrt",
                    "elewise_single_rsqrt": "vector_rsqrt",
                    "elewise_binary_mul": "vector_mul",
                    "elewise_single_VS_mul": "vector_muls",
                    "elewise_binary_div": "vector_div",
                    "elewise_binary_add": "vector_add",
                    "elewise_single_VS_add": "vector_adds",
                    "elewise_binary_min": "vector_min",
                    "elewise_binary_max": "vector_max",
                    "elewise_binary_vcmpv_gt": "vector_gt",
                    "elewise_binary_vcmpv_ge": "vector_ge",
                    "elewise_binary_vcmpv_lt": "vector_lt",
                    "elewise_binary_vcmpv_le": "vector_le",
                    "elewise_binary_vcmpv_eq": "vector_eq",
                    "elewise_binary_vcmpv_ne": "vector_ne",
                    "elewise_binary_or": "vector_or",
                    "elewise_binary_and": "vector_and",
                    "elewise_multiple_mla": "vector_multiple",
                    "elewise_multiple_madd": "vector_multiple",
                    "elewise_multiple_maddrelu": "vector_multiple",
                    "broadcast": "vector_dup",
                    "elewise_binary_sub": "vector_sub",
                    "reduce_sum": "vector_reduce_sum",
                    "tuple_reduce_sum": "vector_reduce_sum"}
        if tensor.op.tag.find("|") != -1:
            str_list = tensor.op.tag.split("|")
            insn = insn_map.get(str_list[0])
        else:
            insn = insn_map.get(tensor.op.tag)
        return insn

    def _get_emit_insn_map_for_broadcast(self, tensor):
        insn_map = {"elewise_binary_mul": "vector_mul_with_broadcast",
                    "elewise_binary_div": "vector_div_with_broadcast",
                    "elewise_binary_add": "vector_add_with_broadcast",
                    "elewise_binary_sub": "vector_sub_with_broadcast",
                    }
        if tensor.op.tag not in insn_map:
            raise RuntimeError("Invalid tag of with broadcast vector instric!")

        return insn_map.get(tensor.op.tag)
    
    def _do_cache_read(self):
        """
        cache read
        """
        self.input_tensor_buffer_map = {}

        sch = self.sch_list[0]
        for tensor in self.input2dst_tensor_map:
            input_tensor = sch.cache_read(tensor, cce.scope_ubuf,
                                        self.input2dst_tensor_map[tensor])
            self.input_tensor_buffer_map[tensor] = input_tensor
        
        self.sch_list[0] = sch
    
    def _do_cache_write(self):
        """
        cache write
        """
        self.mid_tensor_buffer_map = {}
        self.broadcast_tensor_buffers = {}
        sch = self.sch_list[0]
        for tensor in self.mid_tensor_dst_tensor_map:
            buffer_tensor = sch.cache_write(tensor, cce.scope_ubuf)
            self.mid_tensor_buffer_map[tensor] = buffer_tensor

            if tensor in self.input_broadcast_tensors:
                self.broadcast_tensor_buffers.append(buffer_tensor)
        
        self.sch_list[0] = sch
    
    def _do_compute_inline(self):
        """
        compute inline
        """
        sch = self.sch_list[0]
        for tensor in self.mid_tensor_dst_tensor_map:
            if tensor not in self.output_tensor_set:
                sch[tensor].compute_inline()
        self.sch_list[0] = sch
    
    def _get_max_ub_count(self, dtype, shape):
        """
        caculate the max element num loaded in UB buffer
        :return: max element num loaded in UB buffer
        """
        # div 2 for align to fp16
        total_size = get_soc_spec("UB_SIZE") // 2
        dtype_size = DTYPE_WIDTH_MAP.get(dtype)
        total_size = total_size // dtype_size
        if shape not in RESNET_50_SHAPE_LIST:
            # div 2 for double buffer
            total_size = total_size // 2
            if dtype == "float16":
                total_width = 4.5
            else:
                total_width = 2.5
        else:
            key_shape = "_".join(str(i) for i in shape)
            if dtype == "float16":
                if RESNET_50_SPECIAL_MAX_UB_COUNT_FP16_MAP.get(key_shape):
                    total_width = \
                        RESNET_50_SPECIAL_MAX_UB_COUNT_FP16_MAP[key_shape]
                else:
                    total_width = 5.5
            else:
                if RESNET_50_SPECIAL_MAX_UB_COUNT_FP32_MAP.get(key_shape):
                    total_width = \
                        RESNET_50_SPECIAL_MAX_UB_COUNT_FP32_MAP[key_shape]
                else:
                    total_width = 3.5

        align_to = 128

        max_bound = total_width * align_to
        max_ub_count = int(total_size / max_bound * align_to)

        return max_ub_count

    def _do_storage_bound(self):
        UB_SIZE = self.ub_size
        mid_buffer_list = []
        input_buffer_list = []
        for tensor in self.mid_tensor_buffer_map:
            buffer_tensor = self.mid_tensor_buffer_map[tensor]
            mid_buffer_list.append(buffer_tensor)
        for tensor in self.input_tensor_buffer_map:
            buffer_tensor = self.input_tensor_buffer_map[tensor]
            input_buffer_list.append(buffer_tensor)
        tensor_storage_bound_set = set(mid_buffer_list) | set(input_buffer_list)
        # storage bound which is the maximum memory of UB allocated by the backend
        self.storage_bound = int(1024*7)
        add_compile_info("max_ub_count", self.storage_bound)
        for stage_tensor in tensor_storage_bound_set:
            self.schedule[stage_tensor].set_storage_bound(self.storage_bound)
        self.schedule[self.out].set_storage_bound(self.storage_bound)
    
    def _do_tiling(self):
        tiling_case: BNTrainingUpdateGradTilingCase = self.tiling_case
        if not isinstance(tiling_case, BNTrainingUpdateGradTilingCase):
            raise RuntimeError("BNTrainingUpdateGradTilingCase required for BNTrainingUpdateGradSchedule!")
        
        core_num = get_soc_spec("CORE_NUM")

        # Get tiling tensor
        final_out_tensor = self.out

        # Get tiling axes
        block_split_axis_index = tiling_case.block_split_axis_index
        ub_split_axis_index = tiling_case.ub_split_axis_index

        # Get tiling params
        block_factor = tiling_case.block_factor
        self.block_inner = block_factor if block_factor is not None else var("block_factor", (1, None))
        ub_factor = tiling_case.ub_factor
        self.ub_inner = ub_factor if ub_factor is not None else var("ub_factor", (1, None))
        self._need_multi_core = tiling_case.multi_core

        # block tiling
        if block_split_axis_index == 1:
            self.final_out_buffer_list = self.schedule.cache_write(self.output_tensor_set, cce.scope_ubuf)
            final_out_buffer = self.final_out_buffer_list[0]
            self.schedule[final_out_buffer].set_storage_bound(self.storage_bound)

            block_split_axis = final_out_tensor.op.axis[block_split_axis_index]
            self.sum_x_block_outer, self.sum_x_block_inner = self.schedule[final_out_tensor].split(block_split_axis, nparts=core_num)
            self.schedule[final_out_tensor].reorder(self.sum_x_block_outer, 
                                                    self.sum_x_block_inner,
                                                    final_out_tensor.op.axis[0],
                                                    final_out_tensor.op.axis[2],
                                                    final_out_tensor.op.axis[3],
                                                    final_out_tensor.op.axis[4])

            # ub tiling
            if ub_split_axis_index == 0:
                ub_split_axis = final_out_buffer.op.reduce_axis[0]
                self.sum_x_ub_outer, self.sum_x_ub_inner = self.schedule[final_out_buffer].split(ub_split_axis, factor=1)
                self.schedule[final_out_buffer].reorder(final_out_buffer.op.axis[0],
                                                        final_out_buffer.op.axis[1],
                                                        final_out_buffer.op.axis[2],
                                                        final_out_buffer.op.axis[3],
                                                        self.sum_x_ub_outer,
                                                        self.sum_x_ub_inner,
                                                        final_out_buffer.op.reduce_axis[1],
                                                        final_out_buffer.op.reduce_axis[2],
                                                        final_out_buffer.op.axis[4])
            elif ub_split_axis_index == 2:
                ub_split_axis = final_out_buffer.op.reduce_axis[1]
                self.sum_x_ub_outer, self.sum_x_ub_inner = self.schedule[final_out_buffer].split(ub_split_axis, factor=1)
                self.schedule[final_out_buffer].reorder(final_out_buffer.op.axis[0],
                                                        final_out_buffer.op.axis[1],
                                                        final_out_buffer.op.reduce_axis[0],
                                                        final_out_buffer.op.axis[2],
                                                        self.sum_x_ub_outer,
                                                        self.sum_x_ub_inner,
                                                        final_out_buffer.op.reduce_axis[2],
                                                        final_out_buffer.op.axis[3],
                                                        final_out_buffer.op.axis[4])
        else:
            self.block_axis_is_reduce = True
            self.sum_x_block_outer, self.sum_x_block_inner = self.schedule[final_out_tensor].split(final_out_tensor.op.reduce_axis[0], nparts=core_num)
            self.final_out_tensor_ub_rf, _ = self.schedule.rfactor(final_out_tensor, self.sum_x_block_outer)
            final_out_tensor_global_list = self.schedule.cache_write(self.output_tensor_set, "")

            final_tensors_index_res = []
            for tensor in self.output_tensor_set:
                for i, tensor_res in enumerate(self.res):
                    if tensor == tensor_res:
                        final_tensors_index_res.append(i)
                        break
            for i, tensor in enumerate(final_out_tensor_global_list):
                self.res[final_tensors_index_res[i]] = tensor
            
            self.schedule[self.final_out_tensor_ub_rf].set_storage_bound(self.storage_bound)
            self.schedule[self.final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)

            self.final_out_tensor_global = final_out_tensor_global_list[0]
            self.sum_x_ub_outer, self.sum_x_ub_inner = \
                self.schedule[self.final_out_tensor_ub_rf].split(
                self.final_out_tensor_ub_rf.op.reduce_axis[0],
                factor=self.ub_inner)

            self.schedule[self.final_out_tensor_global].set_storage_bound(self.storage_bound)
            self.schedule[self.final_out_tensor_global].reorder(
                self.final_out_tensor_global.op.reduce_axis[0],
                self.final_out_tensor_global.op.axis[0],
                self.final_out_tensor_global.op.axis[1],
                self.final_out_tensor_global.op.axis[2],
                self.final_out_tensor_global.op.axis[3],
                self.final_out_tensor_global.op.axis[4])

            self.schedule[self.final_out_tensor_ub_rf].reorder(
                self.final_out_tensor_ub_rf.op.axis[0],
                self.final_out_tensor_ub_rf.op.reduce_axis[2],
                self.final_out_tensor_ub_rf.op.axis[1],
                self.final_out_tensor_ub_rf.op.axis[2],
                self.final_out_tensor_ub_rf.op.axis[3],
                self.final_out_tensor_ub_rf.op.axis[4],
                self.sum_x_ub_outer, 
                self.sum_x_ub_inner,
                self.final_out_tensor_ub_rf.op.reduce_axis[1],
                self.final_out_tensor_ub_rf.op.axis[5])
    
    def _do_compute_at(self):
        """
        compute at
        """
        sch = self.sch_list[0]
        compute_at_axis = None
        final_out_tensor = None
        final_out_buffer = None

        if not self.block_axis_is_reduce:
            final_out_buffer = self.final_out_buffer_list[0]
            final_out_tensor = self.out
            compute_at_axis = self.sum_x_block_outer
        else:
            final_out_buffer = self.final_out_tensor_ub_rf
            final_out_tensor = self.final_out_tensor_global
            compute_at_axis = self.final_out_tensor_global.op.reduce_axis[0]

        for tensor in self.input_tensor_buffer_map:
            buffer_tensor = self.input_tensor_buffer_map[tensor]
            shape = shape_to_list(tensor.shape)
            if shape == self.shape_x:
                sch[buffer_tensor].compute_at(sch[final_out_buffer], self.sum_x_ub_outer)
            else:
                sch[buffer_tensor].compute_at(sch[final_out_tensor], compute_at_axis)
        
        for tensor in self.mid_tensor_buffer_map:
            buffer_tensor = self.mid_tensor_buffer_map[tensor]
            shape = shape_to_list(tensor.shape)
            if shape == self.shape_x:
                sch[buffer_tensor].compute_at(sch[final_out_buffer], self.sum_x_ub_outer)
            else:
                sch[buffer_tensor].compute_at(sch[final_out_tensor], compute_at_axis)

        if not self.block_axis_is_reduce:
            sch[final_out_buffer].compute_at(sch[self.out], self.sum_x_block_outer)
            block = tvm.thread_axis("blockIdx.x")
            sch[self.out].bind(self.sum_x_block_outer, block)
        else:
            sch[self.final_out_tensor_ub_rf].compute_at(
                sch[self.final_out_tensor_global],
                self.final_out_tensor_global.op.reduce_axis[0])
            sch[self.final_out_tensor_ub_rf].pragma(self.sum_x_ub_outer, "json_info_batchBindOnly", 1)
            block = tvm.thread_axis("blockIdx.x")
            sch[self.final_out_tensor_global].bind(self.final_out_tensor_global.op.reduce_axis[0], block)

        self.sch_list[0] = sch

    def _do_emit_insn(self):
        """
        emit insn
        """
        attrs = dict(storage_bound=[self.ub_size//7])
        sch = self.sch_list[0]
        for tensor in self.input_tensor_buffer_map:
            buffer_tensor = self.input_tensor_buffer_map[tensor]
            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")
        
        for tensor in self.mid_tensor_buffer_map:
            buffer_tensor = self.mid_tensor_buffer_map[tensor]
            if tensor.op.tag == "unified_broadcast":
                insn = "vector_broadcast"
                sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)
                continue
            insn = self._get_emit_insn_map(tensor)
            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)
        
        if not self.block_axis_is_reduce:
            final_out_buffer = self.final_out_buffer_list[0]
            insn = self._get_emit_insn_map(final_out_buffer)
            sch[final_out_buffer].emit_insn(final_out_buffer.op.axis[4], insn)
            sch[self.out].emit_insn(self.out.op.axis[4], "dma_copy")
        else:
            sch[self.final_out_tensor_ub_rf].emit_insn(
                self.sum_x_ub_inner, "vector_reduce_sum")
            sch[self.final_out_tensor_global].emit_insn(
                self.final_out_tensor_global.op.axis[1], "dma_copy")
            sch[self.out].emit_insn(sch[self.out].op.axis[0], "phony_insn")

        self.sch_list[0] = sch
    
    def _do_double_buffer(self):
        sch = self.sch_list[0]
        for tensor in self.input_tensor_buffer_map:
            shape = shape_to_list(tensor.shape)
            if shape == self.shape_x:
                buffer_tensor = self.input_tensor_buffer_map[tensor]
                sch[buffer_tensor].double_buffer()
                sch[buffer_tensor].preload()
                break
        
        self.sch_list[0] = sch
    
    def do_schedule(self, outs, tiling_case: BNTrainingUpdateGradTilingCase):

        self.res = outs

        self.schedule = tvm.create_schedule(self.out.op)

        self.sch_list = [self.schedule]
        self.tiling_case: BNTrainingUpdateGradTilingCase = tiling_case

        self._gen_reversed_subgraph_list()

        for tensor in self.mid_tensor_dst_tensor_map:
            self.schedule[tensor].set_scope(cce.scope_ubuf)

        self._do_cache_read()
        
        self._do_cache_write()

        self._do_storage_bound()

        self._do_tiling()

        self._do_compute_inline()
        
        self._do_compute_at()

        self._do_double_buffer()

        self._do_emit_insn()

        outs = self.res

        return self.schedule