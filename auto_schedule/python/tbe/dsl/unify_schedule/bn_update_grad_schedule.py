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
# storage bound which is the maximum memory of UB allocated by the backend
STORAGE_BOUND = int(7 * 1024)

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
    mode = get_context().get_current_compute().get("mode")

    if mode == CONST:
        bn_training_update_grad_sch: BNUpdateGradSchedule = BNUpdateGradSchedule(graph_info, bn_training_update_grad_info)
        real_schedule = bn_training_update_grad_sch.do_schedule(outs, tiling_case)
        real_schedule.tiling_key = 0
    else:
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

        self.mode = get_context().get_current_compute().get("mode")
        self.out = self.output_tensor_set[0]
        self.storage_bound = STORAGE_BOUND
        self.max_ub_count = STORAGE_BOUND
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

        self.block_axis_is_reduce = False
        self.final_out_tensor_ub_rf = None
        self.final_out_tensor_global = None
        self.is_do_double_buffer = True

        self.block_split_axis_index = None
        self.ub_split_axis_index = None
        self.ub_split_reduce_axis = None
        
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

                    if in_tensor.op.tag.find("unified_broadcast") != -1:
                        self.input_broadcast_tensors.append(cur_tensor)

                self._map_apend(self.tensor_list_dst_tensor_map, in_tensor, cur_tensor)
        
        final_out_tensor_list = list(self.output_tensor_set)

        input_x_tensor = None
        for tensor in self.input_tensor_set:
            if tensor.op.name == "x_input":
                input_x_tensor = tensor
        self.shape_x = shape_to_list(input_x_tensor.shape)

        self.dtype = input_x_tensor.dtype.lower()

        for tensor in self.tensor_list_dst_tensor_map:
            if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                self.input2dst_tensor_map[tensor] = list(set(self.tensor_list_dst_tensor_map[tensor]))
            else:
                self.mid_tensor_dst_tensor_map[tensor] = self.tensor_list_dst_tensor_map[tensor]
            if tensor.op.tag == "unified_broadcast":
                self.broadcast_tensor_list.append(tensor)
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
                    "tuple_reduce_sum": "vector_reduce_sum",
                    "unified_broadcast": "vector_broadcast"}
        if tensor.op.tag.find("|") != -1:
            str_list = tensor.op.tag.split("|")
            insn = insn_map.get(str_list[0])
        else:
            insn = insn_map.get(tensor.op.tag)
        return insn
    
    def _need_dichotomy_add(self, dtype, loop_size):
        if dtype == "float16":
            vector_inst_one_repeat_size = 128
        else:
            vector_inst_one_repeat_size = 64

        return loop_size > vector_inst_one_repeat_size
    
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
        sch = self.sch_list[0]
        for tensor in self.mid_tensor_dst_tensor_map:
            if tensor not in self.cache_write_exclude_tensor:
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
        add_compile_info("max_ub_count", self.storage_bound)
        for stage_tensor in tensor_storage_bound_set:
            self.schedule[stage_tensor].set_storage_bound(self.storage_bound)
        self.schedule[self.out].set_storage_bound(self.storage_bound)

    def _schedule_cut_c1(self):
        sch = self.sch_list[0]

        core_num = get_soc_spec("CORE_NUM")
        block_split_axis_index = self.block_split_axis_index
        ub_split_axis_index = self.ub_split_axis_index
        ub_split_reduce_axis = self.ub_split_reduce_axis

        self._do_cache_read()

        if self.mode == CONST:
            self.block_inner = self.tiling_case.block_factor
            self.ub_inner = self.tiling_case.ub_factor
        else:
            self.block_inner = var("block_factor", (1, None))
            self.ub_inner = var("ub_factor", (1, None))

        self._do_cache_write()

        self._do_compute_inline()

        final_out_tensor = self.out
        self.final_out_buffer_list = self.schedule.cache_write(self.output_tensor_set, cce.scope_ubuf)
        final_out_buffer = self.final_out_buffer_list[0]

        sum_x_c1_axis = final_out_tensor.op.axis[1]
        sum_x_c0_axis = final_out_tensor.op.axis[4]
        sum_x_ub_n_axis = final_out_buffer.op.axis[0]
        sum_x_ub_c1_axis = final_out_buffer.op.axis[1]
        sum_x_ub_h_axis = final_out_buffer.op.axis[2]
        sum_x_ub_w_axis = final_out_buffer.op.axis[3]
        sum_x_ub_c0_axis = final_out_buffer.op.axis[4]

        sum_x_ub_n_reduce_axis = final_out_buffer.op.reduce_axis[0]
        sum_x_ub_h_reduce_axis = final_out_buffer.op.reduce_axis[1]
        sum_x_ub_w_reduce_axis = final_out_buffer.op.reduce_axis[2]

        block_split_axis = final_out_tensor.op.axis[block_split_axis_index]
        self.sum_x_block_outer, self.sum_x_block_inner = self.schedule[final_out_tensor].split(block_split_axis, nparts=core_num)

        is_mean_bound = False
        is_need_mte3_opt = False
        if self.mode == CONST:
            c1_size = self.shape_x[1]
            c0_size = 16
            block_factor = (c1_size + core_num - 1) // core_num
            is_mean_bound = block_factor * c0_size * 2 > self.max_ub_count

            is_need_mte3_opt = \
                c1_size % core_num == 0 and c1_size // core_num > 1 and \
                not is_mean_bound
            
            if is_need_mte3_opt:
                sum_x_block_inner_outer, sum_x_block_inner_inner = \
                    sch[final_out_tensor].split(self.sum_x_block_inner, nparts=1)

        ub_split_axis = final_out_buffer.op.reduce_axis[ub_split_reduce_axis]
        self.sum_x_ub_outer, self.sum_x_ub_inner = self.schedule[final_out_buffer].split(ub_split_axis, factor=self.ub_inner)

        if ub_split_axis_index == 0:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                        sum_x_ub_c1_axis,
                                        sum_x_ub_h_axis,
                                        sum_x_ub_w_axis,
                                        self.sum_x_ub_outer,
                                        self.sum_x_ub_inner,
                                        sum_x_ub_h_reduce_axis,
                                        sum_x_ub_w_reduce_axis,
                                        sum_x_ub_c0_axis)
        elif ub_split_axis_index == 2:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                        sum_x_ub_c1_axis,
                                        sum_x_ub_n_reduce_axis,
                                        sum_x_ub_h_axis,
                                        self.sum_x_ub_outer,
                                        self.sum_x_ub_inner,
                                        sum_x_ub_w_axis,
                                        sum_x_ub_w_reduce_axis,
                                        sum_x_ub_c0_axis)
        else:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                        sum_x_ub_c1_axis,
                                        sum_x_ub_n_reduce_axis,
                                        sum_x_ub_h_axis,
                                        sum_x_ub_h_reduce_axis,
                                        sum_x_ub_w_axis,
                                        self.sum_x_ub_outer,
                                        self.sum_x_ub_inner,
                                        sum_x_ub_c0_axis)

        final_out_tensor = self.out
        final_out_buffer = self.final_out_buffer_list[0]
        compute_at_axis1 = self.sum_x_ub_outer
        compute_at_axis2 = self.sum_x_block_outer
        if is_mean_bound:
            if is_need_mte3_opt:
                compute_at_axis2 = sum_x_block_inner_inner
            else:
                compute_at_axis2 = self.sum_x_block_inner
        self._do_compute_at(final_out_tensor, final_out_buffer, compute_at_axis1, compute_at_axis2)

        if is_need_mte3_opt:
            sch[final_out_buffer].compute_at(sch[final_out_tensor], sum_x_block_inner_outer)
        else:
            sch[final_out_buffer].compute_at(sch[final_out_tensor], self.sum_x_block_inner)

        block = tvm.thread_axis("blockIdx.x")
        sch[self.out].bind(self.sum_x_block_outer, block)

        self._do_storage_bound()
        sch[final_out_buffer].set_storage_bound(self.storage_bound)

        if self.mode == CONST:
            if self.is_do_double_buffer:
                outer_loop = self.shape_x[ub_split_axis_index] // self.ub_inner
                if ub_split_axis_index == 3:
                    outer_loop = outer_loop * self.shape_x[2]
                
                self._do_const_double_buffer(outer_loop)

            sch[final_out_buffer].emit_insn(self.sum_x_ub_inner,
                                            "vector_reduce_sum")
            
            self._do_emit_insn()

            if is_need_mte3_opt:
                sch[final_out_tensor].emit_insn(sum_x_block_inner_inner, "dma_copy")
            else:
                sch[final_out_tensor].emit_insn(sum_x_c0_axis, "dma_copy")
        else:
            self._do_double_buffer()

            self._do_emit_insn()

            insn = self._get_emit_insn_map(final_out_buffer)
            sch[final_out_buffer].emit_insn(self.sum_x_ub_inner, insn)
            sch[self.out].emit_insn(self.out.op.axis[4], "dma_copy")
        self.sch_list[0] = sch
    
    def _schedule_cut_batch(self):
        sch = self.sch_list[0]

        core_num = get_soc_spec("CORE_NUM")
        block_split_axis_index = self.block_split_axis_index
        ub_split_axis_index = self.ub_split_axis_index
        ub_split_reduce_axis = self.ub_split_reduce_axis
        
        self._do_cache_read()
        
        final_out_tensor = self.out

        if self.mode == CONST:
            self.block_inner = self.tiling_case.block_factor
            self.ub_inner = self.tiling_case.ub_factor
        else:
            self.block_inner = var("block_factor", (1, None))
            self.ub_inner = var("ub_factor", (1, None))
        
        self._do_cache_write()
        
        self._do_compute_inline()

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
        
        self.schedule[self.final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)

        self.final_out_tensor_global = final_out_tensor_global_list[0]

        if ub_split_reduce_axis == 0:
            self.sum_x_ub_outer, self.sum_x_ub_inner = \
                self.schedule[self.final_out_tensor_ub_rf].split(
                self.final_out_tensor_ub_rf.op.reduce_axis[-1],
                factor=self.ub_inner)
        else:
            self.sum_x_ub_outer, self.sum_x_ub_inner = \
                self.schedule[self.final_out_tensor_ub_rf].split(
                self.final_out_tensor_ub_rf.op.reduce_axis[ub_split_reduce_axis-1],
                factor=self.ub_inner)
        
        sch[self.final_out_tensor_global].reorder(
            self.final_out_tensor_global.op.reduce_axis[0],
            self.final_out_tensor_global.op.axis[0],
            self.final_out_tensor_global.op.axis[1],  # C1 axis
            self.final_out_tensor_global.op.axis[2],
            self.final_out_tensor_global.op.axis[3],
            self.final_out_tensor_global.op.axis[4]  # C0 axis
        )
        if ub_split_reduce_axis == 0:
            sch[self.final_out_tensor_ub_rf].reorder(
                self.final_out_tensor_ub_rf.op.axis[0],  # N axis
                self.final_out_tensor_ub_rf.op.axis[1],
                self.final_out_tensor_ub_rf.op.axis[2],  # C1 axis
                self.final_out_tensor_ub_rf.op.axis[3],
                self.final_out_tensor_ub_rf.op.axis[4],
                self.sum_x_ub_outer,
                self.sum_x_ub_inner,
                self.final_out_tensor_ub_rf.op.reduce_axis[0],
                self.final_out_tensor_ub_rf.op.reduce_axis[1],
                self.final_out_tensor_ub_rf.op.axis[5]  # C0 axis
            )
        elif ub_split_reduce_axis == 1:
            sch[self.final_out_tensor_ub_rf].reorder(
                self.final_out_tensor_ub_rf.op.axis[0],  # N axis
                self.final_out_tensor_ub_rf.op.reduce_axis[2],
                self.final_out_tensor_ub_rf.op.axis[1],
                self.final_out_tensor_ub_rf.op.axis[2],  # C1 axis
                self.final_out_tensor_ub_rf.op.axis[3],
                self.final_out_tensor_ub_rf.op.axis[4],
                self.sum_x_ub_outer,
                self.sum_x_ub_inner,
                self.final_out_tensor_ub_rf.op.reduce_axis[1],
                self.final_out_tensor_ub_rf.op.axis[5]  # C0 axis
            )
        elif ub_split_reduce_axis == 2:
            sch[self.final_out_tensor_ub_rf].reorder(
                self.final_out_tensor_ub_rf.op.axis[0],  # N axis
                self.final_out_tensor_ub_rf.op.reduce_axis[2],
                self.final_out_tensor_ub_rf.op.axis[1],
                self.final_out_tensor_ub_rf.op.axis[2],  # C1 axis
                self.final_out_tensor_ub_rf.op.axis[3],
                self.final_out_tensor_ub_rf.op.axis[4],
                self.final_out_tensor_ub_rf.op.reduce_axis[0],
                self.sum_x_ub_outer,
                self.sum_x_ub_inner,
                self.final_out_tensor_ub_rf.op.axis[5]  # C0 axis
            )
        
        sch[self.final_out_tensor_ub_rf].compute_at(
            sch[self.final_out_tensor_global],
            self.final_out_tensor_global.op.reduce_axis[0])
        
        final_compute_at_tensor = self.final_out_tensor_global
        final_compute_at_buffer = self.final_out_tensor_ub_rf
        compute_at_axis1 = self.sum_x_ub_outer
        compute_at_axis2 = self.final_out_tensor_global.op.reduce_axis[0]
        self._do_compute_at(final_compute_at_tensor, final_compute_at_buffer, compute_at_axis1, compute_at_axis2)

        sch[self.final_out_tensor_ub_rf].pragma(self.sum_x_ub_outer, "json_info_batchBindOnly", 1)
        block = tvm.thread_axis("blockIdx.x")
        sch[self.final_out_tensor_global].bind(self.final_out_tensor_global.op.reduce_axis[0], block)

        self._do_storage_bound()
        sch[self.final_out_tensor_ub_rf].set_storage_bound(self.storage_bound)
        
        if self.mode == CONST:
            c0_size = 16
            n_size = self.shape_x[0]
            c1_size = self.shape_x[1]
            h_size = self.shape_x[2]
            w_size = self.shape_x[3]

            if self.is_do_double_buffer:
                if ub_split_reduce_axis == 0:
                    outer_loop = c1_size
                elif ub_split_reduce_axis == 1:
                    outer_loop = h_size // self.ub_inner
                    outer_loop = outer_loop * n_size * c1_size
                else:
                    outer_loop = w_size // self.ub_inner
                    outer_loop = outer_loop * n_size * c1_size * h_size

                self._do_const_double_buffer(outer_loop)

            sch[self.final_out_tensor_ub_rf].emit_insn(
                self.sum_x_ub_inner, "vector_reduce_sum")
            
            self._do_emit_insn()

            sch[self.final_out_tensor_global].emit_insn(
                self.final_out_tensor_global.op.axis[1], "dma_copy")
            
            sch[self.out].emit_insn(sch[self.out].op.axis[0], "phony_insn")
        else:
            self._do_double_buffer()

            self._do_emit_insn()

            sch[self.final_out_tensor_ub_rf].emit_insn(
                self.sum_x_ub_inner, "vector_reduce_sum")
            sch[self.final_out_tensor_global].emit_insn(
                self.final_out_tensor_global.op.axis[1], "dma_copy")
            sch[self.out].emit_insn(sch[self.out].op.axis[0], "phony_insn")
        self.sch_list[0] = sch
    
    def _schedule_cut_h_or_w_twice(self):
        sch = self.sch_list[0]

        final_out_tensor = self.out
        core_num = get_soc_spec("CORE_NUM")
        block_split_axis_index = self.block_split_axis_index
        ub_split_axis_index = self.ub_split_axis_index
        ub_split_reduce_axis = self.ub_split_reduce_axis

        self._do_cache_read()
        
        if self.mode == CONST:
            self.block_inner = self.tiling_case.block_factor
            self.ub_inner = self.tiling_case.ub_factor
        else:
            self.block_inner = var("block_factor", (1, None))
            self.ub_inner = var("ub_factor", (1, None))

        self._do_cache_write()
        
        self._do_compute_inline()

        self.sum_x_block_outer, self.sum_x_block_inner =\
            sch[final_out_tensor].split(
                final_out_tensor.op.reduce_axis[ub_split_axis_index-1],
                nparts=core_num)
        
        sch[final_out_tensor].split(self.sum_x_block_inner,
                                    factor=self.ub_inner)
        
        self.final_out_tensor_ub_rf, _ = sch.rfactor(final_out_tensor,
                                                     self.sum_x_block_outer)
        
        final_out_tensor_global_list = self.schedule.cache_write(self.output_tensor_set, "")
        final_tensors_index_res = []
        for tensor in self.output_tensor_set:
            for i, tensor_res in enumerate(self.res):
                if tensor == tensor_res:
                    final_tensors_index_res.append(i)
                    break
        for i, tensor in enumerate(final_out_tensor_global_list):
            self.res[final_tensors_index_res[i]] = tensor
        
        self.schedule[self.final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)

        self.final_out_tensor_global = final_out_tensor_global_list[0]

        sch[self.final_out_tensor_global].reorder(
            self.final_out_tensor_global.op.reduce_axis[0],
            self.final_out_tensor_global.op.axis[0],
            self.final_out_tensor_global.op.axis[1],
            self.final_out_tensor_global.op.axis[2],
            self.final_out_tensor_global.op.axis[3],
            self.final_out_tensor_global.op.axis[4])
        
        reorder_list = []
        for i in range(5):
            reorder_list.append(self.final_out_tensor_ub_rf.op.axis[i])
        
        if self.ub_split_axis_index == 2:
            reorder_list.append(self.final_out_tensor_ub_rf.op.reduce_axis[0])
            reorder_list.append(self.final_out_tensor_ub_rf.op.reduce_axis[2])
            reorder_list.append(self.final_out_tensor_ub_rf.op.reduce_axis[3])
            reorder_list.append(self.final_out_tensor_ub_rf.op.reduce_axis[1])
        else:
            for i in range(4):
                reorder_list.append(self.final_out_tensor_ub_rf.op.reduce_axis[i])
        reorder_list.append(self.final_out_tensor_ub_rf.op.axis[5])

        sch[self.final_out_tensor_ub_rf].reorder(*reorder_list)

        sch[self.final_out_tensor_ub_rf].compute_at(
            sch[self.final_out_tensor_global],
            self.final_out_tensor_global.op.axis[0])
        
        final_compute_at_buffer = self.final_out_tensor_ub_rf
        compute_at_axis = self.final_out_tensor_ub_rf.op.reduce_axis[2]
        self._do_compute_at(final_compute_at_buffer, final_compute_at_buffer, compute_at_axis, compute_at_axis)

        block = tvm.thread_axis("blockIdx.x")
        sch[self.final_out_tensor_global].bind(self.final_out_tensor_global.op.reduce_axis[0], block)

        self._do_storage_bound()
        sch[self.final_out_tensor_ub_rf].set_storage_bound(self.storage_bound)

        if self.mode == CONST:
            outer_loop = self.shape_x[2] // self.ub_inner
            outer_loop = outer_loop * self.shape_x[0] * self.shape_x[1]
            self._do_const_double_buffer(outer_loop)

            sch[self.final_out_tensor_ub_rf].emit_insn(
                self.final_out_tensor_ub_rf.op.reduce_axis[3], 
                "vector_reduce_sum")
            
            self._do_emit_insn()

            sch[self.final_out_tensor_global].emit_insn(
                self.final_out_tensor_global.op.axis[1], "dma_copy")
            sch[final_out_tensor].emit_insn(
                sch[final_out_tensor].op.axis[0], "phony_insn")
        else:
            self._do_double_buffer()

            self._do_emit_insn()

            sch[self.final_out_tensor_ub_rf].emit_insn(
                self.final_out_tensor_ub_rf.op.reduce_axis[3], 
                "vector_reduce_sum")

            sch[self.final_out_tensor_global].emit_insn(
                self.final_out_tensor_global.op.axis[1], "dma_copy")
            sch[final_out_tensor].emit_insn(
                sch[final_out_tensor].op.axis[0], "phony_insn")
        self.sch_list[0] = sch
    
    def _schedule_fuse_h_n(self):
        sch = self.sch_list[0]

        final_out_tensor = self.out
        core_num = get_soc_spec("CORE_NUM")
        half_core_num = core_num // 2
        block_split_axis_index = self.block_split_axis_index
        ub_split_axis_index = self.ub_split_axis_index
        ub_split_reduce_axis = self.ub_split_reduce_axis

        self._do_cache_read()
        
        if self.mode == CONST:
            self.block_inner = self.tiling_case.block_factor
            self.ub_inner = self.tiling_case.ub_factor
        else:
            self.block_inner = var("block_factor", (1, None))
            self.ub_inner = var("ub_factor", (1, None))
        
        self._do_cache_write()
        
        self._do_compute_inline()

        self.sum_x_block_outer, self.sum_x_block_inner =\
            sch[final_out_tensor].split(
                final_out_tensor.op.reduce_axis[1],
                nparts=half_core_num)
        
        sch[final_out_tensor].split(self.sum_x_block_inner,
                                    factor=self.ub_inner)
        fused = sch[final_out_tensor].fuse(final_out_tensor.op.reduce_axis[0],
                                           self.sum_x_block_outer)
        fused_outer, _ = sch[final_out_tensor].split(fused, nparts=core_num)
        self.final_out_tensor_ub_rf, _ = sch.rfactor(final_out_tensor, fused_outer)

        final_out_tensor_global_list = self.schedule.cache_write(self.output_tensor_set, "")
        final_tensors_index_res = []
        for tensor in self.output_tensor_set:
            for i, tensor_res in enumerate(self.res):
                if tensor == tensor_res:
                    final_tensors_index_res.append(i)
                    break
        for i, tensor in enumerate(final_out_tensor_global_list):
            self.res[final_tensors_index_res[i]] = tensor
        
        self.schedule[self.final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)

        self.final_out_tensor_global = final_out_tensor_global_list[0]

        sum_x_global_c1_axis = self.final_out_tensor_global.op.axis[1]
        sum_x_global_c0_axis = self.final_out_tensor_global.op.axis[4]

        sch[self.final_out_tensor_global].reorder(
            self.final_out_tensor_global.op.reduce_axis[0],
            sum_x_global_c1_axis,
            self.final_out_tensor_global.op.axis[0],
            self.final_out_tensor_global.op.axis[2],
            self.final_out_tensor_global.op.axis[3],
            sum_x_global_c0_axis)
        
        sch[self.final_out_tensor_ub_rf].reorder(
            self.final_out_tensor_ub_rf.op.axis[0],
            self.final_out_tensor_ub_rf.op.axis[1],
            self.final_out_tensor_ub_rf.op.reduce_axis[1],
            self.final_out_tensor_ub_rf.op.reduce_axis[2],
            self.final_out_tensor_ub_rf.op.reduce_axis[3],
            self.final_out_tensor_ub_rf.op.reduce_axis[0],
            self.final_out_tensor_ub_rf.op.axis[5])
        
        sch[self.final_out_tensor_ub_rf].compute_at(
            sch[self.final_out_tensor_global],
            sum_x_global_c1_axis)
        
        final_compute_at_buffer = self.final_out_tensor_ub_rf
        compute_at_axis = self.final_out_tensor_ub_rf.op.reduce_axis[2]
        self._do_compute_at(final_compute_at_buffer, final_compute_at_buffer, compute_at_axis, compute_at_axis)

        block = tvm.thread_axis("blockIdx.x")
        sch[self.final_out_tensor_global].bind(self.final_out_tensor_global.op.reduce_axis[0], block)

        if self.mode == CONST:
            self._do_storage_bound()
            sch[self.final_out_tensor_ub_rf].set_storage_bound(self.storage_bound)

            outer_loop = self.shape_x[2] // self.ub_inner
            outer_loop = outer_loop * self.shape_x[0] * self.shape_x[1]
            self._do_const_double_buffer(outer_loop)

            sch[self.final_out_tensor_ub_rf].emit_insn(
                self.final_out_tensor_ub_rf.op.reduce_axis[3], 
                "vector_reduce_sum")
            
            self._do_emit_insn()
            sch[self.final_out_tensor_global].emit_insn(
                self.final_out_tensor_global.op.axis[4], "dma_copy")
            sch[final_out_tensor].emit_insn(
                sch[final_out_tensor].op.axis[1], "phony_insn")
        else:
            self._do_storage_bound()
            sch[self.final_out_tensor_ub_rf].set_storage_bound(self.storage_bound)

            self._do_double_buffer()

            self._do_emit_insn()

            sch[self.final_out_tensor_ub_rf].emit_insn(
                self.final_out_tensor_ub_rf.op.reduce_axis[3], 
                "vector_reduce_sum")

            sch[self.final_out_tensor_global].emit_insn(
                self.final_out_tensor_global.op.axis[4], "dma_copy")
            sch[final_out_tensor].emit_insn(
                sch[final_out_tensor].op.axis[1], "phony_insn")
        self.sch_list[0] = sch
    
    def _schedule_cut_general(self):
        sch = self.sch_list[0]

        core_num = get_soc_spec("CORE_NUM")
        block_split_axis_index = self.block_split_axis_index
        ub_split_axis_index = self.ub_split_axis_index
        ub_split_reduce_axis = self.ub_split_reduce_axis

        self._do_cache_read()

        if self.mode == CONST:
            self.block_inner = self.tiling_case.block_factor
            self.ub_inner = self.tiling_case.ub_factor
        else:
            self.block_inner = var("block_factor", (1, None))
            self.ub_inner = var("ub_factor", (1, None))

        self._do_cache_write()

        self._do_compute_inline()

        final_out_tensor = self.out
        self.final_out_buffer_list = self.schedule.cache_write(self.output_tensor_set, cce.scope_ubuf)
        final_out_buffer = self.final_out_buffer_list[0]

        self.sum_x_ub_outer, self.sum_x_ub_inner = \
            self.schedule[final_out_buffer].split(
                          final_out_buffer.op.reduce_axis[self.ub_split_reduce_axis], 
                          factor=self.ub_inner)
        
        sum_x_c1_axis = final_out_tensor.op.axis[1]
        sum_x_c0_axis = final_out_tensor.op.axis[4]
        sum_x_ub_n_axis = final_out_buffer.op.axis[0]
        sum_x_ub_c1_axis = final_out_buffer.op.axis[1]
        sum_x_ub_h_axis = final_out_buffer.op.axis[2]
        sum_x_ub_w_axis = final_out_buffer.op.axis[3]
        sum_x_ub_c0_axis = final_out_buffer.op.axis[4]

        sum_x_ub_n_reduce_axis = final_out_buffer.op.reduce_axis[0]
        sum_x_ub_h_reduce_axis = final_out_buffer.op.reduce_axis[1]
        sum_x_ub_w_reduce_axis = final_out_buffer.op.reduce_axis[2]

        if self.ub_split_reduce_axis == 1:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                          sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_axis,
                                          self.sum_x_ub_outer,
                                          self.sum_x_ub_inner,
                                          sum_x_ub_w_axis,
                                          sum_x_ub_w_reduce_axis,
                                          sum_x_ub_c0_axis)
        else:
            sch[final_out_buffer].reorder(sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_reduce_axis,
                                          self.sum_x_ub_outer,
                                          self.sum_x_ub_inner,
                                          sum_x_ub_c0_axis)
        
        self._do_compute_at(final_out_buffer, final_out_buffer, self.sum_x_ub_outer, self.sum_x_ub_outer)

        sch[final_out_buffer].compute_at(sch[final_out_tensor], sum_x_c1_axis)

        block = tvm.thread_axis("blockIdx.x")
        sch[final_out_tensor].bind(sum_x_c1_axis, block)

        self._do_storage_bound()
        sch[final_out_buffer].set_storage_bound(self.storage_bound)

        if self.mode == CONST:
            outer_loop = self.shape_x[self.ub_split_axis_index] // self.ub_inner
            if self.ub_split_axis_index == 3:
                outer_loop = outer_loop * self.shape_x[2]
            self._do_const_double_buffer(outer_loop)
            
            sch[final_out_buffer].emit_insn(self.sum_x_ub_inner, "vector_reduce_sum")
            
            self._do_emit_insn()
            sch[final_out_tensor].emit_insn(sum_x_c0_axis, "dma_copy")
        else:
            self._do_double_buffer()

            self._do_emit_insn()
            sch[final_out_buffer].emit_insn(self.sum_x_ub_inner, 
                                            "vector_reduce_sum")
            sch[final_out_tensor].emit_insn(sum_x_c0_axis, "dma_copy")
        self.sch_list[0] = sch

    def _do_compute_at(self, final_out_tensor, final_out_buffer, compute_at_axis1, compute_at_axis2):
        sch = self.sch_list[0]

        for tensor in self.input_tensor_buffer_map:
            buffer_tensor = self.input_tensor_buffer_map[tensor]
            shape = shape_to_list(tensor.shape)
            if shape == self.shape_x:
                sch[buffer_tensor].compute_at(sch[final_out_buffer], compute_at_axis1)
            else:
                sch[buffer_tensor].compute_at(sch[final_out_tensor], compute_at_axis2)
        
        for tensor in self.mid_tensor_buffer_map:
            buffer_tensor = self.mid_tensor_buffer_map[tensor]
            shape = shape_to_list(tensor.shape)
            if shape == self.shape_x:
                sch[buffer_tensor].compute_at(sch[final_out_buffer], compute_at_axis1)
            else:
                sch[buffer_tensor].compute_at(sch[final_out_tensor], compute_at_axis2)
        self.sch_list[0] = sch
    
    def _do_emit_insn(self):
        """
        emit insn
        """
        sch = self.sch_list[0]
        for tensor in self.input_tensor_buffer_map:
            buffer_tensor = self.input_tensor_buffer_map[tensor]
            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")
        
        for tensor in self.mid_tensor_buffer_map:
            buffer_tensor = self.mid_tensor_buffer_map[tensor]
            insn = self._get_emit_insn_map(tensor)
            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)
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
    
    def _do_const_double_buffer(self, outer_loop):
        sch = self.sch_list[0]
        if outer_loop > 2:
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

        self.block_split_axis_index = tiling_case.block_split_axis_index
        self.ub_split_axis_index = tiling_case.ub_split_axis_index

        if self.ub_split_axis_index == 0:
            self.ub_split_reduce_axis = 0
        else:
            self.ub_split_reduce_axis = self.ub_split_axis_index - 1

        self._gen_reversed_subgraph_list()

        if self.shape_x in RESNET_50_SHAPE_LIST:
            self.is_do_double_buffer = False

        for tensor in self.mid_tensor_dst_tensor_map:
            self.schedule[tensor].set_scope(cce.scope_ubuf)

        batch = self.shape_x[0]
        c1_size = self.shape_x[1]
        core_num = get_soc_spec("CORE_NUM")

        if self.block_split_axis_index == 1:
            self._schedule_cut_c1()
        elif self.block_split_axis_index == 2 and self.ub_split_axis_index in (2, 3):
            self._schedule_cut_h_or_w_twice()
        elif self.block_split_axis_index == 5 and self.ub_split_axis_index == 2:
            self._schedule_fuse_h_n()
        elif self.block_split_axis_index == 0:
            self._schedule_cut_batch()
        elif self.block_split_axis_index == 6 and self.ub_split_axis_index in (2, 3):
            self._schedule_cut_general()

        outs = self.res
        return self.schedule