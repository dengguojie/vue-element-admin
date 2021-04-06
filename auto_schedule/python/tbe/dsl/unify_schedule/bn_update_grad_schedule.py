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
BN TRAINING UPDATE GRAD D   
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


@register_schedule(pattern=Pattern.BN_TRAINING_UPDATE_GRAD)
def schedule(outs, tiling_case: BNTrainingUpdateGradTilingCase):
    [outs].clear()

    graph_info = get_context().get_current_compute().get("compute_graph_info")
    bn_training_update_grad_info: BNTrainingUpdateGradInfo = get_context().get_current_compute().get("bn_training_update_grad_info")

    bn_training_update_grad_sch: BNUpdateGradSchedule = BNUpdateGradSchedule(graph_info, bn_training_update_grad_info)
    real_schedule = bn_training_update_grad_sch.do_schedule(tiling_case)
    real_schedule.tiling_key = tiling_case.tiling_key

    return real_schedule


class BNUpdateGradSchedule():
    def __init__(self, graph_info: ComputeGraphInfo, bn_training_update_grad_info: BNTrainingUpdateGradInfo):
        self.bn_training_update_grad_info = bn_training_update_grad_info
        self.mid_tensor_set = tuple(graph_info.mid_tensor_set)
        self.output_tensor_set = tuple(graph_info.output_tensor_set)
        self.input_tensor_set = tuple(graph_info.input_tensor_set)
        self.broadcast_tensor_set = tuple(graph_info.broadcast_tensor_set)

        self.out = self._fake_node(list(self.output_tensor_set))

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
        self.final_out_buffer_list = []
        self.compute_at_axis_list = []
        self.final_out_list = []

        self.shape_x = None
        self.fake_node_buffer = None
        self.ub_size = get_soc_spec("UB_SIZE")
        self.reuse_ub_map = {}
        self.reduce_0_tensor = None
        self.reduce_1_tensor = None
        self.reduce_0_buffer = None
        self.reduce_1_buffer = None
        self.grads_tensor = None
        self.grads_buffer = None
        self.cast_1_buffer = None

        self.reduce_0_outer = None
        self.reduce_0_inner = None
        self.reduce_1_outer = None
        self.reduce_1_inner = None

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
                if in_tensor.op.name == "grads_input":
                    self.grads_tensor = in_tensor
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

        dtype = input_x_tensor.dtype.lower()
        max_ub_count = self._get_max_ub_count(dtype, self.shape_x)
        add_compile_info("max_ub_count", max_ub_count)

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
                    "reduce_sum": "vector_reduce_sum"}
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
            if tensor.op.name == "grads_input":
                if tensor.dtype == "float32":
                    input_tensor = sch.cache_read(tensor, cce.scope_ubuf,
                                        self.input2dst_tensor_map[tensor][-1])
                    self.grads_buffer = sch.cache_read(tensor, cce.scope_ubuf,
                                            self.input2dst_tensor_map[tensor][0])
                
                if tensor.dtype == "float16":
                    input_tensor = sch.cache_read(tensor, cce.scope_ubuf,
                                        self.input2dst_tensor_map[tensor][1])
                    self.grads_buffer = sch.cache_read(tensor, cce.scope_ubuf,
                                        self.input2dst_tensor_map[tensor][0])
                self.input_tensor_buffer_map[tensor] = input_tensor
                continue
            input_tensor = sch.cache_read(tensor, cce.scope_ubuf,
                                        self.input2dst_tensor_map[tensor])
            self.input_tensor_buffer_map[tensor] = input_tensor
        
        for tensor in list(self.output_tensor_set):
            out_tensor = sch.cache_read(tensor, cce.scope_ubuf,
                                        self.tensor_list_dst_tensor_map[tensor])
            self.reuse_ub_map[tensor] = out_tensor
        
        self.sch_list[0] = sch
    
    def _do_cache_write(self):
        """
        cache write
        """
        self.mid_tensor_buffer_map = {}
        self.broadcast_tensor_buffers = {}
        sch = self.sch_list[0]
        for tensor in self.mid_tensor_dst_tensor_map:
            if tensor.op.name == "cast_1":
                self.cast_1_buffer = sch.cache_write(tensor, cce.scope_ubuf)
                continue
            buffer_tensor = sch.cache_write(tensor, cce.scope_ubuf)
            self.mid_tensor_buffer_map[tensor] = buffer_tensor

            if tensor in self.input_broadcast_tensors:
                self.broadcast_tensor_buffers.append(buffer_tensor)
            if tensor.op.name == "reduce_0":
                self.reduce_0_buffer = buffer_tensor
            if tensor.op.name == "reduce_1":
                self.reduce_1_buffer = buffer_tensor 
        
        self.fake_node_buffer = sch.cache_write(self.out, cce.scope_ubuf)
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
        ub_count = self.ub_size // 9
        for tensor in self.tensor_list:
            self.schedule[tensor].set_storage_bound(ub_count)
    
    def _do_tiling(self):
        tiling_case: BNTrainingUpdateGradTilingCase = self.tiling_case
        if not isinstance(tiling_case, BNTrainingUpdateGradTilingCase):
            raise RuntimeError("BNTrainingUpdateGradTilingCase required for BNTrainingUpdateGradSchedule!")
        
        core_num = get_soc_spec("CORE_NUM")

        # Get tiling tensor
        final_out_tensor = self.out
        final_out_buffer = self.fake_node_buffer

        # Get tiling axes
        block_split_axis_index = tiling_case.block_split_axis_index
        ub_split_axis_index = tiling_case.ub_split_axis_index

        # Get tiling params
        block_factor = tiling_case.block_factor
        self.block_inner = block_factor if block_factor is not None else var("block_factor", (1, 4))
        ub_factor = tiling_case.ub_factor
        self.ub_inner = ub_factor if ub_factor is not None else var("ub_factor", (1, 128))
        self._need_multi_core = tiling_case.multi_core

        # block tiling
        block_split_axis = final_out_tensor.op.axis[block_split_axis_index]
        if block_split_axis_index != 0:
            self.schedule[final_out_tensor].reorder(final_out_tensor.op.axis[0], block_split_axis)
        self.sum_x_block_outer, self.sum_x_block_inner = self.schedule[final_out_tensor].split(block_split_axis, nparts=32)

        self.reduce_0_outer, self.reduce_0_inner = self.schedule[self.reduce_0_buffer].split(self.reduce_0_buffer.op.reduce_axis[0], factor=1)
        self.reduce_1_outer, self.reduce_1_inner = self.schedule[self.reduce_1_buffer].split(self.reduce_1_buffer.op.reduce_axis[0], factor=1)

        self.schedule[final_out_tensor].reorder(self.sum_x_block_outer, self.sum_x_block_inner, final_out_tensor.op.axis[0],
                                                final_out_tensor.op.axis[2], final_out_tensor.op.axis[3], final_out_tensor.op.axis[4])

        self.schedule[self.reduce_0_buffer].reorder(self.reduce_0_outer, self.reduce_0_inner, self.reduce_0_buffer.op.axis[0], self.reduce_0_buffer.op.axis[1],
                                                self.reduce_0_buffer.op.axis[2], self.reduce_0_buffer.op.axis[3], self.reduce_0_buffer.op.axis[4])
        self.schedule[self.reduce_1_buffer].reorder(self.reduce_1_outer, self.reduce_1_inner, self.reduce_1_buffer.op.axis[0], self.reduce_1_buffer.op.axis[1],
                                                self.reduce_1_buffer.op.axis[2], self.reduce_1_buffer.op.axis[3], self.reduce_1_buffer.op.axis[4])
        
        self.compute_at_axis = self.sum_x_ub_outer
    
    def _do_compute_at(self):
        """
        compute at
        """
        sch = self.sch_list[0]

        for tensor in self.input_tensor_buffer_map:
            buffer_tensor = self.input_tensor_buffer_map[tensor]
            sch[buffer_tensor].compute_at(sch[self.reduce_0_buffer], self.reduce_0_outer)
        
        for tensor in self.mid_tensor_buffer_map:
            if tensor in self.output_tensor_set:
                continue
            buffer_tensor = self.mid_tensor_buffer_map[tensor]
            sch[buffer_tensor].compute_at(sch[self.reduce_0_buffer], self.reduce_0_outer)

        sch[self.grads_buffer].compute_at(sch[self.reduce_1_buffer], self.reduce_1_outer)
        if self.cast_1_buffer != None:
            sch[self.cast_1_buffer].compute_at(sch[self.reduce_1_buffer], self.reduce_1_outer)

        for tensor in list(self.output_tensor_set):
            buffer_tensor = self.mid_tensor_buffer_map[tensor]
            ub_tensor = self.reuse_ub_map[tensor]
            sch[buffer_tensor].compute_at(sch[self.out], self.sum_x_block_inner)
            sch[ub_tensor].compute_at(sch[self.out], self.sum_x_block_inner)
            sch[tensor].compute_at(sch[self.out], self.sum_x_block_inner)
        
        sch[self.fake_node_buffer].compute_at(sch[self.out], self.sum_x_block_inner)

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
        
        sch[self.grads_buffer].emit_insn(self.grads_buffer.op.axis[0], "dma_copy")

        if self.cast_1_buffer != None:
            sch[self.cast_1_buffer].emit_insn(self.cast_1_buffer.op.axis[4], "vector_conv")
        
        for tensor in self.mid_tensor_buffer_map:
            buffer_tensor = self.mid_tensor_buffer_map[tensor]
            if tensor.op.tag == "unified_broadcast":
                insn = "unified_broadcast"
                sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[4], insn)
                continue
            elif tensor.op.tag == "unknown_broadcast":
                insn = "unified_broadcast"
            else:
                insn = self._get_emit_insn_map(tensor)
            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[4], insn)
        
        for tensor in list(self.output_tensor_set):
            ub_tensor = self.reuse_ub_map[tensor]
            sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy", attrs=attrs)
            sch[ub_tensor].emit_insn(ub_tensor.op.axis[0], "dma_copy")
        
        sch[self.out].emit_insn(self.out.op.axis[0], "phony_insn")
        sch[self.fake_node_buffer].emit_insn(self.fake_node_buffer.op.axis[0], "phony_insn")

        self.sch_list[0] = sch
    
    def _fake_node(self, tensors):
        dtype = tensors[0].dtype
        dim_length = max([len(t.shape) for t in tensors])
        shape = [1] * dim_length
        for tensor_i in tensors:
            if DTYPE_BYTE_MAPPING[tensor_i.dtype] > DTYPE_BYTE_MAPPING[dtype]:
                dtype = tensor_i.type
            shape_i = util.shape_to_list(tensor_i.shape)
            diff = dim_length - len(shape_i)
            shape_i = [1] * diff + shape_i
            for j in range(diff, dim_length):
                if util.equals_one(shape[j]):
                    shape[j] = shape_i[j]
                elif not expr_equal(shape[j], shape_i[j]) and not util.equals_one(shape_i[j]):
                    shape[j] = tvm.max(shape_i[j], shape[j])
            
        def _fake_compute(*indices):
            res_ = tvm.const(1, dtype)
            for tensor in tensors:
                cur_indices = []
                for idx, dim in enumerate(tensor.shape):
                    if util.equals_one(dim):
                        cur_indices.append(0)
                    else:
                        cur_indices.append(indices[idx])
                res_ *= tvm.expr.Cast(dtype, tensor(*cur_indices))
            return res_
        
        with tvm.tag_scope(FAKE_NODE_TAG):
            res = tvm.compute(shape, _fake_compute, name="fake_node")
        
        return res
    
    def do_schedule(self, tiling_case: BNTrainingUpdateGradTilingCase):
        self.schedule = tvm.create_schedule(self.out.op)
        self.sch_list = [self.schedule]
        self.tiling_case: BNTrainingUpdateGradTilingCase = tiling_case

        self._gen_reversed_subgraph_list()

        for tensor in self.mid_tensor_dst_tensor_map:
            if tensor not in list(self.output_tensor_set):
                self.schedule[tensor].set_scope(cce.scope_ubuf)
        self.schedule[self.out].set_scope(cce.scope_ubuf)

        self._do_cache_read()
        
        self._do_cache_write()

        self._do_compute_inline()

        self._do_storage_bound()

        self._do_tiling()

        for tensor in self.output_tensor_set:
            buffer_tensor = self.mid_tensor_buffer_map[tensor]
            ub_tensor = self.reuse_ub_map[tensor]
            self.schedule[buffer_tensor].reused_by(ub_tensor)

        self._do_compute_at()

        block = tvm.thread_axis("blockIdx.x")
        self.schedule[self.out].bind(self.sum_x_block_outer, block)

        self._do_emit_insn()

        return self.schedule