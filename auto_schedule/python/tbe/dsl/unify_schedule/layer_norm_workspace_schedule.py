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
LayerNorm Schedule Remake stage 1
"""

# Standard Package
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Union
from typing import Tuple
from typing import Dict
from typing import Set

from tbe import tvm
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils import shape_util
from ..base import operation
from ..base.operation import get_context
from ..base.operation import register_schedule
from ..base.operation import var
from .constants import Pattern
from .constants import INSN_MAPPING
from .layer_norm_tilingcase import LayerNormInfo
from .layer_norm_tilingcase import LayerNormTilingCase
from . import util
from .util import get_dsl_insn

MAX_NODE_COUNT = 12
PHONY_INSN = "phony_insn"
DMA_COPY = "dma_copy"
# Ascend310 bound value
MAX_BOUND = 8 * 1024


class WorkspaceLayerNormSchedule:
    def __init__(self, graph_info, layer_norm_info, outs):
        self.graph_info = graph_info
        self.layer_norm_info = layer_norm_info
        # foward compute graph
        self.forward_compute_graph_map = self.graph_info.tensor_consumers_map
        self.backward_compute_graph_map = self.graph_info.tensor_producers_map
        self.dim_ln_range = operation.get_context().get_current_compute().get("dim_ln_range")
        self.emit_insn_dict = {}
        self.outs = outs
        self.is_cast = False
        self.is_last_dim_one = False
        self.reduce0_tensor = None
        self.reduce1_tensor = None
        self.sub_to_mul0 = None
        self.sub_to_mul1 = None
        self.mid0_output_gm_tensor = None
        self.mid1_output_gm_tensor = None
        self.input_tensor_ub_list = []
        self.input_tensor_2_dict = {}
        self.beta_gam_ub_list = []
        self.output_tensor_ub_list = []
        self.tensorub2tensor = dict()
        self.tensor2tensorub = dict()
        self.mid_tensor_ub_back_dict = {}

    def get_sub_tensor(self):
        for input_tensor in self.graph_info.input_tensor_set:
            if input_tensor.op.name == "x":
                self.x_tensor = input_tensor
        for x_consumer_tensor in self.graph_info.tensor_consumers_map[self.x_tensor]:
            opname, num = x_consumer_tensor.op.name.split("_")
            if opname == "cast":
                self.is_cast = True

        sub_gm_tensor = None
        sub_gm_num = -1
        for ele_tensor in tuple(self.forward_compute_graph_map):
            if ele_tensor.op.name.startswith("broadcast") or ele_tensor in self.graph_info.input_tensor_set:
                continue
            opname, num = ele_tensor.op.name.split("_")
            if sub_gm_tensor is None and opname == "sub":
                sub_gm_tensor = ele_tensor
                sub_gm_num = int(num)
            elif sub_gm_tensor is not None and opname == "sub" and int(num) < sub_gm_num:
                sub_gm_tensor = ele_tensor
                sub_gm_num = int(num)
        self.outs.append(sub_gm_tensor)
        self.sub_gm_tensor = sub_gm_tensor
        self.shape_dim = len(self.sub_gm_tensor.shape)
        shape = shape_util.shape_to_list(self.sub_gm_tensor.shape)
        if shape[0] == 1:
            self.shape_dim -= 1
        if shape[-1] == 1:
            self.shape_dim -= 1
            self.is_last_dim_one = True
        # find reduce_0, reduce_1, mid_output, end_output

        mid_num = -1
        for red_tensor in self.graph_info.reduce_tensor_set:
            red_tensor_num = int(red_tensor.op.name.split("_")[1])
            if self.reduce0_tensor is None:
                self.reduce0_tensor = red_tensor
                mid_num = red_tensor_num
            elif mid_num > red_tensor_num:
                self.reduce0_tensor, self.reduce1_tensor = red_tensor, self.reduce0_tensor
            else:
                self.reduce1_tensor = red_tensor

        sub_to_mul_num = -1
        for tensor in tuple(self.forward_compute_graph_map[self.sub_gm_tensor]):
            mul_tensor_num = int(tensor.op.name.split("_")[1])
            if self.sub_to_mul0 is None:
                self.sub_to_mul0 = tensor
                sub_to_mul_num = mul_tensor_num
            elif sub_to_mul_num > mul_tensor_num:
                self.sub_to_mul0, self.sub_to_mul1 = tensor, self.sub_to_mul0
            else:
                self.sub_to_mul1 = tensor

        mid_output_num = -1
        for mid_out_tensor in self.graph_info.mid_output_tensor_set:
            tensor_num = int(mid_out_tensor.op.name.split("_")[1])
            if self.mid0_output_gm_tensor is None:
                self.mid0_output_gm_tensor = mid_out_tensor
                mid_output_num = tensor_num
            elif mid_output_num > tensor_num:
                self.mid0_output_gm_tensor, self.mid1_output_gm_tensor = mid_out_tensor, self.mid0_output_gm_tensor
            else:
                self.mid1_output_gm_tensor = mid_out_tensor

    def do_schedule(self, tiling_case):

        if not isinstance(tiling_case, LayerNormTilingCase):
            raise RuntimeError(
                "LayerNormTilingCase required for LayerNormSchedule")
        self.tiling_case = tiling_case
        self.get_sub_tensor()
        self._do_create_schedule()
        self._do_set_var_range()
        self._do_cache_read_write()
        self._do_set_scope()
        self._do_storage_bound()
        self._do_tiling()
        self._do_storage_align()
        self._do_compute_at()
        self._do_emit_insn()
        return self.schedule

    def _do_create_schedule(self):
        self.schedule = tvm.create_schedule(
            [tuple(self.graph_info.endpoint_output_tensor_set)[0].op, self.sub_gm_tensor.op])

    def _do_set_var_range(self):
        res_tensor = tuple(self.graph_info.endpoint_output_tensor_set)[0]
        self.res_shape = res_tensor.shape
        if tuple(self.dim_ln_range) == (1, 1):
            self.is_last_dim_one = True
        if isinstance(self.res_shape[-1], tvm.expr.Var):
            self.schedule.set_var_range(self.res_shape[-1], *self.dim_ln_range)

    def _do_cache_read_write(self):
        def _do_cache_read():
            # raw input params
            for input_tensor in self.graph_info.input_tensor_set:
                for nextest_tensor in self.forward_compute_graph_map[input_tensor]:
                    input_tensor_ub = self.schedule.cache_read(
                        input_tensor, tbe_platform_info.scope_ubuf, [nextest_tensor])
                    self.input_tensor_ub_list.append(input_tensor_ub)
                    self.input_tensor_2_dict[nextest_tensor] = input_tensor_ub
                if input_tensor.op.name == "x":
                    self.x_tensor = input_tensor
                else:
                    self.beta_gam_ub_list.append(input_tensor_ub)

        def _do_cache_write():
            for output_tensor in self.graph_info.output_tensor_set:
                output_tensor_ub = self.schedule.cache_write(output_tensor, tbe_platform_info.scope_ubuf)
                self.output_tensor_ub_list.append(output_tensor_ub)
                self.tensorub2tensor[output_tensor_ub] = output_tensor
                self.tensor2tensorub[output_tensor] = output_tensor_ub

        def _do_reused_by():
            # reused_by mid_tensor
            # middle_output cache_read
            for mid_tensor in self.graph_info.mid_output_tensor_set:
                mid_tensor_ub_back = self.schedule.cache_read(mid_tensor, tbe_platform_info.scope_ubuf,
                                                              self.forward_compute_graph_map[mid_tensor])
                self.schedule[mid_tensor_ub_back].reused_by(self.tensor2tensorub[mid_tensor])
                self.input_tensor_ub_list.append(mid_tensor_ub_back)
                self.mid_tensor_ub_back_dict[mid_tensor] = [self.tensor2tensorub[mid_tensor], mid_tensor_ub_back]

            # find the first sub tensor
            # the first sub tensor in workspace
            sub_1_ub = self.schedule.cache_write(
                self.sub_gm_tensor,
                tbe_platform_info.scope_ubuf
            )

            self.mid_tensor_ub_back_dict[self.sub_gm_tensor] = [sub_1_ub]
            for tensor in tuple(self.forward_compute_graph_map[self.sub_gm_tensor]):
                sub_1_ub_back = self.schedule.cache_read(self.sub_gm_tensor, tbe_platform_info.scope_ubuf, [tensor])
                self.schedule[sub_1_ub_back].reused_by(sub_1_ub)
                self.input_tensor_ub_list.extend([sub_1_ub, sub_1_ub_back])
                self.mid_tensor_ub_back_dict[self.sub_gm_tensor].append(sub_1_ub_back)
                self.input_tensor_2_dict[tensor] = sub_1_ub_back

        # cache read
        _do_cache_read()
        # cache_write
        _do_cache_write()
        # reused by
        _do_reused_by()

    def _do_set_scope(self):
        for tensor in self.forward_compute_graph_map:
            if tensor not in tuple(
                    self.graph_info.output_tensor_set | self.graph_info.input_tensor_set | {self.sub_gm_tensor}):
                self.schedule[tensor].set_scope(tbe_platform_info.scope_ubuf)

    def _do_storage_bound(self):
        UB_SIZE = get_soc_spec("UB_SIZE")
        soc_version = get_soc_spec("SOC_VERSION")
        # analysis coexit tensor node
        tensor_storage_bound_set = set(self.forward_compute_graph_map.keys()) | set(
            self.output_tensor_ub_list) | set(self.input_tensor_ub_list)
        if self.is_cast:
            # fp16 --> fp32
            self.max_ub_size = int(UB_SIZE / MAX_NODE_COUNT / 2)
        else:
            # fp32 or fp16 in 310
            self.max_ub_size = int(UB_SIZE / MAX_NODE_COUNT / 2) if soc_version not in ("Ascend310",) else MAX_BOUND
        for stage_tensor in tensor_storage_bound_set:
            if self.forward_compute_graph_map.get(stage_tensor) and stage_tensor in self.graph_info.input_tensor_set:
                continue
            self.schedule[stage_tensor].set_storage_bound(self.max_ub_size)

    def _do_storage_align(self):
        case = self.tiling_case
        input_format = case.format
        if input_format != "FRACTAL_NZ":
            tensor_storage_bound_set = set(self.forward_compute_graph_map.keys()) | set(
                self.output_tensor_ub_list) | set(self.input_tensor_ub_list)
            reduce_len = len(self.reduce0_tensor.op.reduce_axis)
            output_dtype = tuple(self.graph_info.endpoint_output_tensor_set)[0].dtype
            if output_dtype in ("fp16", "float16"):
                align_num = 16
            else:
                align_num = 8
            if self.is_cast:
                op_name_0 = self.mid0_output_gm_tensor.op.name
                op_name_1 = self.mid1_output_gm_tensor.op.name
                op_name_0_consumer = tuple(self.forward_compute_graph_map[self.mid0_output_gm_tensor])[0].op.name
                op_name_1_consumer = tuple(self.forward_compute_graph_map[self.mid1_output_gm_tensor])[0].op.name

            # if self.shape_dim > 1: operate storage align
            if self.shape_dim > 1 and not self.is_last_dim_one:
                for stage_tensor in tensor_storage_bound_set:
                    if stage_tensor in self.graph_info.input_tensor_set | self.graph_info.output_tensor_set | {self.sub_gm_tensor}:
                        continue
                    op_name = stage_tensor.op.name
                    if op_name.startswith("reduce"):
                        continue
                    if self.is_cast and (op_name.startswith(op_name_0) or op_name.startswith(op_name_0_consumer) or op_name.startswith(op_name_1) or op_name.startswith(op_name_1_consumer)):
                        continue
                    self.schedule[stage_tensor].storage_align(stage_tensor.op.axis[-2], align_num, 0)

    def _do_tiling(self):
        case = self.tiling_case
        reduce_axis_list = case.reduce_axis_list
        input_format = case.format

        # get last endpoint output tensor
        self.res_tensor = tuple(self.graph_info.endpoint_output_tensor_set)[0]

        # get tiling axis
        self.block_split_axis_index = case.block_split_axis_index
        self.ub_split_axis_index = case.ub_split_axis_index
        self.ub_split_axis_index_reduce = case.ub_split_axis_index_reduce

        # Get tiling params
        block_factor = case.block_factor
        ub_factor = case.ub_factor
        ub_fuse_factor = case.ub_fuse_factor

        block_inner = block_factor if block_factor is not None else var("block_factor", (1, None))
        block_factor_1 = case.block_factor_1
        block_inner_1 = block_factor_1 if block_factor_1 is not None else var(
            "block_factor_1", (1, None))
        ub_inner = ub_factor if ub_factor is not None else var("ub_factor", (1, None))
        ub_fuse_inner = ub_fuse_factor if ub_fuse_factor is not None else var("ub_fuse_factor", (1, None))

        # subgraph 0 tiling case
        def do_tiling_subgraph0():
            sub_order = []
            for d_i in range(len(self.sub_gm_tensor.shape)):
                sub_order.append(self.sub_gm_tensor.op.axis[d_i])

            if not self.is_cast:
                reduce_tensor = self.mid_tensor_ub_back_dict[self.mid0_output_gm_tensor][0]
            else:
                reduce_tensor = self.reduce0_tensor

            sub_block_outer, sub_block_inner = self.schedule[self.sub_gm_tensor].split(
                self.sub_gm_tensor.op.axis[self.block_split_axis_index], factor=block_inner)
            sub_ub_fuse_outer, sub_ub_fuse_inner = self.schedule[self.sub_gm_tensor].split(
                sub_block_inner, factor=ub_fuse_inner)
            reduce0_ub_fuse_outer, reduce0_ub_fuse_inner = self.schedule[reduce_tensor].split(
                reduce_tensor.op.reduce_axis[self.ub_split_axis_index_reduce], factor=ub_inner)
            if case.is_split_ub:
                sub_ub_gm_outer, sub_ub_gm_inner = self.schedule[self.sub_gm_tensor].split(
                    self.sub_gm_tensor.op.axis[self.ub_split_axis_index], factor=ub_inner)
            else:
                sub_ub_gm_outer, sub_ub_gm_inner = self.schedule[self.sub_gm_tensor].split(
                    sub_ub_fuse_inner, factor=ub_inner)

            self.sub_block_split_result = [sub_block_outer, sub_block_inner]
            self.sub_ub_fuse_split_result = [sub_ub_fuse_outer, sub_ub_fuse_inner]
            self.sub_ub_split_result = [sub_ub_gm_outer, sub_ub_gm_inner]
            self.reduce0_ub_split_result = [reduce0_ub_fuse_outer, reduce0_ub_fuse_inner]

            return sub_order

        # subgraph1 tiling case
        def do_tiling_subgraph1():
            res_order = []
            for d_i in range(len(self.res_tensor.shape)):
                res_order.append(self.res_tensor.op.axis[d_i])

            if not self.is_cast:
                reduce_tensor = self.mid_tensor_ub_back_dict[self.mid1_output_gm_tensor][0]
            else:
                reduce_tensor = self.reduce1_tensor

            res_block_outer, res_block_inner = self.schedule[self.res_tensor].split(
                self.res_tensor.op.axis[self.block_split_axis_index], factor=block_inner)
            res_ub_fuse_outer, res_ub_fuse_inner = self.schedule[self.res_tensor].split(
                res_block_inner, factor=ub_fuse_inner)
            reduce1_ub_fuse_outer, reduce1_ub_fuse_inner = self.schedule[reduce_tensor].split(
                reduce_tensor.op.reduce_axis[self.ub_split_axis_index_reduce], factor=ub_inner)
            if case.is_split_ub:
                res_ub_gm_outer, res_ub_gm_inner = self.schedule[self.res_tensor].split(
                    self.res_tensor.op.axis[self.ub_split_axis_index], factor=ub_inner)
            else:
                res_ub_gm_outer, res_ub_gm_inner = self.schedule[self.res_tensor].split(
                    res_ub_fuse_inner, factor=ub_inner)
            # judge open multi_core

            if case.multi_core is None:
                raise RuntimeError("Tilingcase didn`t declare multi_core switch")
            if case.multi_core:
                block = tvm.thread_axis("blockIdx.x")
                self.schedule[self.res_tensor].bind(res_block_outer, block)
                self.schedule[self.sub_gm_tensor].bind(self.sub_block_split_result[0], block)

            self.res_block_split_result = [res_block_outer, res_block_inner]
            self.res_ub_fuse_split_result = [res_ub_fuse_outer, res_ub_fuse_inner]
            self.res_ub_split_result = [res_ub_gm_outer, res_ub_gm_inner]
            self.reduce1_ub_split_result = [reduce1_ub_fuse_outer, reduce1_ub_fuse_inner]

            return res_order

        sub_order = do_tiling_subgraph0()
        res_order = do_tiling_subgraph1()

        # reorder part
        if input_format == "FRACTAL_NZ":
            res_axis_order = []
            sub_axis_order = []
            sub_reorder_dict_old2new = {}
            new_index = 0
            # reorder nonreduce axis order before last reduce axis
            for idx, sub_axis in enumerate(sub_order):
                if idx not in reduce_axis_list:
                    sub_axis_order.append(sub_axis)
                    res_axis_order.append(res_order[idx])
                    sub_reorder_dict_old2new[idx] = new_index
                    new_index += 1
                elif idx != reduce_axis_list[-1]:
                    continue
                else:
                    break
            # reorder separated reduce axis together
            # append reduce axis
            for rai in reduce_axis_list:
                sub_axis_order.append(sub_order[rai])
                res_axis_order.append(res_order[rai])
                sub_reorder_dict_old2new[rai] = new_index
                new_index += 1
            # reorder nonreduce axis order after last reduce axis
            for ot in range(reduce_axis_list[-1] + 1, len(sub_order)):
                sub_axis_order.append(sub_order[ot])
                res_axis_order.append(res_order[ot])
                sub_reorder_dict_old2new[ot] = new_index
                new_index += 1
            # reorder split result order
            reordered_block_index = sub_reorder_dict_old2new[self.block_split_axis_index]
            if case.is_split_ub:
                reordered_ub_index = sub_reorder_dict_old2new[self.ub_split_axis_index]
                fin_sub_axis_order = sub_axis_order[:reordered_block_index] + self.sub_block_split_result[:1] + self.sub_ub_fuse_split_result + \
                    sub_axis_order[reordered_block_index + 1:reordered_ub_index] + \
                    self.sub_ub_split_result + sub_axis_order[reordered_ub_index + 1:]

                fin_res_axis_order = res_axis_order[:reordered_block_index] + self.res_block_split_result[:1] + self.res_ub_fuse_split_result + \
                    res_axis_order[reordered_block_index + 1:reordered_ub_index] + \
                    self.res_ub_split_result + res_axis_order[reordered_ub_index + 1:]
            else:
                fin_sub_axis_order = sub_axis_order[:reordered_block_index] + self.sub_block_split_result[:1] + \
                    self.sub_ub_fuse_split_result[:1] + self.sub_ub_split_result + \
                    sub_axis_order[reordered_block_index + 1:]

                fin_res_axis_order = res_axis_order[:reordered_block_index] + self.res_block_split_result[:1] + \
                    self.res_ub_fuse_split_result[:1] + self.res_ub_split_result + \
                    res_axis_order[reordered_block_index + 1:]

            self.schedule[self.sub_gm_tensor].reorder(*fin_sub_axis_order)
            self.schedule[self.res_tensor].reorder(*fin_res_axis_order)

    def _do_compute_at(self):
        # redefine tensor emitinsn axis
        self.ub_split_axis_index = 0

        def _compute_at_func(tensor_key,
                             base_tensor,
                             at_axis_list,
                             not_at_tensor_list,
                             emit_axis_index,
                             is_reduce=False):

            if tensor_key not in (base_tensor, self.mid0_output_gm_tensor, self.mid1_output_gm_tensor):
                self.schedule[tensor_key].compute_at(self.schedule[base_tensor], at_axis_list[0])

                if is_reduce and tensor_key not in self.graph_info.input_tensor_set:
                    self.emit_insn_dict[tensor_key] = {"insn": "", "axis": at_axis_list[1]}
                elif tensor_key not in self.graph_info.input_tensor_set:
                    self.emit_insn_dict[tensor_key] = {"insn": "", "axis": tensor_key.op.axis[emit_axis_index]}

            for tensor_i in self.backward_compute_graph_map[tensor_key]:
                if tensor_i in not_at_tensor_list:
                    continue
                _compute_at_func(tensor_i, base_tensor, at_axis_list, not_at_tensor_list, emit_axis_index, is_reduce)

        def _do_compute_at_0():
            # before reduce
            if self.is_cast:
                mid0_tensor_ub, mid0_tensor_ub_back = self.mid_tensor_ub_back_dict[self.mid0_output_gm_tensor]
                mean_tensor_ub = self.reduce0_tensor

                self.cast_to_mul0 = None
                self.cast_to_sub1 = None
                for tensor in tuple(self.forward_compute_graph_map[self.x_tensor]):
                    cast_tensor_num = int(tensor.op.name.split("_")[1])
                    if self.cast_to_mul0 is None:
                        self.cast_to_mul0 = tensor
                        cast_num = cast_tensor_num
                    elif cast_num > cast_tensor_num:
                        self.cast_to_mul0, self.cast_to_sub1 = tensor, self.cast_to_mul0
                    else:
                        self.cast_to_sub1 = tensor
                x_mul_ub = self.input_tensor_2_dict[self.cast_to_mul0]
                self.schedule[x_mul_ub].compute_at(self.schedule[mean_tensor_ub], self.reduce0_ub_split_result[0])
                self.emit_insn_dict[x_mul_ub] = {
                    "insn": DMA_COPY,
                    "axis": x_mul_ub.op.axis[0]
                }

                _compute_at_func(mean_tensor_ub, mean_tensor_ub, self.reduce0_ub_split_result, [],
                                 0, False)

                self.schedule[mean_tensor_ub].compute_at(self.schedule[self.sub_gm_tensor],
                                                         self.sub_ub_fuse_split_result[0])
                self.schedule[mid0_tensor_ub].compute_at(self.schedule[self.sub_gm_tensor],
                                                         self.sub_ub_fuse_split_result[0])
                self.schedule[self.mid0_output_gm_tensor].compute_at(self.schedule[self.sub_gm_tensor],
                                                                     self.sub_ub_fuse_split_result[0])
                self.schedule[mid0_tensor_ub_back].compute_at(self.schedule[self.sub_gm_tensor],
                                                              self.sub_ub_fuse_split_result[0])
                # cast fp16 to fp32
                special_tensor_0 = tuple(self.graph_info.tensor_consumers_map[self.mid0_output_gm_tensor])[0]
                self.schedule[special_tensor_0].reused_by(mean_tensor_ub)
                self.schedule[special_tensor_0].compute_at(self.schedule[self.sub_gm_tensor],
                                                           self.sub_ub_fuse_split_result[0])
                self.emit_insn_dict[mean_tensor_ub] = {"insn": "", "axis": self.reduce0_ub_split_result[1],
                                                       "attr": {"extra_space": 1024}}
                self.emit_insn_dict[mid0_tensor_ub] = {
                    "insn": "",
                    "axis": mid0_tensor_ub.op.axis[self.block_split_axis_index]
                }
                self.emit_insn_dict[self.mid0_output_gm_tensor] = {
                    "insn": DMA_COPY,
                    "axis": self.mid0_output_gm_tensor.op.axis[self.block_split_axis_index]
                }
                self.emit_insn_dict[mid0_tensor_ub_back] = {
                    "insn": PHONY_INSN,
                    "axis": mid0_tensor_ub_back.op.axis[self.block_split_axis_index],
                }
                self.emit_insn_dict[special_tensor_0] = {
                    "insn": PHONY_INSN,
                    "axis": special_tensor_0.op.axis[self.block_split_axis_index],
                }

                x_sub_ub = self.input_tensor_2_dict[self.cast_to_sub1]
                self.schedule[x_sub_ub].compute_at(self.schedule[self.sub_gm_tensor], self.sub_ub_split_result[0])
                self.emit_insn_dict[x_sub_ub] = {"insn": DMA_COPY, "axis": x_sub_ub.op.axis[0]}

                # mid0_output compute at
                _compute_at_func(self.sub_gm_tensor, self.sub_gm_tensor, self.sub_ub_split_result,
                                 [special_tensor_0, self.x_tensor], 0)
            else:
                mean_tensor_ub, mean_tensor_ub_back = self.mid_tensor_ub_back_dict[self.mid0_output_gm_tensor]

                mul_0_tensor = tuple(self.forward_compute_graph_map[self.x_tensor] - {self.sub_gm_tensor})[0]
                x_mul_ub = self.input_tensor_2_dict[mul_0_tensor]
                self.schedule[x_mul_ub].compute_at(self.schedule[mean_tensor_ub], self.reduce0_ub_split_result[0])

                self.emit_insn_dict[x_mul_ub] = {
                    "insn": DMA_COPY,
                    "axis": x_mul_ub.op.axis[0]
                }

                # mid0_output compute at
                _compute_at_func(self.mid0_output_gm_tensor, mean_tensor_ub, self.reduce0_ub_split_result, [],
                                 0, False)

                # after reduce
                # reduce fuse loop:ub->gm->ub

                self.schedule[mean_tensor_ub].compute_at(self.schedule[self.sub_gm_tensor],
                                                         self.sub_ub_fuse_split_result[0])
                self.schedule[self.mid0_output_gm_tensor].compute_at(self.schedule[self.sub_gm_tensor],
                                                                     self.sub_ub_fuse_split_result[0])
                self.schedule[mean_tensor_ub_back].compute_at(self.schedule[self.sub_gm_tensor],
                                                              self.sub_ub_fuse_split_result[0])
                self.emit_insn_dict[mean_tensor_ub] = {"insn": "", "axis": self.reduce0_ub_split_result[1],
                                                       "attr": {"extra_space": 1024}}

                self.emit_insn_dict[self.mid0_output_gm_tensor] = {
                    "insn": DMA_COPY,
                    "axis": self.mid0_output_gm_tensor.op.axis[self.block_split_axis_index]
                }
                self.emit_insn_dict[mean_tensor_ub_back] = {
                    "insn": PHONY_INSN,
                    "axis": mean_tensor_ub_back.op.axis[self.block_split_axis_index],
                }
                x_sub_ub = self.input_tensor_2_dict[self.sub_gm_tensor]

                self.schedule[x_sub_ub].compute_at(self.schedule[self.sub_gm_tensor], self.sub_ub_split_result[0])
                self.emit_insn_dict[x_sub_ub] = {"insn": DMA_COPY, "axis": x_sub_ub.op.axis[0]}

                # mid0_output compute at
                _compute_at_func(self.sub_gm_tensor, self.sub_gm_tensor, self.sub_ub_split_result,
                                 [self.mid0_output_gm_tensor, self.x_tensor], 0)
            sub_1_ub = self.mid_tensor_ub_back_dict[self.sub_gm_tensor][0]
            self.schedule[sub_1_ub].compute_at(self.schedule[self.sub_gm_tensor], self.sub_ub_split_result[0])
            self.emit_insn_dict[sub_1_ub] = {"insn": "", "axis": sub_1_ub.op.axis[0]}
            self.emit_insn_dict[self.sub_gm_tensor] = {"insn": DMA_COPY, "axis": self.sub_ub_split_result[1]}

        def _do_compute_at_1():
            # before reduce
            if self.is_cast:
                variance_tensor_ub = self.reduce1_tensor
                mid1_tensor_ub, mid1_tensor_ub_back = self.mid_tensor_ub_back_dict[self.mid1_output_gm_tensor]
                sub_1_ub_back_var = self.input_tensor_2_dict[self.sub_to_mul0]
                sub_1_ub_back_res = self.input_tensor_2_dict[self.sub_to_mul1]
                self.schedule[sub_1_ub_back_var].compute_at(self.schedule[variance_tensor_ub],
                                                            self.reduce1_ub_split_result[0])
                self.emit_insn_dict[sub_1_ub_back_var] = {
                    "insn": DMA_COPY,
                    "axis": sub_1_ub_back_var.op.axis[0]
                }
                _compute_at_func(variance_tensor_ub, variance_tensor_ub,
                                 self.reduce1_ub_split_result, [self.sub_gm_tensor], 0, False)
                self.schedule[variance_tensor_ub].compute_at(self.schedule[self.res_tensor],
                                                             self.res_ub_fuse_split_result[0])
                self.schedule[mid1_tensor_ub].compute_at(self.schedule[self.res_tensor],
                                                         self.res_ub_fuse_split_result[0])
                self.schedule[self.mid1_output_gm_tensor].compute_at(self.schedule[self.res_tensor],
                                                                     self.res_ub_fuse_split_result[0])
                self.schedule[mid1_tensor_ub_back].compute_at(self.schedule[self.res_tensor],
                                                              self.res_ub_fuse_split_result[0])
                # cast fp16 to fp32
                special_tensor_1 = tuple(self.graph_info.tensor_consumers_map[self.mid1_output_gm_tensor])[0]
                self.schedule[special_tensor_1].reused_by(variance_tensor_ub)
                self.schedule[special_tensor_1].compute_at(self.schedule[self.res_tensor],
                                                           self.res_ub_fuse_split_result[0])

                self.emit_insn_dict[variance_tensor_ub] = {"insn": "", "axis": self.reduce1_ub_split_result[1],
                                                           "attr": {"extra_space": 1024}}
                self.emit_insn_dict[mid1_tensor_ub] = {
                    "insn": "",
                    "axis": mid1_tensor_ub.op.axis[self.block_split_axis_index],
                }
                self.emit_insn_dict[self.mid1_output_gm_tensor] = {
                    "insn": DMA_COPY,
                    "axis": self.mid1_output_gm_tensor.op.axis[self.block_split_axis_index],
                }
                self.emit_insn_dict[mid1_tensor_ub_back] = {
                    "insn": PHONY_INSN,
                    "axis": mid1_tensor_ub_back.op.axis[self.block_split_axis_index],
                }
                self.emit_insn_dict[special_tensor_1] = {
                    "insn": PHONY_INSN,
                    "axis": special_tensor_1.op.axis[self.block_split_axis_index],
                }
                self.schedule[sub_1_ub_back_res].compute_at(self.schedule[self.res_tensor], self.res_ub_split_result[0])
                self.emit_insn_dict[sub_1_ub_back_res] = {
                    "insn": DMA_COPY,
                    "axis": sub_1_ub_back_res.op.axis[0],
                }
                _compute_at_func(self.res_tensor, self.res_tensor, self.res_ub_split_result,
                                 [self.sub_gm_tensor, special_tensor_1], 0)
            else:
                variance_tensor_ub, variance_tensor_ub_back = self.mid_tensor_ub_back_dict[self.mid1_output_gm_tensor]
                sub_1_ub_back_var = self.input_tensor_2_dict[self.sub_to_mul0]

                sub_1_ub_back_res = self.input_tensor_2_dict[self.sub_to_mul1]

                self.schedule[sub_1_ub_back_var].compute_at(self.schedule[variance_tensor_ub],
                                                            self.reduce1_ub_split_result[0])
                self.emit_insn_dict[sub_1_ub_back_var] = {
                    "insn": DMA_COPY,
                    "axis": sub_1_ub_back_var.op.axis[0]
                }

                _compute_at_func(self.mid1_output_gm_tensor, variance_tensor_ub,
                                 self.reduce1_ub_split_result, [self.sub_gm_tensor], 0, False)

                # after reduce
                # reduce fuse loop:ub->gm->ub
                self.schedule[variance_tensor_ub].compute_at(self.schedule[self.res_tensor],
                                                             self.res_ub_fuse_split_result[0])
                self.schedule[self.mid1_output_gm_tensor].compute_at(self.schedule[self.res_tensor],
                                                                     self.res_ub_fuse_split_result[0])
                self.schedule[variance_tensor_ub_back].compute_at(self.schedule[self.res_tensor],
                                                                  self.res_ub_fuse_split_result[0])

                self.emit_insn_dict[variance_tensor_ub] = {"insn": "", "axis": self.reduce1_ub_split_result[1],
                                                           "attr": {"extra_space": 1024}}
                self.emit_insn_dict[self.mid1_output_gm_tensor] = {
                    "insn": DMA_COPY,
                    "axis": self.mid1_output_gm_tensor.op.axis[self.block_split_axis_index],
                }
                self.emit_insn_dict[variance_tensor_ub_back] = {
                    "insn": PHONY_INSN,
                    "axis": variance_tensor_ub_back.op.axis[self.block_split_axis_index],
                }
                self.schedule[sub_1_ub_back_res].compute_at(self.schedule[self.res_tensor], self.res_ub_split_result[0])
                self.emit_insn_dict[sub_1_ub_back_res] = {
                    "insn": DMA_COPY,
                    "axis": sub_1_ub_back_res.op.axis[0],
                }
                _compute_at_func(self.res_tensor, self.res_tensor, self.res_ub_split_result,
                                 [self.sub_gm_tensor, self.mid1_output_gm_tensor], 0)

            res_ub = self.tensor2tensorub[self.res_tensor]
            self.schedule[res_ub].compute_at(self.schedule[self.res_tensor], self.res_ub_split_result[0])
            self.emit_insn_dict[res_ub] = {"insn": "", "axis": res_ub.op.axis[0]}
            self.emit_insn_dict[self.res_tensor] = {"insn": DMA_COPY, "axis": self.res_ub_split_result[1]}

            for bg_ub in self.beta_gam_ub_list:
                self.schedule[bg_ub].compute_at(self.schedule[self.res_tensor], self.res_ub_split_result[0])
                self.emit_insn_dict[bg_ub] = {"insn": DMA_COPY, "axis": bg_ub.op.axis[0]}

        _do_compute_at_0()
        _do_compute_at_1()

    def _do_emit_insn(self):

        for tensor in self.emit_insn_dict:

            emit_insn_axis = self.emit_insn_dict[tensor]["axis"]
            insn = self.emit_insn_dict[tensor]["insn"]
            attr = self.emit_insn_dict[tensor].get("attr")
            if attr:
                storage_bound_num = attr.get("extra_space")
                if insn == PHONY_INSN:
                    self.schedule[tensor].emit_insn(emit_insn_axis, insn, attrs=dict(storage_bound=[storage_bound_num]))
                    continue
                elif insn == "":
                    insn = get_dsl_insn(tensor)
                self.schedule[tensor].emit_insn(emit_insn_axis, INSN_MAPPING[insn],
                                                attrs=dict(storage_bound=[storage_bound_num]))
            else:
                if insn == PHONY_INSN:
                    self.schedule[tensor].emit_insn(emit_insn_axis, insn)
                    continue
                elif insn == "":
                    insn = get_dsl_insn(tensor)
                self.schedule[tensor].emit_insn(emit_insn_axis, INSN_MAPPING[insn])
