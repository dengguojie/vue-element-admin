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
from ..base.operation import add_compile_info
from .constants import Pattern
from .constants import INSN_MAPPING
from .layer_norm_tilingcase import LayerNormInfo
from .layer_norm_tilingcase import LayerNormTilingCase
from .layer_norm_workspace_schedule import WorkspaceLayerNormSchedule
from . import util
from .util import get_dsl_insn

MAX_NODE_COUNT = 12
LAST_DIM_RANGE_NUM1 = 64
PHONY_INSN = "phony_insn"
DMA_COPY = "dma_copy"


@register_schedule(pattern=Pattern.LayerNorm)
def schedule(outs, tiling_case):
    """
    :param outs:
    :param tiling_case:
    :return:
    """
    graph_info = get_context().get_current_compute().get("compute_graph_info")
    layer_norm_info: LayerNormInfo = get_context().get_current_compute().get("layer_norm_info")
    is_normal = tiling_case.is_normal
    if is_normal:
        sch = NormalLayerNormSchedule(graph_info, layer_norm_info, outs)
    else:
        sch = WorkspaceLayerNormSchedule(graph_info, layer_norm_info, outs)
    real_schedule = sch.do_schedule(tiling_case)
    real_schedule.tiling_key = tiling_case.tiling_key
    return real_schedule


class NormalLayerNormSchedule:
    def __init__(self, graph_info, layer_norm_info, outs):
        self.graph_info = graph_info
        self.layer_norm_info = layer_norm_info
        # foward compute graph
        self.forward_compute_graph_map = self.graph_info.tensor_consumers_map
        self.dim_ln_range = operation.get_context().get_current_compute().get("dim_ln_range")
        self.outs = outs
        self.input_tensor_stage_list = []
        self.is_cast = False
        self.is_last_dim_one = False
        self.reduce0_tensor = None
        self.reduce1_tensor = None
        self.sub_to_mul0 = None
        self.sub_to_mul1 = None
        self.mid0_output_gm_tensor = None
        self.mid1_output_gm_tensor = None
        self.output_tensor_stage_list = []
        self.stage2tensor = {}
        self.tensor2stage = {}
        self.input_tensor_stage_list_1 = []
        self.inline_tensor_list = []
        self.emit_insn_dict = {}

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
        self._do_compute_inline()
        self._do_tiling()
        self._do_compute_at()
        self._do_storage_align()
        self._do_emit_insn()
        self._add_compile_info()
        return self.schedule

    def _do_create_schedule(self):
        self.schedule = tvm.create_schedule(
            tuple(self.graph_info.endpoint_output_tensor_set)[0].op)

    def _do_set_var_range(self):
        res_tensor = tuple(self.graph_info.endpoint_output_tensor_set)[0]
        self.res_shape = res_tensor.shape
        if tuple(self.dim_ln_range) == (1, 1):
            self.is_last_dim_one = True
        if isinstance(self.res_shape[-1], tvm.expr.Var):
            self.schedule.set_var_range(self.res_shape[-1], *self.dim_ln_range)

    def _do_cache_read_write(self):
        def _do_cache_read():
            for input_tensor in self.graph_info.input_tensor_set:
                input_tensor_stage = self.schedule.cache_read(input_tensor, tbe_platform_info.scope_ubuf,
                                                              self.forward_compute_graph_map[input_tensor])
                self.input_tensor_stage_list.append(input_tensor_stage)

        def _do_cache_write():
            for output_tensor in self.graph_info.output_tensor_set:
                output_tensor_stage = self.schedule.cache_write(output_tensor,
                                                                tbe_platform_info.scope_ubuf)
                self.output_tensor_stage_list.append(output_tensor_stage)
                self.stage2tensor[output_tensor_stage] = output_tensor
                self.tensor2stage[output_tensor] = output_tensor_stage
            # fit workspace case
            self.sub_tensor_ub = self.schedule.cache_write(
                self.sub_gm_tensor, tbe_platform_info.scope_ubuf)
            self.sub_gm_tensor_back = self.schedule.cache_read(self.sub_gm_tensor, tbe_platform_info.scope_ubuf,
                                                               [self.sub_to_mul0, self.sub_to_mul1])
            self.schedule[self.sub_gm_tensor_back].reused_by(self.sub_tensor_ub)

            # reused_by mid_tensor
            for mid_ten in self.graph_info.mid_output_tensor_set:
                mid_tensor = self.schedule.cache_read(mid_ten, tbe_platform_info.scope_ubuf,
                                                      self.forward_compute_graph_map[mid_ten])

                self.schedule[mid_tensor].reused_by(self.tensor2stage[mid_ten])
                self.input_tensor_stage_list_1.append(mid_tensor)

        # cache read
        _do_cache_read()
        # cache_write
        _do_cache_write()

    def _do_set_scope(self):
        for tensor in self.forward_compute_graph_map:
            if tensor not in tuple(
                    self.graph_info.output_tensor_set | self.graph_info.input_tensor_set) and tensor != self.sub_gm_tensor:
                self.schedule[tensor].set_scope(tbe_platform_info.scope_ubuf)

    def _do_compute_inline(self):
        # self.shape_dim > 1 operate storage align
        if self.shape_dim > 1:
            for tensor in self.graph_info.input_tensor_set:
                opname = tensor.op.name
                if opname in ("gamma", "beta") and self.is_cast:
                    bc_tensor = tuple(self.forward_compute_graph_map[tuple(
                        self.forward_compute_graph_map[tensor])[0]])[0]
                    self.schedule[bc_tensor].compute_inline()
                    self.inline_tensor_list.append(bc_tensor)

                elif opname in ("gamma", "beta") and not self.is_cast:
                    bc_tensor = tuple(self.forward_compute_graph_map[tensor])[0]
                    self.schedule[bc_tensor].compute_inline()
                    self.inline_tensor_list.append(bc_tensor)

    def _do_storage_bound(self):
        UB_SIZE = get_soc_spec("UB_SIZE")
        # analysis coexit tensor node
        tensor_storage_bound_set = set(self.forward_compute_graph_map.keys()) | set(
            self.output_tensor_stage_list) | set(self.input_tensor_stage_list) | set(self.input_tensor_stage_list_1)
        if self.is_cast:
            # fp16 --> fp32
            self.max_ub_size = int(UB_SIZE / MAX_NODE_COUNT / 2)
        else:
            # fp32
            self.max_ub_size = int(UB_SIZE / MAX_NODE_COUNT / 2)
        for stage_tensor in tensor_storage_bound_set:
            if self.forward_compute_graph_map.get(stage_tensor) and stage_tensor in self.graph_info.input_tensor_set:
                continue
            self.schedule[stage_tensor].set_storage_bound(self.max_ub_size)
        self.schedule[self.sub_tensor_ub].set_storage_bound(self.max_ub_size)
        self.schedule[self.sub_gm_tensor_back].set_storage_bound(self.max_ub_size)

    def _do_storage_align(self):
        case = self.tiling_case
        input_format = case.format
        if input_format != "FRACTAL_NZ":
            tensor_storage_bound_set = set(self.forward_compute_graph_map.keys()) | set(
                self.output_tensor_stage_list) | set(self.input_tensor_stage_list)
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

            # self.shape_dim > 1 operate storage align  and last dim is one
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
                self.schedule[self.sub_tensor_ub].storage_align(self.sub_tensor_ub.op.axis[-2], align_num, 0)
                self.schedule[self.sub_gm_tensor_back].storage_align(self.sub_gm_tensor_back.op.axis[-2], align_num, 0)

    def _do_tiling(self):
        case = self.tiling_case
        reduce_axis_list = case.reduce_axis_list
        input_format = case.format

        # get last endpoint output tensor
        res_tensor = tuple(self.graph_info.endpoint_output_tensor_set)[0]

        if self.dim_ln_range[-1] is not None and self.dim_ln_range[-1] <= LAST_DIM_RANGE_NUM1:
            ub_factor_range = (1, None)
        else:
            ub_factor_range = (1, 255)

        # get tiling axis
        block_split_axis_index = case.block_split_axis_index
        block_split_axis_index_1 = case.block_split_axis_index_1
        ub_split_axis_index = case.ub_split_axis_index
        self.ub_split_axis_index = ub_split_axis_index
        ub_fuse_factor = case.ub_fuse_factor

        # Get tiling params
        block_factor = case.block_factor
        block_inner = block_factor if block_factor is not None else var(
            "block_factor", (1, None))
        block_factor_1 = case.block_factor_1
        block_inner_1 = block_factor_1 if block_factor_1 is not None else var(
            "block_factor_1", (1, None))

        res_axis = []
        for d_i in range(len(res_tensor.shape)):
            res_axis.append(res_tensor.op.axis[d_i])

        ub_factor = case.ub_factor
        ub_inner = ub_factor if ub_factor is not None else var(
            "ub_factor", ub_factor_range)
        ub_fuse_inner = ub_fuse_factor if ub_fuse_factor is not None else var("ub_fuse_factor", (0, 0))
        ub_inner = ub_inner + ub_fuse_inner

        # block tiling

        if case.is_split_ub:
            if block_split_axis_index != block_split_axis_index_1:
                pre_outer, pre_inner = self.schedule[res_tensor].split(res_tensor.op.axis[block_split_axis_index],
                                                                       factor=block_inner)
                suf_outer, suf_inner = self.schedule[res_tensor].split(res_tensor.op.axis[block_split_axis_index_1],
                                                                       nparts=block_inner_1)

                res_axis[block_split_axis_index] = pre_outer
                res_axis[block_split_axis_index_1] = suf_inner
                fuse_axis = self.schedule[res_tensor].fuse(pre_inner, suf_outer)

                ub_outer, ub_inner = self.schedule[res_tensor].split(res_axis[ub_split_axis_index],
                                                                     factor=ub_inner)

                axis_order = [fuse_axis]
                # add special NZ reorder func
                if input_format == "FRACTAL_NZ":
                    for idx, r_axis in enumerate(res_axis):
                        if idx == ub_split_axis_index:
                            axis_order.append(ub_outer)
                            axis_order.append(ub_inner)
                        elif idx not in reduce_axis_list:
                            axis_order.append(r_axis)
                        elif idx != reduce_axis_list[-1]:
                            continue
                        else:
                            break
                    # reorder separated reduce axis together
                    # append reduce axis
                    for rai in reduce_axis_list[:-1]:
                        axis_order.append(res_axis[rai])
                    axis_order = axis_order + res_axis[reduce_axis_list[-1]:]
                else:
                    for idx, r_axis in enumerate(res_axis):
                        if idx == ub_split_axis_index:
                            axis_order.append(ub_outer)
                            axis_order.append(ub_inner)
                        else:
                            axis_order.append(r_axis)
                self.schedule[res_tensor].reorder(*axis_order)
                block_outer, block_inner = fuse_axis, axis_order[1]
            else:
                block_outer, block_inner = self.schedule[res_tensor].split(res_tensor.op.axis[block_split_axis_index],
                                                                           factor=block_inner)
                ub_outer, ub_inner = self.schedule[res_tensor].split(res_axis[ub_split_axis_index], factor=ub_inner)
                axis_order = []
                if input_format == "FRACTAL_NZ":
                    for idx, r_axis in enumerate(res_axis):
                        if idx == block_split_axis_index:
                            axis_order.append(block_outer)
                            axis_order.append(block_inner)
                        elif idx == ub_split_axis_index:
                            axis_order.append(ub_outer)
                            axis_order.append(ub_inner)
                        elif idx not in reduce_axis_list:
                            axis_order.append(r_axis)
                        elif idx != reduce_axis_list[-1]:
                            continue
                        else:
                            break
                    # reorder separated reduce axis together
                    # append reduce axis
                    for rai in reduce_axis_list[:-1]:
                        axis_order.append(res_axis[rai])
                    axis_order = axis_order + res_axis[reduce_axis_list[-1]:]
                    self.schedule[res_tensor].reorder(*axis_order)
        else:
            block_outer, block_inner = self.schedule[res_tensor].split(res_tensor.op.axis[block_split_axis_index],
                                                                       factor=block_inner)
            ub_outer, ub_inner = self.schedule[res_tensor].split(
                block_inner, factor=ub_inner)
            axis_order = []
            if input_format == "FRACTAL_NZ":
                for idx, r_axis in enumerate(res_axis):
                    if idx == ub_split_axis_index:
                        axis_order.append(block_outer)
                        axis_order.append(ub_outer)
                        axis_order.append(ub_inner)
                    elif idx not in reduce_axis_list:
                        axis_order.append(r_axis)
                    elif idx != reduce_axis_list[-1]:
                        continue
                    else:
                        break
                # reorder separated reduce axis together
                # append reduce axis
                for rai in reduce_axis_list[:-1]:
                    axis_order.append(res_axis[rai])
                axis_order = axis_order + res_axis[reduce_axis_list[-1]:]
                self.schedule[res_tensor].reorder(*axis_order)
        self.block_spit_result = [block_outer, block_inner]
        self.ub_split_result = [ub_outer, ub_inner]

    def _do_compute_at(self):
        case = self.tiling_case

        # get last endpoint output tensor
        res_tensor = tuple(self.graph_info.endpoint_output_tensor_set)[0]

        if not self.is_cast:
            non_compute_at_tensor_list = [res_tensor, self.sub_gm_tensor] + self.inline_tensor_list
            for tensor in self.forward_compute_graph_map:
                if tensor not in non_compute_at_tensor_list:
                    self.schedule[tensor].compute_at(
                        self.schedule[res_tensor], self.ub_split_result[0])
            self.schedule[self.sub_tensor_ub].compute_at(
                self.schedule[res_tensor], self.ub_split_result[0])
            self.schedule[self.sub_gm_tensor].compute_at(
                self.schedule[res_tensor], self.ub_split_result[0])
            self.schedule[self.sub_gm_tensor_back].compute_at(
                self.schedule[res_tensor], self.ub_split_result[0])
            for tensor in self.stage2tensor:
                self.schedule[tensor].compute_at(
                    self.schedule[res_tensor], self.ub_split_result[0])
            for stage_tensor in self.input_tensor_stage_list:
                self.schedule[stage_tensor].compute_at(
                    self.schedule[res_tensor], self.ub_split_result[0])
            for input_tensor_stage in self.input_tensor_stage_list_1:
                self.schedule[input_tensor_stage].compute_at(
                    self.schedule[res_tensor], self.ub_split_result[0])
        else:
            for tensor in self.forward_compute_graph_map:
                if tensor in self.inline_tensor_list:
                    continue
                if tensor not in (self.graph_info.input_tensor_set | self.graph_info.output_tensor_set):

                    self.schedule[tensor].compute_at(
                        self.schedule[res_tensor], self.ub_split_result[0])
                    self.emit_insn_dict[tensor] = {
                        "insn": "",
                        "axis": tensor.op.axis[0]
                    }
                elif tensor in self.graph_info.output_tensor_set and tensor != res_tensor:
                    self.schedule[tensor].compute_at(
                        self.schedule[res_tensor], self.ub_split_result[0])
                    self.emit_insn_dict[tensor] = {
                        "insn": DMA_COPY,
                        "axis": tensor.op.axis[0]
                    }
                elif tensor == res_tensor:
                    self.emit_insn_dict[tensor] = {
                        "insn": DMA_COPY,
                        "axis": self.ub_split_result[1]
                    }
            self.schedule[self.sub_tensor_ub].compute_at(
                self.schedule[res_tensor], self.ub_split_result[0])
            self.schedule[self.sub_gm_tensor].compute_at(
                self.schedule[res_tensor], self.ub_split_result[0])
            self.schedule[self.sub_gm_tensor_back].compute_at(
                self.schedule[res_tensor], self.ub_split_result[0])
            self.emit_insn_dict[self.sub_tensor_ub] = {
                "insn": "",
                "axis": self.sub_tensor_ub.op.axis[0]
            }
            self.emit_insn_dict[self.sub_gm_tensor] = {
                "insn": PHONY_INSN,
                "axis": self.sub_gm_tensor.op.axis[0]
            }
            self.emit_insn_dict[self.sub_gm_tensor_back] = {
                "insn": PHONY_INSN,
                "axis": self.sub_gm_tensor_back.op.axis[0]
            }

            # cast fp16->fp32
            special_tensor_0 = tuple(
                self.graph_info.tensor_consumers_map[self.mid0_output_gm_tensor])[0]
            self.schedule[special_tensor_0].reused_by(self.reduce0_tensor)

            special_tensor_1 = tuple(
                self.graph_info.tensor_consumers_map[self.mid1_output_gm_tensor])[0]
            self.schedule[special_tensor_1].reused_by(self.reduce1_tensor)
            self.emit_insn_dict[special_tensor_0] = {
                "insn": PHONY_INSN,
                "axis": special_tensor_0.op.axis[0]
            }
            self.emit_insn_dict[special_tensor_1] = {
                "insn": PHONY_INSN,
                "axis": special_tensor_1.op.axis[0]
            }
            for tensor in self.stage2tensor:
                self.schedule[tensor].compute_at(
                    self.schedule[res_tensor], self.ub_split_result[0])
                self.emit_insn_dict[tensor] = {
                    "insn": "",
                    "axis": tensor.op.axis[0]
                }
            for stage_tensor in self.input_tensor_stage_list:
                self.schedule[stage_tensor].compute_at(
                    self.schedule[res_tensor], self.ub_split_result[0])
                self.emit_insn_dict[stage_tensor] = {
                    "insn": DMA_COPY,
                    "axis": stage_tensor.op.axis[0]
                }
            for input_tensor_stage in self.input_tensor_stage_list_1:
                self.schedule[input_tensor_stage].compute_at(
                    self.schedule[res_tensor], self.ub_split_result[0])
                self.emit_insn_dict[input_tensor_stage] = {
                    "insn": PHONY_INSN,
                    "axis": input_tensor_stage.op.axis[0]
                }
            self.emit_insn_dict[self.reduce0_tensor]["attr"] = {"extra_space": 1024}
            self.emit_insn_dict[self.reduce1_tensor]["attr"] = {"extra_space": 1024}

        # judge open multi_core
        if case.multi_core is None:
            raise RuntimeError("Tilingcase didn`t declare multi_core switch")
        if case.multi_core:
            block = tvm.thread_axis("blockIdx.x")
            self.schedule[res_tensor].bind(self.block_spit_result[0], block)

    def _do_emit_insn(self):
        # emit_insn
        if not self.is_cast:

            res_tensor = tuple(self.graph_info.endpoint_output_tensor_set)[0]
            for tensor in self.forward_compute_graph_map:
                if tensor in self.inline_tensor_list:
                    continue
                if tensor == self.sub_gm_tensor:
                    tensor = self.sub_tensor_ub
                if tensor in self.graph_info.input_tensor_set:
                    continue

                insn = get_dsl_insn(tensor)
                emit_insn_axis = tensor.op.axis[0]

                if tensor == res_tensor:
                    emit_insn_axis = self.ub_split_result[1]
                    insn = DMA_COPY
                if tensor in tuple(self.graph_info.output_tensor_set):
                    insn = DMA_COPY
                if insn == "":
                    insn = DMA_COPY
                self.schedule[tensor].emit_insn(emit_insn_axis, INSN_MAPPING[insn])
            self.schedule[self.sub_gm_tensor].emit_insn(self.sub_gm_tensor.op.axis[0], PHONY_INSN)
            self.schedule[self.sub_gm_tensor_back].emit_insn(self.sub_gm_tensor_back.op.axis[0], PHONY_INSN)
            # emit insn read and write
            for input_tensor_stage in self.input_tensor_stage_list:
                emit_insn_axis = input_tensor_stage.op.axis[0]
                self.schedule[input_tensor_stage].emit_insn(
                    emit_insn_axis, DMA_COPY)
            for input_tensor_stage in self.input_tensor_stage_list_1:
                emit_insn_axis = input_tensor_stage.op.axis[0]
                self.schedule[input_tensor_stage].emit_insn(
                    emit_insn_axis, PHONY_INSN)

            for output_tensor_stage in self.output_tensor_stage_list:
                emit_insn_axis = output_tensor_stage.op.axis[0]
                output_tensor = self.stage2tensor[output_tensor_stage]

                insn = get_dsl_insn(output_tensor)
                if output_tensor in (self.reduce0_tensor, self.reduce1_tensor):
                    self.schedule[output_tensor_stage].emit_insn(
                        emit_insn_axis, INSN_MAPPING[insn], attrs=dict(storage_bound=[1024]))
                else:
                    self.schedule[output_tensor_stage].emit_insn(
                        emit_insn_axis, INSN_MAPPING[insn])
        else:
            for tensor in self.emit_insn_dict:
                emit_insn_axis = self.emit_insn_dict[tensor]["axis"]
                insn = self.emit_insn_dict[tensor]["insn"]
                attr = self.emit_insn_dict[tensor].get("attr")
                if attr:
                    storage_bound_num = attr.get("extra_space")
                    if insn == PHONY_INSN:
                        self.schedule[tensor].emit_insn(
                            emit_insn_axis, insn, attrs=dict(storage_bound=[storage_bound_num]))
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

    def _add_compile_info(self):
        add_compile_info("max_ub_size_normal_fp16", 10 * 1024)
        add_compile_info("max_ub_size_normal_fp32", 10 * 1024)
