#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
extract image patches schedule
"""

from functools import reduce
import math
from tbe import tvm
from tbe.common import platform as tbe_platform
import te.platform as te_platform
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base.operation import var

from . import util
from .constants import Pattern
from .extract_image_patches_tilingcase import TilingStrategy


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    BLOCK_SIZE = 16
    BLOCK_SIZE_INT8 = 32

    DOUBLE_BUFFER = 2
    FP16_SIZE = 2
    INT8_SIZE = 1
    NEED_UB_SPACE_NUM = 2
    SIZE_L1 = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    SIZE_UB = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    LOAD3D_REPEAT_TIME_LIMIT = 255
    FACTOR_N_CEIL = 128
    # `avg_split_ub_size // (ceil((max shape h*w*c can cut by howo col or row) / align_block_size))
    DMA_SPILT_ELEMENT_THRESHOLD = 9069
    DELTA = 0.000001  # aviod div zero, fp32 precision


@register_schedule(pattern=Pattern.EXTRACT_IMAGE_PATCHES)
def schedule(outs, tiling_case):
    """
    schedule for extract_image_patch dynamic shape
    """
    return ExtractImagePatchesSchedule(outs, tiling_case).do_schedule()


class ExtractImagePatchesSchedule:
    """
    ExtractImagePatchesSchedule
    """
    def __init__(self, outs, tiling_case):
        self.output_res = outs[0]
        self._sch = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._tiling_key = self._tiling_case.get("key")
        self.align_block_size = 0
        self.avg_split_ub_size = 0
        self.used_ub_size = 0
        self.type_size = 0
        self.fmap_n = 0
        self.fmap_c1 = 0
        self.fmap_h = 0
        self.fmap_w = 0
        self.fmap_c0 = 0
        self.howo = 0
        self.khkw = 0
        self.dilated_kernel_h = 0
        self.dilated_kernel_w = 0
        self.cut_h_col = 0
        self.cut_h_row_s = 0
        self.cut_h_row = 0
        self.cut_w_row = 0
        self.lcm_out_w = 0
        self.origin_c_in = 0
        self.out_w = 0
        self.c_out = 0
        self.kernel_w = 0
        self.dtype_input = ""
        self.device_core_num = 0
        self.pre_com_fmap_w_c0 = 0
        self.pre_com_fmap_c1_w_c0 = 0
        self.multi_core_factor_0 = 0
        self.move_rate_cut_col = 0.0
        self.extract_params = {}
        self.out_shape = []
        self.setfmatrix_dict = {}

    # 'pylint: disable=too-many-statements,too-many-branches,too-many-locals,too-many-lines
    def do_schedule(self):
        """
        do schedule

        Returns:
            schedule
        """
        self._sch = tvm.create_schedule(self.output_res.op)
        self._sch.tiling_key = self._tiling_key

        setfmatrix_map = self.output_res.op.attrs["setfmatrix_dict"]
        for key, value in setfmatrix_map.items():
            if hasattr(value, "value"):
                self.setfmatrix_dict[key] = value.value
            else:
                self.setfmatrix_dict[key] = value

        extract_map = self.output_res.op.attrs["extract_params"]
        for key, value in extract_map.items():
            if hasattr(value, "value"):
                self.extract_params[key] = value.value
            else:
                self.extract_params[key] = value

        self.multi_core_factor_0 = var("multi_core_factor_0")
        fmap_shape = self.extract_params.get("fmap_shape")
        self.fmap_n = fmap_shape[0]
        self._sch.set_var_range(self.multi_core_factor_0, 3, 20)
        self.fmap_c1 = fmap_shape[1].value
        self.fmap_h = fmap_shape[2].value
        self.fmap_w = fmap_shape[3].value
        self.fmap_c0 = fmap_shape[4].value

        out_h = self.extract_params.get("out_h")
        self.out_w = self.extract_params.get("out_w")
        self.origin_c_in = self.extract_params.get("origin_c_in")

        kernel_h = self.setfmatrix_dict.get("conv_kernel_h")
        self.kernel_w = self.setfmatrix_dict.get("conv_kernel_w")
        dilate_h = self.setfmatrix_dict.get("conv_dilation_h")
        dilate_w = self.setfmatrix_dict.get("conv_dilation_w")
        stride_h = self.setfmatrix_dict.get("conv_stride_h")
        stride_w = self.setfmatrix_dict.get("conv_stride_w")

        graph_tensors = {}
        graph_tensors["ub_res"] = self.output_res.op.input_tensors[0]
        graph_tensors["workspace_res"] = graph_tensors.get("ub_res").op.input_tensors[0]
        graph_tensors["merge_co_ub"] = graph_tensors.get("workspace_res").op.input_tensors[0]
        graph_tensors["merge_hw_ub"] = graph_tensors.get("merge_co_ub").op.input_tensors[0]
        graph_tensors["transpose_ub"] = graph_tensors.get("merge_hw_ub").op.input_tensors[0]
        graph_tensors["split_c1_ub"] = graph_tensors.get("transpose_ub").op.input_tensors[0]
        graph_tensors["fmap_fractal"] = graph_tensors.get("split_c1_ub").op.input_tensors[0]
        graph_tensors["fmap_in_l1"] = graph_tensors.get("fmap_fractal").op.input_tensors[0]

        self._sch[graph_tensors.get("fmap_in_l1")].set_scope(tbe_platform.scope_cbuf)
        self._sch[graph_tensors.get("fmap_fractal")].set_scope(tbe_platform.scope_ubuf)
        self._sch[graph_tensors.get("split_c1_ub")].set_scope(tbe_platform.scope_ubuf)
        self._sch[graph_tensors.get("transpose_ub")].set_scope(tbe_platform.scope_ubuf)
        self._sch[graph_tensors.get("merge_hw_ub")].set_scope(tbe_platform.scope_ubuf)
        self._sch[graph_tensors.get("merge_co_ub")].set_scope(tbe_platform.scope_ubuf)
        self._sch[graph_tensors.get("ub_res")].set_scope(tbe_platform.scope_ubuf)

        workspace_shape = [int(graph_tensors.get("workspace_res").shape[i]) for i in range(1, 4)]

        self.dtype_input = graph_tensors.get("ub_res").dtype
        if self.dtype_input in ("int8", "uint8"):
            self.align_block_size = Constant.BLOCK_SIZE_INT8
            self.type_size = Constant.INT8_SIZE
        else:
            self.align_block_size = Constant.BLOCK_SIZE
            self.type_size = Constant.FP16_SIZE
        self.pre_com_fmap_w_c0 = self.fmap_w * self.fmap_c0 * self.type_size * Constant.DOUBLE_BUFFER
        self.pre_com_fmap_c1_w_c0 = self.fmap_c1 * self.pre_com_fmap_w_c0

        out_hw_up16 = ((out_h * self.out_w - 1) // Constant.BLOCK_SIZE + 1) * Constant.BLOCK_SIZE
        self.dilated_kernel_h = (kernel_h - 1) * dilate_h + 1
        self.dilated_kernel_w = (self.kernel_w - 1) * dilate_w + 1
        self.lcm_out_w = Constant.BLOCK_SIZE // math.gcd(self.out_w, Constant.BLOCK_SIZE) * self.out_w
        self.cut_h_col = (Constant.BLOCK_SIZE // math.gcd(self.out_w, Constant.BLOCK_SIZE) - 1) \
            * stride_h + 1 + self.dilated_kernel_h // 2
        if self.cut_h_col > self.fmap_h:
            self.cut_h_col = self.fmap_h
        # `cut_h_col while cut_hw = Constant.BLOCK_SIZE`
        cut_w_row_s = (Constant.BLOCK_SIZE - 1) * stride_w + 1
        self.cut_h_row_s = max(stride_h, (((cut_w_row_s - 1) // self.fmap_w + 1) - 1) * stride_h + 1)
        self.cut_w_row = cut_w_row_s + self.dilated_kernel_w - 1
        self.cut_h_row = self.cut_h_row_s + self.dilated_kernel_h - 1
        if self.lcm_out_w > out_hw_up16:
            self.lcm_out_w = out_hw_up16

        self.extract_params["cut_h_row"] = self.cut_h_row

        self._sch[graph_tensors.get("ub_res")].buffer_align((1, 1), (1, 1), (1, 1), (1, self.align_block_size))
        self._sch[graph_tensors.get("fmap_fractal")].buffer_align((1, 1), (1, 1), (1, 1), (1, Constant.BLOCK_SIZE),
                                             (1, self.align_block_size))

        self.used_ub_size = Constant.SIZE_UB // self.type_size // Constant.DOUBLE_BUFFER
        self.avg_split_ub_size = self.used_ub_size // Constant.NEED_UB_SPACE_NUM
        self.howo = out_h * self.out_w
        self.khkw = kernel_h * self.kernel_w
        self.c_out = self.khkw * self.fmap_c1 * self.fmap_c0

        self.out_shape = [self.fmap_n, self.howo, self.khkw, self.origin_c_in]
        self.device_core_num = util.get_core_num()
        self.setfmatrix_dict["set_fmatrix"] = 1
        self.setfmatrix_dict["conv_fm_c1"] = self.fmap_c1
        self.setfmatrix_dict["conv_fm_c0"] = self.fmap_c0
        self.setfmatrix_dict["group_flag"] = 1
        self.setfmatrix_dict["l1_group_flag"] = 1
        self.setfmatrix_dict["enable_load3dv2"] = 1

        self._do_tiling(graph_tensors)

        self._sch[graph_tensors.get("fmap_in_l1")].double_buffer()
        self._sch[graph_tensors.get("fmap_fractal")].double_buffer()
        self._sch[graph_tensors.get("transpose_ub")].double_buffer()
        self._sch[graph_tensors.get("ub_res")].double_buffer()
        self._add_compile_info(workspace_shape)
        return self._sch

    def _do_tiling(self, graph_tensors):
        funcs = {TilingStrategy.AXIS_ALIGN: self._do_tiling_axis_align,
                 TilingStrategy.AXIS_NOT_ALIGN: self._do_tiling_axis_not_align}
        funcs.get(self._tiling_strategy)(graph_tensors)

    def _do_tiling_axis_align(self, graph_tensors):
        tiling_param = self._get_tiling_param()
        tiling_factor_and_rate = self._get_tiling_factor(tiling_param, align=True)
        n_factor, howo_factor, khkw_factor, c_factor, split_khkw_mode, move_rate = tiling_factor_and_rate
        tiling_factor = [n_factor, howo_factor, khkw_factor, c_factor]
        res_bind_list, res_axis_list = self._split_multi_core_32b_align(tiling_factor)
        res_axis_outer_list, res_axis_inner_list = self._get_axis_list(self.output_res,
                                                                       res_axis_list,
                                                                       tiling_factor)

        if Constant.SIZE_L1 >= self.fmap_h * self.pre_com_fmap_c1_w_c0:
            compute_at_id = 0
        elif Constant.SIZE_L1 >= self.cut_h_row * self.pre_com_fmap_c1_w_c0 and move_rate != self.move_rate_cut_col:
            compute_at_id = 1
        elif Constant.SIZE_L1 >= self.cut_h_col * self.pre_com_fmap_c1_w_c0 and move_rate == self.move_rate_cut_col:
            compute_at_id = 1
        elif Constant.SIZE_L1 >= self.cut_h_col * self.pre_com_fmap_w_c0 and move_rate == self.move_rate_cut_col:
            compute_at_id = 1
            res_c_out_outer, res_axis_outer_list[3] = self._sch[self.output_res].split(res_axis_outer_list[3],
                                                                                        factor=1)
            res_bind_list.append(res_c_out_outer)
        elif Constant.SIZE_L1 >= self.cut_h_row_s * self.pre_com_fmap_w_c0 and split_khkw_mode:
            compute_at_id = 2
            res_axis_outer_list[2], _ = self._sch[self.output_res].split(res_axis_outer_list[2],
                                                                         factor=max(self.kernel_w //
                                                                                    khkw_factor, 1))
            res_c_out_outer, res_axis_outer_list[3] = self._sch[self.output_res].split(res_axis_outer_list[3],
                                                                                       factor=1)
            res_bind_list.append(res_c_out_outer)
        else:
            compute_at_id = 3

        self._sch[self.output_res].reorder(*(res_bind_list + res_axis_outer_list + res_axis_inner_list))

        self._sch[graph_tensors["fmap_in_l1"]].compute_at(self._sch[self.output_res],
                                                          res_axis_outer_list[compute_at_id])
        self._sch[graph_tensors["fmap_fractal"]].compute_at(self._sch[self.output_res], res_axis_outer_list[3])
        self._sch[graph_tensors["transpose_ub"]].compute_at(self._sch[self.output_res], res_axis_outer_list[3])

        self._sch[graph_tensors["workspace_res"]].compute_inline()
        self._sch[graph_tensors["ub_res"]].compute_inline()
        self._sch[graph_tensors["merge_co_ub"]].compute_inline()
        self._sch[graph_tensors["merge_hw_ub"]].compute_inline()
        self._sch[graph_tensors["split_c1_ub"]].compute_inline()

        block = tvm.thread_axis("blockIdx.x")
        self._sch[self.output_res].bind(res_bind_list[0], block)

        self._sch[graph_tensors["fmap_in_l1"]].emit_insn(
            graph_tensors["fmap_in_l1"].op.axis[0], te_platform.DMA_COPY)
        self._sch[graph_tensors["fmap_fractal"]].emit_insn(
            graph_tensors["fmap_fractal"].op.axis[0], "im2col_v2", self.setfmatrix_dict)
        self._sch[graph_tensors["transpose_ub"]].emit_insn(graph_tensors["transpose_ub"].op.axis[0],
                                                           te_platform.DMA_COPY)
        self._sch[self.output_res].emit_insn(res_axis_inner_list[0], te_platform.DMA_COPY)

    def _do_tiling_axis_not_align(self, graph_tensors):
        # 'pylint: disable=too-many-branches,too-many-lines
        def _schedule_32b_not_aligned(dma_split_axis_id, dma_split_factor, allow_multi_core, graph_tensors,
                                      reg_mov=True):
            """
            schedule, when 32B is not aligned
            """
            tiling_param = self._get_tiling_param()
            tiling_factor_and_rate = self._get_tiling_factor(tiling_param, align=False)
            n_factor, howo_factor, khkw_factor, c_factor, _, move_rate = tiling_factor_and_rate
            tiling_factor = [n_factor, howo_factor, khkw_factor, c_factor]
            if reg_mov:
                reg_mov_ub = self._sch.cache_write(self.output_res, tbe_platform.scope_ubuf)
            if allow_multi_core:
                multi_core_factor = self._get_multi_core_factor_32b_not_aligned(dma_split_axis_id, tiling_factor)
            else:
                multi_core_factor = self.out_shape.copy()

            split_multi_core_axis_list = self._split_multi_core_32b_not_aligned(multi_core_factor,
                                                                                dma_split_axis_id,
                                                                                dma_split_factor,
                                                                                graph_tensors["workspace_res"])
            res_bind_list, res_axis_list, workspace_bind_list, workspace_axis_list, dma_copy_axis = \
                split_multi_core_axis_list

            workspace_axis_outer_list, workspace_axis_inner_list = self._get_axis_list(graph_tensors["workspace_res"],
                                                                                       workspace_axis_list,
                                                                                       tiling_factor)

            if Constant.SIZE_L1 >= self.fmap_h * self.pre_com_fmap_c1_w_c0:
                compute_at_id = 0
            elif Constant.SIZE_L1 >= self.cut_h_row * self.pre_com_fmap_c1_w_c0 and move_rate != self.move_rate_cut_col:
                compute_at_id = 1
            elif Constant.SIZE_L1 >= self.cut_h_col * self.pre_com_fmap_c1_w_c0 and move_rate == self.move_rate_cut_col:
                compute_at_id = 1
            elif Constant.SIZE_L1 >= self.cut_h_col * self.pre_com_fmap_w_c0 and move_rate == self.move_rate_cut_col:
                compute_at_id = 1
                workspace_c_out_outer, workspace_axis_outer_list[3] = self._sch[graph_tensors["workspace_res"]].split(
                    workspace_axis_outer_list[3], factor=1)
                workspace_bind_list.append(workspace_c_out_outer)
            else:
                compute_at_id = 2
                workspace_c_out_outer, workspace_axis_outer_list[3] = self._sch[graph_tensors["workspace_res"]].split(
                    workspace_axis_outer_list[3], factor=1)
                workspace_bind_list.append(workspace_c_out_outer)
                workspace_axis_outer_list[2], _ = self._sch[graph_tensors["workspace_res"]].split(
                    workspace_axis_outer_list[2], factor=max(self.kernel_w // tiling_factor[2], 1))

            self._sch[graph_tensors["workspace_res"]].reorder(
                *(workspace_bind_list + workspace_axis_outer_list + workspace_axis_inner_list))

            self._sch[graph_tensors["fmap_in_l1"]].compute_at(self._sch[graph_tensors["workspace_res"]],
                                                             workspace_axis_outer_list[compute_at_id])

            self._sch[graph_tensors["merge_co_ub"]].compute_at(self._sch[graph_tensors["workspace_res"]],
                                                              workspace_axis_outer_list[3])
            self._sch[graph_tensors["merge_hw_ub"]].compute_at(self._sch[graph_tensors["workspace_res"]],
                                                              workspace_axis_outer_list[3])
            self._sch[graph_tensors["transpose_ub"]].compute_at(self._sch[graph_tensors["workspace_res"]],
                                                               workspace_axis_outer_list[3])
            self._sch[graph_tensors["split_c1_ub"]].compute_at(self._sch[graph_tensors["workspace_res"]],
                                                              workspace_axis_outer_list[3])
            self._sch[graph_tensors["fmap_fractal"]].compute_at(self._sch[graph_tensors["workspace_res"]],
                                                               workspace_axis_outer_list[3])

            self._sch[graph_tensors["ub_res"]].compute_at(self._sch[self.output_res], res_axis_list[dma_split_axis_id])
            if reg_mov:
                self._sch[reg_mov_ub].compute_at(self._sch[self.output_res], res_axis_list[dma_split_axis_id])

            block = tvm.thread_axis("blockIdx.x")
            self._sch[self.output_res].bind(res_bind_list[0], block)
            self._sch[graph_tensors["workspace_res"]].bind(workspace_bind_list[0], block)

            self._sch[graph_tensors["split_c1_ub"]].compute_inline()
            self._sch[graph_tensors["merge_co_ub"]].compute_inline()

            self._sch[graph_tensors["fmap_in_l1"]].emit_insn(graph_tensors["fmap_in_l1"].op.axis[0],
                                                             te_platform.DMA_COPY)
            self._sch[graph_tensors["fmap_fractal"]].emit_insn(graph_tensors["fmap_fractal"].op.axis[0],
                                                               "im2col_v2", self.setfmatrix_dict)
            self._sch[graph_tensors["split_c1_ub"]].emit_insn(graph_tensors["split_c1_ub"].op.axis[0],
                                                              te_platform.DMA_COPY)

            if self.dtype_input in ("int8", "uint8"):
                self._sch[graph_tensors["transpose_ub"]].emit_insn(graph_tensors["transpose_ub"].op.axis[0],
                                                                   te_platform.DMA_COPY)
                self._sch[graph_tensors["merge_hw_ub"]].emit_insn(graph_tensors["merge_hw_ub"].op.axis[0],
                                                                  te_platform.DMA_COPY)
            else:
                self._sch[graph_tensors["transpose_ub"]].emit_insn(graph_tensors["transpose_ub"].op.axis[0],
                                                                   te_platform.insn_cmd.ADDVS)
                self._sch[graph_tensors["merge_hw_ub"]].emit_insn(graph_tensors["merge_hw_ub"].op.axis[0],
                                                                  te_platform.insn_cmd.ADDVS)

            self._sch[graph_tensors["merge_co_ub"]].emit_insn(graph_tensors["merge_co_ub"].op.axis[0],
                                                              te_platform.DMA_COPY)
            self._sch[graph_tensors["workspace_res"]].emit_insn(workspace_axis_inner_list[0], te_platform.DMA_COPY)
            self._sch[graph_tensors["ub_res"]].emit_insn(graph_tensors["ub_res"].op.axis[0],
                                                         te_platform.DMA_COPY)
            if reg_mov:
                if self.origin_c_in == 1 and self.dtype_input not in ("int8", "uint8"):
                    self._sch[reg_mov_ub].emit_insn(reg_mov_ub.op.axis[0], te_platform.REDUCE_SUM)
                else:
                    self._sch[reg_mov_ub].emit_insn(reg_mov_ub.op.axis[0], te_platform.DATA_MOV)
            self._sch[self.output_res].emit_insn(dma_copy_axis, te_platform.DMA_PADDING)

        out_shape_len = len(self.out_shape)
        dma_split_i = 0
        prod = 1
        for i in range(out_shape_len - 1, -1, -1):
            prod = prod * self.out_shape[i]
            if prod > self.align_block_size:
                dma_split_i = i
                break

        for i in range(min(1, dma_split_i), dma_split_i + 1):
            dma_split_factor, align_split, allow_multi_core = self._get_dma_split_factor(i)
            if align_split or i == dma_split_i:
                _schedule_32b_not_aligned(i, dma_split_factor, allow_multi_core,
                                          graph_tensors, reg_mov=(i != out_shape_len - 1))
                break

    # 'pylint: disable=too-many-arguments
    def _get_tiling_param_cut_howo_col(self):
        """
        get params for tiling
        """
        # cut howo col
        max_vm_ub = (self.used_ub_size // self.align_block_size // self.lcm_out_w + self.khkw - 1) // (self.khkw + 1)
        if max_vm_ub > Constant.LOAD3D_REPEAT_TIME_LIMIT:
            max_vm_ub = Constant.LOAD3D_REPEAT_TIME_LIMIT
        max_vm_l1 = Constant.SIZE_L1 // (self.cut_h_col * self.pre_com_fmap_w_c0)
        if max_vm_ub > max_vm_l1:
            max_vm_ub = max_vm_l1
        if max_vm_ub > 1:
            while self.origin_c_in % max_vm_ub != 0:
                max_vm_ub = max_vm_ub - 1
        # cut howo col, move_rate
        # move_rate limit according to mte2 bound
        move_rate = 1 / self.khkw
        return max_vm_ub, move_rate

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _get_tiling_param_cut_howo_row(self, stride_h):
        # cut howo row
        max_vm_ub = self.avg_split_ub_size // self.align_block_size // Constant.BLOCK_SIZE // self.khkw
        max_vm_load3d_limit = Constant.LOAD3D_REPEAT_TIME_LIMIT // self.khkw
        if max_vm_ub > max_vm_load3d_limit:
            max_vm_ub = max_vm_load3d_limit
        max_vm_l1 = Constant.SIZE_L1 // (self.cut_h_row * self.pre_com_fmap_w_c0)
        if max_vm_ub > max_vm_l1:
            max_vm_ub = max_vm_l1
        if max_vm_ub > 1:
            while self.origin_c_in % max_vm_ub != 0:
                max_vm_ub = max_vm_ub - 1

        # cut howo row, move_rate
        # move_rate useful move rate while mte2 data move
        double_loaded = self.dilated_kernel_h // 2 - stride_h
        if double_loaded < 0:
            double_loaded = 0
        slide_dis_h = self.cut_h_row - self.dilated_kernel_h + 1
        slide_times_h = slide_dis_h // stride_h + 1
        slide_dis_w = self.cut_w_row - self.dilated_kernel_w + 1
        move_rate = slide_dis_w / (slide_times_h * self.fmap_w) * (1 - double_loaded / self.cut_h_row)
        return max_vm_ub, move_rate

    # 'pylint: disable=too-many-arguments
    def _get_tiling_param_cut_howo_partial_col(self, stride_h):
        """
        The function is get tiling param cut howo partial col.
        """
        # cut howo col partially
        c_in_align = math.ceil(self.origin_c_in / self.align_block_size) * self.align_block_size
        max_vm_ub = self.avg_split_ub_size // (self.khkw * c_in_align * self.align_block_size)
        max_vm_load3d_limit = Constant.LOAD3D_REPEAT_TIME_LIMIT // self.khkw
        if max_vm_ub > max_vm_load3d_limit:
            max_vm_ub = 0

        w_size = self.fmap_w * c_in_align * self.type_size * Constant.DOUBLE_BUFFER
        max_vm_l1 = Constant.SIZE_L1 // (self.dilated_kernel_h * w_size)
        if Constant.SIZE_L1 < (math.ceil(max_vm_l1 * Constant.BLOCK_SIZE / self.out_w) + 1) * stride_h * w_size \
                or self.cut_h_row > stride_h + self.dilated_kernel_h - 1:
            max_vm_l1 = Constant.SIZE_L1 // (self.cut_h_row * w_size)

        if max_vm_ub > max_vm_l1:
            max_vm_ub = max_vm_l1
        cut_hw_up_w = (max_vm_ub * self.align_block_size + self.out_w - 1) // self.out_w * self.out_w

        # cut howo col partially, move_rate
        # move_rate useful move rate while mte2 data move
        move_rate = max_vm_ub * self.align_block_size / (cut_hw_up_w + Constant.DELTA)
        return max_vm_ub, move_rate

    def _get_tiling_param_cut_howo_min(self):
        # cut howo self.khkw c, minimum cut
        max_vm_ub = self.avg_split_ub_size // (1 * self.align_block_size * Constant.BLOCK_SIZE)
        if max_vm_ub > Constant.LOAD3D_REPEAT_TIME_LIMIT:
            max_vm_ub = Constant.LOAD3D_REPEAT_TIME_LIMIT
        max_vm_l1 = Constant.SIZE_L1 // (self.cut_h_row * self.pre_com_fmap_w_c0)
        if max_vm_ub > max_vm_l1:
            max_vm_ub = max_vm_l1

        return max_vm_ub

    # 'pylint: disable=too-many-arguments
    def _get_tiling_param(self):
        stride_h = self.setfmatrix_dict.get("conv_stride_h")

        max_vm_cut_col, self.move_rate_cut_col = self._get_tiling_param_cut_howo_col()

        max_vm_cut_row, move_rate_cut_row = self._get_tiling_param_cut_howo_row(stride_h)

        max_vm_cut_col_p, move_rate_cut_col_p = \
            self._get_tiling_param_cut_howo_partial_col(stride_h)

        max_vm_cut_min = self._get_tiling_param_cut_howo_min()
        return [max_vm_cut_col, max_vm_cut_row, max_vm_cut_col_p, max_vm_cut_min, self.move_rate_cut_col,
                move_rate_cut_row, move_rate_cut_col_p]

    def _get_tiling_factor(self, tiling_param, align=True):
        n_factor = 1
        howo_factor = self.howo
        khkw_factor = self.khkw
        c_factor = self.origin_c_in

        max_vm_cut_col, max_vm_cut_row, max_vm_cut_col_p, max_vm_cut_min, self.move_rate_cut_col, \
            move_rate_cut_row, move_rate_cut_col_p = tiling_param
        move_rate = 0
        if max_vm_cut_col > 0:
            move_rate = self.move_rate_cut_col
        if move_rate < move_rate_cut_row and max_vm_cut_row > 0:
            move_rate = move_rate_cut_row
        if move_rate < move_rate_cut_col_p and max_vm_cut_col_p > 0:
            move_rate = move_rate_cut_col_p
        split_khkw_mode = False

        if self.lcm_out_w * self.c_out <= self.avg_split_ub_size and \
                self.khkw * self.fmap_c1 <= Constant.LOAD3D_REPEAT_TIME_LIMIT \
                and max_vm_cut_col > 0 and max_vm_cut_row > 0 and \
                Constant.SIZE_L1 >= self.fmap_h * self.pre_com_fmap_c1_w_c0:
            max_v = self.avg_split_ub_size // self.lcm_out_w // self.c_out
            if self.lcm_out_w * max_v < self.howo:
                # if True cut n howo else only cut n
                howo_factor = self.lcm_out_w * max_v
        elif move_rate == self.move_rate_cut_col and max_vm_cut_col > 0:
            # cut howo col
            howo_factor = self.lcm_out_w
            khkw_factor = 1
            if align:
                max_v = max_vm_cut_col
                c_factor = self.align_block_size * max_v
            else:
                c_factor = self.align_block_size
        elif move_rate == move_rate_cut_row and max_vm_cut_row > 0:
            # cut howo row
            howo_factor = Constant.BLOCK_SIZE
            khkw_factor = self.khkw
            if align:
                max_v = max_vm_cut_row
                c_factor = self.align_block_size * max_v
            else:
                c_factor = self.align_block_size
        elif move_rate == move_rate_cut_col_p and max_vm_cut_col_p > 0:
            # cut howo col partially
            howo_factor = Constant.BLOCK_SIZE * max_vm_cut_col_p
            c_factor = self.origin_c_in
            khkw_factor = self.khkw
            max_v = self.fmap_c1
        else:
            # cut howo khkw c
            howo_factor = Constant.BLOCK_SIZE
            khkw_factor = 1
            if align:
                max_v = max_vm_cut_min
                if max_v == 0:
                    max_v = 1
                    split_khkw_mode = True
                # The instruction parameter is uint8 type
                if self.khkw * max_v >= 256:
                    max_v = max(255 // self.khkw, 1)
                c_factor = self.align_block_size * max_v
            else:
                c_factor = self.align_block_size
        return [n_factor, howo_factor, khkw_factor, c_factor, split_khkw_mode, move_rate]

    def _get_axis_list(self, res, res_axis_list, tiling_factor):
        res_n_outer, res_n_inner = self._sch[res].split(res_axis_list[0], factor=tiling_factor[0])
        res_howo_outer, res_howo_inner = self._sch[res].split(res_axis_list[1], factor=tiling_factor[1])
        res_khkw_outer, res_khkw_inner = self._sch[res].split(res_axis_list[2], factor=tiling_factor[2])
        res_c_outer, res_c_inner = self._sch[res].split(res_axis_list[3], factor=self.align_block_size)
        res_c_outer, res_c_outer_inner = self._sch[res].split(res_c_outer, factor=max(tiling_factor[3] //
                                                                                      self.align_block_size, 1))

        res_axis_outer_list = [res_n_outer, res_howo_outer, res_khkw_outer, res_c_outer]
        res_axis_inner_list = [res_n_inner, res_c_outer_inner, res_howo_inner, res_khkw_inner, res_c_inner]
        return res_axis_outer_list, res_axis_inner_list

    def _cal_multi_core_factor(self, m, n, m_list, n_list):
        """
        Return the cut factors for multicore axis.
        """

        m_list = list(set(m_list))
        n_list = list(set(n_list))
        m_list.sort(reverse=True)
        n_list.sort(reverse=True)

        min_cycle_num = m * n
        core_m, core_n = m_list[-1], n_list[-1]

        for i in m_list:
            for j in n_list:
                if i * j > self.device_core_num:
                    continue
                tmp_cycle_num = math.ceil(m / i) * math.ceil(n / j)
                if tmp_cycle_num < min_cycle_num:
                    min_cycle_num = tmp_cycle_num
                    core_m, core_n = i, j
                break
        return core_m, core_n

    def _cal_multi_core_factor_3d(self, m, n, p, m_list, n_list, p_list):
        """
        Return the cut factors for multicore axis.
        """
        m_list = list(set(m_list))
        n_list = list(set(n_list))
        p_list = list(set(p_list))

        m_list.sort(reverse=True)
        n_list.sort(reverse=True)
        p_list.sort(reverse=True)

        min_cycle_num = m * n * p
        core_m, core_n, core_p = m_list[-1], n_list[-1], p_list[-1]

        for i in m_list:
            for j in n_list:
                if i * j > self.device_core_num:
                    continue
                for k in p_list:
                    if i * j * k > self.device_core_num:
                        continue
                    tmp_cycle_num = math.ceil(m / i) * math.ceil(n / j) * math.ceil(p / k)
                    if tmp_cycle_num < min_cycle_num:
                        min_cycle_num = tmp_cycle_num
                        core_m, core_n, core_p = i, j, k
                    break
        return core_m, core_n, core_p

    def _get_dma_split_factor(self, dma_split_axis_id):
        """
        get split factor
        """
        split_eles = reduce(lambda x, y: x * y, self.out_shape[dma_split_axis_id:])
        ele_len = reduce(lambda x, y: x * y, self.out_shape[dma_split_axis_id + 1:])

        def _could_split_multi_core(val):
            if val * ele_len > Constant.DMA_SPILT_ELEMENT_THRESHOLD:
                return False
            tail_len = split_eles % (val * ele_len)
            return (tail_len > self.align_block_size) or (val * ele_len > self.align_block_size and tail_len == 0)

        if _could_split_multi_core(self.out_shape[dma_split_axis_id]):
            return self.out_shape[dma_split_axis_id], True, True

        if dma_split_axis_id == 1 and _could_split_multi_core(self.out_w):  # howo
            return self.out_w, True, True

        if dma_split_axis_id == 2 and _could_split_multi_core(self.kernel_w):  # self.khkw
            return self.kernel_w, True, True

        for val in range(self.align_block_size, self.out_shape[dma_split_axis_id], self.align_block_size):
            if _could_split_multi_core(val):
                return val, (self.out_shape[dma_split_axis_id] % val == 0), True

        return 1, False, False

    def _split_multi_core_32b_not_aligned(self, multi_core_factor, dma_split_axis_id, dma_split_factor, workspace_res):
        """
        split multi core, when 32B is not aligned
        """
        res_axis_list = list(self.output_res.op.axis).copy()
        workspace_axis_list = list(workspace_res.op.axis).copy()

        res_bind_axis_list = [0 for _ in range(dma_split_axis_id)]
        workspace_bind_axis_list = [0 for _ in range(dma_split_axis_id)]
        for i in range(dma_split_axis_id):
            workspace_bind_axis_list[i], workspace_axis_list[i] = self._sch[workspace_res].\
                split(workspace_axis_list[i], factor=multi_core_factor[i])
            res_bind_axis_list[i], res_axis_list[i] = self._sch[self.output_res].\
                split(res_axis_list[i], factor=multi_core_factor[i])
        # 32B not align data copy
        res_axis_list[dma_split_axis_id], dma_copy_axis = self._sch[self.output_res].split(
            res_axis_list[dma_split_axis_id], factor=dma_split_factor)

        self._sch[self.output_res].reorder(*(res_bind_axis_list + res_axis_list[:dma_split_axis_id] +
                                             [dma_copy_axis] + res_axis_list[dma_split_axis_id + 1:]))
        self._sch[workspace_res].reorder(*(workspace_bind_axis_list + workspace_axis_list))

        res_bind_axis = self._sch[self.output_res].fuse(*res_bind_axis_list)
        workspace_bind_axis = self._sch[workspace_res].fuse(*workspace_bind_axis_list)

        return [[res_bind_axis], res_axis_list, [workspace_bind_axis], workspace_axis_list, dma_copy_axis]

    def _get_multi_core_factor_32b_not_aligned(self, dma_split_axis_id, tiling_factor):
        """
        get multi core split factor
        """
        multi_core_factor = self.out_shape.copy()
        if dma_split_axis_id == 0:
            return multi_core_factor
        if dma_split_axis_id == 1:
            multi_core_factor[0] = self.multi_core_factor_0
            return multi_core_factor

        if Constant.SIZE_L1 >= self.fmap_h * self.pre_com_fmap_c1_w_c0:
            howo_align = Constant.BLOCK_SIZE
        elif Constant.SIZE_L1 >= self.cut_h_col * self.pre_com_fmap_c1_w_c0:
            howo_align = self.lcm_out_w
        else:
            howo_align = self.howo

        def _get_core_factor(multi_core_factor, core_n, core_howo):
            multi_core_factor[0] = max(math.ceil(Constant.FACTOR_N_CEIL / core_n), tiling_factor[0])
            multi_core_factor[1] = math.ceil(max(math.ceil(self.out_shape[1] / core_howo), tiling_factor[1]) /
                                             howo_align) * howo_align
            return multi_core_factor

        pre_core_n, pre_core_howo = [1], [1]
        for i in range(1, self.device_core_num + 1):
            multi_core_factor = _get_core_factor(self.out_shape.copy(), i, i)
            pre_core_n.append(math.ceil(Constant.FACTOR_N_CEIL / multi_core_factor[0]))
            pre_core_howo.append(math.ceil(self.out_shape[1] / multi_core_factor[1]))

        core_n, core_howo = self._cal_multi_core_factor(math.ceil(Constant.FACTOR_N_CEIL / tiling_factor[0]),
                                                        math.ceil(self.out_shape[1] / tiling_factor[1]),
                                                        pre_core_n, pre_core_howo)
        multi_core_factor = _get_core_factor(self.out_shape.copy(), core_n, core_howo)

        return multi_core_factor

    def _get_multi_core_factor(self, tiling_factor):
        if Constant.SIZE_L1 >= self.fmap_h * self.pre_com_fmap_c1_w_c0:
            howo_align = Constant.BLOCK_SIZE
        else:
            howo_align = self.lcm_out_w

        def _get_core_factor(multi_core_factor, core_n, core_howo, core_c):
            multi_core_factor[0] = math.ceil(Constant.FACTOR_N_CEIL / core_n)
            multi_core_factor[1] = math.ceil(max(math.ceil(self.out_shape[1] / core_howo), tiling_factor[1]) /
                                             howo_align) * howo_align
            multi_core_factor[3] = math.ceil(max(math.ceil(self.out_shape[3] / core_c), tiling_factor[3]) /
                                             self.align_block_size) * self.align_block_size
            return multi_core_factor

        pre_core_n, pre_core_c, pre_core_howo = [1], [1], [1]
        for i in range(1, self.device_core_num + 1):
            multi_core_factor = _get_core_factor(self.out_shape.copy(), i, i, i)
            pre_core_n.append(math.ceil(Constant.FACTOR_N_CEIL / multi_core_factor[0]))
            pre_core_howo.append(math.ceil(self.out_shape[1] / multi_core_factor[1]))
            pre_core_c.append(math.ceil(self.out_shape[3] / multi_core_factor[3]))

        core_n, core_c, core_howo = self._cal_multi_core_factor_3d(math.ceil(Constant.FACTOR_N_CEIL / tiling_factor[0]),
                                                                   math.ceil(self.out_shape[3] / tiling_factor[3]),
                                                                   math.ceil(self.out_shape[1] / tiling_factor[1]),
                                                                   pre_core_n, pre_core_c, pre_core_howo)
        multi_core_factor = _get_core_factor(self.out_shape.copy(), core_n, core_howo, core_c)
        return multi_core_factor

    def _split_multi_core_32b_align(self, tiling_factor):
        """
        split multi core, when 32B is aligned
        """
        multi_core_factor = self._get_multi_core_factor(tiling_factor)
        res_axis_list = list(self.output_res.op.axis).copy()
        res_bind_axis_list = [0 for _ in res_axis_list]
        for i, _ in enumerate(res_bind_axis_list):
            res_bind_axis_list[i], res_axis_list[i] = self._sch[self.output_res].\
                split(res_axis_list[i], factor=multi_core_factor[i])
        self._sch[self.output_res].reorder(*(res_bind_axis_list + res_axis_list))
        res_bind_axis = self._sch[self.output_res].fuse(*res_bind_axis_list)

        return [res_bind_axis], res_axis_list

    def _add_compile_info(self, workspace_shape):
        add_compile_info("coreNum", self.device_core_num)
        add_compile_info("workspaceDimen", workspace_shape)
        add_compile_info("originCIn", self.origin_c_in)
