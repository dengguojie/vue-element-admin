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
Schedule of conv2d maxpool fusion.
"""
from tbe import tvm
from tbe.common.register import set_fusion_buildcfg
from tbe.common.utils.errormgr import error_manager_cube as err_man
from tbe.dsl.static_schedule.conv_schedule_util import ceil_div
from tbe.dsl.static_schedule.conv_schedule_util import search_op
from tbe.dsl.static_schedule.conv_maxpoolfusion_schedule_util import maxpool_tensor_buffertile
from tbe.dsl.static_schedule.conv_maxpoolfusion_schedule_util import POOLING_STRIDE, POOLING_WINDOW
from te.platform import cce_params


class MaxpoolFusion:
    """
    Class of conv2d + maxpool fusion.
    """
    def __init__(self, res, maxpool_param):
        self.flag = self.config_maxpool_fusion_flag(res)
        self.pooling_tensor_map = maxpool_param.tensor_map
        self.window_size = self.pooling_tensor_map.get(
            "window_size") if self.flag else 0
        self.pooling_out_height, self.pooling_out_width = self.pooling_tensor_map.get(
            "pooling_out") if self.flag else (0, 0)
        self.pooling_padding = self.pooling_tensor_map.get(
            "pooling_padding")
        self.fusion_mode = "{}*{}".format(
            self.window_size, self.window_size) if self.flag else None
        self.pooling_tiling_param = {}
        self.maxpool_quant_flag = self.flag and res.op.tag == "quant"

        if self.flag and self.window_size not in (2, 3):
            err_man.raise_err_specific(
                "conv2d",
                "only support 2*2 or 3*3 window size when conv2d maxpool fusion."
            )

    @staticmethod
    def config_maxpool_fusion_flag(res):
        """
        Check whether it is conv + maxpool fusion.
        """
        if "pooling2d_max" in res.op.tag:
            return True
        if search_op(res, "pooling2d_max_max_pool_res") is not None:
            return True
        return False

    def config_window_stride(self):
        """
        Config window and stride for info_dict.
        """
        pooling_shape = [self.window_size, self.window_size]
        pooling_stride = [2, 2] if self.flag else [0, 0]
        return pooling_shape, pooling_stride

    def cal_pooling_coeff(self):
        """
        Calculate ub space coefficient of the tensors in maxpool fusion compute.
        """
        pooling_coeff = 0
        if self.flag:
            pooling_coeff += 2
        if self.maxpool_quant_flag:
            # AscendQuant op ub space calculation, 1 for fp16 reform_by_vadds or reform_by_vmuls,
            # and 0.5 for int8 cast_i8_ub.
            pooling_coeff += 1.5

        return pooling_coeff

    def modify_tiling(self, tiling, filter_matrix_dim, out_width, conv_param, block_k0):
        """
        Modify tiling for maxpool fusion.
        """
        if self.flag:
            tiling["A_overhead_opt_flag"] = False
            tiling["B_overhead_opt_flag"] = False
            tiling["n_bef_batch_flag"] = False
            tiling["CUB_channel_wise_flag"] = False

            kernel_h = conv_param.filter_h
            kernel_w = conv_param.filter_w
            dilate_h = conv_param.dilate_h
            dilate_w = conv_param.dilate_w

            if not tiling["AL1_shape"]:  # AL1 full load
                tiling["AL1_shape"] = [
                    ((kernel_h - 1) * dilate_h + 1) * ((kernel_w - 1) * dilate_w + 1) * block_k0,
                    self.pooling_out_height,
                    1,
                    1
                    ]

            batch_dim, n_dim, m_dim, group_dim = tiling["block_dim"]
            _, multi_m_al1, _, _ = tiling["AL1_shape"]
            nc_cl0, _, _, _, _, _ = tiling["CL0_matrix"]
            _, mc_cub, _, _, _, _ = tiling["CUB_matrix"]
            weight_n1 = filter_matrix_dim[1]

            mc_cl0 = int(ceil_div(self.window_size * out_width, 16))

            if self.pooling_padding[0] > 0 and m_dim > 2:
                m_dim = 2

            al1_factor = ceil_div(self.pooling_out_height,
                                  ceil_div(self.pooling_out_height, multi_m_al1))
            al1_nparts = ceil_div(self.pooling_out_height, al1_factor)

            if self.fusion_mode == "3*3":
                m_dim = min(m_dim, al1_nparts)
                if out_width % 16 != 0 and al1_nparts % m_dim != 0:
                    m_dim = min(m_dim, 2)

            if tiling["CL0_matrix"][0] * n_dim < weight_n1:
                n_dim = weight_n1 // nc_cl0

            tiling["AL0_matrix"][0] = mc_cl0
            tiling["CUB_matrix"][1] = mc_cl0
            tiling["CL0_matrix"][1] = mc_cl0
            tiling["block_dim"] = [batch_dim, n_dim, m_dim, group_dim]

            pooling_mc_cub = mc_cub

            self.pooling_tiling_param = {
                "pooling_mc_cub": pooling_mc_cub,
                "pooling_al1_factor": al1_factor,
                "pooling_al1_nparts": al1_nparts,
            }

        return tiling

    def modify_cl0_factor(self, cl0_factor):
        """
        Modify cl0_factor for maxpool fusion.
        """
        if self.flag:
            cl0_factor[1] = self.pooling_out_height
        return cl0_factor

    def modify_res_m_cl0_factor(self, res_m_cl0_factor):
        """
        Modify res_m_cl0_factor for maxpool fusion.
        """
        if self.flag:
            res_m_cl0_factor = self.pooling_out_width
        return res_m_cl0_factor

    def modify_cl0_m_factor(self, cl0_ma_factor):
        """
        Modify cl0_m_factor for maxpool fusion.
        """
        if self.flag:
            cl0_ma_factor = self.pooling_tiling_param["pooling_mc_cub"]
        return cl0_ma_factor

    def align_al1_pooling(self, sch, al1):
        """
        Buffer align al1 when pooling fusion.
        """
        if self.flag:
            sch[al1].buffer_align(
                (1, 1),
                (1, 1),
                (1, 1),
                (al1.shape[3], al1.shape[3]),
                (1, 1)
                )

    def special_process(self, sch, res, cub, conv_param, tensor_param, tiling_param,
                        emit_insn_dict, attach_axis_dict):
        """
        Special process for conv + maxpool fusion.
        """
        if self.flag:
            out_width = conv_param.w_out

            fmap_row_major = tensor_param.get("fmap_row_major")
            al1 = tensor_param.get("al1")
            al0 = tensor_param.get("al0")
            bl0 = tensor_param.get("bl0")
            cl0 = tensor_param.get("cl0")

            pingpong_buffer = tiling_param.get("manual_pingpong_buffer")
            _, _, m_dim, _ = tiling_param.get("block_dim")
            blocks = tiling_param.get("blocks")

            bindcore_axis = emit_insn_dict.get("bindcore_axis")

            cub_at_res_axis = attach_axis_dict.get("cub_at_res_axis")
            singlecore_out2al1_loopm_axis = attach_axis_dict.get("singlecore_out2al1_loopm_axis")
            al12al0_loopm_axis = attach_axis_dict.get("al12al0_loopm_axis")
            batchbindonly_pragma_axis = attach_axis_dict.get("batchbindonly_pragma_axis")
            res_m_dim_axis = attach_axis_dict.get("res_m_dim_axis")
            cl0_mo = attach_axis_dict.get("cl0_mo")

            self.set_build_cfg()

            self.process_maxpool_bl0(
                sch,
                tensors=(bl0, res),
                axes=(batchbindonly_pragma_axis, res_m_dim_axis, al12al0_loopm_axis,
                      singlecore_out2al1_loopm_axis, cl0_mo),
                params=(blocks)
                )

            self.process_maxpool_ub_tensors(
                sch,
                tensors=(res),
                axes=(cub_at_res_axis, bindcore_axis, singlecore_out2al1_loopm_axis, al12al0_loopm_axis),
                params=(out_width, m_dim, pingpong_buffer)
                )

            self.process_maxpool_res(
                sch,
                res,
                cub_at_res_axis
                )

            maxpool_tensor_buffertile(
                sch,
                (fmap_row_major, al1, al0, cl0, cub, res),
                (singlecore_out2al1_loopm_axis, al12al0_loopm_axis, cl0_mo),
                (conv_param, m_dim, self.fusion_mode, self.pooling_tiling_param, self.pooling_padding)
                )

    def set_build_cfg(self):
        """
        Set fusion buildcfg for maxpool fusion.
        """
        if self.flag:
            build_config = {"read_write_bank_conflict": 1, "sync_mode": 3}
            set_fusion_buildcfg("conv2d", build_config)

    def set_maxpool_ub_scope(self, sch, body_ops):
        """
        Set scope for ub tensors in maxpool fusion compute.
        """
        if self.flag:
            for lop in body_ops:
                if lop["op"] in (
                        "pooling2d_max_input_5d_data",
                        "pooling2d_max_row_temp_max",
                        "pooling2d_max_row_max",
                        "pooling2d_max_col_temp_max",
                        "pooling2d_max_col_max",
                        "pooling2d_max_trans_vn_node",
                        "pooling2d_max_trans_line_data",
                        "pooling2d_max_ub_reshape"
                    ):
                    sch[lop["dst_buffer"]].set_scope(cce_params.scope_ubuf)

    def process_maxpool_bl0(self, sch, tensors, axes, params):
        """
        Allocate_at for bl0.
        """
        bl0, res = tensors
        batchbindonly_pragma_axis, res_m_dim_axis, al12al0_loopm_axis, singlecore_out2al1_loopm_axis, cl0_mo = axes
        blocks = params

        if self.flag:
            if blocks != 1:
                sch[bl0].allocate_at(sch[res],
                                     batchbindonly_pragma_axis,
                                     run_once_axes=[
                                         al12al0_loopm_axis,
                                         singlecore_out2al1_loopm_axis,
                                         cl0_mo
                                     ])
            else:
                sch[bl0].allocate_at(sch[res],
                                     res_m_dim_axis,
                                     run_once_axes=[
                                         al12al0_loopm_axis,
                                         singlecore_out2al1_loopm_axis,
                                         cl0_mo
                                     ])
            sch[bl0].pragma(bl0.op.axis[0], "filter_reshape", 1)

    def process_maxpool_ub_tensors(self, sch, tensors, axes, params):
        """
        Special process for ub tensors in maxpool fusion compute.
        """
        def process_padding():
            """
            Special process for padding tensors.
            """
            pad_data = self.pooling_tensor_map["max_pooling_pad_data"]
            pad_top = self.pooling_tensor_map["max_pooling_pad_top"]
            pad_bottom = self.pooling_tensor_map["max_pooling_pad_bottom"]
            pad_left = self.pooling_tensor_map["max_pooling_pad_left"]
            pad_right = self.pooling_tensor_map["max_pooling_pad_right"]
            pad_vn = self.pooling_tensor_map["max_pooling_pad_vn"]

            for tensor in (pad_data, pad_top, pad_bottom, pad_left, pad_right, pad_vn):
                sch[tensor].storage_align(tensor.op.axis[3], 16, 0)
                sch[tensor].buffer_align(
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 16),
                    (1, 1),
                )
                sch[tensor].set_scope(cce_params.scope_ubuf)
                sch[tensor].compute_at(sch[res], cub_slice_axis)

            for tensor in (pad_data, pad_top, pad_bottom, pad_left, pad_right):
                sch[pad_vn].reused_by(tensor)

            for stage in (pad_top, pad_bottom, pad_left, pad_right):
                sch[stage].emit_insn(stage.op.axis[0], 'vector_dup',
                                     {"split_select": 1})

            sch[pad_data].emit_insn(pad_data.op.axis[0], 'dma_copy',
                                    {"split_select": 1})
            sch[pad_vn].emit_insn(pad_vn.op.axis[0], 'phony_insn')

        if self.flag:
            res = tensors
            cub_slice_axis, bindcore_axis, singlecore_out2al1_loopm_axis, al12al0_loopm_axis = axes
            out_width, m_dim, pingpong_buffer = params

            input_5d_data = self.pooling_tensor_map["input_5d_data"]
            trans_line_data = self.pooling_tensor_map["trans_line_data"]
            trans_vn_node = self.pooling_tensor_map["trans_vn_node"]
            max_pool_tensors = self.pooling_tensor_map["max_pool_tensors"]
            ub_reshape = self.pooling_tensor_map["ub_reshape"]

            pooling_al1_nparts = self.pooling_tiling_param["pooling_al1_nparts"]
            pooling_al1_factor = self.pooling_tiling_param["pooling_al1_factor"]

            if m_dim != 1:
                block_tile = bindcore_axis % m_dim
            else:
                block_tile = bindcore_axis - bindcore_axis if m_dim == 1 else bindcore_axis % m_dim

            if self.fusion_mode == "3*3":
                sch[input_5d_data].reused_by(trans_line_data)
                sch[input_5d_data].mem_unique()
                sch[trans_vn_node].reused_by(ub_reshape)

                sch[trans_line_data].compute_at(sch[res], cub_slice_axis)
                sch[trans_vn_node].compute_at(sch[res], cub_slice_axis)

                sch[trans_line_data].emit_insn(trans_line_data.op.axis[0], "dma_copy")
                sch[trans_vn_node].emit_insn(trans_vn_node.op.axis[0], "phony_insn")

                if pingpong_buffer["CUB_pbuffer"] == 2:
                    sch[trans_vn_node].double_buffer()

                offset_bound = (block_tile * ceil_div(pooling_al1_nparts, m_dim) * pooling_al1_factor \
                       + singlecore_out2al1_loopm_axis * pooling_al1_factor + al12al0_loopm_axis + 1) * POOLING_STRIDE \
                       - self.pooling_padding[0]
                input_5d_data_offset = (block_tile * ceil_div(pooling_al1_nparts, m_dim) * pooling_al1_factor \
                       + singlecore_out2al1_loopm_axis.var * pooling_al1_factor + al12al0_loopm_axis.var) * \
                        POOLING_STRIDE - self.pooling_padding[0]

                input_5d_data_offset_condition = tvm.any(
                    tvm.all(singlecore_out2al1_loopm_axis.var == 0,
                            al12al0_loopm_axis.var == 0),
                    tvm.all(
                        singlecore_out2al1_loopm_axis.var + al12al0_loopm_axis.var != 0,
                        input_5d_data.op.axis[2] > input_5d_data_offset))

                sch[input_5d_data].set_store_predicate(
                    input_5d_data_offset_condition,
                    partition=True)

                # redefine tensor scope
                sch[trans_line_data].buffer_tile(
                    (None, None),
                    (None, None),
                    (offset_bound, 1),
                    (0, out_width),
                    (None, None),
                )
                sch[input_5d_data].buffer_tile(
                    (None, None),
                    (None, None),
                    (None, POOLING_WINDOW),
                    (None, None),
                    (None, None),
                )

            # input_5d_data
            sch[input_5d_data].compute_at(sch[res], cub_slice_axis)
            co_outer, co_inner = sch[input_5d_data].split(input_5d_data.op.axis[-1], 16)
            sch[input_5d_data].reorder(
                co_outer,
                input_5d_data.op.axis[-3],
                input_5d_data.op.axis[-2],
                co_inner)
            sch[input_5d_data].emit_insn(input_5d_data.op.axis[0], "vector_auto")

            for pooling_tensor in max_pool_tensors:
                sch[pooling_tensor].compute_at(sch[res], cub_slice_axis)
                sch[pooling_tensor].emit_insn(pooling_tensor.op.axis[0], "vector_max")
                sch[pooling_tensor].buffer_align(
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 16),
                    (1, 1),
                )

            # ub_reshape
            sch[ub_reshape].compute_at(sch[res], cub_slice_axis)
            if pingpong_buffer["CUB_pbuffer"] == 2:
                sch[ub_reshape].double_buffer()
            sch[ub_reshape].emit_insn(ub_reshape.op.axis[0], 'vector_auto')

            if "max_pooling_pad_data" in self.pooling_tensor_map:
                process_padding()

            self.pooling_tiling_param.update({"block_tile": block_tile})

    def process_maxpool_res(self, sch, res, cub_at_res_axis):
        """
        Process maxpool_res when conv2d + maxpool + quant fusion enabled.
        """
        if self.maxpool_quant_flag:
            res_pool = self.pooling_tensor_map.get("max_pool_res")
            sch[res_pool].compute_at(sch[res], cub_at_res_axis)
            sch[res_pool].compute_inline()

    def maxpool_al1_preload(self, sch, pingpong_buffer, al1):
        """
        Preload AL1 in maxpool fusion.
        """
        if self.flag and pingpong_buffer["AL1_pbuffer"] == 2:
            sch[al1].preload()
