#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
Sub-function of gemm_integrated_schedule
"""
from tbe.common import platform as tbe_platform


class GemmScheduleContainer:
    """
    This is a container Class used to store all containers
    """
    def __init__(self):
        self.ori_tensors = {}
        self.tensor_map = {}
        self.placeholder_name = None
        self.buffer_reuse_dict = {}
        self.axpy_2_parent = {}
        self.compute_tensors = []
        self.elemwise_tensors = []
        self.matmul_dequant_tensor = []
        self.ele_header_ub_tensors = []
        self.placeholder_tensors = []
        self.dequant_activation_tensor = []
        self.header_ub_tensors = []
        self.tensors_in_aub = []
        self.tensors_in_bub = []
        self.tensors_in_cub = []
        self.tensors_in_l0c = []
        self.matmul_tensors = []
        self.fusion_list = []
        self.fusion_ele = []
        self.tensor_fusion_list = []
        self.compute_inline_list = []
        self.elewise_compute_inline_list = []
        self.fusion_tensor_cub = []
        self.fuse_num_group = []
        self.vector_muls_attr = {}
        self.custom_block_dim = []
        self.axis_core = None
        self.double_out_tensor = []


class GemmScheduleStatusController:
    """
    This is a controller used to control flags like "attach_status"
    """
    def __init__(self):
        self.gm_ub = None
        self.have_batch_a, self.have_batch_b, self.have_batch = False, False, False
        self.have_bias, self.have_c = False, False
        self.need_init_bias = False
        self.a_l1_inline_flag, self.b_l1_inline_flag = False, False
        self.dequant_flag, self.quant_fusion, self.quantify_fusion = False, False, False
        self.requant_fusion, self.dequant_fusion = False, False
        self.reduce_fusion = False
        self.only_use_gevm_gemv_flow = False
        self.round_mode = "vector_conv"
        self.sqrt_flag = False
        self.l1_fusion_and_l1_size_0 = False
        self.input_l1_flag = False
        self.aub_attach_status = "full_load"
        self.bub_attach_status = "full_load"
        self.al1_attach_status = "full_load"
        self.bl1_attach_status = "full_load"
        self.c_l0c_attach_status = "full_load"
        self.al0_attach_status = "full_load"
        self.c_ub_attach_status = "full_load"
        self.ops_data_flow_mode = "fp162fp16"
        self.mad_pattern = tbe_platform.GEMM_MODE
        self.transpose_a, self.transpose_b = False, False
        self.compute_inline_c_ub_fract = False
        self.cube_vector_split = False
        self.mmad_mode = "gemm"
        self.align_a, self.align_b = True, True
        self.a_use_aligned_pattern, self.b_use_aligned_pattern = False, False
        self.matmul_multi_output_flag = False
        self.fusion_multi_output_flag = False
        self.storage_m_bound_change = False
        self.storage_ka_bound_change = False
        self.storage_kb_bound_change = False
        self.storage_n_bound_change = False
        self.cgm_ka_storage_change = False
        self.cgm_kb_storage_change = False
        self.compress_flag = False
        self.int8_not_double_m = False
        self.attach_at_flag = None
        self.split_k_axis_by_tiling = False
        self.split_k = False
        self.over_head_flag = False
        self.batch_broadcast_flag = False


class BufferChecker:
    """ Check whether buffer size exceed bound of memory
    """
    FP16_DTYPE = 2
    FP32_DTYPE = 4
    INPUT_SIZE = {"fp162fp16": 2, "fp162fp32": 2, "int82int32": 1, "int82fp32": 1, "int42int32": 0.5}
    OUTPUT_SIZE = {"fp162fp16": 2, "fp162fp32": 4, "int82int32": 4, "int82fp32": 4, "int42int32": 4}
    DTYPE_WIDTH_MAP = {"uint64": 4, "float16": 1, "float32": 2, "int32": 2, "int16": 1, "uint16": 1,
                       "int8": 0.5, "uint8": 0.5, "int4": 0.25, "bool": 0.5}
    DOUBLE_MULTI = 2
    PRE_UB_MULTIPLIER = 10.0
    THRESHOLD_DATA_NUM = 64
    UB_SIZE = tbe_platform.get_soc_spec("UB_SIZE")
    AUB_M_INDEX = 1
    BUB_N_INDEX = 1
    CUB_M_INDEX = 1

    def __init__(self):
        self.tiling = None
        self.container = None
        self.status_controller = None
        self.block_in = tbe_platform.BLOCK_IN
        self.block_out = tbe_platform.BLOCK_OUT
        self.block_reduce = tbe_platform.BLOCK_REDUCE
        self.format_a = None
        self.format_b = None
        self.format_out = None
        self.a_fused_num = 0
        self.b_fused_num = 0
        self.c_fused_num = 0

    def check_aub_preload(self, tiling, params):
        """ check whether total_ub_size exceed ub_buffer_size when aub preload
        """
        self.tiling = tiling
        self.container = params.get("container")
        self.status_controller = params.get("status_controller")
        if params.get("cache_tiling"):
            return True

        total_ub_size = 0
        total_ub_size += self._aub_size_preload()
        total_ub_size += self._cub_size_preload()
        return total_ub_size <= self.UB_SIZE

    def check_bub_preload(self, tiling, params):
        """ check whether total_ub_size exceed ub_buffer_size when bub preload
        """
        self.tiling = tiling
        self.container = params.get("container")
        self.status_controller = params.get("status_controller")
        if params.get("cache_tiling"):
            return True

        total_ub_size = 0
        total_ub_size += self._bub_size_preload()
        total_ub_size += self._cub_size_preload()
        return total_ub_size <= self.UB_SIZE

    def check_bias_preload(self, tiling, params):
        """ check whether total_ub_size exceed ub_buffer_size when bias preload
        """
        self.tiling = tiling
        self.container = params.get("container")
        self.status_controller = params.get("status_controller")
        if params.get("cache_tiling"):
            return True
        if self.container.fuse_num_group:
            _, _, self.c_fused_num = self.container.fuse_num_group
        self.c_fused_num += 1

        total_ub_size = 0
        total_ub_size += self._aub_size_preload()
        total_ub_size += self._bub_size_preload()
        total_ub_size += self._cub_size_preload(bias_preload=True)
        return total_ub_size <= self.UB_SIZE

    def check_exceed_ub(self, tiling, params):
        """ If storage_align is used, more UB space is used.
            Therefore, check the UB space usage after storage_align is used.
        """
        self.tiling = tiling
        self.container = params.get("container")
        self.status_controller = params.get("status_controller")
        self.format_out = params.get("format_out")
        self.format_a = params.get("format_a")
        self.format_b = params.get("format_b")

        if self.container.fuse_num_group:
            self.a_fused_num, self.b_fused_num, self.c_fused_num = self.container.fuse_num_group
        self.a_fused_num = self.a_fused_num / self.PRE_UB_MULTIPLIER + 1
        self.b_fused_num = self.b_fused_num / self.PRE_UB_MULTIPLIER + 1
        self.c_fused_num += 1

        need_aub_storage_align = (self.container.tensor_map.get("a_ub") is not None) and (self.format_a == "ND")
        need_bub_storage_align = (self.container.tensor_map.get("b_ub") is not None) and (self.format_b == "ND")
        need_cub_storage_align = (((self.container.tensor_map.get("c_add_bias_ub") is not None) or
                                  (self.container.tensor_map.get("before_c_gm") is not None)) and
                                  (not params.get("cache_tiling") and self.format_out == "ND"))

        # compute before storage_align used UB size
        base_buffer_size = 0
        base_buffer_size, a_add_size = self._get_a_ub_storage_align_buffer_size(base_buffer_size,
            need_aub_storage_align, self.status_controller.transpose_a)
        base_buffer_size, b_add_size = self._get_b_ub_storage_align_buffer_size(base_buffer_size,
            need_bub_storage_align, self.status_controller.transpose_b)
        base_buffer_size, c_add_size = self._get_c_ub_storage_align_buffer_size(base_buffer_size,
            need_cub_storage_align)

        base_buffer_size, c_ub_storage_align = self._check_cub_gap(base_buffer_size,
            c_add_size, need_cub_storage_align)
        base_buffer_size, a_ub_storage_align = self._check_aub_gap(base_buffer_size,
            a_add_size, need_aub_storage_align, self.status_controller.transpose_a)
        base_buffer_size, b_ub_storage_align = self._check_bub_gap(base_buffer_size,
            b_add_size, need_bub_storage_align, self.status_controller.transpose_b)

        return a_ub_storage_align, b_ub_storage_align, c_ub_storage_align

    def _aub_size_preload(self):
        """ calculate aub_size when aub preload
        """
        tiling = self.tiling
        aub_size = 0
        if self.container.tensor_map.get("a_ub") is not None:
            aub_k, aub_m = tiling.get("AUB_shape")[0 : self.AUB_M_INDEX + 1]
            a_db = tiling.get("manual_pingpong_buffer").get("AUB_pbuffer")
            aub_size += (aub_m * self.block_in * aub_k *
                         self.INPUT_SIZE.get(self.status_controller.ops_data_flow_mode) * a_db)
        return aub_size

    def _bub_size_preload(self):
        """ calculate bub_size when bub preload
        """
        tiling = self.tiling
        bub_size = 0
        if self.container.tensor_map.get("b_ub") is not None:
            bub_k, bub_n = tiling.get("BUB_shape")[0: self.BUB_N_INDEX + 1]
            b_db = tiling.get("manual_pingpong_buffer").get("BUB_pbuffer")
            bub_size += (bub_k * bub_n * self.block_out *
                         self.INPUT_SIZE.get(self.status_controller.ops_data_flow_mode) * b_db)
        return bub_size

    def _cub_size_preload(self, bias_preload=False):
        """ calculate cub_size when aub/bub/bias preload
        """
        tiling = self.tiling
        cub_n, cub_m = tiling.get("CUB_matrix")[0: self.CUB_M_INDEX + 1]
        cub_size = (cub_n * self.block_out * cub_m * self.block_in * self.c_fused_num *
                    self.OUTPUT_SIZE.get(self.status_controller.ops_data_flow_mode))
        if bias_preload:
            cub_size += (cub_n * self.block_out *
                         self.OUTPUT_SIZE.get(self.status_controller.ops_data_flow_mode))
        return cub_size

    def _get_a_ub_storage_align_buffer_size(self, base_buffer_size, need_aub_storage_align, a_trans):
        """
        calculate extra aub buffer size

        Parameters:
        ------------
        base_buffer_size: int, base ub buffer size
        need_aub_storage_align: bool, whether need aub storage align
        a_trans: bool, transpose of matrix A

        Returns:
        ------------
        base_buffer_size: int, base ub buffer size
        a_add_size: int, extra buffer size
        """
        tiling = self.tiling
        gap_value = self.block_reduce
        a_add_size = 0
        if self.container.tensor_map.get("a_ub") is not None:
            aub_k, aub_m = tiling.get("AUB_shape")[0: self.AUB_M_INDEX + 1]
            aub_m *= self.block_in
            a_db = tiling.get("manual_pingpong_buffer").get("AUB_pbuffer")
            base_buffer_size += (aub_m * aub_k * self.a_fused_num *
                                 self.INPUT_SIZE.get(self.status_controller.ops_data_flow_mode) * a_db)

        if need_aub_storage_align:
            # if use storage_align, need UB size
            a_add_size = (gap_value * (aub_k if a_trans else aub_m) *
                          self.INPUT_SIZE.get(self.status_controller.ops_data_flow_mode) * a_db)
        return base_buffer_size, a_add_size

    def _get_b_ub_storage_align_buffer_size(self, base_buffer_size, need_bub_storage_align, b_trans):
        """
        calculate extra bub buffer size

        Parameters:
        ------------
        base_buffer_size: int, base ub buffer size
        need_bub_storage_align: bool, whether need bub storage align
        b_trans: bool, transpose of matrix B

        Returns:
        ------------
        base_buffer_size: int, base ub buffer size
        b_add_size: int, extra buffer size
        """
        tiling = self.tiling
        gap_value = self.block_reduce
        b_add_size = 0
        if self.container.tensor_map.get("b_ub") is not None:
            bub_k, bub_n = tiling.get("BUB_shape")[0: self.BUB_N_INDEX + 1]
            bub_n *= self.block_out
            b_db = tiling.get("manual_pingpong_buffer").get("BUB_pbuffer")
            base_buffer_size += (bub_k * bub_n * self.b_fused_num *
                                 self.INPUT_SIZE.get(self.status_controller.ops_data_flow_mode) * b_db)

        if need_bub_storage_align:
            # if use storage_align, need UB size
            b_add_size = (gap_value * (bub_n if b_trans else bub_k) *
                          self.INPUT_SIZE.get(self.status_controller.ops_data_flow_mode) * b_db)
        return base_buffer_size, b_add_size

    def _get_c_ub_storage_align_buffer_size(self, base_buffer_size, need_cub_storage_align):
        """
        calculate extra cub buffer size

        Parameters:
        ------------
        base_buffer_size: int, base ub buffer size
        need_bub_storage_align: bool, whether need cub storage align

        Returns:
        ------------
        base_buffer_size: int, base ub buffer size
        c_add_size: int, extra buffer size
        """
        tiling = self.tiling
        c_add_size = 0
        cub_n, cub_m = tiling.get("CUB_matrix")[0: self.CUB_M_INDEX + 1]
        c_db = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")
        base_buffer_size += (cub_n * cub_m * self.block_in * self.block_out * self.c_fused_num *
                             self.OUTPUT_SIZE.get(self.status_controller.ops_data_flow_mode) * c_db)
        if need_cub_storage_align:
            # if use storage_align, need UB size
            if self.container.tensor_map.get("before_c_gm") is not None:
                before_c_gm = self.container.tensor_map.get("before_c_gm")
                data_size = self.DTYPE_WIDTH_MAP.get(before_c_gm.dtype) * self.DOUBLE_MULTI
                c_add_size = cub_n * self.block_out * data_size * c_db
            else:
                data_size = self.FP32_DTYPE
                c_ub_cast_to_fp16 = self.container.tensor_map.get("cast_to_fp16")
                tensor_alpha = self.container.tensor_map.get("alpha")
                if (c_ub_cast_to_fp16 is not None) and (tensor_alpha is None):
                    data_size = self.FP16_DTYPE
                c_add_size = self.block_out * cub_n * cub_m * data_size * c_db
        return base_buffer_size, c_add_size

    def _check_aub_gap(self, base_buffer_size, a_add_size, need_aub_storage_align, a_trans):
        """
        check whether to do aub storage align

        Parameters:
        ------------
        base_buffer_size: int, base ub buffer size
        a_add_size: int, extra buffer size
        need_aub_storage_align: bool, whether need aub storage align
        a_trans: bool, transpose of matrix A

        Returns:
        ------------
        base_buffer_size: int, base ub buffer size
        a_ub_storage_align: bool, whether aub to do storage align
        """
        a_ub_storage_align = False
        if need_aub_storage_align:
            aub_k, aub_m = self.tiling.get("AUB_shape")[0: self.AUB_M_INDEX + 1]
            aub_m *= self.block_in
            judge_value = aub_m if a_trans else aub_k
            a_ub_storage_align = ((judge_value % self.THRESHOLD_DATA_NUM == 0)
                and ((base_buffer_size + a_add_size) <= self.UB_SIZE))
            if a_ub_storage_align:
                base_buffer_size += a_add_size
        return base_buffer_size, a_ub_storage_align

    def _check_bub_gap(self, base_buffer_size, b_add_size, need_bub_storage_align, b_trans):
        """
        check whether to do bub storage align

        Parameters:
        ------------
        base_buffer_size: int, base ub buffer size
        b_add_size: int, extra buffer size
        need_bub_storage_align: bool, whether need bub storage align
        b_trans: bool, transpose of matrix B

        Returns:
        ------------
        base_buffer_size: int, base ub buffer size
        b_ub_storage_align: bool, whether bub to do storage align
        """
        b_ub_storage_align = False
        if need_bub_storage_align:
            bub_k, bub_n = self.tiling.get("BUB_shape")[0: self.BUB_N_INDEX + 1]
            bub_n *= self.block_out
            judge_value = bub_k if b_trans else bub_n
            b_ub_storage_align = ((judge_value % self.THRESHOLD_DATA_NUM == 0)
                and ((base_buffer_size + b_add_size) <= self.UB_SIZE))
            if b_ub_storage_align:
                base_buffer_size += b_add_size
        return base_buffer_size, b_ub_storage_align

    def _check_cub_gap(self, base_buffer_size, c_add_size, need_cub_storage_align):
        """
        check whether to do aub storage align

        Parameters:
        ------------
        base_buffer_size: int, base ub buffer size
        c_add_size: int, extra buffer size
        need_cub_storage_align: bool, whether need cub storage align

        Returns:
        ------------
        base_buffer_size: int, base ub buffer size
        c_ub_storage_align: bool, whether cub to do storage align
        """
        c_ub_storage_align = False
        if need_cub_storage_align:
            c_ub_storage_align = (base_buffer_size + c_add_size <= self.UB_SIZE)
            if c_ub_storage_align:
                base_buffer_size += c_add_size
        return base_buffer_size, c_ub_storage_align
