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
from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.buildcfg import build_config
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.compute.util import int_ceil_div
from tbe.dsl.compute.util import get_value


def get_all_tags(res):
    """
    get all tags
    :param res: tensor
    :return: list
    """
    op_tags = set()

    def get_tag(tensor):
        """
        find all tags
        :param tensor: tensor
        :return: all tags
        """
        tensor_list = tensor.op.input_tensors
        op_tags.add(tensor.op.tag)
        for one_tensor in tensor_list:
            op_tags.add(one_tensor.op.tag)
            get_tag(one_tensor)

    get_tag(res)
    return op_tags


def copy_attrs(src_tensor, dst_tensor):
    for attr_key, value in src_tensor.op.attrs.items():
        dst_tensor.op.attrs[attr_key] = value


def print_ir_matmul(debug_ir_flag, process, sch):
    """
    print ir for input sch
    :param process: tag
    :param sch: schedule
    :return: IR process
    """
    if debug_ir_flag:
        with build_config():
            start = process + " IR start"
            end = process + " IR end\n"
            sch = sch.normalize()
            print(start)
            bounds = tvm.schedule.InferBound(sch)
            stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
            print(stmt)
            print(end)


def debug(debug_param_flag, info, tag=""):
    """
    print log if debug
    :param info:
    :return:
    """
    if debug_param_flag:
        print("----")
        print(tag, info)


class GemmScheduleContainer:
    """
    This is a container Class used to store all containers
    """
    def __init__(self):
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
        self.need_init_bias = False
        self.a_l1_inline_flag, self.b_l1_inline_flag = False, False
        self.quant_fusion, self.quantify_fusion = False, False
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
        self.non_factor_k_flag = False
        self.over_head_flag = False
        self.batch_broadcast_flag = False
        self.batch_broadcast_change_attach = False
        self.flag_l0c_preload = False


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


class UbBufferReuser:
    # This Class is used to set CUB to Reuse AUB/BUB in cache Tiling Mode.
    def __init__(self, tiling, tensor_map, buffer_reuse_dict):
        self.tiling = tiling
        self.tensor_map = tensor_map
        self.buffer_reuse_dict = buffer_reuse_dict

    def set_post_ub_reuse_pre_ub(self, split_k):
        # The following attach flag can only be enabled in cacheTiling mode.
        al1_attach_flag = self.tiling.get("attach_at_flag").get("al1_attach_flag")
        bl1_attach_flag = self.tiling.get("attach_at_flag").get("bl1_attach_flag")
        aub_attach_flag = self.tiling.get("attach_at_flag").get("aub_multi_flag")
        bub_attach_flag = self.tiling.get("attach_at_flag").get("bub_multi_flag")
        al1_full_load = al1_attach_flag == 0
        bl1_full_load = bl1_attach_flag == 0
        aub_full_load = aub_attach_flag == 1
        bub_full_load = bub_attach_flag == 1
        # To avoid Preload Precision Problem or Double Buffer Failure Problem,
        # reused is disable when L1 and UB are full load at the same time.
        aub_vacant_flag = al1_full_load and not aub_full_load
        bub_vacant_flag = bl1_full_load and not bub_full_load

        a_ub = self.tensor_map.get("a_ub")
        a_ub_fract = self.tensor_map.get("a_ub_fract")
        b_ub = self.tensor_map.get("b_ub")
        b_ub_fract = self.tensor_map.get("b_ub_fract")
        if split_k:
            c_ub = self.tensor_map.get("c_ub_fract")
        else:
            c_ub = self.tensor_map.get("cast_to_fp16")
        cub_nz_to_nd = self.tensor_map.get("nz_to_nd")
        # Only reusing nz_to_nd if it is not split K scene.
        nz_to_nd_reused_list = list()
        if aub_vacant_flag and not bub_vacant_flag:
            self._add_to_reused_dict(c_ub, a_ub)
            nz_to_nd_reused_list = [a_ub_fract]
        elif bub_vacant_flag and not aub_vacant_flag:
            self._add_to_reused_dict(c_ub, b_ub)
            nz_to_nd_reused_list = [b_ub_fract]
        elif aub_vacant_flag and bub_vacant_flag:
            self._add_to_reused_dict(c_ub, [a_ub, b_ub])
            nz_to_nd_reused_list = [a_ub_fract, b_ub_fract]
        if not split_k:
            self._add_to_reused_dict(cub_nz_to_nd, nz_to_nd_reused_list)

    def _add_to_reused_dict(self, src_tensor, reuse_tensor):
        # Allowing reuse_tensor(or reuse_tensors) to reuse the spaces of src_tensor.
        if (src_tensor is not None) and (reuse_tensor is not None):
            self.buffer_reuse_dict[src_tensor] = reuse_tensor


class GemmTilingWork:
    """ There are tiling parameters in matmul
    """
    def __init__(self):
        self.tiling = None
        self.block_reduce = tbe_platform.BLOCK_REDUCE
        (self.al0_tiling_batch, self.al0_tiling_ma,
         self.al0_tiling_ka, self.al0_tiling_m0, self.al0_tiling_k0) = 1, 1, 1, 16, 16
        (self.bl0_tiling_batch, self.bl0_tiling_nb,
         self.bl0_tiling_kb, self.bl0_tiling_n0, self.bl0_tiling_k0) = 1, 1, 1, 16, 16
        self.al1_tiling_batch, self.al1_tiling_m, self.al1_tiling_k = 1, 1, 1
        self.bl1_tiling_batch, self.bl1_tiling_n, self.bl1_tiling_k = 1, 1, 1
        (self.aub_tiling_batch, self.aub_tiling_m,
         self.aub_tiling_k, self.aub_tiling_m0, self.aub_tiling_k0) = 1, 1, 1, 16, 16
        (self.bub_tiling_batch, self.bub_tiling_n,
         self.bub_tiling_k, self.bub_tiling_n0, self.bub_tiling_k0) = 1, 1, 1, 16, 16
        (self.cl0_tiling_batch, self.cl0_tiling_nc,
         self.cl0_tiling_mc, self.cl0_tiling_n0, self.cl0_tiling_m0) = 1, 1, 1, 16, 16
        self.factor_shape = {"aub": [], "bub": [], "cub": [], "al0": [], "bl0": [], "cl0": [], "al1": [], "bl1": []}

    @staticmethod
    def get_split_param(cache_tiling, non_factor_k_flag):
        """
        get param for attach at, ceildiv by default, use floordiv to aid simplification in binary scene
        -----------------------
        Return:
            split_param: dict, include factor_ceil_mode, split_ceil_mode, tail_strategy and activate_scope.
        """
        factor_ceil_mode = True
        split_ceil_mode = True
        tail_strategy = "guard_with_if"
        if cache_tiling and not non_factor_k_flag:
            factor_ceil_mode = False
            split_ceil_mode = False
            tail_strategy = "round_up"
        return {"split_ceil_mode": split_ceil_mode, "factor_ceil_mode": factor_ceil_mode,
                "tail_strategy": tail_strategy, "active_scope": "outer"}

    def _set_factor_shape_nz_out(self, cache_tiling, cub_tiling):
        """
        define the tiling factor for attach at when output format is NZ.
        """
        cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0, _, _ = cub_tiling
        self.factor_shape["cub"] = [cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0]
        self.factor_shape["cl0"] = [cache_tiling.get("n_ub_l0_time"), 1, 1, 1]
        if self.tiling.get("attach_at_flag").get("bl1_attach_flag") == 0:
            self.factor_shape["al12ddr"] = [cache_tiling.get("n_bl1"), cache_tiling.get("m_al1"), 1, None]
        elif self.tiling.get("attach_at_flag").get("bl1_attach_flag") == 1:
            self.factor_shape["al12ddr"] = [1, 1, 1, None]
        else:
            self.factor_shape["al12ddr"] = [1, cache_tiling.get("m_al1"), 1, None]

        if self.tiling.get("attach_at_flag").get("al1_attach_flag") == 1:
            self.factor_shape["bl12ddr"] = [cache_tiling.get("n_bl1"), cache_tiling.get("m_al1"), None, 1]
        else:
            self.factor_shape["bl12ddr"] = [cache_tiling.get("n_bl1"),
                                            cache_tiling.get("m_single_core") * cache_tiling.get("m_al1"), None, 1]

    def set_factor_shape(self, cache_tiling, format_info, status_controller):
        """
        define the tiling factor for attach at.
        we split root scope from small to large, the basic sequence is UB->L0->L1.
        the factor is TILING_L0/TILING_UB when split TILING_L0.
        the factor is TILING_L1/TILING_L0 when split TILING_L1.
        """
        attach_at_flag = self.tiling.get("attach_at_flag")
        cub_tiling = self.tiling.get("CUB_matrix")
        cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0, cub_tiling_batch, _ = cub_tiling
        ub_ka = int_ceil_div(self.aub_tiling_k, self.aub_tiling_k0)
        ub_kb = int_ceil_div(self.bub_tiling_k, self.bub_tiling_k0)

        # parent of cub/cl0 is cddr by default, parent of al0/bl0 is cl0 by default.
        # parent of al1/bl1 is cl0 when attach_flag is equals to 2, otherwise parent of al1/bl1 is cddr.
        # cddr has 2 iter_vars(m, n) when output format is ND.
        # cddr has 4 iter_vars(n1, m1, m0, n0) when output format is NZ.
        # cl0 has 6 iter_vars(n1, m1, m0, n0, k1, k0).
        self.factor_shape["al0"] = [
            None, self.al0_tiling_ma, None, self.al0_tiling_m0, self.al0_tiling_ka, self.al0_tiling_k0
        ]
        self.factor_shape["bl0"] = [self.bl0_tiling_nb, None, None, 1, 1, 1]
        if format_info.get("a") == "ND":
            self.factor_shape["aub"] = [self.aub_tiling_m, ub_ka, self.aub_tiling_m0, self.aub_tiling_k0]
        if format_info.get("b") == "ND":
            self.factor_shape["bub"] = [ub_kb, self.bub_tiling_n, self.bub_tiling_n0, self.bub_tiling_k0]
        # only split by kl0_factor when k_al1/k_bl1 larger than kl0
        if attach_at_flag.get("min_kl1_cmp_kl0"):
            if attach_at_flag.get("abkl1_attach_flag") in (0, 1):
                self.factor_shape["al12cl0"] = [None, 1, None, 1, 1, 1]
                self.factor_shape["bl12cl0"] = [1, None, None, 1, cache_tiling.get("kbl0_factor"), 1]
            else:
                self.factor_shape["al12cl0"] = [None, 1, None, 1, cache_tiling.get("kal0_factor"), 1]
                self.factor_shape["bl12cl0"] = [1, None, None, 1, 1, 1]
        if format_info.get("out") == "ND":
            self.factor_shape["cub"] = [cub_tiling_mc_factor * cub_tiling_m0, cub_tiling_nc_factor * cub_tiling_n0]
            self.factor_shape["cl0"] = [1, cache_tiling.get("n_ub_l0_time")]
            self.factor_shape["al12ddr"] = [cache_tiling.get("m_al1"), cache_tiling.get("n_single_core")]
            self.factor_shape["bl12ddr"] = [1, cache_tiling.get("n_bl1")]
        else:
            self._set_factor_shape_nz_out(cache_tiling, cub_tiling)

        if status_controller.split_k_axis_by_tiling:
            self.factor_shape.get("cub").insert(0, 1)
            self.factor_shape.get("cl0").insert(0, 1)
            self.factor_shape.get("al0").insert(0, 1)
            self.factor_shape.get("bl0").insert(0, 1)
            self.factor_shape.get("al12ddr").insert(0, 1)
            self.factor_shape.get("bl12ddr").insert(0, 1)
            if attach_at_flag.get("min_kl1_cmp_kl0"):
                self.factor_shape.get("al12cl0").insert(0, 1)
                self.factor_shape.get("bl12cl0").insert(0, 1)

        if status_controller.have_batch:
            self.factor_shape.get("cub").insert(0, cub_tiling_batch)
            self.factor_shape.get("cl0").insert(0, self.cl0_tiling_batch)
            self.factor_shape.get("aub").insert(0, self.aub_tiling_batch)
            self.factor_shape.get("al0").insert(0, self.al0_tiling_batch)
            self.factor_shape.get("al12ddr").insert(0, self.al1_tiling_batch)
            self.factor_shape.get("bub").insert(0, self.bub_tiling_batch)
            self.factor_shape.get("bl0").insert(0, self.bl0_tiling_batch)
            self.factor_shape.get("bl12ddr").insert(0, self.bl1_tiling_batch)
            if attach_at_flag.get("min_kl1_cmp_kl0"):
                self.factor_shape.get("al12cl0").insert(0, 1)
                self.factor_shape.get("bl12cl0").insert(0, 1)

    def get_a_max_k_bound(self, a_l0a):
        """
        This function is used to get the maximum k bound, which will be used in the
        following calculation to solve bank conflict and to set storage bound.
        """
        a_matrix_dim = [get_value(i) for i in a_l0a.shape]
        k_bound_tiling = (int_ceil_div(a_matrix_dim[-3],
                                       self.tiling.get("AL0_matrix")[1]) * self.tiling.get("AL0_matrix")[1] *
                          self.block_reduce)
        return int_ceil_div(k_bound_tiling, self.tiling.get("block_dim")[-1])

    def get_b_max_k_bound(self, b_l0b, is_dynamic, dynamic_k):
        """
        This function is used to get the maximum k bound, which will be used in the
        following calculation to solve bank conflict and to set storage bound.
        """
        b_matrix_dim = [get_value(i) for i in b_l0b.shape]
        if self.tiling.get("BL0_matrix"):
            k_bound_tiling = (int_ceil_div(b_matrix_dim[-4],
                                           self.tiling.get("BL0_matrix")[0]) *
                              self.tiling.get("BL0_matrix")[0] * self.block_reduce)
            return int_ceil_div(k_bound_tiling, self.tiling.get("block_dim")[-1])
        elif is_dynamic:
            return int_ceil_div(dynamic_k, self.tiling.get("block_dim")[-1]) * self.block_reduce
        else:
            return int_ceil_div(b_matrix_dim[-4], self.tiling.get("block_dim")[-1]) * self.block_reduce