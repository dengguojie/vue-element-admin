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
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.compute.util import get_value
from tbe.dsl.compute.util import int_ceil_div


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


def get_optional_te_var(var_name):
    return None if not get_te_var(var_name) else get_te_var(var_name).get_tvm_var()


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
        self.over_head_flag = False
        self.batch_broadcast_flag = False
        self.batch_broadcast_change_attach = False


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
    block_reduce = tbe_platform.BLOCK_REDUCE
    block_size = tbe_platform.BLOCK_IN
    default_l0_ub_tiling = (1, 1, 1, block_size, block_size)
    l1_attach_at_cl0_flag = 2
    l1_attach_at_ddr_flag = 1
    l1_full_load_flag = 0
    kal1_eq_kbl1_flag = 0
    kal1_gt_kbl1_flag = 1
    kal1_lt_kbl1_flag = 2

    def __init__(self):
        self.tiling = None
        (self.al0_tiling_batch, self.al0_tiling_ma,
         self.al0_tiling_ka, self.al0_tiling_m0, self.al0_tiling_k0) = self.default_l0_ub_tiling
        (self.bl0_tiling_batch, self.bl0_tiling_nb,
         self.bl0_tiling_kb, self.bl0_tiling_n0, self.bl0_tiling_k0) = self.default_l0_ub_tiling
        (self.cl0_tiling_batch, self.cl0_tiling_nc,
         self.cl0_tiling_mc, self.cl0_tiling_n0, self.cl0_tiling_m0) = self.default_l0_ub_tiling
        (self.aub_tiling_batch, self.aub_tiling_m,
         self.aub_tiling_k, self.aub_tiling_m0, self.aub_tiling_k0) = self.default_l0_ub_tiling
        (self.bub_tiling_batch, self.bub_tiling_n,
         self.bub_tiling_k, self.bub_tiling_n0, self.bub_tiling_k0) = self.default_l0_ub_tiling
        self.al1_tiling_batch, self.al1_tiling_m, self.al1_tiling_k = 1, 1, 1
        self.bl1_tiling_batch, self.bl1_tiling_n, self.bl1_tiling_k = 1, 1, 1
        self.factor_shape = {"aub": [], "bub": [], "cub": [], "al0": [], "bl0": [], "cl0": [], "al1": [], "bl1": []}

    @staticmethod
    def get_split_param(cache_tiling_manager):
        """
        get param for attach at, ceildiv by default, use floordiv to aid simplification in binary scene
        -----------------------
        Return:
            split_param: dict, include factor_ceil_mode, split_ceil_mode, tail_strategy and activate_scope.
        """
        factor_ceil_mode = True
        split_ceil_mode = True
        tail_strategy = "guard_with_if"
        if cache_tiling_manager.cache_tiling and not cache_tiling_manager.non_factor_k_flag:
            factor_ceil_mode = False
            split_ceil_mode = False
            tail_strategy = "round_up"
        return {"split_ceil_mode": split_ceil_mode, "factor_ceil_mode": factor_ceil_mode,
                "tail_strategy": tail_strategy, "active_scope": "outer"}

    def config_tiling(self, tiling, cache_tiling, compute_param):
        """
        config tiling variable for cache tiling
        """
        tiling['block_dim'] = [cache_tiling.get('batch_dim'),
                               cache_tiling.get("n_dim"),
                               cache_tiling.get("m_dim"),
                               cache_tiling.get('k_dim')]
        tiling.get('AL1_shape')[0] = cache_tiling.get("kal1_16") * self.block_reduce
        if tiling.get('AL1_shape')[1] == -1:
            tiling.get('AL1_shape')[1] = cache_tiling.get("m_al1")
        tiling.get('BL1_shape')[0] = cache_tiling.get("kbl1_16") * self.block_reduce
        if tiling.get('BL1_shape')[1] == -1:
            tiling.get('BL1_shape')[1] = cache_tiling.get("n_bl1")
        tiling.get('AL0_matrix')[0] = cache_tiling.get("m_l0")
        tiling.get('CL0_matrix')[1] = cache_tiling.get("m_l0")
        tiling.get('CUB_matrix')[1] = cache_tiling.get("m_l0")
        tiling.get('CUB_matrix')[0] = cache_tiling.get("cub_n1")
        tiling.get('AL0_matrix')[1] = cache_tiling.get("k_al0")
        tiling.get('BL0_matrix')[0] = cache_tiling.get("k_bl0")
        tiling.get('BL0_matrix')[1] = cache_tiling.get("n_ub_l0_time") * cache_tiling.get("cub_n1")
        tiling.get('CL0_matrix')[0] = tiling.get('BL0_matrix')[1]
        if compute_param.format_a == "ND":
            tiling.get('AUB_shape')[0] = cache_tiling.get('k_aub') * self.block_reduce
            tiling.get('AUB_shape')[1] = cache_tiling.get('m_aub')
            tiling.get('BUB_shape')[0] = cache_tiling.get('k_bub') * self.block_reduce
            tiling.get('BUB_shape')[1] = cache_tiling.get('n_bub')
        return tiling

    def set_factor_shape(self, cache_tiling, format_info, status_controller):
        """
        define the tiling factor for attach at.
        we split root scope from small to large, the basic sequence is UB->L0->L1.
        the factor is TILING_L0/TILING_UB when split TILING_L0.
        the factor is TILING_L1/TILING_L0 when split TILING_L1.
        """
        attach_at_flag = self.tiling.get("attach_at_flag")
        cub_tiling = self.tiling.get("CUB_matrix")

        ub_ka = int_ceil_div(self.aub_tiling_k, self.aub_tiling_k0)
        ub_kb = int_ceil_div(self.bub_tiling_k, self.bub_tiling_k0)

        if format_info.get("a") == "ND":
            self.factor_shape["aub"] = [self.aub_tiling_m, ub_ka, self.aub_tiling_m0, self.aub_tiling_k0]
        if format_info.get("b") == "ND":
            self.factor_shape["bub"] = [ub_kb, self.bub_tiling_n, self.bub_tiling_n0, self.bub_tiling_k0]

        self._set_factor_shape_to_l0c(cache_tiling, attach_at_flag)
        if format_info.get("out") == "ND":
            self._set_factor_shape_to_out_nd(cache_tiling, cub_tiling)
        else:
            self._set_factor_shape_to_out_nz(cache_tiling, cub_tiling)

        if status_controller.split_k_axis_by_tiling:
            self.factor_shape.get("cub").insert(0, 1)
            self.factor_shape.get("cl0").insert(0, 1)
            self.factor_shape.get("al0").insert(0, 1)
            self.factor_shape.get("bl0").insert(0, 1)
            self.factor_shape.get("al12ddr").insert(0, 1)
            self.factor_shape.get("bl12ddr").insert(0, 1)
            self.factor_shape.get("al12cl0").insert(0, 1)
            self.factor_shape.get("bl12cl0").insert(0, 1)

        if status_controller.have_batch:
            cub_tiling_batch = cub_tiling[-2]
            self.factor_shape.get("cub").insert(0, cub_tiling_batch)
            self.factor_shape.get("cl0").insert(0, self.cl0_tiling_batch)
            self.factor_shape.get("aub").insert(0, self.aub_tiling_batch)
            self.factor_shape.get("al0").insert(0, self.al0_tiling_batch)
            self.factor_shape.get("al12ddr").insert(0, self.al1_tiling_batch)
            self.factor_shape.get("bub").insert(0, self.bub_tiling_batch)
            self.factor_shape.get("bl0").insert(0, self.bl0_tiling_batch)
            self.factor_shape.get("bl12ddr").insert(0, self.bl1_tiling_batch)
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

    def _set_factor_shape_to_l0c(self, cache_tiling, attach_at_flag):
        """
        define the tiling factor for tensor attach at cl0
        parent of al0/bl0 is cl0 by default, parent of al1/bl1 is cl0 in this func
        factor corresponding to [n1, m1, m0, n0, k1, k0]
        """
        self.factor_shape["al0"] = [
            None, self.al0_tiling_ma, None, self.al0_tiling_m0, self.al0_tiling_ka, self.al0_tiling_k0
        ]
        self.factor_shape["bl0"] = [self.bl0_tiling_nb, None, None, 1, 1, 1]
        self.factor_shape["al12cl0"] = [None, 1, None, None, 1, None]
        self.factor_shape["bl12cl0"] = [1, None, None, None, 1, None]
        # only split by kl0_factor when k_al1/k_bl1 larger than kl0
        if attach_at_flag.get("min_kl1_cmp_kl0"):
            if attach_at_flag.get("abkl1_attach_flag") in (self.kal1_eq_kbl1_flag, self.kal1_gt_kbl1_flag):
                self.factor_shape.get("bl12cl0")[-2] = cache_tiling.get("kbl0_factor")
            else:
                self.factor_shape.get("al12cl0")[-2] = cache_tiling.get("kal0_factor")

        # only split by kl1_times when al1/bl1 both attach at cl0
        if (attach_at_flag.get("al1_attach_flag") == self.l1_attach_at_cl0_flag and
            attach_at_flag.get("bl1_attach_flag") == self.l1_attach_at_cl0_flag):
            if attach_at_flag.get("abkl1_attach_flag") == self.kal1_gt_kbl1_flag:
                self.factor_shape.get('al12cl0')[-2] = cache_tiling.get("kl1_times")
            elif attach_at_flag.get("abkl1_attach_flag") == self.kal1_lt_kbl1_flag:
                self.factor_shape.get('bl12cl0')[-2] = cache_tiling.get("kl1_times")

    def _set_factor_shape_to_out_nd(self, cache_tiling, cub_tiling):
        """
        define the tiling factor for tensor attach at ddr when output format is ND.
        parent of cub/cl0 is cddr by default, parent of al1/bl1 is cddr in this function.
        factor corresponding to [m, n]
        """
        cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0, _, _ = cub_tiling
        self.factor_shape["cub"] = [cub_tiling_mc_factor * cub_tiling_m0, cub_tiling_nc_factor * cub_tiling_n0]
        self.factor_shape["cl0"] = [1, cache_tiling.get("n_ub_l0_time")]
        # al1 attach at ddr, bl1 not full load
        self.factor_shape["al12ddr"] = [cache_tiling.get("m_al1"), cache_tiling.get("n_single_core")]
        # al1 attach at ddr, bl1 full load
        if self.tiling.get("attach_at_flag").get("bl1_attach_flag") == self.l1_full_load_flag:
            self.factor_shape.get("al12ddr")[1] = cache_tiling.get("n_single_core")

        # al1/bl1 both attach at ddr
        self.factor_shape["bl12ddr"] = [1, cache_tiling.get("n_bl1")]
        # bl1 attach at ddr, al1 full load
        if self.tiling.get("attach_at_flag").get("al1_attach_flag") == self.l1_full_load_flag:
            self.factor_shape.get("bl12ddr")[0] = cache_tiling.get("m_single_core") * cache_tiling.get("m_al1")
        elif self.tiling.get("attach_at_flag").get("al1_attach_flag") == self.l1_attach_at_cl0_flag:
            self.factor_shape.get("bl12ddr")[0] = cache_tiling.get("m_single_core")

    def _set_factor_shape_to_out_nz(self, cache_tiling, cub_tiling):
        """
        define the tiling factor for tensor attach at ddr when output format is NZ.
        parent of cub/cl0 is cddr by default, parent of al1/bl1 is cddr in this function.
        factor corresponding to [n1, m1, m0, n0]
        """
        cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0, _, _ = cub_tiling
        self.factor_shape["cub"] = [cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0]
        self.factor_shape["cl0"] = [cache_tiling.get("n_ub_l0_time"), 1, 1, 1]

        # al1/bl1 both attach at ddr
        self.factor_shape["al12ddr"] = [1, 1, 1, None]
        # al1 attach at ddr, bl1 full load
        if self.tiling.get("attach_at_flag").get("bl1_attach_flag") == self.l1_full_load_flag:
            self.factor_shape.get("al12ddr")[:2] = [cache_tiling.get("n_bl1"), cache_tiling.get("m_al1")]
        # al1 attach at ddr, bl1 attach at cl0
        elif self.tiling.get("attach_at_flag").get("bl1_attach_flag") == self.l1_attach_at_cl0_flag:
            self.factor_shape.get("al12ddr")[1] = cache_tiling.get("m_al1")

        # al1/bl1 both attach at ddr
        self.factor_shape["bl12ddr"] = [cache_tiling.get("n_bl1"), cache_tiling.get("m_al1"), None, 1]
        # bl1 attach at ddr, al1 not attach at ddr
        if self.tiling.get("attach_at_flag").get("al1_attach_flag") != self.l1_attach_at_ddr_flag:
            self.factor_shape.get("bl12ddr")[1] = cache_tiling.get("m_single_core") * cache_tiling.get("m_al1")


class CceSimplification:
    """
    schedule simplification which can significantly reduce emitted cce code size.
    """
    MAX_ORI_SHAPE_TEMP = 1048560
    MAX_UB_SHAPE = 4096

    def __init__(self, sch, dynamic_para):
        self.tiling = dynamic_para.get("tiling_strategy")
        self.sch = sch
        self.cache_tiling = None
        self.tensor_map = None
        self.status_controller = None
        self.is_align_mode = True if (self.tiling and self.tiling.get("schedule_pattern") == "Aligned") else False
        self.var_manager = VarManage(sch, dynamic_para.get("var_range"))
        self.block_in = tbe_platform.BLOCK_IN
        self.block_out = tbe_platform.BLOCK_OUT
        self.block_reduce = tbe_platform.BLOCK_REDUCE

    def set_kaub_simplification_l1_fullload(self, single_core_k, compute_param):
        """
        config kaub when al1 and aub are full load.
        """
        if compute_param.format_a != "ND":
            return
        aub_full_load = self.tiling.get("attach_at_flag").get("aub_multi_flag") == 1
        k_aub = self.cache_tiling.get('k_aub')
        multi_k_aub_k1 = self.cache_tiling.get("multi_k_aub_l1")
        if aub_full_load and self.tiling["attach_at_flag"].get("min_kl1_cmp_kl0") == 0:
            # When AL1 full load and Aub full load, the reduced axis K is identical.
            self.sch.set_var_value(k_aub, single_core_k)
        elif self.tiling["attach_at_flag"].get("min_kl1_cmp_kl0") == 0:
            self.sch.set_constraint((multi_k_aub_k1 * k_aub == single_core_k).asnode())

    def cce_simplify(self, compute_param, sch_agent, cache_tiling_manager, sch_container):
        """
        handles cce simplification for binary, include set_var_range, pragma, skip_bound_check, etc.
        """
        self.tensor_map = sch_container.tensor_map
        self.var_manager.set_var_range_for_dynamic_scene(compute_param, cache_tiling_manager)
        if cache_tiling_manager.cache_tiling:
            self._enable_skip_bound_check(cache_tiling_manager, sch_container)
            self._buffer_tile_for_simplify(compute_param, sch_agent)
            self._emit_insn_simplyfy_c_gm(compute_param)
            if compute_param.format_a == "ND":
                self._emit_insn_simplify_aub()
                self._emit_insn_simplify_bub()
                self._fuse_l1_axis()
                self._emit_insn_simplify_al1(compute_param)
                self._emit_insn_simplify_bl1()
        elif self.is_align_mode:
            self.sch.set_constraint((self.var_manager.m_var % tbe_platform.BLOCK_IN == 0).asnode())
            self.sch.set_constraint((self.var_manager.k_var % tbe_platform.BLOCK_REDUCE == 0).asnode())
            self.sch.set_constraint((self.var_manager.n_var % tbe_platform.BLOCK_OUT == 0).asnode())

    def _emit_insn_simplyfy_c_gm(self, compute_param):
        """
        add pragma on c_gm for cachetiling
        """
        # This template does not support overlarge imput dimension
        self.sch.set_constraint(self.var_manager.m_var * self.block_in < self.MAX_ORI_SHAPE_TEMP)
        if compute_param.format_a == "ND":
            self.sch.set_constraint(self.var_manager.k_var < self.MAX_ORI_SHAPE_TEMP)
        else:
            self.sch.set_constraint(self.var_manager.k_var * self.block_in < self.MAX_ORI_SHAPE_TEMP)
        self.sch.set_constraint(self.var_manager.n_var * self.block_out < self.MAX_ORI_SHAPE_TEMP)
        c_gm = self.tensor_map.get("c_gm")
        m_l0 = self.cache_tiling.get("m_l0")
        cub_n1 = self.cache_tiling.get("cub_n1")

        self.sch[c_gm].pragma(self.sch[c_gm].leaf_iter_vars[0], "constraint", self.var_manager.n_var - cub_n1 >= 0)
        self.sch[c_gm].pragma(self.sch[c_gm].leaf_iter_vars[0], "constraint",
            tvm.truncmod(((m_l0 * self.block_in) * (cub_n1 * self.block_out)), self.MAX_ORI_SHAPE_TEMP) > 0)
        self.sch[c_gm].pragma(self.sch[c_gm].leaf_iter_vars[0], "constraint",
            ((m_l0 * self.block_in) * (cub_n1 * self.block_out)) < self.MAX_ORI_SHAPE_TEMP)

    def _emit_insn_simplify_aub(self):
        """
        add pragma on aub for ND_in_ND_out cachetiling
        """
        # Set constraints for aub
        multi_k_aub_l1 = self.cache_tiling.get("multi_k_aub_l1")
        multi_m_ub_l1 = self.cache_tiling.get("multi_m_ub_l1")
        k_aub = self.cache_tiling.get("k_aub")
        m_aub = self.cache_tiling.get("m_aub")
        aub_var = [self.cache_tiling.get("k_aub"), self.cache_tiling.get("m_aub")]
        a_align_value = self.cache_tiling.get("a_align_value")
        trans_a = int(self.status_controller.transpose_a)
        # Constraint in m dimension: m_1 is not smaller than m1_single_core and
        # m1_single_core is not smaller than multi_m_ub_l1 * m_aub

        # Constraint in k dimension: k_ori is not smaller than k_single_core and
        # k_single_core is not smaller than multi_k_aub_l1 * k_aub * self.block_reduce
        self.sch.set_constraint(self.var_manager.m_var - multi_m_ub_l1 * m_aub >= 0)
        self.sch.set_constraint(self.var_manager.k_var - multi_k_aub_l1 * k_aub * self.block_reduce >= 0)
        # aligned condition
        constraint_aub = aub_var[trans_a] * self.block_in + tvm.floormod(a_align_value -
            tvm.floormod(aub_var[trans_a] * self.block_in, a_align_value), a_align_value)
        constraint_aub_multi = aub_var[trans_a] * self.block_in * self.block_reduce + tvm.floormod(a_align_value -
            tvm.floormod(aub_var[trans_a] * self.block_in, a_align_value), a_align_value) * self.block_reduce
        self.sch.set_constraint((constraint_aub_multi % self.block_in == 0).asnode())
        self.sch.set_constraint((constraint_aub_multi % (self.block_in * self.block_reduce) == 0).asnode())

    def _emit_insn_simplify_bub(self):
        """
        add pragma on bub for ND_in_ND_out cachetiling
        """
        # Set constraints for bub
        multi_k_bub_l1 = self.cache_tiling.get("multi_k_bub_l1")
        multi_n_ub_l1 = self.cache_tiling.get("multi_n_ub_l1")
        k_bub = self.cache_tiling.get("k_bub")
        n_bub = self.cache_tiling.get("n_bub")
        b_align_value = self.cache_tiling.get("b_align_value")
        trans_b = int(self.status_controller.transpose_b)
        bub_var = [self.cache_tiling.get("n_bub"), self.cache_tiling.get("k_bub")]
        # Constraint: n_1 is not smaller than n1_single_core and
        # n1_single_core is not smaller than n1_single_core multi_n_ub_l1 * n_bub;

        # Constraint: k_ori is not smaller than k_single_core and
        # k_single_core is not smaller than multi_k_bub_l1 * k_bub * self.block_reduce
        self.sch.set_constraint(self.var_manager.n_var - multi_n_ub_l1 * n_bub >= 0)
        self.sch.set_constraint(self.var_manager.k_var - multi_k_bub_l1 * k_bub * self.block_reduce >= 0)
        # aligned condition
        constraint_bub = bub_var[trans_b] * self.block_out + tvm.floormod(
            b_align_value - tvm.floormod(bub_var[trans_b] * self.block_out, b_align_value), b_align_value)
        constraint_bub_multi = bub_var[trans_b] * self.block_out * self.block_reduce + tvm.floormod(
            b_align_value - tvm.floormod(bub_var[trans_b] * self.block_out, b_align_value),
            b_align_value) * self.block_reduce
        self.sch.set_constraint((constraint_bub_multi % self.block_out == 0).asnode())
        self.sch.set_constraint((constraint_bub_multi % (self.block_out * self.block_reduce) == 0).asnode())

    def _emit_insn_simplify_al1(self, compute_param):
        """
        add pragma on al1 for ND_in_ND_out cachetiling
        """
        a_l1 = self.tensor_map.get("a_l1")
        m_l0 = self.cache_tiling.get("m_l0")
        cub_n1 = self.cache_tiling.get("cub_n1")
        a_align_value = self.cache_tiling.get("a_align_value")
        a_ori = [self.var_manager.k_var, self.var_manager.m_var]
        aub_var = [self.cache_tiling.get("k_aub"), self.cache_tiling.get("m_aub")]
        multi_aub_var = [self.cache_tiling.get("multi_k_aub_l1"), self.cache_tiling.get("multi_m_ub_l1")]
        host_var_a = [aub_var[1] * self.block_in, aub_var[0] * self.block_reduce]
        trans_a = int(self.status_controller.transpose_a)

        cons1 = aub_var[trans_a] * self.block_in + tvm.floormod(a_align_value -
            tvm.floormod(aub_var[trans_a] * self.block_in, a_align_value), a_align_value)
        self.sch[a_l1].pragma(self.sch[a_l1].leaf_iter_vars[0], "constraint",
            tvm.div(cons1, self.block_in) - aub_var[trans_a] >= 0)
        self.sch[a_l1].pragma(self.sch[a_l1].leaf_iter_vars[0], "constraint", a_ori[trans_a] - aub_var[trans_a] >= 0)
        self.sch[a_l1].pragma(self.sch[a_l1].leaf_iter_vars[0], "constraint",
            tvm.truncmod((host_var_a[trans_a] * host_var_a[1 - trans_a]), self.MAX_ORI_SHAPE_TEMP) > 0)
        self.sch[a_l1].pragma(self.sch[a_l1].leaf_iter_vars[0], "constraint",
            (host_var_a[trans_a] * host_var_a[1 - trans_a]) < self.MAX_ORI_SHAPE_TEMP)
        self.sch[a_l1].pragma(self.sch[a_l1].leaf_iter_vars[0], "constraint",
            multi_aub_var[0] * aub_var[0] < self.MAX_UB_SHAPE)
        self.sch[a_l1].pragma(self.sch[a_l1].leaf_iter_vars[0], "constraint",
            multi_aub_var[0] * aub_var[0] - aub_var[0] >= 0)
        self.sch[a_l1].pragma(self.sch[a_l1].leaf_iter_vars[0], "constraint",
            tvm.truncmod(aub_var[1] * aub_var[0] * self.block_in * self.block_reduce, self.MAX_ORI_SHAPE_TEMP) > 0)
        if (self.status_controller.al1_attach_status == "c_l0c" and
            not self.status_controller.attach_at_flag.get("min_kl1_cmp_kl0") and compute_param.format_a == "ND"):
            self.sch[a_l1].pragma(self.sch[a_l1].leaf_iter_vars[0], "constraint",
                                  (self.cache_tiling.get("k_l0") - self.cache_tiling.get("k_aub") >= 0))

    def _emit_insn_simplify_bl1(self):
        """
        add pragma on bl1 for ND_in_ND_out cachetiling
        """
        b_l1 = self.tensor_map.get("b_l1")
        b_align_value = self.cache_tiling.get("b_align_value")
        b_ori = [self.var_manager.n_var, self.var_manager.k_var]
        bub_var = [self.cache_tiling.get("n_bub"), self.cache_tiling.get("k_bub")]
        multi_bub_var = [self.cache_tiling.get("multi_n_ub_l1"), self.cache_tiling.get("multi_k_bub_l1")]
        host_var_b = [bub_var[1] * self.block_reduce, bub_var[0] * self.block_out]
        trans_b = int(self.status_controller.transpose_b)

        cons2 = bub_var[trans_b] * self.block_out + tvm.floormod(b_align_value -
            tvm.floormod(bub_var[trans_b] * self.block_out, b_align_value), b_align_value)
        self.sch[b_l1].pragma(self.sch[b_l1].leaf_iter_vars[0], "constraint",
            tvm.div(cons2, self.block_out) - bub_var[trans_b] >= 0)
        self.sch[b_l1].pragma(self.sch[b_l1].leaf_iter_vars[0], "constraint",
            b_ori[trans_b] < self.MAX_ORI_SHAPE_TEMP)
        self.sch[b_l1].pragma(self.sch[b_l1].leaf_iter_vars[0], "constraint",
            tvm.div(b_ori[trans_b], self.block_out) - bub_var[trans_b] >= 0)
        self.sch[b_l1].pragma(self.sch[b_l1].leaf_iter_vars[0], "constraint",
            tvm.truncmod((host_var_b[trans_b] * host_var_b[1 - trans_b]), self.MAX_ORI_SHAPE_TEMP) > 0)
        self.sch[b_l1].pragma(self.sch[b_l1].leaf_iter_vars[0], "constraint",
            (host_var_b[trans_b] * host_var_b[1 - trans_b]) < self.MAX_ORI_SHAPE_TEMP)
        self.sch[b_l1].pragma(self.sch[b_l1].leaf_iter_vars[0], "constraint",
            multi_bub_var[0] * bub_var[0] < self.MAX_UB_SHAPE)
        self.sch[b_l1].pragma(self.sch[b_l1].leaf_iter_vars[0], "constraint",
            multi_bub_var[0] * bub_var[0] - bub_var[0] >= 0)
        self.sch[b_l1].pragma(self.sch[b_l1].leaf_iter_vars[0], "constraint", tvm.truncmod(
            bub_var[1] * bub_var[0] * self.block_out * self.block_reduce, self.MAX_ORI_SHAPE_TEMP) > 0)

    def _enable_skip_bound_check(self, cache_tiling_manager, sch_container):
        """
        use skip_bound_check to ignore iflikely restraint when k is splited by factor
        """
        if cache_tiling_manager.non_factor_k_flag:
            get_context().get_current_compute().get_current_schedule().add(
                "_build_config", {"predicate_realize_bound": True})
        else:
            skip_bound_check_list = [self.tensor_map.get("a_l1"), self.tensor_map.get("b_l1"),
                self.tensor_map.get("a_l0a"), self.tensor_map.get("b_l0b"),
                self.tensor_map.get("c_l0c")]
            skip_bound_check_list += sch_container.tensors_in_aub
            skip_bound_check_list += sch_container.tensors_in_bub
            skip_bound_check_list += sch_container.tensors_in_cub
            for tensor in skip_bound_check_list:
                self.sch[tensor].skip_bound_check()

    def _buffer_tile_for_simplify(self, compute_param, sch_agent):
        """
        align m/k/n axis for tensor in ub and l1
        """
        if compute_param.format_a == "ND":
            al1_k_ext = self.cache_tiling.get("multi_k_aub_l1") * self.cache_tiling.get("k_aub")
            al1_m_ext = self.cache_tiling.get("multi_m_ub_l1") * self.cache_tiling.get("m_aub")
            al1_buffer_tile_list = [(None, al1_m_ext), (None, al1_k_ext), (None, None), (None, None)]
            if compute_param.batch_a:
                al1_buffer_tile_list.insert(0, (None, None))
            sch_agent[self.tensor_map.get("a_l1")].buffer_tile(*al1_buffer_tile_list)
        if compute_param.format_b == "ND":
            bl1_k_ext = self.cache_tiling.get("multi_k_bub_l1") * self.cache_tiling.get("k_bub")
            bl1_n_ext = self.cache_tiling.get("multi_n_ub_l1") * self.cache_tiling.get("n_bub")
            bl1_buffer_tile_list = [(None, bl1_k_ext), (None, bl1_n_ext), (None, None), (None, None)]
            if compute_param.batch_b:
                bl1_buffer_tile_list.insert(0, (None, None))
            sch_agent[self.tensor_map.get("b_l1")].buffer_tile(*bl1_buffer_tile_list)

    def _fuse_l1_axis(self):
        """
        fuse axis for tensor in l1
        """
        if self.status_controller.have_batch:
            self.sch[self.tensor_map.get("a_l1")].fuse(
                self.sch[self.tensor_map.get("a_l1")].leaf_iter_vars[1],
                self.sch[self.tensor_map.get("a_l1")].leaf_iter_vars[2])
            self.sch[self.tensor_map.get("b_l1")].fuse(
                self.sch[self.tensor_map.get("b_l1")].leaf_iter_vars[1],
                self.sch[self.tensor_map.get("b_l1")].leaf_iter_vars[2])
        else:
            self.sch[self.tensor_map.get("a_l1")].fuse(
                self.sch[self.tensor_map.get("a_l1")].leaf_iter_vars[0],
                self.sch[self.tensor_map.get("a_l1")].leaf_iter_vars[1])
            self.sch[self.tensor_map.get("b_l1")].fuse(
                self.sch[self.tensor_map.get("b_l1")].leaf_iter_vars[0],
                self.sch[self.tensor_map.get("b_l1")].leaf_iter_vars[1])


class CacheTilingManager:
    """
    manager tiling vars and cache tiling flags in bianry mode
    """

    def __init__(self, sch, dynamic_para):
        self.sch = sch
        self.tiling_strategy = dynamic_para.get("tiling_strategy")
        self.attach_at_flag = None
        self.non_factor_k_flag = None
        self.cache_tiling = None
        self.k_expr = None
        self.flag_l0c_preload = False

    def config_cache_tiling(self, cce_simplification_obj, compute_param, sch_container):
        """
        config cache tiling variables in binary mode
        """
        sch_container.vector_muls_attr = {'axis_dynamic_shift': 1}
        self.attach_at_flag = self.tiling_strategy.get("attach_at_flag")
        self.non_factor_k_flag = self.tiling_strategy.get("non_factor_k_flag")
        self._get_cache_tiling(compute_param.split_k_flag)
        cce_simplification_obj.cache_tiling = self.cache_tiling
        aub_multi_flag = self.attach_at_flag.get("aub_multi_flag")
        bub_multi_flag = self.attach_at_flag.get("bub_multi_flag")
        if aub_multi_flag == 1:
            self.sch.set_var_value(self.cache_tiling.get("multi_k_aub_l1"), 1)
            self.sch.set_var_value(self.cache_tiling.get("multi_m_ub_l1"), 1)
        if bub_multi_flag == 1:
            self.sch.set_var_value(self.cache_tiling.get("multi_n_ub_l1"), 1)
            self.sch.set_var_value(self.cache_tiling.get("multi_k_bub_l1"), 1)
        abkl1_attach_flag = self.attach_at_flag.get("abkl1_attach_flag")
        if abkl1_attach_flag == 0:
            self.sch.set_var_value(self.cache_tiling.get("kl1_times"), 1)
            self._norange_kal1_kbl1_equal(cce_simplification_obj, compute_param)
        elif abkl1_attach_flag == 1:
            self._norange_kal1(cce_simplification_obj, compute_param)
        else:
            self._norange_kbl1()
        self._set_l0c_preload_flag()

    def cache_tiling_full_load(self, container, status_controller):
        """
        handles the full load scene for binary mode
        """
        if self.cache_tiling:
            c_gm = container.tensor_map.get("c_gm")
            if self.attach_at_flag.get("bl1_attach_flag") == 0:
                self.sch.set_var_value(self.cache_tiling.get("n_single_core"), 1)
                bl1 = container.tensor_map.get("b_l1")
                iter_axis = 1 if len(bl1.shape) == 4 else 3
                # split k_axis by k_dim will create one axis to bind multi core and one axis equal to 1
                if status_controller.split_k_axis_by_tiling:
                    iter_axis += 2
                self.sch[bl1].compute_at(self.sch[c_gm], self.sch[c_gm].leaf_iter_vars[iter_axis])
            if self.attach_at_flag.get("al1_attach_flag") == 0:
                self.sch.set_var_value(self.cache_tiling.get("m_single_core"), 1)
                al1 = container.tensor_map.get("a_l1")
                iter_axis = 1 if len(al1.shape) == 4 else 3
                # split k_axis by k_dim will create one axis to bind multi core and one axis equal to 1
                if status_controller.split_k_axis_by_tiling:
                    iter_axis += 2
                self.sch[al1].compute_at(self.sch[c_gm], self.sch[c_gm].leaf_iter_vars[iter_axis])

    def _get_cache_tiling(self, split_k_flag):
        """
        get basic tiling variables in binary mode
        """
        self.cache_tiling = {
            "batch_dim": get_te_var("batch_dim").get_tvm_var(),
            "batch_single_core": get_te_var("batch_single_core").get_tvm_var(),
            "n_single_core": get_te_var("n_single_core").get_tvm_var(),
            "n_dim": get_te_var("n_dim").get_tvm_var(),
            "n_bl1": get_te_var("n_bl1").get_tvm_var(),
            "n_ub_l0_time": get_te_var("n_ub_l0_time").get_tvm_var(),
            "cub_n1": get_te_var("cub_n1").get_tvm_var(),
            "m_dim": get_te_var("m_dim").get_tvm_var(),
            "m_single_core": get_te_var("m_single_core").get_tvm_var(),
            "m_al1": get_te_var("m_al1").get_tvm_var(),
            "m_l0": get_te_var("m_l0").get_tvm_var(),
            "k_dim": get_te_var("k_dim").get_tvm_var(),
            "k_l0": get_te_var("k_l0").get_tvm_var(),
            "k_al0": get_te_var("k_l0").get_tvm_var(),
            "k_bl0": get_te_var("k_l0").get_tvm_var(),
            "kal1_factor": get_te_var("kal1_factor").get_tvm_var(),
            "kbl1_factor": get_te_var("kbl1_factor").get_tvm_var(),
            "kal0_factor": get_te_var("kal0_factor").get_tvm_var(),
            "kbl0_factor": get_te_var("kbl0_factor").get_tvm_var(),
            "kal1_16": get_te_var("kal1_16").get_tvm_var(),
            "kbl1_16": get_te_var("kbl1_16").get_tvm_var(),
            "kl1_times": get_te_var("kl1_times").get_tvm_var(),
            "m_aub": get_optional_te_var("m_aub"),
            "n_bub": get_optional_te_var("n_bub"),
            "k_aub": get_optional_te_var("k_aub"),
            "k_bub": get_optional_te_var("k_bub"),
            "multi_n_ub_l1": get_optional_te_var("multi_n_ub_l1"),
            "multi_m_ub_l1": get_optional_te_var("multi_m_ub_l1"),
            "multi_k_aub_l1": get_optional_te_var("multi_k_aub_l1"),
            "multi_k_bub_l1": get_optional_te_var("multi_k_bub_l1"),
            "a_align_value": get_optional_te_var("a_align_value"),
            "b_align_value": get_optional_te_var("b_align_value"),
            "aub_align_bound": get_optional_te_var("aub_align_bound"),
            "bub_align_bound": get_optional_te_var("bub_align_bound"),
        }
        self.cache_tiling["kal1_16"] = self.cache_tiling.get("kal0_factor") * self.cache_tiling.get("k_l0")
        self.cache_tiling["kbl1_16"] = self.cache_tiling.get("kbl0_factor") * self.cache_tiling.get("k_l0")
        if not split_k_flag:
            self.cache_tiling["k_dim"] = 1

    def _norange_kal1_kbl1_equal(self, cce_simplification_obj, compute_param):
        """
        config k related tiling variable when kal1 equals kbl1
        """
        kal0_factor = self.cache_tiling.get("kal0_factor")
        kal1_factor = self.cache_tiling.get("kal1_factor")
        k_l0 = self.cache_tiling.get("k_l0")
        k_dim = self.cache_tiling.get("k_dim")
        if not self.attach_at_flag.get("min_kl1_cmp_kl0"):
            self.cache_tiling["k_al0"] = kal0_factor * k_l0
            self.cache_tiling["k_bl0"] = kal0_factor * k_l0
            if self.attach_at_flag.get("al1_attach_flag") == 0:
                self.cache_tiling["k_al0"] = kal1_factor * kal0_factor * k_l0
                self.cache_tiling["k_bl0"] = self.cache_tiling.get("k_al0")
                cce_simplification_obj.set_kaub_simplification_l1_fullload(
                    kal1_factor * kal0_factor * k_l0, compute_param)

        self.cache_tiling["kal1_16"] = kal0_factor * k_l0
        self.cache_tiling["kbl1_16"] = kal0_factor * k_l0
        self.k_expr = kal1_factor * kal0_factor * k_l0 * k_dim

    def _norange_kal1(self, cce_simplification_obj, compute_param):
        """
        config k related tiling variable when kal1 large than kbl1
        """
        kbl0_factor = self.cache_tiling.get("kbl0_factor")
        k_l0 = self.cache_tiling.get("k_l0")
        kl1_times = self.cache_tiling.get("kl1_times")
        k_dim = self.cache_tiling.get("k_dim")
        if not self.attach_at_flag.get("min_kl1_cmp_kl0"):
            self.sch.set_var_value(kbl0_factor, 1)
            self.cache_tiling["k_al0"] = kbl0_factor * k_l0
            self.cache_tiling["k_bl0"] = kbl0_factor * k_l0
        if self.attach_at_flag.get("al1_attach_flag") == 0:
            self.k_expr = self.cache_tiling.get("kbl1_factor") * kbl0_factor * k_l0 * k_dim
            cce_simplification_obj.set_kaub_simplification_l1_fullload(
                self.cache_tiling.get("kbl1_factor") * kbl0_factor * k_l0, compute_param)
        else:
            self.cache_tiling["kal1_16"] = kl1_times * kbl0_factor * k_l0
            self.k_expr = self.cache_tiling.get("kal1_factor") * kl1_times * kbl0_factor * k_l0 * k_dim

    def _norange_kbl1(self):
        """
        config k related tiling variable when kal1 smaller than kbl1
        """
        kal0_factor = self.cache_tiling.get("kal0_factor")
        k_l0 = self.cache_tiling.get("k_l0")
        kl1_times = self.cache_tiling.get("kl1_times")
        k_dim = self.cache_tiling.get("k_dim")
        if not self.attach_at_flag.get("min_kl1_cmp_kl0"):
            self.sch.set_var_value(kal0_factor, 1)
            self.cache_tiling["k_al0"] = kal0_factor * k_l0
            self.cache_tiling["k_bl0"] = kal0_factor * k_l0
        if self.attach_at_flag.get("bl1_attach_flag") == 0:
            self.k_expr = self.cache_tiling.get("kal1_factor") * kal0_factor * k_l0 * k_dim
        else:
            self.cache_tiling["kbl1_16"] = kl1_times * kal0_factor * k_l0
            self.k_expr = self.cache_tiling.get("kbl1_factor") * kl1_times * kal0_factor * k_l0 * k_dim

    def _set_l0c_preload_flag(self):
        l0c_pb = self.tiling_strategy.get("manual_pingpong_buffer").get("CL0_pbuffer")
        al1_attach_flag = self.attach_at_flag.get("al1_attach_flag")
        bl1_attach_flag = self.attach_at_flag.get("bl1_attach_flag")
        # enable l0c_preload for template_5 because only template_5's performance is improved for now
        # 2,2,0 means that l0c_double_buffer, no_k_al1_full_load and bl1_full_load
        self.flag_l0c_preload = (l0c_pb == 2 and al1_attach_flag == 2 and bl1_attach_flag == 0)


class VarManage:
    """
    manage variable and range in shape and tiling
    """
    def __init__(self, sch, var_range):
        self.sch = sch
        self.var_range = var_range
        self.m_var = None
        self.n_var = None
        self.k_var = None
        self.commom_var_range = {
            "range_block_dim": (1, 32),
            "range_64": (1, 64),
            "range_1024": (1, 1024),
            "range_ub": (256, 262144)
        }
        self.binary_shape_name = ("m", "k", "n")

    def set_var_range_for_dynamic_scene(self, compute_param, cache_tiling_manager):
        """
        set range for vars in shape and tiling
        """
        self.m_var = get_optional_te_var(compute_param.m_var_name)
        self.k_var = get_optional_te_var(compute_param.k_var_name)
        self.n_var = get_optional_te_var(compute_param.n_var_name)
        if not cache_tiling_manager.cache_tiling:
            self.sch.set_var_range(self.m_var, *self.var_range.get(compute_param.m_var_name))
            self.sch.set_var_range(self.k_var, *self.var_range.get(compute_param.k_var_name))
            self.sch.set_var_range(self.n_var, *self.var_range.get(compute_param.n_var_name))
            if compute_param.batch_a and self.var_range.get("batch") is not None:
                self.sch.set_var_range(get_optional_te_var("batch"), *self.var_range.get("batch"))
        else:
            self._set_var_range_for_cache_tiling(compute_param, cache_tiling_manager)

    def _init_var_range_dict(self, compute_param):
        range_1024 = self.commom_var_range.get("range_1024")
        range_64 = self.commom_var_range.get("range_64")
        range_ub = self.commom_var_range.get("range_ub")
        range_block_dim = self.commom_var_range.get("range_block_dim")
        var_range_dict = {"batch_dim": range_block_dim, "n_dim": range_block_dim, "m_dim": range_block_dim,
                          "m_single_core": range_1024, "n_single_core": range_1024, "m_al1": range_1024,
                          "n_bl1": range_1024, "cub_n1": range_64, "m_l0": range_64, "k_l0": range_64,
                          "n_ub_l0_time": range_64, "kal0_factor": range_64, "kbl0_factor": range_64,
                          "kal1_factor": range_64, "kbl1_factor": range_64, "kl1_times": range_64}
        if compute_param.format_a == "ND":
            value_range_append = {"m_aub": range_64, "k_aub": range_64, "k_bub": range_64, "n_bub": range_64,
                                  "multi_n_ub_l1": range_64, "multi_m_ub_l1": range_64, "multi_k_aub_l1": range_64,
                                  "multi_k_bub_l1": range_64, "a_align_value":range_1024, "b_align_value":range_1024,
                                  "aub_align_bound": range_ub, "bub_align_bound":range_ub}
            var_range_dict.update(value_range_append)
        if compute_param.split_k_flag:
            value_range_append_dim = {"k_dim": range_block_dim}
            var_range_dict.update(value_range_append_dim)
        return var_range_dict

    def _set_var_range_for_cache_tiling(self, compute_param, cache_tiling_manager):
        """
        set var range for cache tiling
        """
        var_range_dict = self._init_var_range_dict(compute_param)
        for var, var_range in var_range_dict.items():
            self.sch.set_var_range(cache_tiling_manager.cache_tiling.get(var), *var_range)
        if not compute_param.split_k_flag:
            if compute_param.format_a == "ND":
                self.sch.set_var_value(
                    get_te_var(compute_param.k_var_name).get_tvm_var(),
                    cache_tiling_manager.k_expr * tbe_platform.BLOCK_REDUCE)
            else:
                self.sch.set_var_value(get_te_var(compute_param.k_var_name).get_tvm_var(),
                                       cache_tiling_manager.k_expr)
        self.sch.set_var_value(get_optional_te_var('k'), cache_tiling_manager.k_expr)

