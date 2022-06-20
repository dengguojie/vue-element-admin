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
gemm schedule
"""

import os
import math
import functools
from collections.abc import Iterable

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.utils.errormgr import error_manager_cube
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.tiling import get_tiling_type
from tbe.common.tiling import set_tiling_type
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import in_dynamic
from tbe.dsl.boost_schedule_kit import Compare
from tbe.dsl.boost_schedule_kit import ScheduleAgent
from tbe.dsl.boost_schedule_kit import SplitParam
from tbe.dsl.compute import cube_util
from tbe.dsl.compute.util import get_value
from tbe.dsl.compute.util import int_ceil_div
from tbe.dsl.compute.gemm_integrated_compute import GEMMComputeParam
from tbe.dsl.instrinsic import cce_emitinsn_params
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import copy_attrs
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import debug
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import get_all_tags
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import print_ir_matmul
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import CacheTilingManager
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import GemmScheduleContainer
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import GemmScheduleStatusController
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import GemmTilingWork
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import BufferChecker
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import UbBufferReuser
from tbe.dsl.static_schedule.gemm_integrated_schedule_util import CceSimplification
from tbe.dsl.static_schedule.util import L1CommonParam
from tbe.dsl.static_schedule.util import parse_tbe_compile_para
from tbe.dsl.static_schedule.util import CalculateMultiUB


def gemm_schedule(res, sch_list, dynamic_para=None):
    """
    schedule enter
    param:
    res: tensor
    sch_list: list of schedule
    """
    gemm_sch = GemmSchedule(res, sch_list, dynamic_para)
    gemm_sch.gemm_schedule()

    return True


def _check_double_out(outs):
    """ check double out condition for matmul ub fusion
    """
    if not isinstance(outs, list):
        return False
    # 2 means two output tensors
    if len(outs) != 2:
        return False

    for tensor in outs:
        if not isinstance(tensor, tvm.tensor.Tensor):
            return False
        # not support matmul multi-output
        if tensor.op.tag != "gemm" and tensor.op.name == "tensor_c_gm":
            return False

    all_tags = get_all_tags(outs[-1])
    if "matmul" not in all_tags and "gemm" not in all_tags:
        return False

    shape0 = [x.value for x in outs[0].shape]
    shape1 = [x.value for x in outs[1].shape]
    # only support elementwise
    return shape0 == shape1


def reget_matmul_multioutput(outs):
    """ add a virtual node to connect double outs for tvm coding rule
    """
    if _check_double_out(outs):

        out1, out2 = outs
        op_tag = "gemm" if out1.op.tag == "gemm" else "matmul_elewise"
        input_tensor = {"gemm": out1.op.input_tensors[0], "matmul_elewise": out1}
        out1_copy = tvm.compute(
            out1.shape,
            lambda *indices: input_tensor.get(op_tag)(*indices).astype(out1.dtype),
            name=op_tag + "out1",
            tag=op_tag
        )
        # 4 means that virtual node only support four dims for now
        if (len(out1.shape) == 4 and len(out2.shape) == 4) or op_tag == "gemm":
            virtual_res = tvm.compute(
                out1.shape,
                lambda *indices: out1_copy(*indices) + out2(*indices),
                name=op_tag + "_virtual_res",
                tag=op_tag + "_virtual_res"
            )
            outs = [virtual_res, out1_copy, out2]
    return outs


class GemmSchedule:
    """
    schedule enter
    param:
    res: tensor
    sch_list: list of schedule
    dynamic_para: dynamic para from gemm_tilingcase
    """
    DEBUG_PARAM = False
    DEBUG_IR = False
    DYN_ALIGNED_MODE = "Aligned_mode"
    GENERAL_MODE = "General_mode"
    PRE_UB_MULTIPLIER = 10.0
    DTYPE_WIDTH_MAP = {"uint64": 4, "float16": 1, "float32": 2, "int32": 2,
                       "int16": 1, "uint16": 1, "int8": 0.5, "uint8": 0.5,
                       "int4": 0.25, "bool": 0.5}
    INPUT_SIZE = {"fp162fp16": 2, "fp162fp32": 2, "int82int32": 1, "int82fp32": 1, "int42int32": 0.5}
    L1_L0_SIZE = {"fp162fp16": 2, "fp162fp32": 2, "int82int32": 1, "int82fp32": 2, "int42int32": 0.5}
    OUTPUT_SIZE = {"fp162fp16": 2, "fp162fp32": 4, "int82int32": 4, "int82fp32": 4, "int42int32": 4}
    MAD_TYPE = {
        "fp162fp16": "float32",
        "fp162fp32": "float32",
        "int82int32": "int32",
        "int42int32": "int32",
        "int82fp32": "float32"
    }
    emit_fusion_insn_map = {
        "dequant_NZ": "phony_insn",
        "cast_f16_ub": "vector_conv",
        "input_ub": "phony_insn",
        "reform_by_vmuls": "vector_muls",
        "scale_sqrt_ub": "vector_muls",
        "offset_ub": "vector_adds",
        "cast_i8_ub": "vector_conv",
        "reform_by_vadds": "vector_adds"
    }

    reform_tensor_tag_list = ("reform_by_vadds",
                              "reform_by_vmuls",
                              "data_transfer")

    emit_insn_map = {"elewise_single_cast": "vector_conv",
                     "elewise_single_round_d": "vector_conv_round",
                     "elewise_single_VS_max": "vector_maxs",
                     "elewise_single_VS_min": "vector_mins",
                     "elewise_single_ceil": "elewise_single_ceil",
                     "elewise_single_log": "vector_ln",
                     "elewise_single_exp": "vector_exp",
                     "elewise_single_relu": "vector_relu",
                     "elewise_single_abs": "vector_abs",
                     "elewise_single_not": "vector_not",
                     "elewise_single_sqrt": "vector_sqrt",
                     "elewise_single_rsqrt": "vector_rsqrt",
                     "elewise_binary_mul": "vector_mul",
                     "elewise_single_rec": "vector_rec",
                     "elewise_single_VS_mul": "vector_muls",
                     "elewise_binary_div": "vector_div",
                     "elewise_binary_sub": "vector_sub",
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
                     "elewise_binary_cmpsel_gt": "vector_select_gt",
                     "elewise_binary_cmpsel_ge": "vector_select_ge",
                     "elewise_binary_cmpsel_lt": "vector_select_lt",
                     "elewise_binary_cmpsel_le": "vector_select_le",
                     "elewise_binary_cmpsel_eq": "vector_select_eq",
                     "elewise_binary_cmpsel_ne": "vector_select_ne",
                     "elewise_binary_or": "vector_or",
                     "elewise_binary_and": "vector_and",
                     "elewise_multiple_mla": "vector_multiple",
                     "elewise_multiple_madd": "vector_multiple",
                     "elewise_multiple_maddrelu": "vector_multiple",
                     "elewise_multiple_sel": "vector_select_bool",
                     "elewise_binary_scalar_axpy": "vector_axpy",
                     "elewise_binary_cmpsel": "vector_cmpsel",
                     "broadcast": "vector_dup",
                     "emit_insn_elewise_multiple_sel": "elewise_multiple_sel",
                     "emit_insn_elewise_binary_cmp": "elewise_binary_cmp"
                     }

    requant_fusion_insn_map = {"tensor_c_gm": "phony_insn",
                               "tensor_c_ub": "phony_insn",
                               "s32_to_s8": "dma_copy",
                               "data_transfer": "dma_copy"}

    DEQ_SCALE_CHILD_LIST = [
        "dequant",
        "dequant_scale",
        "dequant_sqrt",
    ]
    M1_VAR_NAME = "m"
    N1_VAR_NAME = "n"
    K1_VAR_NAME = "k"
    THRESHOLD_DATA_NUM = 64
    ND_M_INDEX = -2
    ND_N_INDEX = -1
    FRACTAL_NZ_M_INDEX = -3
    FRACTAL_NZ_N_INDEX = -4
    FRACTAL_NZ_M0_INDEX = -2
    FRACTAL_NZ_N0_INDEX = -1
    FRACTAL_Z_M_INDEX = -4
    FRACTAL_Z_N_INDEX = -3
    FRACTAL_Z_KA_INDEX = -3
    FRACTAL_Z_LEN = 4
    FRACTAL_LEN_WITH_BATCH = 5
    BLOCK_BATCH_DIM_INDEX = 0
    BLOCK_N_DIM_INDEX = 1
    BLOCK_M_DIM_INDEX = 2
    ALLOCATE_OFF = 0
    ALLOCATE_HALF = 1
    ALLOCATE_FULL = 2
    KBL1_LARGER_FLAG = 2
    BLOCKS_PER_REPEAT = 8

    # index of tiling
    IDX_BATCH_DIM = 0
    IDX_N_DIM = 1
    IDX_M_DIM = 2
    IDX_K_DIM = 3

    IDX_KBL1 = 0
    IDX_MULTI_N1 = 1

    IDX_KAL1 = 0
    IDX_MULTI_M1 = 1

    UINT16_MAX = 65535

    is_dynamic = False

    def __init__(self, res, sch_list, dynamic_para):
        self.tensor_list = dynamic_para["tensor_list"]
        self.res_ori = res
        self.res = res[-1] if isinstance(res, list) else res
        self.root_tensor = res[-1] if isinstance(res, list) else res
        self.sch = sch_list[0]
        self.sch_list = sch_list
        self.sch_agent = None
        self.kernel_name = "gemm"
        self.dynamic_para = dynamic_para
        self.cache_tiling = None
        # used to control aligned/general mode in dynamic shape
        self.schedule_mode = self.GENERAL_MODE
        self.block_in = tbe_platform.BLOCK_IN
        self.block_out = tbe_platform.BLOCK_OUT
        self.block_reduce = tbe_platform.BLOCK_REDUCE
        self.input_l1_size = 0
        self.in_addr_type, self.out_addr_type = 0, 0
        self.tensor_a_l1_workspace = 0
        self.dynamic_m, self.dynamic_k, self.dynamic_n, self.dynamic_batch = 1, 1, 1, 1
        self.c_col_k0, self.c_col_k1 = 1, 1
        self.optmt_a, self.optmt_b, self.optmt_c = "float16", "float16", "float16"
        self.format_info = {"a": "ND", "b": "ND", "out": "ND"}
        self.ops_format = "ND"
        self.seed_shape = None
        self.get_a_matrix_mode, self.get_b_matrix_mode = "none", "none"
        self.l1_fusion_type = 0
        self.compute_param = GEMMComputeParam()
        self.status_ori_dict = {
            0: Compare.EQUAL,
            1: Compare.LESS_EQ,
            2: Compare.LESS_EQ,
            3: Compare.LESS_EQ
        }
        self.status_dict = {
            0: Compare.EQUAL,
            1: Compare.EQUAL,
            2: Compare.LESS_EQ,
            3: Compare.GREATE_EQ
        }
        self.bind_core_when_full_load_bl1 = False

        # Call initialization function
        self.container = GemmScheduleContainer()
        self.status_controller = GemmScheduleStatusController()
        self.tiling_work = GemmTilingWork()
        self.buffer_checker = BufferChecker()
        self.cce_simplification_obj = CceSimplification(self.sch, dynamic_para)
        self.cache_tiling_manager = CacheTilingManager(self.sch, dynamic_para)
        self.tensor_b_reshape = 0

    @staticmethod
    def _is_full_load(item):
        return item == []

    @staticmethod
    def _bit_width(dtype):
        return {'float16': 16, 'int8': 8, 'float32': 32, 'int4': 4}.get(dtype)

    @staticmethod
    def _prod(lst):
        return functools.reduce(lambda x, y: x * y, lst)

    @staticmethod
    def _get_addr_type(tensor):
        addr_type = 0
        if "addr_type" in tensor.op.attrs and tensor.op.attrs["addr_type"].value == 1:
            addr_type = 1
        return addr_type

    @staticmethod
    def _get_output_format(tensor):
        """
        get output format

        Parameters
        ----------
        tensor : res tensor

        Return :
        output format
        """
        format_out = "FRACTAL_NZ"
        if tensor is None:
            return format_out
        if (tensor.op.attrs is None) or ("format" not in tensor.op.attrs):
            if len(tensor.shape) in (2, 3):
                format_out = "ND"
            return format_out

        format_out = tensor.op.attrs["format"]
        return format_out

    @staticmethod
    def _add_tensor_to_list(tensor, tensors_list_list):
        if tensor is not None:
            for tensors_list in tensors_list_list:
                if tensor not in tensors_list:
                    tensors_list.append(tensor)

    @staticmethod
    def _match_and_get_tensor(tensors, tensor_name):
        """
        match and get tensor
        """
        for i in tensors:
            if tensor_name == i.op.name:
                return i
        return None

    @staticmethod
    def _round_emit_insn(round_mode):
        """
        Obtains the conv instruction by the round mode attr

        Parameters
        ----------
        round_mode: the attr of round mode

        Returns
        -------
        instruction
        """
        if cube_util.is_mini_version():
            emit_insn_str = "vector_conv"
        else:
            if round_mode == "Round":
                emit_insn_str = "vector_conv"
            else:
                raise RuntimeError("Round mode should be Round only, %s is not supported" % round_mode)
        return emit_insn_str

    @staticmethod
    def _map_apend(input_map, key, value):
        """
        map apend
        """
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

    @staticmethod
    def _get_input_l1_paras(tensor):
        input_l1_flag = -1
        input_l1_size = -1
        if 'L1_addr_flag' in tensor.op.attrs:
            input_l1_flag = tensor.op.attrs['L1_addr_flag'].value

        if input_l1_flag == 1:
            if 'L1_valid_size' in tensor.op.attrs:
                input_l1_size = tensor.op.attrs['L1_valid_size'].value
            else:
                input_l1_flag = -1

        return input_l1_flag, input_l1_size

    @staticmethod
    def _get_l1_fusion_type(tensor):
        l1_fusion_type = -1
        if "L1_fusion_type" in tensor.op.attrs:
            l1_fusion_type = tensor.op.attrs["L1_fusion_type"].value
        return l1_fusion_type

    @staticmethod
    def _get_zero_tiling(tiling_dict):
        if all(value == 0 for value in tiling_dict['AL0_matrix']):
            return True
        return False

    @staticmethod
    def _handle_bool_env(env_name, default_value):
        str_env = str(os.environ.get(env_name, None)).lower()
        if str_env not in ('true', 'false'):
            return default_value
        return str_env == 'true'

    @staticmethod
    def _get_compress_block_info(tile_k, tile_n):
        """
        get weigths compress info, like, block size, index size
        """
        block_size_max = 32 * 1024
        block_unit = 512
        data_size = tile_k * tile_n * block_unit
        if data_size > block_size_max:
            error_manager_cube.raise_err_message_cube("block_size cannot be greater than block_size_max")
        return int(data_size)

    def _get_batch_info(self):
        tensor_l0a = self.container.tensor_map.get("a_l0a")
        tensor_l0b = self.container.tensor_map.get("b_l0b")
        tensor_l0c = self.container.tensor_map.get("c_l0c")
        self.status_controller.have_batch_a = len(tensor_l0a.shape) in (3, 5)
        self.status_controller.have_batch_b = len(tensor_l0b.shape) in (3, 5)
        self.status_controller.have_batch = len(tensor_l0c.shape) in (3, 5)
        if "ori_batch_shape" in tensor_l0a.op.attrs and "ori_batch_shape" in tensor_l0b.op.attrs:
            batch_a = [get_value(i) for i in tensor_l0a.op.attrs["ori_batch_shape"]]
            batch_b = [get_value(i) for i in tensor_l0b.op.attrs["ori_batch_shape"]]
            batch_broadcast_flag = batch_a != batch_b
            self.status_controller.batch_broadcast_flag = batch_broadcast_flag
            batch_broadcast_flag &= (len(batch_a) > 0 and self._prod(batch_a) != 1)
            batch_broadcast_flag &= (len(batch_b) > 0 and self._prod(batch_b) != 1)
            self.status_controller.batch_broadcast_change_attach = batch_broadcast_flag

    def _temporarily_enable_fullload_to_bind_multi_core_when_exceed_space(self):
        if self.is_dynamic:
            return

        t = self.tiling_work.tiling
        soc_version = tbe_platform_info.get_soc_spec(tbe_platform_info.FULL_SOC_VERSION)
        size_l1 = tbe_platform_info.get_soc_spec(tbe_platform_info.L1_SIZE)

        status = soc_version == "Ascend310P3" and self.status_controller.quantify_fusion
        status = status and self.get_a_matrix_mode == "Nz2Zz" and self.get_b_matrix_mode == "none"
        status = status and not self._is_full_load(t.get("AL1_shape")) and self._is_full_load(t.get("BL1_shape"))
        status = status and (self._calc_b_l1(False) + self._calc_a_l1(False) > size_l1)
        status = status and (self._calc_b_l1(True) + self._calc_a_l1(False) <= size_l1)

        if status:
            self.bind_core_when_full_load_bl1 = True

    def _calc_b_l1(self, consider_multi_core_in_full_load=True):
        tensor_bl1 = self.container.tensor_map.get("b_l1")
        tensor_bl0 = self.container.tensor_map.get("b_l0b")
        tiling_bl1 = self.tiling_work.tiling.get("BL1_shape")
        tiling_pb_bl1 = self.tiling_work.tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
        n_dim = self.tiling_work.tiling.get("block_dim")[self.IDX_N_DIM]
        *_, _, n1_bl0, n0_bl0, _ = (x.value for x in tensor_bl0.shape) if self._is_full_load(
            self.tiling_work.tiling.get("BL0_matrix")) else self.tiling_work.tiling.get("BL0_matrix")

        if self._is_full_load(tiling_bl1):
            size_shape = self._prod(x.value for x in tensor_bl1.shape)

            if consider_multi_core_in_full_load:
                # consistent with the implementation logic of schedule_agent, using nparts to split the axis
                # get_b_matrix_mode is none
                # cache_read tensor_b
                # k1, n1, k0, n0 with not trans_b
                *_, k1, n1, _, _ = (x.value for x in tensor_bl1.shape)
                if self.status_controller.transpose_b:
                    k1, n1 = n1, k1
                n1_factor = math.ceil(n1 / n_dim)
                size_shape = size_shape // n1 * n1_factor
        else:
            size_shape = tiling_bl1[self.IDX_KBL1] * tiling_bl1[self.IDX_MULTI_N1] * n1_bl0 * n0_bl0
        return size_shape * tiling_pb_bl1 * self._bit_width(tensor_bl1.dtype)

    def _calc_a_l1(self, consider_multi_core_in_full_load=True):
        tensor_al1 = self.container.tensor_map.get("a_l1")
        tensor_al0 = self.container.tensor_map.get("a_l0b")
        tiling_al1 = self.tiling_work.tiling.get("AL1_shape")
        tiling_pb_al1 = self.tiling_work.tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
        m_dim = self.tiling_work.tiling.get("block_dim")[self.IDX_M_DIM]
        *_, m1_al0, _, m0_al0, _ = (x.value for x in tensor_al0.shape) if self._is_full_load(
            self.tiling_work.tiling.get("AL0_matrix")) else self.tiling_work.tiling.get("AL0_matrix")

        if self._is_full_load(tiling_al1):
            size_shape = self._prod(x.value for x in tensor_al1.shape)

            if consider_multi_core_in_full_load:
                # consistent with the implementation logic of schedule_agent, using nparts to split the axis
                # get_a_matrix_mode is Nz2Zz
                # cache_read tensor_a
                # k1, m1, m0, k0 with not trans_a
                *_, k1, m1, _, _ = (x.value for x in tensor_al1.shape)
                if self.status_controller.transpose_a:
                    k1, m1 = m1, k1
                m1_factor = math.ceil(m1 / m_dim)
                size_shape = size_shape // m1 * m1_factor
        else:
            size_shape = tiling_al1[self.IDX_KAL1] * tiling_al1[self.IDX_MULTI_M1] * m1_al0 * m0_al0
        return size_shape * tiling_pb_al1 * self._bit_width(tensor_al1.dtype)

    def gemm_schedule(self):
        """
        the main func of gemm_schedule
        """
        print_ir_matmul(self.DEBUG_IR, "original ir", self.sch)
        self._set_para_for_genenal()
        if in_dynamic():
            self._set_para_for_dynamic()
            self._get_seed_shape()
        self._set_data_layout(self.res)
        self._get_batch_info()
        self._set_buffer_reuse_dict()
        self._tiling_process()
        self._update_flag_after_tiling()
        self._set_nd_out_compute_inline()
        self._atomic_add_k_axis()
        if self.cache_tiling and self.format_info.get("a") == "ND" and self.format_info.get("b") == "ND":
            ub_reuse_obj = UbBufferReuser(self.tiling_work.tiling, self.container.tensor_map,
                                          self.container.buffer_reuse_dict)
            ub_reuse_obj.set_post_ub_reuse_pre_ub(self.status_controller.split_k_axis_by_tiling)
        self.sch_agent = ScheduleAgent(self.sch)
        self._tiling_l0_process()
        self._tiling_l1_process()
        self._tiling_ub_process()
        if self.cache_tiling:
            self.tiling_work.set_factor_shape(self.cache_tiling, self.format_info, self.status_controller)
            self._bind_multi_core()
        c_ub_tiling_shape = self._cub_process()
        self._cl0_process(c_ub_tiling_shape)
        self._al0_process()
        self._bl0_process()
        self._do_l1_ub_process()
        if not self.cache_tiling:
            self._bind_multi_core()
        self._do_emit_insn()
        self._do_buffer_reuse()
        self.sch_agent.apply()
        self._do_buffer_align()
        self._solve_bank_conflict()
        self._solve_split_k_dirty_data()
        a_run_once, b_run_once = self._allocate_axis(self.status_controller.over_head_flag)
        self.cache_tiling_manager.cache_tiling_full_load(self.container, self.status_controller)
        self._double_buffer(a_run_once, b_run_once)
        self._handle_tbe_compile_para()
        self._reorder_axis(self.status_controller.over_head_flag, a_run_once, b_run_once)
        self._do_compute_inline()
        self._mem_process()
        self._set_continuous_axis()
        if self.is_dynamic:
            self.cce_simplification_obj.cce_simplify(
                self.compute_param, self.sch_agent, self.cache_tiling_manager, self.container)
        print_ir_matmul(self.DEBUG_IR, "final ir", self.sch)
        self.tiling_work.tiling.clear()
        self.container.tensor_map.clear()
        return True

    def _set_para_for_genenal(self):
        tensor_map = self.tensor_list[0]
        tensor_l0c = tensor_map.get("c_l0c")
        # set status_controller para
        self.status_controller.need_init_bias = tensor_map.get("bias_init") is not None
        self.status_controller.ops_data_flow_mode = tensor_l0c.op.attrs["ops_data_flow_mode"].value
        self.status_controller.only_use_gevm_gemv_flow = tensor_l0c.op.attrs["only_use_gevm_gemv_flow"].value
        self.status_controller.mmad_mode = tensor_l0c.op.attrs["mmad_mode"].value
        if self.status_controller.mmad_mode in ("gevm", "gemv") and \
            not self.status_controller.only_use_gevm_gemv_flow:
            self.status_controller.mad_pattern = tbe_platform.GEVM_MODE
        self.status_controller.int8_not_double_m = tensor_l0c.op.attrs["int8_not_double_m"].value
        self.status_controller.cube_vector_split = tbe_platform_info.get_soc_spec("CUBE_VECTOR_SPLIT")
        self.status_controller.transpose_a = tensor_l0c.op.attrs["transpose_a"].value
        self.status_controller.transpose_b = tensor_l0c.op.attrs["transpose_b"].value
        self.status_controller.split_k = bool(tensor_l0c.op.attrs["split_k"].value)
        self.status_controller.compress_flag =  tensor_l0c.op.attrs["compress_flag"].value
        self.status_controller.align_a = tensor_l0c.op.attrs["align_a"].value
        self.status_controller.align_b = tensor_l0c.op.attrs["align_b"].value
        # set init para
        self.ops_format = tensor_l0c.op.attrs["ops_format"].value
        self.format_info["a"] = tensor_l0c.op.attrs["format_a"].value
        self.format_info["b"] = tensor_l0c.op.attrs["format_b"].value
        if self.status_controller.ops_data_flow_mode == "int82int32":
            self.block_reduce = tbe_platform.BLOCK_REDUCE_INT8
            self.tiling_work.block_reduce = self.block_reduce
        self.tensor_b_reshape = 1 if tensor_map.get("tensor_b_reshape") is not None else 0
        self.kernel_name =  tensor_l0c.op.attrs["kernel_name"].value
        # set container para
        self.container.placeholder_name = tensor_l0c.op.attrs["placeholder_name"]
        if tensor_map.get("multi_output_list") is not None:
            self.container.double_out_tensor = tensor_map.get("multi_output_list")
            self.status_controller.fusion_multi_output_flag = True
            self.root_tensor = self.res_ori[0]
        if "custom_block_dim_m" in tensor_l0c.op.attrs:
            self.container.custom_block_dim = [
                tensor_l0c.op.attrs["custom_block_dim_m"].value,
                tensor_l0c.op.attrs["custom_block_dim_k"].value,
                tensor_l0c.op.attrs["custom_block_dim_n"].value
            ]
        self._fusion_para()

    def _set_para_for_dynamic(self):
        tensor_map = self.tensor_list[0]
        self.is_dynamic = True
        if self.dynamic_para.get("tiling_strategy").get("schedule_pattern") == "Aligned":
            self.schedule_mode = self.DYN_ALIGNED_MODE
            self.status_controller.a_use_aligned_pattern = True
            self.status_controller.b_use_aligned_pattern = True
        if tensor_map.get("virtual_aub") is not None:
            self.status_controller.a_use_aligned_pattern = \
                tensor_map["virtual_aub"].op.attrs["use_aligned_pattern"].value
        if tensor_map.get("virtual_bub") is not None:
            self.status_controller.a_use_aligned_pattern = \
                tensor_map["virtual_bub"].op.attrs["use_aligned_pattern"].value
        self.cce_simplification_obj.status_controller = self.status_controller

    def _get_real_k_multi_core_axis(self):
        """
        The multi-core loop range must be correctly identified during the operation of removing dirty data when multiple
        cores are bound to the k axis.
        When the bound multi-core is a non-factor, the cyclic range of the inner axis is upward aligned. If the length
        of the X axis is A and N cores are bound, the (N-1)th core has loaded the length A, and if the X axis has no
        segmentation other than the multi-core, In this case, the actual multi-core cycle range is N-1.
        """
        block_dims = self.tiling_work.tiling.get("block_dim")
        m_dim = block_dims[2]
        n_dim = block_dims[1]
        real_multi_core_axis = self.container.axis_core
        if self.is_dynamic:
            block_dims_without_k_dim = n_dim * m_dim
        else:
            tensor_a_l0a = self.container.tensor_map.get("a_l0a")
            m_shape = get_value(tensor_a_l0a.shape[self.FRACTAL_Z_M_INDEX])
            core_align_m = int_ceil_div(m_shape, m_dim)
            if (core_align_m * (m_dim - 1) >= m_shape) and (self.tiling_work.al1_tiling_m == core_align_m):
                m_dim -= 1
            tensor_b_l0b = self.container.tensor_map.get("b_l0b")
            n_shape = get_value(tensor_b_l0b.shape[self.FRACTAL_Z_N_INDEX])
            core_align_n = int_ceil_div(n_shape, n_dim)
            if (core_align_n * (n_dim - 1) >= n_shape) and (self.tiling_work.bl1_tiling_n == core_align_n):
                n_dim -= 1
            block_dims_without_k_dim = n_dim * m_dim
        if block_dims_without_k_dim != 1:
            real_multi_core_axis //= block_dims_without_k_dim
        return real_multi_core_axis

    def _solve_split_k_dirty_data(self):
        # cache_tiling does not have not factor split, do not need this function now
        if not self.status_controller.split_k_axis_by_tiling or self.cache_tiling:
            return

        block_dims = self.tiling_work.tiling.get("block_dim")
        real_multi_core_axis = self._get_real_k_multi_core_axis()
        a_l0 = self.container.tensor_map.get("a_l0a")
        a_l0_shape_k = get_value(a_l0.shape[-3])
        temp_block_k_dim = block_dims[3] - 1
        no_tail_core_k_len = (a_l0_shape_k + temp_block_k_dim) // block_dims[3]
        tail_core_k_len = a_l0_shape_k % no_tail_core_k_len
        lastest_core = a_l0_shape_k // no_tail_core_k_len
        lastest_core_limit = lastest_core
        c_gm = self.container.tensor_map.get("c_gm")
        if self.is_dynamic:
            lastest_core_limit = tvm.select(no_tail_core_k_len * lastest_core == a_l0_shape_k, lastest_core_limit - 1,
                                            lastest_core_limit)
        else:
            if no_tail_core_k_len * lastest_core == a_l0_shape_k:
                lastest_core_limit -= 1
        self.sch[c_gm].set_store_predicate(real_multi_core_axis <= lastest_core_limit, partition=True)
        no_tail_core_condition = self._get_condition_of_multi_k_axis(real_multi_core_axis, lastest_core,
                                                                     no_tail_core_k_len)
        tail_core_condition = self._get_condition_of_multi_k_axis(real_multi_core_axis,
                                                                  lastest_core,
                                                                  tail_core_k_len,
                                                                  is_tail_core=True)
        condition = tvm.any(no_tail_core_condition, tail_core_condition)

        l0_ka = self.tiling_work.al0_tiling_ka * self.block_reduce
        block_size = self.block_in * self.block_reduce
        k_tail_block_stride = (no_tail_core_k_len - no_tail_core_k_len // self.tiling_work.al0_tiling_ka *
                               self.tiling_work.al0_tiling_ka) * block_size
        k_tail_block_stride_tail_core = (tail_core_k_len - tail_core_k_len // self.tiling_work.al0_tiling_ka *
                                         self.tiling_work.al0_tiling_ka) * block_size

        al0_k_dim_full_load = self.status_controller.al0_attach_status != "c_l0c"
        if (not self.is_dynamic and no_tail_core_k_len % self.tiling_work.al0_tiling_ka == 0) or al0_k_dim_full_load:
            return

        self.sch[a_l0].set_store_predicate(
            a_l0.op.axis[-3].var - real_multi_core_axis * no_tail_core_k_len < no_tail_core_k_len, partition=True)

        # the stride of m axis
        ori_stride = l0_ka * self.block_in
        k_tail_block_stride_final = tvm.select(real_multi_core_axis < lastest_core, k_tail_block_stride,
                                               k_tail_block_stride_tail_core)

        stride = tvm.select(condition, k_tail_block_stride_final, ori_stride)
        self.sch[a_l0].bind_buffer(a_l0.op.axis[-4], stride, 0)
        if self.is_dynamic:
            self.sch[a_l0].set_buffer_size(l0_ka * self.tiling_work.al0_tiling_ma * self.block_in)

    def _get_condition_of_multi_k_axis(self, multi_core_axis, block_dim_k, k_len, is_tail_core=False):
        c_l0c = self.container.tensor_map.get("c_l0c")
        # the axis number of l0c is 6
        total_axis_l0c = 6
        scopes_intrins = self.sch_agent[c_l0c].intrin_scopes(total_axis_l0c)
        scope_insn = scopes_intrins[0]
        inner_k_axis = self.sch_agent[c_l0c].get_relate_scope(c_l0c.op.reduce_axis[1], scope_insn)
        inner_k_axis = inner_k_axis[::-1]
        k_axis_values = self._get_k_outer_values(k_len)
        k_axis_values = k_axis_values[::-1]

        for index, axis in enumerate(inner_k_axis):
            if index == 0:
                current_param = self.tiling_work.al0_tiling_ka
                base_condition = axis * current_param
            else:
                base_condition += axis * current_param
            current_param *= k_axis_values[index]
        condition = base_condition >= (k_len // self.tiling_work.al0_tiling_ka * self.tiling_work.al0_tiling_ka)
        if is_tail_core:
            condition = tvm.all(condition, multi_core_axis >= block_dim_k)
        else:
            condition = tvm.all(condition, multi_core_axis < block_dim_k)
        return condition

    def _get_k_outer_values(self, k_len):
        k_axis_values = []
        k_split_info = [
            self.tiling_work.al0_tiling_ka, self.tiling_work.al1_tiling_k // self.block_reduce,
            self.tiling_work.bl1_tiling_k // self.block_reduce
        ]
        if self.status_controller.aub_attach_status == "c_l0c":
            k_split_info.append(self.tiling_work.aub_tiling_k // self.block_reduce)
        if self.status_controller.bub_attach_status == "c_l0c":
            k_split_info.append(self.tiling_work.bub_tiling_k // self.block_reduce)
        k_split_info = list(set(k_split_info))
        k_split_info = sorted(k_split_info)
        k_split_info_new = k_split_info + [k_len]
        before_value = k_split_info[0]
        for i in k_split_info_new[1:]:
            k_axis_values.insert(0, (i + before_value - 1) // before_value)
            before_value = (i + before_value - 1) // before_value * before_value
        return k_axis_values

    def _fusion_para(self):
        res = self.res
        self.out_addr_type = self._get_addr_type(res)  # 0:DDR;1:L1
        self.format_info["out"] = self._get_output_format(res)

    def _get_b_l1_fractal(self):
        b_l0b = self.container.tensor_map.get("b_l0b")
        get_tensor_from_compress = ("tile_L1_n" in b_l0b.op.attrs)

        if get_tensor_from_compress:
            self.container.tensor_map["compress_index"] = b_l0b.op.input_tensors[0]
            self.container.tensor_map["b_l1"] = self.sch.cache_write(b_l0b, tbe_platform_info.scope_cbuf)
        else:
            self.container.tensor_map["b_l1"] = self.sch.cache_read(self.container.tensor_map.get("b_placehold"),
                                                     tbe_platform_info.scope_cbuf,
                                                     [self.container.tensor_map.get("b_l0b")])

    def _check_fusion_before(self):
        if self.container.tensor_map["a_placehold"] is None or self.container.tensor_map["b_placehold"] is None:
            error_manager_cube.raise_err_message_cube("Don't support op + Gemm/MatMul/BatchMatMul ub_fusion")

    def _set_data_layout(self, res):
        self.container.compute_inline_list = []
        self.container.tensor_map = {}
        # for compute at and db
        self.container.compute_tensors = self._get_compute_tensor(res)
        self._set_data_layout_base_tensor()
        if self.status_controller.cube_vector_split:
            self._set_data_layout_cube_vector_split()
        else:
            self._set_data_layout_after_mmad()
            self._set_data_layout_a_matrix()
            self._set_data_layout_b_matrix()
            self._set_data_layout_fusion()
            self._set_data_layout_multi_output()
        print_ir_matmul(self.DEBUG_IR, "ir after set data layout", self.sch)

    def _set_nd_out_compute_inline(self):
        """
        in cache_tiling: cl0->cast_to_fp16->before_cgm(compute_inline)->nz_to_nd->res
        """
        if self.format_info.get("out") == "ND" and self.cache_tiling:
            self._add_tensor_to_list(
                self.container.tensor_map.get("before_c_gm"), [self.container.compute_inline_list])
            self.container.tensors_in_cub.remove(self.container.tensor_map.get("before_c_gm"))

    def _change_l0_gemv(self):
        if self.status_controller.mmad_mode == "gemv":
            self.container.tensor_map["a_l0a"], self.container.tensor_map["b_l0b"] = (
                self.container.tensor_map.get("b_l0b"), self.container.tensor_map.get("a_l0a"))

    def _get_compute_tensor(self, tensor):
        """
        scan all the transient tensor during calculation
        tensor: target tensor which needs to find placeholder tensor
        """
        compute_tensors_local = []
        placeholder_tensors = self.container.placeholder_tensors

        def enter(tensor):
            """
            get compute tensors by search
            """
            if tensor not in compute_tensors_local:
                compute_tensors_local.append(tensor)
            tensor_list = tensor.op.input_tensors
            for one_tensor in tensor_list:
                # check which tensor has not been checked
                if isinstance(one_tensor.op, tvm.tensor.PlaceholderOp):
                    if one_tensor not in placeholder_tensors:
                        placeholder_tensors.append(one_tensor)
                else:
                    if one_tensor not in compute_tensors_local:
                        compute_tensors_local.append(one_tensor)
                        enter(one_tensor)
        enter(tensor)
        return compute_tensors_local

    def _set_data_layout_base_tensor(self):
        tensor_map, all_tensor = self.tensor_list[:2]
        placeholder_name = self.container.placeholder_name
        self.container.tensor_map["c_gm"] = tensor_map.get("tensor_c_gm")
        self.container.tensor_map["a_placehold"] = all_tensor.get(placeholder_name["a"].value)
        self.container.tensor_map["b_placehold"] = all_tensor.get(placeholder_name["b"].value)
        self._check_fusion_before()

        self.container.tensor_map["alpha"] = all_tensor.get(placeholder_name["alpha"].value)
        self.container.tensor_map["beta"] = all_tensor.get(placeholder_name["beta"].value)
        self.container.tensor_map["c_l0c"] = tensor_map.get("c_l0c")
        self.sch[self.container.tensor_map.get("c_l0c")].set_scope(tbe_platform_info.scope_cc)
        if self.status_controller.mmad_mode == "gemv":
            l0a_scope = tbe_platform_info.scope_cb
        else:
            l0a_scope = tbe_platform_info.scope_ca
        if tensor_map.get("a_l0a") is None:
            self.container.tensor_map["a_l0a"] = self.sch.cache_read(
                self.container.tensor_map.get("a_placehold"),
                l0a_scope, [self.container.tensor_map.get("c_l0c")])
        else:
            self.container.tensor_map["a_l0a"] = tensor_map["a_l0a"]
            self.sch[self.container.tensor_map.get("a_l0a")].set_scope(l0a_scope)
        if "mode" in self.container.tensor_map.get("a_l0a").op.attrs:
            self.get_a_matrix_mode = self.container.tensor_map.get("a_l0a").op.attrs["mode"]

        if self.status_controller.mmad_mode == "gemv":
            l0b_scope = tbe_platform_info.scope_ca
        else:
            l0b_scope = tbe_platform_info.scope_cb
        if tensor_map.get("b_l0b") is None:
            self.container.tensor_map["b_l0b"] = self.sch.cache_read(
                self.container.tensor_map.get("b_placehold"),
                l0b_scope, [self.container.tensor_map.get("c_l0c")])
        else:
            self.container.tensor_map["b_l0b"] = tensor_map["b_l0b"]
            self.sch[self.container.tensor_map.get("b_l0b")].set_scope(l0b_scope)
        if "mode" in self.container.tensor_map.get("b_l0b").op.attrs:
            self.get_b_matrix_mode = self.container.tensor_map.get("b_l0b").op.attrs["mode"]

        self.optmt_a = self.container.tensor_map.get("a_l0a").dtype
        self.optmt_b = self.container.tensor_map.get("b_l0b").dtype
        self.optmt_c = self.container.tensor_map.get("c_gm").dtype
        self._add_tensor_to_list(self.container.tensor_map.get("c_l0c"), [self.container.tensors_in_l0c])

    def _set_data_layout_cube_vector_split(self):
        tiling = self.tiling_work.tiling
        self.container.tensor_map["a_l1"] = self.sch.cache_read(self.container.tensor_map.get("a_placehold"),
            tbe_platform_info.scope_cbuf, [self.container.tensor_map.get("a_l0a")])
        if self.src_dtype != "int8" or tiling.get("BL1_shape") is not None:
            self.container.tensor_map["b_l1"] = self.sch.cache_read(
                self.container.tensor_map.get("b_placehold"), tbe_platform_info.scope_cbuf,
                [self.container.tensor_map.get("b_l0b")])

    def _set_tensor_scope(self, tensor, buffer_local):
        if tensor is not None:
            self.sch[tensor].set_scope(buffer_local)

    def _set_data_layout_a_in_nd2zz(self):
        self.container.tensor_map["a_l1"] = self._match_and_get_tensor(self.container.compute_tensors, "tensor_a_l1")
        if self.container.tensor_map.get("a_l1") is not None:
            self.sch[self.container.tensor_map.get("a_l1")].set_scope(tbe_platform_info.scope_cbuf)
        else:
            self.container.tensor_map["a_l1"] = self.sch.cache_write(
                self.container.tensor_map.get("a_l0a"), tbe_platform_info.scope_cbuf)
        if self.status_controller.ops_data_flow_mode == "int82fp32":
            self._get_tensor_and_set_scope("tensor_a_s82f16", tbe_platform_info.scope_ubuf, "a_int82fp16")
        if self.optmt_a == "float16":
            self.container.tensor_map["a_ub_fract"] = self.sch.cache_write(self.container.tensor_map.get("a_l1"),
                                                                 tbe_platform_info.scope_ubuf)

    def _set_data_layout_a_matrix(self):
        tensors_in_aub = self.container.tensors_in_aub
        if self.is_dynamic:
            self._get_tensor_and_set_scope("tensor_a_already_aligned",
                                           tbe_platform_info.scope_ubuf, "a_ub_aligned")
            self._get_tensor_and_set_scope("tensor_a_do_align",
                                           tbe_platform_info.scope_ubuf, "a_ub_general")
            self._add_tensor_to_list(self.container.tensor_map.get("a_ub_aligned"), [tensors_in_aub])
            self._add_tensor_to_list(self.container.tensor_map.get("a_ub_general"), [tensors_in_aub])
        self._get_tensor_and_set_scope("tensor_a_aligned", tbe_platform_info.scope_ubuf, "a_ub")

        if self.get_a_matrix_mode == "none":
            self.container.tensor_map["a_l1"] = self.sch.cache_write(
                self.container.tensor_map.get("a_l0a"), tbe_platform_info.scope_cbuf)
        elif self.get_a_matrix_mode == "nd2Zz_vnchwconv":
            self._get_tensor_and_set_scope("tensor_a_zz_fract_k", tbe_platform_info.scope_ubuf, "a_ub_fract")
            self.container.tensor_map["a_l1"] = self.sch.cache_read(self.container.tensor_map.get("a_ub_fract"),
                tbe_platform_info.scope_cbuf, [self.container.tensor_map.get("a_l0a")])
            if self.status_controller.ops_data_flow_mode == "int82fp32":
                # int82fp32 need cast to fp16
                self._get_tensor_and_set_scope("tensor_a_s82f16", tbe_platform_info.scope_ubuf, "a_int82fp16")
        elif self.get_a_matrix_mode == "nd2Zz_int8":
            if self.status_controller.transpose_a:
                self._get_tensor_and_set_scope("tensor_a_transpose", tbe_platform_info.scope_ubuf, "a_transpose")
            self.container.tensor_map["a_l1"] = self.sch.cache_write(
                self.container.tensor_map.get("a_l0a"), tbe_platform_info.scope_cbuf)
        elif self.get_a_matrix_mode == "Nz2Zz_int82fp32":
            # int82fp32 need cast to fp16
            self._get_tensor_and_set_scope("tensor_a_s82f16", tbe_platform_info.scope_ubuf, "a_int82fp16")
            self.container.tensor_map["a_ub"] = self.sch.cache_read(self.container.tensor_map.get("a_placehold"),
                tbe_platform_info.scope_ubuf, [self.container.tensor_map.get("a_int82fp16")])
            self.container.tensor_map["a_l1"] = self.sch.cache_write(
                self.container.tensor_map.get("a_l0a"), tbe_platform_info.scope_cbuf)
            self.container.tensor_map["a_ub_fract"] = self.sch.cache_write(self.container.tensor_map.get("a_l1"),
                tbe_platform_info.scope_ubuf)
        elif self.get_a_matrix_mode in ("Nz2Zz", "fractal_gemv", "Zz_trans"):
            self.container.tensor_map["a_l1"] = self.sch.cache_read(self.container.tensor_map.get("a_placehold"),
                tbe_platform_info.scope_cbuf, [self.container.tensor_map.get("a_l0a")])
        elif self.get_a_matrix_mode == "nd_gemv":
            self._get_tensor_and_set_scope("tensor_a_nd2zz", tbe_platform_info.scope_cbuf, "a_l1")
            if self.optmt_a == "float16":
                self.container.tensor_map["a_ub_fract"] = self.sch.cache_write(self.container.tensor_map.get("a_l1"),
                                                                     tbe_platform_info.scope_ubuf)
        elif self.get_a_matrix_mode == "nd_gevm":
            self.container.tensor_map["a_l1"] = self.sch.cache_write(
                self.container.tensor_map.get("a_l0a"), tbe_platform_info.scope_cbuf)
            # check in int8
            self._get_tensor_and_set_scope("tensor_a_nd2zz", tbe_platform_info.scope_ubuf, "a_ub_fract")
        elif self.get_a_matrix_mode == "nd2Zz":
            self._set_data_layout_a_in_nd2zz()

        self._handle_virtual_align()
        self._add_tensor_to_list(self.container.tensor_map.get("a_int82fp16"), [tensors_in_aub])
        self._add_tensor_to_list(self.container.tensor_map.get("a_ub"), [tensors_in_aub])
        self._add_tensor_to_list(self.container.tensor_map.get("a_ub_fract"), [tensors_in_aub])
        self._add_tensor_to_list(self.container.tensor_map.get("a_transpose"), [tensors_in_aub])

    def _handle_virtual_align(self):
        a_ub = self.container.tensor_map.get("a_ub")
        # set a_ub as a_ub_virtual_align, a_ub_virtual_align will do nothing, and reuse this two
        if a_ub is not None and "virtual_align" in a_ub.op.attrs:
            self.container.tensor_map["a_ub_virtual_align"] = a_ub
            self.container.tensor_map["a_ub"] = self.sch.cache_read(self.container.tensor_map.get("a_placehold"),
                                                                    tbe_platform_info.scope_ubuf, [a_ub])
            self._add_key_value(self.container.tensor_map.get("a_ub_virtual_align"),
                                self.container.tensor_map.get("a_ub"))
            self._add_tensor_to_list(self.container.tensor_map.get("a_ub_virtual_align"),
                                     [self.container.tensors_in_aub])

    def _set_data_layout_b_matrix(self):
        tensors_in_bub = self.container.tensors_in_bub
        if self.is_dynamic:
            self._get_tensor_and_set_scope("tensor_b_already_aligned",
                                           tbe_platform_info.scope_ubuf, "b_ub_aligned")
            self._get_tensor_and_set_scope("tensor_b_do_align",
                                           tbe_platform_info.scope_ubuf, "b_ub_general")
            self._add_tensor_to_list(self.container.tensor_map.get("b_ub_aligned"), [tensors_in_bub])
            self._add_tensor_to_list(self.container.tensor_map.get("b_ub_general"), [tensors_in_bub])
        self._get_tensor_and_set_scope("tensor_b_aligned", tbe_platform_info.scope_ubuf, "b_ub")

        if self.get_b_matrix_mode == "nd_gemv":
            self._get_tensor_and_set_scope("tensor_b_nd2zz", tbe_platform_info.scope_cbuf, "b_l1")
            if self.optmt_b == "float16":
                self.container.tensor_map["b_ub_fract"] = self.sch.cache_write(self.container.tensor_map.get("b_l1"),
                                                                     tbe_platform_info.scope_ubuf)
        elif self.get_b_matrix_mode == "nd2Zn_vnchwconv":
            self._get_tensor_and_set_scope("tensor_b_zn_fract", tbe_platform_info.scope_ubuf, "b_ub_fract")
            self.container.tensor_map["b_l1"] = self.sch.cache_read(self.container.tensor_map.get("b_ub_fract"),
                tbe_platform_info.scope_cbuf, [self.container.tensor_map.get("b_l0b")])
            if self.status_controller.ops_data_flow_mode == "int82fp32":
                # if int82fp32 need cast to fp16
                self._get_tensor_and_set_scope("tensor_b_s82f16", tbe_platform_info.scope_ubuf, "b_int82fp16")
        elif self.get_b_matrix_mode == "nd2Zn_int8":
            if not self.status_controller.transpose_b:
                self._get_tensor_and_set_scope("tensor_b_transpose", tbe_platform_info.scope_ubuf, "b_transpose")
            self.container.tensor_map["b_l1"] = self.sch.cache_write(
                self.container.tensor_map.get("b_l0b"), tbe_platform_info.scope_cbuf)
        elif self.get_b_matrix_mode == "Zn2Zn_int82fp32":
            self._get_tensor_and_set_scope("tensor_b_s82f16", tbe_platform_info.scope_ubuf, "b_int82fp16")
            self.container.tensor_map["b_ub"] = self.sch.cache_read(self.container.tensor_map.get("b_placehold"),
                tbe_platform_info.scope_ubuf, [self.container.tensor_map.get("b_int82fp16")])
            self.container.tensor_map["b_l1"] = self.sch.cache_write(
                self.container.tensor_map.get("b_l0b"), tbe_platform_info.scope_cbuf)
            self.container.tensor_map["b_ub_fract"] = self.sch.cache_write(self.container.tensor_map.get("b_l1"),
                tbe_platform_info.scope_ubuf)
        elif self.get_b_matrix_mode in ("Nz2Zn", "Nz2Zz", "fractal_gemv", "Zn_trans"):
            self.container.tensor_map["b_l1"] = self.sch.cache_read(self.container.tensor_map.get("b_placehold"),
                tbe_platform_info.scope_cbuf, [self.container.tensor_map.get("b_l0b")])
        elif self.get_b_matrix_mode == "nd2Zn":
            self.container.tensor_map["b_l1"] = self._match_and_get_tensor(
                self.container.compute_tensors, "tensor_b_l1")
            if self.container.tensor_map.get("b_l1") is not None:
                self.sch[self.container.tensor_map.get("b_l1")].set_scope(tbe_platform_info.scope_cbuf)
            else:
                self.container.tensor_map["b_l1"] = self.sch.cache_write(self.container.tensor_map.get("b_l0b"),
                    tbe_platform_info.scope_cbuf)
            self.container.tensor_map["b_ub_fract"] = self.sch.cache_write(self.container.tensor_map.get("b_l1"),
                tbe_platform_info.scope_ubuf)
        else:
            self._get_b_l1_fractal()

        self._add_tensor_to_list(self.container.tensor_map.get("b_int82fp16"), [tensors_in_bub])
        self._add_tensor_to_list(self.container.tensor_map.get("b_ub_fract"), [tensors_in_bub])
        self._add_tensor_to_list(self.container.tensor_map.get("b_ub"), [tensors_in_bub])
        self._add_tensor_to_list(self.container.tensor_map.get("b_transpose"), [tensors_in_bub])

    def _set_scope_bias_in_l0c(self):
        self._get_tensor_and_set_scope("tensor_bias_ub", tbe_platform_info.scope_ubuf, "bias_ub")
        self._get_tensor_and_set_scope("tensor_bias_nz", tbe_platform_info.scope_cc, "bias_l0c")
        self._get_tensor_and_set_scope("tensor_mmad_with_bias", tbe_platform_info.scope_cc, "c_add_bias")
        if self.status_controller.need_init_bias:
            self._get_tensor_and_set_scope('tensor_init_value_of_bias_ub', tbe_platform_info.scope_ubuf,
                'init_value_of_bias_ub')
            self._get_tensor_and_set_scope('tensor_virtual_add_bias', tbe_platform_info.scope_ubuf, 'virtual_add_bias')

    def _set_data_layout_after_mmad(self):
        all_tensor = self.tensor_list[1]
        tensors_in_cub = self.container.tensors_in_cub
        tensors_in_l0c = self.container.tensors_in_l0c
        placeholder_name = self.container.placeholder_name
        self.container.tensor_map["bias"] = all_tensor.get(placeholder_name["bias"].value)
        self.container.tensor_map["tensor_c"] = all_tensor.get(placeholder_name["c"].value)
        self._get_tensor_and_set_scope("tensor_mmad_with_scale", tbe_platform_info.scope_ubuf, "c_ub_fract")

        bias_ub_compute_at = []
        if self.container.tensor_map["bias"] is not None and not self.status_controller.cube_vector_split:
            bias_ub_compute_at = tensors_in_l0c
            self._set_scope_bias_in_l0c()

        if self.container.tensor_map["tensor_c"] is not None:
            bias_ub_compute_at = tensors_in_cub
            self._get_tensor_and_set_scope("tensor_gemm", tbe_platform_info.scope_ubuf, "c_add_bias_ub")
            self._get_tensor_and_set_scope("tensor_bias_aligned", tbe_platform_info.scope_ubuf, "bias_ub")
            if self.status_controller.ops_data_flow_mode == "fp162fp16":
                self._get_tensor_and_set_scope("tensor_b_f32", tbe_platform_info.scope_ubuf, "bias_cast_to_fp32")

            if self.container.tensor_map.get("beta") is not None:
                self._get_tensor_and_set_scope("tensor_beta_bias", tbe_platform_info.scope_ubuf, "beta_bias")
                if self.status_controller.ops_data_flow_mode == "fp162fp16":
                    self._get_tensor_and_set_scope("tensor_beta_f162f32",
                        tbe_platform_info.scope_ubuf, "beta_fp162fp32")
                    self.container.tensor_map["beta_ub"] = self.sch.cache_read(self.container.tensor_map.get("beta"),
                        tbe_platform_info.scope_ubuf, [self.container.tensor_map.get("beta_fp162fp32")])
                else:
                    self.container.tensor_map["beta_ub"] = self.sch.cache_read(self.container.tensor_map.get("beta"),
                        tbe_platform_info.scope_ubuf, [self.container.tensor_map.get("beta_bias")])

        if self.container.tensor_map.get("alpha") is not None:
            self._get_tensor_and_set_scope("tensor_alpha_mmad", tbe_platform_info.scope_ubuf, "alpha_c")
            if self.status_controller.ops_data_flow_mode == "fp162fp16":
                self._get_tensor_and_set_scope("tensor_alpha_f162f32",
                    tbe_platform_info.scope_ubuf, "alpha_fp162fp32")
                self.container.tensor_map["alpha_ub"] = self.sch.cache_read(self.container.tensor_map.get("alpha"),
                    tbe_platform_info.scope_ubuf, [self.container.tensor_map.get("alpha_fp162fp32")])
            else:
                self.container.tensor_map["alpha_ub"] = self.sch.cache_read(self.container.tensor_map.get("alpha"),
                    tbe_platform_info.scope_ubuf, [self.container.tensor_map.get("alpha_c")])

        self._get_tensor_and_set_scope("tensor_gemm_f16", tbe_platform_info.scope_ubuf, "cast_to_fp16")

        self.container.tensor_map["nz_to_nd"] = all_tensor.get("tensor_nz2nd")
        if self.container.tensor_map["nz_to_nd"] is not None:
            self.sch[self.container.tensor_map["nz_to_nd"]].set_scope(tbe_platform_info.scope_ubuf)
        self.container.tensor_map["before_c_gm"] = all_tensor.get("before_c_gm")
        if self.container.tensor_map["before_c_gm"] is not None:
            self.sch[self.container.tensor_map["before_c_gm"]].set_scope(tbe_platform_info.scope_ubuf)

        self._add_tensor_to_list(self.container.tensor_map.get("bias_l0c"), [tensors_in_l0c])
        self._add_tensor_to_list(self.container.tensor_map.get("beta_bias"), [tensors_in_cub])
        self._add_tensor_to_list(self.container.tensor_map.get("c_ub_fract"), [tensors_in_cub])
        self._add_tensor_to_list(self.container.tensor_map.get("alpha_c"), [tensors_in_cub])
        self._add_tensor_to_list(self.container.tensor_map.get("c_add_bias"), [tensors_in_l0c])
        self._add_tensor_to_list(self.container.tensor_map.get("c_add_bias_ub"), [tensors_in_cub])
        self._add_tensor_to_list(self.container.tensor_map.get("cast_to_fp16"), [tensors_in_cub])
        self._add_tensor_to_list(self.container.tensor_map.get("bias_cast_to_fp32"), [tensors_in_cub])
        self._add_tensor_to_list(self.container.tensor_map.get("bias_ub"), [bias_ub_compute_at])
        if self.status_controller.need_init_bias:
            self._add_tensor_to_list(self.container.tensor_map.get("init_value_of_bias_ub"), [bias_ub_compute_at])
            self._add_tensor_to_list(self.container.tensor_map.get("virtual_add_bias"), [bias_ub_compute_at])
        self._add_tensor_to_list(self.container.tensor_map.get("nz_to_nd"), [tensors_in_cub])
        self._add_tensor_to_list(self.container.tensor_map.get("before_c_gm"), [tensors_in_cub])

        cast_to_fp16 = self.container.tensor_map.get("cast_to_fp16")
        c_ub_fract = self.container.tensor_map.get("c_ub_fract")
        self.status_controller.compute_inline_c_ub_fract = ((cast_to_fp16 is not None) and
            (cast_to_fp16.op.input_tensors[0] == c_ub_fract))
        if self.status_controller.compute_inline_c_ub_fract:
            self._add_tensor_to_list(c_ub_fract, [self.container.compute_inline_list])

    def _set_data_layout_multi_output(self):
        """ set tensor_c_gm to ubuf scope and not do compute_inline
        """
        if self.status_controller.fusion_multi_output_flag:
            tensor_c_gm = self.container.tensor_map.get("c_gm")
            self._add_tensor_to_list(tensor_c_gm, [self.container.tensors_in_cub])
            self.sch[tensor_c_gm].set_scope(tbe_platform_info.scope_ubuf)
            if tensor_c_gm in self.container.compute_inline_list:
                self.container.compute_inline_list.remove(tensor_c_gm)

    def _get_tensor_and_set_scope(self, tensor_name, buffer_name, save_name=None):
        if save_name is None:
            save_name = tensor_name
        self.container.tensor_map[save_name] = self._match_and_get_tensor(self.container.compute_tensors, tensor_name)
        if self.container.tensor_map.get(save_name) is not None:
            self.sch[self.container.tensor_map.get(save_name)].set_scope(buffer_name)

    def _set_and_add_tensor(self, tensor, tensors_lists, buffer_type):
        if tensor is not None:
            self.sch[tensor].set_scope(buffer_type)

            for a_list in tensors_lists:
                a_list.append(tensor)

    def _get_quant_fusion_tensor(self):
        matmul_dequant_tensor = self.container.matmul_dequant_tensor
        tensor_fusion_list = self.container.tensor_fusion_list
        if not self.status_controller.quant_fusion:
            return
        for ten_in in self.container.compute_tensors:
            if ten_in == self.res:
                continue
            if ten_in not in matmul_dequant_tensor and ten_in.op.name in self.emit_fusion_insn_map:
                tensor_fusion_list.append(ten_in)

    def _get_elewise_ub_tensors(self, tensor_ele_ub):
        """
        get axpy_ub to axpy_parents[1]_ub dict, in order to set reused_by.
        """
        axpy_and_parent = {}
        for ten_i in self.container.elemwise_tensors:
            if "broadcast" in ten_i.op.tag:
                self.container.compute_inline_list.append(ten_i)
            else:
                if ten_i in self.container.elewise_compute_inline_list:
                    continue
                ele_ub = self.sch.cache_write(ten_i, tbe_platform_info.scope_ubuf)
                tensor_ele_ub.append(ele_ub)
                self.container.elewise_compute_inline_list.append(ten_i)
                if "elewise_binary_scalar_axpy" in ten_i.op.tag:
                    ele_ub_input = self.sch.cache_write(ten_i.op.input_tensors[1], tbe_platform_info.scope_ubuf)
                    self.container.elewise_compute_inline_list.append(ten_i.op.input_tensors[1])
                    axpy_and_parent[ele_ub] = ele_ub_input
        return axpy_and_parent

    def _emit_requant_fusion_insn(self):
        tensor_reform = self.container.tensor_map.get("tensor_reform")
        if tensor_reform is None:
            return
        insn = self.requant_fusion_insn_map.get(tensor_reform.op.name)
        # axiss are batch, n1, m1, n0.outer(2), m0, n0.inner(16), the axis n0.outer should be out of emit_insn axis
        self.sch[tensor_reform].emit_insn(tensor_reform.op.axis[-3], insn)
        return

    def _dequant_fusion_proc(self):
        dequant_tensor = self.container.tensor_map.get("dequant_tensor")
        tensor_sqrt = self.container.tensor_map.get("tensor_sqrt")
        self._emit_insn_func(self.container.tensor_map.get("tensor_deq_ub"), 0, "dma_copy")
        self._dequant_activation_emit_insn_simple()
        dequant_emit_axis, deq_scale_mode = (1, "vector") \
            if "vector" in dequant_tensor.op.tag else (0, "scalar")

        if cube_util.is_ng1_version() or cube_util.is_lhisi_cs_version():
            self.sch[dequant_tensor].emit_insn(
                dequant_tensor.op.axis[dequant_emit_axis], "dma_copy")
        else:
            self.sch[dequant_tensor].pragma(
                dequant_tensor.op.axis[dequant_emit_axis], "deq_scale", deq_scale_mode)
            self._emit_insn_func(tensor_sqrt, 0, "vector_auto")

        self._compute_inline_dequant_output()

    def _compute_inline_dequant_output(self):
        """
        compute inline dequant output tensor when dequant is not the last op.
        """
        compute_inline_list = self.container.compute_inline_list
        dequant_nz = self.container.tensor_map.get("dequant_nz")
        dequant_nd = self.container.tensor_map.get("dequant_nd")
        if dequant_nz is not None and self.res != dequant_nz:
            self._add_tensor_to_list(dequant_nz, [compute_inline_list])
        if dequant_nd is not None and self.res != dequant_nd:
            self._add_tensor_to_list(dequant_nd, [compute_inline_list])

    def _quant_fusion_proc(self):
        input_ub = self.container.tensor_map.get("tensor_input_ub")
        self._add_tensor_to_list(input_ub, [self.container.compute_inline_list])
        for ten_in in self.container.tensor_fusion_list:
            if ten_in.op.name == "cast_i8_ub":
                insn = self._round_emit_insn(self.status_controller.round_mode)
            else:
                insn = self.emit_fusion_insn_map.get(ten_in.op.name)
            if ten_in.op.name in self.reform_tensor_tag_list:
                self.sch[ten_in].emit_insn(ten_in.op.axis[2], insn)
            else:
                self.sch[ten_in].emit_insn(ten_in.op.axis[0], insn)

    def _dequant_activation_emit_insn_simple(self):
        if self.status_controller.dequant_fusion:
            for ten_in in self.container.dequant_activation_tensor:
                if ten_in.op.tag.find("|") != -1:
                    str_list = ten_in.op.tag.split("|")
                    insn = self.emit_insn_map.get(str_list[0])
                else:
                    insn = self.emit_insn_map.get(ten_in.op.tag)
                if ten_in in self.container.header_ub_tensors:
                    insn = "dma_copy"
                if insn is None:
                    insn = "vector_auto"
                if "elewise_binary_scalar_axpy" in ten_in.op.tag:
                    self.sch[ten_in].reused_by(ten_in.op.input_tensors[1])
                self.sch[ten_in].emit_insn(ten_in.op.axis[0], insn)

    def _requant_fusion_proc(self):
        requant_scale = self.container.tensor_map.get("requant_scale")
        tensor_drq = requant_scale.op.input_tensors[1]
        tensor_drq_ub = self.sch.cache_read(tensor_drq, tbe_platform_info.scope_ubuf, [requant_scale])
        self.sch[tensor_drq_ub].emit_insn(tensor_drq_ub.op.axis[0], "dma_copy")

        self._add_tensor_to_list(requant_scale, [self.container.compute_inline_list])
        self._emit_requant_fusion_insn()

    def _quantify_fusion_entry(self):
        if not self.status_controller.quantify_fusion:
            return
        if self.status_controller.requant_fusion:
            self._requant_fusion_proc()
        if self.status_controller.dequant_fusion:
            self._dequant_fusion_proc()
        if self.status_controller.quant_fusion:
            self._quant_fusion_proc()

        reform_fusion = self.status_controller.quant_fusion or self.status_controller.requant_fusion
        if reform_fusion:
            tensor_len_c = len(self.container.tensor_map.get("c_l0c").shape)
            tensor_reform = self.container.tensor_map.get("tensor_reform")
            reform_c_outer, reform_c_inner = self.sch[tensor_reform].split(
                tensor_reform.op.axis[tensor_len_c - 1], factor=16)
            self.sch[tensor_reform].reorder(
                tensor_reform.op.axis[tensor_len_c - 4],
                reform_c_outer, # emit_insn axis
                tensor_reform.op.axis[tensor_len_c - 3],
                tensor_reform.op.axis[tensor_len_c - 2],
                reform_c_inner)
        return

    def _set_scope_fusion(self):
        dequant_tensor_list = []

        if self.out_addr_type == 1:
            self._set_tensor_scope(self.res, tbe_platform_info.scope_cbuf_fusion)
        if self.in_addr_type == 1:
            tensor_a = self.container.tensor_map.get("a_placehold")
            self._set_tensor_scope(tensor_a, tbe_platform_info.scope_cbuf_fusion)

        if self.status_controller.dequant_fusion:
            dequant_tensor_list.append(self.container.tensor_map["dequant_tensor"])
            if self.status_controller.sqrt_flag:
                tensor_sqrt = self.container.tensor_map.get("tensor_sqrt")
                self.sch[tensor_sqrt].set_scope(tbe_platform_info.scope_ubuf)
                dequant_tensor_list.append(tensor_sqrt)
            for tensor in self.container.dequant_activation_tensor:
                self.sch[tensor].set_scope(tbe_platform_info.scope_ubuf)
            for tensor in self.container.tensor_fusion_list:
                self.sch[tensor].set_scope(tbe_platform_info.scope_ubuf)

        for tensor in self.container.fusion_list:
            if tensor in self.container.double_out_tensor:
                continue
            self.sch[tensor].set_scope(tbe_platform_info.scope_ubuf)

        self._get_tensor_deq(dequant_tensor_list)

    def _get_tensor_deq(self, dequant_tensor_list):
        tensor_deq = self._match_and_get_tensor(self.container.placeholder_tensors, "tensor_deq")
        if tensor_deq is None:
            tensor_dequant = self.container.tensor_map.get("dequant_tensor")
            if tensor_dequant is not None:
                tensor_deq = tensor_dequant.op.input_tensors[1]
        if tensor_deq is not None:
            self.container.tensor_map["tensor_deq"] = tensor_deq
            self.container.tensor_map["tensor_deq_ub"] = self.sch.cache_read(
                self.container.tensor_map.get("tensor_deq"), tbe_platform_info.scope_ubuf, dequant_tensor_list)

    def _get_quant_fusion_tensor_and_flag(self):
        fusion_tensor_cub = self.container.fusion_tensor_cub
        for i in self.tensor_list[1].values():
            if i.op.name == "dequant":
                self.container.tensor_map["dequant_tensor"] = i
                fusion_tensor_cub.append(i)
                self.status_controller.dequant_fusion = True
            if "dequant_sqrt" in i.op.name:
                fusion_tensor_cub.append(i)
                self.container.tensor_map["tensor_sqrt"] = i
                self.status_controller.sqrt_flag = True
            if "dequant_NZ" in i.op.name:
                fusion_tensor_cub.append(i)
                self.container.tensor_map["dequant_nz"] = i
            if "dequant_ND" in i.op.name:
                fusion_tensor_cub.append(i)
                self.container.tensor_map["dequant_nd"] = i
            if i.op.tag == "quant":
                self.container.tensor_map["quant"] = i
                self.status_controller.quant_fusion = True
                self.status_controller.round_mode = i.op.attrs["round_mode"]
            if "input_ub" in i.op.name:
                self.container.tensor_map["tensor_input_ub"] = i
            if i.op.tag == "requant_scale" or i.op.tag == "requant_vector":
                self.status_controller.requant_fusion = True
                self.container.tensor_map["requant_scale"] = i
            if i.op.tag == "requant_data_transfer":
                fusion_tensor_cub.append(i)
                self.container.tensor_map["tensor_reform"] = i
        self.status_controller.quantify_fusion = (
            self.status_controller.requant_fusion or self.status_controller.dequant_fusion
            or self.status_controller.quant_fusion)

    def _atomic_add_batch(self, res):
        """
        atomic add according to refactor res
        """
        if "reduce_sum" in res.op.tag:
            self.status_controller.reduce_fusion = True
            res = self.root_tensor
            # set all batch to ddr add
            block_dim_batch = get_value(self.container.tensor_map.get("c_l0c").shape)[0]
            batch_outer, _ = self.sch[res].split(res.op.reduce_axis[0], nparts=block_dim_batch)
            res_after = res
            res_ub = self.sch.rfactor(res, batch_outer)
            self.sch[res_ub].set_scope(tbe_platform_info.scope_ubuf)
            # put reduce axis first
            self.sch[res_after].reorder(self.sch[res_after].op.reduce_axis[0], *self.sch[res_after].op.axis)
            self.sch[res_ub].reorder(self.sch[res_ub].op.reduce_axis[0], *self.sch[res_ub].op.axis[1:])
            self.container.tensor_map["res_atomic_add_ub"] = res_ub
            self.container.tensors_in_cub.append(res_ub)
            print_ir_matmul(self.DEBUG_IR, "after atomic_add", self.sch)

    def _atomic_add_k_axis(self):
        if not self.status_controller.split_k_axis_by_tiling:
            return
        tensor_l0c = self.container.tensor_map.get("c_l0c")
        real_k = self.sch[tensor_l0c].op.reduce_axis[0]

        block_dim_k = self.tiling_work.tiling.get("block_dim")[3]
        if self.cache_tiling:
            factor_k = self.cache_tiling.get("kal1_factor") * self.cache_tiling.get("kal1_16")
            k_multicore, real_k = self.sch[tensor_l0c].split(real_k, factor=factor_k)
        else:
            k_multicore, real_k = self.sch[tensor_l0c].split(real_k, nparts=block_dim_k)
        atomic_add_l0c = self.sch.rfactor(tensor_l0c, k_multicore)
        atomic_add_ddr = tensor_l0c
        self.sch[atomic_add_ddr].reorder(self.sch[atomic_add_ddr].op.reduce_axis[0], *self.sch[atomic_add_ddr].op.axis)
        self.sch[atomic_add_l0c].reorder(atomic_add_l0c.op.reduce_axis[1], atomic_add_l0c.op.reduce_axis[0])
        self.sch[atomic_add_l0c].set_scope(tbe_platform_info.scope_cc)
        self._add_tensor_to_list(atomic_add_l0c, [self.container.tensors_in_l0c])
        self.sch[atomic_add_ddr].set_scope(tbe_platform_info.scope_gm)
        # remove initialize reduce axis with zero
        self.sch[atomic_add_ddr].remove_init()

        atomic_add_ub = self.sch.cache_read(atomic_add_l0c, tbe_platform_info.scope_ubuf, [atomic_add_ddr])
        self._add_tensor_to_list(atomic_add_ub, [self.container.tensors_in_cub])
        self.container.tensor_map["ori_c_ub_fract"] = self.container.tensor_map.get("c_ub_fract")
        self.container.tensor_map["c_ub_fract"] = atomic_add_ub
        copy_attrs(self.container.tensor_map.get("ori_c_ub_fract"), atomic_add_ub)
        self.container.tensors_in_cub.remove(self.container.tensor_map.get("ori_c_ub_fract"))
        self.sch[self.container.tensor_map.get("ori_c_ub_fract")].set_scope(tbe_platform_info.scope_gm)

        self.container.tensor_map["ori_c_l0c"] = self.container.tensor_map.get("c_l0c")
        self.container.tensor_map["c_l0c"] = atomic_add_l0c
        self.container.tensors_in_l0c.remove(atomic_add_ddr)
        self.container.tensor_map["ori_c_gm"] = self.container.tensor_map.get("c_gm")
        self.container.tensor_map["c_gm"] = atomic_add_ddr
        self.root_tensor = atomic_add_ddr
        copy_attrs(self.container.tensor_map.get("ori_c_gm"), atomic_add_ddr)
        self.sch[self.container.tensor_map.get("ori_c_gm")].set_scope(tbe_platform_info.scope_gm)
        self.sch_list.append(atomic_add_ddr)
        print_ir_matmul(self.DEBUG_IR, "ir after atomic_add_k_axis", self.sch)

    def _emit_insn_after_split_k(self):
        if self.status_controller.split_k_axis_by_tiling:
            ori_c_gm = self.container.tensor_map.get("ori_c_gm")
            ori_c_ub_fract = self.container.tensor_map.get("ori_c_ub_fract")
            self.sch[ori_c_gm].emit_insn(self.sch[ori_c_gm].op.axis[0], "phony_insn")
            self.sch[ori_c_ub_fract].emit_insn(self.sch[ori_c_ub_fract].op.axis[0], "phony_insn")
        return True

    def _set_data_layout_fusion(self):
        all_tensor = self.tensor_list[1]
        fusion_tensor_cub = self.container.fusion_tensor_cub
        if all_tensor.get("reform_by_vadds") is not None:
            self.container.tensor_map["tensor_reform"] = all_tensor["reform_by_vadds"]
        if all_tensor.get("reform_by_vmuls") is not None:
            self.container.tensor_map["tensor_reform"] = all_tensor["reform_by_vmuls"]

        self._atomic_add_batch(self.res)
        self._get_quant_fusion_tensor_and_flag()
        debug(self.DEBUG_PARAM, self.status_controller.quant_fusion, "quant_fusion")
        debug(self.DEBUG_PARAM, self.status_controller.requant_fusion, "requant_fusion")
        debug(self.DEBUG_PARAM, self.status_controller.dequant_fusion, "dequant_fusion")

        matmul_end_tensor = self.container.tensor_map.get("c_gm")
        self.container.matmul_tensors = self._get_compute_tensor(matmul_end_tensor)
        debug(self.DEBUG_PARAM, self.container.matmul_tensors, "matmul_tensors")

        self.container.matmul_dequant_tensor = self._get_matmul_dequant_tensor()
        debug(self.DEBUG_PARAM, self.container.matmul_dequant_tensor, "matmul_dequant_tensor")

        self.container.fusion_ele = self._get_elewise_fusion_tensor()
        debug(self.DEBUG_PARAM, self.container.fusion_ele, "fusion_ele")
        (self.status_controller.gm_ub,
         self.container.ele_header_ub_tensors,
         self.container.axpy_2_parent) = self._set_scope_buffer_type(self.container.placeholder_tensors)
        debug(self.DEBUG_PARAM, self.container.elemwise_tensors, "elemwise_tensors")

        self._get_quant_fusion_tensor()
        if self.container.tensor_fusion_list is not None:
            fusion_tensor_cub += self.container.tensor_fusion_list
        debug(self.DEBUG_PARAM, self.container.tensor_fusion_list, "tensor_fusion_list")

        self._get_matmul_dequant_activation_tensor()
        self._add_res_ub(self.container.dequant_activation_tensor)
        fusion_tensor_cub += self.container.dequant_activation_tensor
        debug(self.DEBUG_PARAM, self.container.dequant_activation_tensor, "dequant_activation_tensor")

        self.container.header_ub_tensors = self._get_header_tensor_in_dequant_ew_fusion()
        fusion_tensor_cub += self.container.header_ub_tensors
        debug(self.DEBUG_PARAM, self.container.header_ub_tensors, "header_ub_tensors")

        self._get_fusion_tensor()
        fusion_tensor_cub += self.container.fusion_list
        debug(self.DEBUG_PARAM, self.container.fusion_list, "fusion_list")

        tensor_a = self.container.tensor_map.get("a_placehold")
        tensor_b = self.container.tensor_map.get("b_placehold")
        is_fractal_a = len(tensor_a.shape) in (4, 5)
        is_fractal_b = len(tensor_b.shape) in (4, 5)
        self.in_addr_type = self._get_addr_type(tensor_a)
        self.l1_fusion_type = self._get_l1_fusion_type(tensor_a)
        self.status_controller.input_l1_flag, self.input_l1_size = self._get_input_l1_paras(tensor_a)
        self._check_placeholders_shared(tensor_a, tensor_b)

        l1_fusion_and_l1_size_0 = self._get_l1_fusion_and_l1_size_0_flag(self.l1_fusion_type)
        self.status_controller.l1_fusion_and_l1_size_0 = l1_fusion_and_l1_size_0
        tensor_a_l1_workspace = self._get_tensor_a_l1_workspace(l1_fusion_and_l1_size_0)
        self._set_l1_fusion_workspace_tensor(tensor_a, tensor_a_l1_workspace)
        self._set_l1_fusion_workspace_size(tensor_a_l1_workspace)
        self.tensor_a_l1_workspace = tensor_a_l1_workspace

        if self.status_controller.mmad_mode == "gemv":
            tensor_a_l1 = self.container.tensor_map.get("b_l1")
            tensor_b_l1 = self.container.tensor_map.get("a_l1")
        else:
            tensor_a_l1 = self.container.tensor_map.get("a_l1")
            tensor_b_l1 = self.container.tensor_map.get("b_l1")

        self.status_controller.a_l1_inline_flag = self._fc_tensor_a_l1_inline(
            tensor_a_l1, is_fractal_a, l1_fusion_and_l1_size_0)
        self.status_controller.b_l1_inline_flag = self._fc_tensor_b_l1_inline(
            tensor_b_l1, is_fractal_b, l1_fusion_and_l1_size_0)

        self._set_scope_fusion()

        if self.container.fusion_ele:
            res_ub = self.sch.cache_write(self.res, tbe_platform_info.scope_ubuf)
            self.container.elemwise_tensors.append(res_ub)

        if self.res != self.container.tensor_map.get("c_gm") and not self.status_controller.matmul_multi_output_flag:
            self.container.compute_inline_list.append(self.container.tensor_map.get("c_gm"))
        # if in quant mode, the tensor of matmul l0c_to_ub tensor need inline
        if self.status_controller.quantify_fusion:
            self.container.compute_inline_list.append(self.container.tensor_map.get("c_ub_fract"))

        if self.tensor_b_reshape:
            self.container.compute_inline_list.append(self.container.tensor_map.get("b_placehold"))
        fusion_tensor_cub += self.container.elemwise_tensors

    def _tensor_a_l1_workspace_emit(self):
        if self.status_controller.input_l1_flag == 1:
            self._emit_insn_func(self.tensor_a_l1_workspace, 0, "dma_copy")

    def _fc_tensor_a_l1_inline(self, tensor_a_l1, is_fractal_a, l1_fusion_and_l1_size_0):
        inline_flag = False
        if (((self.in_addr_type == 1 or self.status_controller.input_l1_flag == 1) and is_fractal_a)
            or l1_fusion_and_l1_size_0):
            self._add_tensor_to_list(tensor_a_l1, [self.container.compute_inline_list])
            inline_flag = True
        return inline_flag

    def _fc_tensor_b_l1_inline(self, tensor_b_l1, is_fractal_b, l1_fusion_and_l1_size_0):
        inline_flag = False
        if l1_fusion_and_l1_size_0 and is_fractal_b:
            self._add_tensor_to_list(tensor_b_l1, [self.container.compute_inline_list])
            inline_flag = True
        return inline_flag

    def _set_quant_by_params(self):
        if not self.status_controller.quantify_fusion:
            c_ub_fract = self.container.tensor_map.get("c_ub_fract")
            if c_ub_fract.op.attrs["scale_drq"].value == "ENABLE":
                # tensor_drq is second input for tensor_c_ub
                tensor_drq = c_ub_fract.op.input_tensors[1]
                tensor_drq_ub = self.sch.cache_read(tensor_drq, tbe_platform_info.scope_ubuf, [c_ub_fract])
                self._emit_insn_func(tensor_drq_ub, 0, "dma_copy")
                if c_ub_fract.op.attrs["sqrt_out"].value == "SQRT":
                    # Sqrt Mode
                    self.sch[c_ub_fract].pragma(c_ub_fract.op.axis[0], "deq_scale", "scalar_sqrt")
                else:
                    # No Sqrt Mode
                    self.sch[c_ub_fract].pragma(c_ub_fract.op.axis[0], "deq_scale", "scalar")

    def _get_batch_factors(self, tensor_a_l0a, tensor_b_l0b):
        if self.status_controller.have_batch:
            if self.is_dynamic:
                batch = self.dynamic_batch
            else:
                batch = get_value(tensor_a_l0a.shape[0])
                if self.status_controller.mmad_mode == "gemv":
                    batch = get_value(tensor_b_l0b.shape[0])
        else:
            batch = 0
        return batch

    def _set_l1_fusion_workspace_tensor(self, tensor_a, tensor_a_l1_workspace):
        if self.status_controller.input_l1_flag == 0:
            L1CommonParam.l1_fusion_tensors_map = {}
            L1CommonParam.l1_fusion_tensors_map[tensor_a] = tvm.var("dummy")
        elif self.status_controller.input_l1_flag == 1:
            L1CommonParam.l1_fusion_tensors_map = {}
            L1CommonParam.l1_fusion_tensors_map[tensor_a] = tensor_a_l1_workspace

    def _set_l1_fusion_workspace_size(self, tensor_a_l1_workspace):
        if self.status_controller.input_l1_flag == 1 and self.input_l1_size > 0:
            self.sch[tensor_a_l1_workspace].set_buffer_size(self.input_l1_size)

    def _get_tensor_a_l1_workspace(self, l1_fusion_and_l1_size_0):
        tensor_a_l1_workspace = None
        tensor_a_ub = self.container.tensor_map.get("a_ub")
        tensor_a_l1 = self.container.tensor_map.get("a_l1")
        tensor_a_l0a = self.container.tensor_map.get("a_l0a")
        tensor_a = self.container.tensor_map.get("a_placehold")
        if self.status_controller.input_l1_flag == 1:
            if tensor_a_ub is not None:
                tensor_a_l1_workspace = self.sch.cache_read(tensor_a, tbe_platform_info.scope_cbuf_fusion, tensor_a_ub)
            elif tensor_a_l1 is not None and not l1_fusion_and_l1_size_0:
                tensor_a_l1_workspace = self.sch.cache_read(tensor_a, tbe_platform_info.scope_cbuf_fusion, tensor_a_l1)
            elif tensor_a_l0a is not None and l1_fusion_and_l1_size_0:
                tensor_a_l1_workspace = self.sch.cache_read(tensor_a, tbe_platform_info.scope_cbuf_fusion, tensor_a_l1)
        return tensor_a_l1_workspace

    def _get_l1_fusion_and_l1_size_0_flag(self, l1_fusion_type):

        trans_b = self.status_controller.transpose_b
        is_l1fusion = l1_fusion_type in (0, 1)
        size = tbe_platform_info.get_soc_spec("L1_SIZE")
        if size == 0 and is_l1fusion:
            if trans_b:
                raise RuntimeError(
                    "If the size of L1 is zero, trans_b is not unexpected.")
            return True
        return False

    def _gen_in_out_tensor_map(self, out_tensor, in_out_tensor_map):
        """
        traverse tensors by Depth-First-Search

        Parameters
        ----------
        out_tensor : tensor
            traverse tensors from this tensor,
            traversing its input tensors recursively.

        tensor_list : list
            record tensors in the order of Depth-First-Search.
        """
        if out_tensor is None:
            return
        stack = [out_tensor]
        visited_list = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor not in visited_list:
                    stack.append(in_tensor)
                self._map_apend(in_out_tensor_map, in_tensor, cur_tensor)

    def _check_placeholders_shared(self, tensor_a, tensor_b):
        """check placeholders shared"""
        leaf_tensor = self.tensor_list[2]
        matmul_tensors = self.container.matmul_tensors
        if not self.container.fusion_ele:
            return True
        if tensor_a.op.name in leaf_tensor:
            for ten_i in leaf_tensor.get(tensor_a.op.name):
                if ten_i not in matmul_tensors:
                    raise RuntimeError("matmul placeholders can't be shared "
                                    "with elementwise op")
        if tensor_b.op.name in leaf_tensor:
            for ten_i in leaf_tensor.get(tensor_b.op.name):
                if ten_i not in matmul_tensors:
                    raise RuntimeError("matmul placeholders can't be shared "
                                    "with elementwise op")
        return True

    def _get_fusion_tensor(self):
        matmul_tensors = self.container.matmul_tensors
        fusion_list = self.container.fusion_list
        tensor_c_gm = self.container.tensor_map.get("c_gm")
        if tensor_c_gm != self.res and tensor_c_gm is not None:
            for ten_in in self.container.compute_tensors:
                if ten_in == self.res and "virtual_res" not in self.root_tensor.op.tag:
                    continue
                if ten_in not in matmul_tensors:
                    fusion_list.append(ten_in)

    def _get_header_tensor_in_dequant_ew_fusion(self):
        """
        add header_ub tensor to dequant_activation_tensor.
        """
        dequant_activation_tensor = self.container.dequant_activation_tensor
        header_set = self.container.placeholder_tensors
        header_ub_tensors = []
        comm_2_elwt = {}
        for ten_i in dequant_activation_tensor:
            common_tensors = set(header_set) & set(ten_i.op.input_tensors)
            for common_tensor in common_tensors:
                if common_tensor in comm_2_elwt:
                    comm_2_elwt[common_tensor].append(ten_i)
                else:
                    comm_2_elwt[common_tensor] = [ten_i]
        for common_tensor, ten_in_list in comm_2_elwt.items():
            common_tensor_ub = self.sch.cache_read(
                common_tensor, tbe_platform_info.scope_ubuf, ten_in_list)
            header_ub_tensors.append(common_tensor_ub)
        dequant_activation_tensor += header_ub_tensors
        return header_ub_tensors

    def _add_res_ub(self, dequant_activation_tensor):
        """
        add res_ub tensor to dequant_activation_tensor.
        """
        for tensor in dequant_activation_tensor:
            if tensor == self.res:
                res_ub = self.sch.cache_write(self.res, tbe_platform_info.scope_ubuf)
                dequant_activation_tensor.remove(tensor)
                dequant_activation_tensor.append(res_ub)

    def _get_matmul_dequant_activation_tensor(self):
        dequant_activation_tensor = self.container.dequant_activation_tensor
        tensor_fusion_list = self.container.tensor_fusion_list
        if not self.status_controller.dequant_fusion:
            return
        dequant_nz = self.container.tensor_map.get("dequant_nz")
        dequant_nd = self.container.tensor_map.get("dequant_nd")
        dequant_tensor = self.container.tensor_map.get("dequant_tensor")
        tensor_sqrt = self.container.tensor_map.get("tensor_sqrt")
        quant = self.container.tensor_map.get("quant")
        tensor_front_dequant = self._get_compute_tensor(dequant_tensor)
        for ten_in in self.container.compute_tensors:
            if ten_in not in tensor_front_dequant:
                dequant_activation_tensor.append(ten_in)
        if tensor_sqrt is not None:
            dequant_activation_tensor.remove(tensor_sqrt)
        for ten_quant in tensor_fusion_list:
            if ten_quant in dequant_activation_tensor:
                dequant_activation_tensor.remove(ten_quant)
        dequant_type_tensor = dequant_nz if dequant_nz is not None else dequant_nd
        if dequant_type_tensor in dequant_activation_tensor:
            dequant_activation_tensor.remove(dequant_type_tensor)
        if self.res in dequant_activation_tensor and quant is not None:
            dequant_activation_tensor.remove(self.res)

    def _get_elewise_fusion_tensor(self):
        elemwise_tensors = self.container.elemwise_tensors
        if self.status_controller.quantify_fusion:
            return False
        if self.status_controller.reduce_fusion:
            real_res = self.res.op.input_tensors[0]
        else:
            real_res = self.res
        tensor_c_gm = self.container.tensor_map.get("c_gm")
        if tensor_c_gm != real_res and tensor_c_gm is not None:
            for ten_in in self.container.compute_tensors:
                if ten_in in (self.res, real_res):
                    continue
                if ten_in not in self.container.matmul_tensors and ten_in not in elemwise_tensors:
                    elemwise_tensors.append(ten_in)
            return True
        return False

    def _get_matmul_dequant_tensor(self):
        if self.container.tensor_map.get("quant") is not None and self.container.tensor_map.get(
                "dequant_nz") is not None:
            compute_tensors = self._get_compute_tensor(self.container.tensor_map.get("dequant_nz"))
            return compute_tensors
        return []

    def _no_solution_tiling(self, tiling):
        """Determining that there is no solution to tilling
        and change tiling to default
        Input:
        tiling: dict, the tiling from tiling_query
        -----------------------------
        Return:
            default tiling
        """
        if tiling.get("AL0_matrix") == [1, 1, 32, 16, 1, 1]:
            multi_m, multi_n = 1, 1
            src_dtype = self.container.tensor_map.get("a_placehold").dtype
            dst_dtype = self.container.tensor_map.get("c_gm").dtype
            if src_dtype in ("uint8", "int8") and dst_dtype == "int32":
                multi_m, multi_n = 2, 2
            if self.status_controller.int8_not_double_m or self.format_info.get("a") != "ND":
                multi_m = 1
            if self.format_info.get("b") != "ND":
                multi_n = 1
            block_reduce = self.block_reduce
            block_in = self.block_in
            block_out = self.block_out
            tiling = {
                'AUB_shape': [block_reduce, multi_m, 1, 1],
                'BUB_shape': [block_reduce, multi_n, 1, 1],
                'AL1_shape': [block_reduce, 1, 1, 1],
                'BL1_shape': [block_reduce, 1, 1, 1],
                'AL0_matrix': [multi_m, 1, block_in, block_reduce, 1, 1],
                'BL0_matrix': [1, multi_n, block_out, block_reduce, 1, 1],
                'CL0_matrix': [multi_n, multi_m, block_in, block_out, 1, 1],
                'CUB_matrix': [multi_n, multi_m, block_in, block_out, 1, 1],
                'block_dim': [1, 1, 1, 1],
                'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'batch_bef_group_flag': 0,
                'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0,
                'AUB_channel_wise_flag': None, 'BUB_channel_wise_flag': None,
                'CUB_channel_wise_flag': 0,
                'manual_pingpong_buffer':
                    {'AUB_pbuffer': 1,
                     'BUB_pbuffer': 1,
                     'AL1_pbuffer': 1,
                     'BL1_pbuffer': 1,
                     'AL0_pbuffer': 1,
                     'BL0_pbuffer': 1,
                     'CL0_pbuffer': 1,
                     'CUB_pbuffer': 1,
                     'UBG_pbuffer': 2
                    },
                "attach_same_to_static": False
            }
            if (self.status_controller.mmad_mode in ("gemv", "gevm")
                and (not self.status_controller.only_use_gevm_gemv_flow)):
                tiling["AUB_shape"] = [block_reduce * block_in, 1, 1, 1]
                tiling["AL1_shape"] = [block_reduce * block_in, 1, 1, 1]
                tiling["BL1_shape"] = [block_reduce * block_in, 1, 1, 1]
                tiling["AL0_matrix"] = [1, block_in, 1, block_reduce, 1, 1]
                tiling["BL0_matrix"] = [block_in, 1, block_out, block_reduce, 1, 1]
            b_ub = self.container.tensor_map.get("b_ub")
            if (b_ub is not None) and b_ub.dtype in ("int8", "uint8"):
                tiling["BUB_shape"][1] = tiling.get("BUB_shape")[1] * 2
        return tiling

    def _gemv_tiling(self, tiling):
        if self.status_controller.mmad_mode != "gemv":
            return tiling

        tiling["AUB_shape"], tiling["BUB_shape"] = tiling.get("BUB_shape"), tiling.get("AUB_shape")
        tiling["AL1_shape"], tiling["BL1_shape"] = tiling.get("BL1_shape"), tiling.get("AL1_shape")
        block_dim = tiling.get("block_dim")
        block_dim[1] = 1
        tiling["block_dim"] = block_dim
        return tiling

    def _check_tiling_value(self, tiling):
        if len(tiling.get("block_dim")) != self.FRACTAL_Z_LEN:
            error_manager_cube.raise_err_message_cube("block_dim should be 4 dim")
        batch_dim, n_dim, m_dim, reduce_dim = tiling.get("block_dim")
        is_valid = ((isinstance(batch_dim, int) and batch_dim < 1) or
                    (isinstance(n_dim, int) and n_dim < 1) or
                    (isinstance(m_dim, int) and m_dim < 1) or
                    (isinstance(reduce_dim, int) and reduce_dim < 1))
        if is_valid:
            error_manager_cube.raise_err_message_cube("block_dim cannot be less than 1")

    def _tiling_process(self):
        """
        :param None:
        :return None
        info_dict
        -----------------------------------------------
        strideH: the data format A matrix and B matrix
                 0: a and b both ND input
                 1: a and b both fractal input
                 2: a is fractal b is ND
                 3: a is ND b is fractal
        strideW: 0 indicates that tail block processing is required.
                 (The non-alignment indicates that the number of
                  bytes in the tail block is less than or equal to 32 bytes.)
        padl: Indicates the multiplier of the AUB extra usage multiplied by 10.
              The multiplier is accurate to 1 decimal point.
        padr: Indicates the multiplier of the BUB extra usage multiplied by 10.
              The multiplier is accurate to 1 decimal point.
        dilationH: 1: A and B are not transposed.
                   2: A is transposed and B is not transposed.
                   3: A is not transposed but B is transposed.
                   4: both A and B are transposed.
                   This flag is not read by the GEMM solution space
        """

        op_type_flag = GEMMComputeParam.get_op_type_flag(
            self.format_info.get("a"), self.format_info.get("b"), self.status_controller.mmad_mode)
        n_shape = self.container.tensor_map.get("b_l0b").shape[-3] * self.block_out
        tail_block = 1
        if self.ops_format == "ND":
            tail_block = GEMMComputeParam.check_tail_block(n_shape, self.status_controller.ops_data_flow_mode,
                                                           self.format_info.get("out"), self.is_dynamic)
        a_type = self.container.tensor_map.get("a_placehold").dtype
        b_type = self.container.tensor_map.get("b_placehold").dtype
        a_type, b_type = (b_type, a_type) if self.status_controller.mmad_mode == "gemv" else (a_type, b_type)

        c_type = self.res.dtype
        a_shape, b_shape = self._get_tiling_param()
        a_ub_fuse_num, b_ub_fuse_num, fused_num = self._compute_buffer_used_multi()

        stride_w = GEMMComputeParam.get_stride_w_value(tail_block, self.status_controller.split_k)
        new_fused_num = fused_num
        scalar_size = 0
        if not self.is_dynamic:
            not_count_list = []
            for tensor_item in self.container.compute_inline_list:
                if tensor_item not in self.container.placeholder_tensors:
                    not_count_list.append(tensor_item)
            multi_ub = CalculateMultiUB(self.container.tensor_map.get("c_ub_fract"), self.res, not_count_list)
            ub_res, scalar_size = multi_ub.calculate_start()
            new_fused_num = ub_res / (self.DTYPE_WIDTH_MAP.get(c_type) * 2) - 1
        self.container.fuse_num_group = [a_ub_fuse_num, b_ub_fuse_num, new_fused_num]
        mad_type = self.MAD_TYPE.get(str(self.status_controller.ops_data_flow_mode))
        bias_flag = self.container.tensor_map.get("c_add_bias") is not None
        trans_flag = GEMMComputeParam.get_trans_flag(
            self.status_controller.transpose_a, self.status_controller.transpose_b)
        # in gemv or gevm, k need align to 256
        is_gevm = (int(self.status_controller.mmad_mode in ("gemv", "gevm"))
            and (not self.status_controller.only_use_gevm_gemv_flow))
        info_dict = {
            "op_type": "matmul",
            "A_shape": a_shape, "B_shape": b_shape, "C_shape": None,
            "A_dtype": a_type, "B_dtype": b_type, "C_dtype": c_type,
            "mad_dtype": mad_type,
            "padl": a_ub_fuse_num, "padr": b_ub_fuse_num, "padu": is_gevm,
            "padd": int(self.status_controller.int8_not_double_m and not self.status_controller.transpose_a),
            "strideH": op_type_flag, "strideW": stride_w,
            "strideH_expand": 1, "strideW_expand": 1,
            "dilationH": trans_flag, "dilationW": 0 if self.status_controller.compress_flag else 1,
            "group": 1,
            "bias_flag": bias_flag,
            "fused_double_operand_num": fused_num,
            "shape_a_align": self.status_controller.align_a, "shape_b_align": self.status_controller.align_b,
            "kernel_name": self.kernel_name,
            "scalar_size": scalar_size,
            "batch_type" : (int(self.status_controller.batch_broadcast_flag) * 4 +
                            int(self.status_controller.have_batch_a) * 2 + int(self.status_controller.have_batch_b)),
            "reduce_fusion": int(self.status_controller.reduce_fusion)
        }
        debug(self.DEBUG_PARAM, info_dict, "info_dict")
        if self.is_dynamic:
            tiling = self.dynamic_para.get("tiling_strategy")
            # binary mode
            if -1 in tiling['block_dim']:
                self.res.op.attrs["cache_tiling"] = 1
                self.cache_tiling_manager.config_cache_tiling(
                    self.cce_simplification_obj, self.compute_param, self.container)
                self.cache_tiling = self.cache_tiling_manager.cache_tiling
                tiling = self.tiling_work.config_tiling(tiling, self.cache_tiling, self.compute_param)
        else:
            tiling = self._get_tiling_after_cmp(info_dict, new_fused_num)
        tiling = self._no_solution_tiling(tiling)
        tiling = self._gemv_tiling(tiling)
        tiling = self._check_k_full_load(tiling)
        if not tiling:
            args_dict = {"errCode": "E60114", "reason": "tiling is None", "value": "None"}
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        self._check_tiling_value(tiling)
        self.tiling_work.tiling = tiling
        self.status_controller.attach_at_flag = tiling.get("attach_at_flag")
        debug(self.DEBUG_PARAM, tiling, "auto tiling result")

    def _update_flag_after_tiling(self):
        """
        Update variables that depend on tiling.
        over_head_flag: try use nbuffer. Depend on A_overhead_opt_flag and split_k.
        split_k_axis_by_tiling: Enable the bind multi core in k axis. Depend on block_dim_k.
        """
        # split k not support nbuffer now
        self.status_controller.over_head_flag = bool(
            self.tiling_work.tiling.get("A_overhead_opt_flag")) and (not self.status_controller.split_k)
        # the index 3 of block_dim means block_dim_k
        self.status_controller.split_k_axis_by_tiling = self.tiling_work.tiling.get("block_dim")[3] != 1

    def _get_tiling_from_repository(self, current_tiling_type, info_dict, new_fused_num):
        set_tiling_type("repository_tiling")
        tiling_res = get_tiling(info_dict)
        info_dict["fused_double_operand_num"] = new_fused_num
        tiling_res_new = get_tiling(info_dict)
        set_tiling_type(current_tiling_type)
        is_from_repository = not self._get_zero_tiling(tiling_res)
        is_from_repository_new = not self._get_zero_tiling(tiling_res_new)
        return tiling_res, tiling_res_new, is_from_repository, is_from_repository_new

    def _get_tiling_after_cmp(self, info_dict, new_fused_num):
        tiling_res = None
        current_tiling_type = get_tiling_type()
        enable_tune_bank = self._handle_bool_env("ENABLE_TUNE_BANK", True)
        if current_tiling_type == "auto_tiling" and enable_tune_bank:
            (
                tiling_res,
                tiling_res_new,
                is_from_repository,
                is_from_repository_new
            ) = self._get_tiling_from_repository(current_tiling_type, info_dict, new_fused_num)
            if is_from_repository:
                return tiling_res
            # the op gemm not use auto tiling
            if self.container.tensor_map.get("alpha") is None:
                if is_from_repository_new:
                    return tiling_res_new

                return self._auto_tiling(
                    self.container.fuse_num_group[0],
                    self.container.fuse_num_group[1],
                    self.container.fuse_num_group[2])

        info_dict["fused_double_operand_num"] = new_fused_num
        tiling_res = get_tiling(info_dict)
        return tiling_res

    def _check_k_full_load(self, tiling):
        if not self.is_dynamic or self.cache_tiling:
            return tiling

        if tiling.get("AL1_shape") != []:
            al1_k = tiling.get("AL1_shape")[0] // self.block_reduce
            if al1_k != self.dynamic_k:
                tiling["AL1_shape"][1] = 1

        if tiling.get("BL1_shape") != []:
            bl1_k = tiling.get("BL1_shape")[0] // self.block_reduce
            if bl1_k != self.dynamic_k:
                tiling["BL1_shape"][1] = 1
        return tiling

    def _is_int82fp32_nd(self):
        is_int82fp32_nd = ((self.format_info.get("a") == "ND")
                           and (self.format_info.get("b") == "ND")
                           and (self.status_controller.ops_data_flow_mode == "int82fp32"))
        return is_int82fp32_nd

    def _get_tiling_param(self):
        l0a_shape = [get_value(i) for i in self.container.tensor_map.get("a_l0a").shape]
        l0b_shape = [get_value(i) for i in self.container.tensor_map.get("b_l0b").shape]
        debug(self.DEBUG_PARAM, l0a_shape, "l0a_shape")
        debug(self.DEBUG_PARAM, l0b_shape, "l0b_shape")
        l0a_shape, l0b_shape = (
            (l0b_shape, l0a_shape) if self.status_controller.mmad_mode == "gemv"
            else (l0a_shape, l0b_shape))
        a_shape = [
            1,
            l0a_shape[-3],
            l0a_shape[-4],
            self.block_in,
            self.block_reduce
        ]
        if not self.is_dynamic:
            # becasuse A_shape dimension 2 only 16 bits
            while a_shape[2] >= 65536:
                a_shape[2] //= 2
        b_shape = [
            l0b_shape[-4] * self.block_reduce,
            l0b_shape[-3],
            1,
            1,
            self.block_out
        ]

        if len(l0a_shape) == 5:
            a_shape[0] = l0a_shape[0]
        if self.status_controller.ops_data_flow_mode == "int82fp32":
            a_shape[1] = a_shape[1] // 2
            a_shape[1] = a_shape[1] if a_shape[1] != 0 else 1
            a_shape[4] *= 2

        return a_shape, b_shape

    def _get_seed_shape(self):
        self.seed_shape = list(self.dynamic_para.get("m_k_n_shape"))
        debug(self.DEBUG_PARAM, self.seed_shape, "seed_shape:")
        if self.seed_shape and len(self.seed_shape) in (3, 4):
            self.dynamic_m, self.dynamic_k, self.dynamic_n = self.seed_shape[:3]
            if len(self.seed_shape) == 4:
                self.dynamic_batch = self.seed_shape[3]

    def _tiling_l0_process(self):
        tiling = self.tiling_work.tiling
        if self.status_controller.mmad_mode == "gemv":
            a_l0a = self.container.tensor_map.get("b_l0b")
            b_l0b = self.container.tensor_map.get("a_l0a")
        else:
            a_l0a = self.container.tensor_map.get("a_l0a")
            b_l0b = self.container.tensor_map.get("b_l0b")
        if tiling.get("BL0_matrix") != []:
            (
                self.tiling_work.bl0_tiling_kb,
                self.tiling_work.bl0_tiling_nb,
                self.tiling_work.bl0_tiling_n0,
                self.tiling_work.bl0_tiling_k0,
                self.tiling_work.bl0_tiling_batch,
                _
            ) = tiling.get("BL0_matrix")
        else:
            b_l0b_shape = [get_value(i) for i in b_l0b.shape]
            b_l0b_shape = self._get_dynamic_l0b_shape(b_l0b_shape)
            (
                self.tiling_work.bl0_tiling_kb,
                self.tiling_work.bl0_tiling_nb,
                self.tiling_work.bl0_tiling_n0,
                self.tiling_work.bl0_tiling_k0
            ) = b_l0b_shape[-4:]
            # full load only loads 1 batch
            self.tiling_work.bl0_tiling_batch = 1
            self.tiling_work.bl0_tiling_nb = self.tiling_work.tiling.get("CL0_matrix")[0]
            self.tiling_work.bl0_tiling_kb = self.tiling_work.tiling.get("AL0_matrix")[1]
        self.tiling_work.bl0_tiling_k0 = self.block_reduce

        (
            self.tiling_work.al0_tiling_ma,
            self.tiling_work.al0_tiling_ka,
            self.tiling_work.al0_tiling_m0,
            self.tiling_work.al0_tiling_k0,
            self.tiling_work.al0_tiling_batch,
            _
        ) = self.tiling_work.tiling.get("AL0_matrix")
        self.tiling_work.al0_tiling_k0 = self.block_reduce
        (
            self.tiling_work.cl0_tiling_nc,
            self.tiling_work.cl0_tiling_mc,
            self.tiling_work.cl0_tiling_m0,
            self.tiling_work.cl0_tiling_n0,
            self.tiling_work.cl0_tiling_batch,
            _
        ) = self.tiling_work.tiling.get("CL0_matrix")
        c_l0c = self.container.tensor_map.get("c_l0c")
        self._get_l0c_reduce_dims()

        self.tiling_work.al0_tiling_m0 = tbe_platform.CUBE_MKN[a_l0a.dtype]["mac"][0]
        self.tiling_work.al0_tiling_k0 = tbe_platform.CUBE_MKN[a_l0a.dtype]["mac"][1]
        self.tiling_work.bl0_tiling_k0 = tbe_platform.CUBE_MKN[b_l0b.dtype]["mac"][1]
        self.tiling_work.bl0_tiling_n0 = tbe_platform.CUBE_MKN[b_l0b.dtype]["mac"][2]
        self.tiling_work.cl0_tiling_m0 = tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][0]
        self.tiling_work.cl0_tiling_n0 = tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][2]

        # special handle
        if self.is_dynamic:
            # dynamic shape does not support multi batch
            self.tiling_work.al0_tiling_batch = 1
            self.tiling_work.bl0_tiling_batch = 1
            self.tiling_work.cl0_tiling_batch = 1

    def _get_l0c_reduce_dims(self):
        c_l0c = self.container.tensor_map.get("c_l0c")
        self.c_col_k1, self.c_col_k0 = [get_value(ax.dom.extent) for ax in c_l0c.op.reduce_axis]
        if self.is_dynamic:
            if self.status_controller.split_k_axis_by_tiling:
                self.c_col_k0 = self.dynamic_k
            else:
                self.c_col_k1 = self.dynamic_k
        if self.status_controller.split_k_axis_by_tiling:
            self.c_col_k0, self.c_col_k1 = self.c_col_k1, self.c_col_k0

    def _tiling_l1_process(self):
        self._tiling_al1_process()
        self._tiling_bl1_process()

    def _tiling_al1_process(self):
        tiling = self.tiling_work.tiling
        al1_tiling_batch = 1
        if tiling.get("AL1_shape") != [] and (tiling.get("AL1_shape") is not None):
            al1_tiling_k, al1_tiling_m, al1_tiling_batch, _ = tiling.get("AL1_shape")
            al1_tiling_m *= self.tiling_work.al0_tiling_ma
        else:
            if self.is_dynamic:
                al1_tiling_k = self.dynamic_k * self.block_reduce
                al1_ma = self.dynamic_m
            else:
                if self.status_controller.mmad_mode != "gemv":
                    al0_shape = [get_value(i) for i in self.container.tensor_map.get("a_l0a").shape]
                    al1_tiling_k = al0_shape[-3] * al0_shape[-1]
                    al1_ma = al0_shape[-4]
                else:
                    al0_shape = [get_value(i) for i in self.container.tensor_map.get("b_l0b").shape]
                    al1_tiling_k = al0_shape[-4] * al0_shape[-1]
                    al1_ma = al0_shape[-3]

            al1_tiling_m = (al1_ma + tiling.get("block_dim")[2] - 1) // tiling.get("block_dim")[2]
            al1_tiling_k = (al1_tiling_k + tiling.get("block_dim")[3] - 1) // tiling.get("block_dim")[3]
            if self.status_controller.have_batch_a:
                al1_tiling_batch = self.dynamic_batch if self.is_dynamic else al1_tiling_batch

        self.tiling_work.al1_tiling_batch = al1_tiling_batch
        if self.is_dynamic:
            # dynamic shape does not support multi batch
            self.tiling_work.al1_tiling_batch = 1
        self.tiling_work.al1_tiling_k = al1_tiling_k
        self.tiling_work.al1_tiling_m = al1_tiling_m

    def _tiling_bl1_process(self):
        tiling = self.tiling_work.tiling
        bl1_tiling_batch = 1
        if tiling.get("BL1_shape") != [] and (tiling.get("BL1_shape") is not None):
            bl1_tiling_k, bl1_tiling_n, bl1_tiling_batch, _ = tiling.get("BL1_shape")
            bl1_tiling_n *= self.tiling_work.bl0_tiling_nb
        else:
            if self.is_dynamic:
                bl1_tiling_k = self.dynamic_k * self.block_reduce
                bl1_n = self.dynamic_n
            else:
                if self.status_controller.mmad_mode != "gemv":
                    bl0_shape = [get_value(i) for i in self.container.tensor_map.get("b_l0b").shape]
                    bl1_tiling_k = bl0_shape[-4] * bl0_shape[-1]
                    bl1_n = bl0_shape[-3]
                else:
                    bl0_shape = [get_value(i) for i in self.container.tensor_map.get("a_l0a").shape]
                    bl1_tiling_k = bl0_shape[-3] * bl0_shape[-1]
                    bl1_n = bl0_shape[-4]

            bl1_tiling_n = (bl1_n + tiling.get("block_dim")[1] - 1) // tiling.get("block_dim")[1]
            bl1_tiling_k = (bl1_tiling_k + tiling.get("block_dim")[3] - 1) // tiling.get("block_dim")[3]

        self.tiling_work.bl1_tiling_batch = bl1_tiling_batch
        if self.is_dynamic:
            # dynamic shape does not support multi batch
            self.tiling_work.bl1_tiling_batch = 1
        self.tiling_work.bl1_tiling_k = bl1_tiling_k
        self.tiling_work.bl1_tiling_n = bl1_tiling_n

    def _tiling_ub_process(self):
        if self.format_info.get("a") == "ND" or self.status_controller.ops_data_flow_mode == "int82fp32":
            (self.tiling_work.aub_tiling_k, self.tiling_work.aub_tiling_m,
             self.tiling_work.aub_tiling_batch) = self.tiling_work.tiling.get("AUB_shape")[:3]
            if not self.tiling_work.aub_tiling_batch or self.is_dynamic:
                # dynamic shape does not support multi batch
                self.tiling_work.aub_tiling_batch = 1
        else:
            self.tiling_work.aub_tiling_m, self.tiling_work.aub_tiling_k, self.tiling_work.aub_tiling_batch = 0, 0, 0

        if self.format_info.get("b") == "ND" or self.status_controller.ops_data_flow_mode == "int82fp32":
            (self.tiling_work.bub_tiling_k, self.tiling_work.bub_tiling_n,
             self.tiling_work.bub_tiling_batch) = self.tiling_work.tiling.get("BUB_shape")[:3]
            if not self.tiling_work.bub_tiling_batch or self.is_dynamic:
                # dynamic shape does not support multi batch
                self.tiling_work.bub_tiling_batch = 1
        else:
            self.tiling_work.bub_tiling_k, self.tiling_work.bub_tiling_n, self.tiling_work.bub_tiling_batch = 0, 0, 0

    def _get_dynamic_cub_shape(self, cub_shape):
        if self.is_dynamic:
            if self.cache_tiling:
                cub_shape = [self.dynamic_n, self.dynamic_m] + cub_shape[:-2]
            else:
                dynamic_n = int_ceil_div(self.dynamic_n, self.tiling_work.tiling.get("block_dim")[1])
                dynamic_m = int_ceil_div(self.dynamic_m, self.tiling_work.tiling.get("block_dim")[2])
                cub_shape = [dynamic_n, dynamic_m, self.block_in, self.block_out]
            if self.status_controller.have_batch:
                dynamic_batch = int_ceil_div(self.dynamic_batch,
                                                             self.tiling_work.tiling.get("block_dim")[0])
                cub_shape.insert(0, dynamic_batch)
        return cub_shape

    def _cub_process(self):
        debug(self.DEBUG_PARAM, "-------debug info in cub_process-------")
        cub_tiling = self.tiling_work.tiling.get("CUB_matrix")
        cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0, cub_tiling_batch, _ = cub_tiling
        if self.res.dtype == "int8":
            cub_tiling_nc_factor = int_ceil_div(cub_tiling_nc_factor, 2)
        if self.format_info.get("out") == "ND":
            affine_cub = [cub_tiling_mc_factor * cub_tiling_m0, cub_tiling_nc_factor * cub_tiling_n0]
        else:
            affine_cub = [cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0]
        c_ub_tiling_shape = [cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0]
        if self.status_controller.have_batch:
            c_ub_tiling_shape.insert(0, cub_tiling_batch)
            affine_cub.insert(0, cub_tiling_batch)

        c_ub_fract = self.container.tensor_map.get("c_ub_fract")
        c_ub_shape = [get_value(i) for i in c_ub_fract.shape]
        c_ub_shape = self._get_dynamic_cub_shape(c_ub_shape)
        if self.status_controller.split_k_axis_by_tiling:
            c_ub_shape = c_ub_shape[1:]
        self._fix_affine_params_for_atomic_k(affine_cub)

        if self.status_controller.mmad_mode in ("gemv", "gevm"):
            c_ub_tiling_shape[-2] = 1
        if self.status_controller.attach_at_flag:
            cub_attach_flag = self.status_controller.attach_at_flag.get("cub_attach_flag")
            status = self.status_dict.get(cub_attach_flag)
        else:
            status = Compare.compare(c_ub_tiling_shape, c_ub_shape)
        debug(self.DEBUG_PARAM, [c_ub_tiling_shape, c_ub_shape], "c_ub_tiling_shape with c_ub_shape")
        debug(self.DEBUG_PARAM, [affine_cub, self.root_tensor.shape], "affine_cub with root_tensor's shape")
        self._do_attach_cub(status, c_ub_fract, affine_cub)
        debug(self.DEBUG_PARAM, "-------debug info in cub_process end-------")
        return c_ub_tiling_shape

    def _do_attach_cub(self, status, c_ub_fract, affine_cub):
        if self.container.fusion_tensor_cub:
            self.container.tensors_in_cub += self.container.fusion_tensor_cub

        if self.is_dynamic and not self.cache_tiling:
            status = Compare.LESS_EQ

        if status == Compare.EQUAL:
            pass
        elif status == Compare.LESS_EQ:
            self._do_attach_cub_less_eq(c_ub_fract, affine_cub)
        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "c_ub attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    def _do_attach_cub_less_eq(self, c_ub_fract, affine_cub):
        """ sub func of _do_attach_cub
        """
        if self.status_controller.quant_fusion or self.status_controller.requant_fusion:
            affine_cub[-1] *= 2
            debug(self.DEBUG_PARAM, affine_cub, "affine_cub in quant_fusion and requant_fusion")
        self.sch_agent.attach_at(c_ub_fract, self.root_tensor, affine_shape=affine_cub,
                                 factor_shape=self.tiling_work.factor_shape.get("cub"),
                                 ceil_mode_dict=self.tiling_work.get_split_param(
                                     self.cache_tiling_manager))
        self.status_controller.c_ub_attach_status = "c_gm"

        same_attach_cub = self.container.tensors_in_cub
        for tensor in same_attach_cub:
            if tensor in (c_ub_fract, self.root_tensor):
                continue
            self.sch_agent.same_attach(tensor, c_ub_fract)
        if self.container.double_out_tensor:
            self.sch_agent.same_attach(self.container.double_out_tensor[0], c_ub_fract)

    def _get_dynamic_l0c_shape(self, l0c_shape, have_batch):
        if self.is_dynamic:
            if self.cache_tiling:
                l0c_shape = [self.dynamic_n, self.dynamic_m] + l0c_shape[:-2]
            else:
                dynamic_n = int_ceil_div(self.dynamic_n, self.tiling_work.tiling.get("block_dim")[1])
                dynamic_m = int_ceil_div(self.dynamic_m, self.tiling_work.tiling.get("block_dim")[2])
                l0c_shape = [dynamic_n, dynamic_m, self.block_in, self.block_out]
            if have_batch:
                dynamic_batch = int_ceil_div(self.dynamic_batch,
                                             self.tiling_work.tiling.get("block_dim")[0])
                l0c_shape.insert(0, dynamic_batch)

        return l0c_shape

    def _cl0_process(self, c_ub_tiling_shape):
        debug(self.DEBUG_PARAM, "-------debug info in cl0_process-------")

        cl0_tiling_nc, cl0_tiling_mc = self.tiling_work.cl0_tiling_nc, self.tiling_work.cl0_tiling_mc
        cl0_tiling_m0, cl0_tiling_n0 = self.tiling_work.cl0_tiling_m0, self.tiling_work.cl0_tiling_n0

        if self.format_info.get("out") == "ND":
            affine_l0c = [cl0_tiling_mc * cl0_tiling_m0, cl0_tiling_nc * cl0_tiling_n0]
        else:
            affine_l0c = [cl0_tiling_nc, cl0_tiling_mc, cl0_tiling_m0, cl0_tiling_n0]
        cl0_tiling_shape = [cl0_tiling_nc, cl0_tiling_mc, cl0_tiling_m0, cl0_tiling_n0]

        if self.status_controller.have_batch:
            affine_l0c.insert(0, self.tiling_work.cl0_tiling_batch)
            cl0_tiling_shape.insert(0, self.tiling_work.cl0_tiling_batch)

        c_l0c = self.container.tensor_map.get("c_l0c")
        c_l0c_shape = [get_value(i) for i in c_l0c.shape]
        c_l0c_shape = self._get_dynamic_l0c_shape(c_l0c_shape, self.status_controller.have_batch)
        if self.status_controller.split_k_axis_by_tiling:
            c_l0c_shape = c_l0c_shape[1:]
        self._fix_affine_params_for_atomic_k(affine_l0c)

        if self.status_controller.mmad_mode in ("gemv", "gevm"):
            # add for dsl_mat_d_elm-eltwise-ut8_NZ_0054
            c_ub_tiling_shape[-2] = 16
        if self.status_controller.attach_at_flag:
            cl0_attach_flag = self.status_controller.attach_at_flag.get("cl0_attach_flag")
            status_ori = self.status_ori_dict.get(cl0_attach_flag)
            status = self.status_dict.get(cl0_attach_flag)
        else:
            status_ori = Compare.compare(cl0_tiling_shape, c_l0c_shape)
            status = Compare.compare(cl0_tiling_shape, c_ub_tiling_shape)
        debug(self.DEBUG_PARAM, [cl0_tiling_shape, c_l0c_shape], "cl0_tiling_shape with c_l0c_shape")
        debug(self.DEBUG_PARAM, [cl0_tiling_shape, c_ub_tiling_shape], "cl0_tiling_shape with c_ub_tiling_shape")
        debug(self.DEBUG_PARAM, [affine_l0c, self.root_tensor.shape], "affine_l0c with root_tensor.shape")
        self._do_attach_cl0(status_ori, status, c_l0c, affine_l0c)
        debug(self.DEBUG_PARAM, "-------debug info in cl0_process end -------")

    def _do_attach_cl0(self, status_ori, status, c_l0c, affine_l0c):
        self.status_controller.c_l0c_attach_status = "full_load"

        if self.is_dynamic and not self.cache_tiling:
            status_ori = Compare.LESS_EQ

        if status_ori == Compare.MISC:
            args_dict = {
                "errCode": "E60114",
                "reason": "cl0 attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        elif status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            self.status_controller.c_l0c_attach_status = self.status_controller.c_ub_attach_status
            c_ub_fract = self.container.tensor_map.get("c_ub_fract")
            self.sch_agent.same_attach(c_l0c, c_ub_fract)
        elif status == Compare.GREATE_EQ:
            if self.status_controller.quant_fusion or self.status_controller.requant_fusion:
                affine_l0c[-1] *= 2
                affine_l0c[-4] //= 2

            self.sch_agent.attach_at(c_l0c, self.root_tensor, affine_shape=affine_l0c,
                                     factor_shape=self.tiling_work.factor_shape.get("cl0"),
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
            self.status_controller.c_l0c_attach_status = "c_gm"
        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "tensor_c_l0c attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

        for tensor in self.container.tensors_in_l0c:
            if tensor == c_l0c:
                continue
            self.sch_agent.same_attach(tensor, c_l0c)

    def _get_dynamic_l0a_shape(self, l0a_shape):
        if self.is_dynamic:
            if self.cache_tiling:
                l0a_shape = [self.dynamic_m, self.dynamic_k] + l0a_shape[:-2]
            else:
                dynamic_m = int_ceil_div(self.dynamic_m, self.tiling_work.tiling.get("block_dim")[2])
                l0a_shape = [dynamic_m, self.dynamic_k, self.block_in, self.block_reduce]
            if self.status_controller.have_batch_a:
                dynamic_batch = int_ceil_div(self.dynamic_batch, self.tiling_work.tiling.get("block_dim")[0])
                l0a_shape.insert(0, dynamic_batch)

        return l0a_shape

    def _get_affine_m_full_load_dynamic(self):
        # now only support fractal
        tiling_m, _, _, _ = self.container.tensor_map.get("a_l0a").shape
        tiling_m = int_ceil_div(tiling_m, self.tiling_work.tiling.get("block_dim")[2])
        return tiling_m

    def _get_affine_n_full_load_dynamic(self):
        # now only support fractal
        _, tiling_n, _, _ = self.container.tensor_map.get("b_l0b").shape
        tiling_n = int_ceil_div(tiling_n, self.tiling_work.tiling.get("block_dim")[1])
        return tiling_n

    def _fix_affine_params_for_atomic_k(self, affine_param):
        if self.status_controller.split_k_axis_by_tiling:
            affine_param.insert(0, 1)

    def _get_affine_k_full_load_dynamic(self):
        _, tiling_k, _, _ = self.container.tensor_map.get("a_l0a").shape
        return tiling_k

    def _al0_process(self):
        debug(self.DEBUG_PARAM, "-------debug info in al0_process-------")
        if (self.tiling_work.tiling.get("AL0_matrix") == [] and
            not self.is_dynamic and self.status_controller.have_batch_a):
            return
        l0a2l0c_affine_shape = [
            None,
            self.tiling_work.al0_tiling_ma,
            None,
            self.tiling_work.cl0_tiling_n0,
            self.tiling_work.al0_tiling_ka,
            self.tiling_work.al0_tiling_k0
        ]
        tiling_ori_l0a = [
            self.tiling_work.al0_tiling_ma, self.tiling_work.al0_tiling_ka, self.tiling_work.al0_tiling_m0,
            self.tiling_work.al0_tiling_k0
        ]
        a_l0a = self.container.tensor_map.get("a_l0a")
        if self.status_controller.mmad_mode == "gemv":
            a_l0a = self.container.tensor_map.get("b_l0b")
        else:
            a_l0a = self.container.tensor_map.get("a_l0a")
        a_l0a_shape = [get_value(i) for i in a_l0a.shape]
        a_l0a_shape = self._get_dynamic_l0a_shape(a_l0a_shape)
        al0_tiling_shape = [
            self.tiling_work.al0_tiling_ma, self.tiling_work.al0_tiling_m0, self.tiling_work.al0_tiling_ka,
            self.tiling_work.al0_tiling_k0
        ]
        cl0_tiling_shape = [
            self.tiling_work.cl0_tiling_mc, self.tiling_work.cl0_tiling_m0, self.c_col_k1, self.c_col_k0
        ]
        if self.format_info.get("out") == "ND":
            l0a2out_affine_shape = [self.tiling_work.al0_tiling_ma * self.tiling_work.al0_tiling_m0, None]
        else:
            l0a2out_affine_shape = [None, self.tiling_work.al0_tiling_ma, self.tiling_work.al0_tiling_m0, None]
        if self.status_controller.have_batch_a:
            l0a2l0c_affine_shape.insert(0, self.tiling_work.al0_tiling_batch)
            tiling_ori_l0a.insert(0, self.tiling_work.al0_tiling_batch)
            al0_tiling_shape.insert(0, self.tiling_work.al0_tiling_batch)
            l0a2out_affine_shape.insert(0, self.tiling_work.al0_tiling_batch)
            cl0_tiling_shape.insert(0, self.tiling_work.cl0_tiling_batch)
        elif self.status_controller.have_batch:
            l0a2l0c_affine_shape.insert(0, None)
            l0a2out_affine_shape.insert(0, None)

        if self.status_controller.mmad_mode in ("gevm", "gemv") and not self.status_controller.only_use_gevm_gemv_flow:
            tiling_ori_l0a[-2] = 1
        self._fix_affine_params_for_atomic_k(l0a2out_affine_shape)
        self._fix_affine_params_for_atomic_k(l0a2l0c_affine_shape)

        if self.status_controller.attach_at_flag:
            l0a_attach_flag = self.status_controller.attach_at_flag.get("al0_attach_flag")
            status_ori = self.status_ori_dict.get(l0a_attach_flag)
            status = self.status_dict.get(l0a_attach_flag)
        else:
            status_ori = Compare.compare(tiling_ori_l0a, a_l0a_shape)
            status = Compare.compare(al0_tiling_shape, cl0_tiling_shape)
        debug(self.DEBUG_PARAM, [tiling_ori_l0a, a_l0a_shape], "tiling_ori_l0a with a_l0a_shape")
        debug(self.DEBUG_PARAM, [al0_tiling_shape, cl0_tiling_shape], "al0_tiling_shape with cl0_tiling_shape")
        all_status = (status_ori, status)
        self._do_attach_l0a(all_status, a_l0a, l0a2l0c_affine_shape, l0a2out_affine_shape)
        debug(self.DEBUG_PARAM, "-------debug info in al0_process end-------")

    def _do_attach_l0a(self, all_status, a_l0a, l0a2l0c_affine_shape, l0a2out_affine_shape):
        status_ori, status = all_status
        if self.is_dynamic and not self.cache_tiling:
            status_ori = Compare.LESS_EQ
            status = Compare.LESS_EQ

        if status_ori == Compare.MISC:
            args_dict = {
                "errCode": "E60114",
                "reason": "a_l0a attach error.",
                "value": "compare status = {}".format(status_ori)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        elif status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            self.sch_agent.same_attach(a_l0a, self.container.tensor_map.get("c_l0c"))
            self.status_controller.al0_attach_status = self.status_controller.c_l0c_attach_status
        elif status == Compare.LESS_EQ:
            self.sch_agent.attach_at(
                a_l0a, self.container.tensor_map.get("c_l0c"), affine_shape=l0a2l0c_affine_shape,
                factor_shape=self.tiling_work.factor_shape.get("al0"),
                ceil_mode_dict=self.tiling_work.get_split_param(
                    self.cache_tiling_manager)
            )
            self.status_controller.al0_attach_status = "c_l0c"
        elif status == Compare.GREATE_EQ:
            self.sch_agent.attach_at(a_l0a, self.root_tensor, affine_shape=l0a2out_affine_shape,
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
            self.status_controller.al0_attach_status = "c_gm"
        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "l0a attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    def _get_dynamic_l0b_shape(self, l0b_shape):
        if self.is_dynamic:
            if self.cache_tiling:
                l0b_shape = [self.dynamic_k, self.dynamic_n, self.block_out, self.block_reduce]
            else:
                dynamic_n = int_ceil_div(self.dynamic_n, self.tiling_work.tiling.get("block_dim")[1])
                l0b_shape = [self.dynamic_k, dynamic_n, self.block_out, self.block_reduce]
            if self.status_controller.have_batch_b:
                dynamic_batch = int_ceil_div(self.dynamic_batch, self.tiling_work.tiling.get("block_dim")[0])
                l0b_shape.insert(0, dynamic_batch)
        return l0b_shape

    def _bl0_process(self):
        debug(self.DEBUG_PARAM, "-------debug info in bl0_process-------")
        l0b2l0c_affine_shape = [
            self.tiling_work.bl0_tiling_nb,
            None,
            None,
            self.tiling_work.bl0_tiling_n0,
            self.tiling_work.bl0_tiling_kb,
            self.tiling_work.bl0_tiling_k0
        ]
        if self.format_info.get("out") == "ND":
            l0b2out_affine_shape = [None, self.tiling_work.bl0_tiling_nb * self.tiling_work.bl0_tiling_n0]
        else:
            l0b2out_affine_shape = [self.tiling_work.bl0_tiling_nb, None, None, self.tiling_work.bl0_tiling_n0]

        tiling_ori_l0b = [
            self.tiling_work.bl0_tiling_kb, self.tiling_work.bl0_tiling_nb, self.tiling_work.bl0_tiling_n0,
            self.tiling_work.bl0_tiling_k0
        ]
        bl0_tiling_shape = [
            self.tiling_work.bl0_tiling_nb, self.tiling_work.bl0_tiling_n0, self.tiling_work.bl0_tiling_kb,
            self.tiling_work.bl0_tiling_k0
        ]

        cl0_tiling_shape = [
            self.tiling_work.cl0_tiling_nc, self.tiling_work.cl0_tiling_n0, self.c_col_k1, self.c_col_k0
        ]
        if self.status_controller.mmad_mode == "gemv":
            b_l0b = self.container.tensor_map.get("a_l0a")
        else:
            b_l0b = self.container.tensor_map.get("b_l0b")
        b_l0b_shape = [get_value(i) for i in b_l0b.shape]
        b_l0b_shape = self._get_dynamic_l0b_shape(b_l0b_shape)
        if self.status_controller.have_batch_b:
            l0b2l0c_affine_shape.insert(0, self.tiling_work.bl0_tiling_batch)
            tiling_ori_l0b.insert(0, self.tiling_work.bl0_tiling_batch)
            l0b2out_affine_shape.insert(0, self.tiling_work.bl0_tiling_batch)
            bl0_tiling_shape.insert(0, self.tiling_work.bl0_tiling_batch)
            cl0_tiling_shape.insert(0, self.tiling_work.cl0_tiling_batch)
        elif self.status_controller.have_batch:
            l0b2l0c_affine_shape.insert(0, None)
            l0b2out_affine_shape.insert(0, None)
        self._fix_affine_params_for_atomic_k(l0b2out_affine_shape)
        self._fix_affine_params_for_atomic_k(l0b2l0c_affine_shape)

        debug(self.DEBUG_PARAM, [tiling_ori_l0b, b_l0b_shape], "tiling_ori_l0b, b_l0b_shape")
        debug(self.DEBUG_PARAM, [bl0_tiling_shape, cl0_tiling_shape], "bl0_tiling_shape, cl0_tiling_shape")
        if self.status_controller.attach_at_flag:
            l0b_attach_flag = self.status_controller.attach_at_flag.get("bl0_attach_flag")
            status_ori = self.status_ori_dict.get(l0b_attach_flag)
            status = self.status_dict.get(l0b_attach_flag)
        else:
            status_ori = Compare.compare(tiling_ori_l0b, b_l0b_shape)
            status = Compare.compare(bl0_tiling_shape, cl0_tiling_shape)
        self._do_attach_bl0([status_ori, status], b_l0b, [l0b2l0c_affine_shape, l0b2out_affine_shape])
        debug(self.DEBUG_PARAM, "-------debug info in bl0_process end-------")

    def _do_attach_bl0(self, affine_status, b_l0b, affine_shapes):
        status_ori, status = affine_status
        l0b2l0c_affine_shape, l0b2out_affine_shape = affine_shapes
        if self.is_dynamic and not self.cache_tiling:
            status_ori = Compare.LESS_EQ
            status = Compare.LESS_EQ

        if status_ori == Compare.MISC:
            args_dict = {
                "errCode": "E60114",
                "reason": "b_l0b attach error.",
                "value": "compare status = {}".format(status_ori)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        if status_ori == Compare.EQUAL and (not self.bind_core_when_full_load_bl1):
            pass
        elif status == Compare.EQUAL:
            self.sch_agent.same_attach(b_l0b, self.container.tensor_map.get("c_l0c"))
        elif status == Compare.LESS_EQ:
            self.sch_agent.attach_at(b_l0b, self.container.tensor_map.get("c_l0c"), affine_shape=l0b2l0c_affine_shape,
                                     factor_shape=self.tiling_work.factor_shape.get("bl0"),
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
        elif status == Compare.GREATE_EQ:
            l0b2out_affine_shape = self._fix_affine_out_int8(b_l0b.dtype, l0b2out_affine_shape)
            self.sch_agent.attach_at(b_l0b, self.root_tensor, affine_shape=l0b2out_affine_shape,
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "l0b attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    def _al1_process(self):
        self.status_controller.al1_attach_status = "full_load"
        not_need_process = self.tiling_work.tiling.get("AL1_shape") in (None, []) and (not self.is_dynamic)
        not_need_process = not_need_process and (not self.status_controller.have_batch_a)
        if not_need_process:
            return
        debug(self.DEBUG_PARAM, "-------debug info in al1_process-------")
        al1_tiling_m, al1_tiling_k = self.tiling_work.al1_tiling_m, self.tiling_work.al1_tiling_k
        al0_tiling_m0 = self.tiling_work.al0_tiling_m0
        al0_tiling_k0 = self.tiling_work.al0_tiling_k0
        cl0_tiling_mc = self.tiling_work.cl0_tiling_mc

        cl0_tiling_m0, cl0_tiling_n0 = self.tiling_work.cl0_tiling_m0, self.tiling_work.cl0_tiling_n0

        l1_ma = al1_tiling_m
        l1_ka = al1_tiling_k // al0_tiling_k0 if self.cache_tiling else int_ceil_div(al1_tiling_k, al0_tiling_k0)

        a_l1 = (
            self.container.tensor_map.get("b_l1") if self.status_controller.mmad_mode == "gemv"
            else self.container.tensor_map.get("a_l1"))
        tiling_ori_al1 = [l1_ma, l1_ka]

        a_l0a = (
            self.container.tensor_map.get("b_l0b") if self.status_controller.mmad_mode == "gemv"
            else self.container.tensor_map.get("a_l0a"))
        a_l0a_shape = [get_value(i) for i in a_l0a.shape]
        m_shape = a_l0a_shape[-4]
        k_shape = a_l0a_shape[-3]
        if self.is_dynamic:
            m_shape = self.dynamic_m
            k_shape = self.dynamic_k
        al1_shape = [m_shape, k_shape]
        al1_shape[0] = int_ceil_div(al1_shape[0], self.tiling_work.tiling.get("block_dim")[2])

        l1a2l0c_affine_shape = [
            None,
            l1_ma,
            None,
            cl0_tiling_n0,
            l1_ka,
            al0_tiling_k0
        ]

        al1_tiling_shape = [l1_ma, al0_tiling_m0, l1_ka, al0_tiling_k0]
        # add bl1_tiling_n in order to out n_inner axis down
        l1a2out_affine_shape = [self.tiling_work.bl1_tiling_n, l1_ma, al0_tiling_m0, None]
        if self.format_info.get("out") == "ND":
            al1_n_shape = None
            if self.cache_tiling:
                al1_n_shape = self.cache_tiling.get('n_single_core') * self.tiling_work.bl1_tiling_n * cl0_tiling_n0
            l1a2out_affine_shape = [l1_ma * al0_tiling_m0, al1_n_shape]

        tiling_ori_al1[-2] = 1 if self.status_controller.mmad_mode in ("gevm", "gemv") else tiling_ori_al1[-2]
        cl0_tiling_shape = [cl0_tiling_mc, cl0_tiling_m0, self.c_col_k1, self.c_col_k0]
        if self.status_controller.have_batch_a:
            al1_ori_batch = self.dynamic_batch if self.is_dynamic else a_l1.shape[0].value
            al1_shape.insert(0, al1_ori_batch)
            al1_shape[0] = (al1_shape[0] + self.tiling_work.tiling.get("block_dim")[0] -
                            1) // self.tiling_work.tiling.get("block_dim")[0]
            tiling_ori_al1.insert(0, self.tiling_work.al1_tiling_batch)
            l1a2l0c_affine_shape.insert(0, self.tiling_work.al1_tiling_batch)
            l1a2out_affine_shape.insert(0, self.tiling_work.al1_tiling_batch)
            al1_tiling_shape.insert(0, self.tiling_work.al1_tiling_batch)
            cl0_tiling_shape.insert(0, self.tiling_work.cl0_tiling_batch)
        elif self.status_controller.have_batch:
            l1a2l0c_affine_shape.insert(0, None)
            l1a2out_affine_shape.insert(0, None)
        self._fix_affine_params_for_atomic_k(l1a2l0c_affine_shape)
        self._fix_affine_params_for_atomic_k(l1a2out_affine_shape)

        status_ori, status = self._al1_attach_modus(al1_tiling_shape, cl0_tiling_shape,
                                                    tiling_ori_al1, al1_shape)
        debug(self.DEBUG_PARAM, [al1_tiling_shape, cl0_tiling_shape],
              "al1_tiling_shape with cl0_tiling_shape")
        debug(self.DEBUG_PARAM, [tiling_ori_al1, al1_shape], "tiling_ori_al1 with al1_shape")
        self._do_attach_al1(status_ori, status, a_l1, l1a2l0c_affine_shape, l1a2out_affine_shape)
        debug(self.DEBUG_PARAM, "-------debug info in al1_process end-------")

    def _al1_attach_modus(self, al1_tiling_shape, cl0_tiling_shape, tiling_ori_al1, al1_shape):
        if self.status_controller.attach_at_flag:
            al1_attach_flag = self.status_controller.attach_at_flag.get("al1_attach_flag")
            status_ori = self.status_ori_dict.get(al1_attach_flag)
            status = self.status_dict.get(al1_attach_flag)
        else:
            status = Compare.compare(al1_tiling_shape, cl0_tiling_shape)
            status_ori = Compare.compare(tiling_ori_al1, al1_shape)
            if self.status_controller.batch_broadcast_change_attach and self.format_info.get("a") == "ND":
                status_ori = Compare.LESS_EQ

        return status_ori, status

    def _do_attach_al1(self, status_ori, status, a_l1, l1a2l0c_affine_shape, l1a2out_affine_shape):
        if self.is_dynamic and not self.cache_tiling and not self.tiling_work.tiling.get("attach_same_to_static"):
            status_ori = Compare.LESS_EQ
            status = Compare.LESS_EQ

        if status_ori == Compare.MISC:
            args_dict = {
                "errCode": "E60114",
                "reason": "a_l1 attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        elif status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            self.status_controller.al1_attach_status = self.status_controller.c_l0c_attach_status
            # reduce duplicate loading, probably not only in cachetiling scene
            if self.cache_tiling and self.format_info.get("a") == "ND":
                self.sch_agent.attach_at(a_l1, self.root_tensor, affine_shape=l1a2out_affine_shape,
                                         factor_shape=self.tiling_work.factor_shape.get("al12ddr"),
                                         ceil_mode_dict=self.tiling_work.get_split_param(
                                             self.cache_tiling_manager))
            else:
                self.sch_agent.same_attach(a_l1, self.container.tensor_map.get("c_l0c"))
        elif status == Compare.LESS_EQ:
            self.status_controller.al1_attach_status = "c_l0c"
            self.sch_agent.attach_at(a_l1, self.container.tensor_map.get("c_l0c"), affine_shape=l1a2l0c_affine_shape,
                                     factor_shape=self.tiling_work.factor_shape.get("al12cl0"),
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
        elif status == Compare.GREATE_EQ:
            self.status_controller.al1_attach_status = "c_gm"
            l1a2out_affine_shape = self._fix_affine_out_int8(a_l1.dtype, l1a2out_affine_shape)
            self.sch_agent.attach_at(a_l1, self.root_tensor, affine_shape=l1a2out_affine_shape,
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "a_l1 attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    def _bl1_process(self):
        self._temporarily_enable_fullload_to_bind_multi_core_when_exceed_space()

        self.status_controller.bl1_attach_status = "full_load"
        not_need_bl1_process = self.tiling_work.tiling.get("BL1_shape") in (None, []) and (not self.is_dynamic)
        not_need_bl1_process = not_need_bl1_process and (
            not self.status_controller.have_batch_b) and (not self.bind_core_when_full_load_bl1)
        if not_need_bl1_process:
            return
        debug(self.DEBUG_PARAM, "-------debug info in bl1_process-------")
        bl1_tiling_n, bl1_tiling_k = self.tiling_work.bl1_tiling_n, self.tiling_work.bl1_tiling_k
        bl0_tiling_n0, bl0_tiling_k0 = self.tiling_work.bl0_tiling_n0, self.tiling_work.bl0_tiling_k0
        cl0_tiling_nc, cl0_tiling_n0 = self.tiling_work.cl0_tiling_nc, self.tiling_work.cl0_tiling_n0

        l1_nb = bl1_tiling_n
        l1_kb = bl1_tiling_k // bl0_tiling_k0 if self.cache_tiling else int_ceil_div(bl1_tiling_k, bl0_tiling_k0)

        l1b2l0c_affine_shape = [
            l1_nb,
            None,
            None,
            bl0_tiling_n0,
            l1_kb,
            bl0_tiling_k0
        ]

        bl1_tiling_shape = [l1_nb, bl0_tiling_n0, l1_kb, bl0_tiling_k0]
        l1b2out_affine_shape = self._get_l1b2out_affine_shape()
        cl0_tiling_shape = [cl0_tiling_nc, cl0_tiling_n0, self.c_col_k1, self.c_col_k0]

        b_l1 = (
            self.container.tensor_map.get("a_l1") if self.status_controller.mmad_mode == "gemv"
            else self.container.tensor_map.get("b_l1"))
        tiling_ori_bl1 = [l1_kb, l1_nb]
        if self.is_dynamic:
            n_shape = self.dynamic_n
            k_shape = self.dynamic_k
        else:
            b_l0b = (
                self.container.tensor_map.get("a_l0a") if self.status_controller.mmad_mode == "gemv"
                else self.container.tensor_map.get("b_l0b"))
            b_l0b_shape = [get_value(i) for i in b_l0b.shape]
            n_shape = b_l0b_shape[-3]
            k_shape = b_l0b_shape[-4]

        bl1_shape = [k_shape, n_shape]
        if not self.bind_core_when_full_load_bl1:
            bl1_shape[1] = int_ceil_div(bl1_shape[1], self.tiling_work.tiling.get("block_dim")[1])

        if self.status_controller.have_batch_b:
            bl1_ori_batch = self.dynamic_batch if self.is_dynamic else b_l1.shape[0].value
            bl1_shape.insert(0, bl1_ori_batch)
            bl1_shape[0] = int_ceil_div(bl1_shape[0], self.tiling_work.tiling.get("block_dim")[0])
            l1b2l0c_affine_shape.insert(0, self.tiling_work.bl1_tiling_batch)
            tiling_ori_bl1.insert(0, self.tiling_work.bl1_tiling_batch)
            bl1_tiling_shape.insert(0, self.tiling_work.bl1_tiling_batch)
            l1b2out_affine_shape.insert(0, self.tiling_work.bl1_tiling_batch)
            cl0_tiling_shape.insert(0, self.tiling_work.cl0_tiling_batch)
        elif self.status_controller.have_batch:
            l1b2l0c_affine_shape.insert(0, None)
            l1b2out_affine_shape.insert(0, None)
        self._fix_affine_params_for_atomic_k(l1b2l0c_affine_shape)
        self._fix_affine_params_for_atomic_k(l1b2out_affine_shape)

        debug(self.DEBUG_PARAM, [bl1_tiling_shape, cl0_tiling_shape], "bl1_tiling_shape with cl0_tiling_shape")
        debug(self.DEBUG_PARAM, [tiling_ori_bl1, bl1_shape], "tiling_ori_bl1 with bl1_shape")
        status_ori, status = self._bl1_attach_modus(bl1_tiling_shape, cl0_tiling_shape,
                             tiling_ori_bl1, bl1_shape)
        self._do_attach_bl1(status_ori, status, b_l1, l1b2l0c_affine_shape, l1b2out_affine_shape)
        debug(self.DEBUG_PARAM, "-------debug info in bl1_process end-------")

    def _get_l1b2out_affine_shape(self):
        """
        get l1b2out_affine_shape
        """
        bl1_m_shape = None
        l1b2out_affine_shape = [bl1_m_shape, self.tiling_work.bl1_tiling_n * self.tiling_work.bl0_tiling_n0]
        if self.format_info.get("out") != "ND":
            l1b2out_affine_shape = [self.tiling_work.bl1_tiling_n, None, None, self.tiling_work.bl0_tiling_n0]
            template_bl1_full_load = (
                self.cache_tiling and self.status_controller.attach_at_flag.get("bl1_attach_flag") == 0)
            if template_bl1_full_load:
                l1b2out_affine_shape[1] = self.tiling_work.al1_tiling_m
        return l1b2out_affine_shape

    def _bl1_attach_modus(self, bl1_tiling_shape, cl0_tiling_shape, tiling_ori_bl1, bl1_shape):
        if self.status_controller.attach_at_flag:
            bl1_attach_flag = self.status_controller.attach_at_flag.get("bl1_attach_flag")
            status_ori = self.status_ori_dict.get(bl1_attach_flag)
            status = self.status_dict.get(bl1_attach_flag)
        else:
            status = Compare.compare(bl1_tiling_shape, cl0_tiling_shape)
            status_ori = Compare.compare(tiling_ori_bl1, bl1_shape)
            if self.status_controller.batch_broadcast_change_attach and self.format_info.get("b") == "ND":
                status_ori = Compare.LESS_EQ

        return status_ori, status

    def _do_attach_bl1(self, status_ori, status, b_l1, l1b2l0c_affine_shape, l1b2out_affine_shape):
        if self.is_dynamic and not self.cache_tiling and not self.tiling_work.tiling.get("attach_same_to_static"):
            status_ori = Compare.LESS_EQ
            status = Compare.LESS_EQ

        if status_ori == Compare.MISC:
            args_dict = {
                "errCode": "E60114",
                "reason": "b_l1 attach error.",
                "value": "compare status = {}".format(status_ori)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        elif status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            self.status_controller.bl1_attach_status = self.status_controller.c_l0c_attach_status
            if self.cache_tiling:
                self.sch_agent.attach_at(b_l1, self.root_tensor, affine_shape=l1b2out_affine_shape,
                                         factor_shape=self.tiling_work.factor_shape.get("bl12ddr"),
                                         ceil_mode_dict=self.tiling_work.get_split_param(
                                             self.cache_tiling_manager))
            else:
                self.status_controller.bl1_attach_status = self.status_controller.c_l0c_attach_status
                self.sch_agent.same_attach(b_l1, self.container.tensor_map.get("c_l0c"))
        elif status == Compare.LESS_EQ:
            self.status_controller.bl1_attach_status = "c_l0c"
            self.sch_agent.attach_at(b_l1, self.container.tensor_map.get("c_l0c"), affine_shape=l1b2l0c_affine_shape,
                                     factor_shape = self.tiling_work.factor_shape.get("bl12cl0"),
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
        elif status == Compare.GREATE_EQ:
            self.status_controller.bl1_attach_status = "c_gm"
            l1b2out_affine_shape = self._fix_affine_out_int8(b_l1.dtype, l1b2out_affine_shape)
            self.sch_agent.attach_at(b_l1, self.root_tensor, affine_shape=l1b2out_affine_shape,
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "b_l1 attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    def _fix_affine_out_int8(self, tensor_dtype, affine_shape):
        """
        if input and output both int8, tiling_n only half
        """
        if self.res.dtype == "int8" and tensor_dtype in ("int8", "uint8"):
            n_dim_index = -1 if self.format_info.get("out") == "ND" else -4
            n_factor = affine_shape[n_dim_index] // 2
            affine_shape[n_dim_index] = 1 if n_factor == 0 else n_factor
        return affine_shape

    def _renew_aub_m(self, a_ub_ori_shape):
        index_offset = 1 if self.status_controller.have_batch_a else 0
        if self.format_info.get("a") == "ND":
            a_ub_ori_shape[index_offset] = int_ceil_div(a_ub_ori_shape[index_offset],
                self.tiling_work.tiling.get("block_dim")[2] * self.block_in) * self.block_in
        else:
            if self.status_controller.transpose_a:
                a_ub_ori_shape[index_offset] = int_ceil_div(a_ub_ori_shape[index_offset],
                    self.tiling_work.tiling.get("block_dim")[2])
            else:
                a_ub_ori_shape[1 + index_offset] = int_ceil_div(a_ub_ori_shape[1 + index_offset],
                    self.tiling_work.tiling.get("block_dim")[2])
        if self.status_controller.have_batch_a and not self.is_dynamic:
            a_ub_ori_shape[0] = int_ceil_div(a_ub_ori_shape[0],
                                                             self.tiling_work.tiling.get("block_dim")[0])

    def _get_dynamic_aub_shape(self, aub_shape, aub_tiling):
        if self.is_dynamic:
            # m1, k1*k0
            aub_shape = [self.dynamic_m, self.dynamic_k * self.block_reduce]
            aub_tiling = [self.tiling_work.aub_tiling_m, self.tiling_work.aub_tiling_k]
            if self.status_controller.have_batch_a:
                # tiling add batch at the same palce of static mode
                aub_shape.insert(0, self.dynamic_batch)

        return aub_shape, aub_tiling

    def _aub_process(self):

        debug(self.DEBUG_PARAM, "-------debug info in aub_process-------")
        a_ub = self.container.tensor_map.get("a_ub")
        if a_ub in (None, []):
            return
        transpose_a = self.status_controller.transpose_a
        aub_tiling_k, aub_tiling_m = self.tiling_work.aub_tiling_k, self.tiling_work.aub_tiling_m
        aub_tiling_k0 = self.block_reduce
        aub_tiling_m0 = 1 if self.status_controller.mmad_mode == "gevm" else self.block_in
        l1_ma, al1_tiling_k = self.tiling_work.al1_tiling_m, self.tiling_work.al1_tiling_k
        al0_tiling_m0, al0_tiling_k0 = self.tiling_work.al0_tiling_m0, self.tiling_work.al0_tiling_k0
        cl0_tiling_mc = self.tiling_work.cl0_tiling_mc
        cl0_tiling_m0, cl0_tiling_n0 = self.tiling_work.cl0_tiling_m0, self.tiling_work.cl0_tiling_n0
        l1_ka = (al1_tiling_k + al0_tiling_k0 - 1) // al0_tiling_k0
        ub_ka = (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0

        # a_ub_ori_shape([m, k]) and tiling_ori_aub are used to compare full load (pass) or not
        a_ub_ori_shape = [get_value(i) for i in a_ub.shape]
        tiling_ori_aub = [aub_tiling_m * aub_tiling_m0, ub_ka * aub_tiling_k0]

        # tiling_ori_al1, tiling_ori_aub_with_l1 are used to choose compute at tensor
        tiling_ori_al1 = [l1_ma, l1_ka, al0_tiling_m0, al0_tiling_k0]
        tiling_ori_aub_with_l1 = [aub_tiling_m, ub_ka, aub_tiling_m0, aub_tiling_k0]
        aub_l1_affine_shape = [aub_tiling_m, ub_ka, aub_tiling_m0, aub_tiling_k0]

        # tiling_ori_aub is [m, k], in trans scene aub_ori_shape is [k, m]
        if transpose_a and len(a_ub_ori_shape) in (2, 3):
            a_ub_ori_shape[-2:] = [a_ub_ori_shape[-1], a_ub_ori_shape[-2]]
        # in Nz2Zz_int82fp32 scene, ori_shape is 4d
        if self.get_a_matrix_mode == "Nz2Zz_int82fp32":
            tiling_ori_aub = [ub_ka // 2, aub_tiling_m, aub_tiling_m0, aub_tiling_k0 * 2]
        # in nd2Zz_vnchwconv scene L1 tensor is 3d
        elif self.get_a_matrix_mode == "nd2Zz_vnchwconv":
            tiling_ori_al1 = [l1_ma, l1_ka * al0_tiling_k0, al0_tiling_m0]
            tiling_ori_aub_with_l1 = [aub_tiling_m, ub_ka * aub_tiling_k0, aub_tiling_m0]
            if transpose_a:
                aub_l1_affine_shape = [ub_ka, aub_tiling_m * aub_tiling_m0, aub_tiling_k0]
            else:
                aub_l1_affine_shape = [aub_tiling_m, ub_ka * aub_tiling_k0, aub_tiling_m0]
        elif self.get_a_matrix_mode in ("nd2Zz_int8", "nd2Zz", "nd_gemv", "nd_gevm"):
            tiling_ori_aub_with_l1 = [aub_tiling_m, ub_ka, al0_tiling_m0, aub_tiling_k0]
        a_ub_ori_shape, tiling_ori_aub = self._get_dynamic_aub_shape(a_ub_ori_shape, tiling_ori_aub)

        aub_out_affine_shape = [None, aub_tiling_m, aub_tiling_m0, None]
        if self.format_info.get("out") == "ND":
            aub_out_affine_shape = [aub_tiling_m * aub_tiling_m0, None]

        self._renew_aub_m(a_ub_ori_shape)

        cl0_tiling_shape = [cl0_tiling_mc, cl0_tiling_m0, self.c_col_k1, self.c_col_k0]
        aub_tiling_shape_with_lc0 = [
            aub_tiling_m,
            aub_tiling_m0,
            (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
            aub_tiling_k0
        ]
        aub_l0c_affine_shape = [
            None,
            aub_tiling_m,
            None,
            cl0_tiling_n0,
            (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
            aub_tiling_k0
        ]
        if self.status_controller.have_batch_a:
            tiling_ori_aub.insert(0, self.tiling_work.aub_tiling_batch)
            tiling_ori_al1.insert(0, self.tiling_work.al1_tiling_batch)
            aub_l1_affine_shape.insert(0, self.tiling_work.aub_tiling_batch)
            aub_out_affine_shape.insert(0, self.tiling_work.aub_tiling_batch)
            aub_tiling_shape_with_lc0.insert(0, self.tiling_work.aub_tiling_batch)
            aub_l0c_affine_shape.insert(0, self.tiling_work.aub_tiling_batch)
            cl0_tiling_shape.insert(0, self.tiling_work.cl0_tiling_batch)
            tiling_ori_aub_with_l1.insert(0, self.tiling_work.aub_tiling_batch)
        elif self.status_controller.have_batch:
            aub_out_affine_shape.insert(0, None)
            aub_l0c_affine_shape.insert(0, None)
        self._fix_affine_params_for_atomic_k(aub_l0c_affine_shape)
        self._fix_affine_params_for_atomic_k(aub_out_affine_shape)

        compare_params = [
            tiling_ori_aub, a_ub_ori_shape,
            tiling_ori_aub_with_l1, tiling_ori_al1,
            aub_tiling_shape_with_lc0, cl0_tiling_shape
        ]
        status_ori, status_l1, status_l0c = self._aub_attach_modus(compare_params)

        debug(self.DEBUG_PARAM, [tiling_ori_aub, a_ub_ori_shape], "tiling_ori_aub with a_ub_ori_shape")
        debug(self.DEBUG_PARAM, [tiling_ori_aub_with_l1, tiling_ori_al1],
              "tiling_ori_aub_with_l1 with tiling_ori_al1")
        debug(self.DEBUG_PARAM, [aub_tiling_shape_with_lc0, cl0_tiling_shape],
              "aub_tiling_shape_with_lc0 cl0_tiling_shape")
        self._do_attach_aub(status_ori, status_l1, status_l0c, a_ub,
            aub_l1_affine_shape, aub_l0c_affine_shape, aub_out_affine_shape)
        debug(self.DEBUG_PARAM, "-------debug info in aub_process end-------")

    def _aub_attach_modus(self, compare_params):
        (tiling_ori_aub, a_ub_ori_shape, tiling_ori_aub_with_l1,
            tiling_ori_al1, aub_tiling_shape_with_lc0, cl0_tiling_shape) = compare_params
        if self.status_controller.attach_at_flag:
            status_ori = Compare.LESS_EQ
            status_l1 = Compare.LESS_EQ
            status_l0c = Compare.GREATE_EQ
        else:
            status_ori = Compare.compare(tiling_ori_aub, a_ub_ori_shape)
            status_l1 = Compare.compare(tiling_ori_aub_with_l1, tiling_ori_al1)
            status_l0c = Compare.compare(aub_tiling_shape_with_lc0, cl0_tiling_shape)
            if self.status_controller.batch_broadcast_change_attach and self.format_info.get("a") == "ND":
                status_ori = Compare.LESS_EQ
        return status_ori, status_l1, status_l0c

    def _do_attach_aub(self, status_ori, status_l1, status_l0c, a_ub,
        aub_l1_affine_shape, aub_l0c_affine_shape, aub_out_affine_shape):
        if self.is_dynamic and status_ori == Compare.EQUAL:
            status_ori = Compare.LESS_EQ
            self.status_controller.aub_attach_status = "c_gm"
        if status_ori == Compare.EQUAL:
            pass
        elif status_l1 == Compare.EQUAL:
            self.status_controller.aub_attach_status = self.status_controller.al1_attach_status
            self.sch_agent.same_attach(a_ub, self.container.tensor_map.get("a_l1"))
        elif status_l1 == Compare.LESS_EQ:
            self.status_controller.aub_attach_status = "a_l1"
            self.sch_agent.attach_at(a_ub, self.container.tensor_map.get("a_l1"), aub_l1_affine_shape,
                                     factor_shape=self.tiling_work.factor_shape.get("aub"),
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
        else:
            if status_l0c == Compare.EQUAL:
                self.status_controller.aub_attach_status = "c_gm"
                self.sch_agent.same_attach(a_ub, self.container.tensor_map.get("c_l0c"))
            elif status_l0c == Compare.LESS_EQ:
                self.status_controller.aub_attach_status = "c_l0c"
                self.sch_agent.attach_at(a_ub, self.container.tensor_map.get("c_l0c"),
                                         affine_shape=aub_l0c_affine_shape,
                                         ceil_mode_dict=self.tiling_work.get_split_param(
                                             self.cache_tiling_manager))
            else:
                self.status_controller.aub_attach_status = "c_gm"
                self.sch_agent.attach_at(a_ub, self.root_tensor, affine_shape=aub_out_affine_shape,
                                         ceil_mode_dict=self.tiling_work.get_split_param(
                                             self.cache_tiling_manager))

        same_attach_tensors = self.container.tensors_in_aub
        for tensor in same_attach_tensors:
            if tensor == a_ub:
                continue
            self.sch_agent.same_attach(tensor, a_ub)

    def _renew_bub_n(self, b_ub_ori_shape):
        index_offset = 1 if self.status_controller.have_batch_b else 0
        block_n = self.tiling_work.tiling.get("block_dim")[1]
        if self.format_info.get("b") == "ND":
            b_ub_ori_shape[1 + index_offset] = int_ceil_div(b_ub_ori_shape[1 + index_offset],
                                                                  block_n * self.block_out) * self.block_out
        elif self.format_info.get("b") == "FRACTAL_Z":
            b_ub_ori_shape[1 + index_offset] = int_ceil_div(b_ub_ori_shape[1 + index_offset], block_n)
        else:
            if self.status_controller.transpose_b:
                b_ub_ori_shape[1 + index_offset] = int_ceil_div(b_ub_ori_shape[1 + index_offset],
                                                                                block_n)
            else:
                b_ub_ori_shape[index_offset] = int_ceil_div(b_ub_ori_shape[index_offset], block_n)
        if self.status_controller.have_batch_b and not self.is_dynamic:
            b_ub_ori_shape[0] = int_ceil_div(b_ub_ori_shape[0],
                                                             self.tiling_work.tiling.get("block_dim")[0])

    def _get_dynamic_bub_shape(self, bub_shape, bub_tiling):
        if self.is_dynamic:
            # k1 * k0, n1
            bub_shape = [self.dynamic_k * self.block_reduce, self.dynamic_n]
            bub_tiling = [self.tiling_work.bub_tiling_k, self.tiling_work.bub_tiling_n]
            if self.status_controller.have_batch_b:
                bub_shape.insert(0, self.tiling_work.bub_tiling_batch)

        return bub_shape, bub_tiling

    def _bub_process(self):
        b_ub = self.container.tensor_map.get("b_ub")
        if b_ub in (None, []):
            return
        debug(self.DEBUG_PARAM, "-------debug info in bub_process-------")
        l1_kb = (self.tiling_work.bl1_tiling_k + self.tiling_work.bl0_tiling_k0 -
                 1) // self.tiling_work.bl0_tiling_k0
        bub_tiling_k0, bub_tiling_n0 = self.block_reduce, self.block_out
        ub_kb = (self.tiling_work.bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0
        b_ub_ori_shape = [get_value(i) for i in b_ub.shape]
        tiling_ori_bub = [self.tiling_work.bub_tiling_k, self.tiling_work.bub_tiling_n * bub_tiling_n0]
        tiling_ori_bl1 = [
            l1_kb, self.tiling_work.bl1_tiling_n, self.tiling_work.bl0_tiling_n0, self.tiling_work.bl0_tiling_k0
        ]
        tiling_ori_bub_with_l1 = [ub_kb, self.tiling_work.bub_tiling_n, bub_tiling_n0, bub_tiling_k0]
        bub_l1_affine_shape = [ub_kb, self.tiling_work.bub_tiling_n, bub_tiling_n0, bub_tiling_k0]
        if len(b_ub_ori_shape) in (2, 3):
            if self.status_controller.transpose_b:
                b_ub_ori_shape[-2:] = [b_ub_ori_shape[-1], b_ub_ori_shape[-2]]

        if self.get_b_matrix_mode == "nd2Zn_vnchwconv":
            if self.status_controller.transpose_b:
                bub_l1_affine_shape = [self.tiling_work.bub_tiling_n, self.tiling_work.bub_tiling_k, bub_tiling_n0]
            else:
                bub_l1_affine_shape = [ub_kb, self.tiling_work.bub_tiling_n * bub_tiling_n0, bub_tiling_k0]
        elif self.get_b_matrix_mode == "Zn2Zn_int82fp32":
            tiling_ori_bub = [ub_kb // 2, self.tiling_work.bub_tiling_n, bub_tiling_n0, bub_tiling_k0 * 2]
        elif self.get_b_matrix_mode == "Nz2Zn":
            tiling_ori_bub = [self.tiling_work.bub_tiling_n, ub_kb, bub_tiling_k0, bub_tiling_n0]
            bub_l1_affine_shape = [self.tiling_work.bub_tiling_n, ub_kb, bub_tiling_k0, bub_tiling_n0]

        b_ub_ori_shape, tiling_ori_bub = self._get_dynamic_bub_shape(b_ub_ori_shape, tiling_ori_bub)
        self._renew_bub_n(b_ub_ori_shape)
        if self.format_info.get("out") == "ND":
            bub_out_affine_shape = [None, self.tiling_work.bub_tiling_n * bub_tiling_n0]
        else:
            bub_out_affine_shape = [self.tiling_work.bub_tiling_n, None, None, bub_tiling_n0]

        bub_tiling_shape_with_lc0 = [
            self.tiling_work.bub_tiling_n,
            bub_tiling_n0,
            (self.tiling_work.bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
            bub_tiling_k0
        ]
        cl0_tiling_shape = [
            self.tiling_work.cl0_tiling_nc, self.tiling_work.cl0_tiling_n0, self.c_col_k1, self.c_col_k0
        ]

        bub_l0c_affine_shape = [
            self.tiling_work.bub_tiling_n,
            None,
            None,
            self.tiling_work.bl0_tiling_n0,
            (self.tiling_work.bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
            bub_tiling_k0
        ]
        if self.status_controller.have_batch_b:
            tiling_ori_bub.insert(0, self.tiling_work.bub_tiling_batch)
            tiling_ori_bl1.insert(0, self.tiling_work.bl1_tiling_batch)
            bub_l1_affine_shape.insert(0, self.tiling_work.bub_tiling_batch)
            bub_out_affine_shape.insert(0, self.tiling_work.bub_tiling_batch)
            bub_tiling_shape_with_lc0.insert(0, self.tiling_work.bub_tiling_batch)
            cl0_tiling_shape.insert(0, self.tiling_work.cl0_tiling_batch)
            bub_l0c_affine_shape.insert(0, self.tiling_work.bub_tiling_batch)
            tiling_ori_bub_with_l1.insert(0, self.tiling_work.bub_tiling_batch)
        elif self.status_controller.have_batch:
            bub_out_affine_shape.insert(0, None)
            bub_l0c_affine_shape.insert(0, None)
        self._fix_affine_params_for_atomic_k(bub_l0c_affine_shape)
        self._fix_affine_params_for_atomic_k(bub_out_affine_shape)

        compare_params = [
            tiling_ori_bub, b_ub_ori_shape,
            tiling_ori_bub_with_l1, tiling_ori_bl1,
            bub_tiling_shape_with_lc0, cl0_tiling_shape
        ]
        status_ori, status_l1, status_l0c = self._bub_attach_modus(compare_params)

        debug(self.DEBUG_PARAM, [tiling_ori_bub, b_ub_ori_shape], "tiling_ori_bub with b_ub_ori_shape")
        debug(self.DEBUG_PARAM, [tiling_ori_bub_with_l1, tiling_ori_bl1], "tiling_ori_bub_with_l1 with tiling_ori_bl1")
        debug(self.DEBUG_PARAM, [bub_tiling_shape_with_lc0, cl0_tiling_shape],
            "bub_tiling_shape_with_lc0 with cl0_tiling_shape")
        self._do_attach_bub(status_ori, status_l1, status_l0c, b_ub,
            bub_l1_affine_shape, bub_l0c_affine_shape, bub_out_affine_shape)
        debug(self.DEBUG_PARAM, "-------debug info in bub_process end-------")

    def _bub_attach_modus(self, compare_params):
        (tiling_ori_bub, b_ub_ori_shape, tiling_ori_bub_with_l1, tiling_ori_bl1,
            bub_tiling_shape_with_lc0, cl0_tiling_shape) = compare_params
        if self.status_controller.attach_at_flag:
            status_ori = Compare.LESS_EQ
            status_l1 = Compare.LESS_EQ
            status_l0c = Compare.GREATE_EQ
        else:
            status_ori = Compare.compare(tiling_ori_bub, b_ub_ori_shape)
            status_l1 = Compare.compare(tiling_ori_bub_with_l1, tiling_ori_bl1)
            status_l0c = Compare.compare(bub_tiling_shape_with_lc0, cl0_tiling_shape)
            if self.status_controller.batch_broadcast_change_attach and self.format_info.get("b") == "ND":
                status_ori = Compare.LESS_EQ
        return status_ori, status_l1, status_l0c

    def _do_attach_bub(self, status_ori, status_l1, status_l0c, b_ub,
        bub_l1_affine_shape, bub_l0c_affine_shape, bub_out_affine_shape):
        if self.is_dynamic and status_ori == Compare.EQUAL:
            self.status_controller.bub_attach_status = "c_gm"
            status_ori = Compare.LESS_EQ
        if status_ori == Compare.MISC:
            args_dict = {
                "errCode": "E60114",
                "reason": "bub attach error.",
                "value": "compare status = {}".format(status_ori)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        elif status_ori == Compare.EQUAL:
            pass
        elif status_l1 == Compare.EQUAL:
            self.status_controller.bub_attach_status = self.status_controller.bl1_attach_status
            self.sch_agent.same_attach(b_ub, self.container.tensor_map.get("b_l1"))
        elif status_l1 == Compare.LESS_EQ:
            self.status_controller.bub_attach_status = "b_l1"
            self.sch_agent.attach_at(b_ub, self.container.tensor_map.get("b_l1"), bub_l1_affine_shape,
                                     factor_shape=self.tiling_work.factor_shape.get("bub"),
                                     ceil_mode_dict=self.tiling_work.get_split_param(
                                         self.cache_tiling_manager))
        else:
            if status_l0c == Compare.EQUAL:
                self.status_controller.bub_attach_status = "c_gm"
                self.sch_agent.same_attach(b_ub, self.container.tensor_map.get("c_l0c"))
            elif status_l0c == Compare.LESS_EQ:
                self.status_controller.bub_attach_status = "c_l0c"
                self.sch_agent.attach_at(b_ub, self.container.tensor_map.get("c_l0c"),
                                         affine_shape=bub_l0c_affine_shape,
                                         ceil_mode_dict=self.tiling_work.get_split_param(
                                             self.cache_tiling_manager))
            else:
                self.status_controller.bub_attach_status = "c_gm"
                bub_out_affine_shape = self._fix_affine_out_int8(b_ub.dtype, bub_out_affine_shape)
                self.sch_agent.attach_at(b_ub, self.root_tensor, affine_shape=bub_out_affine_shape,
                                         ceil_mode_dict=self.tiling_work.get_split_param(
                                             self.cache_tiling_manager))

        for tensor in self.container.tensors_in_bub:
            if tensor == b_ub:
                continue
            self.sch_agent.same_attach(tensor, b_ub)

    def _get_cache_tiling_axis_order(self):
        """
        get axis_order in cache_tiling scene
        """
        abkl1_attach_flag = self.status_controller.attach_at_flag.get("abkl1_attach_flag")
        if abkl1_attach_flag == self.KBL1_LARGER_FLAG:
            axis_order = ["bl1", "al1"]
            axis_ub_order = ["bub", "aub"]
        else:
            axis_order = ["al1", "bl1"]
            axis_ub_order = ["aub", "bub"]
        if self.format_info.get("a") == "ND":
            axis_order = axis_order + axis_ub_order
        return axis_order

    def _do_l1_ub_process(self):
        """
        do l1 ub process by correct order
        """
        if self.status_controller.attach_at_flag:
            axis_order = self._get_cache_tiling_axis_order()
        else:
            # get order
            order_dict = {
                "aub": [self.tiling_work.aub_tiling_k // self.block_reduce, self.tiling_work.aub_tiling_batch],
                "bub": [self.tiling_work.bub_tiling_k // self.block_reduce, self.tiling_work.bub_tiling_batch],
                "al1": [
                    self.tiling_work.al1_tiling_k // int(self.tiling_work.al0_tiling_k0),
                    self.tiling_work.al1_tiling_batch
                ],
                "bl1": [
                    self.tiling_work.bl1_tiling_k // int(self.tiling_work.bl0_tiling_k0),
                    self.tiling_work.bl1_tiling_batch
                ],
            }
            tmp_order = sorted(order_dict.items(), key=lambda d: [d[1][0], d[1][1]], reverse=True)
            axis_order = [i[0] for i in tmp_order]

            def _adjust_order(axis_order, ub_tag, l1_tag):
                if (axis_order.index(ub_tag) > axis_order.index(l1_tag) and
                    order_dict.get(ub_tag) == order_dict.get(l1_tag)):
                    index_ub = axis_order.index(ub_tag)
                    index_l1 = axis_order.index(l1_tag)
                    axis_order[index_ub] = l1_tag
                    axis_order[index_l1] = ub_tag

            _adjust_order(axis_order, "aub", "al1")
            _adjust_order(axis_order, "bub", "bl1")

        for tag in axis_order[::-1]:
            if tag == "bl1":
                self._bl1_process()
            elif tag == "al1":
                self._al1_process()
            elif tag == "bub":
                self._bub_process()
            else:
                self._aub_process()

    def _bind_core_cache_tiling(self, root_tensor, input_axes_list):
        """
        bind multi core for cache tiling
        """
        ax_batch, ax_reduce, ax_n, ax_m = input_axes_list
        batch_dim, n_dim, m_dim, k_dim = self.tiling_work.tiling.get("block_dim")
        m_factor = (self.cache_tiling.get("m_single_core") * self.cache_tiling.get("m_al1") *
                    self.cache_tiling.get("m_l0"))
        n_factor = (self.cache_tiling.get("n_single_core") * self.cache_tiling.get("n_bl1") *
                    self.cache_tiling.get("n_ub_l0_time") * self.cache_tiling.get("cub_n1"))
        if self.format_info.get("out") == "ND":
            m_factor *= self.block_in
            n_factor *= self.block_out
        split_params = SplitParam(tail_strategy="shift_inwards", active_scope="inner")
        ax_m_out, ax_m_inner = self.sch_agent[root_tensor].split(ax_m, factor=m_factor, split_params=split_params)
        ax_n_out, ax_n_inner = self.sch_agent[root_tensor].split(ax_n, factor=n_factor, split_params=split_params)
        multi_core_axes_list = [ax_n_out, ax_m_out]
        single_core_axes_list = [ax_n_inner, ax_m_inner]
        self.container.axis_core = ax_n_out * m_dim + ax_m_out
        if self.status_controller.have_batch:
            ax_batch_out, ax_batch_inner = self.sch_agent[root_tensor].split(
                ax_batch, factor=self.cache_tiling.get("batch_single_core"), split_params=split_params)
            multi_core_axes_list.insert(0, ax_batch_out)
            single_core_axes_list.insert(0, ax_batch_inner)
            self.container.axis_core += ax_batch_out * (n_dim * m_dim)
        if self.status_controller.split_k_axis_by_tiling:
            ax_k_out, ax_k_inner = self.sch_agent[root_tensor].split(ax_reduce, factor=1)
            multi_core_axes_list.insert(0, ax_k_out)
            single_core_axes_list.insert(0, ax_k_inner)
            self.container.axis_core += ax_k_out * (batch_dim * n_dim * m_dim)
        split_reorder_axes_list = multi_core_axes_list + single_core_axes_list
        self.sch[root_tensor].reorder(*split_reorder_axes_list)
        return multi_core_axes_list

    def _bind_multi_core(self):
        axis_mn = self.sch_agent[self.root_tensor].get_active_scopes()
        ax_batch = 1
        offset_value = 1 if self.status_controller.split_k_axis_by_tiling else 0
        if (not self.status_controller.have_batch) and (not self.status_controller.reduce_fusion):
            upper_range = 2 + offset_value
            ax_m, ax_n = axis_mn[offset_value:upper_range]
        else:
            upper_range = 3 + offset_value
            ax_batch, ax_m, ax_n = axis_mn[offset_value:upper_range]
        if self.format_info.get("out") != "ND":
            ax_m, ax_n = ax_n, ax_m

        batch_dim, n_dim, m_dim, reduce_dim = self.tiling_work.tiling.get("block_dim")
        if self.status_controller.reduce_fusion:
            batch_dim = get_value(self.container.tensor_map.get("c_l0c").shape)[0]
        if self.cache_tiling:
            ax_reduce = axis_mn[0] if self.status_controller.split_k_axis_by_tiling else None
            axes_list = self._bind_core_cache_tiling(self.root_tensor, [ax_batch, ax_reduce, ax_n, ax_m])
            self.sch.bind_axes(axes_list, tvm.thread_axis("blockIdx.x"))
        else:
            axis_list = [ax_m, ax_n]
            axis_dim = [m_dim, n_dim]
            if self.status_controller.have_batch or self.status_controller.reduce_fusion:
                axis_list.insert(0, ax_batch)
                axis_dim.insert(0, batch_dim)
            if self.status_controller.split_k_axis_by_tiling:
                axis_list.insert(0, axis_mn[0])
                axis_dim.insert(0, reduce_dim)

            self.container.axis_core = self.sch_agent[self.root_tensor].bind_core(axis_list, axis_dim)

    def _buffer_align_func(self, tensor, have_batch, *align_args):

        if tensor is not None:
            if have_batch:
                self.sch[tensor].buffer_align((1, 1), *align_args)
            else:
                self.sch[tensor].buffer_align(*align_args)

    def _do_buffer_align(self):
        have_batch_b = self.status_controller.have_batch_b
        have_batch_a = self.status_controller.have_batch_a
        have_batch = self.status_controller.have_batch
        self._do_buffer_align_l0c()
        self._set_requant_transfer_buffer_align()
        is_int82fp32_nd = self._is_int82fp32_nd()

        self._buffer_align_func(self.container.tensor_map.get("b_transpose"), have_batch_b, (1, 32), (1, 32))
        self._buffer_align_func(self.container.tensor_map.get("a_transpose"), have_batch_a, (1, 32), (1, 32))
        if is_int82fp32_nd:
            self._buffer_align_func(self.container.tensor_map.get("b_ub"), have_batch_b, (1, 32), (1, 32))
            self._buffer_align_func(self.container.tensor_map.get("a_ub"), have_batch_a, (1, 32), (1, 32))

        if self.format_info.get("out") == "ND":
            self._buffer_align_func(self.container.tensor_map.get("c_add_bias_ub"), have_batch, (1, 16), (1, 16))
            self._buffer_align_func(self.container.tensor_map.get("beta_bias"), have_batch, (1, 16), (1, 16))

        cast_to_fp16 = self.container.tensor_map.get("cast_to_fp16")
        if cast_to_fp16 is not None:
            if len(cast_to_fp16.shape) in (2, 3):
                self._buffer_align_func(self.container.tensor_map.get("cast_to_fp16"), have_batch, (1, 16), (1, 16))
            else:
                self._buffer_align_func(self.container.tensor_map.get("cast_to_fp16"),
                    have_batch, (1, 1), (1, 1), (1, 16), (1, 16))

        if self.status_controller.split_k_axis_by_tiling:
            self._buffer_align_func(self.container.tensor_map.get("c_ub_fract"), have_batch, (1, 1), (1, 1), (1, 1),
                                    (1, 16), (1, 16))
        else:
            self._buffer_align_func(self.container.tensor_map.get("c_ub_fract"), have_batch, (1, 1), (1, 1), (1, 16),
                                    (1, 16))
        self._buffer_align_func(
            self.container.tensor_map.get("bias_l0c"), have_batch, (1, 1), (1, 1), (1, 16), (1, 16))
        self._buffer_align_func(
            self.container.tensor_map.get("c_add_bias"), have_batch, (1, 1), (1, 1), (1, 16), (1, 16))

        if self.status_controller.mmad_mode == "gevm":
            self._buffer_align_func(
                self.container.tensor_map.get("a_l1"), have_batch_a, (1, 1), (1, 1), (1, 16), (1, 16))

    def _double_buffer(self, a_run_once, b_run_once):
        tiling = self.tiling_work.tiling
        if (tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") == 2
            and (a_run_once == self.ALLOCATE_OFF)):
            self.sch[self.container.tensor_map.get("a_l1")].double_buffer()
        if (tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") == 2
            and (b_run_once == self.ALLOCATE_OFF)):
            self.sch[self.container.tensor_map.get("b_l1")].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") == 2:
            self.sch[self.container.tensor_map.get("a_l0a")].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") == 2:
            self.sch[self.container.tensor_map.get("b_l0b")].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == 2:
            for tensor in self.container.tensors_in_l0c:
                self.sch[tensor].double_buffer()
                if self.cache_tiling_manager.flag_l0c_preload:
                    self.sch[tensor].preload()
        self._double_buffer_aub_bub()
        self._double_buffer_cub()

    def _double_buffer_aub_bub(self):
        tiling = self.tiling_work.tiling
        params = {"container": self.container, "status_controller": self.status_controller,
                  "cache_tiling": self.cache_tiling}
        # disable aub_bub_preload when l0c_preload enabled for accuracy error
        if tiling.get("manual_pingpong_buffer").get("AUB_pbuffer") == 2:
            if (self.container.tensor_map.get("a_ub") is not None and
                (not self.is_dynamic or self.cache_tiling) and
                self.buffer_checker.check_aub_preload(tiling, params) and
                (not self.cache_tiling_manager.flag_l0c_preload)):
                self.sch[self.container.tensor_map.get("a_ub")].preload()
                if (self.container.tensor_map.get("a_ub_aligned") is not None and
                    self.container.tensor_map.get("a_ub_general") is not None):
                    self.sch[self.container.tensor_map.get("a_ub_aligned")].preload()
                    self.sch[self.container.tensor_map.get("a_ub_general")].preload()
            for tensor in self.container.tensors_in_aub:
                self.sch[tensor].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("BUB_pbuffer") == 2:
            if (self.container.tensor_map.get("b_ub") is not None and
                (not self.is_dynamic or self.cache_tiling) and
                self.buffer_checker.check_bub_preload(tiling, params) and
                (not self.cache_tiling_manager.flag_l0c_preload)):
                self.sch[self.container.tensor_map.get("b_ub")].preload()
                if (self.container.tensor_map.get("b_ub_aligned") is not None and
                    self.container.tensor_map.get("b_ub_general") is not None):
                    self.sch[self.container.tensor_map.get("b_ub_aligned")].preload()
                    self.sch[self.container.tensor_map.get("b_ub_general")].preload()
            for tensor in self.container.tensors_in_bub:
                self.sch[tensor].double_buffer()

    def _double_buffer_cub(self):
        tiling = self.tiling_work.tiling
        params = {"container": self.container, "status_controller": self.status_controller,
                  "cache_tiling": self.cache_tiling}
        if tiling.get("manual_pingpong_buffer").get("CUB_pbuffer") == 2:
            bias_ub = self.container.tensor_map.get("bias_ub")
            if bias_ub is not None:
                if not self.is_dynamic and self.buffer_checker.check_bias_preload(tiling, params):
                    self.sch[bias_ub].preload()
                self.sch[bias_ub].double_buffer()

                if self.status_controller.need_init_bias:
                    if not self.is_dynamic and self.buffer_checker.check_bias_preload(tiling, params):
                        self.sch[self.container.tensor_map.get('init_value_of_bias_ub')].preload()
                        self.sch[self.container.tensor_map.get('virtual_add_bias')].preload()
                    self.sch[self.container.tensor_map.get('init_value_of_bias_ub')].double_buffer()
                    self.sch[self.container.tensor_map.get('virtual_add_bias')].double_buffer()
            for tensor in self.container.tensors_in_cub:
                if tensor in (self.res, self.container.tensor_map.get("c_gm")):
                    continue
                self.sch[tensor].double_buffer()
            if self.status_controller.fusion_multi_output_flag:
                self.sch[self.container.tensor_map.get("c_gm")].double_buffer()

    def _emit_insn_func(self, insn_tensor, insn_axis_num, insn_tag, insn_dict=None, mode=0, offset=0):
        normal_mode = 0
        if insn_tensor is not None:
            tensor_len = len(insn_tensor.shape) + offset
            if mode == normal_mode:
                scope_insn = self.sch_agent[insn_tensor].op.axis[insn_axis_num]
            else:
                scopes_intrins = self.sch_agent[insn_tensor].intrin_scopes(tensor_len)
                scope_insn = scopes_intrins[insn_axis_num]

            if insn_dict is None:
                self.sch_agent[insn_tensor].emit_insn(scope_insn, insn_tag)
            else:
                self.sch_agent[insn_tensor].emit_insn(scope_insn, insn_tag, insn_dict)

    def _pragma_func(self, insn_tensor, insn_axis_num, insn_tag):
        if insn_tensor is not None:
            self.sch_agent[insn_tensor].pragma(insn_tensor.op.axis[insn_axis_num], insn_tag, insn_axis_num)

    def _do_emit_insn_l0c(self):
        # emit insn for l0c
        c_l0c = self.container.tensor_map.get("c_l0c")
        scopes_intrins = self.sch_agent[c_l0c].intrin_scopes(6)
        scope_insn = scopes_intrins[0]
        if self.status_controller.split_k_axis_by_tiling:
            reduce_axis_index = 1
        else:
            reduce_axis_index = 0
        inner_k_axis = self.sch_agent[c_l0c].get_relate_scope(c_l0c.op.reduce_axis[reduce_axis_index], scope_insn)
        if inner_k_axis:
            mad_dict = {
                "mad_pattern": self.status_controller.mad_pattern,
                "k_outer": self.sch_agent[c_l0c].get_relate_scope(c_l0c.op.reduce_axis[reduce_axis_index], scope_insn)
            }
        else:
            (inner_nb, inner_mb, inner_mp, inner_np, inner_kb, inner_kp) = scopes_intrins
            inner_ko, inner_ki = self.sch_agent[c_l0c].split(inner_kb, nparts=1)
            self.sch_agent[c_l0c].reorder(
                inner_ko, inner_nb, inner_mb, inner_mp, inner_np, inner_ki, inner_kp
            )
            mad_dict = {"mad_pattern": self.status_controller.mad_pattern, "k_outer": [inner_ko]}

        if self.container.tensor_map.get("c_add_bias") is not None:
            self.sch_agent[c_l0c].pragma(scope_insn, "replace_output", 0)
            mad_dict["init_bias"] = 1

        self.sch_agent[c_l0c].emit_insn(scope_insn, "mad", mad_dict)

    def _do_emit_insn(self):
        self._do_emit_insn_aub()
        self._do_emit_insn_bub()
        self._emit_insn_nz_to_nd()

        # only in |gemm|ND Nz| all data type|
        self._emit_insn_func(self.container.tensor_map.get("alpha_ub"), 0, "dma_copy")
        self._emit_insn_func(self.container.tensor_map.get("beta_ub"), 0, "dma_copy")
        self._emit_insn_func(self.container.tensor_map.get("alpha_c"), 0, "vector_muls", mode=1)
        self._emit_insn_func(self.container.tensor_map.get("beta_bias"), 0, "vector_muls")
        self._emit_insn_func(self.container.tensor_map.get("bias_ub"), 0, "dma_copy")
        # only in matmul ND out solve nonline problem
        self._emit_insn_func(self.container.tensor_map.get("before_c_gm"), 0, "vector_muls")

        if self.status_controller.need_init_bias:
            self._emit_insn_func(self.container.tensor_map.get("init_value_of_bias_ub"), 0, "dma_copy")
            self._emit_insn_func(self.container.tensor_map.get("virtual_add_bias"), 0, "phony_insn")

        # only in |matmul|ND Nz|all data type|
        self._emit_insn_func(self.container.tensor_map.get("bias_l0c"), 0, "dma_copy")
        self._emit_insn_func(self.container.tensor_map.get("c_add_bias"), 0, "phony_insn")
        self._pragma_func(self.container.tensor_map.get("bias_l0c"), 0, "reuse_output")
        self._pragma_func(self.container.tensor_map.get("c_add_bias"), 0, "replace_output")

        #only in |gemm|ND Nz|fp162fp16|
        self._emit_insn_func(self.container.tensor_map.get("alpha_fp162fp32"), 0, "vector_conv")
        self._emit_insn_func(self.container.tensor_map.get("beta_fp162fp32"), 0, "vector_conv")
        self._emit_insn_func(self.container.tensor_map.get("bias_cast_to_fp32"), 0, "vector_conv", mode=1)

        c_ub_fract = self.container.tensor_map.get("c_ub_fract")
        cast_to_fp16_cmd = "dma_copy" if self.status_controller.compute_inline_c_ub_fract else "vector_conv"
        # only in |gemm matmul|ND Nz|to fp16|
        self._emit_insn_func(self.container.tensor_map.get("cast_to_fp16"), 0, cast_to_fp16_cmd, mode=1)

        # only in gemm int82fp16
        self._emit_insn_func(self.container.tensor_map.get("a_int82fp16"), 0, "vector_conv", mode=1)
        self._emit_insn_func(self.container.tensor_map.get("b_int82fp16"), 0, "vector_conv", mode=1)

        # common
        self._emit_insn_func(self.container.tensor_map.get("a_l0a"), 0, "dma_copy", mode=1)
        self._emit_insn_func(self.container.tensor_map.get("a_l1"), 0, "dma_copy", mode=1)
        self._tensor_b_emit_insn()

        # fusion
        c_ub_fract = self.container.tensor_map.get("c_ub_fract")
        if c_ub_fract.op.attrs["scale_drq"].value != "ENABLE":
            self._emit_insn_func(self.container.tensor_map.get("c_ub_fract"), 0, "dma_copy", mode=1)
        else:
            self._set_quant_by_params()

        self._choose_dma_copy_for_res()
        self._do_emit_insn_l0c()

        # fusion
        self._quantify_fusion_entry()
        self._tensor_a_l1_workspace_emit()
        self._emit_insn_elemwise_tensor()
        self._do_emit_insn_multi_output()

        if self.status_controller.reduce_fusion:
            self._emit_insn_func(self.container.tensor_map.get("res_atomic_add_ub"), 0, "dma_copy", mode=1)

        self._emit_insn_after_split_k()

    def _do_emit_insn_multi_output(self):
        if self.status_controller.gm_ub is not None:
            if not self.status_controller.matmul_multi_output_flag:
                self._emit_insn_func(self.container.tensor_map.get("c_gm"), 0, "dma_copy", mode=1)
            else:
                self._emit_insn_for_multi_output()
            self._emit_insn_func(self.status_controller.gm_ub, 0, "phony_insn", mode=1)
        elif self.container.double_out_tensor:
            self._emit_insn_for_multi_output()
            self._emit_insn_func(self.container.double_out_tensor[0], 0, "dma_copy", mode=1)
            self._emit_insn_func(self.root_tensor, 0, "phony_insn", mode=1)

    def _emit_insn_for_multi_output(self):
        if len(self.container.tensor_map.get("c_gm").shape) in (4, 5):
            gm_n_outer, gm_n_inner = self.sch_agent[self.container.tensor_map.get("c_gm")].split(
                self.container.tensor_map.get("c_gm").op.axis[-4], nparts=1)
            gm_m_outer, gm_m_inner = self.sch_agent[self.container.tensor_map.get("c_gm")].split(
                self.container.tensor_map.get("c_gm").op.axis[-3], nparts=1)
            self.sch_agent[self.container.tensor_map.get("c_gm")].reorder(
                gm_n_outer, gm_m_outer, gm_n_inner, gm_m_inner)
            if self.status_controller.fusion_multi_output_flag:
                self.sch_agent[self.container.tensor_map.get("c_gm")].emit_insn(gm_n_inner, "phony_insn")
            else:
                self.sch_agent[self.container.tensor_map.get("c_gm")].emit_insn(gm_n_inner, "dma_copy")
        else:
            gm_n_outer, gm_n_inner = self.sch_agent[self.container.tensor_map.get("c_gm")].split(
                self.container.tensor_map.get("c_gm").op.axis[-1], nparts=1)
            gm_m_outer, gm_m_inner = self.sch_agent[self.container.tensor_map.get("c_gm")].split(
                self.container.tensor_map.get("c_gm").op.axis[-2], nparts=1)
            self.sch_agent[self.container.tensor_map.get("c_gm")].reorder(
                gm_m_outer, gm_n_outer, gm_m_inner, gm_n_inner)
            self.sch_agent[self.container.tensor_map.get("c_gm")].emit_insn(gm_m_inner, "dma_copy")

    def _do_emit_insn_for_tensor_aub(self):
        if self.is_dynamic:
            # only in |batch_matmul/ matmul|nd|
            if self.status_controller.a_use_aligned_pattern:
                align_insn_dict = {"map_policy" : "2d"}
                self._emit_insn_func(self.container.tensor_map.get("a_ub_aligned"), 0, "dma_copy",
                                     insn_dict=align_insn_dict, mode=1)
                self._emit_insn_func(self.container.tensor_map.get("a_ub_general"), 0, "phony_insn", mode=1)
            else:
                self._emit_insn_func(self.container.tensor_map.get("a_ub_aligned"), 0, "phony_insn", mode=1)
                self._emit_insn_func(self.container.tensor_map.get("a_ub_general"), 0, "dma_copy", mode=1)
            self._emit_insn_func(self.container.tensor_map.get("a_ub"), 0, "phony_insn", mode=1)
        else:
            # only in |gemm matmul|nd|all| or |matmul|nz|int82fp32| etc
            self._emit_insn_func(self.container.tensor_map.get("a_ub"), 0, "dma_copy", mode=1)
            self._emit_insn_func(self.container.tensor_map.get("a_ub_virtual_align"), 0, "phony_insn", mode=1)

    def _emit_insn_elemwise_tensor(self):
        for ten_in in self.container.elemwise_tensors:
            if ten_in.op.tag.find("|") != -1:
                str_list = ten_in.op.tag.split("|")
                insn = self.emit_insn_map.get(str_list[0])
            else:
                insn = self.emit_insn_map.get(ten_in.op.tag)
            if ten_in in self.container.ele_header_ub_tensors:
                insn = "dma_copy"
            if insn is None:
                insn = "vector_auto"
            self._emit_insn_func(ten_in, 0, insn)

    def _do_emit_insn_aub(self):
        offset_a = 1 if self.status_controller.have_batch_a else 0
        self._do_emit_insn_for_tensor_aub()

        a_cast_and_reshape = (
            (self.status_controller.ops_data_flow_mode == "int82fp32") and (self.format_info.get("a") == "FRACTAL_NZ"))
        a_only_reshape = (
            (self.status_controller.mmad_mode in ("gevm", "gemv")) or (self.get_a_matrix_mode == "nd2Zz_int8"))
        a_only_reshape = a_only_reshape or (self.get_a_matrix_mode == "nd2Zz" and self.is_dynamic)
        if a_cast_and_reshape:
            # only in |gemm matmul|nz|int82fp32|
            a_ub_fract = self.container.tensor_map.get("a_ub_fract")
            if self.status_controller.have_batch_a:
                _, a_ub_scope_outer, a_ub_scope_inner, _, _ = self.sch_agent[a_ub_fract].get_active_scopes()
            else:
                a_ub_scope_outer, a_ub_scope_inner, _, _ = self.sch_agent[a_ub_fract].get_active_scopes()
            self.sch_agent[a_ub_fract].split(a_ub_scope_inner, 2)
            self.sch_agent[a_ub_fract].emit_insn(a_ub_scope_outer, "vector_auto")
        elif a_only_reshape:
            # The pass side causes the vector instruction to have performance regression in some scenarios,
            # so this restriction is added
            reshape_cmd = "vector_muls" if (self.format_info.get("a") == "ND"
                                            and self.format_info.get("b") == "ND") else "dma_copy"
            self._emit_insn_func(
                self.container.tensor_map.get("a_ub_fract"), 0, reshape_cmd, insn_dict=self.container.vector_muls_attr)
        else:
            # only in |gemm matmul|ND|fp162fp16 fp162fp32 int82fp32|
            self._emit_insn_func(self.container.tensor_map.get("a_ub_fract"), 1 + offset_a, "vnchwconv", mode=1)

        # only in |gemm|ND|int82int32|
        a_transpose = self.container.tensor_map.get("a_transpose")
        if a_transpose is not None:
            m_outer, m_inner = self.sch_agent[a_transpose].split(a_transpose.op.axis[1 + offset_a], factor=32)
            self.sch_agent[a_transpose].reorder(m_outer, a_transpose.op.axis[offset_a], m_inner)
            self.sch_agent[a_transpose].emit_insn(self.sch_agent[a_transpose].op.axis[offset_a], "vnchwconv")

    def _do_emit_insn_for_tensor_bub(self):
        if self.is_dynamic:
            # only in |batch_matmul/matmul|nd|
            if self.status_controller.b_use_aligned_pattern:
                align_insn_dict = {"map_policy" : "2d"}
                self._emit_insn_func(self.container.tensor_map.get("b_ub_aligned"), 0, "dma_copy",
                                     insn_dict=align_insn_dict, mode=1)
                self._emit_insn_func(self.container.tensor_map.get("b_ub_general"), 0, "phony_insn", mode=1)
            else:
                self._emit_insn_func(self.container.tensor_map.get("b_ub_aligned"), 0, "phony_insn", mode=1)
                self._emit_insn_func(self.container.tensor_map.get("b_ub_general"), 0, "dma_copy", mode=1)
            self._emit_insn_func(self.container.tensor_map.get("b_ub"), 0, "phony_insn", mode=1)
        else:
            # only in |gemm matmul|nd|all| or |matmul|nz|int82fp32| etc
            self._emit_insn_func(self.container.tensor_map.get("b_ub"), 0, "dma_copy", mode=1)

    def _do_emit_insn_bub(self):
        offset_b = 1 if self.status_controller.have_batch_b else 0
        self._do_emit_insn_for_tensor_bub()

        b_cast_and_reshape = (
            (self.status_controller.ops_data_flow_mode == "int82fp32") and (self.format_info.get("b") == "FRACTAL_Z"))
        b_only_reshape = (self.status_controller.mmad_mode == "gemv") or (self.get_b_matrix_mode == "nd2Zn_int8")
        b_only_reshape = b_only_reshape or (self.get_b_matrix_mode == "nd2Zn" and self.is_dynamic)
        if b_cast_and_reshape:
            # only in |gemm matmul|nz|int82fp32|
            b_ub_fract = self.container.tensor_map.get("b_ub_fract")
            if self.status_controller.have_batch_b:
                _, b_ub_scope_outer, _, _, _ = self.sch_agent[b_ub_fract].get_active_scopes()
            else:
                b_ub_scope_outer, _, _, _ = self.sch_agent[b_ub_fract].get_active_scopes()
            b_ub_outer_outer, _ = self.sch_agent[b_ub_fract].split(b_ub_scope_outer, 2)
            self.sch_agent[b_ub_fract].emit_insn(b_ub_outer_outer, "vector_auto")
        elif b_only_reshape:
            # The pass side causes the vector instruction to have performance regression in some scenarios,
            # so this restriction is added
            reshape_cmd = "vector_muls" if (self.format_info.get("a") == "ND"
                                            and self.format_info.get("b") == "ND") else "dma_copy"
            self._emit_insn_func(
                self.container.tensor_map.get("b_ub_fract"), 0, reshape_cmd, insn_dict=self.container.vector_muls_attr)
        else:
            # only in |gemm matmul|ND|fp162fp16 fp162fp32 int82fp32|
            self._emit_insn_func(self.container.tensor_map.get("b_ub_fract"), 1 + offset_b, "vnchwconv", mode=1)

        # only in |gemm|ND|int82int32|
        b_transpose = self.container.tensor_map.get("b_transpose")
        if b_transpose is not None:
            k_outer, k_inner = self.sch_agent[b_transpose].split(b_transpose.op.axis[1 + offset_b], factor=32)
            self.sch_agent[b_transpose].reorder(k_outer, b_transpose.op.axis[offset_b], k_inner)
            self.sch_agent[b_transpose].emit_insn(self.sch_agent[b_transpose].op.axis[offset_b], "vnchwconv")

    def _emit_insn_nz_to_nd(self):
        c_add_bias_ub = self.container.tensor_map.get("c_add_bias_ub")
        nz_to_nd = self.container.tensor_map.get("nz_to_nd")
        fract_add_nd_to_nd = (self.format_info.get("out") == "ND") and (c_add_bias_ub is not None)

        if fract_add_nd_to_nd:
            self._cut_axis_for_nz_to_nd(c_add_bias_ub, "vector_add")
        elif c_add_bias_ub is not None:
            # only in |gemm|Nz|all data type|
            self._emit_insn_func(c_add_bias_ub, 0, "vector_add", mode=1)

        if nz_to_nd is not None:
            self._cut_axis_for_nz_to_nd(nz_to_nd, "vector_muls", attrs=self.container.vector_muls_attr)

    def _cut_axis_for_nz_to_nd(self, ori_tensor, emit_insn_cmd, attrs=None):
        if attrs is None:
            attrs = {}
        # only in |gemm|ND|all data type|
        if self.status_controller.have_batch:
            scope_batch, scope_outer, scope_inner = self.sch_agent[ori_tensor].get_active_scopes()
        else:
            scope_outer, scope_inner = self.sch_agent[ori_tensor].get_active_scopes()
        split_params = SplitParam(
            self.tiling_work.get_split_param(self.cache_tiling_manager),
            "guard_with_if", "outer")
        outer_outer, outer_inner = self.sch_agent[ori_tensor].split(scope_outer, self.block_in,
                                                                    split_params=split_params)
        inner_outer, inner_inner = self.sch_agent[ori_tensor].split(scope_inner, self.block_out,
                                                                    split_params=split_params)
        self.sch_agent[ori_tensor].reorder(outer_outer, inner_outer, outer_inner, inner_inner)
        if self.cache_tiling:
            m_inner_outer, _ = self.sch_agent[ori_tensor].split(outer_inner, self.BLOCKS_PER_REPEAT,
                                                                split_params=split_params)
            self.sch_agent[ori_tensor].reorder(m_inner_outer, outer_outer, inner_outer)
            self.sch_agent[ori_tensor].emit_insn(outer_outer, emit_insn_cmd, attrs=attrs)
        elif self.status_controller.have_batch:
            self.sch_agent[ori_tensor].emit_insn(scope_batch, emit_insn_cmd, attrs=attrs)
        else:
            self.sch_agent[ori_tensor].emit_insn(outer_inner, emit_insn_cmd, attrs=attrs)

    def _tensor_b_emit_insn(self):
        """
        tensor_b_l1 emit insn operation for compress or not
        """
        b_l0b = self.container.tensor_map.get("b_l0b")
        b_l1 = self.container.tensor_map.get("b_l1")
        if self.status_controller.compress_flag:
            host_axis = self._get_index_at_axis()
            compress_index = self.container.tensor_map.get("compress_index")
            if not self.status_controller.b_l1_inline_flag:
                b_l1.op.attrs["tile_L1_k"] = b_l0b.op.attrs["tile_L1_k"]
                b_l1.op.attrs["tile_L1_n"] = b_l0b.op.attrs["tile_L1_n"]
                # k_l1_tile n_l1_tile host_axis
                k_l1_tile = self.tiling_work.bl1_tiling_k // self.block_reduce
                n_l1_tile = self.tiling_work.bl1_tiling_n
                self._set_compress_info(b_l1, compress_index, k_l1_tile, n_l1_tile, host_axis)
                self._emit_insn_func(b_l0b, 0, "dma_copy", mode=1)
            else:
                k_l0_tile = self.tiling_work.bl0_tiling_kb
                n_l0_tile = self.tiling_work.bl0_tiling_nb
                self._set_compress_info(b_l0b, compress_index, k_l0_tile, n_l0_tile, host_axis)
        else:
            if not self.status_controller.b_l1_inline_flag:
                self._emit_insn_func(b_l1, 0, "dma_copy", mode=1)
            self._emit_insn_func(b_l0b, 0, "dma_copy", mode=1)

    def _add_key_value(self, key, value):
        buffer_reuse_dict = self.container.buffer_reuse_dict
        if (key is not None) and (value is not None):
            buffer_reuse_dict[key] = value

    def _set_buffer_reuse_dict(self):
        self._add_key_value(self.container.tensor_map.get("c_ub_fract"), self.container.tensor_map.get("alpha_c"))
        if self.format_info.get("out") == "FRACTAL_NZ":
            self._add_key_value(
                self.container.tensor_map.get("c_add_bias_ub"), self.container.tensor_map.get("alpha_c"))
        else:
            self._add_key_value(
                self.container.tensor_map.get("c_add_bias_ub"), self.container.tensor_map.get("beta_bias"))

        if self.status_controller.ops_data_flow_mode == "fp162fp16":
            self._add_key_value(
                self.container.tensor_map.get("beta_fp162fp32"), self.container.tensor_map.get("beta_ub"))
            self._add_key_value(
                self.container.tensor_map.get("alpha_fp162fp32"), self.container.tensor_map.get("alpha_ub"))
            self._add_key_value(
                self.container.tensor_map.get("beta_bias"), self.container.tensor_map.get("bias_cast_to_fp32"))
        else:
            self._add_key_value(self.container.tensor_map.get("beta_bias"), self.container.tensor_map.get("bias_ub"))

        for axpy, parent in self.container.axpy_2_parent.items():
            self.container.buffer_reuse_dict[parent] = axpy

        if self.status_controller.need_init_bias:
            self._add_key_value(self.container.tensor_map.get("virtual_add_bias"),
                [self.container.tensor_map.get("bias_ub"), self.container.tensor_map.get("init_value_of_bias_ub")])

        if self.status_controller.fusion_multi_output_flag:
            self._add_key_value(self.container.tensor_map.get("cast_to_fp16"), self.container.tensor_map.get("c_gm"))
        # Enable aub/bub to select schedule pattern
        a_ub = self.container.tensor_map.get("a_ub")
        b_ub = self.container.tensor_map.get("b_ub")
        if a_ub is not None and self.is_dynamic:
            self._add_key_value(
                self.container.tensor_map.get("a_ub_aligned"), [a_ub, self.container.tensor_map.get("a_ub_general")])
        if b_ub is not None and self.is_dynamic:
            self._add_key_value(
                self.container.tensor_map.get("b_ub_aligned"), [b_ub, self.container.tensor_map.get("b_ub_general")])

    def _do_buffer_reuse(self):
        for bereused_tensor, tensor in self.container.buffer_reuse_dict.items():
            if (bereused_tensor is not None) and (tensor is not None):
                if isinstance(tensor, Iterable):
                    self.sch[bereused_tensor].reused_by(*tensor)
                else:
                    self.sch[bereused_tensor].reused_by(tensor)

    def _init_run_once_flag(self):
        a_run_once, b_run_once = self.ALLOCATE_OFF, self.ALLOCATE_OFF

        dtype_byte = self.DTYPE_WIDTH_MAP.get(self.container.tensor_map.get("a_l1").dtype) * 2
        size = tbe_platform_info.get_soc_spec("L1_SIZE") // dtype_byte // 2
        block_dim = self.tiling_work.tiling.get("block_dim")

        core_inner_m = int_ceil_div(
            self.container.tensor_map.get("a_l0a").shape[self.FRACTAL_Z_M_INDEX].value * self.block_in,
            block_dim[self.BLOCK_M_DIM_INDEX])
        core_inner_n = int_ceil_div(
            self.container.tensor_map.get("b_l0b").shape[self.FRACTAL_Z_N_INDEX].value * self.block_out,
            block_dim[self.BLOCK_N_DIM_INDEX])
        k_shape = self.container.tensor_map.get("a_l0a").shape[self.FRACTAL_Z_KA_INDEX].value * self.block_reduce

        m_l1_shape = self.tiling_work.al1_tiling_m * self.block_in
        n_l1_shape = self.tiling_work.bl1_tiling_n * self.block_out
        m_max_num = int_ceil_div(core_inner_m, m_l1_shape) * m_l1_shape
        n_max_num = int_ceil_div(core_inner_n, n_l1_shape) * n_l1_shape

        tensor_a_num = m_max_num * k_shape
        tensor_b_num = n_max_num * k_shape
        if m_max_num * k_shape <= size:
            a_run_once = self.ALLOCATE_FULL
        elif m_l1_shape * k_shape <= size:
            a_run_once = self.ALLOCATE_HALF

        if n_max_num * k_shape <= size:
            b_run_once = self.ALLOCATE_FULL
        elif n_l1_shape * k_shape <= size:
            b_run_once = self.ALLOCATE_HALF

        if a_run_once == self.ALLOCATE_HALF and b_run_once == self.ALLOCATE_HALF:
            aprts_a = core_inner_m // m_l1_shape
            aprts_b = core_inner_n // n_l1_shape
            if tensor_a_num * aprts_b < tensor_b_num * aprts_a:
                a_run_once = self.ALLOCATE_OFF
            else:
                b_run_once = self.ALLOCATE_OFF

        batch = 0
        if self.status_controller.have_batch:
            batch = self.container.tensor_map.get("c_l0c").shape[0].value
        batch_double = False
        if batch > 1:
            if tensor_a_num <= size and tensor_b_num <= size:
                batch_double = True

        double_once = self.ALLOCATE_OFF
        if core_inner_m != m_l1_shape and core_inner_n != n_l1_shape:
            double_once = self.ALLOCATE_HALF

        return a_run_once, b_run_once, batch_double, double_once

    def _allocate_axis(self, enable_nbuffer):
        if not enable_nbuffer or self.is_dynamic:
            return self.ALLOCATE_OFF, self.ALLOCATE_OFF
        a_run_once, b_run_once, batch_double, double_once = self._init_run_once_flag()
        axis_outer = self.sch_agent[self.root_tensor].get_active_scopes()
        if self.format_info.get("out") == "FRACTAL_NZ":
            m_outer = axis_outer[self.FRACTAL_NZ_M_INDEX]
            n_outer = axis_outer[self.FRACTAL_NZ_N_INDEX]
        else:
            m_outer = axis_outer[self.ND_M_INDEX]
            n_outer = axis_outer[self.ND_N_INDEX]
        m_outer, n_outer = (n_outer, m_outer) if self.status_controller.mmad_mode == "gemv" else (m_outer, n_outer)
        out_axis = [m_outer, n_outer]
        if (self.status_controller.al1_attach_status == "full_load"
            or (self.status_controller.al1_attach_status == "c_l0c"
            and self.status_controller.c_l0c_attach_status == "full_load")):
            a_run_once = self.ALLOCATE_OFF
        if (self.status_controller.bl1_attach_status == "full_load"
            or (self.status_controller.bl1_attach_status == "c_l0c"
                and self.status_controller.c_l0c_attach_status == "full_load")):
            b_run_once = self.ALLOCATE_OFF
        if batch_double:
            if double_once != self.ALLOCATE_OFF:
                a_run_once, b_run_once = self._do_allocate_axis(a_run_once, b_run_once, out_axis)
        else:
            a_run_once, b_run_once = self._do_allocate_axis(a_run_once, b_run_once, out_axis)

        return a_run_once, b_run_once

    def _do_allocate_axis(self, a_run_once, b_run_once, out_axis):
        m_outer, n_outer = out_axis
        al1_ddr_to_l1_flag = (self.in_addr_type == 0 and (not self.status_controller.l1_fusion_and_l1_size_0)
                              and self.status_controller.input_l1_flag != 1)
        if a_run_once != self.ALLOCATE_OFF and al1_ddr_to_l1_flag:
            tensor_a_l1 = self.container.tensor_map.get("a_l1")
            self.sch[tensor_a_l1].allocate_at(self.sch[self.root_tensor], n_outer, run_once_axes=[n_outer])
            self.sch[tensor_a_l1].mem_unique()
        else:
            a_run_once = self.ALLOCATE_OFF
        # unzip donot support allocate_at
        if (b_run_once != self.ALLOCATE_OFF and (not self.status_controller.l1_fusion_and_l1_size_0)
            and (not self.status_controller.compress_flag)):
            tensor_b_l1 = self.container.tensor_map.get("b_l1")
            self.sch[tensor_b_l1].allocate_at(self.sch[self.root_tensor], m_outer, run_once_axes=[m_outer])
            self.sch[tensor_b_l1].mem_unique()
        else:
            b_run_once = self.ALLOCATE_OFF
        return a_run_once, b_run_once

    def _reorder_axis(self, enable_nbuffer, tensor_a_reuse_local, tensor_b_reuse_local):
        not_need_nbuffer = (tensor_a_reuse_local == self.ALLOCATE_OFF) and (tensor_b_reuse_local == self.ALLOCATE_OFF)
        if not enable_nbuffer or not_need_nbuffer:
            return
        axis_outer = self.sch_agent[self.root_tensor].get_active_scopes()
        if self.format_info.get("out") == "FRACTAL_NZ":
            m_outer = axis_outer[self.FRACTAL_NZ_M_INDEX]
            n_outer = axis_outer[self.FRACTAL_NZ_N_INDEX]
            l1_reuse_axis_outter = n_outer
            l1_reuse_axis_inner = m_outer
        else:
            m_outer = axis_outer[self.ND_M_INDEX]
            n_outer = axis_outer[self.ND_N_INDEX]
            l1_reuse_axis_outter = m_outer
            l1_reuse_axis_inner = n_outer
        if tensor_a_reuse_local == self.ALLOCATE_HALF and tensor_b_reuse_local != self.ALLOCATE_HALF:
            l1_reuse_axis_outter = m_outer
            l1_reuse_axis_inner = n_outer
        elif tensor_a_reuse_local != self.ALLOCATE_HALF and tensor_b_reuse_local == self.ALLOCATE_HALF:
            l1_reuse_axis_outter = n_outer
            l1_reuse_axis_inner = m_outer
        elif tensor_a_reuse_local == self.ALLOCATE_HALF and tensor_b_reuse_local == self.ALLOCATE_HALF:
            l1_reuse_axis_outter = m_outer
            l1_reuse_axis_inner = n_outer

        self._do_reorder_axis(l1_reuse_axis_outter, l1_reuse_axis_inner)

    def _do_reorder_axis(self, outer_axis, inner_axis):
        axis_outer = self.sch_agent[self.root_tensor].get_active_scopes()
        reorder_list = [outer_axis, inner_axis]
        if self.format_info.get("out") == "FRACTAL_NZ":
            reorder_list.append(axis_outer[self.FRACTAL_NZ_M0_INDEX])
            reorder_list.append(axis_outer[self.FRACTAL_NZ_N0_INDEX])
        if self.status_controller.have_batch:
            reorder_list.insert(0, axis_outer[0])
        self.sch[self.root_tensor].reorder(*reorder_list)

        if self.format_info.get("out") == "ND":
            tensor_at_res_stage = []
            attach_dict = self.sch_agent.get_attach_dict()
            for i, j in attach_dict.items():
                if j.op.name == self.root_tensor.op.name:
                    tensor_at_res_stage.append(i)
            tensor_at_res = []
            for tensor_stage in tensor_at_res_stage:
                for tensor in self.container.tensor_map.values():
                    if tensor is None:
                        continue
                    if tensor_stage.op.name == tensor.op.name:
                        tensor_at_res.append(tensor)
                        break
            for i in tensor_at_res:
                self.sch[i].compute_at(self.sch[self.res], inner_axis)

    def do_aub_storage_align(self):
        """
        solve the bank conflict of aub
        """
        # the data gap in ub
        gap_value = self.block_reduce
        tiling = self.tiling_work.tiling
        # solve bank conflict in aub/bub
        aub_k, aub_m, _, _ = tiling.get("AUB_shape")
        aub_m *= self.block_in
        # the data stride in ub
        a_align_value = (aub_m + gap_value) if self.status_controller.transpose_a else (aub_k + gap_value)
        if self.status_controller.transpose_a:
            self.status_controller.storage_m_bound_change = True
        else:
            self.status_controller.storage_ka_bound_change = True
        a_int82fp16 = self.container.tensor_map.get("a_int82fp16")
        a_normalize_ub = self.container.tensor_map.get("a_ub")
        # when the Inner axis is K and attach to C_gm, k_aligned value
        # may be larger than the tiling value.
        if self.status_controller.aub_attach_status == "c_gm" and not self.status_controller.transpose_a:
            self.status_controller.cgm_ka_storage_change = True
            max_k_bound = self.tiling_work.get_a_max_k_bound(self.container.tensor_map.get("a_l0a"))
            a_align_value = tvm.select(max_k_bound % self.THRESHOLD_DATA_NUM == 0,
                                       max_k_bound + gap_value, 1)
        if self.cache_tiling:
            a_align_value = self.cache_tiling.get("a_align_value")

        if a_int82fp16 is not None:
            self.sch[a_int82fp16].storage_align(a_int82fp16.op.axis[-2], a_align_value, 0)
        elif (a_normalize_ub.dtype == "float16") or (self.container.tensor_map.get("a_transpose") is not None):
            self.sch[a_normalize_ub].storage_align(a_normalize_ub.op.axis[-2], a_align_value, 0)
            if self.is_dynamic:
                a_ub_aligned = self.container.tensor_map.get("a_ub_aligned")
                a_ub_general = self.container.tensor_map.get("a_ub_general")
                self.sch[a_ub_aligned].storage_align(a_ub_aligned.op.axis[-2], a_align_value, 0)
                self.sch[a_ub_general].storage_align(a_ub_general.op.axis[-2], a_align_value, 0)

    def do_bub_storage_align(self):
        """
        solve bub bank conflict by storage align
        """
        # the data gap in ub
        gap_value = self.block_reduce
        tiling = self.tiling_work.tiling
        # solve bank conflict in aub/bub
        bub_k, bub_n, _, _ = tiling.get("BUB_shape")
        bub_n *= self.block_out
        # the data stride in ub
        b_align_value = (bub_k + gap_value) if self.status_controller.transpose_b else (bub_n + gap_value)
        if self.status_controller.transpose_b:
            self.status_controller.storage_kb_bound_change = True
        else:
            self.status_controller.storage_n_bound_change = True
        b_int82fp16 = self.container.tensor_map.get("b_int82fp16")
        b_normalize_ub = self.container.tensor_map.get("b_ub")
        # when the Inner axis is K and attach to C_gm, k_aligned value
        # may be larger than the tiling value.
        if self.status_controller.bub_attach_status == "c_gm" and self.status_controller.transpose_b:
            max_k_bound = self.tiling_work.get_b_max_k_bound(self.container.tensor_map.get("b_l0b"),
                                                             self.is_dynamic, self.dynamic_k)
            b_align_value = tvm.select(max_k_bound % self.THRESHOLD_DATA_NUM == 0, max_k_bound + gap_value, 1)
        if self.cache_tiling:
            b_align_value = self.cache_tiling.get("b_align_value")
        b_transpose = self.container.tensor_map.get("b_transpose")
        if b_int82fp16 is not None:
            self.sch[b_int82fp16].storage_align(b_int82fp16.op.axis[-2], b_align_value, 0)
        elif (b_normalize_ub.dtype == "float16") or (b_transpose is not None):
            self.sch[b_normalize_ub].storage_align(b_normalize_ub.op.axis[-2], b_align_value, 0)
            if self.is_dynamic:
                b_ub_aligned = self.container.tensor_map.get("b_ub_aligned")
                b_ub_general = self.container.tensor_map.get("b_ub_general")
                self.sch[b_ub_aligned].storage_align(b_ub_aligned.op.axis[-2], b_align_value, 0)
                self.sch[b_ub_general].storage_align(b_ub_general.op.axis[-2], b_align_value, 0)

    def _solve_bank_conflict(self):
        """
        solve bank conflict by storage_align
        if aub_k or bub_n bigger than THRESHOLD_DATA_NUM,
        use storage_align to solve bank conflict of aub/bub

        c_ub always conflict, must be use storage_align
        Input: None
        ---------------------------------
        Return: None
        """
        tiling = self.tiling_work.tiling
        params = {
            "container": self.container,
            "status_controller": self.status_controller,
            "format_out": self.format_info.get("out"),
            "format_a": self.format_info.get("a"),
            "format_b": self.format_info.get("b"),
            "cache_tiling": self.cache_tiling
        }
        a_ub_storage_align, b_ub_storage_align, c_ub_storage_align = self.buffer_checker.check_exceed_ub(tiling, params)
        # three point not consider now:
        # 1. Although tiling does not lead to bank conflict, may lead to bank conflict after align in dynamic
        # 2. same to 1.,tiling lead to bank conflict, may not lead to bank conflict after align in dynamic
        # 3. When the tensor is on the c_gm, bank conflict is not enabled
        a_ub_storage_align, b_ub_storage_align = self._disable_solve_bank_conflict_in_dynamic(
            a_ub_storage_align, b_ub_storage_align)
        if a_ub_storage_align or (self.cache_tiling and self.format_info.get("a") == "ND"):
            self.do_aub_storage_align()
        if b_ub_storage_align or (self.cache_tiling and self.format_info.get("b") == "ND"):
            self.do_bub_storage_align()
        # solve bank conflict in cub
        self._solve_bank_conflict_cub(c_ub_storage_align)

    def _solve_bank_conflict_cub(self, c_ub_storage_align):
        if c_ub_storage_align:
            before_c_gm = self.container.tensor_map.get("before_c_gm")
            cast_to_fp16 = self.container.tensor_map.get("cast_to_fp16")
            if (before_c_gm is not None) and (cast_to_fp16 is not None):
                c_gap_value = self.block_out * self.block_in * self.tiling_work.tiling.get(
                    "CUB_matrix")[1] + self.block_out
                self.sch[cast_to_fp16].storage_align(cast_to_fp16.op.axis[-4], c_gap_value, 0)
                if self.is_dynamic:
                    cur_bound_bound = c_gap_value * self.tiling_work.tiling.get("CUB_matrix")[0]
                    self.sch[cast_to_fp16].set_buffer_size(cur_bound_bound)
            else:
                c_gap_value = (self.block_out + 1) * self.block_in
                c_ub_fract = self.container.tensor_map.get("c_ub_fract")
                alpha_c_ub = self.container.tensor_map.get("alpha_c")
                if c_ub_fract not in self.container.compute_inline_list:
                    self.sch[c_ub_fract].storage_align(c_ub_fract.op.axis[-3], c_gap_value, 0)
                    if alpha_c_ub is not None:
                        self.sch[alpha_c_ub].storage_align(alpha_c_ub.op.axis[-3], c_gap_value, 0)

    def _disable_solve_bank_conflict_in_dynamic(self, a_ub_storage_align, b_ub_storage_align):
        if (self.status_controller.aub_attach_status == "c_gm"
            and not self.status_controller.transpose_a and self.is_dynamic):
            a_ub_storage_align = False
        if (self.status_controller.bub_attach_status == "c_gm"
            and self.status_controller.transpose_b and self.is_dynamic):
            b_ub_storage_align = False

        return a_ub_storage_align, b_ub_storage_align

    def _renew_block_dim(self):
        """
        if tail data small then 16(output=fp16) or 32(output=int32)
        close multi core
        """
        c_gm = self.container.tensor_map.get("c_gm")
        if self.status_controller.ops_data_flow_mode == "int82int32":
            multi_core_min_slice = 32
        else:
            multi_core_min_slice = 16

        if (c_gm.shape[1].value * self.OUTPUT_SIZE.get(
            self.status_controller.ops_data_flow_mode) < multi_core_min_slice):
            self.tiling_work.tiling["block_dim"] = [1, 1, 1, 1]

    def _do_compute_inline(self):
        self.container.elewise_compute_inline_list += self.container.compute_inline_list
        for tensor in self.container.elewise_compute_inline_list:
            if tensor in self.container.double_out_tensor:
                continue
            self.sch[tensor].compute_inline()

    def _set_requant_transfer_buffer_align(self):
        if not self.status_controller.requant_fusion:
            return
        requant_data_transfer = self.container.tensor_map.get("tensor_reform")
        unchanged = 1
        if self.status_controller.have_batch:
            self.sch[requant_data_transfer].buffer_align((unchanged, unchanged),
                                                        (unchanged, unchanged),
                                                        (unchanged, unchanged),
                                                        (unchanged, 16),
                                                        (unchanged, 16))
        else:
            self.sch[requant_data_transfer].buffer_align((unchanged, unchanged),
                                                        (unchanged, unchanged),
                                                        (unchanged, 16),
                                                        (unchanged, 16))
        return

    def _do_buffer_align_l0c(self):
        c_l0c = self.container.tensor_map.get("c_l0c")
        block_reduce = self.block_reduce
        block_out = self.block_out
        unchanged = 1
        if self.status_controller.mmad_mode in ("gevm", "gemv"):
            align_args = [
                (unchanged, unchanged),
                (unchanged, unchanged),
                (unchanged, block_out),
                (unchanged, block_out),
                (unchanged, unchanged),
                (unchanged, block_reduce)
            ]
        else:
            align_args = [
                (unchanged, unchanged),
                (unchanged, unchanged),
                (unchanged, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][0]),
                (unchanged, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][2]),
                (unchanged, unchanged),
                (unchanged, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][1])
            ]

        if self.status_controller.have_batch:
            align_args.insert(0, (unchanged, unchanged))
        if self.status_controller.split_k_axis_by_tiling:
            align_args.insert(0, (unchanged, unchanged))
            align_args[-1], align_args[-2] = align_args[-2], align_args[-1]
        self.sch[c_l0c].buffer_align(*align_args)

    def _set_compress_info(self, compress_tensor, compress_index, tile_k, tile_n, out_axis):
        """
        set weigths compress info
        """
        if out_axis is None:
            raise RuntimeError("compress index axis is None, it's error.")
        engine, ratios, channel, mode = tbe_platform_info.get_soc_spec("UNZIP")
        frac_size = 512

        index_shape = compress_index.shape
        dim_k = compress_tensor.shape[0].value
        dim_n = compress_tensor.shape[1].value

        tile_k_value = compress_tensor.op.attrs["tile_L1_k"]
        tile_n_value = compress_tensor.op.attrs["tile_L1_n"]
        block_dim_n = compress_tensor.op.attrs["block_dim_n"]

        block_size = self._get_compress_block_info(tile_k, tile_n)
        k_value = tile_k

        self.sch.set_var_range(tile_k_value, k_value, k_value)
        self.sch.set_var_range(tile_n_value, tile_n, tile_n)

        batch_dim, n_dim, m_dim, reduce_dim = self.tiling_work.tiling.get("block_dim")
        self.sch.set_var_range(block_dim_n, n_dim, n_dim)

        k_block_num = (dim_k + k_value - 1) // k_value
        n_block_num = (dim_n + tile_n - 1) // tile_n
        index_size = k_block_num * n_block_num

        tight_len = 2
        if mode == 1:
            tight_len = 8
        index_size = index_size * tight_len

        self.sch.set_var_range(index_shape[0], int(index_size), int(index_size))

        conflict = tvm.make.Call("int32", "tvm_tuple",
                                (block_size, index_size, mode, engine,
                                channel, ratios, k_value, tile_n, n_dim),
                                tvm.expr.Call.PureIntrinsic, None, 0)
        self.sch[compress_tensor].pragma(compress_tensor.op.axis[0],
                                    "json_info_compress_parameters", conflict)
        tensor_len = len(compress_tensor.shape)
        # transform data to continue by block_size
        al1_multi_m = 1
        if self.tiling_work.tiling.get("AL1_shape"):
            al1_multi_m = self.tiling_work.tiling.get("AL1_shape")[self.IDX_MULTI_M1]
        if batch_dim * n_dim * m_dim * reduce_dim > 1 or (isinstance(al1_multi_m, int) and al1_multi_m > 1):
            # unzip don't support set hoist_axis when multi core or al1_multi_m > 1
            self.sch[compress_tensor].emit_insn(compress_tensor.op.axis[tensor_len - self.FRACTAL_Z_LEN],
                                                "unzip",
                                                {"compress_mode": mode,
                                                "block_size": block_size})
        else:
            self.sch[compress_tensor].emit_insn(compress_tensor.op.axis[tensor_len - self.FRACTAL_Z_LEN],
                                                "unzip",
                                                {"compress_mode": mode,
                                                "block_size": block_size,
                                                "hoist_axis": out_axis})

    def _get_index_at_axis(self):
        axis = self.sch_agent[self.root_tensor].get_active_scopes()
        axis_n = axis[0]
        axis_m = axis[1]

        if self.format_info.get("out") == "ND":
            axis_n, axis_m = axis_m, axis_n

        block_dim_m = self.tiling_work.tiling.get("block_dim")[2]
        m_shape = self.container.tensor_map.get("a_l0a").shape[-4].value
        m_factor = (m_shape + block_dim_m - 1) // block_dim_m
        index_at_axis = axis_m if self.tiling_work.al1_tiling_m * 2 < m_factor else -1
        if index_at_axis != -1 and self.status_controller.bl1_attach_status == 'full_load':
            index_at_axis = self.sch_agent[self.container.tensor_map.get('b_l1')].get_active_scopes()[0]
        return index_at_axis

    def _choose_dma_copy_for_res(self):
        """choose dma copy pattern"""
        # with_transpose is the flag to use emit_insn dma_copy_matmul_transpose
        # this flag set from confusion_transpose_d
        real_res = self.res
        if self.status_controller.split_k_axis_by_tiling:
            real_res = self.container.tensor_map.get("c_gm")
        with_transpose = hasattr(real_res, "matmul_with_transpose")
        emit_insn_cmd = "dma_copy"
        if with_transpose:
            # get matrix axis shapes
            tensor_a_l0a = self.container.tensor_map.get("a_l0a")
            tensor_b_l0b = self.container.tensor_map.get("b_l0b")
            m_shape = self.dynamic_m * self.block_in
            n_shape = self.dynamic_n * self.block_out
            if not self.is_dynamic:
                m_shape = get_value(tensor_a_l0a.shape[-4]) * self.block_in
                n_shape = get_value(tensor_b_l0b.shape[-3]) * self.block_out
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_m", m_shape)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_n", n_shape)
            block_dim = self.tiling_work.tiling.get("block_dim")
            batch = self._get_batch_factors(tensor_a_l0a, tensor_b_l0b)
            n_factors, m_factors = block_dim[1], block_dim[2]
            m_l0_shape = self.tiling_work.tiling.get("AL0_matrix")[0] * self.block_in
            n_l0_shape = self.tiling_work.tiling.get("BL0_matrix")[1] * self.block_out

            cce_emitinsn_params.cceEmitParamsIns.insert_param("batch", batch)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_m_blk", m_factors)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_n_blk", n_factors)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_m_split", m_l0_shape)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_n_split", n_l0_shape)
            emit_insn_cmd = "dma_copy_matmul_transpose"

        out_insn_dict = {}
        if (self.schedule_mode == self.GENERAL_MODE and
            ((not self.status_controller.align_a) or (not self.status_controller.align_b))):
            out_insn_dict["no_overlap"] = 1
        elif self.schedule_mode == self.DYN_ALIGNED_MODE:
            out_insn_dict["no_overlap"] = 0
            out_insn_dict["gm_no_sync"] = 1
            out_insn_dict["map_policy"] = "2d"
        offset_res = 1 if self.status_controller.split_k_axis_by_tiling else 0
        if self.cache_tiling:
            res_axis = -2 if self.format_info.get("out") == "ND" else -4
            self.sch_agent[real_res].emit_insn(self.sch[real_res].leaf_iter_vars[res_axis], emit_insn_cmd,
                                               out_insn_dict)
        else:
            self._emit_insn_func(real_res, 0, emit_insn_cmd, insn_dict=out_insn_dict, mode=1, offset=offset_res)

        # temp
        overload_flag = True
        self._set_overload_flag(overload_flag, real_res, offset=offset_res)

    def _combine_cce_pragma(self, tensor, axis_list):
        """
        add pragma for emit_insn, in cachetiling scene, pass check whether can Combine multiple cce
        """
        id_val = tvm.make.Call("int32", "axis_group", [0, "append"], tvm.expr.Call.Extern, None, 0)
        for axis in axis_list:
            self.sch[tensor].pragma(axis, "axis_group", id_val)

    def _set_scope_buffer_type(self, header_tensors):
        gm_ub = None
        ele_header_ub_tensors = []
        if not self.container.fusion_ele:
            return gm_ub, ele_header_ub_tensors, {}

        in_out_tensor_map = {}
        self._gen_in_out_tensor_map(self.root_tensor, in_out_tensor_map)
        tensor_c_gm = self.container.tensor_map.get("c_gm")
        # multi output fusion with elementwise
        if isinstance(self.res_ori, list) and self.container.fusion_ele and tensor_c_gm in self.res_ori:
            self.status_controller.matmul_multi_output_flag = True
            gm_ub = self.sch.cache_read(tensor_c_gm, tbe_platform_info.scope_ubuf,
                                        in_out_tensor_map.get(tensor_c_gm))
            self.container.fusion_tensor_cub.append(gm_ub)
            self.container.fusion_tensor_cub.append(tensor_c_gm)
            self.sch[tensor_c_gm.op.input_tensors[0]].reused_by(gm_ub)

        tensor_ele_ub = []
        for ten_i in header_tensors:
            if in_out_tensor_map.get(ten_i)[0] not in self.container.matmul_tensors:
                ele_ub = self.sch.cache_read(ten_i, tbe_platform_info.scope_ubuf, in_out_tensor_map.get(ten_i))
                tensor_ele_ub.append(ele_ub)
                ele_header_ub_tensors.append(ele_ub)

        axpy_2_parent = self._get_elewise_ub_tensors(tensor_ele_ub)
        self.container.elemwise_tensors = tensor_ele_ub
        return gm_ub, ele_header_ub_tensors, axpy_2_parent

    def _set_overload_flag(self, overload_flag, flag_on_tensor, offset=0):
        """
        set overload flag
        """
        current_op = self.sch[flag_on_tensor]
        pragma_axis = self.sch_agent[flag_on_tensor].nlast_scopes(offset + 2)[0]
        if current_op is not None and pragma_axis is not None:
            if overload_flag:
                current_op.pragma(pragma_axis, "json_info_cache_read_mode", 0)
            else:
                current_op.pragma(pragma_axis, "json_info_cache_read_mode", 1)

    def _set_buffer_used_multi_custom(self, fused_num):
        all_tags = get_all_tags(self.res)
        if "dropout_broadcast" in all_tags:
            fused_num = 2.5
        return fused_num

    def _compute_buffer_used_multi(self):
        """
        Calculates the number of times the space used. The value is based on fp16.
        """
        buffer_reuse_dict = self.container.buffer_reuse_dict

        def enter(tensor_list, fix_dtype, not_count_list=None):
            """
            the enter to calculate buffer used multi
            not_count_list: tensors in this list do not use memory space.
            """
            counted_tensor_list = []
            fused_num = 0
            for tensor in tensor_list:
                if not_count_list is not None:
                    full_not_count_list = (self.container.elewise_compute_inline_list +
                        self.container.compute_inline_list + not_count_list)
                else:
                    full_not_count_list = (self.container.elewise_compute_inline_list +
                        self.container.compute_inline_list)
                if tensor in full_not_count_list:
                    continue
                if tensor in buffer_reuse_dict:
                    anothor_tensor = buffer_reuse_dict.get(tensor)
                    if anothor_tensor in counted_tensor_list:
                        continue
                    a_dtype_width = self.DTYPE_WIDTH_MAP.get(tensor.dtype)
                    b_dtype_width = self.DTYPE_WIDTH_MAP.get(anothor_tensor.dtype)
                    cur_dtype_num = a_dtype_width if a_dtype_width > b_dtype_width else b_dtype_width
                    cur_tensor = tensor if a_dtype_width > b_dtype_width else anothor_tensor
                    small_one_tensor = anothor_tensor if a_dtype_width > b_dtype_width else tensor
                    # small_one is conuted too
                    counted_tensor_list.append(small_one_tensor)
                else:
                    cur_dtype_num = self.DTYPE_WIDTH_MAP.get(tensor.dtype)
                    cur_tensor = tensor
                if cur_tensor in counted_tensor_list:
                    continue
                counted_tensor_list.append(cur_tensor)
                fused_num += cur_dtype_num

            fused_num = fused_num / self.DTYPE_WIDTH_MAP.get(fix_dtype) - 1
            return fused_num

        if self.container.tensor_map.get("a_ub") is not None:
            a_not_count_list = (
                [self.container.tensor_map.get("a_ub_aligned"),
                 self.container.tensor_map.get("a_ub_general")] if self.is_dynamic else None)
            a_fused_num = int(enter(self.container.tensors_in_aub, self.container.tensor_map.get("a_ub").dtype,
                                    a_not_count_list) * self.PRE_UB_MULTIPLIER)
        else:
            a_fused_num = 0
        if self.container.tensor_map.get("b_ub") is not None:
            b_not_count_list = (
                [self.container.tensor_map.get("b_ub_aligned"),
                self.container.tensor_map.get("b_ub_general")] if self.is_dynamic else None)
            b_fused_num = int(enter(self.container.tensors_in_bub, self.container.tensor_map.get("b_ub").dtype,
                                    b_not_count_list) * self.PRE_UB_MULTIPLIER)
        else:
            b_fused_num = 0

        # if not fusion
        if self.container.tensor_map.get("c_gm") == self.res:
            c_fused_num = int(enter(self.container.tensors_in_cub, self.res.dtype))
        else:
            # elewise fusion not support alpha and beta and C now
            c_ub_fract = self.container.tensor_map.get("c_ub_fract")
            c_ub_cast_to_fp16 = self.container.tensor_map.get("cast_to_fp16")
            c_ub = c_ub_cast_to_fp16 if c_ub_cast_to_fp16 is not None else c_ub_fract
            nz_to_nd = self.container.tensor_map.get("nz_to_nd")
            ub_res_byte = self._get_scope_byte_size(c_ub, nz_to_nd)
            ub_res_byte = self._get_ub_res_byte(self._get_out_tensors_width,
                                                self.container.dequant_activation_tensor,
                                                self.container.fusion_ele, self.res,
                                                ub_res_byte)
            c_fused_num = ub_res_byte // self.DTYPE_WIDTH_MAP.get(self.res.dtype) - 1
            c_fused_num = self._set_buffer_used_multi_custom(c_fused_num)
        return a_fused_num, b_fused_num, c_fused_num

    def _get_scope_byte_size(self, tensor_ub, tensor_ub_fract):
        """
        get unit byte size for buffer scope
        """
        # Calculating tiling para need a_ub info
        ub_byte = 0
        if tensor_ub is not None:
            ub_byte = int(self.DTYPE_WIDTH_MAP.get(tensor_ub.dtype) * 2)
            if tensor_ub_fract is not None:
                ub_byte = ub_byte * 2
        return ub_byte

    def _get_ub_res_byte(self, _get_out_tensors_width, dequant_activation_tensor,
                         fusion_ele, res, ub_res_byte):
        """
        calculate res ub byte by width
        """
        if fusion_ele or dequant_activation_tensor:
            width = _get_out_tensors_width(res)
            res_byte = self.DTYPE_WIDTH_MAP.get(res.dtype)
            if ub_res_byte < width * res_byte:
                ub_res_byte = width * res_byte
        return ub_res_byte

    def _get_out_tensors_width(self, out_tensor):
        """
        get max width for tensors

        Parameters
        ----------
        out_tensor : tensor
            need to count all its input tensorss

        Return
        ------
            max width for tensors
        """
        in_out_tensor_map = {}
        self._gen_in_out_tensor_map(out_tensor, in_out_tensor_map)
        stack = [out_tensor]
        width = len(stack)
        visited_list = []
        tmp_stack = stack
        tensor_c_gm = self.container.tensor_map.get("c_gm")
        matmul_end_tensor = tensor_c_gm.op.input_tensors[0]
        while tmp_stack:
            for tens in tmp_stack:
                if "broadcast" in tens.op.tag:
                    stack.remove(tens)
                    continue
                if tens in in_out_tensor_map:
                    def calc_width_mid(width):
                        """
                        get mid tensor width
                        """
                        all_out = True
                        for out_ten in in_out_tensor_map.get(tens):
                            if out_ten not in visited_list:
                                all_out = False
                        if all_out and (tens not in visited_list):
                            visited_list.append(tens)
                            stack.remove(tens)
                            # the shape of deq_scale is very small
                            if tens.op.tag not in self.DEQ_SCALE_CHILD_LIST:
                                for in_ten in tens.op.input_tensors:
                                    if in_ten not in stack and \
                                            in_ten != matmul_end_tensor:
                                        stack.append(in_ten)
                            else:
                                stack.append(tens.op.input_tensors[0])
                            width_local = 0
                            cast_flag = False
                            for ele in stack:
                                width_local = width_local + self.DTYPE_WIDTH_MAP.get(ele.dtype)
                                if self.DTYPE_WIDTH_MAP.get(ele.dtype) == 2:
                                    cast_flag = True
                            if width_local == 2 and cast_flag:
                                width_local = 3
                            if width_local > width:
                                width = width_local
                        return width

                    width = calc_width_mid(width)

                else:
                    def calc_width_tail(width):
                        """
                        get tail tesnor width
                        """
                        visited_list.append(tens)
                        stack.remove(tens)
                        for in_ten in tens.op.input_tensors:
                            if in_ten not in stack and in_ten != matmul_end_tensor:
                                stack.append(in_ten)
                        width_local = 0
                        cast_flag = False
                        for ele in stack:
                            width_local = width_local + self.DTYPE_WIDTH_MAP.get(ele.dtype)
                            if self.DTYPE_WIDTH_MAP.get(ele.dtype) == 2:
                                cast_flag = True
                        if width_local == 2 and cast_flag:
                            width_local = 3
                        if width_local > width:
                            width = width_local
                        return width

                    width = calc_width_tail(width)

            tmp_stack = []
            for ele in stack:
                tmp_stack.append(ele)
        return width

    def _mem_process(self):
        if self.is_dynamic:
            self.sch.sequential_malloc(tbe_platform_info.scope_cbuf)
            self.sch.sequential_malloc(tbe_platform_info.scope_ca)
            self.sch.sequential_malloc(tbe_platform_info.scope_cb)
            self.sch.sequential_malloc(tbe_platform_info.scope_cc)
            self.sch.sequential_malloc(tbe_platform_info.scope_ubuf)

            # get l1
            if self.format_info.get("a") == "ND":
                aub_storage_bound, aub_fract_storage_bound = self._get_aub_bound()
                # a_ub is normalized so the storage used for (M,K) is the same as (M1, K1 , M0, K0)
                self.sch[self.container.tensor_map.get("a_ub")].set_buffer_size(aub_storage_bound)
                self.sch[self.container.tensor_map.get("a_ub_aligned")].set_buffer_size(aub_storage_bound)
                self.sch[self.container.tensor_map.get("a_ub_general")].set_buffer_size(aub_storage_bound)
                self.sch[self.container.tensor_map.get("a_ub_fract")].set_buffer_size(aub_fract_storage_bound)
            if self.format_info.get("b") == "ND":
                bub_storage_bound, bub_fract_storage_bound = self._get_bub_bound()
                self.sch[self.container.tensor_map.get("b_ub")].set_buffer_size(bub_storage_bound)
                self.sch[self.container.tensor_map.get("b_ub_aligned")].set_buffer_size(bub_storage_bound)
                self.sch[self.container.tensor_map.get("b_ub_general")].set_buffer_size(bub_storage_bound)
                self.sch[self.container.tensor_map.get("b_ub_fract")].set_buffer_size(bub_fract_storage_bound)
            self.sch[self.container.tensor_map.get("a_l1")].set_buffer_size(self._get_al1_bound())
            self.sch[self.container.tensor_map.get("b_l1")].set_buffer_size(self._get_bl1_bound())

            if self.cache_tiling:
                cub_bound = self.cache_tiling.get("cub_n1") * self.cache_tiling.get(
                    "m_l0") * self.block_in * self.block_out
                cl0_bound = cub_bound * self.cache_tiling.get("n_ub_l0_time")
                self.sch[self.container.tensor_map.get("c_l0c")].set_buffer_size(cl0_bound)
                self.sch[self.container.tensor_map.get("c_ub_fract")].set_buffer_size(cub_bound)
                if self.format_info.get("out") == "ND":
                    self.sch[self.container.tensor_map.get("cast_to_fp16")].set_buffer_size(cub_bound)
                    self.sch[self.container.tensor_map.get("nz_to_nd")].set_buffer_size(cub_bound)

            # mem_unique
            self.sch[self.container.tensor_map.get("a_l1")].mem_unique()
            self.sch[self.container.tensor_map.get("b_l1")].mem_unique()
            self.sch[self.container.tensor_map.get("a_l0a")].mem_unique()
            self.sch[self.container.tensor_map.get("b_l0b")].mem_unique()
            # tensor is used_by can't mem_unique; tensor is not reused_by must be unique
            c_ub = self.res.op.input_tensors[0]
            if not self.format_info.get("out") == "ND":
                self.sch[c_ub].mem_unique()
            bias_ub = self.container.tensor_map.get("bias_ub")
            if bias_ub is not None and not self.status_controller.need_init_bias:
                self.sch[bias_ub].mem_unique()
            else:
                self.sch[self.container.tensor_map.get("c_l0c")].mem_unique()

    def _get_max_k_bound(self):
        """
        This function is used to get the maximum k bound, which will be used in the
        following calculation to solve bank conflict and to set storage bound.
        """
        a_matrix_dim = [get_value(i) for i in self.container.tensor_map.get("a_l0a").shape]
        k_bound_tiling = (int_ceil_div(a_matrix_dim[-3], self.tiling_work.tiling.get("AL0_matrix")[1])
            * self.tiling_work.tiling.get("AL0_matrix")[1] * self.block_reduce)
        if self.cache_tiling:
            k_bound_tiling = a_matrix_dim[-3] * self.block_reduce
            if self.status_controller.split_k_axis_by_tiling:
                k_bound_tiling = tvm.div(k_bound_tiling, self.tiling_work.tiling.get("block_dim")[self.IDX_K_DIM])
        return k_bound_tiling

    def _get_max_m_bound(self):
        """
        This function is used to get the maximum m bound, which will be used in the
        following calculation to set storage bound.
        """
        a_matrix_dim = [get_value(i) for i in self.container.tensor_map.get("a_l0a").shape]
        if self.cache_tiling:
            m_bound = (self.cache_tiling.get("m_single_core") * self.cache_tiling.get("m_al1") *
                       self.cache_tiling.get("m_l0") * self.block_in)
        else:
            if self.tiling_work.tiling.get("block_dim")[2] == 1:
                m_bound = a_matrix_dim[-4] * self.block_in
            else:
                m_parts = int_ceil_div(a_matrix_dim[-4], self.tiling_work.tiling.get("CL0_matrix")[1])
                m_factors = int_ceil_div(m_parts, self.tiling_work.tiling.get("block_dim")[2])
                m_bound = m_factors * self.tiling_work.tiling.get("CL0_matrix")[1] * self.block_in
        return m_bound

    def _get_max_n_bound(self):
        """
        This function is used to get the maximum n bound, which will be used in the
        following calculation to set storage bound.
        """
        b_matrix_dim = [get_value(i) for i in self.container.tensor_map.get("b_l0b").shape]
        if self.cache_tiling:
            n_bound = (self.cache_tiling.get("n_single_core") * self.cache_tiling.get("n_bl1") *
                       self.cache_tiling.get("n_ub_l0_time") * self.cache_tiling.get("cub_n1") * self.block_out)
        else:
            if self.tiling_work.tiling.get("block_dim")[1] == 1:
                n_bound = b_matrix_dim[-3] * self.block_out
            else:
                n_parts = int_ceil_div(b_matrix_dim[-3], self.tiling_work.tiling.get("CL0_matrix")[0])
                n_factors = int_ceil_div(n_parts, self.tiling_work.tiling.get("block_dim")[1])
                n_bound = n_factors * self.tiling_work.tiling.get("CL0_matrix")[0] * self.block_out
        return n_bound

    def _get_aub_bound(self):
        gap_value = self.block_reduce
        m_bound = self.tiling_work.aub_tiling_m * self.block_in
        m_bound = (m_bound + gap_value) if self.status_controller.storage_m_bound_change else m_bound
        if self.status_controller.aub_attach_status == "c_gm":
            max_k_bound = self.tiling_work.get_a_max_k_bound(self.container.tensor_map.get("a_l0a"))
            k_bound_not_align = max_k_bound
            # If having bank conflict
            k_bound = tvm.select(
                max_k_bound % self.THRESHOLD_DATA_NUM == 0,
                max_k_bound + gap_value, max_k_bound) if self.status_controller.cgm_ka_storage_change else max_k_bound
        else:
            k_bound = self.tiling_work.aub_tiling_k
            k_bound_not_align = self.tiling_work.aub_tiling_k
            k_bound = (k_bound + gap_value) if self.status_controller.storage_ka_bound_change else k_bound
        aub_bound = m_bound * k_bound
        # aub_fract needn't solve bank conflict
        aub_fract_bound = self.tiling_work.aub_tiling_m * self.block_in * k_bound_not_align
        if self.cache_tiling:
            aub_bound = self.cache_tiling.get("aub_align_bound")
            aub_fract_bound = self.cache_tiling.get("k_aub") * self.cache_tiling.get("m_aub") * \
                self.block_in * self.block_reduce
        return aub_bound, aub_fract_bound

    def _get_bub_bound(self):
        gap_value = self.block_reduce
        n_bound = self.tiling_work.bub_tiling_n * self.block_out
        n_bound = (n_bound + gap_value) if self.status_controller.storage_n_bound_change else n_bound
        if self.status_controller.bub_attach_status == "c_gm":
            max_k_bound = self.tiling_work.get_b_max_k_bound(self.container.tensor_map.get("b_l0b"),
                                                             self.is_dynamic, self.dynamic_k)
            k_bound_not_align = max_k_bound
            # If having bank conflict
            k_bound = tvm.select(
                max_k_bound % self.THRESHOLD_DATA_NUM == 0,
                max_k_bound + gap_value, max_k_bound) if self.status_controller.cgm_kb_storage_change else max_k_bound
        else:
            k_bound = self.tiling_work.bub_tiling_k
            k_bound_not_align = self.tiling_work.bub_tiling_k
            k_bound = (k_bound + gap_value) if self.status_controller.storage_kb_bound_change else k_bound
        bub_bound = n_bound * k_bound
        # aub_fract needn't solve bank conflict
        bub_fract_bound = self.tiling_work.bub_tiling_n * self.block_out * k_bound_not_align
        if self.cache_tiling:
            bub_bound = self.cache_tiling.get("bub_align_bound")
            bub_fract_bound = self.cache_tiling.get("k_bub") * self.cache_tiling.get("n_bub") * \
                self.block_out * self.block_reduce
        return bub_bound, bub_fract_bound

    def _get_al1_bound(self):
        if self.tiling_work.tiling.get("AL1_shape") and self.status_controller.al1_attach_status != "full_load":
            m_bound = self.tiling_work.tiling.get("AL1_shape")[1] * self.tiling_work.tiling.get(
                "CL0_matrix")[1] * self.block_in
            if self.status_controller.al1_attach_status == "c_gm":
                k_bound = self._get_max_k_bound()
            else:
                k_bound = self.tiling_work.tiling.get("AL1_shape")[0]
        else:
            k_bound = self._get_max_k_bound()
            m_bound = self._get_max_m_bound()

        al1_bound = m_bound * k_bound
        return al1_bound

    def _get_bl1_bound(self):
        if self.tiling_work.tiling.get("BL1_shape") and self.status_controller.bl1_attach_status != "full_load":
            n_bound = self.tiling_work.tiling.get("BL1_shape")[1] * self.tiling_work.tiling.get(
                "CL0_matrix")[0] * self.block_out
            if self.status_controller.bl1_attach_status == "c_gm":
                k_bound = self._get_max_k_bound()
            else:
                k_bound = self.tiling_work.tiling.get("BL1_shape")[0]
        else:
            k_bound = self._get_max_k_bound()
            n_bound = self._get_max_n_bound()
        bl1_bound = n_bound * k_bound
        return bl1_bound

    def _set_continuous_axis(self):
        """ add pragma for pass to check whether or not the address is continus and combine cce orders
        """
        if self.cache_tiling and self.format_info.get("a") != "FRACTAL_NZ":
            a_l0a = self.container.tensor_map.get("a_l0a")
            axis_list = [self.sch[a_l0a].leaf_iter_vars[-3], self.sch[a_l0a].leaf_iter_vars[-4]]
            self._combine_cce_pragma(a_l0a, axis_list)
        if self.cache_tiling:
            b_l0b = self.container.tensor_map.get("b_l0b")
            axis_list = [self.sch[b_l0b].leaf_iter_vars[-3], self.sch[b_l0b].leaf_iter_vars[-4]]
            self._combine_cce_pragma(b_l0b, axis_list)
        if self.format_info.get("out") == "ND" and self.is_dynamic:
            real_res = self.res
            if self.status_controller.split_k_axis_by_tiling:
                real_res = self.container.tensor_map.get("c_gm")
            axis_list = [self.sch[real_res].leaf_iter_vars[-2], self.sch[real_res].leaf_iter_vars[-1]]
            self._combine_cce_pragma(real_res, axis_list)

    def _auto_tiling(self, aub_num, bub_num, cub_num):
        # a_ub_byte b_ub_byte ub_res_byte l1a_byte l1b_byte l0a_byte l0b_byte l0c_byte
        bytes_info = self._get_bytes_info(aub_num, bub_num, cub_num)
        m_shape = self.container.tensor_map.get("a_l0a").shape[self.FRACTAL_Z_M_INDEX].value * self.block_in
        mkn_shape = (m_shape,
                     self.container.tensor_map.get("a_l0a").shape[self.FRACTAL_Z_KA_INDEX].value * self.block_reduce,
                     self.container.tensor_map.get("b_l0b").shape[self.FRACTAL_Z_N_INDEX].value * self.block_out)
        schedule_info_dict = {}
        schedule_info_dict["l1_fusion_type"] = self.l1_fusion_type
        schedule_info_dict["dequant_fusion"] = self.status_controller.dequant_fusion
        schedule_info_dict["date_transfer_fusion"] = (
            self.status_controller.quant_fusion or self.status_controller.requant_fusion)
        schedule_info_dict["mmad_mode"] = self.status_controller.mmad_mode
        schedule_info_dict["is_b_nz"] = self.format_info.get("b") == "FRACTAL_NZ"
        schedule_info_dict["block_in"] = self.block_in
        schedule_info_dict["block_out"] = self.block_out
        schedule_info_dict["block_reduce"] = self.block_reduce
        schedule_info_dict["b_trans"] = self.status_controller.transpose_b
        if self.container.custom_block_dim != []:
            schedule_info_dict["custom_block_dim"] = self.container.custom_block_dim

        l0c_shape = self.container.tensor_map.get("c_l0c").shape
        batch_shape = 0
        if len(l0c_shape) == self.FRACTAL_LEN_WITH_BATCH:
            batch_shape = get_value(l0c_shape[0])

        compute_tiling = ComputeTiling(bytes_info, mkn_shape, schedule_info_dict, batch_shape)
        return compute_tiling.compute_tiling_enter()

    def _get_bytes_info(self, aub_num, bub_num, cub_num):
        ub_fused_num_multi = 10
        double_multi = 2
        a_ub_byte = 0
        if self.container.tensor_map.get("a_ub") is not None:
            a_ub_byte = ((aub_num // ub_fused_num_multi + 1)
                * int(self.DTYPE_WIDTH_MAP.get(self.container.tensor_map.get("a_placehold").dtype) * double_multi))
        b_ub_byte = 0
        if self.container.tensor_map.get("b_ub") is not None:
            b_ub_byte = ((bub_num // ub_fused_num_multi + 1)
                * int(self.DTYPE_WIDTH_MAP.get(self.container.tensor_map.get("b_placehold").dtype) * double_multi))
        ub_res_byte = (cub_num + 1) * int(self.DTYPE_WIDTH_MAP.get(self.root_tensor.dtype) * double_multi)
        l1a_byte = int(self.DTYPE_WIDTH_MAP.get(self.container.tensor_map.get("a_l1").dtype) * double_multi)
        l1b_byte = int(self.DTYPE_WIDTH_MAP.get(self.container.tensor_map.get("b_l1").dtype) * double_multi)
        l0a_byte = int(self.DTYPE_WIDTH_MAP.get(self.container.tensor_map.get("a_l0a").dtype) * double_multi)
        l0b_byte = int(self.DTYPE_WIDTH_MAP.get(self.container.tensor_map.get("b_l0b").dtype) * double_multi)
        l0c_byte = int(self.DTYPE_WIDTH_MAP.get(self.container.tensor_map.get("c_l0c").dtype) * double_multi)
        return [a_ub_byte, b_ub_byte, ub_res_byte, l1a_byte, l1b_byte,
                l0a_byte, l0b_byte, l0c_byte]

    def _handle_tbe_compile_para(self):
        tbe_compile_para = self.tiling_work.tiling.get("tbe_compile_para")
        if tbe_compile_para:
            _, tbe_sch_control_para = parse_tbe_compile_para(tbe_compile_para)
            preload_flag = tbe_sch_control_para.get("preload")
            if preload_flag and (self.tiling_work.tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == 2):
                for tensor in self.container.tensors_in_l0c:
                    self.sch[tensor].preload()


class ComputeTiling:
    """
    get tiling by compute
    """
    DOUBLE_VALUE = 2
    CORE_NUM_THRITY = 30
    CORE_NUM_THRITY_TWO = 32
    CORE_NUM_EIGHT = 8
    MKN_M_INDEX = 0
    MKN_K_INDEX = 1
    MKN_N_INDEX = 2

    def __init__(self, bytes_info, mkn_shape, schedule_info_dict, batch_shape=0):
        self.mkn_shape = mkn_shape
        self.batch_shape = batch_shape
        self.bytes_info = bytes_info
        self.l1_fusion_type = schedule_info_dict.get("l1_fusion_type")
        self.dequant_fusion = schedule_info_dict.get("dequant_fusion")
        self.date_transfer_fusion = schedule_info_dict.get("date_transfer_fusion")
        self.mmad_mode = schedule_info_dict.get("mmad_mode")
        self.is_b_nz = schedule_info_dict.get("is_b_nz")
        self.block_in = schedule_info_dict.get("block_in")
        self.block_out = schedule_info_dict.get("block_out")
        self.block_reduce = schedule_info_dict.get("block_reduce")
        self.b_trans = schedule_info_dict.get("b_trans")
        self.custom_block_dim = schedule_info_dict.get("custom_block_dim", [])

    @staticmethod
    def _get_l1fusion_device_core_num(is_l1fusion):
        """
        get the device core num
        :param is_l1fusion: is l1 fusion
        :return: device core num
        """
        if is_l1fusion:
            device_core_num = 1
        else:
            device_core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
        return device_core_num

    @staticmethod
    def _get_core_map():
        """
        the knowledge of matmul schedule core tiling
        """
        shape_map = {(1024, 20480, 1024): (4, 7),
                    (4096, 20480, 1024): (7, 4),
                    (20480, 4096, 1024): (15, 2),
                    (1024, 20480, 4096): (4, 7),
                    (1024, 12288, 4096): (4, 7),
                    (4096, 12288, 1024): (7, 4)
                    }
        return shape_map

    @staticmethod
    def get_shape_map():
        """
        the knowledge of matmul schedule tiling
        """
        shape_map = {(1664, 4096, 1024, -1, 2): "176_320_176_176_80_176_2_2",
                    (1664, 4096, 1024, -1, 4): "176_320_176_176_80_176_2_2",
                    (1664, 1024, 4096, -1, 2): "240_512_128_240_64_128_2_2",
                    (1664, 1024, 4096, -1, 8): "240_512_64_240_64_64_2_2",
                    (1664, 16, 1024, -1, 2): "832_16_128_832_16_32_1_2",
                    (1664, 1024, 1024, -1, 2): "240_512_128_240_64_128_2_2",
                    (1664, 1024, 1024, -1, 4): "240_512_128_240_64_128_2_2",
                    (832, 4096, 1024, -1, 2): "176_320_176_176_80_176_2_2",
                    (832, 4096, 1024, -1, 4): "176_320_176_176_80_176_2_2",
                    (832, 1024, 4096, -1, 2): "240_512_128_240_64_128_2_2",
                    (832, 1024, 4096, -1, 8): "240_512_64_240_64_64_2_2",
                    (832, 1024, 1024, -1, 2): "240_512_128_240_64_128_2_2",
                    (832, 1024, 1024, -1, 4): "240_512_128_240_64_128_2_2",
                    (832, 16, 1024, -1, 2): "832_16_128_832_16_32_1_2",
                    (1280, 16, 768, -1, 2): "640_16_192_640_16_48_1_2",
                    (1280, 768, 768, -1, 2): "336_384_96_336_48_96_2_2",
                    (320, 64, 320, -1, 2): "320_64_192_320_48_96_1_2",
                    (1280, 768, 3072, -1, 2): "336_384_96_336_48_96_2_2",
                    (1280, 16, 768, -1, 2): "640_16_192_640_16_48_1_2",
                    (320, 64, 320, -1, 2): "320_64_192_320_48_96_1_2",
                    (1280, 768, 768, -1, 4): "320_384_96_320_48_96_2_2",
                    (16, -1, 4096, 0, 2): "16_16_1024_16_16_1024_2_2"
                    }

        return shape_map

    @staticmethod
    def get_mini_frac_shape_map():
        """
        the knowledge of matmul schedule tiling
        """
        shape_map = {(304, -1, 4096, -1, 2): "304_80_192_304_80_192_2_2",
                    (304, -1, 4096, -1, 4): "304_80_192_304_80_192_2_2",
                    (304, -1, 4096, -1, 6): "304_80_192_304_80_192_2_2"
                    }

        return shape_map

    @staticmethod
    def get_cloud_shape_map(core_num, core_num_thrity):
        """
        the knowledge of matmul schedule tiling
        """
        if core_num == core_num_thrity:
            shape_map = {(1024, 20480, 1024, -1, 4): "256_512_160_256_64_160_2_2",
                        (12288, 1024, 4096, -1, 2): "208_512_128_208_64_128_2_2",
                        (12288, 4096, 1024, -1, 2): "208_512_128_208_64_128_2_2",
                        (1024, 12288, 1024, -1, 4): "208_512_176_208_64_176_2_2",
                        (12288, 1024, 1024, -1, 2): "208_512_128_208_64_128_2_2",
                        (12288, 1024, 1024, -1, 4): "208_512_128_208_64_128_2_2"
                        }
        else:
            shape_map = {(18432, 1024, 1024, 1, 2): "144_512_256_144_64_256_2_2"
            }

        return shape_map

    @staticmethod
    def get_mdc_shape_map():
        """
        the knowledge of matmul schedule tiling
        """
        shape_map = {(1024, 768, 768, -1, 4): "256_768_384_128_256_128_2_2",
                    (1024, 768, 3072, 1, 6): "128_384_32_128_96_32_3_2",
                    (2048, 768, 3072, 1, 6): "128_384_32_128_96_32_3_2",
                    (1024, 768, 768, -1, 6): "256_768_384_128_256_128_2_2",
                    (512, 768, 768, 1, 2): "128_96_192_128_96_192_2_2"
                    }

        return shape_map

    @staticmethod
    def _get_refresh_core_factors(m_factors, n_factors, batch):
        """
        get refresh
        """
        if batch > 1:
            m_factors = 1
            n_factors = 1

        return m_factors, n_factors

    @staticmethod
    def _get_special_l0_factor(src_shape, m_l0_shape, k_l0_shape, n_l0_shape):
        """
        get temp factors
        """
        m_shape = src_shape[0]
        k_shape = src_shape[1]
        n_shape = src_shape[2]
        if m_shape * n_shape * k_shape == m_l0_shape * n_l0_shape * k_l0_shape and \
                m_l0_shape != 1:
            m_l0_shape = int((m_l0_shape // 2))
            if int((m_l0_shape % 16)) != 0:
                m_l0_shape = int((m_l0_shape + 15) // 16 * 16)

        src_shape = [m_shape, k_shape, n_shape]
        if src_shape == [256, 64, 256]:
            m_l0_shape = 256
            k_l0_shape = 64
            n_l0_shape = 128
        elif src_shape == [256, 256, 64]:
            m_l0_shape = 64
            k_l0_shape = 256
            n_l0_shape = 64
        return m_l0_shape, k_l0_shape, n_l0_shape

    def compute_tiling_enter(self):
        """
        the enter of compute tiling
        Return: tiling, type dict
        """
        # compute the core num of m and n
        m_factors, n_factors = self._get_perfect_core_num()
        m_factors, n_factors = self._get_knowledge_core(self.mkn_shape, m_factors, n_factors)
        k_factors = 1
        if self.custom_block_dim != []:
            m_factors, k_factors, n_factors = self.custom_block_dim
        m_var = [self.mkn_shape[self.MKN_M_INDEX], m_factors]
        n_var = [self.mkn_shape[self.MKN_N_INDEX], n_factors]
        k_var = [self.mkn_shape[self.MKN_K_INDEX], k_factors]
        batch, core_inner_m, core_inner_n, core_inner_k = self._get_batch_factors(m_var, n_var, k_var)
        m_factors, n_factors = self._get_refresh_core_factors(m_factors, n_factors, batch)
        ub_reserve_buff = 0
        if self.dequant_fusion:
            # quant parameter is fixed float16, it's 2 bytes
            # just support scalar now, not support vector yet
            ub_reserve_buff = tbe_platform.BLOCK_OUT * 2

        n_cut_even = self._is_need_n_cut_even(core_inner_n)
        a_ub_byte, b_ub_byte, ub_res_byte, l1a_byte, l1b_byte, l0a_byte, l0b_byte, l0c_byte = self.bytes_info
        ub_res_byte = int(math.ceil(ub_res_byte))
        if self.mmad_mode != "gemm":
            core_inner_m = 1
        get_tiling_shape = tvm.get_global_func("cce.matmul_tiling_gen")
        tiling_shape = get_tiling_shape(core_inner_m, core_inner_k, core_inner_n,
                                        a_ub_byte, b_ub_byte, l1a_byte, l1b_byte, l0a_byte,
                                        l0b_byte, l0c_byte, ub_res_byte, ub_reserve_buff,
                                        n_cut_even, int(self.is_b_nz))

        m_shape, k_shape, n_shape = self.mkn_shape
        b_trans = self.b_trans
        shape_tiling_args = (m_shape, k_shape, n_shape, b_trans, ub_res_byte)
        tiling_shape = self._get_knowledge_tiling(shape_tiling_args, self.is_b_nz, tiling_shape)
        tiled_shape = tiling_shape.split('_')
        m_l1_shape, k_l1_shape, n_l1_shape, m_l0_shape, k_l0_shape, n_l0_shape = [int(i) for i in tiled_shape[:6]]

        after_multicore_shape = [core_inner_m, core_inner_k, core_inner_n]
        m_l0_shape, k_l0_shape, n_l0_shape = self._get_special_l0_factor(after_multicore_shape, m_l0_shape,
            k_l0_shape, n_l0_shape)
        m_l1_shape, k_l1_shape, n_l1_shape = m_l0_shape, k_l0_shape, n_l0_shape

        # compute L1 to L0 tiling params
        m_l0_tile = (m_l0_shape + self.block_in - 1) // self.block_in
        k_l0_tile = (k_l0_shape + self.block_reduce - 1) // self.block_reduce
        n_l0_tile = (n_l0_shape + self.block_out - 1) // self.block_out

        # compute GM to L1 tiling params
        m_l1_tile = (m_l1_shape + self.block_in - 1) // self.block_in
        k_l1_tile = (k_l1_shape + self.block_reduce - 1) // self.block_reduce
        n_l1_tile = (n_l1_shape + self.block_out - 1) // self.block_out

        is_l1fusion = self.l1_fusion_type in (0, 1)
        core_num = self._get_l1fusion_device_core_num(is_l1fusion)
        batch_factor = 1
        if batch > 1:
            batch_factor = batch
            if batch > core_num:
                batch_factor = core_num

        tiling_factors = [
            batch_factor, m_factors, n_factors, k_factors,
            m_l1_tile, k_l1_tile, n_l1_tile,
            m_l0_tile, k_l0_tile, n_l0_tile]
        double_buffers = self._get_double_buffer(tiling_factors, l0c_byte)
        return self._assembly_tiling(tiling_factors, double_buffers)

    def _assembly_tiling(self, tiling_factors, double_buffers):
        block_reduce = self.block_reduce
        block_in = self.block_in
        block_out = self.block_out
        [batch_factor, m_factor, n_factor, k_factor,
        m_l1_tile, k_l1_tile, n_l1_tile,
        m_l0_tile, k_l0_tile, n_l0_tile] = tiling_factors
        tiling = {
            'AUB_shape': [k_l1_tile * block_reduce, m_l1_tile, 1, 1],
            'BUB_shape': [k_l1_tile * block_reduce, n_l1_tile, 1, 1],
            'AL1_shape': [k_l1_tile * block_reduce, m_l1_tile // m_l0_tile, 1, 1],
            'BL1_shape': [k_l1_tile * block_reduce, n_l1_tile // n_l0_tile, 1, 1],
            'AL0_matrix': [m_l0_tile, k_l0_tile, block_in, block_reduce, 1, 1],
            'BL0_matrix': [k_l0_tile, n_l0_tile, block_out, block_reduce, 1, 1],
            'CL0_matrix': [n_l0_tile, m_l0_tile, block_in, block_out, 1, 1],
            'CUB_matrix': [n_l0_tile, m_l0_tile, block_in, block_out, 1, 1],
            'block_dim': [batch_factor, n_factor, m_factor, k_factor],
            'n_bef_batch_flag': 0,
            'n_bef_group_flag': 0,
            'batch_bef_group_flag': 0,
            'A_overhead_opt_flag': 1,
            'B_overhead_opt_flag': 1,
            'AUB_channel_wise_flag': None,
            'BUB_channel_wise_flag': None,
            'CUB_channel_wise_flag': 0,
            'manual_pingpong_buffer':
            {'AUB_pbuffer': double_buffers[0],
            'BUB_pbuffer': double_buffers[1],
            'AL1_pbuffer': double_buffers[2],
            'BL1_pbuffer': double_buffers[3],
            'AL0_pbuffer': double_buffers[4],
            'BL0_pbuffer': double_buffers[5],
            'CL0_pbuffer': double_buffers[6],
            'CUB_pbuffer': double_buffers[7],
            'UBG_pbuffer': 1
            }
        }

        return tiling

    def _get_double_buffer(self, tiling_factors, l0c_byte):
        m_shape, k_shape, n_shape = self.mkn_shape
        [_, _, _, _,
         m_l1_tile, k_l1_tile, n_l1_tile,
         m_l0_tile, _, n_l0_tile] = tiling_factors

        al1_db = 2
        if m_l1_tile * self.block_in == m_shape and k_l1_tile * self.block_reduce == k_shape:
            al1_db = 1
        bl1_db = 2
        if n_l1_tile * self.block_out == n_shape and k_l1_tile * self.block_reduce == k_shape:
            bl1_db = 1

        l0c_db = 2
        l0c_size = tbe_platform_info.get_soc_spec("L0C_SIZE")
        if m_l0_tile * n_l0_tile * self.block_in * self.block_out * l0c_byte * self.DOUBLE_VALUE > l0c_size:
            l0c_db = 1

        return [2, 2, al1_db, bl1_db, 2, 2, l0c_db, 2]

    def _get_perfect_core_num(self):
        """
        :param input_shape_1:the tensor_a shape
        :param input_shape_2:the tensor_b shape
        :return:core_num
        """
        m_shape, k_shape, n_shape = self.mkn_shape
        frac_size = self.block_in
        is_l1fusion = self.l1_fusion_type in (0, 1)
        core_num = self._get_l1fusion_device_core_num(is_l1fusion)
        m_axis_outer = (m_shape + frac_size - 1) // frac_size
        if m_shape == 1:
            m_axis_outer = 1
            n_axis_outer = (n_shape + frac_size - 1) // frac_size
            if n_axis_outer > core_num:
                return 1, core_num
            return 1, 1

        m_axis_outer = m_shape // frac_size
        n_axis_outer = n_shape // frac_size
        if (m_axis_outer * n_axis_outer) <= core_num:
            return m_axis_outer, n_axis_outer
        tensor_a_size = m_shape * k_shape
        tensor_b_size = n_shape * k_shape
        min_copy_size = core_num * (tensor_a_size + tensor_b_size)
        m_factor = 1
        n_factor = 1

        for i in range(1, core_num + 1):
            # judge cur_factor
            if core_num % i != 0:
                continue

            cur_m_factor = i
            cur_n_factor = core_num // i
            if cur_m_factor > m_axis_outer or (m_axis_outer // cur_m_factor) == 0:
                continue
            if cur_n_factor > n_axis_outer or (n_axis_outer // cur_n_factor) == 0:
                continue

            cur_copy_size = cur_n_factor * tensor_a_size + cur_m_factor * \
                tensor_b_size
            temp_m_shape = m_shape
            temp_n_shape = n_shape
            if m_axis_outer % m_factor != 0:
                temp_m_shape = (((m_axis_outer // cur_m_factor) + 1) *
                                cur_m_factor) * frac_size

            if n_shape % n_factor != 0:
                temp_n_shape = (((n_axis_outer // cur_n_factor) + 1) *
                                cur_n_factor) * frac_size

            cur_copy_size = cur_n_factor * (temp_m_shape * k_shape) + \
                cur_m_factor * (temp_n_shape * k_shape)
            if cur_copy_size < min_copy_size:
                min_copy_size = cur_copy_size
                m_factor = cur_m_factor
                n_factor = cur_n_factor

        return m_factor, n_factor

    def _get_knowledge_core(self, shape_mkn_args, m_factors, n_factors):
        """
        get knowledge of core set

        Parameters
        ----------
        shape_mkn_args : list, shape info

        m_factors: core split in m_factor

        n_factors: core split in n_factors

        Returns
        -------
        m_factors, n_factors, value of m, n core split
        """
        shape_map = {}
        core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
        if core_num == self.CORE_NUM_THRITY:
            shape_map = self._get_core_map()

        if shape_map.get(shape_mkn_args) is not None:
            m_factors, n_factors = shape_map[shape_mkn_args]

        return m_factors, n_factors

    def _get_batch_factors(self, m_var, n_var, k_var):
        """
        get batch vars
        """
        m_shape = m_var[0]
        m_factors = m_var[1]
        n_shape = n_var[0]
        n_factors = n_var[1]
        core_inner_m = m_shape
        core_inner_n = n_shape
        core_inner_k = k_var[0]
        batch = self.batch_shape
        if batch in (0, 1):
            block_in = self.block_in
            block_out = self.block_out
            block_reduce = self.block_reduce
            if m_shape != 1:
                core_inner_m = (((m_shape + block_in - 1) // block_in + (m_factors - 1)) // m_factors) * block_in
            core_inner_n = (((n_shape + block_out - 1) // block_out + (n_factors - 1)) // n_factors) * block_out
            core_inner_k = (((k_var[0] + block_reduce - 1) // block_reduce +
                             (k_var[1] - 1)) // k_var[1]) * block_reduce
        return batch, core_inner_m, core_inner_n, core_inner_k

    def _is_need_n_cut_even(self, core_inner_n):
        if not self.date_transfer_fusion:
            return False
        if core_inner_n == 16:
            return False
        return True

    def _get_knowledge_tiling(self, shape_tiling_args, is_b_nz, tiling_shape):
        """
        get knowledge tiling for matmul schedule
        """
        m_shape, k_shape, n_shape, b_trans, ub_res_byte = shape_tiling_args
        b_trans_val = -1
        if b_trans is not None:
            b_trans_val = 1 if b_trans else 0
        shape_args = (m_shape, k_shape, n_shape, b_trans_val, ub_res_byte)

        shape_map = {}
        core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
        if core_num == self.DOUBLE_VALUE:
            if is_b_nz:
                shape_map = self.get_shape_map()
            else:
                shape_map = self.get_mini_frac_shape_map()
        elif core_num in (self.CORE_NUM_THRITY, self.CORE_NUM_THRITY_TWO):
            shape_map = self.get_cloud_shape_map(core_num, self.CORE_NUM_THRITY)
        elif core_num == self.CORE_NUM_EIGHT:
            shape_map = self.get_mdc_shape_map()
        if shape_map.get(shape_args) is not None:
            tiling_shape = shape_map[shape_args]
        else:
            shape_args = (m_shape, k_shape, n_shape, -1, ub_res_byte)
            if shape_map.get(shape_args) is not None:
                tiling_shape = shape_map[shape_args]
            else:
                shape_args = (m_shape, -1, n_shape, b_trans_val, ub_res_byte)
                if shape_map.get(shape_args) is not None:
                    tiling_shape = shape_map[shape_args]
                else:
                    shape_args = (m_shape, -1, n_shape, -1, ub_res_byte)
                    if shape_map.get(shape_args) is not None:
                        tiling_shape = shape_map[shape_args]

        return tiling_shape
