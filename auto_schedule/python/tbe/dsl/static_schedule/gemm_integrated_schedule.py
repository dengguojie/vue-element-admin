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
from enum import Enum
from queue import Queue
import functools
from collections.abc import Iterable

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.buildcfg import build_config
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import in_dynamic
from tbe.dsl.compute import cube_util
from tbe.dsl.boost_schedule_kit import Compare
from tbe.dsl.boost_schedule_kit import ScheduleAgent
from tbe.dsl.instrinsic import cce_emitinsn_params
from tbe.dsl.static_schedule.util import L1CommonParam


def gemm_schedule(res, sch_list, dynamic_para=None):
    """schedule enter
    param:
    res: tensor
    sch_list: list of schedule
    """
    gemm_sch = GemmSchedule(res, sch_list[0], dynamic_para)
    gemm_sch.gemm_schedule()

    return True


class GemmSchedule(object):
    """schedule enter
    param:
    res: tensor
    sch_list: list of schedule
    dynamic_para: dynamic para from gemm_tilingcase
    """
    DEBUG_PARAM = False
    DEBUG_IR = False
    DTYPE_WIDTH_MAP = {"uint64": 4, "float16": 1, "float32": 2, "int32": 2,
                       "int16": 1, "uint16": 1, "int8": 0.5, "uint8": 0.5,
                       "int4": 0.25, "bool": 0.5}
    BYTES_DTYPE = {"uint64": 8, "float16": 2, "float32": 4, "int32": 4,
                    "int16": 2, "uint16": 2, "int8": 1, "uint8": 1,
                    "int4": 0.5, "bool": 1}
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
    emit_fusion_insn_map = {"dequant_NZ": "phony_insn",
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

    is_dynamic = False
    def __init__(self, res, sch, dynamic_para):
        self.res_ori = res
        self.res = res[-1] if isinstance(res, list) else res
        self.root_tensor = res[-1] if isinstance(res, list) else res
        self.sch = sch
        self.sch_agent = None
        self.kernel_name = "gemm"
        self.tiling = None
        self.dynamic_para = dynamic_para
        self.block_in = tbe_platform.BLOCK_IN
        self.block_out = tbe_platform.BLOCK_OUT
        self.block_reduce = tbe_platform.BLOCK_REDUCE
        self.ori_tensors = dict()
        self.TENSOR_MAP = dict()
        self.placeholder_name = dict()
        self.buffer_reuse_dict = dict()
        self.axpy_2_parent = dict()
        self.ori_tensor = dict()
        self.compute_tensors = list()
        self.elemwise_tensors = list()
        self.matmul_dequant_tensor = list()
        self.ele_header_ub_tensors = list()
        self.placeholder_tensors = list()
        self.dequant_activation_tensor = list()
        self.header_ub_tensors = list()
        self.tensors_in_aub = list()
        self.tensors_in_bub = list()
        self.tensors_in_cub = list()
        self.tensors_in_l0c = list()
        self.matmul_tensors = list()
        self.fusion_list = list()
        self.fusion_ele = list()
        self.tensor_fusion_list = list()
        self.compute_inline_list = list()
        self.elewise_compute_inline_list = list()
        self.fusion_tensor_cub = list()
        self.fuse_num_group = list()
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
        self.input_l1_flag, self.input_l1_size = False, 0
        self.in_addr_type, self.out_addr_type = 0, 0
        self.tensor_a_l1_workspace = 0
        self.dynamic_m, self.dynamic_k, self.dynamic_n, self.dynamic_batch = 1, 1, 1, 1
        (self.al0_tiling_batch, self.al0_tiling_ma,
         self.al0_tiling_ka, self.al0_tiling_m0, self.al0_tiling_k0) = 1, 1, 1, 16, 16
        (self.bl0_tiling_batch, self.bl0_tiling_nb,
         self.bl0_tiling_kb, self.bl0_tiling_n0, self.bl0_tiling_k0) = 1, 1, 1, 16, 16
        self.al1_tiling_batch, self.al1_tiling_m, self.al1_tiling_k = 1, 1, 1
        self.bl1_tiling_batch, self.bl1_tiling_n, self.bl1_tiling_k = 1, 1, 1
        (self.aub_tiling_batch, self.aub_tiling_m,
         self.aub_tiling_k, self.aub_tiling_m0, self.aub_tiling_k0)  = 1, 1, 1, 16, 16
        (self.bub_tiling_batch, self.bub_tiling_n,
         self.bub_tiling_k, self.bub_tiling_n0, self.bub_tiling_k0)  = 1, 1, 1, 16, 16
        (self.cl0_tiling_batch, self.cl0_tiling_nc,
         self.cl0_tiling_mc, self.cl0_tiling_n0, self.cl0_tiling_m0) = 1, 1, 1, 16, 16
        self.c_col_k0, self.c_col_k1 = 1, 1
        self.optmt_a, self.optmt_b, self.optmt_c = "float16", "float16", "float16"
        self.format_info_a, self.format_info_b = "", ""
        self.format_a, self.format_b, self.format_out = "ND", "ND", "ND"
        self.al1_attach_status = "full_load"
        self.bl1_attach_status = "full_load"
        self.c_l0c_attach_status = "full_load"
        self.c_ub_attach_status = "full_load"
        self.ops_format = "ND"
        self.ops_data_flow_mode = "fp162fp16"
        self.mad_pattern = tbe_platform.GEMM_MODE
        self.transpose_a, self.transpose_b = False, False
        self.compute_inline_c_ub_fract = False
        self.seed_shape = None
        self.input_l1_flag = False
        self.cube_vector_split = False
        self.mmad_mode = "gemm"
        self.align_a, self.align_b = True, True
        self.get_a_matrix_mode, self.get_b_matrix_mode = "none", "none"
        self.compress_flag = False
        self.int8_not_double_m = False
        self.multi_output_flag = False

    @staticmethod
    def _get_all_tags(res):
        """
        get all tags
        :param res: tensor
        :return: list
        """
        tensor_tags = set()

        def get_tag(tenosr):
            """
            find all tag
            :param tensor: tensor
            :return: all tags
            """
            tensor_list = tenosr.op.input_tensors
            tensor_tags.add(tenosr.op.tag)
            for one_tensor in tensor_list:
                tensor_tags.add(one_tensor.op.tag)
                get_tag(one_tensor)

        get_tag(res)
        return tensor_tags

    @staticmethod
    def _get_value(shape_object):
        """
        get the value if shape object when having attr value
        """
        return shape_object.value if hasattr(shape_object, "value") else shape_object

    @staticmethod
    def _get_all_tensors(res):
        """
        get all tensor
        :param res: tensor
        :return: list
        """

        all_tensor = dict()
        all_tensor["res"] = res

        def get(tensor):
            """
            find all tensor
            :param tensor: c_gm
            :return: all tensor
            """

            tensor_list = tensor.op.input_tensors
            for one_tensor in tensor_list:
                # check which tensor has not been checked
                if one_tensor.op.name not in all_tensor:
                    all_tensor[one_tensor.op.name] = one_tensor
                    get(one_tensor)

        get(res)
        return all_tensor

    @staticmethod
    def _int_ceil_div(divisor_a, divisor_b):
        """
        round up function
        :param divisor_a: int.
        :param divisor_b: int.
        :return: int
        """
        if divisor_b == 0:
            args_dict = {
                "errCode": "E60114",
                "reason": "division by zero",
                "value": "divisor_b = {}".format(divisor_b)
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        return (divisor_a + divisor_b - 1) // divisor_b

    def _print_ir_matmul(self, process, sch):
        """
        print ir for input sch
        :param process: tag
        :param sch: schedule
        :return: IR process
        """
        if self.DEBUG_IR:
            with build_config():
                start = process + " IR start"
                end = process + " IR end\n"
                sch = sch.normalize()
                print(start)
                bounds = tvm.schedule.InferBound(sch)
                stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
                print(stmt)
                print(end)

    def _print_debug(self, info, tag=""):
        """
        print log if debug
        :param info:
        :return:
        """
        if self.DEBUG_PARAM:
            print("----")
            print(tag, info)

    def _get_batch_info(self):
        tensor_l0a = self.TENSOR_MAP.get("a_l0a")
        tensor_l0b = self.TENSOR_MAP.get("b_l0b")
        tensor_l0c = self.TENSOR_MAP.get("c_l0c")
        self.have_batch_a = len(tensor_l0a.shape) in (3, 5)
        self.have_batch_b = len(tensor_l0b.shape) in (3, 5)
        self.have_batch = len(tensor_l0c.shape) in (3, 5)

    def gemm_schedule(self):
        """
        the main func of gemm_schedule
        """
        self._print_ir_matmul("orgin ir", self.sch)
        self.ori_tensors = self._get_all_tensors(self.res)
        self._get_global_para(self.ori_tensors)
        self._get_seed_shape()
        self._set_data_layout(self.res)
        self._print_ir_matmul("after data layout", self.sch)
        self._get_batch_info()
        self._set_buffer_reuse_dict()
        self._tiling_process()
        self.sch_agent = ScheduleAgent(self.sch)
        self._tiling_l0_process()
        self._tiling_l1_process()
        self._tiling_ub_process()
        c_ub_tiling_shape = self._cub_process()
        self._cl0_process(c_ub_tiling_shape)
        self._l0a_process()
        self._l0b_process()
        self._do_l1_ub_process()
        self._bind_multi_core()
        self._do_emit_insn()
        self._do_buffer_reuse()
        self.sch_agent.apply()
        self._do_buffer_align()
        self._slove_bank_conflict()
        a_run_once, b_run_once = False, False
        self._double_buffer(a_run_once, b_run_once)
        self._do_compute_inline()
        self._mem_process()
        self._set_var_range_for_dynamic()
        self._print_ir_matmul("finial", self.sch)
        self.tiling.clear()
        self.TENSOR_MAP.clear()
        return True

    def _get_global_para(self, ori_tensors):
        if in_dynamic():
            self.is_dynamic = True
        self.root_tensor = self.res
        tensor_l0c = ori_tensors.get("tensor_c_matrix")
        self.ops_format = tensor_l0c.op.attrs["ops_format"].value
        self.have_bias = tensor_l0c.op.attrs["have_bias"].value
        self.have_c = tensor_l0c.op.attrs["have_c"].value
        self.format_a = tensor_l0c.op.attrs["format_a"].value
        self.format_b = tensor_l0c.op.attrs["format_b"].value
        self.align_a = tensor_l0c.op.attrs["align_a"].value
        self.align_b = tensor_l0c.op.attrs["align_b"].value
        self.ops_data_flow_mode = tensor_l0c.op.attrs["ops_data_flow_mode"].value
        self.only_use_gevm_gemv_flow = tensor_l0c.op.attrs["only_use_gevm_gemv_flow"].value
        self.int8_not_double_m = tensor_l0c.op.attrs["int8_not_double_m"].value
        self.placeholder_name = tensor_l0c.op.attrs["placeholder_name"]
        # user self.placeholder_name to avoid the inconsistency of placeholder names
        # in the fusion scene and the single operator scene
        self.need_init_bias = ori_tensors[self.placeholder_name['bias'].value].op.attrs[
            "ori_shape"][-1].value % 16 != 0 if self.have_bias else False
        self.compress_flag = tensor_l0c.op.attrs["compress_flag"].value
        self.kernel_name = tensor_l0c.op.attrs["kernel_name"].value
        self.cube_vector_split = tbe_platform_info.get_soc_spec("CUBE_VECTOR_SPLIT")
        self.block_reduce = tbe_platform.BLOCK_REDUCE
        if self.ops_data_flow_mode == "int82int32":
            self.block_reduce = tbe_platform.BLOCK_REDUCE_INT8

        self.transpose_a = tensor_l0c.op.attrs["transpose_a"].value
        self.transpose_b = tensor_l0c.op.attrs["transpose_b"].value
        self.mmad_mode = tensor_l0c.op.attrs["mmad_mode"].value
        self.mad_pattern = tbe_platform.GEVM_MODE if self.mmad_mode in ("gevm", "gemv") else tbe_platform.GEMM_MODE
        self.mad_pattern = tbe_platform.GEMM_MODE if self.only_use_gevm_gemv_flow else self.mad_pattern
        self._fusion_para()
        self._print_debug(self.ops_format, "ops_format")
        self._print_debug(self.ops_data_flow_mode, "ops_data_flow_mode")
        self._print_debug(self.cube_vector_split, "cube_vector_split")
        self._print_debug(self.align_a, "align_a")
        self._print_debug(self.align_b, "align_b")

    def _fusion_para(self):
        res = self.res
        self.out_addr_type = self._get_addr_type(res) # 0:DDR;1:L1
        self.format_out = self._get_output_format(res)

    def _get_addr_type(self, tensor):
        addr_type = 0
        if "addr_type" in tensor.op.attrs and tensor.op.attrs["addr_type"].value == 1:
            addr_type = 1
        return addr_type

    def _get_output_format(self, tensor):
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

    def _get_b_l1_fractal(self):
        TENSOR_MAP = self.TENSOR_MAP
        b_l0b = TENSOR_MAP.get("b_l0b")
        get_tensor_from_compress = ("tile_L1_n" in b_l0b.op.attrs)

        if get_tensor_from_compress:
            TENSOR_MAP["compress_index"] = b_l0b.op.input_tensors[0]
            TENSOR_MAP["b_l1"] = self.sch.cache_write(b_l0b, tbe_platform_info.scope_cbuf)
        else:
            TENSOR_MAP["b_l1"] = self.sch.cache_read(TENSOR_MAP["b_placehold"],
                                                     tbe_platform_info.scope_cbuf,
                                                     [TENSOR_MAP["b_l0b"]])

    def _set_data_layout(self, res):
        self.compute_inline_list = list()
        self.TENSOR_MAP = dict()
        # for compute at and db

        self.compute_tensors = self._get_compute_tensor(res)
        self._set_fusion_flag()
        self._set_data_layout_base_tensor()
        if self.cube_vector_split:
            self._set_data_layout_cube_vector_split()
        else:
            self._set_data_layout_after_mmad()
            self._set_data_layout_a_matrix()
            self._set_data_layout_b_matrix()
            self._set_data_layout_fusion()

    def _set_fusion_flag(self):
        self.dequant_flag = "dequant_NZ" in self.compute_tensors or "dequant_ND" in self.compute_tensors
        self.sqrt_flag = self.dequant_flag and "dequant_sqrt" in self.compute_tensors

    def _change_l0_gemv(self):
        TENSOR_MAP = self.TENSOR_MAP
        if self.mmad_mode == "gemv":
            TENSOR_MAP["a_l0a"], TENSOR_MAP["b_l0b"] = TENSOR_MAP["b_l0b"], TENSOR_MAP["a_l0a"]

    def _get_compute_tensor(self, tensor):
        """
        scan all the transient tensor during calculation
        tensor: target tensor which needs to find placeholder tensor
        """
        compute_tensors_local = list()
        placeholder_tensors = self.placeholder_tensors
        def enter(tensor):
            """
            get compute tensors by search
            """
            if tensor not in compute_tensors_local:
                compute_tensors_local.append(tensor)
            tensor_list = tensor.op.input_tensors
            for one_tensor in tensor_list:
                # check which tensor has not been checked
                if one_tensor not in compute_tensors_local:
                    if isinstance((one_tensor.op), tvm.tensor.PlaceholderOp):
                        placeholder_tensors.append(one_tensor)
                    else:
                        compute_tensors_local.append(one_tensor)
                        enter(one_tensor)
        enter(tensor)
        return compute_tensors_local

    def _set_data_layout_base_tensor(self):
        sch = self.sch
        TENSOR_MAP = self.TENSOR_MAP
        placeholder_name = self.placeholder_name
        compute_tensors = self.compute_tensors
        placeholder_tensors = self.placeholder_tensors
        TENSOR_MAP["c_gm"] = self._match_and_get_tensor(compute_tensors, "tensor_c_gm")
        TENSOR_MAP["a_placehold"] = self._match_and_get_tensor(placeholder_tensors, placeholder_name["a"])
        TENSOR_MAP["b_placehold"] = self._match_and_get_tensor(placeholder_tensors, placeholder_name["b"])
        TENSOR_MAP["alpha"] = self._match_and_get_tensor(placeholder_tensors, placeholder_name["alpha"])
        TENSOR_MAP["beta"] = self._match_and_get_tensor(placeholder_tensors, placeholder_name["beta"])
        TENSOR_MAP["c_l0c"] = self._match_and_get_tensor(compute_tensors, "tensor_c_matrix")
        sch[TENSOR_MAP["c_l0c"]].set_scope(tbe_platform_info.scope_cc)
        load_a_matrix = self.format_a == "FRACTAL_Z" and (not self.transpose_a)
        load_a_matrix = load_a_matrix and (not ((self.format_a == "FRACTAL_Z") and (self.mmad_mode == "gemv")))
        l0a_scope = tbe_platform_info.scope_cb if self.mmad_mode == "gemv" else tbe_platform_info.scope_ca
        if load_a_matrix:
            TENSOR_MAP["a_l0a"] = sch.cache_read(TENSOR_MAP["a_placehold"], l0a_scope, [TENSOR_MAP["c_l0c"]])
        else:
            TENSOR_MAP["a_l0a"] = self._match_and_get_tensor(compute_tensors, "tensor_a_matrix")
            sch[TENSOR_MAP["a_l0a"]].set_scope(l0a_scope)

        if "mode" in TENSOR_MAP["a_l0a"].op.attrs:
            self.get_a_matrix_mode = TENSOR_MAP["a_l0a"].op.attrs["mode"]

        if "format_info" in TENSOR_MAP["a_l0a"].op.attrs:
            self.format_info_a = TENSOR_MAP["a_l0a"].op.attrs["format_info"]
        else:
            self.format_info_a = {
                "format_in_a_l1": "Zz",
                "format_in_a_ub": "none"
            }

        load_b_matrix = (((self.format_b == "FRACTAL_Z")
            or (self.ops_format == "FRACTAL_NZ" and self.ops_data_flow_mode == "int82int32"))
            and (not self.ops_data_flow_mode == "int82fp32")
            and (not self.compress_flag)
            and (self.mmad_mode != "gemv"))
        l0b_scope = tbe_platform_info.scope_ca if self.mmad_mode == "gemv" else tbe_platform_info.scope_cb
        if load_b_matrix:
            TENSOR_MAP["b_l0b"] = sch.cache_read(TENSOR_MAP["b_placehold"], l0b_scope, [TENSOR_MAP["c_l0c"]])
        else:
            TENSOR_MAP["b_l0b"] = self._match_and_get_tensor(compute_tensors, "tensor_b_matrix")
            sch[TENSOR_MAP["b_l0b"]].set_scope(l0b_scope)

        if "mode" in TENSOR_MAP["b_l0b"].op.attrs:
            self.get_b_matrix_mode = TENSOR_MAP["b_l0b"].op.attrs["mode"]

        if "format_info" in TENSOR_MAP["b_l0b"].op.attrs:
            self.format_info_b = TENSOR_MAP["b_l0b"].op.attrs["format_info"]
        else:
            self.format_info_b = {
                "format_in_b_l1": "Zn",
                "format_in_b_ub": "none"
            }

        self.optmt_a = TENSOR_MAP["a_l0a"].dtype
        self.optmt_b = TENSOR_MAP["b_l0b"].dtype
        self.optmt_c = TENSOR_MAP["c_gm"].dtype
        self._add_tensor_to_list(TENSOR_MAP.get("c_l0c"), [self.tensors_in_l0c])

    def _set_data_layout_cube_vector_split(self):
        tiling = self.tiling
        sch = self.sch
        TENSOR_MAP = self.TENSOR_MAP

        TENSOR_MAP["a_l1"] = sch.cache_read(TENSOR_MAP["a_placehold"],
            tbe_platform_info.scope_cbuf, [TENSOR_MAP["a_l0a"]])
        if self.src_dtype != "int8" or tiling.get("BL1_shape") is not None:
            TENSOR_MAP["b_l1"] = sch.cache_read(TENSOR_MAP["b_placehold"], tbe_platform_info.scope_cbuf,
                [TENSOR_MAP["b_l0b"]])

    def _add_tensor_to_list(self, tensor, tensors_list_list):
        if tensor is not None:
            for tensors_list in tensors_list_list:
                if tensor not in tensors_list:
                    tensors_list.append(tensor)

    def _set_tensor_scope(self, tensor, buffer_local):
        if tensor is not None:
            self.sch[tensor].set_scope(buffer_local)

    def _set_data_layout_a_matrix(self):
        sch = self.sch
        TENSOR_MAP = self.TENSOR_MAP
        placeholder_tensors = self.placeholder_tensors
        compute_tensors = self.compute_tensors
        tensors_in_aub = self.tensors_in_aub

        self._get_tensor_and_set_scope("tensor_a_normalize_ub", tbe_platform_info.scope_ubuf, "a_ub")
        if self.get_a_matrix_mode == "none":
            TENSOR_MAP["a_l1"] = sch.cache_write(TENSOR_MAP["a_l0a"], tbe_platform_info.scope_cbuf)
        elif self.get_a_matrix_mode == "nd2Zz_vnchwconv":
            self._get_tensor_and_set_scope("tensor_a_matrix_fract_k", tbe_platform_info.scope_ubuf, "a_ub_fract")
            TENSOR_MAP["a_l1"] = sch.cache_read(TENSOR_MAP["a_ub_fract"],
                tbe_platform_info.scope_cbuf, [TENSOR_MAP["a_l0a"]])
            if self.ops_data_flow_mode == "int82fp32":
                # int82fp32 need cast to fp16
                self._get_tensor_and_set_scope("tensor_a_int82fp16", tbe_platform_info.scope_ubuf, "a_int82fp16")
        elif self.get_a_matrix_mode == "nd2Zz_int8":
            if self.transpose_a:
                self._get_tensor_and_set_scope("a_transpose", tbe_platform_info.scope_ubuf, "a_transpose")
            TENSOR_MAP["a_l1"] = sch.cache_write(TENSOR_MAP["a_l0a"], tbe_platform_info.scope_cbuf)
        elif self.get_a_matrix_mode == "Nz2Zz_int82fp32":
            # int82fp32 need cast to fp16
            self._get_tensor_and_set_scope("tensor_a_int82fp16", tbe_platform_info.scope_ubuf, "a_int82fp16")
            TENSOR_MAP["a_ub"] = sch.cache_read(TENSOR_MAP["a_placehold"], tbe_platform_info.scope_ubuf,
                [TENSOR_MAP["a_int82fp16"]])
            TENSOR_MAP["a_l1"] = sch.cache_write(TENSOR_MAP["a_l0a"], tbe_platform_info.scope_cbuf)
            TENSOR_MAP["a_ub_fract"] = sch.cache_write(TENSOR_MAP["a_l1"], tbe_platform_info.scope_ubuf)
        elif self.get_a_matrix_mode in ("Nz2Zz", "fractal_gemv", "Zz_trans"):
            TENSOR_MAP["a_l1"] = sch.cache_read(TENSOR_MAP["a_placehold"], tbe_platform_info.scope_cbuf,
                [TENSOR_MAP["a_l0a"]])
        elif self.get_a_matrix_mode == "nd_gemv":
            self._get_tensor_and_set_scope("tensor_a_fract", tbe_platform_info.scope_cbuf, "a_l1")
            if self.optmt_a == "float16":
                TENSOR_MAP["a_ub_fract"] = sch.cache_write(TENSOR_MAP["a_l1"], tbe_platform_info.scope_ubuf)
        elif self.get_a_matrix_mode == "nd_gevm":
            TENSOR_MAP["a_l1"] = sch.cache_write(TENSOR_MAP["a_l0a"], tbe_platform_info.scope_cbuf)
            # check in int8
            self._get_tensor_and_set_scope("tensor_a_fract", tbe_platform_info.scope_ubuf, "a_ub_fract")
        elif self.get_a_matrix_mode == "nd2Zz":
            TENSOR_MAP["a_l1"] = sch.cache_write(TENSOR_MAP["a_l0a"], tbe_platform_info.scope_cbuf)
            if self.ops_data_flow_mode == "int82fp32":
                self._get_tensor_and_set_scope("tensor_a_int82fp16", tbe_platform_info.scope_ubuf, "a_int82fp16")
            if self.optmt_a == "float16":
                TENSOR_MAP["a_ub_fract"] = sch.cache_write(TENSOR_MAP["a_l1"], tbe_platform_info.scope_ubuf)

        self._add_tensor_to_list(TENSOR_MAP.get("a_int82fp16"), [tensors_in_aub])
        self._add_tensor_to_list(TENSOR_MAP.get("a_ub"), [tensors_in_aub])
        self._add_tensor_to_list(TENSOR_MAP.get("a_ub_fract"), [tensors_in_aub])
        self._add_tensor_to_list(TENSOR_MAP.get("a_transpose"), [tensors_in_aub])

    def _set_data_layout_b_matrix(self):
        sch = self.sch
        TENSOR_MAP = self.TENSOR_MAP
        placeholder_tensors = self.placeholder_tensors
        compute_tensors = self.compute_tensors
        tensors_in_bub = self.tensors_in_bub

        self._get_tensor_and_set_scope("tensor_b_normalize_ub", tbe_platform_info.scope_ubuf, "b_ub")

        if self.get_b_matrix_mode == "nd_gemv":
            self._get_tensor_and_set_scope("tensor_b_fract", tbe_platform_info.scope_cbuf, "b_l1")
            if self.optmt_b == "float16":
                TENSOR_MAP["b_ub_fract"] = sch.cache_write(TENSOR_MAP["b_l1"], tbe_platform_info.scope_ubuf)
        elif self.get_b_matrix_mode == "nd2Zn_vnchwconv":
            self._get_tensor_and_set_scope("tensor_b_matrix_fract", tbe_platform_info.scope_ubuf, "b_ub_fract")
            TENSOR_MAP["b_l1"] = sch.cache_read(TENSOR_MAP["b_ub_fract"],
                tbe_platform_info.scope_cbuf, [TENSOR_MAP["b_l0b"]])
            if self.ops_data_flow_mode == "int82fp32":
                # if int82fp32 need cast to fp16
                self._get_tensor_and_set_scope("tensor_b_int82fp16", tbe_platform_info.scope_ubuf, "b_int82fp16")
        elif self.get_b_matrix_mode == "nd2Zn_int8":
            if not self.transpose_b:
                self._get_tensor_and_set_scope("b_transpose", tbe_platform_info.scope_ubuf, "b_transpose")
            TENSOR_MAP["b_l1"] = sch.cache_write(TENSOR_MAP["b_l0b"], tbe_platform_info.scope_cbuf)
        elif self.get_b_matrix_mode == "Zn2Zn_int82fp32":
            self._get_tensor_and_set_scope("tensor_b_int82fp16", tbe_platform_info.scope_ubuf, "b_int82fp16")
            TENSOR_MAP["b_ub"] = sch.cache_read(TENSOR_MAP["b_placehold"], tbe_platform_info.scope_ubuf,
                [TENSOR_MAP["b_int82fp16"]])
            TENSOR_MAP["b_l1"] = sch.cache_write(TENSOR_MAP["b_l0b"], tbe_platform_info.scope_cbuf)
            TENSOR_MAP["b_ub_fract"] = sch.cache_write(TENSOR_MAP["b_l1"], tbe_platform_info.scope_ubuf)
        elif self.get_b_matrix_mode in ("Nz2Zn", "Nz2Zz", "fractal_gemv", "Zn_trans"):
            TENSOR_MAP["b_l1"] = sch.cache_read(TENSOR_MAP["b_placehold"], tbe_platform_info.scope_cbuf,
                [TENSOR_MAP["b_l0b"]])
        else:
            self._get_b_l1_fractal()

        self._add_tensor_to_list(TENSOR_MAP.get("b_int82fp16"), [tensors_in_bub])
        self._add_tensor_to_list(TENSOR_MAP.get("b_ub_fract"), [tensors_in_bub])
        self._add_tensor_to_list(TENSOR_MAP.get("b_ub"), [tensors_in_bub])
        self._add_tensor_to_list(TENSOR_MAP.get("b_transpose"), [tensors_in_bub])

    def _set_data_layout_after_mmad(self):
        sch = self.sch
        TENSOR_MAP = self.TENSOR_MAP
        placeholder_tensors = self.placeholder_tensors
        compute_tensors = self.compute_tensors
        tensors_in_cub = self.tensors_in_cub
        tensors_in_l0c = self.tensors_in_l0c
        placeholder_name = self.placeholder_name
        bias = self._match_and_get_tensor(placeholder_tensors, placeholder_name["bias"])
        tensor_c = self._match_and_get_tensor(placeholder_tensors, placeholder_name["c"])
        TENSOR_MAP["tensor_c"] = tensor_c
        TENSOR_MAP["bias"] = bias

        alpha = self._match_and_get_tensor(placeholder_tensors, placeholder_name["alpha"])
        beta = self._match_and_get_tensor(placeholder_tensors, placeholder_name["beta"])
        self._get_tensor_and_set_scope("c_ub_fract", tbe_platform_info.scope_ubuf, "c_ub_fract")

        add_bias_in_l0c = self.have_bias and (not self.cube_vector_split)
        add_bias_in_fb = self.have_bias and self.cube_vector_split
        add_bias_in_ub = self.have_c
        bias_ub_compute_at = list()
        if add_bias_in_l0c:
            bias_ub_compute_at = tensors_in_l0c
            self._get_tensor_and_set_scope("tensor_bias_l0c", tbe_platform_info.scope_cc, "bias_l0c")
            self._get_tensor_and_set_scope("tensor_c_add_bias", tbe_platform_info.scope_cc, "c_add_bias")
            self._get_tensor_and_set_scope("tensor_bias_ub", tbe_platform_info.scope_ubuf, "bias_ub")

            if self.need_init_bias:
                self._get_tensor_and_set_scope('tensor_init_value_of_bias_ub', tbe_platform_info.scope_ubuf, 'init_value_of_bias_ub')
                self._get_tensor_and_set_scope('tensor_virtual_add_bias', tbe_platform_info.scope_ubuf, 'virtual_add_bias')

        if add_bias_in_fb:
            pass
        if add_bias_in_ub:
            bias_ub_compute_at = tensors_in_cub
            self._get_tensor_and_set_scope("tensor_c_add_bias_ub", tbe_platform_info.scope_ubuf, "c_add_bias_ub")
            self._get_tensor_and_set_scope("tensor_bias_normalize_ub", tbe_platform_info.scope_ubuf, "bias_ub")
            if self.ops_data_flow_mode == "fp162fp16":
                self._get_tensor_and_set_scope("tensor_bias_cast_to_fp32",
                    tbe_platform_info.scope_ubuf, "bias_cast_to_fp32")

            if beta is not None:
                self._get_tensor_and_set_scope("tensor_beta_bias", tbe_platform_info.scope_ubuf, "beta_bias")
                if self.ops_data_flow_mode == "fp162fp16":
                    self._get_tensor_and_set_scope("tensor_beta_fp162fp32",
                        tbe_platform_info.scope_ubuf, "beta_fp162fp32")
                    TENSOR_MAP["beta_ub"] = sch.cache_read(TENSOR_MAP["beta"], tbe_platform_info.scope_ubuf,
                        [TENSOR_MAP["beta_fp162fp32"]])
                else:
                    TENSOR_MAP["beta_ub"] = sch.cache_read(TENSOR_MAP["beta"], tbe_platform_info.scope_ubuf,
                        [TENSOR_MAP["beta_bias"]])

        if alpha is not None:
            self._get_tensor_and_set_scope("tensor_alpha_c", tbe_platform_info.scope_ubuf, "alpha_c")
            if self.ops_data_flow_mode == "fp162fp16":
                self._get_tensor_and_set_scope("tensor_alpha_fp162fp32",
                    tbe_platform_info.scope_ubuf, "alpha_fp162fp32")
                TENSOR_MAP["alpha_ub"] = sch.cache_read(TENSOR_MAP["alpha"], tbe_platform_info.scope_ubuf,
                    [TENSOR_MAP["alpha_fp162fp32"]])
            else:
                TENSOR_MAP["alpha_ub"] = sch.cache_read(TENSOR_MAP["alpha"], tbe_platform_info.scope_ubuf,
                    [TENSOR_MAP["alpha_c"]])

        self._get_tensor_and_set_scope("tensor_cast_to_fp16", tbe_platform_info.scope_ubuf, "cast_to_fp16")

        TENSOR_MAP["nz_to_nd"] = self._match_and_get_tensor(compute_tensors, "nz_to_nd")
        if TENSOR_MAP["nz_to_nd"] is not None:
            sch[TENSOR_MAP["nz_to_nd"]].set_scope(tbe_platform_info.scope_ubuf)

        self._add_tensor_to_list(TENSOR_MAP.get("bias_l0c"), [tensors_in_l0c])
        self._add_tensor_to_list(TENSOR_MAP.get("beta_bias"), [tensors_in_cub])
        self._add_tensor_to_list(TENSOR_MAP.get("c_ub_fract"), [tensors_in_cub])
        self._add_tensor_to_list(TENSOR_MAP.get("alpha_c"), [tensors_in_cub])
        self._add_tensor_to_list(TENSOR_MAP.get("c_add_bias"), [tensors_in_l0c])
        self._add_tensor_to_list(TENSOR_MAP.get("c_add_bias_ub"), [tensors_in_cub])
        self._add_tensor_to_list(TENSOR_MAP.get("cast_to_fp16"), [tensors_in_cub])
        self._add_tensor_to_list(TENSOR_MAP.get("bias_cast_to_fp32"), [tensors_in_cub])
        self._add_tensor_to_list(TENSOR_MAP.get("bias_ub"), [bias_ub_compute_at])
        if self.need_init_bias:
            self._add_tensor_to_list(TENSOR_MAP.get("init_value_of_bias_ub"), [bias_ub_compute_at])
            self._add_tensor_to_list(TENSOR_MAP.get("virtual_add_bias"), [bias_ub_compute_at])
        self._add_tensor_to_list(TENSOR_MAP.get("nz_to_nd"), [tensors_in_cub])

        cast_to_fp16 = TENSOR_MAP.get("cast_to_fp16")
        c_ub_fract = TENSOR_MAP.get("c_ub_fract")
        self.compute_inline_c_ub_fract = ((cast_to_fp16 is not None)
            and (cast_to_fp16.op.input_tensors[0] == c_ub_fract))
        if self.compute_inline_c_ub_fract:
            self._add_tensor_to_list(c_ub_fract, [self.compute_inline_list])

    def _get_tensor_and_set_scope(self, tensor_name, buffer_name, save_name=None):
        if save_name is None:
            save_name = tensor_name
        TENSOR_MAP = self.TENSOR_MAP
        TENSOR_MAP[save_name] = self._match_and_get_tensor(self.compute_tensors, tensor_name)
        if TENSOR_MAP[save_name] is not None:
            self.sch[TENSOR_MAP[save_name]].set_scope(buffer_name)

    def _set_and_add_tensor(self, tensor, tensors_lists, buffer_type):
        if tensor is not None:
            self.sch[tensor].set_scope(buffer_type)

            for a_list in tensors_lists:
                a_list.append(tensor)

    def _get_quant_fusion_tensor(self):
        matmul_dequant_tensor = self.matmul_dequant_tensor
        tensor_fusion_list = self.tensor_fusion_list
        if not self.quant_fusion:
            return
        for ten_in in self.compute_tensors:
            if ten_in == self.res:
                continue
            if ten_in not in matmul_dequant_tensor and ten_in.op.name in self.emit_fusion_insn_map:
                tensor_fusion_list.append(ten_in)

    def _match_and_get_tensor(self, tensors, tensor_name):
        """
        match and get tensor
        """
        for i in tensors:
            if tensor_name == i.op.name:
                return i
        return None

    def _get_elewise_ub_tensors(self, tensor_ele_ub):
        """
        get axpy_ub to axpy_parents[1]_ub dict, in order to set reused_by.
        """
        sch = self.sch
        elemwise_tensors = self.elemwise_tensors
        axpy_and_parent = list()
        for ten_i in elemwise_tensors:
            if "elewise_binary_scalar_axpy" in ten_i.op.tag:
                axpy_and_parent.append([ten_i, ten_i.op.input_tensors[1]])

        for ten_i in elemwise_tensors:
            if "broadcast" in ten_i.op.tag:
                self.compute_inline_list.append(ten_i)
            else:
                ele_ub = sch.cache_write(ten_i, tbe_platform_info.scope_ubuf)
                for index, (axpy, parent) in enumerate(axpy_and_parent):
                    if ten_i == axpy:
                        axpy_and_parent[index][0] = ele_ub
                    if ten_i == parent:
                        axpy_and_parent[index][1] = ele_ub
                tensor_ele_ub.append(ele_ub)
                self._add_tensor_to_list(ten_i, [self.elewise_compute_inline_list])
        if axpy_and_parent:
            return dict(axpy_and_parent)
        return dict()

    def _emit_requant_fusion_insn(self):
        sch = self.sch
        tensor_reform = self.TENSOR_MAP.get("requant_data_transfer")
        if tensor_reform is None:
            return
        insn = self.requant_fusion_insn_map.get(tensor_reform.op.name)
        self.sch[tensor_reform].emit_insn(tensor_reform.op.axis[2], insn)

        return

    def _dequant_fusion_proc(self):
        dequant_tensor = self.TENSOR_MAP.get("dequant_tensor")
        tensor_sqrt = self.TENSOR_MAP.get("tensor_sqrt")
        sch = self.sch
        self._emit_insn_func(self.TENSOR_MAP.get("tensor_deq_ub"), 0, "dma_copy")
        self._dequant_activation_emit_insn_simple()
        dequant_emit_axis, deq_scale_mode = (1, "vector") \
            if "vector" in dequant_tensor.op.tag else (0, "scalar")

        if cube_util.is_ng1_version() or cube_util.is_lhisi_cs_version():
            sch[dequant_tensor].emit_insn(
                dequant_tensor.op.axis[dequant_emit_axis], "dma_copy")
        else:
            sch[dequant_tensor].pragma(
                dequant_tensor.op.axis[dequant_emit_axis], "deq_scale", deq_scale_mode)
            self._emit_insn_func(tensor_sqrt, 0, "vector_auto")

        self._compute_inline_dequant_output()

    def _compute_inline_dequant_output(self):
        """
        compute inline dequant output tensor when dequant is not the last op.
        """
        compute_inline_list = self.compute_inline_list
        dequant_nz = self.TENSOR_MAP.get("dequant_nz")
        dequant_nd = self.TENSOR_MAP.get("dequant_nd")
        if dequant_nz is not None and self.res != dequant_nz:
            self._add_tensor_to_list(dequant_nz, [compute_inline_list])
        if dequant_nd is not None and self.res != dequant_nd:
            self._add_tensor_to_list(dequant_nd, [compute_inline_list])

    def _round_emit_insn(self, round_mode):
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
                emit_insn_str = "vector_conv_round"
            elif round_mode == "Ceil":
                emit_insn_str = "vector_conv_ceil"
            elif round_mode == "Floor":
                emit_insn_str = "vector_conv_floor"
            elif round_mode == "Trunc":
                emit_insn_str = "vector_conv_trunc"
            else:
                emit_insn_str = "vector_conv"
        return emit_insn_str

    def _quant_fusion_proc(self):
        sch = self.sch
        input_ub = self.TENSOR_MAP.get("tensor_input_ub")
        self._add_tensor_to_list(input_ub, [self.compute_inline_list])
        for ten_in in self.tensor_fusion_list:
            if ten_in.op.name == "cast_i8_ub":
                insn = self._round_emit_insn(self.round_mode)
            else:
                insn = self.emit_fusion_insn_map.get(ten_in.op.name)
            if ten_in.op.name in self.reform_tensor_tag_list:
                sch[ten_in].emit_insn(ten_in.op.axis[2], insn)
            else:
                sch[ten_in].emit_insn(ten_in.op.axis[0], insn)

    def _dequant_activation_emit_insn_simple(self):
        sch = self.sch
        if self.dequant_fusion:
            for ten_in in self.dequant_activation_tensor:
                if ten_in.op.tag.find("|") != -1:
                    str_list = ten_in.op.tag.split("|")
                    insn = self.emit_insn_map.get(str_list[0])
                else:
                    insn = self.emit_insn_map.get(ten_in.op.tag)
                if ten_in in self.header_ub_tensors:
                    insn = "dma_copy"
                if insn is None:
                    insn = "vector_auto"
                if "elewise_binary_scalar_axpy" in ten_in.op.tag:
                    sch[ten_in].reused_by(ten_in.op.input_tensors[1])
                sch[ten_in].emit_insn(ten_in.op.axis[0], insn)

    def _requant_fusion_proc(self):
        requant_scale = self.TENSOR_MAP.get("requant_scale")
        tensor_drq = requant_scale.op.input_tensors[1]
        tensor_drq_ub = self.sch.cache_read(tensor_drq, tbe_platform_info.scope_ubuf, [requant_scale])
        self.sch[tensor_drq_ub].emit_insn(tensor_drq_ub.op.axis[0], "dma_copy")

        self._add_tensor_to_list(requant_scale, [self.compute_inline_list])
        self._emit_requant_fusion_insn()

    def _quantify_fusion_entry(self):
        if not self.quantify_fusion:
            return

        if self.requant_fusion:
            self._requant_fusion_proc()

        if self.dequant_fusion:
            self._dequant_fusion_proc()

        if self.quant_fusion:
            self._quant_fusion_proc()

        sch = self.sch
        reform_fusion = self.quant_fusion or self.requant_fusion
        if reform_fusion:
            tensor_len_c = len(self.TENSOR_MAP.get("c_l0c").shape)
            tensor_reform = self.TENSOR_MAP.get("tensor_reform")
            reform_c_outer, reform_c_inner = sch[tensor_reform].split(
                tensor_reform.op.axis[tensor_len_c - 1], factor=16)
            sch[tensor_reform].reorder(
                tensor_reform.op.axis[tensor_len_c - 4],
                tensor_reform.op.axis[tensor_len_c - 3],
                reform_c_outer,
                tensor_reform.op.axis[tensor_len_c - 2],
                reform_c_inner)

        return

    def _set_scope_fusion(self):
        sch = self.sch
        TENSOR_MAP = self.TENSOR_MAP
        dequant_tensor_list = list()

        if self.out_addr_type == 1:
            self._set_tensor_scope(self.res, tbe_platform_info.scope_cbuf_fusion)

        if self.in_addr_type == 1:
            tensor_a = self.TENSOR_MAP.get("a_placehold")
            self._set_tensor_scope(tensor_a, tbe_platform_info.scope_cbuf_fusion)

        if self.dequant_fusion:
            dequant_tensor_list.append(TENSOR_MAP["dequant_tensor"])
            if self.sqrt_flag:
                tensor_sqrt = TENSOR_MAP.get("tensor_sqrt")
                sch[tensor_sqrt].set_scope(tbe_platform_info.scope_ubuf)
                dequant_tensor_list.append(tensor_sqrt)
            for tensor in self.dequant_activation_tensor:
                sch[tensor].set_scope(tbe_platform_info.scope_ubuf)
            for tensor in self.tensor_fusion_list:
                sch[tensor].set_scope(tbe_platform_info.scope_ubuf)

        for tensor in self.fusion_list:
            sch[tensor].set_scope(tbe_platform_info.scope_ubuf)

        self._get_tensor_deq(dequant_tensor_list)

    def _get_tensor_deq(self, dequant_tensor_list):
        TENSOR_MAP = self.TENSOR_MAP
        tensor_deq = self._match_and_get_tensor(self.placeholder_tensors, "tensor_deq")
        if tensor_deq is None:
            tensor_dequant = TENSOR_MAP.get("dequant_tensor")
            if tensor_dequant is not None:
                tensor_deq = tensor_dequant.op.input_tensors[1]
        if tensor_deq is not None:
            TENSOR_MAP["tensor_deq"] = tensor_deq
            TENSOR_MAP["tensor_deq_ub"] = self.sch.cache_read(TENSOR_MAP["tensor_deq"],
                tbe_platform_info.scope_ubuf, dequant_tensor_list)

    def _find_tensor_and_get_flag(self, compute_tensors):
        TENSOR_MAP = self.TENSOR_MAP
        fusion_tensor_cub = self.fusion_tensor_cub
        for i in compute_tensors:
            if i.op.name == "dequant":
                TENSOR_MAP["dequant_tensor"] = i
                fusion_tensor_cub.append(i)
                self.dequant_fusion = True
            if "dequant_sqrt" in i.op.name:
                fusion_tensor_cub.append(i)
                TENSOR_MAP["tensor_sqrt"] = i
                self.sqrt_flag = True
            if "dequant_NZ" in i.op.name:
                fusion_tensor_cub.append(i)
                TENSOR_MAP["dequant_nz"] = i
            if "dequant_ND" in i.op.name:
                fusion_tensor_cub.append(i)
                TENSOR_MAP["dequant_nd"] = i
                dequant_nd_fract = True
            if i.op.tag == "quant":
                TENSOR_MAP["quant"] = i
                self.quant_fusion = True
                self.round_mode = i.op.attrs["round_mode"]
            if "input_ub" in i.op.name:
                TENSOR_MAP["tensor_input_ub"] = i
            if i.op.tag == "requant_scale" or i.op.tag == "requant_vector":
                self.requant_fusion = True
                TENSOR_MAP["requant_scale"] = i
            if i.op.tag == "requant_data_transfer":
                fusion_tensor_cub.append(i)
                TENSOR_MAP["requant_data_transfer"] = i

    def _set_reduce_fusion_flag(self, res):
        self.reduce_fusion = "reduce_sum" in res.op.tag
        self._print_debug(self.reduce_fusion, "reduce_fusion:")

    def _atomic_add(self):
        """
        atomic add according to refactor res
        """
        if not self.reduce_fusion:
            return
        res = self.root_tensor
        sch = self.sch
        # set all batch to ddr add
        block_dim_batch = self._get_value(self.TENSOR_MAP.get("c_l0c").shape)[0]
        batch_outer, _ = sch[res].split(res.op.reduce_axis[0], nparts = block_dim_batch)
        res_after = res
        res_ub = sch.rfactor(res, batch_outer)
        sch[res_ub].set_scope(tbe_platform_info.scope_ubuf)
        # put reduce axis first
        sch[res_after].reorder(sch[res_after].op.reduce_axis[0], *sch[res_after].op.axis)
        sch[res_ub].reorder(sch[res_ub].op.reduce_axis[0], *sch[res_ub].op.axis[1:])
        self.TENSOR_MAP["res_atomic_add_ub"] = res_ub
        self.tensors_in_cub.append(res_ub)
        self._print_ir_matmul("after atomic_add", sch)

    def _set_data_layout_fusion(self):
        TENSOR_MAP = self.TENSOR_MAP
        fusion_tensor_cub = self.fusion_tensor_cub
        compute_tensors = self._get_compute_tensor(self.res)

        self._set_reduce_fusion_flag(self.res)
        self._atomic_add()
        self._find_tensor_and_get_flag(compute_tensors)
        self.quantify_fusion = self.requant_fusion or self.dequant_fusion

        self._print_debug(self.quant_fusion, "quant_fusion")
        self._print_debug(self.requant_fusion, "requant_fusion")
        self._print_debug(self.dequant_fusion, "dequant_fusion")

        TENSOR_MAP["tensor_reform_by_vadds"] = self._match_and_get_tensor(compute_tensors, "reform_by_vadds")
        TENSOR_MAP["tensor_reform_by_vmuls"] = self._match_and_get_tensor(compute_tensors, "reform_by_vmuls")

        matmul_end_tensor = TENSOR_MAP.get("c_gm")
        self.matmul_tensors = self._get_compute_tensor(matmul_end_tensor)
        self._print_debug(self.matmul_tensors, "matmul_tensors")

        self.matmul_dequant_tensor = self._get_matmul_dequant_tensor()
        self._print_debug(self.matmul_dequant_tensor, "matmul_dequant_tensor")

        self.fusion_ele = self._get_elewise_fusion_tensor()
        self._print_debug(self.fusion_ele, "fusion_ele")
        (self.gm_ub,
         self.ele_header_ub_tensors,
         self.axpy_2_parent) = self._set_scope_buffer_type(self.placeholder_tensors)
        self._print_debug(self.elemwise_tensors, "elemwise_tensors")

        self._get_quant_fusion_tensor()
        if self.tensor_fusion_list is not None:
            fusion_tensor_cub += self.tensor_fusion_list
        self._print_debug(self.tensor_fusion_list, "tensor_fusion_list")

        self._get_matmul_dequant_activation_tensor()
        self._add_res_ub(self.dequant_activation_tensor)
        fusion_tensor_cub += self.dequant_activation_tensor
        self._print_debug(self.dequant_activation_tensor, "dequant_activation_tensor")

        self.header_ub_tensors = self._get_header_tensor_in_dequant_ew_fusion()
        fusion_tensor_cub += self.header_ub_tensors
        self._print_debug(self.header_ub_tensors, "header_ub_tensors")

        TENSOR_MAP["tensor_reform"] = self._get_reform_tensor()
        self._print_debug(TENSOR_MAP["tensor_reform"], "tensor_reform")

        self._get_fusion_tensor()
        fusion_tensor_cub += self.fusion_list
        self._print_debug(self.fusion_list, "fusion_list")

        tensor_a = TENSOR_MAP.get("a_placehold")
        tensor_b = TENSOR_MAP.get("b_placehold")
        is_fractal_a = len(tensor_a.shape) in (4, 5)
        is_fractal_b = len(tensor_b.shape) in (4, 5)
        self.in_addr_type = self._get_addr_type(tensor_a)
        l1_fusion_type = self._get_l1_fusion_type(tensor_a)
        self.input_l1_flag, self.input_l1_size = self._get_input_l1_paras(tensor_a)
        self._check_placeholders_shared(tensor_a, tensor_b, self.res)

        l1_fusion_and_l1_size_0 = self._get_l1_fusion_and_l1_size_0_flag(l1_fusion_type)
        self.l1_fusion_and_l1_size_0 = l1_fusion_and_l1_size_0
        tensor_a_l1_workspace = self._get_tensor_a_l1_workspace(l1_fusion_and_l1_size_0)
        self._set_l1_fusion_workspace_tensor(tensor_a, tensor_a_l1_workspace)
        self._set_l1_fusion_workspace_size(tensor_a_l1_workspace)
        self.tensor_a_l1_workspace = tensor_a_l1_workspace

        if self.mmad_mode == "gemv":
            tensor_a_l1 = self.TENSOR_MAP.get("b_l1")
            tensor_b_l1 = self.TENSOR_MAP.get("a_l1")
        else:
            tensor_a_l1 = self.TENSOR_MAP.get("a_l1")
            tensor_b_l1 = self.TENSOR_MAP.get("b_l1")

        self.a_l1_inline_flag = self._fc_tensor_a_l1_inline(tensor_a_l1, is_fractal_a, l1_fusion_and_l1_size_0)
        self.b_l1_inline_flag = self._fc_tensor_b_l1_inline(tensor_b_l1, is_fractal_b, l1_fusion_and_l1_size_0)

        self._set_scope_fusion()

        if self.fusion_ele:
            res_ub = self.sch.cache_write(self.res, tbe_platform_info.scope_ubuf)
            self.elemwise_tensors.append(res_ub)

        if self.res != TENSOR_MAP.get("c_gm") and not self.multi_output_flag:
            self.compute_inline_list.append(TENSOR_MAP.get("c_gm"))
        compute_inline_c_ub = self.quantify_fusion
        if compute_inline_c_ub:
            self.compute_inline_list.append(TENSOR_MAP.get("c_ub_fract"))

        fusion_tensor_cub += self.elemwise_tensors

    def _tensor_a_l1_workspace_emit(self):
        if self.input_l1_flag == 1:
            self._emit_insn_func(self.tensor_a_l1_workspace, 0, "dma_copy")

    def _fc_tensor_a_l1_inline(self, tensor_a_l1, is_fractal_a, l1_fusion_and_l1_size_0):
        inline_flag = False
        if ((self.in_addr_type == 1 or self.input_l1_flag == 1) and is_fractal_a) or l1_fusion_and_l1_size_0:
            self._add_tensor_to_list(tensor_a_l1, [self.compute_inline_list])
            inline_flag = True
        return inline_flag

    def _fc_tensor_b_l1_inline(self, tensor_b_l1, is_fractal_b, l1_fusion_and_l1_size_0):
        inline_flag = False
        if l1_fusion_and_l1_size_0 and is_fractal_b:
            self._add_tensor_to_list(tensor_b_l1, [self.compute_inline_list])
            inline_flag = True
        return inline_flag

    def _set_quant_by_params(self):
        sch = self.sch
        if not self.quantify_fusion:
            c_ub_fract = self.TENSOR_MAP.get("c_ub_fract")
            if c_ub_fract.op.attrs["scale_drq"].value == "ENABLE":
                # tensor_drq is second input for tensor_c_ub
                tensor_drq = c_ub_fract.op.input_tensors[1]
                tensor_drq_ub = sch.cache_read(tensor_drq, tbe_platform_info.scope_ubuf, [c_ub_fract])
                self._emit_insn_func(tensor_drq_ub, 0, "dma_copy")
                if c_ub_fract.op.attrs["sqrt_out"].value == "SQRT":
                    # Sqrt Mode
                    sch[c_ub_fract].pragma(c_ub_fract.op.axis[0], "deq_scale", "scalar_sqrt")
                else:
                    # No Sqrt Mode
                    sch[c_ub_fract].pragma(c_ub_fract.op.axis[0], "deq_scale", "scalar")

    def _get_batch_factors(self, tensor_a_l0a, tensor_b_l0b):
        if self.have_batch:
            if self.is_dynamic:
                batch = self.dynamic_batch
            else:
                batch = self._get_value(tensor_a_l0a.shape[0])
                if self.mmad_mode == "gemv":
                    batch = self._get_value(tensor_b_l0b.shape[0])
        else:
            batch = 0
        return batch

    def _set_l1_fusion_workspace_tensor(self, tensor_a, tensor_a_l1_workspace):
        if self.input_l1_flag == 0:
            L1CommonParam.l1_fusion_tensors_map = {}
            L1CommonParam.l1_fusion_tensors_map[tensor_a] = tvm.var("dummy")
        elif self.input_l1_flag == 1:
            L1CommonParam.l1_fusion_tensors_map = {}
            L1CommonParam.l1_fusion_tensors_map[tensor_a] = tensor_a_l1_workspace

    def _set_l1_fusion_workspace_size(self, tensor_a_l1_workspace):
        if self.input_l1_flag == 1 and self.input_l1_size > 0:
            self.sch[tensor_a_l1_workspace].set_storage_bound(self.input_l1_size)

    def _get_tensor_a_l1_workspace(self, l1_fusion_and_l1_size_0):
        tensor_a_l1_workspace = None
        sch = self.sch
        tensor_a_ub = self.TENSOR_MAP.get("a_ub")
        tensor_a_l1 = self.TENSOR_MAP.get("a_l1")
        tensor_a_l0a = self.TENSOR_MAP.get("a_l0a")
        tensor_a = self.TENSOR_MAP.get("a_placehold")
        if self.input_l1_flag == 1:
            if tensor_a_ub is not None:
                tensor_a_l1_workspace = sch.cache_read(tensor_a, tbe_platform_info.scope_cbuf_fusion, tensor_a_ub)
            elif tensor_a_l1 is not None and not l1_fusion_and_l1_size_0:
                tensor_a_l1_workspace = sch.cache_read(tensor_a, tbe_platform_info.scope_cbuf_fusion, tensor_a_l1)
            elif tensor_a_l0a is not None and l1_fusion_and_l1_size_0:
                tensor_a_l1_workspace = sch.cache_read(tensor_a, tbe_platform_info.scope_cbuf_fusion, tensor_a_l1)
        return tensor_a_l1_workspace

    def _get_l1_fusion_and_l1_size_0_flag(self, l1_fusion_type):

        trans_b = self.transpose_b
        is_l1fusion = l1_fusion_type in (0, 1)
        size = tbe_platform_info.get_soc_spec("L1_SIZE")
        if size == 0 and is_l1fusion:
            if trans_b:
                raise RuntimeError(
                    "If the size of L1 is zero, trans_b is not unexpected.")
            return True
        return False

    def _map_apend(self, input_map, key, value):
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

    def _check_placeholders_shared(self, tensor_a, tensor_b, res):
        """check placeholders shared"""
        matmul_tensors = self.matmul_tensors
        if not self.fusion_ele:
            return None

        in_out_tensor_map = {}
        self._gen_in_out_tensor_map(res, in_out_tensor_map)
        if tensor_a in in_out_tensor_map:
            for ten_i in in_out_tensor_map[tensor_a]:
                if ten_i not in matmul_tensors:
                    raise RuntimeError("matmul placeholders can't be shared "
                                    "with elementwise op")
        if tensor_b in in_out_tensor_map:
            for ten_i in in_out_tensor_map[tensor_b]:
                if ten_i not in matmul_tensors:
                    raise RuntimeError("matmul placeholders can't be shared "
                                    "with elementwise op")

    def _get_input_l1_paras(self, tensor):
        input_l1_flag = -1
        input_l1_size = None
        if 'L1_addr_flag' in tensor.op.attrs:
            input_l1_flag = tensor.op.attrs['L1_addr_flag'].value

        if input_l1_flag == 0:
            input_l1_size = -1
        elif input_l1_flag == 1:
            if 'L1_valid_size' in tensor.op.attrs:
                input_l1_size = tensor.op.attrs['L1_valid_size'].value
            else:
                input_l1_flag = -1
        else:
            pass

        return input_l1_flag, input_l1_size

    def _get_l1_fusion_type(self, tensor):
        l1_fusion_type = -1
        if "L1_fusion_type" in tensor.op.attrs:
            l1_fusion_type = tensor.op.attrs["L1_fusion_type"].value
        return l1_fusion_type

    def _get_fusion_tensor(self):
        matmul_tensors = self.matmul_tensors
        fusion_list = self.fusion_list
        tensor_c_gm = self.TENSOR_MAP.get("c_gm")
        if tensor_c_gm != self.res and tensor_c_gm is not None:
            for ten_in in self.compute_tensors:
                if ten_in == self.res:
                    continue
                if ten_in not in matmul_tensors:
                    fusion_list.append(ten_in)

    def _get_reform_tensor(self):
        tensor_reform_by_vadds = self.TENSOR_MAP.get("tensor_reform_by_vadds")
        tensor_reform_by_vmuls = self.TENSOR_MAP.get("tensor_reform_by_vmuls")
        requant_data_transfer = self.TENSOR_MAP.get("requant_data_transfer")
        if tensor_reform_by_vadds is not None:
            return tensor_reform_by_vadds
        if tensor_reform_by_vmuls is not None:
            return tensor_reform_by_vmuls
        if requant_data_transfer is not None:
            return requant_data_transfer
        return None

    def _get_header_tensor_in_dequant_ew_fusion(self):
        """
        add header_ub tensor to dequant_activation_tensor.
        """
        sch = self.sch
        dequant_activation_tensor = self.dequant_activation_tensor
        header_set = set(self.placeholder_tensors)
        header_ub_tensors = list()
        comm_2_elwt = dict()
        for ten_i in dequant_activation_tensor:
            common_tensors = header_set & set(ten_i.op.input_tensors)
            for common_tensor in common_tensors:
                if common_tensor in comm_2_elwt:
                    comm_2_elwt[common_tensor].append(ten_i)
                else:
                    comm_2_elwt[common_tensor] = [ten_i]
        for common_tensor, ten_in_list in comm_2_elwt.items():
            common_tensor_ub = sch.cache_read(
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
        dequant_activation_tensor = self.dequant_activation_tensor
        tensor_fusion_list = self.tensor_fusion_list
        if not self.dequant_fusion:
            return None
        TENSOR_MAP = self.TENSOR_MAP
        dequant_nz = TENSOR_MAP.get("dequant_nz")
        dequant_nd = TENSOR_MAP.get("dequant_nd")
        dequant_tensor = TENSOR_MAP.get("dequant_tensor")
        tensor_sqrt = TENSOR_MAP.get("tensor_sqrt")
        quant = TENSOR_MAP.get("quant")
        tensor_front_dequant = self._get_compute_tensor(dequant_tensor)
        for ten_in in self.compute_tensors:
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
        elemwise_tensors = self.elemwise_tensors
        if self.quantify_fusion:
            return False
        if self.reduce_fusion:
            real_res = self.res.op.input_tensors[0]
        else:
            real_res = self.res
        tensor_c_gm = self.TENSOR_MAP.get("c_gm")
        if tensor_c_gm != real_res and tensor_c_gm is not None:
            for ten_in in self.compute_tensors:
                if ten_in in (self.res, real_res):
                    continue
                if ten_in not in self.matmul_tensors and ten_in not in elemwise_tensors:
                    elemwise_tensors.append(ten_in)

            return True
        return False

    def _get_matmul_dequant_tensor(self):
        TENSOR_MAP = self.TENSOR_MAP
        if TENSOR_MAP.get("quant") is not None:
            compute_tensors = self._get_compute_tensor(TENSOR_MAP.get("dequant_nz"))
            return compute_tensors
        return list()

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
            src_dtype = self.TENSOR_MAP["a_placehold"].dtype
            dst_dtype = self.TENSOR_MAP["c_gm"].dtype
            if src_dtype in ("uint8", "int8") and dst_dtype == "int32":
                multi_m, multi_n = 2, 2
            if self.int8_not_double_m or self.format_a != "ND":
                multi_m = 1
            if self.format_b != "ND":
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
                'n_bef_batch_flag': 0,
                'n_bef_group_flag': 0,
                'batch_bef_group_flag': 0,
                'A_overhead_opt_flag': 0,
                'B_overhead_opt_flag': 0,
                'AUB_channel_wise_flag': None,
                'BUB_channel_wise_flag': None,
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
                }
            }
            if self.mmad_mode in ("gemv", "gevm") and (not self.only_use_gevm_gemv_flow):
                tiling["AUB_shape"] = [block_reduce * block_in, 1, 1, 1]
                tiling["AL1_shape"] = [block_reduce * block_in, 1, 1, 1]
                tiling["BL1_shape"] = [block_reduce * block_in, 1, 1, 1]
                tiling["AL0_matrix"] = [1, block_in, 1, block_reduce, 1, 1]
                tiling["BL0_matrix"] = [block_in, 1, block_out, block_reduce, 1, 1]
            b_ub = self.TENSOR_MAP.get("b_ub")
            if (b_ub is not None) and b_ub.dtype in ("int8", "uint8"):
                tiling["BUB_shape"][1] = tiling["BUB_shape"][1] * 2
        return tiling

    def _gemv_tiling(self, tiling):
        if self.mmad_mode != "gemv":
            return tiling

        tiling["AUB_shape"], tiling["BUB_shape"] = tiling["BUB_shape"], tiling["AUB_shape"]
        tiling["AL1_shape"], tiling["BL1_shape"] = tiling["BL1_shape"], tiling["AL1_shape"]
        block_dim = tiling.get("block_dim")
        block_dim[1] = 1
        tiling["block_dim"] = block_dim
        return tiling

    def _get_op_type_flag(self):
        """
        0: a and b both ND input
        1: a is fractal b is ND
        2: a is ND b is fractal
        3: a and b both fractal input
        """
        if (self.format_a == "ND" and self.format_b == "ND"):
            op_type_flag = 0
        elif (self.format_a != "ND" and self.format_b == "ND"):
            op_type_flag = 2
        elif (self.format_a == "ND" and self.format_b != "ND"):
            op_type_flag = 3
        else:
            op_type_flag = 1
        if self.mmad_mode == "gemv" and op_type_flag in (2, 3):
            change_value_dict = {
                2: 3,
                3: 2
            }
            op_type_flag = change_value_dict.get(op_type_flag)

        return op_type_flag

    def _tiling_process(self):
        """
        :param None:
        :return tiling result and data_byte
        info_dict
        -----------------------------------------------
        strideH: the data format A matrix and B matrix
            0: a and b both ND input
            1: a is fractal b is ND
            2: a is ND b is fractal
            3: a and b both fractal input
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

        op_type_flag = self._get_op_type_flag()
        tail_block = self._check_tail_block() if self.ops_format == "ND" else 1

        a_type = self.TENSOR_MAP["a_placehold"].dtype
        b_type = self.TENSOR_MAP["b_placehold"].dtype
        a_type, b_type = (b_type, a_type) if self.mmad_mode == "gemv" else (a_type, b_type)

        c_type = self.res.dtype
        a_shape, b_shape = self._get_tiling_param()
        a_ub_fuse_num, b_ub_fuse_num = self._compute_ab_buffer()

        not_count_list = []
        for tensor_in in self.compute_inline_list:
            if tensor_in not in self.placeholder_tensors:
                not_count_list.append(tensor_in)
        calculate_multi_ub = CalculateMultiUB(self.TENSOR_MAP.get("c_ub_fract"), self.res, not_count_list)
        ub_res = calculate_multi_ub.calculate_multi_ub_enter()
        fused_num = ub_res / self.BYTES_DTYPE[c_type] - 1
        # Distinguish between fused add non fused scenes with same input parameter
        if "te_fused_op_mat_mul_mul" in self.kernel_name:
            fused_num += 1

        self.fuse_num_group = [a_ub_fuse_num, b_ub_fuse_num, fused_num]
        mad_type = self.MAD_TYPE.get(str(self.ops_data_flow_mode))
        bias_flag = self.TENSOR_MAP.get("c_add_bias") is not None
        trans_flag = self._get_trans_flag(self.transpose_a, self.transpose_b)
        # in gemv or gevm, k need align to 256
        is_gevm = int(self.mmad_mode in ("gemv", "gevm")) and (not self.only_use_gevm_gemv_flow)
        info_dict = {
            "op_type": "matmul",
            "A_shape": a_shape,
            "B_shape": b_shape,
            "C_shape": None,
            "A_dtype": a_type,
            "B_dtype": b_type,
            "C_dtype": c_type,
            "mad_dtype": mad_type,
            "padl": a_ub_fuse_num,
            "padr": b_ub_fuse_num,
            "padu": is_gevm,
            "padd": int(self.int8_not_double_m and not self.transpose_a),
            "strideH": op_type_flag,
            "strideW": tail_block,
            "strideH_expand": 1,
            "strideW_expand": 1,
            "dilationH": trans_flag,
            "dilationW": 0 if self.compress_flag else 1,
            "group": 1,
            "bias_flag": bias_flag,
            "fused_double_operand_num": fused_num,
            "shape_a_align": self.align_a,
            "shape_b_align": self.align_b,
            "kernel_name": self.kernel_name
        }
        self._print_debug(info_dict, "info_dict")
        if self.is_dynamic:
            tiling = self.dynamic_para.get("tiling_strategy")
        else:
            tiling = get_tiling(info_dict)
        tiling = self._no_solution_tiling(tiling)
        tiling = self._gemv_tiling(tiling)
        tiling = self._check_k_full_load(tiling)
        if not tiling:
            args_dict = {
                "errCode": "E60114",
                "reason": "tiling is None",
                "value": "None"
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        self.tiling = tiling
        self._print_debug(tiling, "auto tiling result")

    def _check_k_full_load(self, tiling):
        if not self.is_dynamic:
            return tiling
        l0a_k = tiling.get("AL0_matrix")[1]
        l0c_n, l0c_m, _, _, _, _ = tiling.get("CL0_matrix")
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
        is_int82fp32_nd = ((self.format_a == "ND")
                           and (self.format_b == "ND")
                           and (self.ops_data_flow_mode == "int82fp32"))
        return is_int82fp32_nd

    def _get_tiling_param(self):
        a_type = self.TENSOR_MAP["a_placehold"].dtype
        b_type = self.TENSOR_MAP["b_placehold"].dtype
        c_type = self.TENSOR_MAP["c_gm"].dtype
        l0a_shape = [self._get_value(i) for i in self.TENSOR_MAP["a_l0a"].shape]
        l0b_shape = [self._get_value(i) for i in self.TENSOR_MAP["b_l0b"].shape]
        self._print_debug(l0a_shape, "l0a_shape")
        self._print_debug(l0b_shape, "l0b_shape")
        l0a_shape, l0b_shape = (l0b_shape, l0a_shape) if self.mmad_mode == "gemv" else (l0a_shape, l0b_shape)

        is_int82fp32_nd = self._is_int82fp32_nd()

        a_shape = [
            1,
            l0a_shape[-3],
            l0a_shape[-4],
            self.block_in,
            self.block_reduce
        ]
        if not self.is_dynamic:
            # becasuse A_shape dimension 2 only 16 bits
            while(a_shape[2] >= 65536):
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
        if self.ops_data_flow_mode == "int82fp32":
            a_shape[1] = a_shape[1] // 2
            a_shape[1] = a_shape[1] if a_shape[1] != 0 else 1
            a_shape[4] *= 2

        return a_shape, b_shape

    def _get_seed_shape(self):
        if not self.is_dynamic:
            return
        self.seed_shape = list(self.dynamic_para.get("m_k_n_shape"))
        self._print_debug(self.seed_shape, "seed_shape:")
        if self.mmad_mode == "gemv":
            self.seed_shape[0], self.seed_shape[2] = self.seed_shape[2], self.seed_shape[0]

        if len(self.seed_shape) == 4:
            self.dynamic_m, self.dynamic_k, self.dynamic_n, self.dynamic_batch = self.seed_shape
        else:
            self.dynamic_m, self.dynamic_k, self.dynamic_n = self.seed_shape
            self.dynamic_batch = None

    def _check_tail_block(self):

        n_shape = int(self.TENSOR_MAP["b_l0b"].shape[-3]) * self.block_out
        if self.ops_data_flow_mode in ("int82int32", "int82fp32"):
            divide_factor = 32
            data_num = 8
        elif self.ops_data_flow_mode == "fp162fp16":
            divide_factor = 16
            data_num = 16
        else:
            divide_factor = 16
            data_num = 8

        tail_block_flag = 0
        if 1 <= n_shape <= divide_factor or n_shape % divide_factor == 0:
            tail_block_flag = 1
        elif n_shape % divide_factor < data_num:
            tail_block_flag = 0
        return tail_block_flag

    def _get_trans_flag(self, transpose_a, transpose_b):
        trans_flag = 1
        if transpose_a:
            if transpose_b:
                trans_flag = 4
            else:
                trans_flag = 2
        elif transpose_b:
            trans_flag = 3

        return trans_flag

    def _tiling_l0_process(self):
        tiling = self.tiling
        if self.mmad_mode == "gemv":
            a_l0a = self.TENSOR_MAP.get("b_l0b")
            b_l0b = self.TENSOR_MAP.get("a_l0a")
        else:
            a_l0a = self.TENSOR_MAP.get("a_l0a")
            b_l0b = self.TENSOR_MAP.get("b_l0b")
        if tiling.get("BL0_matrix") != []:
            (
                self.bl0_tiling_kb,
                self.bl0_tiling_nb,
                self.bl0_tiling_n0,
                self.bl0_tiling_k0,
                self.bl0_tiling_batch,
                _
            ) = tiling.get("BL0_matrix")
        else:
            b_l0b_shape = list(self._get_value(i) for i in b_l0b.shape)
            b_l0b_shape = self._get_dynamic_l0b_shape(b_l0b_shape, self.have_batch_b)
            (
                self.bl0_tiling_kb,
                self.bl0_tiling_nb,
                self.bl0_tiling_n0,
                self.bl0_tiling_k0
            ) = b_l0b_shape[-4:]
            self.bl0_tiling_batch = (b_l0b_shape[0] if self.have_batch_b else 0) // tiling.get("block_dim")[0]
            self.bl0_tiling_nb = self.tiling.get("CL0_matrix")[0]
            self.bl0_tiling_kb = self.tiling.get("AL0_matrix")[1]
        self.bl0_tiling_k0 = self.block_reduce

        (
            self.al0_tiling_ma,
            self.al0_tiling_ka,
            self.al0_tiling_m0,
            self.al0_tiling_k0,
            self.al0_tiling_batch,
            _
        ) = self.tiling.get("AL0_matrix")
        self.al0_tiling_k0 = self.block_reduce
        (
            self.cl0_tiling_nc,
            self.cl0_tiling_mc,
            self.cl0_tiling_m0,
            self.cl0_tiling_n0,
            self.cl0_tiling_batch,
            _
        ) = self.tiling.get("CL0_matrix")
        c_l0c = self.TENSOR_MAP.get("c_l0c")

        self.c_col_k1, self.c_col_k0 = list(self._get_value(ax.dom.extent) for ax in c_l0c.op.reduce_axis)
        if self.is_dynamic:
            self.c_col_k1 = self.dynamic_k

        self.al0_tiling_m0 = tbe_platform.CUBE_MKN[a_l0a.dtype]["mac"][0]
        self.al0_tiling_k0 = tbe_platform.CUBE_MKN[a_l0a.dtype]["mac"][1]
        self.bl0_tiling_k0 = tbe_platform.CUBE_MKN[b_l0b.dtype]["mac"][1]
        self.bl0_tiling_n0 = tbe_platform.CUBE_MKN[b_l0b.dtype]["mac"][2]
        self.cl0_tiling_m0 = tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][0]
        self.cl0_tiling_n0 = tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][2]

        # special handle
        self.al0_tiling_batch = 1
        self.bl0_tiling_batch = 1
        self.cl0_tiling_batch = 1

    def _tiling_l1_process(self):
        self._tiling_al1_process()
        self._tiling_bl1_process()

    def _tiling_al1_process(self):
        tiling = self.tiling
        if tiling.get("AL1_shape") != [] and (tiling.get("AL1_shape") is not None):
            al1_tiling_k, al1_tiling_m, al1_tiling_batch, _ = tiling.get("AL1_shape")
            al1_tiling_m *= self.al0_tiling_ma
        else:
            if self.is_dynamic:
                al1_tiling_k = self.dynamic_k * self.block_reduce
                al1_ma = self.dynamic_m
            else:
                if self.mmad_mode != "gemv":
                    al0_shape = list(self._get_value(i) for i in self.TENSOR_MAP.get("a_l0a").shape)
                    al1_tiling_k = al0_shape[-3] * al0_shape[-1]
                    al1_ma = al0_shape[-4]
                else:
                    al0_shape = list(self._get_value(i) for i in self.TENSOR_MAP.get("b_l0b").shape)
                    al1_tiling_k = al0_shape[-4] * al0_shape[-1]
                    al1_ma = al0_shape[-3]

            al1_tiling_m = (al1_ma + tiling.get("block_dim")[2] - 1) // tiling.get("block_dim")[2]
            al1_tiling_batch = 0
            if self.have_batch_a:
                al1_tiling_batch = self.dynamic_batch if self.is_dynamic else al0_shape[0]

        al1_tiling_batch = 1
        self.al1_tiling_batch = al1_tiling_batch
        self.al1_tiling_k = al1_tiling_k
        self.al1_tiling_m = al1_tiling_m

    def _tiling_bl1_process(self):
        tiling = self.tiling
        b_l1 = self.TENSOR_MAP.get("b_l1") if self.mmad_mode != "gemv" else self.TENSOR_MAP.get("a_l1")
        b_l1_shape = list(self._get_value(i) for i in b_l1.shape)

        if tiling.get("BL1_shape") != [] and (tiling.get("BL1_shape") is not None):
            bl1_tiling_k, bl1_tiling_n, bl1_tiling_batch, _ = tiling.get("BL1_shape")
            bl1_tiling_n *= self.bl0_tiling_nb
        else:
            if self.is_dynamic:
                bl1_tiling_k = self.dynamic_k * self.block_reduce
                bl1_n =self.dynamic_n
            else:
                if self.mmad_mode != "gemv":
                    bl0_shape = list(self._get_value(i) for i in self.TENSOR_MAP.get("b_l0b").shape)
                    bl1_tiling_k = bl0_shape[-4] * bl0_shape[-1]
                    bl1_n = bl0_shape[-3]
                else:
                    bl0_shape = list(self._get_value(i) for i in self.TENSOR_MAP.get("a_l0a").shape)
                    bl1_tiling_k = bl0_shape[-3] * bl0_shape[-1]
                    bl1_n = bl0_shape[-4]

            bl1_tiling_n = (bl1_n + tiling.get("block_dim")[1] - 1) // tiling.get("block_dim")[1]
            bl1_tiling_batch = 0
            if self.have_batch_b:
                bl1_tiling_batch = self.dynamic_batch if self.is_dynamic else bl0_shape[0]

        bl1_tiling_batch = 1
        self.bl1_tiling_batch = bl1_tiling_batch
        self.bl1_tiling_k = bl1_tiling_k
        self.bl1_tiling_n = bl1_tiling_n

    def _tiling_ub_process(self):
        if self.format_a == "ND" or self.ops_data_flow_mode == "int82fp32":
            self.aub_tiling_k, self.aub_tiling_m, self.aub_tiling_batch = self.tiling.get("AUB_shape")[:3]
            self.aub_tiling_batch = 1
        else:
            self.aub_tiling_m, self.aub_tiling_k, self.aub_tiling_batch = 0, 0, 0

        if self.format_b == "ND" or self.ops_data_flow_mode == "int82fp32":
            self.bub_tiling_k, self.bub_tiling_n, self.bub_tiling_batch = self.tiling.get("BUB_shape")[:3]
            self.bub_tiling_batch = 1
        else:
            self.bub_tiling_k, self.bub_tiling_n, self.bub_tiling_batch = 0, 0, 0

    def _get_dynamic_cub_shape(self, cub_shape, have_batch):
        if self.is_dynamic:
            cub_shape = [self.dynamic_n, self.dynamic_m, self.block_in, self.block_out]
            if have_batch:
                cub_shape.insert(0, self.dynamic_batch)

        return cub_shape

    def _cub_process(self):
        self._print_debug("-------debug info in cub_process-------")
        sch = self.sch
        cub_tiling = self.tiling.get("CUB_matrix")
        cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0, cub_tiling_batch, _ = cub_tiling
        if self.res.dtype == "int8":
            cub_tiling_nc_factor = self._int_ceil_div(cub_tiling_nc_factor, 2)
        if self.format_out == "ND":
            affine_cub = [
                cub_tiling_mc_factor * cub_tiling_m0,
                cub_tiling_nc_factor * cub_tiling_n0
            ]
        else:
            affine_cub = [cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, cub_tiling_n0]
        c_ub_tiling_shape = [
            cub_tiling_nc_factor,
            cub_tiling_mc_factor,
            cub_tiling_m0,
            cub_tiling_n0
        ]
        if self.have_batch:
            c_ub_tiling_shape.insert(0, cub_tiling_batch)
            affine_cub.insert(0, cub_tiling_batch)

        c_ub_fract = self.TENSOR_MAP.get("c_ub_fract")
        c_ub_shape = list(self._get_value(i) for i in c_ub_fract.shape)
        c_ub_shape = self._get_dynamic_cub_shape(c_ub_shape, self.have_batch)

        if self.mmad_mode in ("gemv", "gevm"):
            c_ub_tiling_shape[-2] = 1
        status = Compare.compare(c_ub_tiling_shape, c_ub_shape)
        self._print_debug([c_ub_tiling_shape, c_ub_shape], "c_ub_tiling_shape with c_ub_shape")
        self._print_debug([affine_cub, self.root_tensor.shape], "affine_cub with root_tensor's shape")
        self.c_ub_attach_status = "full_load"
        self._do_attach_cub(status, c_ub_fract, affine_cub)
        self._print_debug("-------debug info in cub_process end-------")
        return c_ub_tiling_shape

    def _do_attach_cub(self, status, c_ub_fract, affine_cub):
        sch_agent = self.sch_agent
        if self.fusion_tensor_cub:
            self.tensors_in_cub += self.fusion_tensor_cub
        same_attach_cub = self.tensors_in_cub

        special_handle_dynamic = status == Compare.EQUAL and self.is_dynamic and self.tiling.get("CUB_matrix") != []
        if special_handle_dynamic:
            status = Compare.LESS_EQ
        if status == Compare.EQUAL:
            pass
        elif status == Compare.LESS_EQ:
            if self.quant_fusion or self.requant_fusion:
                affine_cub[-1] *= 2
                self._print_debug(affine_cub, "affine_cub in quant_fusion and requant_fusion")
            sch_agent.attach_at(c_ub_fract, self.root_tensor, affine_shape = affine_cub)
            self.c_ub_attach_status = "c_gm"

            for tensor in same_attach_cub:
                if (tensor == c_ub_fract) or (tensor == self.root_tensor):
                    continue
                sch_agent.same_attach(tensor, c_ub_fract)

        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "c_ub attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    def _get_dynamic_l0c_shape(self, l0c_shape, have_batch):
        if self.is_dynamic:
            l0c_shape = [self.dynamic_n, self.dynamic_m, self.block_in, self.block_out]
            if have_batch:
                l0c_shape.insert(0, self.dynamic_batch)

        return l0c_shape

    def _cl0_process(self, c_ub_tiling_shape):
        self._print_debug("-------debug info in cl0_process-------")
        sch_agent = self.sch_agent

        cl0_tiling_nc, cl0_tiling_mc =self.cl0_tiling_nc, self.cl0_tiling_mc
        cl0_tiling_m0, cl0_tiling_n0 =self.cl0_tiling_m0, self.cl0_tiling_n0

        if self.format_out == "ND":
            affine_l0c = [cl0_tiling_mc * cl0_tiling_m0, cl0_tiling_nc * cl0_tiling_n0]
        else:
            affine_l0c = [cl0_tiling_nc, cl0_tiling_mc, cl0_tiling_m0, cl0_tiling_n0]

        cl0_tiling_shape = [cl0_tiling_nc, cl0_tiling_mc, cl0_tiling_m0, cl0_tiling_n0]

        if self.have_batch:
            affine_l0c.insert(0, self.cl0_tiling_batch)
            cl0_tiling_shape.insert(0, self.cl0_tiling_batch)

        c_l0c = self.TENSOR_MAP.get("c_l0c")
        c_l0c_shape = list(self._get_value(i) for i in c_l0c.shape)
        c_l0c_shape = self._get_dynamic_l0c_shape(c_l0c_shape, self.have_batch)
        status_ori = Compare.compare(cl0_tiling_shape, c_l0c_shape)
        if self.mmad_mode in ("gemv", "gevm"):
            # add for dsl_mat_d_elm-eltwise-ut8_NZ_0054
            c_ub_tiling_shape[-2] = 16
        status = Compare.compare(cl0_tiling_shape, c_ub_tiling_shape)
        self._print_debug([cl0_tiling_shape, c_l0c_shape], "cl0_tiling_shape with c_l0c_shape")
        self._print_debug([cl0_tiling_shape, c_ub_tiling_shape], "cl0_tiling_shape with c_ub_tiling_shape")
        self._print_debug([affine_l0c, self.root_tensor.shape], "affine_l0c with root_tensor.shape")
        self.c_l0c_attach_status = "full_load"
        self._do_attach_cl0(status_ori, status, c_l0c, affine_l0c)
        self._print_debug("-------debug info in cl0_process end -------")

    def _do_attach_cl0(self, status_ori, status, c_l0c, affine_l0c):
        sch_agent = self.sch_agent
        self.c_l0c_attach_status = "full_load"
        special_handle_dynamic = (status_ori == Compare.EQUAL
            and self.is_dynamic and self.tiling.get("CL0_matrix") != [])
        if special_handle_dynamic:
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
            self.c_l0c_attach_status = self.c_ub_attach_status
            c_ub_fract = self.TENSOR_MAP["c_ub_fract"]
            sch_agent.same_attach(c_l0c, c_ub_fract)
        elif status == Compare.GREATE_EQ:
            if self.quant_fusion or self.requant_fusion:
                affine_l0c[-1] *= 2
                affine_l0c[-4] //= 2

            sch_agent.attach_at(c_l0c, self.root_tensor, affine_shape = affine_l0c)
            self.c_l0c_attach_status = "c_gm"
        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "tensor_c_l0c attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

        for tensor in self.tensors_in_l0c:
            if tensor == c_l0c:
                continue
            sch_agent.same_attach(tensor, c_l0c)

    def _get_dynamic_l0a_shape(self, l0a_shape, have_batch):
        if self.is_dynamic:
            l0a_shape = [self.dynamic_m, self.dynamic_k, self.block_in, self.block_reduce]
            if have_batch:
                l0a_shape.insert(0, self.dynamic_batch)

        return l0a_shape

    def _get_affine_m_full_load_dynamic(self):
        # now only support fractal
        tiling_m, _, _, _ = self.TENSOR_MAP.get("a_l0a").shape
        tiling_m = self._int_ceil_div(tiling_m, self.tiling["block_dim"][2])
        return tiling_m

    def _get_affine_n_full_load_dynamic(self):
        # now only support fractal
        _, tiling_n, _, _ = self.TENSOR_MAP.get("b_l0b").shape
        tiling_n = self._int_ceil_div(tiling_n, self.tiling["block_dim"][1])
        return tiling_n

    def _get_affine_k_full_load_dynamic(self):
        _, tiling_k, _, _ = self.TENSOR_MAP.get("a_l0a").shape
        return tiling_k

    def _l0a_process(self):
        self._print_debug("-------debug info in l0a_process-------")
        not_need_process = (self.tiling.get("AL0_matrix") == []) and (not self.is_dynamic)
        not_need_process = not_need_process and self.have_batch_a
        if not_need_process:
            return
        sch_agent = self.sch_agent
        al0_tiling_ma, al0_tiling_ka = self.al0_tiling_ma, self.al0_tiling_ka
        al0_tiling_m0, al0_tiling_k0 = self.al0_tiling_m0, self.al0_tiling_k0
        cl0_tiling_nc, cl0_tiling_mc = self.cl0_tiling_nc, self.cl0_tiling_mc
        cl0_tiling_m0, cl0_tiling_n0 = self.cl0_tiling_m0, self.cl0_tiling_n0

        l0a2l0c_affine_shape = [
            None,
            al0_tiling_ma,
            None,
            cl0_tiling_n0,
            al0_tiling_ka,
            al0_tiling_k0
        ]
        tiling_ori_l0a = [al0_tiling_ma, al0_tiling_ka, al0_tiling_m0, al0_tiling_k0]
        a_l0a = self.TENSOR_MAP.get("a_l0a")
        if self.mmad_mode == "gemv":
            a_l0a = self.TENSOR_MAP.get("b_l0b")
        else:
            a_l0a = self.TENSOR_MAP.get("a_l0a")
        a_l0a_shape = list(self._get_value(i) for i in a_l0a.shape)
        a_l0a_shape = self._get_dynamic_l0a_shape(a_l0a_shape, self.have_batch_a)
        al0_tiling_shape = [al0_tiling_ma, al0_tiling_m0, al0_tiling_ka, al0_tiling_k0]
        cl0_tiling_shape = [cl0_tiling_mc, cl0_tiling_m0, self.c_col_k1, self.c_col_k0]
        if self.format_out == "ND":
            l0a2out_affine_shape = [al0_tiling_ma * al0_tiling_m0, None]
        else:
            l0a2out_affine_shape = [None, al0_tiling_ma, al0_tiling_m0, None]
        if self.have_batch_a:
            l0a2l0c_affine_shape.insert(0, self.al0_tiling_batch)
            tiling_ori_l0a.insert(0, self.al0_tiling_batch)
            al0_tiling_shape.insert(0, self.al0_tiling_batch)
            l0a2out_affine_shape.insert(0, self.al0_tiling_batch)
            cl0_tiling_shape.insert(0, self.cl0_tiling_batch)
        elif self.have_batch:
            l0a2l0c_affine_shape.insert(0, None)
            l0a2out_affine_shape.insert(0, None)

        if self.mmad_mode in ("gevm", "gemv") and not self.only_use_gevm_gemv_flow:
            tiling_ori_l0a[-2] = 1

        status_ori = Compare.compare(tiling_ori_l0a, a_l0a_shape)
        status = Compare.compare(al0_tiling_shape, cl0_tiling_shape)
        self._print_debug([tiling_ori_l0a, a_l0a_shape], "tiling_ori_l0a with a_l0a_shape")
        self._print_debug([al0_tiling_shape, cl0_tiling_shape], "al0_tiling_shape with cl0_tiling_shape")
        self._do_attach_l0a(status_ori, status, a_l0a, l0a2l0c_affine_shape, l0a2out_affine_shape)
        self._print_debug("-------debug info in l0a_process end-------")

    def _do_attach_l0a(self, status_ori, status, a_l0a, l0a2l0c_affine_shape, l0a2out_affine_shape):
        sch_agent = self.sch_agent
        special_handle_dynamic = (status_ori == Compare.EQUAL
            and self.is_dynamic and self.tiling.get("AL0_matrix") != [])
        if special_handle_dynamic:
            sch_agent.attach_at(a_l0a, self.root_tensor, affine_shape = l0a2out_affine_shape)
            return
        elif status_ori == Compare.MISC:
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
            sch_agent.same_attach(a_l0a, self.TENSOR_MAP.get("c_l0c"))
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(a_l0a, self.TENSOR_MAP.get("c_l0c"), affine_shape = l0a2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            sch_agent.attach_at(a_l0a, self.root_tensor, affine_shape = l0a2out_affine_shape)
        else:
            args_dict = {
                "errCode": "E60114",
                "reason": "l0a attach error.",
                "value": "compare status = {}".format(status)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    def _get_dynamic_l0b_shape(self, l0b_shape, have_batch):
        if self.is_dynamic:
            l0b_shape = [self.dynamic_k, self.dynamic_n, self.block_out, self.block_reduce]
            if have_batch:
                l0b_shape.insert(0, self.dynamic_batch)
        return l0b_shape

    def _l0b_process(self):
        self._print_debug("-------debug info in l0b_process-------")

        bl0_tiling_kb, bl0_tiling_nb = self.bl0_tiling_kb, self.bl0_tiling_nb
        bl0_tiling_n0, bl0_tiling_k0 = self.bl0_tiling_n0, self.bl0_tiling_k0
        cl0_tiling_mc, cl0_tiling_nc = self.cl0_tiling_mc, self.cl0_tiling_nc
        cl0_tiling_m0, cl0_tiling_n0 = self.cl0_tiling_m0, self.cl0_tiling_n0
        l0b2l0c_affine_shape = [
            bl0_tiling_nb,
            None,
            None,
            bl0_tiling_n0,
            bl0_tiling_kb,
            bl0_tiling_k0
        ]
        if self.format_out == "ND":
            l0b2out_affine_shape = [None, bl0_tiling_nb * bl0_tiling_n0]
        else:
            l0b2out_affine_shape = [bl0_tiling_nb, None, None, bl0_tiling_n0]

        tiling_ori_l0b = [bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_k0]
        bl0_tiling_shape = [bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_kb, bl0_tiling_k0]

        cl0_tiling_shape = [cl0_tiling_nc, cl0_tiling_n0, self.c_col_k1, self.c_col_k0]
        if self.mmad_mode == "gemv":
            b_l0b = self.TENSOR_MAP.get("a_l0a")
        else:
            b_l0b = self.TENSOR_MAP.get("b_l0b")
        b_l0b_shape = list(self._get_value(i) for i in b_l0b.shape)
        b_l0b_shape = self._get_dynamic_l0b_shape(b_l0b_shape, self.have_batch_b)
        if self.have_batch_b:
            l0b2l0c_affine_shape.insert(0, self.bl0_tiling_batch)
            tiling_ori_l0b.insert(0, self.bl0_tiling_batch)
            l0b2out_affine_shape.insert(0, self.bl0_tiling_batch)
            bl0_tiling_shape.insert(0, self.bl0_tiling_batch)
            cl0_tiling_shape.insert(0, self.cl0_tiling_batch)
        elif self.have_batch:
            l0b2l0c_affine_shape.insert(0, None)
            l0b2out_affine_shape.insert(0, None)

        self._print_debug([tiling_ori_l0b, b_l0b_shape], "tiling_ori_l0b, b_l0b_shape")
        self._print_debug([bl0_tiling_shape, cl0_tiling_shape], "bl0_tiling_shape, cl0_tiling_shape")
        status_ori = Compare.compare(tiling_ori_l0b, b_l0b_shape)
        status = Compare.compare(bl0_tiling_shape, cl0_tiling_shape)
        self._do_attach_l0b(status_ori, status, b_l0b, l0b2l0c_affine_shape, l0b2out_affine_shape)
        self._print_debug("-------debug info in l0b_process end-------")

    def _do_attach_l0b(self, status_ori, status, b_l0b, l0b2l0c_affine_shape, l0b2out_affine_shape):
        sch_agent = self.sch_agent
        special_handle_dynamic = (status_ori == Compare.EQUAL and self.is_dynamic)
        if special_handle_dynamic:
            sch_agent.attach_at(b_l0b, self.root_tensor, affine_shape = l0b2out_affine_shape)
            return
        elif status_ori == Compare.MISC:
            args_dict = {
                "errCode": "E60114",
                "reason": "b_l0b attach error.",
                "value": "compare status = {}".format(status_ori)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        elif status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            sch_agent.same_attach(b_l0b, self.TENSOR_MAP.get("c_l0c"))
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(b_l0b, self.TENSOR_MAP.get("c_l0c"), affine_shape = l0b2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            l0b2out_affine_shape = self._fix_affine_out_int8(b_l0b.dtype, l0b2out_affine_shape)
            sch_agent.attach_at(b_l0b, self.root_tensor, affine_shape = l0b2out_affine_shape)
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
        self.al1_attach_status = "full_load"
        not_need_process = self.tiling.get("AL1_shape") in (None, []) and (not self.is_dynamic)
        not_need_process = not_need_process and (not self.have_batch_a)
        if not_need_process:
            return
        self._print_debug("-------debug info in al1_process-------")
        al1_tiling_m, al1_tiling_k = self.al1_tiling_m, self.al1_tiling_k
        al0_tiling_ma, al0_tiling_m0 = self.al0_tiling_ma, self.al0_tiling_m0
        al0_tiling_k0 = self.al0_tiling_k0
        cl0_tiling_mc = self.cl0_tiling_mc

        cl0_tiling_m0, cl0_tiling_n0 = self.cl0_tiling_m0, self.cl0_tiling_n0

        l1_ma = al1_tiling_m
        l1_ka = (al1_tiling_k + al0_tiling_k0 - 1) // al0_tiling_k0

        a_l1 = self.TENSOR_MAP.get("b_l1") if self.mmad_mode == "gemv" else self.TENSOR_MAP.get("a_l1")
        tiling_ori_al1 = [l1_ma, l1_ka]
        if self.is_dynamic:
            m_shape = self.dynamic_m
            k_shape = self.dynamic_k
        else:
            a_l0a = self.TENSOR_MAP.get("b_l0b") if self.mmad_mode == "gemv" else self.TENSOR_MAP.get("a_l0a")
            a_l0a_shape =[self._get_value(i) for i in a_l0a.shape]
            m_shape = a_l0a_shape[-4]
            k_shape = a_l0a_shape[-3]
        al1_shape = [m_shape, k_shape]
        al1_shape[0] = self._int_ceil_div(al1_shape[0], self.tiling.get("block_dim")[2])

        l1a2l0c_affine_shape = [
            None,
            l1_ma,
            None,
            cl0_tiling_n0,
            l1_ka,
            al0_tiling_k0
        ]

        al1_tiling_shape = [l1_ma, al0_tiling_m0, l1_ka, al0_tiling_k0]

        if self.format_out == "ND":
            l1a2out_affine_shape = [l1_ma * al0_tiling_m0, None]
        else:
            # add bl1_tiling_n in order to out n_inner axis down
            l1a2out_affine_shape = [self.bl1_tiling_n, l1_ma, al0_tiling_m0, None]

        tiling_ori_al1[-2] = 1 if self.mmad_mode in ("gevm", "gemv") else tiling_ori_al1[-2]
        cl0_tiling_shape = [cl0_tiling_mc, cl0_tiling_m0, self.c_col_k1, self.c_col_k0]
        if self.have_batch_a:
            al1_ori_batch = self.dynamic_batch if self.is_dynamic else a_l1.shape[0].value
            al1_shape.insert(0, al1_ori_batch)
            al1_shape[0] = (al1_shape[0] + self.tiling.get("block_dim")[0] - 1) // self.tiling.get("block_dim")[0]
            tiling_ori_al1.insert(0, self.al1_tiling_batch)
            l1a2l0c_affine_shape.insert(0, self.al1_tiling_batch)
            l1a2out_affine_shape.insert(0, self.al1_tiling_batch)
            al1_tiling_shape.insert(0, self.al1_tiling_batch)
            cl0_tiling_shape.insert(0, self.cl0_tiling_batch)
        elif self.have_batch:
            l1a2l0c_affine_shape.insert(0, None)
            l1a2out_affine_shape.insert(0, None)

        status = Compare.compare(al1_tiling_shape, cl0_tiling_shape)
        status_ori = Compare.compare(tiling_ori_al1, al1_shape)
        self._print_debug([al1_tiling_shape, cl0_tiling_shape], "al1_tiling_shape with cl0_tiling_shape")
        self._print_debug([tiling_ori_al1, al1_shape], "tiling_ori_al1 with al1_shape")
        self._do_attach_al1(status_ori, status, a_l1, l1a2l0c_affine_shape, l1a2out_affine_shape)
        self._print_debug("-------debug info in al1_process end-------")

    def _do_attach_al1(self, status_ori, status, a_l1, l1a2l0c_affine_shape, l1a2out_affine_shape):
        sch_agent = self.sch_agent
        special_handle_dynamic = (status_ori == Compare.EQUAL
            and self.is_dynamic and self.tiling.get("AL1_shape") != [])
        if special_handle_dynamic:
            sch_agent.attach_at(a_l1, self.root_tensor, affine_shape = l1a2out_affine_shape)
            self.al1_attach_status = "c_gm"
            return
        elif status_ori == Compare.MISC:
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
            self.al1_attach_status = self.c_l0c_attach_status
            sch_agent.same_attach(a_l1, self.TENSOR_MAP.get("c_l0c"))
        elif status == Compare.LESS_EQ:
            self.al1_attach_status = "c_l0c"
            sch_agent.attach_at(a_l1, self.TENSOR_MAP.get("c_l0c"), affine_shape = l1a2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            self.al1_attach_status = "c_gm"
            sch_agent.attach_at(a_l1, self.root_tensor, affine_shape = l1a2out_affine_shape)
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
        self.bl1_attach_status = "full_load"
        not_need_bl1_process = self.tiling.get("BL1_shape") in (None, []) and (not self.is_dynamic)
        not_need_bl1_process = not_need_bl1_process and (not self.have_batch_b)
        if not_need_bl1_process:
            return
        self._print_debug("-------debug info in bl1_process-------")
        bl1_tiling_n, bl1_tiling_k = self.bl1_tiling_n, self.bl1_tiling_k
        bl0_tiling_nb, bl0_tiling_n0 = self.bl0_tiling_nb, self.bl0_tiling_n0
        bl0_tiling_k0 = self.bl0_tiling_k0

        cl0_tiling_nc = self.cl0_tiling_nc
        cl0_tiling_m0, cl0_tiling_n0 = self.cl0_tiling_m0, self.cl0_tiling_n0

        l1_nb = bl1_tiling_n
        l1_kb = (bl1_tiling_k + bl0_tiling_k0 - 1) // bl0_tiling_k0

        l1b2l0c_affine_shape = [
            l1_nb,
            None,
            None,
            bl0_tiling_n0,
            l1_kb,
            bl0_tiling_k0
        ]

        bl1_tiling_shape = [l1_nb, bl0_tiling_n0, l1_kb, bl0_tiling_k0]

        if self.format_out == "ND":
            l1b2out_affine_shape = [None, l1_nb * bl0_tiling_n0]
        else:
            l1b2out_affine_shape = [l1_nb, None, None, bl0_tiling_n0]
        cl0_tiling_shape = [cl0_tiling_nc, cl0_tiling_n0, self.c_col_k1, self.c_col_k0]

        b_l1 = self.TENSOR_MAP.get("a_l1") if self.mmad_mode == "gemv" else self.TENSOR_MAP.get("b_l1")
        tiling_ori_bl1 = [l1_kb, l1_nb]
        if self.is_dynamic:
            n_shape = self.dynamic_n
            k_shape = self.dynamic_k
        else:
            b_l0b = self.TENSOR_MAP.get("a_l0a") if self.mmad_mode == "gemv" else self.TENSOR_MAP.get("b_l0b")
            b_l0b_shape = [self._get_value(i) for i in b_l0b.shape]
            n_shape = b_l0b_shape[-3]
            k_shape = b_l0b_shape[-4]

        bl1_shape = [k_shape, n_shape]
        bl1_shape[1] = self._int_ceil_div(bl1_shape[1], self.tiling.get("block_dim")[1])

        if self.have_batch_b:
            bl1_ori_batch = self.dynamic_batch if self.is_dynamic else b_l1.shape[0].value
            bl1_shape.insert(0, bl1_ori_batch)
            bl1_shape[0] = self._int_ceil_div(bl1_shape[0], self.tiling.get("block_dim")[0])
            l1b2l0c_affine_shape.insert(0, self.bl1_tiling_batch)
            tiling_ori_bl1.insert(0, self.bl1_tiling_batch)
            bl1_tiling_shape.insert(0, self.bl1_tiling_batch)
            l1b2out_affine_shape.insert(0, self.bl1_tiling_batch)
            cl0_tiling_shape.insert(0, self.cl0_tiling_batch)
        elif self.have_batch:
            l1b2l0c_affine_shape.insert(0, None)
            l1b2out_affine_shape.insert(0, None)

        self._print_debug([bl1_tiling_shape, cl0_tiling_shape], "bl1_tiling_shape with cl0_tiling_shape")
        self._print_debug([tiling_ori_bl1, bl1_shape], "tiling_ori_bl1 with bl1_shape")
        status = Compare.compare(bl1_tiling_shape, cl0_tiling_shape)
        status_ori = Compare.compare(tiling_ori_bl1, bl1_shape)
        self._do_attach_bl1(status_ori, status, b_l1, l1b2l0c_affine_shape, l1b2out_affine_shape)
        self._print_debug("-------debug info in bl1_process end-------")

    def _do_attach_bl1(self, status_ori, status, b_l1, l1b2l0c_affine_shape, l1b2out_affine_shape):
        sch_agent = self.sch_agent
        if status_ori == Compare.EQUAL and self.is_dynamic:
            sch_agent.attach_at(b_l1, self.root_tensor, affine_shape = l1b2out_affine_shape)
            self.bl1_attach_status = "c_gm"
            return
        elif status_ori == Compare.MISC:
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
            self.bl1_attach_status = self.c_l0c_attach_status
            sch_agent.same_attach(b_l1, self.TENSOR_MAP.get("c_l0c"))
        elif status == Compare.LESS_EQ:
            self.bl1_attach_status = "c_l0c"
            sch_agent.attach_at(b_l1, self.TENSOR_MAP.get("c_l0c"), affine_shape = l1b2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            self.bl1_attach_status = "c_gm"
            l1b2out_affine_shape = self._fix_affine_out_int8(b_l1.dtype, l1b2out_affine_shape)
            sch_agent.attach_at(b_l1, self.root_tensor, affine_shape = l1b2out_affine_shape)
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
            n_dim_index = -1 if self.format_out == "ND" else -4
            n_factor = affine_shape[n_dim_index] // 2
            affine_shape[n_dim_index] = 1 if n_factor == 0 else n_factor
        return affine_shape

    def _renew_aub_m(self, a_ub_ori_shape):
        index_offset = 1 if self.have_batch_a else 0
        if self.format_a == "ND":
            if self.transpose_a:
                a_ub_ori_shape[1 + index_offset] = self._int_ceil_div(a_ub_ori_shape[1 + index_offset],
                    self.tiling.get("block_dim")[2])
            else:
                a_ub_ori_shape[index_offset] = self._int_ceil_div(a_ub_ori_shape[index_offset],
                    self.tiling.get("block_dim")[2])
        else:
            if self.transpose_a:
                a_ub_ori_shape[index_offset] = self._int_ceil_div(a_ub_ori_shape[index_offset],
                    self.tiling.get("block_dim")[2])
            else:
                a_ub_ori_shape[1 + index_offset] = self._int_ceil_div(a_ub_ori_shape[1 + index_offset],
                    self.tiling.get("block_dim")[2])

    def _get_dynamic_aub_shape(self, aub_shape, aub_tiling, have_batch):
        if self.is_dynamic:
            aub_shape = [self.dynamic_m, self.dynamic_k * self.block_reduce]
            aub_tiling = [self.aub_tiling_m, self.aub_tiling_k]
            if have_batch:
                aub_shape.insert(0, self.dynamic_batch)
                aub_tiling.insert(0, self.aub_tiling_batch)

        return aub_shape, aub_tiling

    def _aub_process(self):

        self._print_debug("-------debug info in aub_process-------")
        a_ub = self.TENSOR_MAP.get("a_ub")
        if a_ub in (None, []):
            return
        sch_agent = self.sch_agent
        transpose_a = self.transpose_a
        aub_tiling_k, aub_tiling_m = self.aub_tiling_k, self.aub_tiling_m
        aub_tiling_k0 = self.block_reduce
        aub_tiling_m0 = 1 if self.mmad_mode == "gevm" else self.block_in
        l1_ma, al1_tiling_k = self.al1_tiling_m, self.al1_tiling_k
        al0_tiling_ma, al0_tiling_m0, al0_tiling_k0 = self.al0_tiling_ma, self.al0_tiling_m0, self.al0_tiling_k0
        cl0_tiling_nc, cl0_tiling_mc = self.cl0_tiling_nc, self.cl0_tiling_mc
        cl0_tiling_m0, cl0_tiling_n0 = self.cl0_tiling_m0, self.cl0_tiling_n0
        l1_ka = (al1_tiling_k + al0_tiling_k0 - 1) // al0_tiling_k0
        ub_ka = (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0

        a_ub_ori_shape = list(self._get_value(i) for i in a_ub.shape)
        if self.get_a_matrix_mode == "nd2Zz_vnchwconv":
            if transpose_a:
                tiling_ori_aub = [
                    ub_ka * aub_tiling_k0,
                    aub_tiling_m * aub_tiling_m0
                ]
                tiling_ori_al1 = [l1_ka, l1_ma * al0_tiling_m0, al0_tiling_k0]
                aub_l1_affine_shape = [ub_ka, aub_tiling_m * aub_tiling_m0, aub_tiling_k0]
                tiling_ori_aub_with_l1 = [ub_ka, aub_tiling_m * aub_tiling_m0, aub_tiling_k0]
            else:
                tiling_ori_aub = [
                    aub_tiling_m * aub_tiling_m0,
                    ub_ka * aub_tiling_k0
                ]
                tiling_ori_al1 = [l1_ma, l1_ka * al0_tiling_k0, al0_tiling_m0]
                aub_l1_affine_shape = [aub_tiling_m, ub_ka * aub_tiling_k0, aub_tiling_m0]
                tiling_ori_aub_with_l1 = [aub_tiling_m, ub_ka * aub_tiling_k0, aub_tiling_m0]
        elif self.get_a_matrix_mode == "Nz2Zz_int82fp32":
            tiling_ori_aub = [
                ub_ka // 2,
                aub_tiling_m,
                aub_tiling_m0,
                aub_tiling_k0 * 2
            ]
            tiling_ori_al1 = [l1_ma, l1_ka, al0_tiling_m0, al0_tiling_k0]
            aub_l1_affine_shape = [
                aub_tiling_m,
                ub_ka,
                aub_tiling_m0,
                aub_tiling_k0
            ]
            tiling_ori_aub_with_l1 = [
                aub_tiling_m,
                ub_ka,
                aub_tiling_m0,
                aub_tiling_k0
            ]
        elif self.get_a_matrix_mode in ("nd2Zz_int8", "nd2Zz", "nd_gemv", "nd_gevm"):
            if transpose_a:
                tiling_ori_aub = [ub_ka * aub_tiling_k0, aub_tiling_m * aub_tiling_m0]
            else:
                tiling_ori_aub = [aub_tiling_m * aub_tiling_m0, ub_ka * aub_tiling_k0]
            tiling_ori_aub_with_l1 = [aub_tiling_m, ub_ka, al0_tiling_m0, aub_tiling_k0]
            tiling_ori_al1 = [l1_ma, l1_ka, al0_tiling_m0, al0_tiling_k0]
            aub_l1_affine_shape = [aub_tiling_m, ub_ka, aub_tiling_m0, aub_tiling_k0]

        a_ub_ori_shape, tiling_ori_aub = self._get_dynamic_aub_shape(a_ub_ori_shape, tiling_ori_aub, self.have_batch_a)

        if self.format_out == "ND":
            aub_out_affine_shape = [aub_tiling_m * aub_tiling_m0, None]
        else:
            aub_out_affine_shape = [None, aub_tiling_m, aub_tiling_m0, None]

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
        if self.have_batch_a:
            a_ub_ori_shape[0] = self._int_ceil_div(a_ub_ori_shape[0], self.tiling.get("block_dim")[0])
            tiling_ori_aub.insert(0, self.aub_tiling_batch)
            tiling_ori_al1.insert(0, self.al1_tiling_batch)
            aub_l1_affine_shape.insert(0, self.aub_tiling_batch)
            aub_out_affine_shape.insert(0, self.aub_tiling_batch)
            aub_tiling_shape_with_lc0.insert(0, self.aub_tiling_batch)
            aub_l0c_affine_shape.insert(0, self.aub_tiling_batch)
            cl0_tiling_shape.insert(0, self.cl0_tiling_batch)
            tiling_ori_aub_with_l1.insert(0, self.aub_tiling_batch)
        elif self.have_batch:
            aub_out_affine_shape.insert(0, None)
            aub_l0c_affine_shape.insert(0, None)

        status_ori = Compare.compare(tiling_ori_aub, a_ub_ori_shape)
        status_l1 = Compare.compare(tiling_ori_aub_with_l1, tiling_ori_al1)
        status_l0c = Compare.compare(aub_tiling_shape_with_lc0, cl0_tiling_shape)
        self._print_debug([tiling_ori_aub, a_ub_ori_shape], "tiling_ori_aub with a_ub_ori_shape")
        self._print_debug([tiling_ori_aub_with_l1, tiling_ori_al1], "tiling_ori_aub_with_l1 with tiling_ori_al1")
        self._print_debug([aub_tiling_shape_with_lc0, cl0_tiling_shape], "aub_tiling_shape_with_lc0 cl0_tiling_shape")
        self._do_attach_aub(status_ori, status_l1, status_l0c, a_ub,
            aub_l1_affine_shape, aub_l0c_affine_shape, aub_out_affine_shape)
        self._print_debug("-------debug info in aub_process end-------")

    def _do_attach_aub(self, status_ori, status_l1, status_l0c, a_ub,
        aub_l1_affine_shape, aub_l0c_affine_shape, aub_out_affine_shape):
        sch_agent = self.sch_agent
        if status_ori == Compare.EQUAL:
            pass
        elif status_l1 == Compare.EQUAL:
            sch_agent.same_attach(a_ub, self.TENSOR_MAP.get("a_l1"))
        elif status_l1 == Compare.LESS_EQ:
            sch_agent.attach_at(a_ub, self.TENSOR_MAP.get("a_l1"), aub_l1_affine_shape)
        else:
            if status_l0c == Compare.EQUAL:
                sch_agent.same_attach(a_ub, self.TENSOR_MAP.get("c_l0c"))
            elif status_l0c == Compare.LESS_EQ:
                sch_agent.attach_at(a_ub, self.TENSOR_MAP.get("c_l0c"), affine_shape = aub_l0c_affine_shape)
            else:
                sch_agent.attach_at(a_ub, self.root_tensor, affine_shape = aub_out_affine_shape)
        
        same_attach_tensors = self.tensors_in_aub
        for tensor in same_attach_tensors:
            if tensor == a_ub:
                continue
            sch_agent.same_attach(tensor, a_ub)

    def _renew_bub_n(self, b_ub_ori_shape):
        index_offset = 1 if self.have_batch_b else 0
        block_n = self.tiling.get("block_dim")[1]
        if self.format_b in ("ND", "FRACTAL_Z"):
            if self.transpose_b:
                b_ub_ori_shape[index_offset] = self._int_ceil_div(b_ub_ori_shape[index_offset], block_n)
            else:
                b_ub_ori_shape[1 + index_offset] = self._int_ceil_div(b_ub_ori_shape[1 + index_offset], block_n)
        else:
            if self.transpose_b:
                b_ub_ori_shape[1 + index_offset] = self._int_ceil_div(b_ub_ori_shape[1 + index_offset], block_n)
            else:
                b_ub_ori_shape[index_offset] = self._int_ceil_div(b_ub_ori_shape[index_offset], block_n)

    def _get_dynamic_bub_shape(self, bub_shape, bub_tiling, have_batch):
        if self.is_dynamic:
            bub_shape = [self.dynamic_k * self.block_reduce, self.dynamic_n]
            bub_tiling = [self.bub_tiling_k, self.bub_tiling_n]
            if have_batch:
                bub_shape.insert(0, self.bub_tiling_batch)
                bub_tiling.insert(0, self.dynamic_batch)

        return bub_shape, bub_tiling

    def _bub_process(self):

        b_ub = self.TENSOR_MAP.get("b_ub")
        if b_ub in (None, []):
            return
        self._print_debug("-------debug info in bub_process-------")
        transpose_b = self.transpose_b
        cl0_tiling_nc, cl0_tiling_n0 = self.cl0_tiling_nc, self.cl0_tiling_n0
        cl0_tiling_m0 = self.cl0_tiling_m0
        bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_k0 = self.bl0_tiling_nb, self.bl0_tiling_n0, self.bl0_tiling_k0
        l1_nb, bl1_tiling_k = self.bl1_tiling_n, self.bl1_tiling_k
        l1_kb = (bl1_tiling_k + bl0_tiling_k0 - 1) // bl0_tiling_k0
        bub_tiling_n, bub_tiling_k = self.bub_tiling_n, self.bub_tiling_k
        bub_tiling_k0, bub_tiling_n0 = self.block_reduce, self.block_out
        ub_kb = (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0
        b_ub_ori_shape = list(self._get_value(i) for i in b_ub.shape)

        if self.get_b_matrix_mode == "nd2Zn_vnchwconv":
            if transpose_b:
                tiling_ori_bub = [bub_tiling_n * bub_tiling_n0, bub_tiling_k]
                tiling_ori_bl1 = [l1_nb, l1_kb * bl0_tiling_k0, bl0_tiling_n0]
                bub_l1_affine_shape = [bub_tiling_n, bub_tiling_k, bub_tiling_n0]
                tiling_ori_bub_with_l1 = [bub_tiling_n, bub_tiling_k, bub_tiling_n0]
            else:
                tiling_ori_bub = [bub_tiling_k, bub_tiling_n * bub_tiling_n0]
                tiling_ori_bl1 = [l1_kb, l1_nb * bl0_tiling_n0, bl0_tiling_k0]
                bub_l1_affine_shape = [ub_kb, bub_tiling_n * bub_tiling_n0, bub_tiling_k0]
                tiling_ori_bub_with_l1 = [ub_kb, bub_tiling_n * bub_tiling_n0, bub_tiling_k0]
        elif self.get_b_matrix_mode in ("nd2Zn_int8", "nd2Zn", "nd_gemv"):
            if transpose_b:
                tiling_ori_bub = [bub_tiling_n * bub_tiling_n0, bub_tiling_k]
            else:
                tiling_ori_bub = [bub_tiling_k, bub_tiling_n * bub_tiling_n0]
            tiling_ori_bub_with_l1 = [ub_kb, bub_tiling_n, bl0_tiling_n0, bl0_tiling_k0]
            tiling_ori_bl1 = [l1_kb, l1_nb, bl0_tiling_n0, bl0_tiling_k0]
            bub_l1_affine_shape = [
                ub_kb,
                bub_tiling_n,
                bub_tiling_n0,
                bub_tiling_k0
            ]
        elif self.get_b_matrix_mode == "Zn2Zn_int82fp32":
            tiling_ori_bub = [
                ub_kb // 2,
                bub_tiling_n,
                bub_tiling_n0,
                bub_tiling_k0 * 2
            ]
            tiling_ori_bub_with_l1 = [
                ub_kb,
                bub_tiling_n,
                bub_tiling_n0,
                bub_tiling_k0
            ]
            tiling_ori_bl1 = [l1_kb, l1_nb, bl0_tiling_n0, bl0_tiling_k0]
            bub_l1_affine_shape = [
                ub_kb,
                bub_tiling_n,
                bub_tiling_n0,
                bub_tiling_k0
            ]
        elif self.get_b_matrix_mode == "Nz2Zn":
            tiling_ori_bub = [
                bub_tiling_n,
                ub_kb,
                bub_tiling_k0,
                bub_tiling_n0
            ]
            tiling_ori_bub_with_l1 = [
                bub_tiling_n,
                ub_kb,
                bub_tiling_k0,
                bub_tiling_n0
            ]
            tiling_ori_bl1 = [l1_nb, l1_kb, bl0_tiling_k0, bl0_tiling_n0]
            bub_l1_affine_shape = [
                bub_tiling_n,
                ub_kb,
                bub_tiling_k0,
                bub_tiling_n0
            ]

        b_ub_ori_shape, tiling_ori_bub = self._get_dynamic_bub_shape(b_ub_ori_shape,
            tiling_ori_bub, self.have_batch_b)
        self._renew_bub_n(b_ub_ori_shape)
        if self.format_out == "ND":
            bub_out_affine_shape = [None, bub_tiling_n * bub_tiling_n0]
        else:
            bub_out_affine_shape = [bub_tiling_n, None, None, bub_tiling_n0]

        bub_tiling_shape_with_lc0 = [
            bub_tiling_n,
            bub_tiling_n0,
            (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
            bub_tiling_k0
        ]
        cl0_tiling_shape = [cl0_tiling_nc, cl0_tiling_n0, self.c_col_k1, self.c_col_k0]

        bub_l0c_affine_shape = [
            bub_tiling_n,
            None,
            None,
            bl0_tiling_n0,
            (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
            bub_tiling_k0
        ]
        if self.have_batch_b:
            b_ub_ori_shape[0] = self._int_ceil_div(b_ub_ori_shape[0], self.tiling.get("block_dim")[0])
            tiling_ori_bub.insert(0, self.bub_tiling_batch)
            tiling_ori_bl1.insert(0, self.bl1_tiling_batch)
            bub_l1_affine_shape.insert(0, self.bub_tiling_batch)
            bub_out_affine_shape.insert(0, self.bub_tiling_batch)
            bub_tiling_shape_with_lc0.insert(0, self.bub_tiling_batch)
            cl0_tiling_shape.insert(0, self.cl0_tiling_batch)
            bub_l0c_affine_shape.insert(0, self.bub_tiling_batch)
            tiling_ori_bub_with_l1.insert(0, self.bub_tiling_batch)
        elif self.have_batch:
            bub_out_affine_shape.insert(0, None)
            bub_l0c_affine_shape.insert(0, None)

        status_ori = Compare.compare(tiling_ori_bub, b_ub_ori_shape)
        status_l1 = Compare.compare(tiling_ori_bub_with_l1, tiling_ori_bl1)
        status_l0c = Compare.compare(bub_tiling_shape_with_lc0, cl0_tiling_shape)
        self._print_debug([tiling_ori_bub, b_ub_ori_shape], "tiling_ori_bub with b_ub_ori_shape")
        self._print_debug([tiling_ori_bub_with_l1, tiling_ori_bl1], "tiling_ori_bub_with_l1 with tiling_ori_bl1")
        self._print_debug([bub_tiling_shape_with_lc0, cl0_tiling_shape],
            "bub_tiling_shape_with_lc0 with cl0_tiling_shape")
        self._do_attach_bub(status_ori, status_l1, status_l0c, b_ub,
            bub_l1_affine_shape, bub_l0c_affine_shape, bub_out_affine_shape)
        self._print_debug("-------debug info in bub_process end-------")

    def _do_attach_bub(self, status_ori, status_l1, status_l0c, b_ub,
        bub_l1_affine_shape, bub_l0c_affine_shape, bub_out_affine_shape):
        sch_agent = self.sch_agent
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
            sch_agent.same_attach(b_ub, self.TENSOR_MAP.get("b_l1"))
        elif status_l1 == Compare.LESS_EQ:
            sch_agent.attach_at(b_ub, self.TENSOR_MAP.get("b_l1"), bub_l1_affine_shape)
        else:
            if status_l0c == Compare.EQUAL:
                sch_agent.same_attach(b_ub, self.TENSOR_MAP.get("c_l0c"))
            elif status_l0c == Compare.LESS_EQ:
                sch_agent.attach_at(
                    b_ub, self.TENSOR_MAP.get("c_l0c"), affine_shape = bub_l0c_affine_shape)
            else:
                bub_out_affine_shape = self._fix_affine_out_int8(b_ub.dtype, bub_out_affine_shape)
                sch_agent.attach_at(b_ub, self.root_tensor, affine_shape = bub_out_affine_shape)

        for tensor in self.tensors_in_bub:
            if tensor == b_ub:
                continue
            sch_agent.same_attach(tensor, b_ub)

    def _do_l1_ub_process(self):
        # get order
        k_dict = {
            "aub": self.aub_tiling_k // self.block_reduce,
            "bub": self.bub_tiling_k // self.block_reduce,
            "al1": self.al1_tiling_k // int(self.al0_tiling_k0),
            "bl1": self.bl1_tiling_k // int(self.bl0_tiling_k0)
        }
        tmp_order = sorted(k_dict.items(), key=lambda d: d[1], reverse=True)
        axis_order =[i[0] for i in tmp_order]

        def _adjust_order(axis_order, ub_tag, l1_tag):
            if axis_order.index(ub_tag) > axis_order.index(l1_tag) and k_dict.get(ub_tag) == k_dict.get(l1_tag):
                index_ub = axis_order.index(ub_tag)
                index_l1 = axis_order.index(l1_tag)
                axis_order[index_ub] = l1_tag
                axis_order[index_l1]= ub_tag

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

    def _bind_multi_core(self):
        root_tensor = self.res
        axis_mn = self.sch_agent[root_tensor].get_active_scopes()
        root_tensor_len = len(root_tensor.shape)
        if self.reduce_fusion:
            root_tensor_len += 1
        if root_tensor_len in (2, 4):
            if self.format_out == "ND":
                ax_m, ax_n =axis_mn[:2]
            else:
                ax_n, ax_m =axis_mn[:2]
        else:
            if self.format_out == "ND":
                ax_batch, ax_m, ax_n = axis_mn[:3]
            else:
                ax_batch, ax_n, ax_m = axis_mn[:3]
        batch_dim, n_dim, m_dim, _ = self.tiling.get("block_dim")
        if self.reduce_fusion:
            batch_dim = self._get_value(self.TENSOR_MAP.get("c_l0c").shape)[0]
        if root_tensor_len in (2, 4):
            ax_core = self.sch_agent[root_tensor].bind_core([ax_m, ax_n], [m_dim, n_dim])
        else:
            ax_core = self.sch_agent[root_tensor].bind_core([ax_batch, ax_m, ax_n], [batch_dim, m_dim, n_dim])
        self.sch_agent.root_stage_at(root_tensor, ax_core)

    def _buffer_align_func(self, tensor, have_batch=False, *align_args):

        if tensor is not None:
            if have_batch:
                self.sch[tensor].buffer_align((1, 1), *align_args)
            else:
                self.sch[tensor].buffer_align(*align_args)

    def _do_buffer_align(self):
        sch = self.sch
        TENSOR_MAP = self.TENSOR_MAP
        have_batch_b = self.have_batch_b
        have_batch_a = self.have_batch_a
        have_batch = self.have_batch
        self._do_buffer_align_l0c()
        self._set_requant_transfer_buffer_align()
        is_int82fp32_nd = self._is_int82fp32_nd()

        self._buffer_align_func(TENSOR_MAP.get("b_transpose"), have_batch_b, (1, 32), (1, 32))
        self._buffer_align_func(TENSOR_MAP.get("a_transpose"), have_batch_a, (1, 32), (1, 32))
        if is_int82fp32_nd:
            self._buffer_align_func(TENSOR_MAP.get("b_ub"), have_batch_b, (1, 32), (1, 32))
            self._buffer_align_func(TENSOR_MAP.get("a_ub"), have_batch_a, (1, 32), (1, 32))

        if self.format_out == "ND":
            self._buffer_align_func(TENSOR_MAP.get("c_add_bias_ub"), have_batch, (1, 16), (1, 16))
            self._buffer_align_func(TENSOR_MAP.get("beta_bias"), have_batch, (1, 16), (1, 16))

        cast_to_fp16 = TENSOR_MAP.get("cast_to_fp16")
        if cast_to_fp16 is not None:
            if len(cast_to_fp16.shape) in (2, 3):
                self._buffer_align_func(TENSOR_MAP.get("cast_to_fp16"), have_batch, (1, 16), (1, 16))
            else:
                self._buffer_align_func(TENSOR_MAP.get("cast_to_fp16"), have_batch, (1, 1), (1, 1), (1, 16), (1, 16))
            
        self._buffer_align_func(TENSOR_MAP.get("c_ub_fract"), have_batch, (1, 1), (1, 1), (1, 16), (1, 16))
        self._buffer_align_func(TENSOR_MAP.get("bias_l0c"), have_batch, (1, 1), (1, 1), (1, 16), (1, 16))
        self._buffer_align_func(TENSOR_MAP.get("c_add_bias"), have_batch, (1, 1), (1, 1), (1, 16), (1, 16))

        if self.mmad_mode == "gevm":
            self._buffer_align_func(self.TENSOR_MAP.get("a_l1"), have_batch_a, (1, 1), (1, 1), (1, 16), (1, 16))

    def _double_buffer(self, a_run_once, b_run_once):
        tiling = self.tiling
        sch = self.sch
        if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") == 2 and not a_run_once:
            sch[self.TENSOR_MAP.get("a_l1")].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") == 2 and not b_run_once:
            sch[self.TENSOR_MAP.get("b_l1")].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") == 2:
            sch[self.TENSOR_MAP.get("a_l0a")].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") == 2:
            sch[self.TENSOR_MAP.get("b_l0b")].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == 2:
            for tensor in self.tensors_in_l0c:
                sch[tensor].double_buffer()
        self._double_buffer_ub()

    def _double_buffer_ub(self):
        tiling = self.tiling
        sch = self.sch
        if tiling.get("manual_pingpong_buffer").get("AUB_pbuffer") == 2:
            if self.TENSOR_MAP.get("a_ub") is not None:
                sch[self.TENSOR_MAP.get("a_ub")].preload()
            for tensor in self.tensors_in_aub:
                self.sch[tensor].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("BUB_pbuffer") == 2:
            if self.TENSOR_MAP.get("b_ub") is not None:
                sch[self.TENSOR_MAP.get("b_ub")].preload()
            for tensor in self.tensors_in_bub:
                self.sch[tensor].double_buffer()
        if tiling.get("manual_pingpong_buffer").get("CUB_pbuffer") == 2:
            bias_ub = self.TENSOR_MAP.get("bias_ub")
            if bias_ub is not None:
                sch[bias_ub].preload()
                sch[bias_ub].double_buffer()

                if self.need_init_bias:
                    sch[self.TENSOR_MAP['init_value_of_bias_ub']].preload()
                    sch[self.TENSOR_MAP['init_value_of_bias_ub']].double_buffer()
                    sch[self.TENSOR_MAP['virtual_add_bias']].preload()
                    sch[self.TENSOR_MAP['virtual_add_bias']].double_buffer()
            for tensor in self.tensors_in_cub:
                if tensor in (self.res, self.TENSOR_MAP.get("c_gm")):
                    continue
                sch[tensor].double_buffer()

    def _emit_insn_func(self, insn_tensor, insn_axis_num, insn_tag, insn_dict=None, mode=0):

        normal_mode = 0
        if insn_tensor is not None:
            tensor_len = len(insn_tensor.shape)
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

    def _do_emit_insn(self):
        sch_agent = self.sch_agent
        self._do_emit_insn_aub()
        self._do_emit_insn_bub()
        self._emit_insn_nz_to_nd()

        # only in |gemm|ND Nz| all data type|
        self._emit_insn_func(self.TENSOR_MAP.get("alpha_ub"), 0, "dma_copy")
        self._emit_insn_func(self.TENSOR_MAP.get("beta_ub"), 0, "dma_copy")
        self._emit_insn_func(self.TENSOR_MAP.get("alpha_c"), 0, "vector_muls", mode=1)
        self._emit_insn_func(self.TENSOR_MAP.get("beta_bias"), 0, "vector_muls")
        self._emit_insn_func(self.TENSOR_MAP.get("bias_ub"), 0, "dma_copy")
        if self.need_init_bias:
            self._emit_insn_func(self.TENSOR_MAP.get("init_value_of_bias_ub"), 0, "dma_copy")
            self._emit_insn_func(self.TENSOR_MAP.get("virtual_add_bias"), 0, "phony_insn")

        # only in |matmul|ND Nz|all data type|
        self._emit_insn_func(self.TENSOR_MAP.get("bias_l0c"), 0, "dma_copy")
        self._emit_insn_func(self.TENSOR_MAP.get("c_add_bias"), 0, "phony_insn")
        self._pragma_func(self.TENSOR_MAP.get("bias_l0c"), 0, "reuse_output")
        self._pragma_func(self.TENSOR_MAP.get("c_add_bias"), 0, "replace_output")

        #only in |gemm|ND Nz|fp162fp16|
        self._emit_insn_func(self.TENSOR_MAP.get("alpha_fp162fp32"), 0, "vector_conv")
        self._emit_insn_func(self.TENSOR_MAP.get("beta_fp162fp32"), 0, "vector_conv")
        self._emit_insn_func(self.TENSOR_MAP.get("bias_cast_to_fp32"), 0, "vector_conv", mode=1)

        cast_to_fp16 = self.TENSOR_MAP.get("cast_to_fp16")
        c_ub_fract = self.TENSOR_MAP.get("c_ub_fract")
        cast_to_fp16_cmd = "dma_copy" if self.compute_inline_c_ub_fract else "vector_conv"
        # only in |gemm matmul|ND Nz|to fp16|
        self._emit_insn_func(self.TENSOR_MAP.get("cast_to_fp16"), 0, cast_to_fp16_cmd, mode=1)

        # only in gemm int82fp16
        self._emit_insn_func(self.TENSOR_MAP.get("a_int82fp16"), 0, "vector_conv", mode=1)
        self._emit_insn_func(self.TENSOR_MAP.get("b_int82fp16"), 0, "vector_conv", mode=1)

        # common
        self._emit_insn_func(self.TENSOR_MAP.get("a_l0a"), 0, "dma_copy", mode=1)
        self._emit_insn_func(self.TENSOR_MAP.get("a_l1"), 0, "dma_copy", mode=1)
        self._tensor_b_emit_insn()

        # fusion
        c_ub_fract = self.TENSOR_MAP.get("c_ub_fract")
        if c_ub_fract.op.attrs["scale_drq"].value != "ENABLE":
            self._emit_insn_func(self.TENSOR_MAP.get("c_ub_fract"), 0, "dma_copy", mode=1)
        else:
            self._set_quant_by_params()

        self._choose_dma_copy_for_res(self.res)

        # emit insn for l0c
        c_l0c = self.TENSOR_MAP.get("c_l0c")
        scopes_intrins = sch_agent[c_l0c].intrin_scopes(6)
        scope_insn = scopes_intrins[0]
        inner_k_axis = sch_agent[c_l0c].get_relate_scope(c_l0c.op.reduce_axis[0], scope_insn)

        if inner_k_axis:
            mad_dict = {
                "mad_pattern": self.mad_pattern,
                "k_outer": sch_agent[c_l0c].get_relate_scope(c_l0c.op.reduce_axis[0], scope_insn)
            }
        else:
            (
                inner_nb,
                inner_mb,
                inner_mp,
                inner_np,
                inner_kb,
                inner_kp
            ) = scopes_intrins
            inner_ko, inner_ki = sch_agent[c_l0c].split(inner_kb, nparts=1)
            sch_agent[c_l0c].reorder(
                inner_ko, inner_nb, inner_mb, inner_mp, inner_np, inner_ki, inner_kp
            )
            mad_dict = {"mad_pattern": self.mad_pattern, "k_outer": [inner_ko]}

        if self.TENSOR_MAP.get("c_add_bias") is not None:
            sch_agent[c_l0c].pragma(scope_insn, "replace_output", 0)
            mad_dict["init_bias"] = 1

        sch_agent[c_l0c].emit_insn(scope_insn, "mad", mad_dict)

        # fusion
        self._quantify_fusion_entry()
        self._tensor_a_l1_workspace_emit()
        for ten_in in self.elemwise_tensors:
            if ten_in.op.tag.find("|") != -1:
                str_list = ten_in.op.tag.split("|")
                insn = self.emit_insn_map.get(str_list[0])
            else:
                insn = self.emit_insn_map.get(ten_in.op.tag)
            if ten_in in self.ele_header_ub_tensors:
                insn = "dma_copy"
            if insn is None:
                insn = "vector_auto"
            self._emit_insn_func(ten_in, 0, insn)

        if self.gm_ub is not None:
            if not self.multi_output_flag:
                self._emit_insn_func(self.TENSOR_MAP.get("c_gm"), 0, "dma_copy", mode=1)
            else:
                self._emit_insn_for_multi_output()
            self._emit_insn_func(self.gm_ub, 0, "phony_insn", mode=1)

        if self.reduce_fusion:
            self._emit_insn_func(self.TENSOR_MAP.get("res_atomic_add_ub"), 0, "dma_copy", mode=1)

    def _emit_insn_for_multi_output(self):
        sch_agent = self.sch_agent
        if len(self.TENSOR_MAP.get("c_gm").shape) in (4, 5):
            gm_n_outer, gm_n_inner = sch_agent[self.TENSOR_MAP.get("c_gm")].split(
                self.TENSOR_MAP.get("c_gm").op.axis[-4], nparts=1)
            gm_m_outer, gm_m_inner = sch_agent[self.TENSOR_MAP.get("c_gm")].split(
                self.TENSOR_MAP.get("c_gm").op.axis[-3], nparts=1)
            sch_agent[self.TENSOR_MAP.get("c_gm")].reorder(gm_n_outer, gm_m_outer, gm_n_inner, gm_m_inner)
            sch_agent[self.TENSOR_MAP.get("c_gm")].emit_insn(gm_n_inner, "dma_copy")
        else:
            gm_n_outer, gm_n_inner = sch_agent[self.TENSOR_MAP.get("c_gm")].split(
                self.TENSOR_MAP.get("c_gm").op.axis[-1], nparts=1)
            gm_m_outer, gm_m_inner = sch_agent[self.TENSOR_MAP.get("c_gm")].split(
                self.TENSOR_MAP.get("c_gm").op.axis[-2], nparts=1)
            sch_agent[self.TENSOR_MAP.get("c_gm")].reorder(gm_m_outer, gm_n_outer, gm_m_inner, gm_n_inner)
            sch_agent[self.TENSOR_MAP.get("c_gm")].emit_insn(gm_m_inner, "dma_copy")

    def _do_emit_insn_aub(self):
        sch_agent = self.sch_agent
        offset_a = 1 if self.have_batch_a else 0

        # only in |gemm matmul|nd|all| or |matmul|nz|int82fp32| etc
        self._emit_insn_func(self.TENSOR_MAP.get("a_ub"), 0, "dma_copy", mode=1)

        a_cast_and_reshape = (self.ops_data_flow_mode == "int82fp32") and (self.format_a == "FRACTAL_NZ")
        a_only_reshape = (self.mmad_mode in ("gevm", "gemv")) or (self.get_a_matrix_mode == "nd2Zz_int8")
        if a_cast_and_reshape:
            # only in |gemm matmul|nz|int82fp32|
            a_ub_fract = self.TENSOR_MAP.get("a_ub_fract")
            if self.have_batch_a:
                _, a_ub_scope_outer, a_ub_scope_inner, _, _ = sch_agent[a_ub_fract].get_active_scopes()
            else:
                a_ub_scope_outer, a_ub_scope_inner, _, _ = sch_agent[a_ub_fract].get_active_scopes()
            sch_agent[a_ub_fract].split(a_ub_scope_inner, 2)
            sch_agent[a_ub_fract].emit_insn(a_ub_scope_outer, "vector_auto")
        elif a_only_reshape:
            self._emit_insn_func(self.TENSOR_MAP.get("a_ub_fract"), 0, "dma_copy")
        else:
            # only in |gemm matmul|ND|fp162fp16 fp162fp32 int82fp32|
            self._emit_insn_func(self.TENSOR_MAP.get("a_ub_fract"), 1 + offset_a, "vnchwconv", mode=1)

        # only in |gemm|ND|int82int32|
        a_transpose = self.TENSOR_MAP.get("a_transpose")
        if a_transpose is not None:
            m_outer, m_inner = sch_agent[a_transpose].split(a_transpose.op.axis[1 + offset_a], factor=32)
            sch_agent[a_transpose].reorder(m_outer, a_transpose.op.axis[offset_a], m_inner)
            sch_agent[a_transpose].emit_insn(sch_agent[a_transpose].op.axis[offset_a], "vnchwconv")

    def _do_emit_insn_bub(self):
        sch_agent = self.sch_agent
        offset_b = 1 if self.have_batch_b else 0
        # only in |gemm matmul|nd|all| or |matmul|nz|int82fp32| etc
        self._emit_insn_func(self.TENSOR_MAP.get("b_ub"), 0, "dma_copy", mode=1)

        b_cast_and_reshape = (self.ops_data_flow_mode == "int82fp32") and (self.format_b == "FRACTAL_Z")
        b_only_reshape = (self.mmad_mode == "gemv") or (self.get_b_matrix_mode == "nd2Zn_int8")
        if b_cast_and_reshape:
            # only in |gemm matmul|nz|int82fp32|
            b_ub_fract = self.TENSOR_MAP.get("b_ub_fract")
            if self.have_batch_b:
                _, b_ub_scope_outer, b_ub_scope_inner, _, _ = sch_agent[b_ub_fract].get_active_scopes()
            else:
                b_ub_scope_outer, b_ub_scope_inner, _, _ = sch_agent[b_ub_fract].get_active_scopes()
            b_ub_outer_outer, _ = sch_agent[b_ub_fract].split(b_ub_scope_outer, 2)
            sch_agent[b_ub_fract].emit_insn(b_ub_outer_outer, "vector_auto")
        elif b_only_reshape:
            self._emit_insn_func(self.TENSOR_MAP.get("b_ub_fract"), 0, "dma_copy")
        else:
            # only in |gemm matmul|ND|fp162fp16 fp162fp32 int82fp32|
            self._emit_insn_func(self.TENSOR_MAP.get("b_ub_fract"), 1 + offset_b, "vnchwconv", mode=1)

        # only in |gemm|ND|int82int32|
        b_transpose = self.TENSOR_MAP.get("b_transpose")
        if b_transpose is not None:
            k_outer, k_inner = sch_agent[b_transpose].split(b_transpose.op.axis[1 + offset_b], factor=32)
            sch_agent[b_transpose].reorder(k_outer, b_transpose.op.axis[offset_b], k_inner)
            sch_agent[b_transpose].emit_insn(sch_agent[b_transpose].op.axis[offset_b], "vnchwconv")

    def _emit_insn_nz_to_nd(self):
        sch_agent = self.sch_agent
        c_add_bias_ub = self.TENSOR_MAP.get("c_add_bias_ub")
        nz_to_nd = self.TENSOR_MAP.get("nz_to_nd")
        fract_add_nd_to_nd = (self.format_out == "ND") and (c_add_bias_ub is not None)

        if fract_add_nd_to_nd:
            self._cut_axis_for_nz_to_nd(c_add_bias_ub, "vector_add")
        elif c_add_bias_ub is not None:
            # only in |gemm|Nz|all data type|
            self._emit_insn_func(c_add_bias_ub, 0, "vector_add", mode=1)

        if nz_to_nd is not None:
            self._cut_axis_for_nz_to_nd(nz_to_nd, "vector_auto")

    def _cut_axis_for_nz_to_nd(self, ori_tensor, emit_insn_cmd):
        sch_agent = self.sch_agent
        # only in |gemm|ND|all data type|
        if self.have_batch:
            scope_batch, scope_outer, scope_inner = sch_agent[ori_tensor].get_active_scopes()
        else:
            scope_outer, scope_inner = sch_agent[ori_tensor].get_active_scopes()
        outer_outer, outer_inner = sch_agent[ori_tensor].split(scope_outer, self.block_in)
        inner_outer, inner_inner = sch_agent[ori_tensor].split(scope_inner, self.block_out)
        sch_agent[ori_tensor].reorder(outer_outer, inner_outer, outer_inner, inner_inner)
        if self.have_batch:
            sch_agent[ori_tensor].emit_insn(scope_batch, emit_insn_cmd)
        else:
            sch_agent[ori_tensor].emit_insn(outer_inner, emit_insn_cmd)

    def _tensor_b_emit_insn(self):
        """
        tensor_b_l1 emit insn operation for compress or not
        """
        b_l0b = self.TENSOR_MAP.get("b_l0b")
        b_l1 = self.TENSOR_MAP.get("b_l1")
        if self.compress_flag:
            host_axis = self._get_index_at_axis()
            compress_index = self.TENSOR_MAP.get("compress_index")
            if not self.b_l1_inline_flag:
                b_l1.op.attrs["tile_L1_k"] = b_l0b.op.attrs["tile_L1_k"]
                b_l1.op.attrs["tile_L1_n"] = b_l0b.op.attrs["tile_L1_n"]
                # k_l1_tile n_l1_tile host_axis
                k_l1_tile = self.bl1_tiling_k // self.block_reduce
                n_l1_tile = self.bl1_tiling_n
                self._set_compress_info(self.sch, b_l1, compress_index, k_l1_tile, n_l1_tile, host_axis)
                self._emit_insn_func(b_l0b, 0, "dma_copy", mode=1)
            else:
                k_l0_tile = self.bl0_tiling_kb
                n_l0_tile = self.bl0_tiling_nb
                self._set_compress_info(self.sch, b_l0b, compress_index, k_l0_tile, n_l0_tile, host_axis)
        else:
            if not self.b_l1_inline_flag:
                self._emit_insn_func(b_l1, 0, "dma_copy", mode=1)
            self._emit_insn_func(b_l0b, 0, "dma_copy", mode=1)

    def _add_key_value(self, key, value):
        buffer_reuse_dict = self.buffer_reuse_dict
        if (key is not None) and (value is not None):
            buffer_reuse_dict[key] = value

    def _set_buffer_reuse_dict(self):
        TENSOR_MAP = self.TENSOR_MAP
        self._add_key_value(TENSOR_MAP.get("c_ub_fract"), TENSOR_MAP.get("alpha_c"))
        if self.format_out == "FRACTAL_NZ":
            self._add_key_value(TENSOR_MAP.get("c_add_bias_ub"), TENSOR_MAP.get("alpha_c"))
        else:
            self._add_key_value(TENSOR_MAP.get("c_add_bias_ub"), TENSOR_MAP.get("beta_bias"))

        if self.ops_data_flow_mode == "fp162fp16":
            self._add_key_value(TENSOR_MAP.get("beta_fp162fp32"), TENSOR_MAP.get("beta_ub"))
            self._add_key_value(TENSOR_MAP.get("alpha_fp162fp32"), TENSOR_MAP.get("alpha_ub"))
            self._add_key_value(TENSOR_MAP.get("beta_bias"), TENSOR_MAP.get("bias_cast_to_fp32"))
        else:
            self._add_key_value(TENSOR_MAP.get("beta_bias"), TENSOR_MAP.get("bias_ub"))

        for axpy, parent in self.axpy_2_parent.items():
            self.buffer_reuse_dict[parent] = axpy

        if self.need_init_bias:
            self._add_key_value(TENSOR_MAP.get("virtual_add_bias"), [TENSOR_MAP.get("bias_ub"), TENSOR_MAP.get("init_value_of_bias_ub")])

    def _do_buffer_reuse(self):
        for bereused_tensor, tensor in self.buffer_reuse_dict.items():
            if (bereused_tensor is not None) and (tensor is not None):
                if isinstance(tensor, Iterable):
                    self.sch[bereused_tensor].reused_by(*tensor)
                else:
                    self.sch[bereused_tensor].reused_by(tensor)

    def _compute_run_once_flag(self):
        # 0: not run_once
        # 1: run_once on c_l0c
        # 2: run_once on c_gm
        run_once_flag, both_not_run_once, c_l0c_aparts, core_inner, data_size = self._init_run_once_flag()
        a_run_once, b_run_once = run_once_flag
        c_l0c_aparts_a, c_l0c_aparts_b = c_l0c_aparts
        core_inner_n, core_inner_m = core_inner
        al1_data_size, bl1_data_size = data_size
        if self.al1_attach_status == "c_l0c" and self.bl1_attach_status == "c_l0c":
            # if true b reload more, if false a reload more
            reload_flag = (al1_data_size * c_l0c_aparts_b) < (bl1_data_size * c_l0c_aparts_a)
            if both_not_run_once:
                a_run_once = 0 if reload_flag else a_run_once
                b_run_once = b_run_once if reload_flag else 0
        elif self.al1_attach_status == "c_gm" and self.bl1_attach_status == "c_gm":
            # if true b reload more, if false a reload more
            reload_flag = (al1_data_size * core_inner_n) < (bl1_data_size * core_inner_m)
            if both_not_run_once:
                a_run_once = 0 if reload_flag else a_run_once
                b_run_once = b_run_once if reload_flag else 0
        else:
            a_run_once, b_run_once = self._diff_compute_at_only_open_one(a_run_once, b_run_once)

        return a_run_once, b_run_once

    def _diff_compute_at_only_open_one(self, a_run_once, b_run_once):
        if self.al1_attach_status == "c_l0c" and self.bl1_attach_status == "c_gm":
            b_run_once = 0
        elif self.al1_attach_status == "c_gm" and self.bl1_attach_status == "c_l0c":
            a_run_once = 0
        return a_run_once, b_run_once

    def _init_run_once_flag(self):

        block_in = self.block_in
        block_out = self.block_out
        block_reduce = self.block_reduce

        dtype_byte = self.DTYPE_WIDTH_MAP.get(self.TENSOR_MAP.get("a_l1").dtype) * 2
        size = tbe_platform_info.get_soc_spec("L1_SIZE") // dtype_byte
        a_l1_db = self.tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
        b_l1_db = self.tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
        c_l0c = self.TENSOR_MAP.get("c_l0c")
        orgin_k = self.TENSOR_MAP.get("a_l0a").shape[1].value * block_reduce
        orgin_m = self.TENSOR_MAP.get("a_l0a").shape[0].value * block_in
        orgin_n = self.TENSOR_MAP.get("b_l0b").shape[1].value * block_out
        cl0_tiling_nc, cl0_tiling_mc = self.cl0_tiling_nc, self.cl0_tiling_mc
        al1_tiling_m, al1_tiling_k = self.al1_tiling_m, self.al1_tiling_k
        bl1_tiling_n, bl1_tiling_k = self.bl1_tiling_n, self.bl1_tiling_k
        tiling_k0 = self.al0_tiling_k0
        
        c_l0c_aparts_b = self._int_ceil_div(cl0_tiling_nc, bl1_tiling_n)
        c_l0c_aparts_a = self._int_ceil_div(cl0_tiling_mc, al1_tiling_m)
        al1_data_size = al1_tiling_m * block_in * al1_tiling_k * a_l1_db
        bl1_data_size = bl1_tiling_n * block_out * bl1_tiling_k * b_l1_db
        al1_run_once_size_in_cl0 = orgin_k * cl0_tiling_mc * block_in
        bl1_run_once_size_in_cl0 = orgin_k * cl0_tiling_nc * block_out
        block_dim = self.tiling.get("block_dim")
        core_inner_m = self._int_ceil_div(orgin_m, block_dim[2])
        core_inner_n = self._int_ceil_div(orgin_n, block_dim[1])
        al1_run_once_size_in_cgm = orgin_k * core_inner_m
        bl1_run_once_size_in_cgm = orgin_k * core_inner_n

        a_run_once = 2 if ((al1_run_once_size_in_cgm + bl1_data_size) < size) else 0
        a_run_once = 1 if ((a_run_once == 0) and ((al1_run_once_size_in_cl0 + bl1_data_size) < size)) else 0
        b_run_once = 2 if ((bl1_run_once_size_in_cgm + al1_data_size) < size) else 0
        b_run_once = 1 if ((b_run_once == 0) and ((bl1_run_once_size_in_cl0 + al1_data_size) < size)) else 0

        both_not_run_once = (a_run_once != 0) and (b_run_once != 0)
        if self.al1_attach_status == "c_gm" and self.bl1_attach_status == "c_gm":
            c_l0c_aparts_b, c_l0c_aparts_a = core_inner_n, core_inner_m
        run_once_flag = [a_run_once, b_run_once]
        c_l0c_aparts = [c_l0c_aparts_a, c_l0c_aparts_b]
        core_inner = [core_inner_n, core_inner_m]
        data_size = [al1_data_size, bl1_data_size]
        return [run_once_flag, both_not_run_once, c_l0c_aparts, core_inner, data_size]

    def _allocate_axis(self):
        a_run_once = False
        b_run_once = False
        if self.format_a == "ND" or self.format_b == "ND" or self.is_dynamic:
            return a_run_once, b_run_once
        sch = self.sch
        c_l0c = self.TENSOR_MAP.get("c_l0c")
        outer_axis_l0c = self.sch_agent[c_l0c].get_active_scopes()
        n_outer_l0c = outer_axis_l0c[0]
        m_outer_l0c = outer_axis_l0c[1]

        outer_axis_c_gm = self.sch_agent[self.res].get_active_scopes()
        m_outer_cgm = outer_axis_c_gm[0]
        n_outer_cgm = outer_axis_c_gm[1]
        if self.format_out != "ND":
            m_outer_cgm, n_outer_cgm = n_outer_cgm, m_outer_cgm

        a_run_once_base = (self.in_addr_type == 0
                            and (not self.l1_fusion_and_l1_size_0)
                            and self.input_l1_flag != 1
                            and self.al1_attach_status != "full_load")

        tensor_a_reuse_local, tensor_b_reuse_local = self._compute_run_once_flag()
        a_run_once_cgm = ((tensor_a_reuse_local == 2)
            and a_run_once_base
            and (not (self.al1_attach_status == "c_l0c" and self.c_l0c_attach_status == "full_load")))
        a_run_once_cl0c = (tensor_a_reuse_local == 1
                           and a_run_once_base
                           and self.al1_attach_status == "c_l0c")
        tensor_a_l1 = self.TENSOR_MAP.get("a_l1") if self.mmad_mode != "gemv" else self.TENSOR_MAP.get("b_l1")

        if a_run_once_cgm:
            sch[tensor_a_l1].allocate_at(sch[self.res], n_outer_cgm, run_once_axes=[n_outer_cgm])
            sch[tensor_a_l1].mem_unique()
        elif a_run_once_cl0c:
            sch[tensor_a_l1].allocate_at(sch[c_l0c], n_outer_l0c, run_once_axes=[n_outer_l0c])
            sch[tensor_a_l1].mem_unique()

        b_run_once_base = (not self.l1_fusion_and_l1_size_0) and (self.bl1_attach_status != "full_load")
        b_run_once_cgm = ((tensor_b_reuse_local == 2)
            and b_run_once_base
            and (not (self.bl1_attach_status == "c_l0c" and self.c_l0c_attach_status == "full_load")))
        b_run_once_cl0c = (tensor_b_reuse_local == 1
                           and b_run_once_base
                           and self.bl1_attach_status == "c_l0c")

        tensor_b_l1 = self.TENSOR_MAP.get("b_l1") if self.mmad_mode != "gemv" else self.TENSOR_MAP.get("a_l1")
        if b_run_once_cgm:
            sch[tensor_b_l1].allocate_at(sch[self.res], m_outer_cgm, run_once_axes=[m_outer_cgm])
            sch[tensor_b_l1].mem_unique()
        elif b_run_once_cl0c:
            sch[tensor_b_l1].allocate_at(sch[c_l0c], m_outer_l0c, run_once_axes=[m_outer_l0c])
            sch[tensor_b_l1].mem_unique()
        a_run_once = a_run_once_cgm or a_run_once_cl0c
        b_run_once = b_run_once_cgm or b_run_once_cl0c
        return a_run_once, b_run_once

    def _slove_bank_conflict(self):
        """slove bank conflict by storage_align
        if aub_k or bub_n bigger than threshold_data_num,
        use storage_align to slove bank conflict of aub/bub

        c_ub always conflict, must be use storage_align
        Input: None
        ---------------------------------
        Return: None
        """
        if self.format_out != "ND":
            return
        (a_ub_storage_align,
         b_ub_storage_align,
         c_ub_storage_align) = self._check_exceed_ub(self.transpose_a, self.transpose_b)

        # the data gap in ub
        gap_value = self.block_reduce
        c_gap_value = (self.block_out + 1) * self.block_in
        tiling = self.tiling
        TENSOR_MAP = self.TENSOR_MAP
        # slove bank conflict in aub/bub
        sch = self.sch
        if a_ub_storage_align:
            aub_k, aub_m, _, _ = tiling.get("AUB_shape")
            aub_m *= self.block_in
            # the data stride in ub
            a_align_value = (aub_m + gap_value) if self.transpose_a else (aub_k + gap_value)
            a_int82fp16 = TENSOR_MAP.get("a_int82fp16")
            a_normalize_ub = TENSOR_MAP.get("a_ub")
            src_dtype = a_normalize_ub.dtype
            a_transpose = TENSOR_MAP.get("a_transpose")
            if a_int82fp16 is not None:
                sch[a_int82fp16].storage_align(a_int82fp16.op.axis[0], a_align_value, 0)
            elif (src_dtype == "float16") or (a_transpose is not None):
                sch[a_normalize_ub].storage_align(a_normalize_ub.op.axis[0], a_align_value, 0)

        if b_ub_storage_align:
            bub_k, bub_n, _, _ = tiling.get("BUB_shape")
            bub_n *= self.block_out
            # the data stride in ub
            b_align_value = (bub_k + gap_value) if self.transpose_b else (bub_n + gap_value)
            b_int82fp16 = TENSOR_MAP.get("b_int82fp16")
            b_normalize_ub = TENSOR_MAP.get("b_ub")
            src_dtype = b_normalize_ub.dtype
            b_transpose = TENSOR_MAP.get("b_transpose")
            if b_int82fp16 is not None:
                sch[b_int82fp16].storage_align(b_int82fp16.op.axis[0], b_align_value, 0)
            elif (src_dtype == "float16") or (b_transpose is not None):
                sch[b_normalize_ub].storage_align(b_normalize_ub.op.axis[0], b_align_value, 0)

        # slove bank conflict in cub
        self._slove_bank_conflict_cub(c_ub_storage_align, c_gap_value)

    def _slove_bank_conflict_cub(self, c_ub_storage_align, c_gap_value):
        TENSOR_MAP = self.TENSOR_MAP
        sch = self.sch
        if c_ub_storage_align:
            c_ub_fract = TENSOR_MAP.get("c_ub_fract")
            alpha_c_ub = TENSOR_MAP.get("alpha_c")
            if c_ub_fract not in self.compute_inline_list:
                sch[c_ub_fract].storage_align(c_ub_fract.op.axis[1], c_gap_value, 0)
                if alpha_c_ub is not None:
                    sch[alpha_c_ub].storage_align(alpha_c_ub.op.axis[1], c_gap_value, 0)

    def _check_exceed_ub(self, a_trans, b_trans):
        """
        if storage_align is used, more UB space is used.
        Therefore, check the UB space usage after storage_align is used.
        Input:
            a_trans: bool, Indicates whether matrix A is transposed.
            b_trans: bool, Indicates whether matrix B is transposed.
        -----------------------
        Return:
            a_ub_storage_align: bool, Matrix A uses storage_align.
            b_ub_storage_align: bool, Matrix B uses storage_align.
            c_ub_storage_align: bool, Matrix C uses storage_align.
        """
        tiling = self.tiling
        need_aub_storage_align = (self.TENSOR_MAP.get("a_ub") is not None)
        need_bub_storage_align = (self.TENSOR_MAP.get("b_ub") is not None)
        need_cub_storage_align = (self.TENSOR_MAP.get("c_add_bias_ub") is not None) and (self.format_out == "ND")

        threshold_data_num = 64
        gap_value = self.block_reduce
        ub_buffer = tbe_platform_info.get_soc_spec("UB_SIZE")

        a_ub_storage_align = False
        b_ub_storage_align = False
        c_ub_storage_align = False

        # get fused num for compute use UB size
        a_fused_num, b_fused_num, c_fused_num = self.fuse_num_group
        a_fused_num = a_fused_num / 10.0 + 1
        b_fused_num = b_fused_num / 10.0 + 1
        c_fused_num += 1
        # compute before storage_align used UB size
        base_buffer_size = 0
        base_buffer_size, a_add_size = self._get_a_ub_storage_align_buffer_size(base_buffer_size,
            need_aub_storage_align, gap_value, a_fused_num, a_trans)
        base_buffer_size, b_add_size = self._get_b_ub_storage_align_buffer_size(base_buffer_size,
            need_bub_storage_align, gap_value, b_fused_num, b_trans)
        base_buffer_size, c_add_size = self._get_c_ub_storage_align_buffer_size(base_buffer_size,
            need_cub_storage_align, c_fused_num)

        c_ub_storage_align = need_cub_storage_align and (base_buffer_size + c_add_size <= ub_buffer)
        base_buffer_size += c_add_size if c_ub_storage_align else 0
        if need_aub_storage_align:
            aub_k, aub_m = tiling.get("AUB_shape")[0:2]
            aub_m *= self.block_in
            judge_value = aub_m if a_trans else aub_k
            a_ub_storage_align = (need_aub_storage_align
                and (judge_value % threshold_data_num == 0)
                and ((base_buffer_size + a_add_size)<= ub_buffer))
            base_buffer_size += a_add_size if a_ub_storage_align else 0

        if need_bub_storage_align:
            bub_k, bub_n = tiling.get("BUB_shape")[0:2]
            bub_n *= self.block_out
            judge_value = bub_k if b_trans else bub_n
            b_ub_storage_align = (need_bub_storage_align
                and (judge_value % threshold_data_num == 0)
                and ((base_buffer_size + b_add_size) <= ub_buffer))
            base_buffer_size += b_add_size if b_ub_storage_align else 0

        return a_ub_storage_align, b_ub_storage_align, c_ub_storage_align

    def _get_a_ub_storage_align_buffer_size(self, base_buffer_size,
        need_aub_storage_align, gap_value, a_fused_num, a_trans):
        tiling = self.tiling
        a_add_size = 0
        if need_aub_storage_align:
            aub_k, aub_m = tiling.get("AUB_shape")[0:2]
            aub_m *= self.block_in
            a_db = tiling.get("manual_pingpong_buffer").get("AUB_pbuffer")
            base_buffer_size += (aub_m * aub_k * a_fused_num *
                                    self.INPUT_SIZE.get(self.ops_data_flow_mode) * a_db)
            # if use storage_align, need UB size
            a_add_size = (gap_value * (aub_k if a_trans else aub_m) * 
                            self.INPUT_SIZE.get(self.ops_data_flow_mode) * a_db)
        return base_buffer_size, a_add_size

    def _get_b_ub_storage_align_buffer_size(self, base_buffer_size,
        need_bub_storage_align, gap_value, b_fused_num, b_trans):
        tiling = self.tiling
        b_add_size = 0
        if need_bub_storage_align:
            bub_k, bub_n = tiling.get("BUB_shape")[0:2]
            bub_n *= self.block_out
            b_db = tiling.get("manual_pingpong_buffer").get("BUB_pbuffer")
            base_buffer_size += (bub_k * bub_n * b_fused_num *
                                    self.INPUT_SIZE.get(self.ops_data_flow_mode) * b_db)
            # if use storage_align, need UB size
            b_add_size = (gap_value * (bub_n if b_trans else bub_k) *
                            self.INPUT_SIZE.get(self.ops_data_flow_mode) * b_db)
        return base_buffer_size, b_add_size

    def _get_c_ub_storage_align_buffer_size(self, base_buffer_size, need_cub_storage_align, c_fused_num):
        tiling = self.tiling
        c_add_size = 0
        float32_int32_size = 4
        if need_cub_storage_align:
            cub_n, cub_m = tiling.get("CUB_matrix")[0:2]
            c_db = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")
            base_buffer_size += (cub_n * cub_m * self.block_in * self.block_out *
                                    c_fused_num * self.OUTPUT_SIZE.get(self.ops_data_flow_mode) * c_db)
            # if use storage_align, need UB size
            c_add_size = self.block_out * cub_n * cub_m * float32_int32_size * c_db
        return base_buffer_size, c_add_size

    def _renew_block_dim(self):
        """
        if tail data small then 16(output=fp16) or 32(output=int32)
        close multi core
        """
        c_gm = self.TENSOR_MAP.get("c_gm")
        if self.ops_data_flow_mode == "int82int32":
            multi_core_min_slice = 32
        else:
            multi_core_min_slice = 16

        if (c_gm.shape[1].value * self.OUTPUT_SIZE.get(self.ops_data_flow_mode) < multi_core_min_slice):
            self.tiling["block_dim"] = [1, 1, 1, 1]

    def _do_compute_inline(self):
        self.elewise_compute_inline_list += self.compute_inline_list
        for tensor in self.elewise_compute_inline_list:
            self.sch[tensor].compute_inline()

    def _set_requant_transfer_buffer_align(self):
        requant_fusion = self.requant_fusion
        requant_data_transfer = self.TENSOR_MAP.get("requant_data_transfer")
        if not requant_fusion:
            return

        unchanged = 1
        sch = self.sch
        if self.have_batch:
            sch[requant_data_transfer].buffer_align((unchanged, unchanged),
                                                    (unchanged, unchanged),
                                                    (unchanged, unchanged),
                                                    (unchanged, 16),
                                                    (unchanged, 16))
        else:
            sch[requant_data_transfer].buffer_align((unchanged, unchanged),
                                                    (unchanged, unchanged),
                                                    (unchanged, 16),
                                                    (unchanged, 16))
        return

    def _do_buffer_align_l0c(self):
        c_l0c = self.TENSOR_MAP.get("c_l0c")
        sch = self.sch
        block_reduce = self.block_reduce
        block_out = self.block_out
        unchanged = 1
        if self.mmad_mode in ("gevm", "gemv"):
            if self.have_batch:
                sch[c_l0c].buffer_align((unchanged, unchanged),
                                        (unchanged, unchanged),
                                        (unchanged, unchanged),
                                        (unchanged, block_out),
                                        (unchanged, block_out),
                                        (unchanged, unchanged),
                                        (unchanged, block_reduce))
            else:
                sch[c_l0c].buffer_align((unchanged, unchanged),
                                        (unchanged, unchanged),
                                        (unchanged, block_out),
                                        (unchanged, block_out),
                                        (unchanged, unchanged),
                                        (unchanged, block_reduce))
        else:
            if self.have_batch:
                sch[c_l0c].buffer_align(
                    (unchanged, unchanged),
                    (unchanged, unchanged),
                    (unchanged, unchanged),
                    (unchanged, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][0]),
                    (unchanged, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][2]),
                    (unchanged, unchanged),
                    (unchanged, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][1])
                )
            else:
                sch[c_l0c].buffer_align(
                    (unchanged, unchanged),
                    (unchanged, unchanged),
                    (unchanged, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][0]),
                    (unchanged, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][2]),
                    (unchanged, unchanged),
                    (unchanged, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][1])
                )

    def _get_compress_block_info(self, tile_k, tile_n):
        """
        get weigths compress info, like, block size, index size
        """
        block_size_max = 32 * 1024
        block_unit = 512
        data_size = tile_k * tile_n * block_unit
        size_max = block_size_max
        if 0 < data_size < size_max:
            size_max = data_size
        block_size = block_unit
        for block_idx in range(size_max, 0, block_unit * (-1)):
            if data_size % block_idx == 0:
                block_size = block_idx
                break

        return int(block_size)

    def _set_compress_info(self, sch,  # pylint: disable=R0913, R0914
                           compress_tensor, compress_index,
                           tile_k, tile_n, out_axis):
        """
        set weigths compress info
        """
        if out_axis is None:
            raise RuntimeError("compress index axis is None, it's error.")
        sch = self.sch
        engine, ratios, channel, mode = tbe_platform_info.get_soc_spec("UNZIP")
        frac_size = 512

        index_shape = compress_index.shape
        dim_k = compress_tensor.shape[0].value
        dim_n = compress_tensor.shape[1].value

        tile_k_value = compress_tensor.op.attrs["tile_L1_k"]
        tile_n_value = compress_tensor.op.attrs["tile_L1_n"]

        block_size = self._get_compress_block_info(tile_k, tile_n)
        k_value = block_size // (tile_n * frac_size)

        sch.set_var_range(tile_k_value, k_value, k_value)
        sch.set_var_range(tile_n_value, tile_n, tile_n)

        k_block_num = (dim_k + k_value - 1) // k_value
        n_block_num = (dim_n + tile_n - 1) // tile_n
        index_size = k_block_num * n_block_num

        tight_len = 2
        if mode == 1:
            tight_len = 8
        index_size = index_size * tight_len

        sch.set_var_range(index_shape[0], int(index_size), int(index_size))

        conflict = tvm.make.Call("int32", "tvm_tuple",
                                (block_size, index_size, mode, engine,
                                channel, ratios, k_value, tile_n),
                                tvm.expr.Call.PureIntrinsic, None, 0)
        sch[compress_tensor].pragma(compress_tensor.op.axis[0],
                                    "json_info_compress_parameters", conflict)
        tensor_len = len(compress_tensor.shape)
        # transform data to continue by block_size
        sch[compress_tensor].emit_insn(compress_tensor.op.axis[tensor_len - 4],
                                       "unzip",
                                       {"compress_mode": mode,
                                        "block_size": block_size,
                                        "hoist_axis": out_axis})

    def _get_index_at_axis(self):
        axis = self.sch_agent[self.root_tensor].get_active_scopes()
        axis_n = axis[0]
        axis_m = axis[1]

        if self.format_out == "ND":
            axis_n, axis_m = axis_m, axis_n

        block_dim_m = self.tiling.get("block_dim")[2]
        m_shape = self.TENSOR_MAP.get("a_l0a").shape[-4].value
        m_factor = (m_shape + block_dim_m - 1) // block_dim_m
        index_at_axis = axis_m if self.al1_tiling_m * 2 < m_factor else -1
        return index_at_axis

    def _choose_dma_copy_for_res(self, res):
        """choose dma copy pattern"""
        # with_transpose is the flag to use emit_insn dma_copy_matmul_transpose
        # this flag set from confusion_transpose_d
        with_transpose = hasattr(res, "matmul_with_transpose")
        if with_transpose:
            # get matrix axis shapes
            tensor_a_l0a = self.TENSOR_MAP.get("a_l0a")
            tensor_b_l0b = self.TENSOR_MAP.get("b_l0b")
            if not self.is_dynamic:
                m_shape = self._get_value(tensor_a_l0a.shape[-4]) * self.block_in
                n_shape = self._get_value(tensor_b_l0b.shape[-3]) * self.block_out
            else:
                m_shape = self.dynamic_m * self.block_in
                n_shape = self.dynamic_n * self.block_out
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_m", m_shape)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_n", n_shape)
            block_dim = self.tiling.get("block_dim")
            batch = self._get_batch_factors(tensor_a_l0a, tensor_b_l0b)
            n_factors, m_factors = block_dim[1], block_dim[2]
            m_l0_shape = self.tiling.get("AL0_matrix")[0] * self.block_in
            n_l0_shape = self.tiling.get("BL0_matrix")[1] * self.block_out

            cce_emitinsn_params.cceEmitParamsIns.insert_param("batch", batch)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_m_blk", m_factors)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_n_blk", n_factors)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_m_split", m_l0_shape)
            cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_n_split", n_l0_shape)
            emit_insn_cmd = "dma_copy_matmul_transpose"
        else:
            emit_insn_cmd = "dma_copy"

        out_insn_dict = None
        if (not self.align_a) or (not self.align_b):
            out_insn_dict = {"no_overlap": 1}
        self._emit_insn_func(res, 0, emit_insn_cmd, insn_dict=out_insn_dict, mode=1)
        # temp
        overload_flag = True
        self._set_overload_flag(overload_flag, res)

    def _set_scope_buffer_type(self, header_tensors):
        res = self.res
        sch = self.sch
        gm_ub = None
        ele_header_ub_tensors = list()
        if not self.fusion_ele:
            return gm_ub, ele_header_ub_tensors, dict()

        in_out_tensor_map = dict()
        self._gen_in_out_tensor_map(res, in_out_tensor_map)
        tensor_c_gm = self.TENSOR_MAP.get("c_gm")
        # multi output fusion with elementwise
        multi_output_flag = isinstance(self.res_ori, list)
        multi_output_flag = multi_output_flag and self.fusion_ele and tensor_c_gm in self.res_ori
        self.multi_output_flag = multi_output_flag
        if multi_output_flag:
            gm_ub = sch.cache_read(tensor_c_gm, tbe_platform_info.scope_ubuf,
                                   in_out_tensor_map[tensor_c_gm])
            self.fusion_tensor_cub.append(gm_ub)
            self.fusion_tensor_cub.append(tensor_c_gm)
            self.sch[tensor_c_gm.op.input_tensors[0]].reused_by(gm_ub)

        tensor_ele_ub = list()
        header_tensors = list(set(header_tensors))
        for ten_i in header_tensors:
            if in_out_tensor_map[ten_i][0] not in self.matmul_tensors:
                ele_ub = sch.cache_read(ten_i, tbe_platform_info.scope_ubuf, in_out_tensor_map[ten_i])
                tensor_ele_ub.append(ele_ub)
                ele_header_ub_tensors.append(ele_ub)

        axpy_2_parent = self._get_elewise_ub_tensors(tensor_ele_ub)
        elemwise_tensors = self.elemwise_tensors
        elemwise_tensors.clear()
        for ten_i in tensor_ele_ub:
            elemwise_tensors.append(ten_i)
        return gm_ub, ele_header_ub_tensors, axpy_2_parent

    def _set_overload_flag(self, overload_flag, flag_on_tensor):
        """
        set overload flag
        """
        current_op = self.sch[flag_on_tensor]
        tensor_len = len(flag_on_tensor.shape)
        scopes_intrins = self.sch_agent[flag_on_tensor].intrin_scopes(tensor_len)
        pragma_axis = scopes_intrins[0]
        if current_op is not None and pragma_axis is not None:
            if overload_flag:
                current_op.pragma(pragma_axis, "json_info_cache_read_mode", 0)
            else:
                current_op.pragma(pragma_axis, "json_info_cache_read_mode", 1)

    def _compute_ab_buffer(self):
        """
        Calculates the number of times the space used. The value is based on fp16.
        """
        buffer_reuse_dict = self.buffer_reuse_dict
        def enter(tensor_list, fix_dtype):
            """
            the enter to calculate buffer used multi
            """
            conuted_tensor_list = list()
            not_need_conuted_tensor_list = list()
            fused_num = 0
            for tensor in tensor_list:
                if tensor in self.elewise_compute_inline_list + self.compute_inline_list:
                    continue
                if tensor in buffer_reuse_dict:
                    anthor_tensor = buffer_reuse_dict.get(tensor)
                    if anthor_tensor in conuted_tensor_list:
                        continue
                    a_dtype_width = self.DTYPE_WIDTH_MAP.get(tensor.dtype)
                    b_dtype_width = self.DTYPE_WIDTH_MAP.get(anthor_tensor.dtype)
                    cur_dtype_num = a_dtype_width if a_dtype_width > b_dtype_width else b_dtype_width
                    cur_tensor = tensor if a_dtype_width > b_dtype_width else anthor_tensor
                    small_one_tensor = anthor_tensor if a_dtype_width > b_dtype_width else tensor
                    # small_one is conuted too
                    conuted_tensor_list.append(small_one_tensor)
                else:
                    cur_dtype_num = self.DTYPE_WIDTH_MAP.get(tensor.dtype)
                    cur_tensor = tensor
                if cur_tensor in conuted_tensor_list:
                    continue
                conuted_tensor_list.append(cur_tensor)
                fused_num += cur_dtype_num

            fused_num = fused_num / self.DTYPE_WIDTH_MAP.get(fix_dtype) - 1
            return fused_num

        if self.TENSOR_MAP.get("a_ub") is not None:
            a_fused_num = int(enter(self.tensors_in_aub, self.TENSOR_MAP.get("a_ub").dtype)*10)
        else:
            a_fused_num = 0
        if self.TENSOR_MAP.get("b_ub") is not None:
            b_fused_num = int(enter(self.tensors_in_bub, self.TENSOR_MAP.get("b_ub").dtype)*10)
        else:
            b_fused_num = 0

        return a_fused_num, b_fused_num

    def _mem_process(self):
        sch = self.sch
        TENSOR_MAP = self.TENSOR_MAP
        if self.is_dynamic:
            sch.disable_allocate(tbe_platform_info.scope_cbuf)
            sch.disable_allocate(tbe_platform_info.scope_ca)
            sch.disable_allocate(tbe_platform_info.scope_cb)
            sch.disable_allocate(tbe_platform_info.scope_cc)
            sch.disable_allocate(tbe_platform_info.scope_ubuf)

            # get l1 bound
            sch[TENSOR_MAP.get("a_l1")].set_storage_bound(self._get_al1_bound())
            sch[TENSOR_MAP.get("b_l1")].set_storage_bound(self._get_bl1_bound())

            # mem_unique
            sch[TENSOR_MAP.get("a_l1")].mem_unique()
            sch[TENSOR_MAP.get("b_l1")].mem_unique()
            sch[TENSOR_MAP.get("a_l0a")].mem_unique()
            sch[TENSOR_MAP.get("b_l0b")].mem_unique()
            c_ub = self.res.op.input_tensors[0]
            sch[c_ub].mem_unique()
            bias_ub = TENSOR_MAP.get("bias_ub")
            if bias_ub is not None:
                sch[bias_ub].mem_unique()
            else:
                sch[TENSOR_MAP.get("c_l0c")].mem_unique()

    def _get_al1_bound(self):
        a_matrix_dim = [self._get_value(i) for i in self.TENSOR_MAP["a_l0a"].shape]
        if self.tiling["AL1_shape"]:
            m_bound = self.tiling["AL1_shape"][1] * self.tiling["CL0_matrix"][1] * self.block_in
            if self.al1_attach_status == "c_gm":
                k_bound = (self._int_ceil_div(a_matrix_dim[-3], self.tiling.get("AL0_matrix")[1])
                    * self.tiling.get("AL0_matrix")[1] * self.block_reduce)
            else:
                k_bound = self.tiling["AL1_shape"][0]
            al1_bound = m_bound * k_bound
        else:
            k_shape = a_matrix_dim[-3]
            k_shape = (self._int_ceil_div(k_shape, self.tiling.get("AL0_matrix")[1])
                * self.tiling.get("AL0_matrix")[1] * self.block_reduce)
            k_bound = k_shape
            if self.tiling["block_dim"][2] == 1:
                m_bound = a_matrix_dim[-4] * self.block_in
            else:
                m_parts = self._int_ceil_div(a_matrix_dim[-4], self.tiling["CL0_matrix"][1])
                m_factors = self._int_ceil_div(m_parts, self.tiling["block_dim"][2])
                m_bound = m_factors * self.tiling["CL0_matrix"][1] * self.block_in
            al1_bound = m_bound * k_bound
        return al1_bound

    def _get_bl1_bound(self):
        b_matrix_dim = [self._get_value(i) for i in self.TENSOR_MAP["b_l0b"].shape]
        if self.tiling["BL1_shape"]:
            n_bound = self.tiling["BL1_shape"][1] * self.tiling["CL0_matrix"][0] * self.block_out
            if self.bl1_attach_status == "c_gm":
                k_bound = (self._int_ceil_div(b_matrix_dim[-4], self.tiling.get("AL0_matrix")[1])
                    * self.tiling.get("AL0_matrix")[1] * self.block_reduce)
            else:
                k_bound = self.tiling["BL1_shape"][0]
            bl1_bound = n_bound * k_bound
        else:
            k_shape = b_matrix_dim[-4]
            k_shape = (self._int_ceil_div(k_shape, self.tiling.get("AL0_matrix")[1])
                * self.tiling.get("AL0_matrix")[1] * self.block_reduce)
            k_bound = k_shape
            if self.tiling["block_dim"][1] == 1:
                n_bound = b_matrix_dim[-3] * self.block_out
            else:
                n_parts = self._int_ceil_div(b_matrix_dim[-3], self.tiling["CL0_matrix"][0])
                n_factors = self._int_ceil_div(n_parts, self.tiling["block_dim"][1])
                n_bound = n_factors * self.tiling["CL0_matrix"][0] * self.block_out
            bl1_bound = n_bound * k_bound
        return bl1_bound

    def _set_var_range_for_dynamic(self):
        if not self.is_dynamic:
            return
        m_shape = self.TENSOR_MAP.get("a_l0a").shape[-4]
        k_shape = self.TENSOR_MAP.get("a_l0a").shape[-3]
        n_shape = self.TENSOR_MAP.get("b_l0b").shape[-3]

        var_range_dict = self.dynamic_para.get("var_range")
        self.sch.set_var_range(m_shape, *var_range_dict.get("m"))
        self.sch.set_var_range(k_shape, *var_range_dict.get("k"))
        self.sch.set_var_range(n_shape, *var_range_dict.get("n"))
        batch_range = var_range_dict.get("batch")
        if self.have_batch and (batch_range is not None):
            batch_shape = self.TENSOR_MAP.get("c_l0c").shape[0]
            self.sch.set_var_range(batch_shape, *batch_range)


class FormatType(Enum):
    """
    format type
    """
    FRACTAL_NZ = 0
    ND = 1


class CalculateMultiUB(object):

    BYTES_DTYPE = {"uint64": 8, "float16": 2, "float32": 4, "int32": 4,
                    "int16": 2, "uint16": 2, "int8": 1, "uint8": 1,
                    "int4": 0.5, "bool": 1}

    def __init__(self, start_tensor, end_tensor, not_count_list):
        self.start_tensor = start_tensor
        self.end_tensor = end_tensor
        self.not_count_list = not_count_list
        self.tensor_occur_times = dict()
        self.ub_res = 0
        self.end_tensor_shape = 0

    def calculate_multi_ub_enter(self):
        self.end_tensor_shape = functools.reduce(lambda x, y: x*y, self.end_tensor.shape)
        self.end_tensor_shape = self.end_tensor_shape.value if hasattr(self.end_tensor_shape, "value") else self.end_tensor_shape
        self._calculate_multi_ub_auto()
        return self.ub_res

    def _calculate_multi_ub_auto(self):
        tensor_q = Queue()
        tensor_q.put(self.end_tensor)
        while not tensor_q.empty():
            tensor_out = tensor_q.get()
            if tensor_out == self.start_tensor:
                self._compute_result(tensor_out)
                continue
            merge_flag = False
            input_tensors = list(tensor_out.op.input_tensors)
            for tensor_in in input_tensors:
                if tensor_in in self.tensor_occur_times.keys():
                    continue
                else:
                    if tensor_in in self.not_count_list:
                        if tensor_in != self.start_tensor:
                            self._merge_compute_inline(tensor_in, input_tensors)
                        continue
                    tensor_q.put(tensor_in)
                    self.tensor_occur_times[tensor_in] = 1
                    if merge_flag:
                        continue
                    if self._can_merge(tensor_out, tensor_in):
                        merge_flag = True
            if not merge_flag:
                self._compute_result(tensor_out)
        return

    def _merge_compute_inline(self, tensor, input_tensors):
        tensor_not_count_q = Queue()
        tensor_not_count_q.put(tensor)
        while not tensor_not_count_q.empty():
            cur_tensor = tensor_not_count_q.get()
            for next_tensor in list(cur_tensor.op.input_tensors):
                if next_tensor in self.not_count_list:
                    if next_tensor != self.start_tensor:
                        tensor_not_count_q.put(next_tensor)
                else:
                    input_tensors.append(next_tensor)
        return

    def _can_merge(self, tensor_out, tensor_in):
        tensor_out_dtype = tensor_out.dtype
        tensor_out_format = FormatType.FRACTAL_NZ if len(tensor_out.shape) in (4, 5) else FormatType.ND
        tensor_in_dtype = tensor_in.dtype
        tensor_in_format = FormatType.FRACTAL_NZ if len(tensor_in.shape) in (4, 5) else FormatType.ND

        if self._not_count(tensor_in):
            return False
        can_merge = (tensor_out_dtype == tensor_in_dtype) and (tensor_out_format == tensor_in_format)
        return can_merge

    def _compute_result(self, tensor):
        if self._not_count(tensor):
            return
        self.ub_res += self.BYTES_DTYPE[tensor.dtype]
        return

    def _not_count(self, tensor):
        if tensor in self.not_count_list:
            return True
        shape_size = functools.reduce(lambda x, y: x*y, tensor.shape)
        shape_size = shape_size.value if hasattr(shape_size, "value") else shape_size
        if shape_size == 1 or shape_size * 4 <= self.end_tensor_shape:
            return True
        return False