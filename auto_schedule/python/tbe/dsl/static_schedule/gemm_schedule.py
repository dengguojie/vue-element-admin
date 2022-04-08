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

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.buildcfg import build_config
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.dsl.base.operation import in_dynamic
from . import gemm_schedule_util as util

K_AXIS_ALIGN_FACTOR = 2


def _get_value(shape_object):
    """
    get the value of shape_object when having attr "value"
    """
    return shape_object.value if hasattr(shape_object, "value") else shape_object


def update_gemm_multiout_list(outs):
    """
    remove virtual res from out_list in multi_out scene
    """
    if not isinstance(outs, (list, tuple)):
        return outs
    for tensor in outs:
        if tensor.op.tag == "gemm_virtual_res":
            outs.remove(tensor)
    return outs


def print_ir_matmul(process, sch):
    """
    print ir for input sch
    :param process: tag
    :param sch: schedule
    :return: IR process
    """
    if process == "debug":
        with build_config():
            start = process + " IR start"
            end = process + " IR end\n"
            sch = sch.normalize()
            print(start)
            bounds = tvm.schedule.InferBound(sch)
            stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
            print(stmt)
            print(end)


class GemmScheduleV2:
    """
    class of gemm schedule
    """
    def __init__(self, res, sch_list, dynamic_para):
        self.res_list = list(res) if isinstance(res, (list, tuple)) else [res]
        self.res = res[0] if isinstance(res, (list, tuple)) else res
        self.sch = sch_list[0]
        self.dynamic_para = dynamic_para
        self.trans_a = False
        self.trans_b = False
        self.fuse_num = 0

    @staticmethod
    def get_al1_bound(tiling, tensor_map):
        """
        cal the l1 bound of al1
        """
        # get parts from gm to l0c every core of m direction: ceil(m1 // blockdim_m, m_l0c)
        l0c_parts = util.int_ceil_div(_get_value(tensor_map.get("a_l0a").shape[-4]) // tiling.get("block_dim")[2],
                                      tiling.get("CL0_matrix")[1])
        if tiling.get("AL1_shape"):
            # get parts from gm to L1: l0c_parts // multiple_of_L1_and_L0
            al1_parts = util.int_ceil_div(l0c_parts, tiling.get("AL1_shape")[1])
            # k on Al1
            k_bound = tiling.get("AL1_shape")[0]
        else:
            # Al1 full load, move all data in one time
            al1_parts = 1
            # total K of input: k0 * k1
            k_bound = _get_value(tensor_map.get("a_l0a").shape[-3]) * _get_value(tensor_map.get("a_l0a").shape[-1])
        # parts from from gm to l0c ignore multi core
        l0c_bounds = util.int_ceil_div(_get_value(tensor_map.get("a_l0a").shape[-4]), tiling.get("CL0_matrix")[1])
        # multiple of al1 and l0c space
        al1_bounds = util.int_ceil_div(l0c_bounds, al1_parts)
        # divide blockdim since l0c_bounds ignored multicore to get m on L1
        m_bound = (util.int_ceil_div(al1_bounds, tiling.get("block_dim")[2]) * tiling.get("CL0_matrix")[1] *
                   tiling.get("CL0_matrix")[2])
        return k_bound * m_bound

    @staticmethod
    def get_bl1_bound(tiling, tensor_map):
        """
        cal the l1 bound of bl1
        """
        l0c_parts = util.int_ceil_div(_get_value(tensor_map.get("b_l0b").shape[-3]) // tiling.get("block_dim")[1],
                                      tiling.get("CL0_matrix")[0])
        if tiling.get("BL1_shape"):
            bl1_parts = util.int_ceil_div(l0c_parts, tiling.get("BL1_shape")[1])
            k_bound = tiling.get("BL1_shape")[0]
        else:
            bl1_parts = 1
            k_bound = _get_value(tensor_map.get("b_l0b").shape[-4]) * _get_value(tensor_map.get("b_l0b").shape[-1])
        l0c_bounds = util.int_ceil_div(_get_value(tensor_map.get("b_l0b").shape[-3]), tiling.get("CL0_matrix")[0])
        bl1_bounds = util.int_ceil_div(l0c_bounds, bl1_parts)
        m_bound = (util.int_ceil_div(bl1_bounds, tiling.get("block_dim")[1]) * tiling.get("CL0_matrix")[0] *
                   tiling.get("CL0_matrix")[3])
        return k_bound * m_bound

    def _init_tiling_input(self, tensor_map):
        """
        init the input of tiling
        """
        a_l0a = tensor_map["a_l0a"]
        b_l0b = tensor_map["b_l0b"]
        l0a_shape = util.shape_to_list(a_l0a.shape)
        l0b_shape = util.shape_to_list(b_l0b.shape)
        self.trans_a = ("transpose_a" in a_l0a.op.attrs and a_l0a.op.attrs["transpose_a"] == "true")
        self.trans_b = ("transpose_b" in b_l0b.op.attrs and b_l0b.op.attrs["transpose_b"] == "true")
        if (not self.trans_a ^ self.trans_b) and a_l0a.dtype == "float32":
            # for some unaligned cases, shape_a=(2,4), shape_b=(4,16) for example, shape_a_l1 will be aligned as
            # (1,1,16,8) while shape_b_l1 is (2,1,16,8), the shapes on L0 are (1,1,16,8) and (2, 1, 16, 8), ka != kb
            l0a_shape[-3] = util.int_ceil_align(l0a_shape[-3], K_AXIS_ALIGN_FACTOR)
            l0b_shape[-4] = util.int_ceil_align(l0b_shape[-4], K_AXIS_ALIGN_FACTOR)
        # a_shape dim: batch_a, k1, m1, m0, k0
        a_shape = [1, l0a_shape[-3], l0a_shape[-4], l0a_shape[-2], l0a_shape[-1]]
        a_shape[0] = l0a_shape[0] if len(l0a_shape) == 5 else 1

        # b_shape dim: K1*k0, n1, 1, 1, n0
        b_shape = [l0b_shape[-4] * l0b_shape[-1], l0b_shape[-3], 1, 1, l0b_shape[-2]]

        trans_flag = 1
        if self.trans_a:
            trans_flag += 1
        if self.trans_b:
            trans_flag += 2

        return a_shape, b_shape, trans_flag

    def _get_tiling_matmul(self, tensor_map):
        """
        get tiling of matmul
        """
        kernel_name = tensor_map.get("c_l0c").op.attrs["kernel_name"]
        a_shape, b_shape, trans_flag = self._init_tiling_input(tensor_map)
        fixpipe_flag = util.get_fixpipe_flag(tensor_map)
        self.fuse_num = util.get_fused_num(tensor_map)
        info_dict = {
                "op_type": "matmul",
                "A_shape": a_shape,
                "B_shape": b_shape,
                "C_shape": None,
                "A_dtype": tensor_map["a_placehold"].dtype,
                "B_dtype": tensor_map["b_placehold"].dtype,
                "C_dtype": tensor_map["c_gm"].dtype,
                "mad_dtype": "int32" if tensor_map["a_placehold"].dtype == "int8" else "float32",
                "padl": 0,
                "padr": 0,
                "padu": 0,
                "padd": 0,
                "strideH": 1,
                "strideW": 1,
                "strideH_expand": fixpipe_flag,
                "strideW_expand": 1,
                "dilationH": trans_flag,
                "dilationW": 1,
                "group": 1,
                "bias_flag": tensor_map.get("bias_bt") is not None,
                "fused_double_operand_num": self.fuse_num,
                "kernel_name": kernel_name.value
            }
        tiling = get_tiling(info_dict)
        return tiling

    def _mem_process(self, tiling, tensor_map):
        al1_bound = self.get_al1_bound(tiling, tensor_map)
        bl1_bound = self.get_bl1_bound(tiling, tensor_map)
        a_l1, b_l1, a_l0a, b_l0b, c_l0c = (
            tensor_map["a_l1"],
            tensor_map["b_l1"],
            tensor_map["a_l0a"],
            tensor_map["b_l0b"],
            tensor_map["c_l0c"]
        )
        if in_dynamic():
            self.sch.sequential_malloc(tbe_platform_info.scope_cbuf)
            self.sch.sequential_malloc(tbe_platform_info.scope_ca)
            self.sch.sequential_malloc(tbe_platform_info.scope_cb)
            self.sch.sequential_malloc(tbe_platform_info.scope_cc)
            self.sch.sequential_malloc(tbe_platform_info.scope_ubuf)
            self.sch.sequential_malloc("local.BT")
            # get l1 bound
            self.sch[a_l1].set_buffer_size(al1_bound)
            self.sch[b_l1].set_buffer_size(bl1_bound)
            # mem_unique
            mem_unique_list = [a_l1, b_l1, a_l0a, b_l0b, c_l0c]
            if tensor_map.get("bias_l1") is not None:
                mem_unique_list += [tensor_map["bias_l1"], tensor_map["bias_bt"]]
            for mem_unique_mem in mem_unique_list:
                self.sch[mem_unique_mem].mem_unique()

    def gemm_schedule(self):
        """
        schedule enter
        param:
        res: tensor
        sch_list: list of schedule
        """
        sch = self.sch
        tensor_map = {}

        # get all tensor of the compute
        all_tensor, leaf_tensor = util.get_all_tensors(self.res)

        # set scope for simple a*b=c
        tensor_map = util.set_matmul_scope(all_tensor, sch, tensor_map)
        # if modify output fusion, please modify this function
        tensor_map = util.set_out_scope(all_tensor, leaf_tensor, sch, tensor_map, self.res_list)
        print_ir_matmul("after set scope", sch)
        if in_dynamic():
            tiling = self.dynamic_para["tiling_strategy"]
        else:
            tiling = self._get_tiling_matmul(tensor_map)
            tiling = util.check_tiling(tiling, tensor_map)

        # get factor and parts from tiling
        l0c_factor, ub_factor_parts, al1_parts, bl1_parts = util.get_aicore_factor(tiling, tensor_map)
        if len(self.res.shape) in (util.MATMUL_LEN_ND, util.BATCH_MATMUL_LEN_ND):
            l0c_factor = [
                tiling["CL0_matrix"][0]*tiling["CL0_matrix"][3],
                tiling["CL0_matrix"][1]*tiling["CL0_matrix"][2]
            ]
        # split l0c, al1, bl1
        l1_m_axis, l1_n_axis = util.split_mn_l0c_l1(self.res, sch, l0c_factor, al1_parts, bl1_parts)
        # split upon block
        batch_inner = util.split_mn_block(self.res, sch, tiling, l1_m_axis, l1_n_axis)
        # reorder m and n of l1 upon minimun memory
        if util.reorder_l1_mn_axis(tiling, al1_parts[1], bl1_parts[1]):
            sch[self.res].reorder(l1_m_axis[0], l1_n_axis[0])
        # split when need hanble ub
        if len(self.res.shape) in (util.MATMUL_LEN_ND, util.BATCH_MATMUL_LEN_ND):
            ub_factor_parts = [
                tiling["CUB_matrix"][0]*tiling["CUB_matrix"][3],
                tiling["CUB_matrix"][1]*tiling["CUB_matrix"][2]
            ]
        if tensor_map.get("ub_eltwise") is None:
            ub_factor_parts = None
        c_gm_emit_axis = util.split_ub(self.res, sch, l1_m_axis, l1_n_axis, ub_factor_parts)
        # attach tensor of l0c and bias, c_slice_axis is l1_m_axis[1]
        sch[tensor_map.get("c_l0c")].compute_at(sch[self.res], l1_m_axis[1])
        util.do_buffer_align(sch, tensor_map, self.trans_a, self.trans_b)
        util.attach_of_ub(sch, tensor_map, c_gm_emit_axis[2])
        util.attach_of_bias_table(sch, tensor_map, bl1_parts, l1_m_axis[1], batch_inner)
        util.attach_of_fixpipe(sch, tensor_map, bl1_parts, c_gm_emit_axis[2], batch_inner)
        k_axis = util.split_k(tensor_map.get("c_l0c"), sch, tiling["AL0_matrix"][1], al1_parts[0], bl1_parts[0])
        # attch of l1 tensor and l0 tensor, l1_attch_axis = l1a_k_axis, l1b_k_axis, l0_k_axis, l1a_m_axis, l1b_n_axis
        util.attach_of_l1_l0(
            sch,
            tensor_map,
            [*k_axis, l1_m_axis[0], l1_n_axis[0]],
            al1_parts,
            bl1_parts
        )
        # double buffer function
        util.double_buffer_func(sch, tensor_map, tiling)
        # emit_insn function, emit_axis is for l0c and res emit_insn
        util.emit_insn_func(sch, tensor_map, k_axis, c_gm_emit_axis)
        print_ir_matmul("final IR", sch)

        # dynamic tensor
        self._mem_process(tiling, tensor_map)

        # clear global cache
        tiling.clear()
        tensor_map.clear()
        return True


def gemm_schedule(res, sch_list, dynamic_para=None):
    """
    schedule enter
    param:
    res: tensor
    sch_list: list of schedule
    """
    gemm_sch = GemmScheduleV2(res, sch_list, dynamic_para)
    return gemm_sch.gemm_schedule()
