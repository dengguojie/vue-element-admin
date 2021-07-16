#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
layer_norm_cube schedule
"""

from tbe.common import platform as cce
from tbe.common.platform import CORE_NUM
from tbe.common.platform.platform_info import get_soc_spec
from tbe import tvm


def _print_ir_conv(process, sch):
    """
    print ir for input sch

    Parameter:
    --------------------------------------------------------------
    :param process: tag
    :param sch: schedule
    :return: IR process
    ---------------------------------------------------------------
    """
    if "debug" in process:
        start = process + " IR start"
        end = process + " IR end\n"
        sch = sch.normalize()
        print(start)
        bounds = tvm.schedule.InferBound(sch)
        stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
        print(stmt)
        print(end)


def check_reget_multioutput_list(outs):
    """
    check the multioutput of layernorm
    """
    if not isinstance(outs, list):
        return False

    if len(outs) != 3:
        return False
    res_out, mean_out, var_out = outs
    if res_out.op.tag == "res_out" and mean_out.op.tag == "mean_out" and var_out.op.tag == "var_out":
        return True
    else:
        return False


def reget_layernorm_multioutput(outs):
    """
    return multioutputs to the virtual output
    """
    if check_reget_multioutput_list(outs):
        res_out, mean_out, var_out = outs
        virtual_res = tvm.compute(res_out.shape,
                                  lambda *indices:
                                  res_out(*indices)
                                  + mean_out(*indices[:-4], indices[-3]*16+indices[-2], 0)
                                  + var_out(*indices[:-4], indices[-3]*16+indices[-2], 0),
                                  name='cube_layer_norm',
                                  tag="cube_layer_norm")
        return [virtual_res] + outs
    else:
        return outs


def _get_all_tensors(res):
    """
    get all tensor
    :param res: tensor
    :return: list
    """

    all_tensor = dict()

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


def _set_tensor_scope(sch, des_tensor):
    """
    set the scope of tensor
    """
    Tensor_MAP= {}
    out_tensor = []
    ub_tensor = []
    for tensor_name, tensor_member in des_tensor.items():
        if "out" in tensor_name:
            out_tensor.append(tensor_member)
        elif tensor_name in ("x", "gamma", "beta"):
            pass
        elif "l0a" in tensor_name:
            Tensor_MAP["x_l0a"] = tensor_member
            sch[tensor_member].set_scope(cce.scope_ca)
        elif "l0b" in tensor_name:
            if tensor_name == "x_l0b":
                Tensor_MAP["x_l0b"] = tensor_member
            else:
                Tensor_MAP["one_l0b"] = tensor_member
            sch[tensor_member].set_scope(cce.scope_cb)
        elif "l0c" in tensor_name:
            sch[tensor_member].set_scope(cce.scope_cc)
            if tensor_name == "x_sum_l0c":
                Tensor_MAP["x_l0c"] = tensor_member
            else:
                Tensor_MAP["xx_l0c"] = tensor_member
        elif "l1" in tensor_name:
            Tensor_MAP["x_l1"] = tensor_member
            sch[tensor_member].set_scope(cce.scope_cbuf)
        else:
            sch[tensor_member].set_scope(cce.scope_ubuf)
            ub_tensor.append(tensor_member)
        Tensor_MAP["ub_tensor"] = ub_tensor
        Tensor_MAP["out_tensor"] = out_tensor
    return Tensor_MAP


def _get_tiling(res):
    k_dims = res.shape[-4].value
    tiling = \
        {'L1_shape': [1, 1, 16, 16],
         "AL1_shape" : [1, 1],
         'CL0_matrix': [k_dims, 1, 16, 16],
         'CUB_matrix': [k_dims, 1, 16, 16]
         }

    batch_m = res.shape[-3].value
    for batch_mem in res.shape[:-4]:
        batch_m *= batch_mem.value
    core_num = get_soc_spec(CORE_NUM)

    if batch_m < core_num:
        tiling["block_dim"] = batch_m
    else:
        tiling["block_dim"] = core_num

    return tiling


class CceLayerNormCubeOp(object):
    """
    CceLayerNormCubeOp: schedule definition of layernorm

    Functions
    ----------
    __init__ : initialization

    schedule : schedule definition of layernorm

    """

    def __init__(self, scope):
        """
        initialization

        Parameters:
        ----------
        scope : scope definition

        Returns
        -------
        None
        """
        self._scope = scope
        self._sch = None
        self._res = None
        self._batch = False

    def _split_l0c_and_ub(self, tiling):
        """
        split m, n based ub and l0c unit
        """
        n_l0c_factor, m_l0c_factor, m_l0c_block_factor = tiling["CL0_matrix"][:3]
        n_ub_factor, m_ub_factor, m_ub_block_factor = tiling["CUB_matrix"][:3]

        # (N, M, 16, 16)
        n_outer, n_inner = self._sch[self._res].split(self._res.op.axis[-4], factor=n_l0c_factor)
        m_outer, m_inner = self._sch[self._res].split(self._res.op.axis[-3], factor=m_l0c_factor)

        m_ub_parts = m_l0c_factor // m_ub_factor
        n_ub_parts = n_l0c_factor // n_ub_factor

        n_inner_outer, n_inner_inner = self._sch[self._res].split(n_inner, nparts=n_ub_parts)
        m_inner_outer, m_inner_inner = self._sch[self._res].split(m_inner, nparts=m_ub_parts)

        if m_l0c_block_factor // m_ub_block_factor != 1:
            block_m_ub_parts = m_l0c_block_factor // m_ub_block_factor
            m_block_outer, m_m_block_inner = self.sch[self._res].split(
                self._res.op.axis[-2], nparts=block_m_ub_parts)
            self._sch[self._res].reorder(n_outer,
                                     m_outer,
                                     n_inner_outer,
                                     m_inner_outer,
                                     m_block_outer,
                                     n_inner_inner,
                                     m_inner_inner)
            m_inner_outer = self._sch[self._res].fuse(m_inner_outer, m_block_outer)
        else:
            self._sch[self._res].reorder(n_outer, m_outer, n_inner_outer, m_inner_outer, n_inner_inner, m_inner_inner)

        return n_outer, m_outer, m_inner_outer, n_inner_inner

    def _split_multi_block(self, tiling, n_outer, m_outer):
        """
        split m axis based on block dims
        """
        block_parts = tiling["block_dim"]
        self._sch[self._res].reorder(m_outer, n_outer)
        if self._batch:
            m_outer = self._sch[self._res].fuse(*self._sch[self._res].op.axis[:-4], m_outer)
        m_outer_outer, m_outer_inner = self._sch[self._res].split(m_outer, nparts=block_parts)

        block_axis_outer, block_axis_inner = self._sch[self._res].split(m_outer_outer, 1)
        blockidx = tvm.thread_axis("blockIdx.x")
        self._sch[self._res].bind(block_axis_outer, blockidx)

        return block_axis_inner, m_outer_inner

    def _split_l1(self, tiling, m_outer_inner):
        """
        split m of AL1 unit
        """
        al1_m = tiling["AL1_shape"][1]
        m_outer_inner_outer, m_outer_inner_inner = self._sch[self._res].split(m_outer_inner, al1_m)
        return m_outer_inner_outer, m_outer_inner_inner

    def _attach_at_ub(self, tensor_map, ub_at_axis, m_outer_inner_inner, block_axis_inner):
        """
        attach ub tensor in res tensor 
        """
        for out_tensor in tensor_map["out_tensor"]:
            if out_tensor.op.name in ("mean_out", "var_out"):
                self._sch[out_tensor].compute_at(self._sch[self._res], m_outer_inner_inner)
            else:
                self._sch[out_tensor].compute_at(self._sch[self._res], ub_at_axis)

        at_cub_list = ["cast_res", "cast_x_ub", "x_ub", "scale_add",
                       "scale_mul", "mean_mul_var", "x_sub_mean"]
        for ub_tensor in tensor_map["ub_tensor"]:
            if ub_tensor.op.name in at_cub_list:
                self._sch[ub_tensor].compute_at(self._sch[self._res], ub_at_axis)
            elif "gamma" in ub_tensor.op.name or "beta" in ub_tensor.op.name:
                self._sch[ub_tensor].compute_at(self._sch[self._res], block_axis_inner)
            else:
                self._sch[ub_tensor].compute_at(self._sch[self._res], m_outer_inner_inner)

    def _attach_l0c(self, x_l0c, xx_l0c, m_outer_inner_inner):
        """
        attach l0c tensor in res tensor 
        """
        self._sch[x_l0c].compute_at(self._sch[self._res ], m_outer_inner_inner)
        self._sch[xx_l0c].compute_at(self._sch[self._res ], m_outer_inner_inner)

    def _attach_l0ab(self, tensor_map, m_outer_inner_inner, m_outer_inner_outer, block_axis_inner):
        """
        attach l0a and l0c tensor in res tensor 
        """
        self._sch[tensor_map["one_l0b"]].compute_at(self._sch[self._res], block_axis_inner)
        self._sch[tensor_map["x_l0a"]].compute_at(self._sch[self._res], m_outer_inner_inner)

        self._sch[tensor_map["x_l0b"]].compute_at(self._sch[self._res], m_outer_inner_inner)
        self._sch[tensor_map["x_l1"]].compute_at(self._sch[self._res], m_outer_inner_outer)

    def _double_buffer(self, tensor_map):
        """
        enable doubel
        """
        for tensor_ub in tensor_map["ub_tensor"]:
            if "gamma" not in tensor_ub.op.name and "beta" not in tensor_ub.op.name:
                self._sch[tensor_ub].double_buffer()
        for item in ("x_l1", "x_l0c", "xx_l0c", "x_l0b", "x_l0a"):
            self._sch[tensor_map[item]].double_buffer()

    def _emit_insn_out(self, tensor_map, n_inner_inner):
        """
        emit insn of ddr tensor which from ub to ddr
        """
        self._sch[self._res].emit_insn(n_inner_inner, "phony_insn")
        for out_tensor in tensor_map["out_tensor"]:
            self._sch[out_tensor].emit_insn(self._sch[out_tensor].op.axis[0], "dma_copy")

    def _emit_insn_ub(self, tensor_map):
        """
        emit insn of all ub tensor
        """
        for ub_tensor in tensor_map["ub_tensor"]:
            if ub_tensor.op.name in ("xx_sum_cub", "x_sum_cub", "x_ub", "input_gamma_ub", "input_beta_ub"):
                self._sch [ub_tensor].emit_insn(self._sch [ub_tensor].op.axis[0], "dma_copy")
            elif ub_tensor.op.name in ("trans_var", "trans_mean"):
                ub_ori_shape_len = len(ub_tensor.shape) - 2
                self._sch [ub_tensor].buffer_align(
                    *(((1, 1),) * ub_ori_shape_len),
                    (16, 16),
                    (16, 16)
                )
                self._sch [ub_tensor].emit_insn(self._sch [ub_tensor].op.axis[0], "dma_copy")
            elif ub_tensor.op.name == "xx_sum_cub_scalar":
                self._sch [ub_tensor].compute_inline()
            elif ub_tensor.op.name == "xx_sum_cub_broadcast":
                self._sch [ub_tensor].emit_insn(self._sch [ub_tensor].op.axis[0], "vector_broadcast")
            elif ub_tensor.op.name in ("scale_mul", "scale_add"):
                block_num_fp32 = 8
                k_inner_out, k_inner_inner = self._sch [ub_tensor].split(self._sch [ub_tensor].op.axis[-1], factor=block_num_fp32)
                self._sch [ub_tensor].reorder(k_inner_out,
                                       *self._sch [ub_tensor].op.axis[0:-1],
                                       k_inner_inner)
                self._sch [ub_tensor].emit_insn(self._sch [ub_tensor].op.axis[0], "vector_auto")
            else:
                self._sch [ub_tensor].emit_insn(self._sch [ub_tensor].op.axis[0], "vector_auto")

    def _emit_insn_cube(self, tensor_map):
        """
        emit insn of tensor in cube cal
        """
        for tensor_name in tensor_map.keys():
            if tensor_name in ("x_l1", "x_l0a", "x_l0b"):
                self._sch[tensor_map[tensor_name]].emit_insn(self._sch[tensor_map[tensor_name]].op.axis[0], "dma_copy")
            elif tensor_name in ("x_l0c", "xx_l0c"):
                k_outer, _ = tensor_map[tensor_name].op.reduce_axis
                k_outer_outer, k_outer_inner = self._sch[tensor_map[tensor_name]].split(k_outer, nparts=1)
                self._sch[tensor_map[tensor_name]].reorder(k_outer_outer, *tensor_map[tensor_name].op.axis)
                mad_dict = {"mad_pattern": 0, "k_outer": [k_outer_outer]}
                self._sch[tensor_map[tensor_name]].emit_insn(self._sch[tensor_map[tensor_name]].op.axis[-4], "mad", mad_dict)
            elif tensor_name == "one_l0b":
                self._sch[tensor_map[tensor_name]].emit_insn(self._sch[tensor_map[tensor_name]].op.axis[0], "set_2d")

    def _emit_insn(self, tensor_map, n_inner_inner):
        """
        emit insn of all tensor
        """
        self._emit_insn_out(tensor_map, n_inner_inner)
        self._emit_insn_ub(tensor_map)
        self._emit_insn_cube(tensor_map)

    def schedule(self, res, spec_node_list, sch_list, tiling_case=None):
        """
        auto_schedule for cce AI-CORE.
        For now, only layernorm supported.

        Parameters
        ----------
        res : tvm.tensor

        spec_node_list: tvm.tensor list

        sch_list: use sch_list[0] to return layernorm schedule

        tiling_case: fix tiling for sch

        Returns
        -------
        True for success, False for no schedule
        """

        sch = sch_list[0]
        self._sch = sch
        self._res = res
        if len(res.shape) > 4:
            self._batch = True
    
        # get all tensor of the sch
        all_tensor = _get_all_tensors(res)
        # set the scope of different tensor
        tensor_map = _set_tensor_scope(sch, all_tensor)

        x_l0c = tensor_map["x_l0c"]
        xx_l0c = tensor_map["xx_l0c"]

        # get the split factor and nparts of case
        if tiling_case is None:
            tiling = _get_tiling(res)
        else:
            tiling = tiling_case

        # split mini parts in ub and L0c
        n_outer, m_outer, ub_at_axis, n_inner_inner = self._split_l0c_and_ub(tiling)

        # blind blocks
        block_axis_inner, m_outer_inner = self._split_multi_block(tiling, n_outer, m_outer)

        # split parts from L1 -> L0C
        m_outer_inner_outer, m_outer_inner_inner = self._split_l1(tiling, m_outer_inner)

        # compute at of tensor
        self._attach_at_ub(tensor_map, ub_at_axis, m_outer_inner_inner, block_axis_inner)
        self._attach_l0c(x_l0c, xx_l0c, m_outer_inner_inner)
        self._attach_l0ab(tensor_map, m_outer_inner_inner, m_outer_inner_outer, block_axis_inner)

        # emit insn of tensor
        self._emit_insn(tensor_map, n_inner_inner)

        _print_ir_conv("final IR", sch)
        tensor_map.clear()
        all_tensor.clear()
        tiling.clear()

        return True