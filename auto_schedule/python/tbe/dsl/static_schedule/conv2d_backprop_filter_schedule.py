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
conv2d backprop filter schudule.
"""

from __future__ import absolute_import
from __future__ import print_function

from tbe import tvm
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling import tiling_api
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.compute.conv2d_backprop_filter_compute import DynamicConv2dBpFilterParams
from tbe.dsl.compute import cube_util
from tbe.dsl.static_schedule.util import parse_tbe_compile_para
from tbe.common import platform as tbe_platform
from tbe.common.context import op_context


# for debug, delete before publish
DEBUG_MODE = False
# disable double buffer, set True
DEBUG_DOUBLE_BUFFER_OFF = False

CUBE_DIM = 16
CUBE_MUL_SHAPE = 256
OPEN_DOUBLE_BUFFER = 2
DEFAULT_TILING_CASE = 32
LOOSE_LINE_CONDITION = 2
KB_2_B = 1024

# the bytes length of several dtype
BIT_RATIO_DICT = {
    "int32": 4,
    "float32": 4,
    "float16": 2,
    "uint8": 1,
    "int8": 1,
    "uint4": 0.5,
    "int4": 0.5,
    "bfloat16": 2,
}

FIXPIPE_SCOPE_MAP = {
    "quant_scale_0": "local.FB0",
    "relu_weight_0": "local.FB1",
    "relu_weight_1": "local.FB2",
    "quant_scale_1": "local.FB3"
}


def _ceil_div(dividend, divisor):
    """
    do division and round up to an integer

    """
    if divisor == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        error_manager_util.raise_runtime_error(dict_args)
    return (dividend + divisor - 1) // divisor


def _align(x_1, x_2):
    """
    do align

    """
    if x_2 == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        error_manager_util.raise_runtime_error(dict_args)
    return (x_1 + x_2 - 1) // x_2*x_2


def _get_precision_mode(op_type):
    """
    get calculation mode, high_performance or high_precision
    """
    context = op_context.get_context()
    op_infos = context.get_op_info() if context else {}
    if not op_infos:
        op_infos = {}
    for op_info in op_infos:
        if op_info.op_type == op_type:
            return op_info.precision_mode
    return ""


class CceConv2dBackpropFilterOp:
    """
    CceConv2dBackpropFilterOp: schedule definition of conv2d_backprop_filter

    Functions
    ----------
    __init__ : initialization

    schedule : schedule definition of conv2d_backprop_filter

    """

    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        """
        initialization

        Parameters:
        ----------
        scope : scope definition

        need_tensorize : whether needs tensorize

        need_pragma : whether needs pragma

        Returns
        -------
        None
        """
        self.scope = scope
        self.need_tensorize = need_tensorize
        self.need_pragma = need_pragma
        self.spec_node_list = []
        self.cube_vector_split = tbe_platform_info.get_soc_spec("CUBE_VECTOR_SPLIT")
        self.l1_size = tbe_platform_info.get_soc_spec("L1_SIZE")  # L1 size
        self._corenum = tbe_platform_info.get_soc_spec("CORE_NUM")
        self._loc_size = tbe_platform_info.get_soc_spec("L0C_SIZE")
        self._lob_size = tbe_platform_info.get_soc_spec("L0B_SIZE")
        self.c0_size = tbe_platform.C0_SIZE
        self.l0b_dma_flag = False
        self.tensor_map = {}
        self.var_map = {}

    def schedule(self, res, spec_node_list, sch_list, dynamic_para=None):
        """
        schedule definition of conv2d_backprop_filter

        Parameters:
        ----------
        res :

        spec_node_list :

        sch_list:

        dynamic_para : A dict of dynamic shape parameters

        Returns
        -------
        None
        """

        self.spec_node_list = spec_node_list

        def _get_dyn_mode(dynamic_para):
            if not dynamic_para:
                return None
            var_range = dynamic_para.get("var_range")
            if not var_range:
                return None
            if "batch" in var_range and len(var_range) == 1:
                return "dynamic_batch"
            for var in ("dedy_h", "dedy_w", "fmap_h", "fmap_w"):
                if var in var_range:
                    return "dynamic_hw"
            return None

        def _tiling_shape_check():
            """
            do tiling shape paramters general check

            """

            al1_shape = tiling.get("AL1_shape")
            bl1_shape = tiling.get("BL1_shape")
            al0_matrix = tiling.get("AL0_matrix")
            bl0_matrix = tiling.get("BL0_matrix")
            cl0_matrix = tiling.get("CL0_matrix")
            if al1_shape and al1_shape[1] < 1:
                    dict_args = {}
                    dict_args['errCode'] = "E64007"
                    dict_args['axis_name'] = "m"
                    dict_args['param_name'] = "AL1_shape[1]"
                    dict_args['param_value'] = str(al1_shape[1])
                    error_manager_util.raise_runtime_error(dict_args)

            if bl1_shape and bl1_shape[1] < 1:
                    dict_args = {}
                    dict_args['errCode'] = "E64007"
                    dict_args['axis_name'] = "n"
                    dict_args['param_name'] = "BL1_shape[1]"
                    dict_args['param_value'] = str(bl1_shape[1])
                    error_manager_util.raise_runtime_error(dict_args)

            if al0_matrix:
                if al0_matrix[0] != cl0_matrix[1]:
                    dict_args = {}
                    dict_args['errCode'] = "E64008"
                    dict_args['axis_name'] = 'm_axis'
                    dict_args['param_1'] = "AL0_matrix"
                    dict_args['param_2'] = "CL0_matrix"
                    dict_args['value_1'] = str(al0_matrix[0])
                    dict_args['value_2'] = str(cl0_matrix[1])
                    error_manager_util.raise_runtime_error(dict_args)

            if bl0_matrix:
                if bl0_matrix[1] != cl0_matrix[0]:
                    dict_args = {}
                    dict_args['errCode'] = "E64008"
                    dict_args['axis_name'] = 'n_axis'
                    dict_args['param_1'] = "BL0_matrix"
                    dict_args['param_2'] = "CL0_matrix"
                    dict_args['value_1'] = str(bl0_matrix[1])
                    dict_args['value_2'] = str(cl0_matrix[0])
                    error_manager_util.raise_runtime_error(dict_args)

            if al0_matrix and bl0_matrix:
                if al0_matrix[1] != bl0_matrix[0]:
                    dict_args = {}
                    dict_args['errCode'] = "E64008"
                    dict_args['axis_name'] = 'k_axis'
                    dict_args['param_1'] = "AL0_matrix"
                    dict_args['param_2'] = "BL0_matrix"
                    dict_args['value_1'] = str(al0_matrix[1])
                    dict_args['value_2'] = str(bl0_matrix[0])
                    error_manager_util.raise_runtime_error(dict_args)

        def _tiling_buffer_check():
            """
            Do buffer paramters general check

            """
            block_cout = tiling.get("block_dim")

            al1_pbuff = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
            bl1_pbuff = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
            al0_pbuff = tiling.get("manual_pingpong_buffer").get("AL0_pbuffer")
            bl0_pbuff = tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
            l0c_pbuff = tiling.get("manual_pingpong_buffer").get("CL0_pbuffer")
            cub_pbuff = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")
            cl0_matrix = tiling.get("CL0_matrix")
            cub_matrix = tiling.get("CUB_matrix")
            if (not self.cube_vector_split
                and (cl0_matrix[0] % cub_matrix[0] != 0 or cl0_matrix[1] != cub_matrix[1])):
                dict_args = {}
                dict_args['errCode'] = "E64009"
                error_manager_util.raise_runtime_error(dict_args)
            # blockIdx must be positive int
            dim_i = 0
            for dim_x in block_cout:
                if dim_x < 1:
                    dict_args = {}
                    dict_args["errCode"] = "E64004"
                    dict_args["param_name"] = "tiling.block_dim"
                    dict_args["axis_rule"] = "positive int"
                    dict_args["wrong_axis"] = str(dim_i)
                    dict_args["actual_value"] = str(dim_x)
                    error_manager_util.raise_runtime_error(dict_args)
                dim_i = dim_i + 1

            def _gen_dict_args(name, value):
                dict_args = {}
                dict_args["errCode"] = "E64010"
                dict_args["buffer_name"] = name
                dict_args["value"] = str(value)
                return dict_args
            # only support no dbuffer/ dbuffer
            if al1_pbuff not in (1, 2):
                dict_args = _gen_dict_args("AL1_pbuffer", al1_pbuff)
                error_manager_util.raise_runtime_error(dict_args)
                error_manager_util.raise_runtime_error(dict_args)

            if bl1_pbuff not in (1, 2):
                dict_args = _gen_dict_args("BL1_pbuffer", bl1_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

            if al0_pbuff not in (1, 2):
                dict_args = _gen_dict_args("AL0_pbuffer", al0_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

            if bl0_pbuff not in (1, 2):
                dict_args = _gen_dict_args("BL0_pbuffer", bl0_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

            if l0c_pbuff not in (1, 2):
                dict_args = _gen_dict_args("L0C_pbuffer", l0c_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

            if cub_pbuff not in (1, 2):
                dict_args = _gen_dict_args("CUB_pbuffer", cub_pbuff)
                error_manager_util.raise_runtime_error(dict_args)

        def _atomic_add(sch, res_cc):
            """
            achieve atomic add according to refactor dw_cc

            """

            # redefine dw_ddr, dw_ub, dw_cc to achieve atomic write
            batch, real_k = sch[res_cc].op.reduce_axis
            batch_dim_factor = _ceil_div(batch_fmap, block_dim_batch)
            batch_dim_factor = tvm.max(1, batch_dim_factor)
            if 'batch' in self.var_map:
                batch_core, batch_in = sch[res_cc].split(
                    batch, factor=batch_dim_factor)
            else:
                batch_core, batch_in = sch[res_cc].split(
                    batch, nparts=block_dim_batch)

            if self.var_map and \
                not DynamicConv2dBpFilterParams.flag_all_one_case:
                # for dynamic hw, the reduce axis of res_cc dose not cut k0
                flag_bl1k_less_than_wo = tiling.get('flag_bl1k_less_than_wo')
                hw_single_core_factor = _ceil_div(hw_pad_1 * CUBE_DIM, \
                                                    block_dim_hw)
                hw_single_core_factor = _align(hw_single_core_factor, \
                                                    dw_k * width_grads * CUBE_DIM)
                if not flag_bl1k_less_than_wo:
                    hw_single_core_factor = _ceil_div(_ceil_div(hw_pad_1, dw_k), block_dim_hw)
                    hw_single_core_factor = hw_single_core_factor * dw_k * CUBE_DIM
                k_1_multicore, real_k = sch[res_cc].split(real_k, hw_single_core_factor)
                sch[res_cc].reorder(k_1_multicore, batch_core, batch_in, real_k)
            else:
                real_k, k_in = sch[res_cc].split(real_k, self.c0_size)
                hw_single_core_factor = _ceil_div(hw_pad_1 * self.c0_size, block_dim_hw)
                k_al1 = tiling.get("AL1_shape")[0] if tiling.get("AL1_shape") else dw_k * self.c0_size
                k_bl1 = tiling.get("BL1_shape")[0] if tiling.get("BL1_shape") else dw_k * self.c0_size
                k_one_core_max = max(k_al1, k_bl1, dw_k * self.c0_size)
                hw_single_core_factor = _align(hw_single_core_factor, k_one_core_max) // self.c0_size
                k_1_multicore, real_k = sch[res_cc].split(real_k, hw_single_core_factor)
                sch[res_cc].reorder(k_1_multicore, batch_core, batch_in, real_k, k_in)

            fused_atomic_write = sch[res_cc].fuse(k_1_multicore, batch_core)

            # after rfactor op, dw_cc becomes dw_ddr, original dw_ub and dw_ddr
            # will be dropped
            res_ddr = res_cc
            res_cc = sch.rfactor(res_ddr, fused_atomic_write)
            sch[res_cc].set_scope(tbe_platform_info.scope_cc)
            if self.cube_vector_split:
                res_ub = None
            else:
                res_ub = sch.cache_read(res_cc, tbe_platform_info.scope_ubuf, [res_ddr])
            if dw_trans_flag:
                sch[res_ddr].set_scope(tbe_platform_info.scope_cc)
            return res_cc, res_ub, res_ddr

        def _full_k_check():
            """
            set flag whether axis K is fully loaded in L0A and L0B
            return:
            -------
            full_k_l0a: 1 or 0,
                        1 means K is fully loaded in L0A
            full_k_l0b: 1 or 0,
                        1 means K is fully loaded in L0B
            """

            # if k is fully load in BL1 and
            # there is multi load in N1 and N1 in BL1
            # isn't aligned to kernel_height*kernel_width, then align to it
            if tiling.get("BL1_shape") and \
                    tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1] \
                    % (kernel_height * kernel_width) != 0:
                tiling.get("BL1_shape")[1] = _align(tiling.get("BL1_shape")[1] *
                                                tiling.get("BL0_matrix")[1],
                                                kernel_height * kernel_width) \
                    // tiling.get("BL0_matrix")[1]

            # whether axis K is fully loaded in L0A and L0B
            # excluding axis batch
            if not tiling.get("AL0_matrix"):
                full_k_l0a = 1
            else:
                full_k_l0a = tiling.get("AL0_matrix")[1] \
                    // _ceil_div(hw_pad_1, block_dim_hw)
            if DEBUG_MODE:
                print("full_k_in_l0a", full_k_l0a)

            if not tiling.get("BL0_matrix"):
                full_k_l0b = 1
            else:
                full_k_l0b = tiling.get("BL0_matrix")[0] \
                    // _ceil_div(hw_pad_1, block_dim_hw)
            if DEBUG_MODE:
                print("full_k_in_l0b", full_k_l0b)
            return full_k_l0a, full_k_l0b

        def _compute_tiling_parts():
            """
            compute the parts or the factors of tensors

            """

            # ka and kb may be different,
            # the min value corresponds to one MMAD,
            # the larger one is []
            if tiling.get("AL0_matrix"):  # dw_k equals to ka if L0A needs tiling
                dw_k = tiling.get("AL0_matrix")[1]
            elif tiling.get("BL0_matrix"):
                dw_k = tiling.get("BL0_matrix")[0]
            else:  # both fully loaded
                dw_k = hw_pad_1 // block_dim_hw

            if not tiling.get("AL0_matrix"):  # if grads no tiling in L0A
                tiling["AL1_shape"] = []  # then no tiling in L1

            # dw_cc is (fmap_channel_1*kernel_height*kernel_width,
            #          grads_channel_1, C0_grads, C0_fmap)
            dw_tiling_factor = [tiling.get("CL0_matrix")[0],
                                tiling.get("CL0_matrix")[1]]
            # nparts N, nparts M
            # dw_tiling_nparts only describe the nparts from single core to L0
            dw_tiling_nparts = \
                [_ceil_div(fkk // block_dim_cin, dw_tiling_factor[0]),
                 _ceil_div(_ceil_div(cout_g // CUBE_DIM, dw_tiling_factor[1]),
                           block_dim_cout)]

            # tiling parameters of dw_ub
            dw_ub_tiling_factor = [tiling.get("CUB_matrix")[0],
                                   tiling.get("CUB_matrix")[1]]
            dw_ub_tiling_nparts = [_ceil_div(dw_tiling_factor[0],
                                             dw_ub_tiling_factor[0]),
                                   _ceil_div(dw_tiling_factor[1],
                                             dw_ub_tiling_factor[1])]
            hw_single_core_factor = _ceil_div(hw_pad_1, block_dim_hw)
            if not self.var_map:
                hw_single_core_factor = _ceil_div(hw_pad_1 * self.c0_size, block_dim_hw)
                k_al1 = tiling.get("AL1_shape")[0] if tiling.get("AL1_shape") else dw_k * self.c0_size
                k_bl1 = tiling.get("BL1_shape")[0] if tiling.get("BL1_shape") else dw_k * self.c0_size
                k_one_core_max = max(k_al1, k_bl1, dw_k * self.c0_size)
                hw_single_core_factor = _align(hw_single_core_factor, k_one_core_max) // self.c0_size
            # only support loading one batch to L1 at a time for now
            # cout:out->single core(sc)->L1
            if tiling.get("AL1_shape"):  # if grads needs tiling in L1
                if len(tiling.get("AL1_shape")) == 1:  # but no C_1 tiling info
                    tiling["AL1_shape"] = \
                        tiling.get("AL1_shape") + [1]
                # nparts K1 in L1, nparts M1 in L1

                grads_l1_tiling_nparts = [hw_single_core_factor //
                                          (tiling.get("AL1_shape")[0] // self.c0_size),
                                          dw_tiling_nparts[1]
                                          // tiling.get("AL1_shape")[1]]
            else:
                grads_l1_tiling_nparts = [1, 1]

            if tiling.get("BL1_shape"):  # if fmap needs tiling in L1
                if len(tiling.get("BL1_shape")) == 1:  # but no fkk tiling info
                    tiling["BL1_shape"] = \
                        tiling.get("BL1_shape") + [1]  # tiling fkk=1
                # DDR to L1 [nparts K1, nparts N1]
                fmap_l1_tiling_nparts = [hw_single_core_factor //
                                         (tiling.get("BL1_shape")[0] // self.c0_size),
                                         dw_tiling_nparts[0]
                                         // tiling.get("BL1_shape")[1]]
            else:
                fmap_l1_tiling_nparts = [1, 1]

            # during L1 to L0 [nparts N1, nparts M1]
            l1_2_l0_tiling_nparts = \
                [dw_tiling_nparts[0] // fmap_l1_tiling_nparts[1],
                 dw_tiling_nparts[1] // grads_l1_tiling_nparts[1]]


            if DEBUG_MODE:
                print("dw_tiling_factor", dw_tiling_factor)
                print("dw_tiling_nparts", dw_tiling_nparts)
                print("dw_ub_tiling_factor", dw_ub_tiling_factor)
                print("dw_ub_tiling_nparts", dw_ub_tiling_nparts)
                print("grads_l1_tiling_nparts", grads_l1_tiling_nparts)
                print("fmap_l1_tiling_nparts", fmap_l1_tiling_nparts)
                print("l1_2_l0_tiling_nparts", l1_2_l0_tiling_nparts)
                print("dw_k", dw_k)

            tiling_parts_dict = {}
            tiling_parts_dict["dw_tiling_factor"] = dw_tiling_factor
            tiling_parts_dict["dw_tiling_nparts"] = dw_tiling_nparts
            tiling_parts_dict["dw_ub_tiling_factor"] = dw_ub_tiling_factor
            tiling_parts_dict["dw_ub_tiling_nparts"] = dw_ub_tiling_nparts
            tiling_parts_dict["grads_l1_tiling_nparts"] = \
                                                    grads_l1_tiling_nparts
            tiling_parts_dict["fmap_l1_tiling_nparts"] = fmap_l1_tiling_nparts
            tiling_parts_dict["l1_2_l0_tiling_nparts"] = l1_2_l0_tiling_nparts
            tiling_parts_dict["dw_k"] = dw_k
            return tiling_parts_dict

        def _compute_tiling_factors():
            fmap_l1_tiling_factor_k, grads_l1_tiling_factor_k = None, None
            if self.var_map and not flag_all_one_case:
                if reduce_split_mode:
                    if tiling.get("AL1_shape"):
                        grads_l1_tiling_factor_k = \
                            tiling.get("AL1_shape")[0] // (dw_k * CUBE_DIM)
                    if tiling.get("BL1_shape") and tiling.get("AL1_shape"):
                        fmap_l1_tiling_factor_k = \
                            tiling.get("BL1_shape")[0] // tiling.get("AL1_shape")[0]
                else:
                    if tiling.get("BL1_shape"):
                        fmap_l1_tiling_factor_k = \
                            tiling.get("BL1_shape")[0] // (dw_k * CUBE_DIM)
                    if tiling.get("BL1_shape") and tiling.get("AL1_shape"):
                        grads_l1_tiling_factor_k = \
                            tiling.get("AL1_shape")[0] // tiling.get("BL1_shape")[0]
            return grads_l1_tiling_factor_k, fmap_l1_tiling_factor_k

        def _reduce_split_mode():
            reduce_split_mode = True
            if self.var_map:
                if tiling.get("AL1_shape") and tiling.get("BL1_shape"):
                    # grads and fmap need tiling in L1
                    reduce_split_mode = \
                              tiling.get("AL1_shape")[0] < tiling.get("BL1_shape")[0]
                elif tiling.get("AL1_shape"):
                    # only grads needs tiling in L1
                    reduce_split_mode = True
                elif tiling.get("BL1_shape"):
                    # only fmap needs tiling in L1
                    reduce_split_mode = False
                else:
                    # Neither grads nor fmap need tiling in L1
                    reduce_split_mode = False
            else:
                reduce_split_mode = \
                          grads_l1_tiling_nparts[0] > fmap_l1_tiling_nparts[0]
            return reduce_split_mode

        def _l0_attach():
            """
            achieve Al0 and Bl0 compute at loc or ddr

            """
            if self.var_map and not flag_all_one_case:
                l0a_attach_mode = (dynamic_l0a_attach == "dw_ddr")
                l0b_attach_mode = (dynamic_l0b_attach == "dw_ddr")
            else:
                l0a_attach_mode = \
                         ((batch_num_sc == 1) and (full_k_in_l0a == 1))
                l0b_attach_mode = \
                         ((batch_num_sc == 1) and (full_k_in_l0b == 1))

            if tiling.get("AL0_matrix"):
                if l0a_attach_mode:
                    # L0A data is more than that L0C needed, attach to dw_ddr
                    sch[grads_fractal].compute_at(sch[dw_ddr], c_grads_mad_at)
                    l0a_attach_scope = dw_ddr
                    l0a_attach_axis = c_grads_mad_at
                else:
                    sch[grads_fractal].compute_at(sch[dw_cc], hw_mad_1_mad_at)
                    l0a_attach_scope = dw_cc
                    l0a_attach_axis = hw_mad_1_mad_at
            else:  # else: fully load, attach to thread_axis
                sch[grads_fractal].compute_at(sch[dw_ddr], fused_multi_core)
                l0a_attach_scope = dw_ddr
                l0a_attach_axis = fused_multi_core

            if tiling.get("BL0_matrix"):
                if l0b_attach_mode:
                    sch[fmap_fractal].compute_at(sch[dw_ddr], c_fmap_mad_at)
                    l0b_attach_scope = dw_ddr
                    l0b_attach_axis = c_fmap_mad_at
                else:
                    sch[fmap_fractal].compute_at(sch[dw_cc], hw_mad_1_mad_at)
                    l0b_attach_scope = dw_cc
                    l0b_attach_axis = hw_mad_1_mad_at
            else:  # else: fully load, attach to thread_axis
                sch[fmap_fractal].compute_at(sch[dw_ddr], fused_multi_core)
                l0b_attach_scope = dw_ddr
                l0b_attach_axis = fused_multi_core
            return [l0a_attach_scope, l0a_attach_axis, l0b_attach_scope, l0b_attach_axis]

        def _al1_attach():
            """
            achieve Al1 compute at l0c or ddr

            """
            if self.var_map and not flag_all_one_case:
                al1_attach_mode = (dynamic_al1_attach == "dw_cc")
            else:
                al1_attach_mode = \
                        (grads_l1_tiling_nparts[0] != 1 or batch_num_sc != 1)
            if reorder_l1_mn:
                run_once_n_dim = [c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at] + c_fmap_mad_at_list
            else:
                run_once_n_dim = c_fmap_mad_at_list
            del_n0_outer_flag = l0a_attach_axis == c_grads_mad_at and reorder_flag

            def _grad_matrix_attach(run_once_n_dim):
                if tiling.get("AL1_shape"):
                    # if axis K needs split, then attach to dw_cc
                    if al1_attach_mode:
                        al1_attach_axis = al1_at_axis
                        al1_attach_scope = dw_cc
                        if tiling.get("A_overhead_opt_flag"):
                            sch[grads_matrix].allocate_at(sch[al1_attach_scope], al1_at_axis)
                            al1_attach_axis = l0a_attach_axis
                    else:  # if axis K fully load in L1, attach to dw_ddr
                        al1_attach_axis = c_grads_l1_at
                        al1_attach_scope = dw_ddr
                        if tiling.get("A_overhead_opt_flag"):
                            # the list of axis is c_grads_mad_at, c_fmap_mad_at
                            run_once_n_dim_tmp = list(set(run_once_n_dim) - set(c_fmap_mad_at_list)) \
                                if del_n0_outer_flag else run_once_n_dim
                            sch[grads_matrix].allocate_at(sch[dw_ddr], c_grads_l1_at, run_once_axes=run_once_n_dim_tmp)
                            al1_attach_scope = l0a_attach_scope
                            al1_attach_axis = l0a_attach_axis
                else:  # else: fully load, attach to thread_axis
                    al1_attach_axis = fused_multi_core
                    al1_attach_scope = dw_ddr
                    if tiling.get("A_overhead_opt_flag") and tiling.get("AL0_matrix"):
                        if del_n0_outer_flag:
                            # the list of axis is c_grads_mad_at, c_fmap_mad_at
                            run_once_n_dim = list(set(run_once_n_dim) - set(c_fmap_mad_at_list))
                        sch[grads_matrix].allocate_at(sch[dw_ddr], fused_multi_core, run_once_axes=run_once_n_dim)
                        al1_attach_scope = l0a_attach_scope
                        al1_attach_axis = l0a_attach_axis
                sch[grads_matrix].compute_at(sch[al1_attach_scope], al1_attach_axis)
                if grads_trans_flag:
                    sch[grads].compute_at(sch[al1_attach_scope], al1_attach_axis)
            _grad_matrix_attach(run_once_n_dim)

        def _bl1_attach():
            """
            achieve Bl1 compute at l0c or ddr

            """

            fmap_matrix_flag = not self.var_map or flag_all_one_case
            if self.var_map and not flag_all_one_case:
                bl1_attach_mode = (dynamic_bl1_attach == "dw_cc")
            else:
                bl1_attach_mode = \
                         (fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1)
            run_once_mdim = [c_grads_mad_at, ] if reorder_l1_mn else [c_grads_mad_at, c_grads_l1_at]

            def _fmap_l1_attach(run_once_mdim):
                if tiling.get("BL1_shape"):
                    # if axis K needs split, then attach to dw_cc
                    if bl1_attach_mode:
                        bl1_attach_axis = bl1_at_axis
                        bl1_attach_scope = dw_cc
                        if not flag_all_one_case:
                            sch[fmap_l1].compute_at(sch[bl1_attach_scope], bl1_attach_axis)
                    else:  # if axis K fully load in L1, attach to dw_ddr
                        bl1_attach_axis = c_fmap_l1_at
                        bl1_attach_scope = dw_ddr
                        if not flag_all_one_case:
                            if tiling.get("B_overhead_opt_flag"):
                                if not reorder_flag:
                                    # the list of axis is c_fmap_mad_at, c_grads_mad_at
                                    run_once_mdim = list(set(run_once_mdim) - {c_grads_mad_at})
                                sch[fmap_l1].allocate_at(sch[dw_ddr], c_fmap_l1_at,
                                                         run_once_axes=run_once_mdim + run_once_ndim)
                                bl1_attach_scope = dw_ddr
                                bl1_attach_axis = c_fmap_mad_at
                            if self.l0b_dma_flag:
                                bl1_attach_axis = c_fmap_mad_at
                            sch[fmap_l1].compute_at(sch[bl1_attach_scope], bl1_attach_axis)
                else:  # else: fully load, attach to thread_axis
                    bl1_attach_axis = fused_multi_core
                    bl1_attach_scope = dw_ddr
                    if not flag_all_one_case:
                        if tiling.get("B_overhead_opt_flag") and tiling.get("AL0_matrix"):
                            if not reorder_flag:
                                # the list of axis is c_fmap_mad_at, c_grads_mad_at
                                run_once_mdim = list(set(run_once_mdim) - {c_grads_mad_at, })
                            sch[fmap_l1].allocate_at(sch[dw_ddr], fused_multi_core,
                                                     run_once_axes=run_once_mdim + run_once_ndim)
                            bl1_attach_scope = dw_ddr
                            bl1_attach_axis = c_fmap_mad_at
                        sch[fmap_l1].compute_at(sch[bl1_attach_scope], bl1_attach_axis)
                return bl1_attach_scope, bl1_attach_axis
            bl1_attach_scope, bl1_attach_axis = _fmap_l1_attach(run_once_mdim)
            if fmap_matrix_flag and not self.l0b_dma_flag:
                sch[fmap_matrix].compute_at(sch[bl1_attach_scope], bl1_attach_axis)
            # fmap_l1 axes:
            #   group, batch, hw, fkk(cin_1*hw*wk), mad->16, cin_0->16
            if fmap_ub is not None:
                sch[fmap_ub].compute_at(sch[fmap_l1], sch[fmap_l1].op.axis[4])

        def _double_buffer():
            """
            achieve double_buffer

            """
            if not DEBUG_DOUBLE_BUFFER_OFF:
                if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[grads_matrix].double_buffer()
                    if grads_trans_flag:
                        sch[grads].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    if not flag_all_one_case:
                        sch[fmap_l1].double_buffer()
                    else:
                        sch[fmap_matrix].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[grads_fractal].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[fmap_fractal].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[dw_cc].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("CUB_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    if not self.cube_vector_split:
                        sch[dw_ub].double_buffer()

        def _emit_insn():
            """
            achieve emit_insn

            """
            setfmatrix_dict = {}
            setfmatrix_dict["conv_kernel_h"] = kernel_height
            setfmatrix_dict["conv_kernel_w"] = kernel_width
            setfmatrix_dict["conv_padding_top"] = pad_up
            setfmatrix_dict["conv_padding_bottom"] = pad_down
            setfmatrix_dict["conv_padding_left"] = pad_left
            setfmatrix_dict["conv_padding_right"] = pad_right
            setfmatrix_dict["conv_stride_h"] = stride_height
            setfmatrix_dict["conv_stride_w"] = stride_width
            setfmatrix_dict["conv_fm_c"] = featuremap_channel
            setfmatrix_dict["conv_fm_h"] = featuremap_height
            setfmatrix_dict["conv_fm_w"] = featuremap_width
            mad_dict = {"mad_pattern": 3,
                        "k_outer":
                            [batch_insn_o, hw_mad_1_l1_out_at,
                            hw_mad_1_l1_in_at, hw_mad_1_mad_at]}

            if self.var_map and not flag_all_one_case:
                setfmatrix_dict["set_fmatrix"] = 1
                setfmatrix_dict["conv_fm_c1"] = c1_fmap
                setfmatrix_dict["conv_fm_c0"] = c0_fmap
            else:
                setfmatrix_dict["conv_dilation_h"] = dilation_height
                setfmatrix_dict["conv_dilation_w"] = dilation_width
                mad_dict["mad_pattern"] = 2

            if self.cube_vector_split and _get_precision_mode("conv2d_backprop_filter") == "high_performance":
                mad_dict["hf32"] = 1

            setfmatrix_dict_0 = {}
            setfmatrix_dict_0["conv_kernel_h"] = kernel_height
            setfmatrix_dict_0["conv_kernel_w"] = kernel_width
            setfmatrix_dict_0["conv_padding_top"] = pad_up
            setfmatrix_dict_0["conv_padding_bottom"] = pad_down
            setfmatrix_dict_0["conv_padding_left"] = pad_left
            setfmatrix_dict_0["conv_padding_right"] = pad_right
            setfmatrix_dict_0["conv_stride_h"] = stride_height
            setfmatrix_dict_0["conv_stride_w"] = stride_width
            setfmatrix_dict_0["conv_fm_c"] = featuremap_channel
            setfmatrix_dict_0["conv_fm_h"] = featuremap_height
            setfmatrix_dict_0["conv_fm_w"] = featuremap_width
            setfmatrix_dict_0["group_flag"] = 1
            setfmatrix_dict_0["l1_group_flag"] = 1

            if self.var_map and not flag_all_one_case:
                setfmatrix_dict_0["set_fmatrix"] = 0
                setfmatrix_dict_0["conv_fm_c1"] = c1_fmap
                setfmatrix_dict_0["conv_fm_c0"] = c0_fmap
            else:
                setfmatrix_dict_0["conv_dilation_h"] = dilation_height
                setfmatrix_dict_0["conv_dilation_w"] = dilation_width

            # move grads from ddr to L1
            # when load3d special case, emit insn after H to avoid floor_div IR
            grads_ub2l1_idx = 3 if flag_w_one_case else 0
            if grads_trans_flag:
                # grads_matrix axes(fp32): batch, mad_1, cout_1, mad_0->16, cout_0->8
                sch[grads].compute_inline()
                sch[grads_matrix].emit_insn(grads_matrix.op.axis[0], 'dma_copy')
            else:
                sch[grads_matrix].emit_insn(grads_matrix.op.axis[grads_ub2l1_idx], 'dma_copy')
            # move grads from L1 to L0A
            grads_fractal_emit_axis = grads_fractal.op.axis[0]
            if in_dtype == "float32":
                # grads_fractal axes: group, batch, cout_1, mad_1, cout_0->16, mad_0->8
                # split the axes to match dma pattern for emit_insn: 2, 2, 8, 8
                sch[grads_fractal].split(grads_fractal.op.axis[-2], factor=self.c0_size)
                _, grads_fractal_inner = sch[grads_fractal].split(grads_fractal.op.axis[-3], factor=2)
                grads_fractal_emit_axis = grads_fractal_inner
            sch[grads_fractal].emit_insn(grads_fractal_emit_axis, 'dma_copy')

            # move fmap from ddr to L1
            if not flag_all_one_case:
                if self.var_map:
                    sch[fmap_l1].emit_insn(fmap_l1.op.axis[0],
                                                'dma_copy', setfmatrix_dict)
                    sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[1],
                                                'im2col_v2', setfmatrix_dict_0)
                else:
                    if self.l0b_dma_flag:
                        if fmap_ub is not None:
                            # fmap_ub axes:
                            #   batch, cin_1, fmap_h, fmap_w, cin_0->16
                            # process 16 data at one time
                            sch[fmap_ub].emit_insn(fmap_ub.op.axis[4], 'dma_copy')
                        # fmap_l1 axes:
                        #   group, batch, hw_mad_1, fkk(cin_1*hk*wk), mad_0->16, cin_0->16
                        sch[fmap_l1].emit_insn(fmap_l1.op.axis[5], 'dma_copy')
                        sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], 'dma_copy')
                    else:
                        if fmap_trans_flag:
                            sch[fmap_l1].emit_insn(fmap_l1.op.axis[1],
                                                   'dma_copy', {"layout_transform": "nd2nz"})
                        else:
                            sch[fmap_l1].emit_insn(fmap_l1.op.axis[0], 'dma_copy')
                        sch[fmap_matrix].emit_insn(fmap_matrix.op.axis[0],
                                                   'set_fmatrix', setfmatrix_dict)
                        if in_dtype == "float32":
                            # fmap_fractal axes:
                            #       group, batch, hw_mad_1, fkk(cin_1*hk*wk), cin_0->16, mad_0->8
                            # split axes to match im2col pattern, make sure the last 2 axes is 8*8
                            # then reorder axes so that the sequence is as below:
                            #       group, 2, [emit here] batch, hw_mad_1, fkk, cin_0_s->8, mad_0->8
                            cin_outer, cin_inner = sch[fmap_fractal].split(fmap_fractal.op.axis[-2],
                                                                           factor=self.c0_size)
                            fmap_fractal_axes = fmap_fractal.op.axis[1:]
                            fmap_fractal_axes[-2] = cin_inner
                            sch[fmap_fractal].reorder(fmap_fractal.op.axis[0], cin_outer, *fmap_fractal_axes)
                        sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[1], 'im2col')
            else:
                if fmap_trans_flag:
                    sch[fmap].emit_insn(fmap.op.axis[1], 'dma_copy', {"layout_transform": "nd2nz"})
                    sch[fmap_matrix].emit_insn(fmap_matrix.op.axis[0], 'phony_insn')
                else:
                    sch[fmap_matrix].emit_insn(fmap_matrix.op.axis[0], 'dma_copy')
                # fmap_fractal emit_insn
                if in_dtype == "float32":
                    sch[fmap_fractal].split(fmap_fractal.op.axis[-2], factor=self.c0_size)
                sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0],
                                            'dma_copy')

            # move dw from L0C to UB
            if not self.cube_vector_split:
                sch[dw_ub].emit_insn(dw_ub.op.axis[0], 'dma_copy')

            sch[dw_cc].emit_insn(batch_insn, 'mad', mad_dict)

            # manage fixpipe memory
            for fixpipe_l1_mem in self.tensor_map.get("fixpipe_l1", []):
                sch[fixpipe_l1_mem].emit_insn(fixpipe_l1_mem.op.axis[0], "dma_copy")
            if self.tensor_map.get("fixpipe_l1_eltwise") is not None:
                fixpipe_l1_eltwise = self.tensor_map.get("fixpipe_l1_eltwise")
                sch[fixpipe_l1_eltwise].emit_insn(fixpipe_l1_eltwise.op.axis[0], "dma_copy")
            for fixpipe_fb_mem in self.tensor_map.get("fixpipe_fb", {}).values():
                sch[fixpipe_fb_mem].emit_insn(fixpipe_fb_mem.op.axis[0], "dma_copy")

            # move dw from UB to ddr
            if dw_trans_flag:
                sch[dw_ddr].emit_insn(c_grads_mad_insn, 'dma_copy', {"layout_transform": "nz2nd"})
            else:
                sch[dw_ddr].emit_insn(c_fmap_2_ub_insn, 'dma_copy')
            sch_list.append(dw_ddr)


        def _get_value(ele):
            res_ele = [ele.value if isinstance(ele, tvm.expr.IntImm) else \
                                                                    ele][0]
            return res_ele

        def _get_load3d_para():
            # load_3d parameters
            if self.var_map and not \
                                DynamicConv2dBpFilterParams.flag_all_one_case:
                stride_height, stride_width, pad_up, pad_down, pad_left, \
                pad_right, kernel_height, kernel_width, dilation_height, \
                dilation_width = (
                    _get_value(fmap_fractal.op.attrs['stride'][0]),
                    _get_value(fmap_fractal.op.attrs['stride'][1]),
                    _get_value(fmap_fractal.op.attrs['pad'][0]),
                    _get_value(fmap_fractal.op.attrs['pad'][1]),
                    _get_value(fmap_fractal.op.attrs['pad'][2]),
                    _get_value(fmap_fractal.op.attrs['pad'][3]),
                    _get_value(fmap_fractal.op.attrs['kernel_size'][2]),
                    _get_value(fmap_fractal.op.attrs['kernel_size'][3]),
                    _get_value(fmap_fractal.op.attrs['dilation'][2]),
                    _get_value(fmap_fractal.op.attrs['dilation'][3])
                    )
            else:
                stride_height, stride_width, pad_up, pad_down, pad_left, \
                pad_right, kernel_height, kernel_width, dilation_height, \
                dilation_width = (
                    fmap_matrix.op.attrs['stride'][0].value,
                    fmap_matrix.op.attrs['stride'][1].value,
                    fmap_matrix.op.attrs['pad'][0].value,
                    fmap_matrix.op.attrs['pad'][1].value,
                    fmap_matrix.op.attrs['pad'][2].value,
                    fmap_matrix.op.attrs['pad'][3].value,
                    fmap_matrix.op.attrs['kernel_size'][2].value,
                    fmap_matrix.op.attrs['kernel_size'][3].value,
                    fmap_matrix.op.attrs['dilation'][2].value,
                    fmap_matrix.op.attrs['dilation'][3].value
                    )
            return stride_height, stride_width, pad_up, pad_down, pad_left, \
                   pad_right, kernel_height, kernel_width, dilation_height, \
                   dilation_width

        def _set_var_range():
            if self.var_map:
                var_range = dynamic_para.get("var_range")
            if 'batch' in self.var_map:
                sch.set_var_range(batch_fmap, *var_range.get('batch'))
            if 'fmap_h' in self.var_map:
                sch.set_var_range(height_fmap, *var_range.get('fmap_h'))
                sch.set_var_range(height_grads, *var_range.get('dedy_h'))
            if 'fmap_w' in self.var_map:
                sch.set_var_range(width_fmap, *var_range.get('fmap_w'))
                sch.set_var_range(width_grads, *var_range.get('dedy_w'))

        def _get_tiling():
            if not self.var_map:
                info_dict = {
                    "op_type": "conv2d_backprop_filter",
                    "A_shape": list(grads_shape),
                    "B_shape": list(fmap_shape),
                    "C_shape": list(weight_shape),
                    "A_dtype": str(grads.dtype),
                    "B_dtype": str(fmap.dtype),
                    "C_dtype": str(dw_cc.dtype),
                    "mad_dtype": str(dw_cc.dtype),
                    "padl": pad_left,
                    "padr": pad_right,
                    "padu": pad_up,
                    "padd": pad_down,
                    "strideH": stride_height,
                    "strideW": stride_width,
                    "strideH_expand": 1,
                    "strideW_expand": 1,
                    "dilationH": dilation_height,
                    "dilationW": dilation_width,
                    "group": real_g,
                    "bias_flag": 0,
                    "fused_double_operand_num": 0,
                    "fusion_type": 0,
                    "kernel_name": kernel_name.value,
                }
                tiling = tiling_api.get_tiling(info_dict)
            else:
                tiling = dynamic_para.get('tiling')
            # for float32 case, k dim in L0 should be even
            # so that we can combine 2*8 to get a 16 length axis
            # This block should be deleted later after tiling support this
            if in_dtype == "float32":
                tiling["AL0_matrix"][1] = 2
                tiling["BL0_matrix"][0] = 2
                # limitation by load3d instruction, n dim must be 1 in fp32
                # otherwise we cannot combine 2 C0=8 in k dim
                tiling["BL0_matrix"][1] = 1
                tiling["CL0_matrix"][0] = 1
            return tiling


        def _handle_tbe_compile_para():
            tbe_compile_para = tiling.get("tbe_compile_para")
            sch.tbe_compile_para, preload = parse_tbe_compile_para(tbe_compile_para)
            if preload:
                if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == 2:
                    sch[dw_cc].preload()


        def _get_attach_flag():
            dynamic_l0a_attach = None
            dynamic_l0b_attach = None
            dynamic_al1_attach = None
            dynamic_bl1_attach = None
            if self.var_map and \
                not DynamicConv2dBpFilterParams.flag_all_one_case:
                dynamic_l0a_attach = dynamic_para.get('dynamic_l0a_attach')
                dynamic_l0b_attach = dynamic_para.get('dynamic_l0b_attach')
                dynamic_al1_attach = dynamic_para.get('dynamic_al1_attach')
                dynamic_bl1_attach = dynamic_para.get('dynamic_bl1_attach')
            return dynamic_l0a_attach, dynamic_l0b_attach, \
                   dynamic_al1_attach, dynamic_bl1_attach

        # for dynamic
        self.var_map = DynamicConv2dBpFilterParams.var_map

        # ####################### get computing graph #######################
        # support NZ2ND transdata
        # dw_ddr -> dw_res_trans
        sch = sch_list[0]
        dw_res_trans = None
        dw_trans_flag = False
        fmap_trans_flag = False
        grads_trans_flag = False
        if res.op.tag == "FZ_trans_NHWC":
            dw_trans_flag = True
            dw_res_trans = res
            dw_ddr = res.op.input_tensors[0]
        elif res.op.tag == "fixpipe_reform":
            dw_trans_flag = True
            dw_res_trans = res
            dw_ddr = res
        else:
            dw_ddr = res

        # support channel split
        # dw_cc -> dw_ddr
        c_split_flag = False
        if dw_ddr.op.tag == "conv2d_backprop_filter_c_split":
            c_split_flag = True
            dw_cc = dw_ddr.op.input_tensors[0]
        elif dw_ddr.op.tag == "fixpipe_reform":
            fixpipe_tensor = dw_ddr.op.input_tensors[0]
            dw_cc = fixpipe_tensor.op.input_tensors[0]

            vector_params = fixpipe_tensor.op.attrs["vector_params"]
            vector_tensors = fixpipe_tensor.op.attrs["vector_tensors"]
            fixpipe_fb_dict = {}
            fixpipe_l1_list = []

            for idx, params_mem in enumerate(vector_params):
                fixpipe_input = vector_tensors[idx]
                fixpipe_input_l1 = sch.cache_read(fixpipe_input, tbe_platform_info.scope_cbuf, [fixpipe_tensor])
                fixpipe_scope_name = FIXPIPE_SCOPE_MAP.get(params_mem.value)
                if fixpipe_scope_name:
                    fixpipe_input_fb = sch.cache_read(fixpipe_input_l1, fixpipe_scope_name, [fixpipe_tensor])
                    fixpipe_l1_list.append(fixpipe_input_l1)
                    fixpipe_fb_dict[fixpipe_scope_name] = fixpipe_input_fb
                else:
                    self.tensor_map["fixpipe_l1_eltwise"] = fixpipe_input_l1
            sch[fixpipe_tensor].compute_inline()
            self.tensor_map["fixpipe_fb"] = fixpipe_fb_dict
            self.tensor_map["fixpipe_l1"] = fixpipe_l1_list
        else:
            dw_cc = dw_ddr

        grads_fractal = dw_cc.op.input_tensors[0]
        grads_matrix = grads_fractal.op.input_tensors[0]
        grads = grads_matrix.op.input_tensors[0]
        fmap_fractal = dw_cc.op.input_tensors[1]
        if fmap_fractal.op.tag == "fmap_2_fractal_dma":
            fmap_fractal_before = fmap_fractal.op.input_tensors[0]
            fmap_matrix = fmap_fractal_before.op.input_tensors[0]
            self.l0b_dma_flag = True
        else:
            fmap_matrix = fmap_fractal.op.input_tensors[0]
        fmap = fmap_matrix.op.input_tensors[0]
        fmap_ub = None
        if fmap.op.tag == "fmap_ub_for_dma":
            fmap_ub = fmap_matrix.op.input_tensors[0]
            fmap = fmap_ub.op.input_tensors[0]

        if fmap.op.tag == "NHWC_trans_5HD":
            fmap_trans_flag = True
        if grads.op.tag == "NHWC_trans_5HD":
            grads_trans_flag = True

        group_dict = fmap_matrix.op.attrs['group_dict']
        kernel_name = dw_ddr.op.attrs["kernel_name"]

        # fmap_dtype is same as outbackprop, use fmap_dtype to get C0 size
        in_dtype = fmap.dtype.lower()
        if in_dtype == "float32":
            # mac[1] meas k axis
            self.c0_size = tbe_platform.CUBE_MKN.get(in_dtype).get("mac")[1]

        # ########################extract parameters##########################

        def _get_default_tiling():

            def _get_factors(val, val_max):
                """
                get the factor of val that smaller than val_max
                """
                factor_max = min(val, val_max)
                for m_fac in range(factor_max, 0, -1):
                    if val % m_fac == 0:
                        return m_fac
                return 1

            def _get_nbl0():
                """
                nbl0 can not too large to over space.
                nbl0 can be 1, kernel_width, kernel_width * kernel_height
                """
                if kernel_width * kernel_height <= self._lob_size // KB_2_B:
                    nbl0 = kernel_width * kernel_height
                elif kernel_width <= self._lob_size // KB_2_B:
                    return kernel_width
                else:
                    nbl0 = 1
                return nbl0

            def _get_kbl1():
                """
                give the max kbl1 that fmap load kernel_h
                or kernel_h+stride_h in the hi direction in L1
                """
                if width_grads < CUBE_DIM:
                    return CUBE_DIM
                if width_grads % CUBE_DIM == 0:
                    return width_grads

                kbl1_before = width_grads // CUBE_DIM
                # if kbl1_before is not factor of K, it needs to recalculate
                c_k1 = _ceil_div(hw_pad_1, CUBE_DIM)
                k_npart = _ceil_div(c_k1, kbl1_before)
                kbl1 = c_k1 // k_npart * CUBE_DIM
                return kbl1

            def _cal_bl1size():
                """
                calculate bl1_size, if kbl1 is a factor of Wo,
                it must loads Hk in H direction
                else it must loads Hk+stride_h in H direction
                """
                if width_grads % kbl1 == 0:
                    phol1 = 1
                else:
                    phol1 = 2
                pbl1hi = (phol1 - 1) * stride_height + (kernel_height - 1) * dilation_height + 1
                bl1_size = pbl1hi * width_fmap * CUBE_DIM * BIT_RATIO_DICT.get(in_dtype, 2)
                return bl1_size

            nbl0 = _get_nbl0()
            l0c_mal0_max = self._loc_size // KB_2_B // nbl0
            loa_mal0_max = self._lob_size // KB_2_B
            mal0 = _get_factors(c1_grads, min(l0c_mal0_max, loa_mal0_max))
            kbl1 = _get_kbl1()

            block_batch = _get_factors(batch_grads, self._corenum)
            out_dtype = dw_cc.dtype.lower()
            if nbl0 * mal0 * CUBE_DIM * CUBE_DIM * BIT_RATIO_DICT.get(out_dtype, 4) * OPEN_DOUBLE_BUFFER \
                < self._loc_size:
                cl0_pbuffer = 2
            else:
                cl0_pbuffer = 1
            bl1_size = _cal_bl1size()
            al1_size_double = mal0 * CUBE_MUL_SHAPE * BIT_RATIO_DICT.get(in_dtype, 2) * OPEN_DOUBLE_BUFFER
            if al1_size_double + bl1_size <= self.l1_size:
                bl1_pbuffer = 2
            else:
                bl1_pbuffer = 1

            if self.l0b_dma_flag:
                kbl1 = CUBE_DIM
                mal0 = 1
                nbl0 = 1

            default_tiling = {
                'AUB_shape': None, 'BUB_shape': None,
                'AL1_shape': [CUBE_DIM, 1, 1, 1], 'BL1_shape': [kbl1, 1, 1, 1],
                'AL0_matrix': [mal0, 1, CUBE_DIM, CUBE_DIM, 1],
                'BL0_matrix': [1, nbl0, CUBE_DIM, CUBE_DIM, 1],
                'CL0_matrix': [nbl0, mal0, CUBE_DIM, CUBE_DIM, 1],
                'CUB_matrix': [nbl0, mal0, CUBE_DIM, CUBE_DIM, 1],
                'block_dim': [block_batch, 1, 1, 1],
                'cout_bef_batch_flag': 0,
                'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0,
                'manual_pingpong_buffer': {
                    'AUB_pbuffer': 1, 'BUB_pbuffer': 1,
                    'AL1_pbuffer': 1,
                    'BL1_pbuffer': bl1_pbuffer,
                    'AL0_pbuffer': 2, 'BL0_pbuffer': 2,
                    'CL0_pbuffer': cl0_pbuffer,
                    'CUB_pbuffer': cl0_pbuffer,
                    'UBG_pbuffer': 1}}
            return default_tiling

        cin1_g = group_dict["cin1_g"].value
        cout_g = group_dict["cout_g"].value
        real_g = group_dict["real_g"].value

        batch_grads, c1_grads, height_grads, width_grads, c0_grads \
            = cube_util.shape_to_list(grads.shape)
        grads_shape = [batch_grads, cout_g // c0_grads,
                       height_grads, width_grads, c0_grads]
        batch_fmap, c1_fmap, height_fmap, width_fmap, c0_fmap \
            = cube_util.shape_to_list(fmap.shape)
        fmap_shape = [batch_fmap, cin1_g, height_fmap, width_fmap, c0_fmap]
        if in_dtype == "float32":
            # grads_matrix axes(fp32): batch, mad_1, cout_1, mad_0->16, cout_0->8
            _, grads_matrix_howo_1, \
            grads_matrix_c1, grads_matrix_c16, grads_matrix_c0 = cube_util.shape_to_list(grads_matrix.shape)
            grads_matrix_howo = grads_matrix_howo_1 * grads_matrix_c16
        else:
            _, grads_matrix_c1, \
            grads_matrix_howo, grads_matrix_c0 = cube_util.shape_to_list(grads_matrix.shape)
        _, fkk, _, _ = cube_util.shape_to_list(dw_cc.shape)
        _, _, hw_pad_1, _, _, _ = cube_util.shape_to_list(fmap_fractal.shape)

        stride_height, stride_width, pad_up, pad_down, pad_left, \
        pad_right, kernel_height, kernel_width, dilation_height, \
        dilation_width \
            = _get_load3d_para()

        featuremap_channel = c1_fmap*c0_fmap
        featuremap_height = height_fmap
        featuremap_width = width_fmap
        kw_dilation = (kernel_width - 1) * dilation_width + 1
        weight_shape = [cout_g, cin1_g,
                        kernel_height, kernel_width, c0_fmap]

        _set_var_range()
        dynamic_l0a_attach, dynamic_l0b_attach, dynamic_al1_attach, \
        dynamic_bl1_attach \
            = _get_attach_flag()

        def _flag_all_one():
            # special supporting for a unique case, there are 2 conditions:
            # (1) height & weight of x/output_backprop/filter are all 1
            # (2) strides is [1,1]
            flag_all_one_case = False
            flag_conv1d_case = False
            flag_w_one_case = False

            height_all_one = False
            width_all_one = False
            if stride_height == 1 and height_grads == 1 and height_fmap == 1 \
                    and kernel_height == 1:
                height_all_one = True
            if stride_width == 1 and width_grads == 1 and width_fmap == 1 \
                    and kernel_width == 1:
                width_all_one = True
            if height_all_one and width_all_one:
                flag_all_one_case = True
                if DEBUG_MODE:
                    print("schedule: this is all one case,"
                          " using special branch")
            if height_all_one and not width_all_one:
                flag_conv1d_case = True
            if height_grads != 1 and width_grads == 1:
                flag_w_one_case = True
            if 'fmap_h' in self.var_map or 'fmap_w' in self.var_map:
                dynamic_hw_tiling = dynamic_para.get("tiling")
                flag_conv1d_case = dynamic_hw_tiling.get("flag_conv1d_case")
            return flag_all_one_case, flag_conv1d_case, flag_w_one_case
        flag_all_one_case, flag_conv1d_case, flag_w_one_case = _flag_all_one()
        tiling = _get_tiling()

        # for dynamic_mode w_one_case
        if self.var_map:
            flag_all_one_case = False
            w_one_flag = tiling.get("w_one_flag")
            sch.set_var_value(self.var_map.get("w_one_flag"), w_one_flag)
            sch.set_var_range(self.var_map.get("w_one_flag"), w_one_flag, w_one_flag)
            flag_w_one_case = w_one_flag == 2
            if flag_w_one_case:
                width_grads *= 2
                grads_shape[3] = width_grads

        if DEBUG_MODE:
            print("grads_shape to tiling_query", grads_shape)
            print("fmap_shape to tiling_query", fmap_shape)
            print("weight_shape to tiling_query", weight_shape)
            print("pad to tiling_query", pad_left, pad_right, pad_up, pad_down)
            print("stride to tiling_query", stride_height, stride_width)
            print("dilation to tiling_query", dilation_height, dilation_width)
            print("Conv2dBackpropFilter: returned from auto_tiling", tiling)
            print("kernel_name:", kernel_name)
        _tiling_shape_check()
        _tiling_buffer_check()
        # if no valid tiling found, the flag is as follows
        if tiling.get("AL0_matrix")[2] == DEFAULT_TILING_CASE:
            tiling = _get_default_tiling()

        batch_num = batch_grads

        def _get_block_dim():
            if tiling.get("AUB_shape"):
                block_dim_hw = tiling.get("AUB_shape")[0]
            else:
                block_dim_hw = 1
            block_dim_batch = tiling.get("block_dim")[0]
            if 'batch' in self.var_map:
                block_dim_batch = tvm.min(block_dim_batch, batch_fmap)
            block_dim_cout = tiling.get("block_dim")[2]
            block_dim_cin = tiling.get("block_dim")[1]
            block_dim_group = tiling.get("block_dim")[3]
            return block_dim_hw, block_dim_batch, \
                   block_dim_cout, block_dim_cin, block_dim_group

        block_dim_hw, block_dim_batch, \
        block_dim_cout, block_dim_cin, block_dim_group \
            = _get_block_dim()

        if in_dtype == "float32":
            # in float32 case, k axis length will be reduce by half each repeatition
            mul_align_length = CUBE_MUL_SHAPE // 2
            k_align_length = CUBE_DIM // 2
        else:
            mul_align_length = CUBE_MUL_SHAPE
            k_align_length = CUBE_DIM

        if grads_trans_flag:
            sch[grads].set_scope(tbe_platform_info.scope_cbuf)
        sch[grads_matrix].set_scope(tbe_platform_info.scope_cbuf)
        sch[grads_matrix].storage_align(
            sch[grads_matrix].op.axis[1], mul_align_length, 0)

        sch[grads_fractal].set_scope(tbe_platform_info.scope_ca)
        sch[grads_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, 1),
                                        (1, CUBE_DIM), (1, k_align_length))

        def _load3d_fmap_l1_process():
            # shape info:
            # fmap_shape_original_matrix is (batch_size,
            #                               grads_height*grads_width,
            #                               fmap_channel_1,
            #                               kernel_height,
            #                               kernel_width,
            #                               C0_fmap)
            if not self.var_map:
                if fmap_trans_flag:
                    fmap_l1 = fmap
                    sch[fmap].set_scope(tbe_platform_info.scope_cbuf)
                elif fmap_ub is not None:
                    sch[fmap_ub].set_scope(tbe_platform_info.scope_ubuf)
                    fmap_l1 = sch.cache_read(fmap_ub, tbe_platform_info.scope_cbuf, [fmap_matrix])
                else:
                    fmap_l1 = sch.cache_read(fmap, tbe_platform_info.scope_cbuf, [fmap_matrix])
                if not flag_conv1d_case:
                    sch[fmap_matrix].buffer_align((1, 1),
                                                (width_grads, width_grads),
                                                (1, 1),
                                                (kernel_height, kernel_height),
                                                (kernel_width, kernel_width),
                                                (1, CUBE_DIM))
                else:
                    sch[fmap_matrix].buffer_align((1, 1), (1, 1), (1, 1),
                                              (1, 1),
                                              (kernel_width, kernel_width),
                                              (1, CUBE_DIM))
            else:
                fmap_l1 = fmap_matrix
            return fmap_l1

        if not flag_all_one_case:
            fmap_l1 = _load3d_fmap_l1_process()
        else:
            sch[fmap_matrix].storage_align(
                sch[fmap_matrix].op.axis[1], mul_align_length, 0)

        sch[fmap_matrix].set_scope(tbe_platform_info.scope_cbuf)

        if self.l0b_dma_flag:
            sch[fmap_fractal_before].set_scope(tbe_platform_info.scope_cbuf)
            sch[fmap_fractal_before].buffer_align((1, 1), (1, 1), (1, 1), (1, 1),
                                                  (1, CUBE_DIM), (1, k_align_length))
        sch[fmap_fractal].set_scope(tbe_platform_info.scope_cb)
        sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, 1),
                                       (1, CUBE_DIM), (1, k_align_length))

        full_k_in_l0a, full_k_in_l0b = _full_k_check()

        tiling_parts_dict = _compute_tiling_parts()
        dw_tiling_factor = tiling_parts_dict.get("dw_tiling_factor")
        dw_tiling_nparts = tiling_parts_dict.get("dw_tiling_nparts")
        dw_ub_tiling_factor = tiling_parts_dict.get("dw_ub_tiling_factor")

        grads_l1_tiling_nparts = tiling_parts_dict.get("grads_l1_tiling_nparts")
        fmap_l1_tiling_nparts = tiling_parts_dict.get("fmap_l1_tiling_nparts")

        l1_2_l0_tiling_nparts = tiling_parts_dict.get("l1_2_l0_tiling_nparts")
        dw_k = tiling_parts_dict.get("dw_k")
        reduce_split_mode = _reduce_split_mode()
        grads_l1_tiling_factor_k, fmap_l1_tiling_factor_k = \
                                                    _compute_tiling_factors()

        dw_cc, dw_ub, dw_rfactor = _atomic_add(sch, dw_cc)
        if not c_split_flag:
            # c_split: dw_cc -> dw_rfactor -> dw_ddr
            # not c_split: dw_cc -> dw_ddr(dw_rfactor)
            dw_ddr = dw_rfactor
        # #######################tiling parameters analyze####################
        batch_num_sc = batch_num // block_dim_batch
        if DEBUG_MODE:
            print("start analyzing tiling parameters")
            print("axis K: block_dim_batch", block_dim_batch)
            print("axis K: block_dim_hw", block_dim_hw)
            print("axis N: block_dim_cin", block_dim_cin)
            print("axis M: block_dim_cout", block_dim_cout)
            print("axis G: block_dim_group", block_dim_group)
            print("batch_num_sc", batch_num_sc)

        def _get_n_factor():
            # for N axis, if Hk and Wk needs split, do explict split
            if not flag_all_one_case:
                if tiling.get("BL1_shape"):
                    # n1 in L1
                    nc_cc = tiling.get("CL0_matrix")[0] * tiling.get("BL1_shape")[1]
                else:
                    # BL1 is full load
                    nc_cc = c1_fmap*kernel_width*kernel_height//block_dim_cin
                factor_kw = _ceil_div(kernel_width, nc_cc)
                factor_kh = _ceil_div(kernel_width*kernel_height, nc_cc) // factor_kw
                if DEBUG_MODE:
                    factor_c1 = _ceil_div(c1_fmap*kernel_width*kernel_height //
                                          block_dim_cin, nc_cc) // factor_kw // factor_kh
                    print("N axis split in L1",
                          factor_c1, factor_kh, factor_kw)
            else:
                factor_kw = 1
                factor_kh = 1
            return factor_kw, factor_kh

        factor_kw, factor_kh = _get_n_factor()

        # #############################split axis N##########################
        if c_split_flag:
            sch[dw_rfactor].compute_inline(instant=True)
        if c_split_flag and not dw_trans_flag:
            # dw_shape is (real_g, fmap_channel_1, kernel_height*kernel_width,
            #              grads_channel, C0_fmap)
            g_multicore, g_axis = sch[dw_ddr].split(sch[dw_ddr].op.axis[0],
                                                    nparts=block_dim_group)
            c_fmap_multicore, c_fmap_mad \
                = sch[dw_ddr].split(sch[dw_ddr].op.axis[1], nparts=block_dim_cin)
            # split c_fmap_mad with factor=2 according to EmitInsn Channel_split
            c_fmap_mad_c1, c_fmap_mad_insn = sch[dw_ddr].split(c_fmap_mad, factor=2)
            c_fmap_mad_at = sch[dw_ddr].op.axis[2]
            c_fmap_l1_ori, c_fmap_mad_at \
                = sch[dw_ddr].split(c_fmap_mad_at, nparts=fmap_l1_tiling_nparts[1])
            # split n dim
            c_fmap_l1_out, c_fmap_l1_at = sch[dw_ddr].split(c_fmap_l1_ori, factor_kw)
            c_fmap_l1_c1, c_fmap_l1_kh = sch[dw_ddr].split(c_fmap_l1_out, factor_kh)
            # split axis M, M axis located at 4th axis
            c_grads_mad_at, c_grads_mad_insn \
                = sch[dw_ddr].split(sch[dw_ddr].op.axis[3], dw_tiling_factor[1]*CUBE_DIM)
            c_grads_multicore, c_grads_mad_at \
                = sch[dw_ddr].split(c_grads_mad_at, nparts=block_dim_cout)
            c_grads_l1_at, c_grads_mad_at = \
                sch[dw_ddr].split(c_grads_mad_at, nparts=grads_l1_tiling_nparts[1])
            # reorder according to requirments of mmad EmitInsn
            sch[dw_ddr].reorder(sch[dw_ddr].op.reduce_axis[0],
                                g_multicore,
                                c_grads_multicore, c_fmap_multicore, g_axis,
                                c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at,
                                c_grads_l1_at,
                                c_fmap_mad_at, c_grads_mad_at,
                                c_fmap_mad_c1, c_fmap_mad_insn, c_grads_mad_insn,
                                sch[dw_ddr].op.axis[4])
        elif not dw_trans_flag:
            # dw_shape is (real_g, fmap_channel_1*kernel_height*kernel_width,
            #              grads_channel_1, C0_grads, C0_fmap)
            g_multicore, g_axis = sch[dw_ddr].split(sch[dw_ddr].op.axis[0],
                                                    nparts=block_dim_group)
            c_fmap_multicore, c_fmap_mad_at \
                = sch[dw_ddr].split(sch[dw_ddr].op.axis[1], nparts=block_dim_cin)
            c_fmap_mad_at, c_fmap_mad_insn \
                = sch[dw_ddr].split(c_fmap_mad_at, nparts=dw_tiling_nparts[0])
            c_fmap_l1_ori, c_fmap_mad_at \
                = sch[dw_ddr].split(c_fmap_mad_at, nparts=fmap_l1_tiling_nparts[1])
            # split n dim
            c_fmap_l1_out, c_fmap_l1_at = sch[dw_ddr].split(c_fmap_l1_ori, factor_kw)
            c_fmap_l1_c1, c_fmap_l1_kh = sch[dw_ddr].split(c_fmap_l1_out, factor_kh)
            # split axis M
            c_grads_mad_at, c_grads_mad_insn \
                = sch[dw_ddr].split(sch[dw_ddr].op.axis[2], dw_tiling_factor[1]*CUBE_DIM)
            c_grads_multicore, c_grads_mad_at \
                = sch[dw_ddr].split(c_grads_mad_at, nparts=block_dim_cout)
            c_grads_l1_at, c_grads_mad_at = \
                sch[dw_ddr].split(c_grads_mad_at, nparts=grads_l1_tiling_nparts[1])
            # reorder according to requirments of mmad EmitInsn
            sch[dw_ddr].reorder(sch[dw_ddr].op.reduce_axis[0],
                                g_multicore, c_grads_multicore,
                                c_fmap_multicore, g_axis,
                                c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at,
                                c_grads_l1_at,
                                c_fmap_mad_at, c_grads_mad_at,
                                c_fmap_mad_insn, c_grads_mad_insn, sch[dw_ddr].op.axis[3])
        else:
            sch[dw_ddr].compute_inline(instant=True)
            dw_ddr = dw_res_trans
            ddr_batch, ddr_hw, ddr_c = sch[dw_ddr].op.axis
            # split the tensor axis to get [group, grads_c, hw, fmap_c1, fmap_c0]
            ddr_g, ddr_n = sch[dw_ddr].split(ddr_batch, nparts=real_g)
            ddr_c1, ddr_c0 = sch[dw_ddr].split(ddr_c, factor=16)
            # split multiple core axis
            g_multicore, g_axis = sch[dw_ddr].split(ddr_g, nparts=real_g)
            # split n axis
            c_fmap_multicore, c_fmap_mad_at = sch[dw_ddr].split(ddr_c1, nparts=block_dim_cin)
            c_fmap_mad_at, c_fmap_mad_insn = sch[dw_ddr].split(c_fmap_mad_at, nparts=dw_tiling_nparts[0])
            c_fmap_l1_c1, c_fmap_mad_at = sch[dw_ddr].split(c_fmap_mad_at, nparts=fmap_l1_tiling_nparts[1])
            c_fmap_l1_kh, c_fmap_l1_at = sch[dw_ddr].split(ddr_hw, factor_kw)
            # split m axis
            c_grads_mad_at, c_grads_mad_insn = sch[dw_ddr].split(ddr_n, dw_tiling_factor[1] * CUBE_DIM)
            c_grads_multicore, c_grads_mad_at = sch[dw_ddr].split(c_grads_mad_at, nparts=block_dim_cout)
            c_grads_l1_at, c_grads_mad_at = sch[dw_ddr].split(c_grads_mad_at, nparts=grads_l1_tiling_nparts[1])
            # reorder according to requirments of mmad EmitInsn
            sch[dw_ddr].reorder(sch[dw_ddr].op.reduce_axis[0],
                                g_multicore,
                                c_grads_multicore, c_fmap_multicore, g_axis,
                                c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at,
                                c_grads_l1_at,
                                c_fmap_mad_at, c_grads_mad_at,
                                c_fmap_mad_insn, c_grads_mad_insn, ddr_c0)

        def _allocate_at_split(c_fmap_mad_at):
            run_once_ndim = []
            c_fmap_mad_at_list = [c_fmap_mad_at, ]
            factor_c = _ceil_div(fkk // block_dim_cin, dw_tiling_nparts[0])
            if tiling.get("B_overhead_opt_flag"):
                if kernel_height * kernel_width % factor_c == 0:
                    nbuffer_size = kernel_height * kernel_width // factor_c
                else:
                    nbuffer_size = kernel_height * kernel_width
                if nbuffer_size <= tiling.get("BL1_shape")[1] and tiling.get("BL1_shape")[1] % nbuffer_size == 0:
                    c_fmap_run_once, c_fmap_mad_at = sch[dw_ddr].split(c_fmap_mad_at, nbuffer_size)
                    run_once_ndim = [c_fmap_mad_at, ]
                    c_fmap_mad_at_list = [c_fmap_run_once, c_fmap_mad_at]
            return run_once_ndim, c_fmap_mad_at_list, c_fmap_mad_at
        run_once_ndim, c_fmap_mad_at_list, c_fmap_mad_at = _allocate_at_split(c_fmap_mad_at)

        def _ub_and_cc_attach():
            # optimization by move small loops to outer
            reorder_flag = False
            # during L1 to L0, if M loop is smaller, then move to outer
            if l1_2_l0_tiling_nparts[0] > l1_2_l0_tiling_nparts[1]:
                sch[dw_ddr].reorder(c_grads_mad_at, *c_fmap_mad_at_list)
                reorder_flag = True
            reorder_l1_mn = False
            # during sc to L1, if M loop is smaller, then move to outer
            if fmap_l1_tiling_nparts[1] > grads_l1_tiling_nparts[1]:
                sch[dw_ddr].reorder(c_grads_l1_at,
                                    c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at)
                reorder_l1_mn = True

            if not self.cube_vector_split:
                # dw_ub attach
                # dw_ub split
                c_fmap_2_ub_at, c_fmap_2_ub_insn \
                    = sch[dw_ddr].split(c_fmap_mad_insn, dw_ub_tiling_factor[0])
                # dw_ub attach
                sch[dw_ub].compute_at(sch[dw_ddr], c_fmap_2_ub_at)
            else:
                c_fmap_2_ub_insn = c_fmap_mad_insn

            # dw attach
            if reorder_flag:
                sch[dw_cc].compute_at(sch[dw_ddr], c_fmap_mad_at)
            else:
                sch[dw_cc].compute_at(sch[dw_ddr], c_grads_mad_at)
            return c_fmap_2_ub_insn, reorder_flag, reorder_l1_mn
        c_fmap_2_ub_insn, reorder_flag, reorder_l1_mn = _ub_and_cc_attach()

        def _dw_cc_split():
            # get the 3 reduce axis of dw_cc
            batch_axis_sc, k_1_axis_sc, k_0 = sch[dw_cc].op.reduce_axis
            # dw_k is the part for one MMAD
            hw_mad_1_mad_at, hw_mad_1_mad_insn \
                = sch[dw_cc].split(k_1_axis_sc, dw_k)
            # mad_pattern :2 , the 1st axis should be 1, so do a fake split
            batch_insn_o, batch_insn = sch[dw_cc].split(batch_axis_sc, 1)

            # K of AL1 and BL1 can be different, there are 2 split methods
            # on which one is larger
            if reduce_split_mode:
                hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                    hw_mad_1_mad_at, nparts=grads_l1_tiling_nparts[0])
                hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                    hw_mad_1_l1_at, nparts=fmap_l1_tiling_nparts[0])
                al1_at_axis = hw_mad_1_l1_in_at
                bl1_at_axis = hw_mad_1_l1_out_at
            else:
                hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                    hw_mad_1_mad_at, nparts=fmap_l1_tiling_nparts[0])
                hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                    hw_mad_1_l1_at, nparts=grads_l1_tiling_nparts[0])
                al1_at_axis = hw_mad_1_l1_out_at
                bl1_at_axis = hw_mad_1_l1_in_at

            # split dw_cc.op.axis[0](N1), factor is one MMAD
            fkk_mad_at, fkk_mad_insn \
                = sch[dw_cc].split(sch[dw_cc].op.axis[2], dw_tiling_factor[0])

            # split dw_cc.op.axis[1](M1*M0), factor is one MMAD
            lc_mad_at, lc_mad_insn \
                = sch[dw_cc].split(sch[dw_cc].op.axis[3],
                                dw_tiling_factor[1] * CUBE_DIM)
            sch[dw_cc].reorder(fkk_mad_at, lc_mad_at, sch[dw_cc].op.axis[0],
                               batch_insn_o, hw_mad_1_l1_out_at,
                               hw_mad_1_l1_in_at, hw_mad_1_mad_at,
                               batch_insn, fkk_mad_insn, lc_mad_insn,
                               sch[dw_cc].op.axis[4], hw_mad_1_mad_insn, k_0)
            return al1_at_axis, bl1_at_axis, hw_mad_1_mad_at, batch_insn_o, \
                   hw_mad_1_l1_out_at, hw_mad_1_l1_in_at, batch_insn

        def _dynamic_hw_dw_cc_split():
            # get the 2 reduce axis of dw_cc
            batch_axis_sc, k_1_axis_sc = sch[dw_cc].op.reduce_axis
            # dw_k is the part for one MMAD
            hw_mad_1_mad_at, hw_mad_1_mad_insn \
                = sch[dw_cc].split(k_1_axis_sc, dw_k * CUBE_DIM)
            # mad_pattern :2 , the 1st axis should be 1, so do a fake split
            batch_insn_o, batch_insn = sch[dw_cc].split(batch_axis_sc, 1)
            if reduce_split_mode:
                # the factor of grads_l1 is smaller than fmap_l1
                hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                    hw_mad_1_mad_at, grads_l1_tiling_factor_k)
                if tiling.get("AL1_shape") and tiling.get("BL1_shape"):
                    # grads and fmap need tiling in L1
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, fmap_l1_tiling_factor_k)
                elif tiling.get("AL1_shape"):
                    # only grads needs tiling in L1
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, nparts=fmap_l1_tiling_nparts[0])
                al1_at_axis = hw_mad_1_l1_in_at
                bl1_at_axis = hw_mad_1_l1_out_at
            else:
                # the factor of fmap_l1 is smaller than grads_l1
                if tiling.get("AL1_shape") and tiling.get("BL1_shape"):
                    # grads and fmap need tiling in L1
                    hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                        hw_mad_1_mad_at, fmap_l1_tiling_factor_k)
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, grads_l1_tiling_factor_k)
                elif tiling.get("BL1_shape"):
                    # only fmap needs tiling in L1
                    hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(hw_mad_1_mad_at, fmap_l1_tiling_factor_k)
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, nparts=grads_l1_tiling_nparts[0])
                else:
                    # Neither grads nor fmap need tiling in L1
                    hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                        hw_mad_1_mad_at, nparts=fmap_l1_tiling_nparts[0])
                    hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                        hw_mad_1_l1_at, nparts=grads_l1_tiling_nparts[0])
                al1_at_axis, bl1_at_axis = hw_mad_1_l1_out_at, hw_mad_1_l1_in_at
            # split dw_cc.op.axis[0](N1), factor is one MMAD
            fkk_mad_at, fkk_mad_insn \
                = sch[dw_cc].split(sch[dw_cc].op.axis[2], dw_tiling_factor[0])

            # split dw_cc.op.axis[1](M1*M0), factor is one MMAD
            lc_mad_at, lc_mad_insn = sch[dw_cc].split(sch[dw_cc].op.axis[3], dw_tiling_factor[1] * CUBE_DIM)
            sch[dw_cc].reorder(fkk_mad_at, lc_mad_at, sch[dw_cc].op.axis[0],
                               batch_insn_o, hw_mad_1_l1_out_at,
                               hw_mad_1_l1_in_at, hw_mad_1_mad_at,
                               batch_insn, fkk_mad_insn, lc_mad_insn,
                               sch[dw_cc].op.axis[4], hw_mad_1_mad_insn)
            return al1_at_axis, bl1_at_axis, hw_mad_1_mad_at, batch_insn_o, \
                   hw_mad_1_l1_out_at, hw_mad_1_l1_in_at, batch_insn

        if self.var_map and not flag_all_one_case:
            al1_at_axis, bl1_at_axis, hw_mad_1_mad_at, batch_insn_o, \
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at, batch_insn \
                = _dynamic_hw_dw_cc_split()
        else:
            al1_at_axis, bl1_at_axis, hw_mad_1_mad_at, batch_insn_o, \
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at, batch_insn \
                = _dw_cc_split()

        # #############################multi core#############################
        def _bind_core():
            fused_multi_core = \
                sch[dw_ddr].fuse(sch[dw_ddr].op.reduce_axis[0], g_multicore,
                                 c_grads_multicore, c_fmap_multicore)
            fused_multi_core, pragma_at = \
                sch[dw_ddr].split(fused_multi_core, 1)
            block = tvm.thread_axis("blockIdx.x")
            sch[dw_ddr].bind(fused_multi_core, block)

            blocks = (block_dim_batch * block_dim_cin *
                        block_dim_cout * block_dim_hw * block_dim_group)
            if blocks == block_dim_batch:
                sch[dw_ddr].pragma(pragma_at,
                                'json_info_batchBindOnly')
            return fused_multi_core

        fused_multi_core = _bind_core()

        def _split_w_for_conv1d():
            # the offset according to multicore
            block_div = (block_dim_batch * block_dim_cout * block_dim_cin * block_dim_group)
            hw_block_offset = fused_multi_core // block_div * _ceil_div(hw_pad_1, block_dim_hw) * CUBE_DIM
            if self.var_map:
                bool_dw_cc = dynamic_bl1_attach == "dw_cc"
            else:
                bool_dw_cc = tiling.get("BL1_shape") and (fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1)
            if bool_dw_cc:
                if 'fmap_h' in self.var_map or 'fmap_w' in self.var_map:
                    bl1_k = tiling.get("BL1_shape")[0]
                    al1_k = grads_matrix_howo
                    if tiling.get("AL1_shape"):
                        al1_k = tiling.get("AL1_shape")[0]
                    if bl1_at_axis == hw_mad_1_l1_in_at:
                        # hw splited two times before BL1 attach
                        hw_parts_offset = (hw_mad_1_l1_out_at.var * al1_k +
                                        hw_mad_1_l1_in_at.var * bl1_k)
                    else:
                        # hw splited one time before BL1  attach
                        hw_parts_offset = hw_mad_1_l1_out_at.var * bl1_k
                else:
                    if not self.var_map and grads_l1_tiling_nparts[0] > fmap_l1_tiling_nparts[0]:
                        # hw splited one time before BL1  attach
                        hw_parts_offset = hw_mad_1_l1_out_at * tiling.get("BL1_shape")[0]
                    else:
                        # hw splited 2 times before BL1 attach
                        hw_parts_offset = \
                            ((hw_mad_1_l1_out_at * fmap_l1_tiling_nparts[0] //
                                grads_l1_tiling_nparts[0] + hw_mad_1_l1_in_at) *
                                tiling.get("BL1_shape")[0])

                hw_offset = hw_block_offset + hw_parts_offset
            else:
                # k axis is full load
                hw_offset = hw_block_offset
            # the offset of w according to that of k_axis
            hw_offset_with_pad = stride_width * hw_offset - pad_left
            # the extend of w according to that of k_axis
            if not tiling.get("BL1_shape"):
                kbl1_data = _ceil_div(hw_pad_1 * CUBE_DIM, block_dim_hw)
            else:
                kbl1_data = tiling.get("BL1_shape")[0]
            hw_extend = (kbl1_data - 1) * stride_width + kw_dilation
            if self.var_map:
                sch[fmap_l1].buffer_tile((None, None),
                                        (None, None),
                                        (None, None),
                                        (None, None),
                                        (hw_offset_with_pad, hw_extend),
                                        (None, None))
            else:
                sch[fmap_l1].buffer_tile((None, None),
                                        (None, None),
                                        (None, None),
                                        (hw_offset_with_pad, hw_extend),
                                        (None, None))

        def _get_bl1_bound():
            """
            for bl1_bound set for dynamic

            Returns :
            bl1_bound: dynamic_mode param for storage_bound

            additional_rows: int
                param for buffer_tile in multi-core cases
                -1 indicates not used

            ho_len: int
                actually number of lines loaded in dynamic_batch
                -1 indicates not used
            """

            def _set_additional_rows(bl1_k, width_grads, height_grads):
                """
                additional rows set for load3d

                Returns : 1. Exp for dynamic_hw
                    2. int for dynamic_batch
                """

                if 'fmap_h' in self.var_map or 'fmap_w' in self.var_map:
                    # dynamic_hw returns exp
                    # generally fix shape:
                    # 1. kbl1 small than wo: ho is 1 if wo % kbl1 is 0 else ho is 2
                    # 2. kbl1 bigger wo: ho is ceilDiv(kbl1, wo) if kbl1%wo is 0 else ho is ceilDiv(kbl1,wo) + 1
                    # dynamic_hw need to consider multi_core tail blocks: hw_single_core%wo == 0
                    return tvm.select(
                               bl1_k <= width_grads,
                               tvm.select(
                                   tvm.all(
                                       tvm.floormod(width_grads, bl1_k) == 0,
                                       tvm.floormod(hw_single_core_factor,
                                                    width_grads) == 0),
                                   1,
                                   LOOSE_LINE_CONDITION),
                               tvm.select(
                                   tvm.any(tvm.all(
                                       tvm.floormod(bl1_k, width_grads) == 0,
                                       tvm.floormod(hw_single_core_factor,
                                                    width_grads) == 0),
                                       bl1_k > width_grads * height_grads),
                                   0,
                                   tvm.select(
                                       tvm.floormod(bl1_k*2, width_grads) == 0,
                                       1,
                                       LOOSE_LINE_CONDITION)))

                if  (bl1_k % width_grads == 0) or (bl1_k > width_grads * height_grads):
                    # fully loaded dont extra line
                    return 0

                if (bl1_k * 2 % width_grads == 0
                    or bl1_k % width_grads == 1):
                    # special cases need only 1 extra line
                    return 1

                # other situations need 2 extra lines
                return 2

            if tiling.get("BL1_shape"):
                bl1_k = tiling.get("BL1_shape")[0]
                bl1_k_full = bl1_k
                if flag_all_one_case:
                    additional_rows = -1
                    ho_len = -1
                elif ('fmap_w' not in self.var_map
                      and bl1_k < width_grads):
                    additional_rows = -1
                    ho_len = 2
                    if not flag_conv1d_case:
                        # check if only need to load int times of bl0
                        ho_len = 1 if (width_grads % bl1_k == 0) \
                            else LOOSE_LINE_CONDITION
                        hi_max = kernel_height + (ho_len - 1) * stride_height
                        bl1_k_full = width_fmap * hi_max
                else:
                    # load3d can not split width_grads
                    additional_rows = _set_additional_rows(bl1_k, width_grads, height_grads)

                    ho_len = tvm.floordiv(bl1_k, width_grads) + additional_rows
                    hi_max = kernel_height + (ho_len - 1) * stride_height
                    bl1_k_full = hi_max * width_fmap

                bl1_bound = bl1_k_full * tiling.get("BL1_shape")[1] * \
                            tiling.get("BL0_matrix")[1] // \
                            (kernel_height * kernel_width) * c0_fmap

            else:
                bl1_k_full = _ceil_div(width_fmap * height_fmap, block_dim_hw)
                bl1_k_full = _align(bl1_k_full, CUBE_DIM)
                bl1_bound = bl1_k_full * c1_fmap * c0_fmap
                ho_len = height_fmap
                additional_rows = -1

            if flag_conv1d_case:
                bl1_hi = 1
                if not tiling.get("BL1_shape"):
                    kbl1_data = _ceil_div(hw_pad_1 * CUBE_DIM, block_dim_hw)
                    nbl1_c1 = c1_fmap
                else:
                    kbl1_data = tiling.get("BL1_shape")[0]
                    nbl1_c1 = ((tiling.get("BL1_shape")[1] *
                               tiling.get("BL0_matrix")[1]) //
                               (kernel_height * kernel_width))
                bl1_wi = (kbl1_data - 1) * stride_width + kw_dilation
                bl1_k_full = bl1_hi * bl1_wi
                bl1_bound = bl1_k_full * nbl1_c1 * c0_fmap

            return bl1_bound, additional_rows, ho_len

        def _get_al1_bound():
            # al1 set storage bound
            if tiling.get("AL1_shape"):
                al1_m = tiling.get("AL1_shape")[1] * \
                              tiling.get("AL0_matrix")[0] * CUBE_DIM
                al1_k = tiling.get("AL1_shape")[0]
                al1_bound = al1_k * al1_m
            else:
                al1_m = grads_matrix_c1 * grads_matrix_c0
                al1_bound = grads_matrix_howo * al1_m
            return al1_bound

        def _dynamic_bl1_buffer_tile(ho_len):
            # buffer_tile for dynamic mode

            def _set_tile_mode():
                # set buffer tile mode for dynamic

                if dynamic_bl1_attach == "dw_cc":
                    # hw need both multi-core offset and k_axis offset
                    return "tile_h_dw_cc"

                if dynamic_bl1_attach == "dw_ddr":
                    # hw only need multi_core offset
                    return "tile_h_dw_ddr"

                return "None"

            def _set_tile_params(ho_len, tile_mode):
                ho_min = 0

                # axis_k offset
                wi_min = -pad_left
                wi_extent = (width_grads - 1) * stride_width + kw_dilation

                bl1_k = tiling.get("BL1_shape")[0]
                al1_k = grads_matrix_howo
                if tiling.get("AL1_shape"):
                    al1_k = tiling.get("AL1_shape")[0]

                if bl1_at_axis == hw_mad_1_l1_in_at:
                    # hw splited two times before BL1 attach
                    axis_k_var = (hw_mad_1_l1_out_at.var * al1_k +
                                  hw_mad_1_l1_in_at.var * bl1_k)
                else:
                    # hw splited one time before BL1 attach
                    axis_k_var = hw_mad_1_l1_out_at.var * bl1_k

                if 'batch' in self.var_map:
                    # multi_core offset
                    block_div = block_dim_cout * block_dim_cin * block_dim_group
                    multi_core_offset = tvm.floordiv(
                                            tvm.floordiv(fused_multi_core, block_div),
                                            tvm.floordiv(batch_fmap - 1,
                                                         _ceil_div(batch_fmap,
                                                                   block_dim_batch)) +
                                            1) * hw_single_core_factor

                    if tile_mode == "tile_h_dw_cc":
                        ho_min = tvm.floordiv(multi_core_offset + axis_k_var,
                                              width_grads)

                    elif tile_mode == "tile_h_dw_ddr":
                        ho_min = tvm.floordiv(multi_core_offset, width_grads)

                elif 'fmap_h' in self.var_map or 'fmap_w' in self.var_map:
                    # multi_core offset
                    block_div = (block_dim_batch * block_dim_cout * block_dim_cin * block_dim_group)
                    multi_core_offset = fused_multi_core // block_div * \
                                      hw_single_core_factor

                    if tile_mode == "tile_h_dw_cc":
                        hw_block_offset = multi_core_offset + axis_k_var
                        ho_min = tvm.floordiv(hw_block_offset, width_grads)

                    elif tile_mode == "tile_h_dw_ddr":
                        hw_block_offset = multi_core_offset
                        ho_min = tvm.floordiv(hw_block_offset, width_grads)

                    # dynamic_hw need process multi_core tail blocks
                    ho_len = _ceil_div(tvm.floormod(hw_block_offset,
                                                     width_grads) +
                                       bl1_k,
                                       width_grads)

                hi_min = ho_min * stride_height - pad_up

                # Calculate the min and extent of the h dimension bound
                hi_extent = kernel_height + (ho_len - 1) * stride_height

                return hi_min, hi_extent, wi_min, wi_extent

            if not flag_all_one_case and not flag_conv1d_case:
                tile_mode = _set_tile_mode()

                hi_min, hi_extent, wi_min, wi_extent = \
                    _set_tile_params(ho_len, tile_mode)

                if "tile_h" not in tile_mode:
                    hi_min, hi_extent = None, None
                sch[fmap_l1].buffer_tile((None, None), (None, None), (None, None),
                                        (hi_min, hi_extent),
                                        (wi_min, wi_extent),
                                        (None, None))

            # mem management in dynamic mode
            sch[grads_matrix].set_buffer_size(al1_bound)
            sch[fmap_matrix].set_buffer_size(bl1_bound)

        def _dynamic_memory_management():
            # sequential_malloc
            sch.sequential_malloc(tbe_platform_info.scope_cbuf)
            sch.sequential_malloc(tbe_platform_info.scope_ca)
            sch.sequential_malloc(tbe_platform_info.scope_cb)
            sch.sequential_malloc(tbe_platform_info.scope_cc)
            if not self.cube_vector_split:
                sch.sequential_malloc(tbe_platform_info.scope_ubuf)

            # mem_unique
            sch[grads_matrix].mem_unique()
            sch[fmap_matrix].mem_unique()
            sch[grads_fractal].mem_unique()
            sch[fmap_fractal].mem_unique()
            sch[dw_cc].mem_unique()
            if not self.cube_vector_split:
                sch[dw_ub].mem_unique()

        if flag_conv1d_case:
            _split_w_for_conv1d()
        if self.l0b_dma_flag:
            sch[fmap_l1].compute_inline()
            sch[fmap_matrix].compute_inline()
            fmap_l1 = fmap_fractal_before
        l0a_attach_scope, l0a_attach_axis, _, _ = _l0_attach()
        _al1_attach()
        _bl1_attach()
        _double_buffer()
        _emit_insn()
        _handle_tbe_compile_para()
        if self.var_map:
            flag_bl1k_less_than_wo = tiling.get('flag_bl1k_less_than_wo')
            hw_single_core_factor = _ceil_div(hw_pad_1, block_dim_hw) * \
                                    CUBE_DIM
            hw_single_core_factor = _align(hw_single_core_factor,
                                           dw_k * width_grads * CUBE_DIM)
            if not flag_bl1k_less_than_wo:
                hw_single_core_factor = _ceil_div(_ceil_div(hw_pad_1, dw_k), block_dim_hw)
                hw_single_core_factor = hw_single_core_factor * dw_k * CUBE_DIM
            al1_bound = _get_al1_bound()
            bl1_bound, _, ho_len = _get_bl1_bound()
            _dynamic_bl1_buffer_tile(ho_len)
            _dynamic_memory_management()

        return True
