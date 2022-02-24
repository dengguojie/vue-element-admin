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
dilation schedule
"""
from functools import reduce

from tbe import tvm
from tbe.common.platform import CORE_NUM
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util
from . import gemm_schedule_util as util


def _cal_mini_ub(shape_input, shape_out, dtype):
    """
    calculate the mininum ub
    """
    return reduce(lambda x, y: x * y, shape_input) + reduce(lambda x, y: x * y, shape_out) * util.DATA_SIZE.get(dtype)


def _get_attach_axis(input_x, res, core_num):
    """
    get the axis for compute at
    """
    shape_input = util.shape_to_list(input_x.shape)
    shape_out = util.shape_to_list(res.shape)
    block_inner_parts = util.int_ceil_div(shape_out[0], core_num)
    shape_input = [block_inner_parts, *shape_input[1:]]
    shape_out = [block_inner_parts, *shape_out[1:]]

    w_dim = 2
    ub_size_max = tbe_platform_info.get_soc_spec("UB_SIZE")
    mini_ub_size = _cal_mini_ub(shape_input[w_dim:], shape_out[w_dim:], input_x.dtype)
    if mini_ub_size > ub_size_max:
        args_dict = {
            "errCode": "E60114",
            "reason": "mini split exceed UB Buffer",
            "value": "tiling size = {}".format(mini_ub_size)
        }
        raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

    double_flag = False
    attach_axis = w_dim
    for attach_dim in range(0, w_dim + 1):
        ub_size = _cal_mini_ub(shape_input[attach_dim:], shape_out[attach_dim:], input_x.dtype)
        if ub_size <= ub_size_max:
            double_flag = True
            attach_axis = attach_dim
            break
    return double_flag, attach_axis


def _set_tensor_scope(sch, res, input_x, init_ub):
    """
    set ub scope for tensor
    """
    dilation_ub = sch.cache_write(res, tbe_platform_info.scope_ubuf)
    x_ub = sch.cache_read(input_x, tbe_platform_info.scope_ubuf, [dilation_ub])
    sch[init_ub].set_scope(tbe_platform_info.scope_ubuf)
    return dilation_ub, x_ub


def _bind_multiblock(sch, res):
    """
    blind multi block upon batch axis
    """
    core_num = tbe_platform_info.get_soc_spec(CORE_NUM)
    batch_dim = util.shape_to_list(res.shape)[0]
    if core_num > batch_dim:
        core_num = batch_dim
    block_outer, block_inner = sch[res].split(res.op.axis[0], nparts=core_num)
    block_outer_outer, block_outer_inner = sch[res].split(block_outer, 1)
    blockidx = tvm.thread_axis("blockIdx.x")
    sch[res].bind(block_outer_outer, blockidx)
    return [block_outer_outer, block_outer_inner, block_inner], core_num


def _dilation_emitinsn(sch, res, dilation_ub, input_dtype):
    """
    emit insn for all tensor
    """
    dilations_para = [i.value for i in res.op.attrs["dilations_para"]]
    dilation_n_dim, dilation_h_dim, dilation_w_dim, _ = sch[dilation_ub].op.axis
    h_dim_outer, h_dim_inner = sch[dilation_ub].split(dilation_h_dim, dilations_para[1])
    w_dim_outer, w_dim_inner = sch[dilation_ub].split(dilation_w_dim, dilations_para[2])
    sch[dilation_ub].reorder(h_dim_inner, w_dim_inner, dilation_n_dim, h_dim_outer, w_dim_outer)
    sch[dilation_ub].unroll(h_dim_inner)
    sch[dilation_ub].unroll(w_dim_inner)

    dilation_emit = "vector_muls" if input_dtype in ("float16", "float32") else "dma_copy"
    sch[dilation_ub].emit_insn(sch[dilation_ub].op.axis[0], dilation_emit)


def dilation_schedule(res, sch_list):
    """
    schedule enter, auto schedule for cce AI-CORE

    Parameters:
    :param res: tvm.tensor
    :param sch_list: list of schedule, use sch_list[0] to return dilation schedule

    Returns:
    True for sucess, False for no schedule
    """
    sch = sch_list[0]
    x = res.op.input_tensors[0]
    init_ub = res.op.input_tensors[1]

    # set scope
    dilation_ub, x_ub = _set_tensor_scope(sch, res, x, init_ub)
    # blind multi block
    block_axis, core_num = _bind_multiblock(sch, res)

    # get attach axis
    double_flag, attach_axis = _get_attach_axis(x, res, core_num)
    # attach at
    attach_dim = [*block_axis[1:], sch[res].op.axis[1]]
    sch[dilation_ub].compute_at(sch[res], attach_dim[attach_axis])
    sch[init_ub].compute_at(sch[res], attach_dim[attach_axis])
    sch[x_ub].compute_at(sch[res], attach_dim[attach_axis])
    # unroll for dilation
    _dilation_emitinsn(sch, res, dilation_ub, x.dtype)
    # double buffer
    if double_flag:
        sch[dilation_ub].double_buffer()
        sch[init_ub].double_buffer()
        sch[x_ub].double_buffer()
    # reuseby and emit insn
    sch[dilation_ub].reused_by(init_ub)
    sch[init_ub].emit_insn(sch[init_ub].op.axis[0], "vector_dup")
    sch[x_ub].emit_insn(sch[x_ub].op.axis[0], "dma_copy")
    if attach_axis == 0:
        sch[res].emit_insn(block_axis[2], "dma_copy")
    else:
        sch[res].emit_insn(sch[res].op.axis[attach_axis], "dma_copy")
    return True

