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
group_normalization_forward_training_reduce
"""
from __future__ import absolute_import
from __future__ import division
from functools import reduce as reduceIns

from tbe import tvm
from tbe.common.utils import shape_to_list
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform import scope_ubuf
from tbe.common.platform import SOC_VERSION
from tbe.common.platform import ASCEND_610
from tbe.common.platform import ASCEND_615
from tbe.common.platform import ASCEND_710
from tbe.common.platform import ASCEND_910
from tbe.common.platform import ASCEND_920A
from tbe.common.utils import log

from .util import DTYPE_WIDTH_MAP
from .util import get_reduce_axis_num

FORMAT_NCHW = "NCHW"
FORMAT_NHWC = "NHWC"


def get_max_ub_count(dtype):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    # div 2 for align to fp16
    total_size = get_soc_spec("UB_SIZE") // 2
    dtype_size = DTYPE_WIDTH_MAP.get(dtype)
    total_size = total_size // dtype_size
    total_size = total_size // 2  # div 2 for double buffer
    total_width = 4
    if not total_width:
        raise RuntimeError("Can not calculate with no compute")
    align_to = 128

    max_bound = total_width * align_to
    max_ub_count = int(total_size // max_bound * align_to)

    return max_ub_count


def sch_cut_not_reduce_axis_nchw(
        sch_list, res_list, sum_x, square_sum_x,
        data_ub, cast_0_ub, data_mul_ub,
        block_split_axis, block_inner,
        ub_split_axis, ub_inner, is_res_need_split):
    '''
    gn_reduce schedule for cut not reduce axis
    '''
    sch = sch_list[0]
    res_list[0] = sum_x
    res_list[1] = square_sum_x

    _, sum_x_ub = sch.cache_write([square_sum_x, sum_x],
                                  scope_ubuf)

    sum_x_block_outer, sum_x_block_inner = \
        sch[sum_x].split(sum_x.op.axis[block_split_axis],
                         factor=block_inner)

    sum_x_emit_axis = sum_x_block_inner
    block_fused_axis = sum_x_block_outer

    if block_split_axis == 1:
        block_fused_axis = sch[sum_x].fuse(block_fused_axis,
                                           sum_x.op.axis[0])

    if ub_split_axis > 1:
        sum_x_ub_outer, sum_x_ub_inner = \
            sch[sum_x_ub].split(sum_x_ub.op.reduce_axis[ub_split_axis - 2],
                                factor=ub_inner)
        sum_x_ub_emit_axis = sum_x_ub_inner
    else:
        sum_x_ub_outer, sum_x_ub_inner = \
            sch[sum_x_ub].split(sum_x_ub.op.axis[ub_split_axis],
                                factor=ub_inner)
        sum_x_ub_emit_axis = sum_x_ub.op.reduce_axis[0]

    sch[data_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
    sch[data_mul_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)

    if is_res_need_split:
        sum_x_block_inner_outer, sum_x_block_inner_inner = \
            sch[sum_x].split(sum_x_block_inner,
                             factor=ub_inner)

        sum_x_emit_axis = sum_x_block_inner_inner
        sch[sum_x_ub].compute_at(sch[sum_x], sum_x_block_inner_outer)
    else:
        sch[sum_x_ub].compute_at(sch[sum_x], block_fused_axis)

    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x].bind(block_fused_axis, block)

    sch[data_ub].double_buffer()

    sch[sum_x_ub].emit_insn(sum_x_ub_emit_axis, "vector_reduce_sum")

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")

    sch[sum_x].emit_insn(sum_x_emit_axis, "dma_copy")

    sch_list[0] = sch


def sch_cut_reduce_axis_nchw(
        sch_list, res_list, sum_x, square_sum_x,
        data_ub, cast_0_ub, data_mul_ub,
        block_split_axis, block_inner,
        ub_split_axis, ub_inner):
    '''
    group norm reduce schedule for cut reduce axis
    '''
    sch = sch_list[0]

    sum_x_block_outer, _ = \
        sch[sum_x].split(sum_x.op.reduce_axis[block_split_axis - 2],
                         factor=block_inner)

    block_fused_axis = sum_x_block_outer
    if block_split_axis == 3:
        block_fused_axis = sch[sum_x].fuse(sum_x.op.reduce_axis[0],
                                           block_fused_axis)
    elif block_split_axis == 4:
        block_fused_axis = sch[sum_x].fuse(sum_x.op.reduce_axis[0],
                                           sum_x.op.reduce_axis[1],
                                           block_fused_axis)

    sum_x_ub_rf, _ = sch.rfactor(sum_x, block_fused_axis)

    sum_x_global, square_sum_x_global = \
        sch.cache_write([sum_x, square_sum_x], "")
    res_list[0] = sum_x_global
    res_list[1] = square_sum_x_global

    sch[sum_x_ub_rf].set_scope(scope_ubuf)

    split_reduce_axis_index = ub_split_axis - block_split_axis
    # after rfactor, the new reduce axis is the last reduce axis
    reduce_axis_list = sum_x_ub_rf.op.reduce_axis[-1:] + \
        sum_x_ub_rf.op.reduce_axis[0:-1]

    sum_x_rf_outer, sum_x_rf_inner = sch[sum_x_ub_rf].split(
        reduce_axis_list[split_reduce_axis_index],
        factor=ub_inner)

    sch[sum_x_global].reorder(sum_x_global.op.reduce_axis[0],
                              sum_x_global.op.axis[0],
                              sum_x_global.op.axis[1],
                              sum_x_global.op.axis[2],
                              sum_x_global.op.axis[3],
                              sum_x_global.op.axis[4])

    reorder_axis_list = []
    reorder_axis_list.append(sum_x_ub_rf.op.axis[2])
    reorder_axis_list.append(sum_x_ub_rf.op.axis[0])
    reorder_axis_list.append(sum_x_ub_rf.op.axis[1])
    reorder_axis_list += reduce_axis_list[0:split_reduce_axis_index]
    reorder_axis_list.append(sum_x_rf_outer)
    reorder_axis_list.append(sum_x_rf_inner)
    reorder_axis_list += reduce_axis_list[split_reduce_axis_index + 1:]
    sch[sum_x_ub_rf].reorder(*reorder_axis_list)

    sch[sum_x_ub_rf].compute_at(sch[sum_x_global],
                                sum_x_global.op.reduce_axis[0])

    sch[data_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)

    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)
    sch[data_mul_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)

    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x_global].bind(sum_x_global.op.reduce_axis[0], block)

    sch[data_ub].double_buffer()

    sch[sum_x_ub_rf].emit_insn(sum_x_rf_inner, "vector_reduce_sum")

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")

    sch[sum_x_global].emit_insn(sum_x_global.op.axis[0], "dma_copy")

    sch[sum_x].emit_insn(sch[sum_x].op.axis[0], "phony_insn")
    sch_list[0] = sch


def sch_cut_not_reduce_axis_nhwc(
        sch_list, res_list, sum_x, square_sum_x,
        data_ub, cast_0_ub, data_mul_ub,
        block_split_axis, block_inner,
        ub_split_axis, ub_inner, group_size, dtype):
    '''
    gn_reduce schedule for cut not reduce axis
    '''
    sch = sch_list[0]
    res_list[0] = sum_x
    res_list[1] = square_sum_x

    _, sum_x_ub = sch.cache_write([square_sum_x, sum_x],
                                  scope_ubuf)

    sum_x_block_outer, sum_x_block_inner = \
        sch[sum_x].split(sum_x.op.axis[block_split_axis],
                         factor=block_inner)

    block_fused_axis = sum_x_block_outer

    if block_split_axis == 3:
        sch[sum_x].reorder(sum_x_block_outer,
                           sum_x.op.axis[0],
                           sum_x_block_inner)

    reorder_axis_list = []
    if ub_split_axis == 4:
        sum_x_ub_outer, sum_x_ub_inner = \
            sch[sum_x_ub].split(sum_x_ub.op.reduce_axis[2],
                                factor=ub_inner)
        sum_x_ub_emit_axis = sum_x_ub_inner
        reorder_axis_list += sum_x_ub.op.axis[0:3]
        reorder_axis_list += sum_x_ub.op.reduce_axis[0:2]
        reorder_axis_list += sum_x_ub.op.axis[3:]
        reorder_axis_list.append(sum_x_ub_outer)
        reorder_axis_list.append(sum_x_ub_inner)
    elif ub_split_axis == 3:
        sum_x_ub_outer, sum_x_ub_inner = \
            sch[sum_x_ub].split(sum_x_ub.op.axis[ub_split_axis],
                                factor=ub_inner)
        sum_x_ub_emit_axis = sum_x_ub.op.reduce_axis[-1]

        reorder_axis_list += sum_x_ub.op.axis[0:3]
        reorder_axis_list += sum_x_ub.op.reduce_axis[0:2]
        reorder_axis_list.append(sum_x_ub_outer)
        reorder_axis_list.append(sum_x_ub_inner)
        reorder_axis_list += sum_x_ub.op.axis[4:]
        reorder_axis_list += sum_x_ub.op.reduce_axis[2:]
    elif ub_split_axis in [1, 2]:
        sum_x_ub_outer, sum_x_ub_inner = \
            sch[sum_x_ub].split(sum_x_ub.op.reduce_axis[ub_split_axis - 1],
                                factor=ub_inner)
        sum_x_ub_emit_axis = sum_x_ub_inner

        reorder_axis_list += sum_x_ub.op.axis[0:3]
        if ub_split_axis == 1:
            reorder_axis_list.append(sum_x_ub_outer)
            reorder_axis_list.append(sum_x_ub_inner)
            reorder_axis_list += sum_x_ub.op.reduce_axis[1:2]
        else:
            reorder_axis_list += sum_x_ub.op.reduce_axis[0:1]
            reorder_axis_list.append(sum_x_ub_outer)
            reorder_axis_list.append(sum_x_ub_inner)
        reorder_axis_list += sum_x_ub.op.axis[3:]
        reorder_axis_list += sum_x_ub.op.reduce_axis[2:]
    else:
        sum_x_ub_outer, sum_x_ub_inner = \
            sch[sum_x_ub].split(sum_x_ub.op.axis[0],
                                factor=ub_inner)
        sum_x_ub_emit_axis = sum_x_ub_inner
        for axis in sum_x_ub.op.reduce_axis:
            if axis.dom.extent.value != 1:
                sum_x_ub_emit_axis = axis

        reorder_axis_list.append(sum_x_ub_outer)
        reorder_axis_list.append(sum_x_ub_inner)
        reorder_axis_list += sum_x_ub.op.axis[1:3]
        reorder_axis_list += sum_x_ub.op.reduce_axis[0:2]
        reorder_axis_list += sum_x_ub.op.axis[3:]
        reorder_axis_list += sum_x_ub.op.reduce_axis[2:]

    sch[sum_x_ub].reorder(*reorder_axis_list)

    sch[data_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
    sch[data_mul_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)

    sch[sum_x_ub].compute_at(sch[sum_x], block_fused_axis)

    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x].bind(block_fused_axis, block)

    sch[data_ub].double_buffer()
    sch[sum_x_ub].emit_insn(sum_x_ub_emit_axis, "vector_reduce_sum")

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")

    sch[sum_x].emit_insn(sum_x_block_inner, "dma_copy")

    def _do_storage_align():
        align_size = 16
        align_size_fp32 = 8
        if dtype == "float32":
            align_size = align_size_fp32

        if group_size % align_size == 0:
            return

        sch[data_ub].storage_align(sch[data_ub].op.axis[3], align_size, 0)
        if cast_0_ub is not None:
            sch[cast_0_ub].storage_align(
                sch[cast_0_ub].op.axis[3], align_size_fp32, 0)
        sch[data_mul_ub].storage_align(
            sch[data_mul_ub].op.axis[3], align_size_fp32, 0)
        sch[sum_x].storage_align(
            sch[sum_x].op.axis[3], align_size_fp32, 0)

    _do_storage_align()

    sch_list[0] = sch


def _get_cut_reduceaxis_nhwc_reorder_list(
        sum_x_ub_rf, block_split_axis, ub_split_axis,
        reduce_axis_list, sum_x_rf_outer, sum_x_rf_inner):
    """
    get reorder axis list
    :param sum_x_ub_rf:
    :param block_split_axis:
    :param ub_split_axis:
    :param reduce_axis_list:
    :param sum_x_rf_outer:
    :param sum_x_rf_inner:
    :return:
    """
    reorder_axis_list = []
    if block_split_axis == 1:
        # axis: N, outer,  1,   1,  G1,  1
        # reduce axis:   inner, W,      G0
        # ub_split_axis maybe [inner, W, G1, G0]
        reorder_axis_list.append(sum_x_ub_rf.op.axis[1])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[0])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[2])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[3])

        if ub_split_axis == 1:
            reorder_axis_list.append(sum_x_rf_outer)
            reorder_axis_list.append(sum_x_rf_inner)
            reorder_axis_list.append(reduce_axis_list[1])
            reorder_axis_list.append(sum_x_ub_rf.op.axis[4])
            reorder_axis_list.append(sum_x_ub_rf.op.axis[5])
            reorder_axis_list.append(reduce_axis_list[2])
        elif ub_split_axis == 2:
            reorder_axis_list.append(reduce_axis_list[0])
            reorder_axis_list.append(sum_x_rf_outer)
            reorder_axis_list.append(sum_x_rf_inner)
            reorder_axis_list.append(sum_x_ub_rf.op.axis[4])
            reorder_axis_list.append(sum_x_ub_rf.op.axis[5])
            reorder_axis_list.append(reduce_axis_list[2])
        elif ub_split_axis == 3:
            reorder_axis_list.append(reduce_axis_list[0])
            reorder_axis_list.append(reduce_axis_list[1])
            reorder_axis_list.append(sum_x_rf_outer)
            reorder_axis_list.append(sum_x_rf_inner)
            reorder_axis_list.append(sum_x_ub_rf.op.axis[5])
            reorder_axis_list.append(reduce_axis_list[2])
        else:
            reorder_axis_list.append(reduce_axis_list[0])
            reorder_axis_list.append(reduce_axis_list[1])
            reorder_axis_list.append(sum_x_ub_rf.op.axis[4])
            reorder_axis_list.append(sum_x_ub_rf.op.axis[5])
            reorder_axis_list.append(sum_x_rf_outer)
            reorder_axis_list.append(sum_x_rf_inner)
    elif block_split_axis == 2:
        # axis:  N, Houter, 1,   G1,    1
        # reduce axis:    inner,      G0
        # H and W_outer fused as Houter
        # ub_split_axis maybe [inner, G1, G0]
        reorder_axis_list.append(sum_x_ub_rf.op.axis[1])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[0])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[2])

        if ub_split_axis == 2:
            reorder_axis_list.append(sum_x_rf_outer)
            reorder_axis_list.append(sum_x_rf_inner)
            reorder_axis_list.append(sum_x_ub_rf.op.axis[3])
            reorder_axis_list.append(sum_x_ub_rf.op.axis[4])
            reorder_axis_list.append(reduce_axis_list[1])
        elif ub_split_axis == 3:
            reorder_axis_list.append(reduce_axis_list[0])
            reorder_axis_list.append(sum_x_rf_outer)
            reorder_axis_list.append(sum_x_rf_inner)
            reorder_axis_list.append(sum_x_ub_rf.op.axis[4])
            reorder_axis_list.append(reduce_axis_list[1])
        else:
            reorder_axis_list.append(reduce_axis_list[0])
            reorder_axis_list.append(sum_x_ub_rf.op.axis[3])
            reorder_axis_list.append(sum_x_ub_rf.op.axis[4])
            reorder_axis_list.append(sum_x_rf_outer)
            reorder_axis_list.append(sum_x_rf_inner)
    else:
        # axis: N,      1,  1,   G1, outer,     1
        # reduce axis:  H,  W,              inner
        # ub split axis maybe   inner,
        reorder_axis_list.append(sum_x_ub_rf.op.axis[4])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[0])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[1])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[2])
        reorder_axis_list.append(reduce_axis_list[0])
        reorder_axis_list.append(reduce_axis_list[1])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[3])
        reorder_axis_list.append(sum_x_ub_rf.op.axis[5])
        reorder_axis_list.append(sum_x_rf_outer)
        reorder_axis_list.append(sum_x_rf_inner)

    return reorder_axis_list


def sch_cut_reduce_axis_nhwc(
        sch_list, res_list, sum_x, square_sum_x,
        data_ub, cast_0_ub, data_mul_ub,
        block_split_axis, block_inner,
        ub_split_axis, ub_inner, group_size, dtype):
    '''
    group norm reduce schedule for cut reduce axis
    '''
    sch = sch_list[0]

    block_split_reduce_axis = block_split_axis - 1
    if block_split_axis == 4:
        block_split_reduce_axis = 2
    sum_x_block_outer, _ = \
        sch[sum_x].split(sum_x.op.reduce_axis[block_split_reduce_axis],
                         factor=block_inner)

    block_fused_axis = sum_x_block_outer

    if block_split_axis == 2:
        block_fused_axis = sch[sum_x].fuse(sum_x.op.reduce_axis[0],
                                           block_fused_axis)

    sum_x_ub_rf, _ = sch.rfactor(sum_x, block_fused_axis)

    sum_x_global, square_sum_x_global = \
        sch.cache_write([sum_x, square_sum_x], "")
    res_list[0] = sum_x_global
    res_list[1] = square_sum_x_global

    sch[sum_x_ub_rf].set_scope(scope_ubuf)

    # after rfactor, the new reduce axis is the last reduce axis
    if block_split_axis in [1, 2]:
        reduce_axis_list = sum_x_ub_rf.op.reduce_axis[-1:] + \
            sum_x_ub_rf.op.reduce_axis[0:-1]
    else:
        reduce_axis_list = sum_x_ub_rf.op.reduce_axis[:]

    if ub_split_axis in [1, 2, 4]:
        split_reduce_axis_index = ub_split_axis - 1
        if block_split_axis == 2:
            split_reduce_axis_index -= 1

        if ub_split_axis == 4:
            split_reduce_axis_index = 2
        sum_x_rf_outer, sum_x_rf_inner = sch[sum_x_ub_rf].split(
            reduce_axis_list[split_reduce_axis_index],
            factor=ub_inner)
    else:
        sum_x_rf_outer, sum_x_rf_inner = sch[sum_x_ub_rf].split(
            sum_x_ub_rf.op.axis[ub_split_axis],
            factor=ub_inner)

    sch[sum_x_global].reorder(sum_x_global.op.reduce_axis[0],
                              sum_x_global.op.axis[0],
                              sum_x_global.op.axis[1],
                              sum_x_global.op.axis[2],
                              sum_x_global.op.axis[3],
                              sum_x_global.op.axis[4])

    reorder_axis_list = _get_cut_reduceaxis_nhwc_reorder_list(
        sum_x_ub_rf, block_split_axis, ub_split_axis,
        reduce_axis_list, sum_x_rf_outer, sum_x_rf_inner
    )
    sch[sum_x_ub_rf].reorder(*reorder_axis_list)

    sch[sum_x_ub_rf].compute_at(sch[sum_x_global],
                                sum_x_global.op.reduce_axis[0])

    sch[data_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)

    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)
    sch[data_mul_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)

    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x_global].bind(sum_x_global.op.reduce_axis[0], block)

    sch[data_ub].double_buffer()

    sch[sum_x_ub_rf].emit_insn(sum_x_rf_inner, "vector_reduce_sum")

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")

    sch[sum_x_global].emit_insn(sum_x_global.op.axis[0], "dma_copy")

    sch[sum_x].emit_insn(sch[sum_x].op.axis[0], "phony_insn")

    def _do_storage_align():
        align_size = 16
        align_size_fp32 = 8
        if dtype == "float32":
            align_size = align_size_fp32

        if group_size % align_size == 0:
            return

        sch[data_ub].storage_align(sch[data_ub].op.axis[3], align_size, 0)
        if cast_0_ub is not None:
            sch[cast_0_ub].storage_align(
                sch[cast_0_ub].op.axis[3], align_size_fp32, 0)
        sch[data_mul_ub].storage_align(
            sch[data_mul_ub].op.axis[3], align_size_fp32, 0)

    _do_storage_align()

    sch_list[0] = sch


def _is_supported_atomic_add(reduce_tensor):
    """
    :return: Bool
    """
    # cloud, fp32
    if reduce_tensor is None:
        return False
    dtype = reduce_tensor.dtype
    if dtype != "float32":
        return False
    soc_ver = get_soc_spec(SOC_VERSION)
    is_version_support = (soc_ver in (ASCEND_910, ASCEND_920A, ASCEND_610, ASCEND_615, ASCEND_710))
    if not is_version_support:
        return False
    tag = reduce_tensor.op.tag
    if tag.find("sum") != -1:
        return True
    return False


GN_REDUCE_NCHW_TILING_MAP = {
    "1_48_1_35_35_float32_Ascend710": (1, 8, 1, 2),
    "1_48_1_39_39_float32_Ascend710": (1, 8, 1, 2),
    "1_1536_1_9_9_float32_Ascend710": (1, 192, 1, 12),
    "2_1024_1_14_14_float32_Ascend710": (1, 256, 1, 4),
}


def _gn_reduce_nchw_tiling(shape_before_reduce,
                           is_support_atomic_add,
                           max_ub_count, dtype):
    """
    get gn_reduce tiling of nchw
    :param shape_before_reduce:
    :param max_ub_count:
    :param dtype:
    :return:
    """
    core_num = get_soc_spec("CORE_NUM")
    soc_version = get_soc_spec(SOC_VERSION)
    shape_key = "_".join(str(i) for i in shape_before_reduce) + \
                "_" + dtype + "_" + soc_version
    if shape_key in GN_REDUCE_NCHW_TILING_MAP:
        block_split_axis, block_inner, \
        ub_split_axis, ub_inner = \
            GN_REDUCE_NCHW_TILING_MAP[shape_key]

        return block_split_axis, block_inner, ub_split_axis, ub_inner

    n_size = shape_before_reduce[0]
    group_nums = shape_before_reduce[1]

    one_core_data_threshold = 32
    one_core_min_data_size_fp32 = 8

    def _get_noreduce_block_tiling():
        split_axis = 0
        if n_size >= core_num:
            block_factor = (n_size + core_num - 1) // core_num
            res_nums = block_factor * group_nums
        elif n_size * group_nums >= core_num:
            split_axis = 1
            block_factor = group_nums
            for i in range(1, group_nums + 1, 1):
                if n_size * i < core_num:
                    continue
                block_factor = (group_nums + i - 1) // i
                break
            res_nums = block_factor
        else:
            return None, None, None

        return split_axis, block_factor, res_nums

    def _get_ub_tiling(block_axis, block_factor):
        ub_axis = len(shape_before_reduce) - 1
        ub_factor = shape_before_reduce[-1]

        tmp_size = 1
        i = len(shape_before_reduce) - 1
        is_find_ub_tiling = False
        for i in range(len(shape_before_reduce) - 1, block_axis, -1):
            if tmp_size*shape_before_reduce[i] < max_ub_count:
                tmp_size *= shape_before_reduce[i]
                continue
            for j in range(shape_before_reduce[i], 0, -1):
                if j*tmp_size > max_ub_count:
                    continue
                is_find_ub_tiling = True
                ub_axis = i
                ub_factor = j
                break
            if is_find_ub_tiling:
                break

        if not is_find_ub_tiling:
            is_block_no_tile = \
                shape_before_reduce[block_axis] % block_factor == 0
            for j in range(block_factor, 0, -1):
                if not is_block_no_tile and block_factor % j != 0:
                    continue
                if j*tmp_size > max_ub_count:
                    continue

                ub_axis = block_axis
                ub_factor = j
                break

        if ub_axis <= 1:
            reduce_size = reduceIns(lambda x, y: x*y,
                                    shape_before_reduce[2:])

            if reduce_size * DTYPE_WIDTH_MAP[dtype] * 2 % 32 != 0:
                ub_axis = 2
                ub_factor = shape_before_reduce[2]

        return ub_axis, ub_factor

    def _get_no_atomic_tiling(block_axis, block_factor, res_nums):
        new_block_factor = block_factor
        new_block_axis = block_axis
        if res_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
            if block_axis == 0:
                new_block_factor = n_size
            else:
                i = block_factor
                for i in range(block_factor, group_nums):
                    if group_nums % i != 0:
                        continue
                    if i * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                        continue
                    new_block_factor = i
                if i == group_nums:
                    new_block_axis = 0
                    new_block_factor = n_size

        ub_axis, ub_factor = \
            _get_ub_tiling(new_block_factor, new_block_axis)

        return block_axis, block_factor, ub_axis, ub_factor

    block_split_axis, block_inner, one_core_res_nums = \
        _get_noreduce_block_tiling()

    if not is_support_atomic_add:
        # block_split_axis may be None
        block_split_axis, block_inner, ub_split_axis, ub_inner =\
            _get_no_atomic_tiling(block_split_axis,
                                  block_inner, one_core_res_nums)

        if block_split_axis == 1:
            _is_multi_core_need_fused = True

        return block_split_axis, block_inner, ub_split_axis, ub_inner

    is_cut_reduce_axis = \
        block_split_axis is None or \
        one_core_res_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32

    def _find_cut_reduce_tiling(new_core_num):
        if new_core_num > shape_before_reduce[2] > new_core_num // 2:
            return 2, 1, shape_before_reduce[2]

        tmp_size = 1
        for i in range(2, len(shape_before_reduce), 1):
            if tmp_size*shape_before_reduce[i] < new_core_num:
                tmp_size *= shape_before_reduce[i]
                continue
            for j in range(1, shape_before_reduce[i] + 1):
                if j*tmp_size < new_core_num:
                    continue

                block_axis = i
                if j*tmp_size % new_core_num > 0 and j > 1:
                    j -= 1
                block_factor = (shape_before_reduce[i] + j - 1) // j
                block_outer = \
                    tmp_size * \
                    (shape_before_reduce[i] + block_factor - 1) // \
                    block_factor
                return block_axis, block_factor, block_outer


    def _find_new_block_tiling(not_cut_reduce_axis_max_core):
        reduce_size = reduceIns(lambda i, j: i*j, shape_before_reduce[2:])
        new_core_num = core_num

        if (reduce_size // core_num) < one_core_data_threshold:
            is_find_new_core_num = False
            for i in range(core_num - 1, 0, -1):
                if (reduce_size // i) >= one_core_data_threshold:
                    new_core_num = i
                    is_find_new_core_num = True
                    break
            if not is_find_new_core_num:
                return False, None, None

        block_axis, block_factor, block_outer = \
            _find_cut_reduce_tiling(new_core_num)

        if block_outer > not_cut_reduce_axis_max_core:
            return True, block_axis, block_factor
        return False, None, None


    def _find_not_reduce_block_tiling():
        # cut not reduce axis
        if group_nums * DTYPE_WIDTH_MAP[dtype] * 2 > 32:
            block_axis = 0
            block_factor = 1
            for i in range(1, group_nums + 1):
                if i * DTYPE_WIDTH_MAP[dtype] * 2 < 32 or \
                    (group_nums % i != 0 and \
                     group_nums % i * DTYPE_WIDTH_MAP[dtype] * 2 < 32):
                    continue
                block_axis = 1
                block_factor = i
                block_outer = n_size * (group_nums + i - 1) // i
                break
        else:
            block_axis = 0
            block_factor = n_size
            block_outer = 1
            for i in range(1, n_size + 1):
                if i * group_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32 or \
                    (n_size % i != 0 and \
                     n_size % i * group_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32):
                    continue
                block_axis = 0
                block_factor = i
                block_outer = (n_size + i - 1) // i
                break
        return block_axis, block_factor, block_outer

    def _get_block_outer_size(shape, block_axis, block_factor):
        if block_axis == 0:
            block_outer_size = shape[0] // block_factor
        elif block_axis == 1:
            block_outer_size = shape[0] * shape[1] // block_factor
        else:
            block_outer_size = shape[block_axis] // block_factor

            for i in range(2, block_axis, 1):
                block_outer_size *= shape[i]

        return block_outer_size

    def _check_can_multi_core_not_cut_reduce_axis():
        if group_nums >= one_core_min_data_size_fp32:
            if n_size > 1:
                return True

            if (group_nums + one_core_min_data_size_fp32 - 1) // \
                    one_core_min_data_size_fp32 > 1:
                return True

        for i in range(1, n_size, 1):
            if i * group_nums < one_core_min_data_size_fp32:
                continue
            if (n_size + i - 1) // i > 1:
                return True
        return False

    if is_cut_reduce_axis:
        # only can cut no reduce axis
        block_split_axis_not_reduce, block_inner_not_reduce, \
        block_outer_not_reduce = _find_not_reduce_block_tiling()

        # cut reduce axis as block
        is_find_new_block_tiling, new_block_split_axis, \
        new_block_inner = \
            _find_new_block_tiling(block_outer_not_reduce)

        if is_find_new_block_tiling:
            block_split_axis = new_block_split_axis
            block_inner = new_block_inner
        else:
            block_split_axis = block_split_axis_not_reduce
            block_inner = block_inner_not_reduce

    ub_split_axis, ub_inner = \
        _get_ub_tiling(block_split_axis, block_inner)

    return block_split_axis, block_inner, ub_split_axis, ub_inner


def _gn_reduce_nhwc_tiling(shape_before_reduce,
                           is_support_atomic_add,
                           max_ub_count, dtype):
    """
    get gn_reduce tiling of nhwc

    :param shape_before_reduce:
    :param max_ub_count:
    :param dtype:
    :return:
    """
    def _shape_mul(shape):
        if not shape:
            return 1
        return reduceIns(lambda x, y: x*y, shape)

    group_size = shape_before_reduce[-1]
    core_num = get_soc_spec("CORE_NUM")

    n_size = shape_before_reduce[0]
    group_nums = shape_before_reduce[3]

    def _get_noreduce_block_tiling():
        """
        cut not reduce axis
        :return:
        """
        outer = core_num
        if n_size >= core_num:
            block_axis = 0
            block_factor = (n_size + core_num - 1) // core_num
            res_nums = block_factor * group_nums
        else:
            block_axis = 0
            if group_nums >= 8:
                block_factor = 1
                res_nums = group_nums
                outer = n_size
            else:
                block_factor = n_size
                res_nums = n_size*group_nums
                outer = 1
                for i in range(1, n_size + 1):
                    if i * group_nums >= 8:
                        block_factor = i
                        res_nums = i*group_nums
                        outer = (n_size + block_factor - 1) // block_factor

        return block_axis, block_factor, res_nums, outer

    def _get_ub_tiling(block_axis, block_factor):
        ub_axis = len(shape_before_reduce) - 1
        ub_factor = shape_before_reduce[-1]

        shape_new = shape_before_reduce[:]
        shape_new[block_axis] = block_factor
        align_size_fp32 = 8

        total_size = \
            _shape_mul(shape_new[:-1]) * \
            (group_size + align_size_fp32 - 1) // align_size_fp32 *\
            align_size_fp32
        if total_size < max_ub_count:
            return block_axis, block_factor

        tmp_size = 1
        is_find_tiling = False
        for i in range(len(shape_new) - 1, block_axis, -1):
            dim = shape_new[i]
            if i == len(shape_new) - 1:
                dim = (dim + align_size_fp32 - 1) // align_size_fp32 *\
                      align_size_fp32

            if tmp_size*dim < max_ub_count:
                tmp_size *= dim
                continue

            for j in range(dim, 0, -1):
                if i == block_axis and dim % j != 0:
                    continue
                if j*tmp_size > max_ub_count:
                    continue

                is_find_tiling = True
                ub_axis = i
                ub_factor = j
                break
            if is_find_tiling:
                break

        if not is_find_tiling:
            for j in range(block_factor, 0, -1):
                if shape_new[block_axis] % j != 0:
                    continue
                if j*tmp_size > max_ub_count:
                    continue

                ub_axis = block_axis
                ub_factor = j
                break

        return ub_axis, ub_factor

    def _update_block_tiling(res_nums, block_axis, block_factor):
        if res_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
            if block_axis == 0:
                i = block_factor
                for i in range(block_factor, n_size, 1):
                    if n_size % i != 0:
                        continue
                    if i * group_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                        continue
                    block_factor = i
                if i == n_size:
                    block_axis = 0
                    block_factor = n_size
            else:
                i = block_factor
                for i in range(block_factor, group_nums):
                    if group_nums % i != 0:
                        continue
                    if n_size * i * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                        continue
                    block_factor = i

                if i == group_nums:
                    block_axis = 0
                    block_factor = n_size

        return block_axis, block_factor

    block_split_axis, block_inner, one_core_res_nums, block_outer = \
        _get_noreduce_block_tiling()

    if not is_support_atomic_add:
        block_split_axis, block_inner = \
            _update_block_tiling(block_split_axis, block_inner,
                                 one_core_res_nums)

        ub_split_axis, ub_inner = _get_ub_tiling(block_split_axis, block_inner)

        return block_split_axis, block_inner, ub_split_axis, ub_inner

    is_need_cut_reduce_axis = \
        block_outer < core_num or \
        one_core_res_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32 or \
        (block_split_axis == 3 and block_inner*DTYPE_WIDTH_MAP[dtype]*2 < 32)

    if is_need_cut_reduce_axis:
        # cut reduce axis as block
        h_size = shape_before_reduce[1]
        w_size = shape_before_reduce[2]
        group_size = shape_before_reduce[4]

        new_block_outer = 1
        if h_size >= core_num:
            new_block_split_axis = 1
            new_block_inner = (h_size + core_num - 1) // core_num
            new_block_outer = \
                (h_size + new_block_inner - 1) // new_block_inner
        elif h_size*w_size >= core_num:
            new_block_split_axis = 2
            new_block_inner = w_size // ((core_num + h_size - 1) // h_size)
            new_block_outer = \
                h_size * (w_size + new_block_inner - 1) // new_block_inner
        elif group_size >= core_num:
            one_time_data_size_threshold = 128
            if (group_size + core_num - 1) // core_num >= \
                    one_time_data_size_threshold:
                new_block_split_axis = 4
                new_block_inner = (group_size + core_num - 1) // core_num
                new_block_outer = \
                    (group_size + new_block_inner - 1) // new_block_inner
            else:
                new_block_split_axis = 2
                new_block_inner = 1
                new_block_outer = h_size * w_size

        if new_block_outer > block_outer:
            block_split_axis = new_block_split_axis
            block_inner = new_block_inner

    ub_split_axis, ub_inner = _get_ub_tiling(block_split_axis, block_inner)

    return block_split_axis, block_inner, ub_split_axis, ub_inner


def _gn_reduce_shc_do_cache(sch, input_tensor, sum_x, square_sum_x):
    is_cast = False
    if input_tensor.dtype == "float16":
        data_mul = square_sum_x.op.input_tensors[1]
        cast_0 = data_mul.op.input_tensors[0]
        is_cast = True
    else:
        data_mul = square_sum_x.op.input_tensors[1]

    data = input_tensor
    if is_cast:
        data_ub = sch.cache_read(data, scope_ubuf, [cast_0])
        cast_0_ub = \
            sch.cache_read(cast_0, scope_ubuf, [data_mul, sum_x])
    else:
        data_ub = sch.cache_read(data, scope_ubuf, [data_mul, sum_x])
        cast_0_ub = None

    data_mul_ub = sch.cache_read(data_mul, scope_ubuf, [sum_x])

    sch[data_mul].compute_inline()
    if is_cast:
        sch[cast_0].compute_inline()

    return data_ub, cast_0_ub, data_mul_ub


def _check_gn_reduce_params(shape_res, shape_input, res_dtype):
    """
    check gn_reduce params
    """
    if len(shape_res) != len(shape_input):
        raise RuntimeError("GnReduceSchedule only support keepdim is True!")

    if res_dtype != "float32":
        raise RuntimeError("GnReduceSchedule only support res is float32!")


def _check_res_need_split(shape_input,
                          block_split_axis, block_inner,
                          ub_split_axis, ub_inner,
                          _format, max_ub_count):
    """
    check result whether need split
    when one core result is greater than max_ub_count, need split
    """
    # reserved
    _ = ub_inner
    if _format == "NCHW":
        if block_split_axis in [2, 3, 4]:
            # cut reduce axis
            return False
        if block_split_axis < ub_split_axis:
            if block_split_axis == 0:
                one_core_result_size = block_inner * shape_input[1]
            else:
                one_core_result_size = block_inner
        else:
            one_core_result_size = block_inner
    else:
        if block_split_axis in [1, 2, 4]:
            # cut reduce axis
            return False
        if block_split_axis == ub_split_axis:
            one_core_result_size = block_inner
        else:
            one_core_result_size = block_inner * shape_input[3]

    return one_core_result_size > max_ub_count


def gn_reduce_schedule(res, input_tensors):
    """
    group_norm reduce schedule
    """
    log.info("gn_reduce_schedule start!")
    input_tensor = input_tensors[0]
    sum_x = res[0]
    square_sum_x = res[1]
    shape_input = shape_to_list(input_tensor.shape)

    shape_res = shape_to_list(sum_x.shape)

    res_dtype = sum_x[0].dtype

    _check_gn_reduce_params(shape_res, shape_input, res_dtype)

    reduce_axis_index = get_reduce_axis_num(sum_x)
    if reduce_axis_index == [1, 2, 4]:
        _format = FORMAT_NHWC
        group_size = shape_input[-1]
        tiling_func = _gn_reduce_nhwc_tiling
    else:
        _format = FORMAT_NCHW
        group_size = shape_input[2]
        tiling_func = _gn_reduce_nchw_tiling

    dtype = input_tensor.dtype.lower()

    is_support_atomic_add = _is_supported_atomic_add(sum_x)

    max_ub_count = get_max_ub_count(dtype)

    block_split_axis, block_inner, ub_split_axis, ub_inner = \
        tiling_func(
            shape_input, is_support_atomic_add,
            max_ub_count, res_dtype)

    is_res_need_split = \
        _check_res_need_split(
            shape_input,
            block_split_axis, block_inner,
            ub_split_axis, ub_inner,
            _format, max_ub_count
        )

    log.debug("gn_reduce_schedule tiling, " +
              "block_axis=%d, block_inner=%d, ub_axis=%d, ub_inner=%d",
              block_split_axis, block_inner, ub_split_axis, ub_inner)

    if is_res_need_split:
        one_time_dma_size = ub_inner
        dma_tile = block_inner % ub_inner

        is_block_conflict = one_time_dma_size < 8 or dma_tile < 8
        if is_block_conflict:
            return None

    sch = tvm.create_schedule([sum_x.op])
    sch_list = [sch]

    data_ub, cast_0_ub, data_mul_ub = \
        _gn_reduce_shc_do_cache(sch, input_tensor,
                                sum_x, square_sum_x)

    if _format == FORMAT_NCHW:
        if block_split_axis < 2 and block_split_axis <= ub_split_axis:
            sch_cut_not_reduce_axis_nchw(
                sch_list, res, sum_x, square_sum_x,
                data_ub, cast_0_ub, data_mul_ub,
                block_split_axis, block_inner,
                ub_split_axis, ub_inner, is_res_need_split)
        elif 2 <= block_split_axis <= ub_split_axis:
            sch_cut_reduce_axis_nchw(
                sch_list, res, sum_x,
                square_sum_x, data_ub, cast_0_ub,
                data_mul_ub, block_split_axis, block_inner,
                ub_split_axis, ub_inner)
        else:
            raise RuntimeError("gn_reduce_schedule not support!")
    else:
        if block_split_axis in [0, 3]:
            sch_cut_not_reduce_axis_nhwc(
                sch_list, res, sum_x, square_sum_x,
                data_ub, cast_0_ub, data_mul_ub,
                block_split_axis, block_inner,
                ub_split_axis, ub_inner, group_size, dtype)
        elif block_split_axis in [1, 2, 4]:
            sch_cut_reduce_axis_nhwc(
                sch_list, res, sum_x, square_sum_x,
                data_ub, cast_0_ub, data_mul_ub,
                block_split_axis, block_inner,
                ub_split_axis, ub_inner, group_size, dtype)
        else:
            raise RuntimeError("gn_reduce_schedule not support!")

    sch = sch_list[0]
    log.info("gn_reduce_schedule end!")

    return sch
