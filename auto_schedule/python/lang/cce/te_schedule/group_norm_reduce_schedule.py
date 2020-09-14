#!/usr/bin/env python # pylint: disable=too-many-lines
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

group_normalization_forward_training_reduce
"""

from __future__ import absolute_import
from __future__ import division
from functools import reduce as reduceIns
import te.lang.cce
from te import tvm
from te import platform as cceconf
from te.platform.cce_conf import CceProductParams as pver
from te.platform import log
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
    total_size = cceconf.get_soc_spec("UB_SIZE") // 2
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
        ub_split_axis, ub_inner):
    '''
    gn_reduce schedule for cut not reduce axis
    '''
    sch = sch_list[0]
    res_list[0] = sum_x
    res_list[1] = square_sum_x

    _, sum_x_ub = sch.cache_write([square_sum_x, sum_x],
                                  cceconf.scope_ubuf)

    sum_x_block_outer, sum_x_block_inner = \
        sch[sum_x].split(sum_x.op.axis[block_split_axis],
                         factor=block_inner)

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

    sch[sum_x_ub_rf].set_scope(cceconf.scope_ubuf)

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
        ub_split_axis, ub_inner):
    '''
    gn_reduce schedule for cut not reduce axis
    '''
    sch = sch_list[0]
    res_list[0] = sum_x
    res_list[1] = square_sum_x

    _, sum_x_ub = sch.cache_write([square_sum_x, sum_x],
                                  cceconf.scope_ubuf)

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
        ub_split_axis, ub_inner):
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

    sch[sum_x_ub_rf].set_scope(cceconf.scope_ubuf)

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
    is_version_support = pver().is_cloud_version() or \
                         pver().is_1951_version()
    if not is_version_support:
        return False
    tag = reduce_tensor.op.tag
    if tag.find("sum") != -1:
        return True
    return False


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
    core_num = cceconf.get_soc_spec("CORE_NUM")

    n_size = shape_before_reduce[0]
    group_nums = shape_before_reduce[1]

    def _get_noreduce_block_tiling():
        split_axis = 0
        if n_size >= core_num:
            block_factor = (n_size + core_num - 1) // core_num
            res_nums = block_factor * group_nums
        elif n_size * group_nums >= core_num:
            split_axis = 1
            block_factor = group_nums
            for i in range(group_nums, 0, -1):
                if group_nums % i != 0:
                    continue
                if n_size * (group_nums // i) < core_num:
                    continue
                block_factor = group_nums // i
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
            if i == block_axis + 1:
                ub_axis = block_axis + 1
                ub_factor = shape_before_reduce[block_axis + 1]
            elif i == block_axis:
                for j in range(block_factor, 0, -1):
                    if block_factor % j != 0:
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

    def _find_new_block_tiling():
        tmp_size = 1
        for i in range(2, len(shape_before_reduce), 1):
            if shape_before_reduce[i] < core_num:
                tmp_size *= shape_before_reduce[i]
                continue
            for j in range(1, shape_before_reduce[i] + 1):
                if shape_before_reduce[i] % j != 0:
                    continue
                if j*tmp_size < core_num:
                    continue

                block_axis = i
                block_factor = shape_before_reduce[i] // j
                return True, block_axis, block_factor

        return False, None, None

    def _find_not_reduce_block_tiling():
        # cut not reduce axis
        if group_nums * DTYPE_WIDTH_MAP[dtype] * 2 > 32:
            block_axis = 0
            block_factor = 1
            for i in range(1, group_nums + 1):
                if i * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                    continue
                block_axis = 1
                block_factor = i
                break
        else:
            block_axis = 0
            block_factor = n_size
            for i in range(1, n_size + 1):
                if i * group_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                    continue
                block_axis = 0
                block_factor = i
                break
        return block_axis, block_factor

    if is_cut_reduce_axis:
        # cut reduce axis as block
        is_find_new_block_tiling, block_split_axis, block_inner = \
            _find_new_block_tiling()

        if not is_find_new_block_tiling:
            block_split_axis, block_inner =\
                _find_not_reduce_block_tiling()

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

    core_num = cceconf.get_soc_spec("CORE_NUM")

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
            block_factor = n_size // core_num
            res_nums = block_factor * group_nums
        elif group_nums >= core_num:
            block_axis = 3
            block_factor = group_nums // core_num
            res_nums = n_size * block_factor
        else:
            if n_size > group_nums:
                if group_nums*DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                    # one core result less than 32B
                    block_axis = 0
                    block_factor = n_size
                    res_nums = n_size * group_nums
                    outer = 1
                else:
                    block_axis = 0
                    block_factor = 1
                    res_nums = group_nums
                    outer = n_size
            else:
                if n_size*DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                    # one core result less than 32B
                    block_axis = 0
                    block_factor = n_size
                    res_nums = n_size * group_nums
                    outer = 1
                else:
                    block_axis = 3
                    block_factor = 1
                    res_nums = n_size
                    outer = group_nums

        return block_axis, block_factor, res_nums, outer

    def _get_ub_tiling(block_axis, block_factor):
        ub_axis = len(shape_before_reduce) - 1
        ub_factor = shape_before_reduce[-1]

        shape_new = shape_before_reduce[:]
        shape_new[block_axis] = block_factor

        if _shape_mul(shape_new) < max_ub_count:
            return block_axis, block_factor

        tmp_size = 1
        is_find_tiling = False
        for i in range(len(shape_new) - 1, block_axis, -1):
            if tmp_size*shape_new[i] < max_ub_count:
                tmp_size *= shape_new[i]
                continue

            for j in range(shape_new[i], 0, -1):
                if i == block_axis and shape_new[i] % j != 0:
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

    _is_need_dichotomy_add = False

    if _shape_mul(shape_before_reduce) < max_ub_count:
        return block_split_axis, block_inner, block_split_axis, block_inner

    is_need_cut_reduce_axis = \
        block_outer < core_num or \
        one_core_res_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32 or \
        (block_split_axis == 3 and block_inner*DTYPE_WIDTH_MAP[dtype]*2 < 32)

    if is_need_cut_reduce_axis:
        # cut reduce axis as block
        h_size = shape_before_reduce[1]
        w_size = shape_before_reduce[2]
        group_size = shape_before_reduce[4]

        if h_size >= core_num:
            block_split_axis = 1
            block_inner = h_size // core_num
        elif h_size*w_size >= core_num:
            block_split_axis = 2
            block_inner = w_size // (core_num // h_size)
        elif group_size >= core_num:
            block_split_axis = 4
            block_inner = group_size // core_num

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
        data_ub = sch.cache_read(data, cceconf.scope_ubuf, [cast_0])
        cast_0_ub = \
            sch.cache_read(cast_0, cceconf.scope_ubuf, [data_mul, sum_x])
    else:
        data_ub = sch.cache_read(data, cceconf.scope_ubuf, [data_mul, sum_x])
        cast_0_ub = None

    data_mul_ub = sch.cache_read(data_mul, cceconf.scope_ubuf, [sum_x])

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


def gn_reduce_schedule(res, input_tensors):
    """
    group_norm reduce schedule
    """
    log.info("gn_reduce_schedule start!")
    input_tensor = input_tensors[0]
    sum_x = res[0]
    square_sum_x = res[1]
    shape_input = te.lang.cce.util.shape_to_list(input_tensor.shape)

    shape_res = te.lang.cce.util.shape_to_list(sum_x.shape)

    res_dtype = sum_x[0].dtype

    _check_gn_reduce_params(shape_res, shape_input, res_dtype)

    reduce_axis_index = get_reduce_axis_num(sum_x)
    if reduce_axis_index == [1, 2, 4]:
        _format = FORMAT_NHWC
        group_size = shape_input[-1]
        if group_size*4 % 32 != 0:
            # not 32B aligned, not process
            return None
        tiling_func = _gn_reduce_nhwc_tiling
    else:
        _format = FORMAT_NCHW
        tiling_func = _gn_reduce_nchw_tiling

    dtype = input_tensor.dtype.lower()

    is_support_atomic_add = _is_supported_atomic_add(sum_x)

    max_ub_count = get_max_ub_count(dtype)

    block_split_axis, block_inner, ub_split_axis, ub_inner = \
        tiling_func(
            shape_input, is_support_atomic_add,
            max_ub_count, res_dtype)

    log.debug("gn_reduce_schedule tiling, " + \
              "block_axis=%d, block_inner=%d, ub_axis=%d, ub_inner=%d",
              block_split_axis, block_inner, ub_split_axis, ub_inner)

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
                ub_split_axis, ub_inner)
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
                ub_split_axis, ub_inner)
        elif block_split_axis in [1, 2, 4]:
            sch_cut_reduce_axis_nhwc(
                sch_list, res, sum_x, square_sum_x,
                data_ub, cast_0_ub, data_mul_ub,
                block_split_axis, block_inner,
                ub_split_axis, ub_inner)
        else:
            raise RuntimeError("gn_reduce_schedule not support!")

    sch = sch_list[0]
    log.info("gn_reduce_schedule end!")

    return sch
