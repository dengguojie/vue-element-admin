#!/usr/bin/env python
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

batch_normalization_forward_training_update
"""
# pylint: import-error, unused-import, ungrouped-imports
from __future__ import absolute_import
from __future__ import division
from functools import reduce
from math import sqrt
import te.lang.cce
from te import tvm
from te import platform as cceconf
from te.platform.cce_conf import CceProductParams as pver
import te.platform.cce_params as cce
from te.platform.fusion_manager import fusion_manager
from .util import get_nearest_factor
from .util import DTYPE_WIDTH_MAP
from te.platform import log

MAX_SHAPE_NUM = 10000000
BN_TYPE = 0
IN_TYPE = 1
GN_TYPE = 2

def get_max_ub_count(dtype, op_type):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    # div 2 for align to fp16
    total_size = cceconf.get_soc_spec("UB_SIZE") // 2
    dtype_size = DTYPE_WIDTH_MAP.get(dtype)
    total_size = total_size // dtype_size
    total_size = total_size // 2  # div 2 for double buffer
    if op_type == BN_TYPE:
        total_width_other = 7
        total_width_cloud = 3
    elif op_type == IN_TYPE:
        total_width_other = 9
        total_width_cloud = 7
    elif op_type == GN_TYPE:
        total_width_other = 9
        total_width_cloud = 7

    if pver().is_cloud_version():
        total_width = total_width_cloud
    else:
        total_width = total_width_other

    align_to = 128
    max_bound = total_width * align_to
    max_ub_count = int(total_size // max_bound * align_to)

    return max_ub_count


def get_ub_tiling(shape, block_tiling_axis, block_tiling_inner_loop,
                  max_ub_count):
    """
    get ub tiling
    """
    last_axis = len(shape) - 1
    ub_split_inner = 1
    ub_split_axis = 0
    if block_tiling_axis < 0 or block_tiling_axis > last_axis:
        return ub_split_axis, ub_split_inner

    bound_size = max_ub_count
    split_axis = block_tiling_axis
    step = -1
    temp_size = 1
    need_split = False
    for i in range(last_axis, block_tiling_axis + step, step):
        temp_size = temp_size * shape[i]
        if temp_size >= bound_size:
            split_axis = i
            temp_size = temp_size / shape[i]
            need_split = True
            break

    split_size = 1
    # split the split axis
    if need_split:
        for i in range(1, shape[split_axis] + 1, 1):
            if (temp_size * i) == bound_size:
                split_size = i
                break
            if (temp_size * i) > bound_size:
                split_size = i - 1
                split_size = get_nearest_factor(shape[split_axis],
                                                split_size)
                break
    else:
        split_size = block_tiling_inner_loop

    if split_axis == block_tiling_axis \
                     and split_size > block_tiling_inner_loop:
        split_size = block_tiling_inner_loop

    ub_split_inner = split_size
    ub_split_axis = split_axis

    return ub_split_axis, ub_split_inner


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


def _gen_reversed_subgraph_list(out_tensor, tensor_list_map,
                                tensor_list_dst_tensor_map,
                                input_broadcast_tensors):
    """traverse tensors by Depth-First-Search

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
                tensor_list_map[in_tensor.name] = in_tensor

                if in_tensor.op.tag.find("broadcast") != -1:
                    input_broadcast_tensors.append(cur_tensor)

            _map_apend(tensor_list_dst_tensor_map, in_tensor, cur_tensor)


def _find_split_axis(shape, begin_axis, end_axis, bound_size):
    axis_num = len(shape)
    if begin_axis >= axis_num or begin_axis < 0 \
            or end_axis >= axis_num or end_axis < 0:
        return 0, 1
    if begin_axis < end_axis:
        step = 1
    else:
        step = -1
    split_axis = end_axis
    temp_size = 1
    need_split = False
    for i in range(begin_axis, end_axis + step, step):
        temp_size = temp_size * shape[i]
        if temp_size >= bound_size:
            split_axis = i
            temp_size = temp_size / shape[i]
            need_split = True
            break

    split_size = 1
    # split the split axis
    if need_split:
        for i in range(1, shape[split_axis] + 1, 1):
            if shape[split_axis] % i == 0:
                if (temp_size * i) == bound_size:
                    split_size = i
                    break
                if (temp_size * i) > bound_size:
                    i = i - 1
                    while shape[split_axis] % i != 0:
                        i = i - 1
                    split_size = i
                    break

    return split_axis, split_size


def _in_update_need_double_buffer(shape, block_axis, block_tiling_inner_loop,
                                  ub_axis, ub_tiling_inner_loop):
    if ub_axis < block_axis or ub_axis < 0 or block_axis < 0:
        return False
    if ub_axis == block_axis:
        one_core_loop_number = block_tiling_inner_loop
    else:
        ub_tiling_outer_loop = shape[ub_axis] // ub_tiling_inner_loop
        one_core_loop_number = block_tiling_inner_loop * ub_tiling_outer_loop

    for i in range(block_axis + 1, ub_axis, 1):
        one_core_loop_number = one_core_loop_number * shape[i]

    return one_core_loop_number > 1


def _need_double_buffer_for_param_buffer_reuse(
        shape, block_axis,
        block_tiling_inner_loop,
        ub_axis, ub_tiling_inner_loop):
    if ub_axis < block_axis or ub_axis < 0 or block_axis < 0:
        return False

    batch = shape[0]

    if batch % 2 == 0:
        return True

    if ub_axis == block_axis:
        one_core_loop_number = block_tiling_inner_loop
    else:
        ub_tiling_outer_loop = shape[ub_axis] // ub_tiling_inner_loop
        one_core_loop_number = block_tiling_inner_loop * ub_tiling_outer_loop

    for i in range(block_axis + 1, ub_axis, 1):
        one_core_loop_number = one_core_loop_number * shape[i]

    return one_core_loop_number > 1


def bn_update_schedule_model_parallel( # pylint: disable=R0912, R0913, R0914, R0915
        shape_x, sch_list, phony_out,
        phony_out_ub, max_ub_count, x_input,
        input_tensor_buffer_tensor_map,
        mid_tensor_buffer_tensor_map,
        mid_out_tensor_list,
        shape_x_tensor_list,
        mid_out_tensor_read_buffer_map,
        phony_tensor_list,
        input_broadcast_tensor_buffers):
    """
    do schedule for model parallel case
    """
    sch = sch_list[0]
    core_num = cceconf.get_soc_spec("CORE_NUM")

    block_split_axis = 1
    res_block_outer, res_block_inner = sch[phony_out].split(
        phony_out.op.axis[block_split_axis], nparts=core_num)

    n_size = shape_x[0]
    c1_size = shape_x[1]
    h_size = shape_x[2]
    w_size = shape_x[3]
    c0_size = shape_x[4]

    # ub utilization ratio is small, so use "model parallel"
    # c1_size axis as block_axis and n_size axis as ub split axis
    # can raise dma copy data size and dichotomy efficiency
    ub_split_axis = 0
    ub_split_inner = 1
    if c1_size >= core_num and c1_size % core_num == 0:
        n_inner = n_size
    else:
        n_inner = n_size // core_num

    for i in range(n_inner, 0, -1):
        if n_inner % i != 0:
            continue
        if h_size*w_size*c0_size*i > max_ub_count:
            continue
        ub_split_inner = i
        break

    res_ub_outer, res_ub_inner = \
        sch[phony_out].split(phony_out.op.axis[ub_split_axis],
                             factor=ub_split_inner)

    reordered_axis_list = []
    reordered_axis_list.append(res_block_outer)
    reordered_axis_list.append(res_block_inner)
    reordered_axis_list.append(res_ub_outer)
    reordered_axis_list.append(res_ub_inner)
    reordered_axis_list.append(phony_out.op.axis[2])
    reordered_axis_list.append(phony_out.op.axis[3])
    reordered_axis_list.append(phony_out.op.axis[4])
    sch[phony_out].reorder(*reordered_axis_list)

    mean_compute_at_axis = res_block_outer

    block = tvm.thread_axis("blockIdx.x")
    sch[phony_out].bind(res_block_outer, block)

    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[buffer_tensor].compute_at(sch[phony_out], mean_compute_at_axis)
        else:
            sch[buffer_tensor].compute_at(sch[phony_out], res_ub_outer)

    for i in mid_tensor_buffer_tensor_map:
        buffer_tensor = mid_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[buffer_tensor].compute_at(sch[phony_out], mean_compute_at_axis)
        else:
            sch[buffer_tensor].compute_at(sch[phony_out], res_ub_outer)

    for i in mid_out_tensor_list:
        if i not in shape_x_tensor_list:
            sch[i].compute_at(sch[phony_out], mean_compute_at_axis)
        else:
            sch[i].compute_at(sch[phony_out], res_ub_outer)

    for i in mid_out_tensor_read_buffer_map:
        buffer_tensor = mid_out_tensor_read_buffer_map[i]
        if i not in shape_x_tensor_list:
            sch[buffer_tensor].compute_at(sch[phony_out], mean_compute_at_axis)
        else:
            sch[buffer_tensor].compute_at(sch[phony_out], res_ub_outer)

    sch[phony_out_ub].compute_at(sch[phony_out], res_ub_outer)

    x_input_ub = input_tensor_buffer_tensor_map[x_input]
    sch[x_input_ub].double_buffer()

    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")

    batch = shape_x[0]
    c1_size = shape_x[1]
    c0_size = shape_x[4]

    for i in mid_tensor_buffer_tensor_map:
        buffer_tensor = mid_tensor_buffer_tensor_map[i]
        if i not in phony_tensor_list:
            if buffer_tensor in input_broadcast_tensor_buffers:
                shape = i.shape
                shape_size = reduce(lambda i, j: i * j, shape)
                # pylint: disable=too-many-boolean-expressions
                if shape_size.value // (batch*c1_size*c0_size) == 1:
                    insn = _get_emit_insn_map(i)
                else:
                    if i.op.tag.find("|") != -1:
                        str_list = i.op.tag.split("|")
                        tag = str_list[0]
                    else:
                        tag = i.op.tag

                    if tag == "elewise_binary_mul":
                        insn = "vector_mul_with_broadcast"
                    elif tag == "elewise_binary_add":
                        insn = "vector_add_with_broadcast"
                    else:
                        raise RuntimeError("Invalid tag")
            else:
                insn = _get_emit_insn_map(i)

            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)
        else:
            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0],
                                         "phony_insn")

    for i in mid_out_tensor_list:
        sch[i].emit_insn(i.op.axis[0], "dma_copy")

        if i in mid_out_tensor_read_buffer_map:
            phony_read_buffer = mid_out_tensor_read_buffer_map[i]
            sch[phony_read_buffer].emit_insn(
                phony_read_buffer.op.axis[0],
                "phony_insn")

    sch[phony_out_ub].emit_insn(phony_out_ub.op.axis[0], "phony_insn")
    sch[phony_out].emit_insn(res_ub_inner, "phony_insn")

    return sch


def _get_in_tensor_cnt(out_tensor):
    """get all input tensor count for current tensor

    Parameters
    ----------
    out_tensor : tensor
        need to count all its input tensorss

    Return
    ------
        count value for out_tensor inpute tensors
    """
    if out_tensor is None:
        return 0
    stack = [out_tensor]
    visited_list = []
    in_count = 0
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
            in_count = in_count + 1
    return in_count


def _check_is_model_para_case(shape_x, max_ub_count):
    """
    check is model paraller case
    :param shape_x:
    :param max_ub_count:
    :return:bool is_model_para
    """
    batch = shape_x[0]
    c1_size = shape_x[1]
    h_size = shape_x[2]
    w_size = shape_x[3]
    c0_size = shape_x[4]

    core_num = cceconf.get_soc_spec("CORE_NUM")

    if max_ub_count // (h_size*w_size*c0_size) < 2:
        return False

    if c1_size >= core_num and c1_size % core_num != 0:
        return False

    if c1_size < core_num and c1_size != 16:
        return False

    if batch % core_num != 0:
        return False

    return True


def _check_is_c1_match(c1_size, core_num):
    """
    check whether c1 is match
    """
    return c1_size < core_num or \
           (c1_size > core_num and c1_size % core_num != 0)


def _is_shape_contain_prime(shape):
    """
    check shape is contain prime that big than 5000
    :param shape:
    :return:
    """
    h_size = shape[2]
    w_size = shape[3]

    def _is_prime(num):
        for i in range(2, int(sqrt(num) + 1)):
            if num % i == 0:
                return False
        return True

    prime_threadhold = 5000
    return (h_size > prime_threadhold and _is_prime(h_size)) or \
           (w_size > prime_threadhold and _is_prime(w_size))


def _get_update_condition(cut_mode, block, core_num,
                          shape,# pylint: disable=too-many-statements
                          block_split_axis,
                          block_split_size):
    """
    when moving mean and var need condition, get condition
    """
    condition = []
    batch = shape[0]
    c1_size = shape[1]
    h_size = shape[2]

    if cut_mode == "cut_batch":
        condition.append(block < 1)
    elif cut_mode == "cut_batch_c1":
        condition.append(block < core_num // batch)
    elif cut_mode == "cut_batch_c1_hw":
        if c1_size == 1:
            condition.append(block < 1)
        else:
            block_factor = block_split_size
            if block_split_axis == 3 and block_split_size == 1:
                block_factor = h_size
            condition.append(block % block_factor < 1)
            condition.append(block < c1_size*block_factor)

    return condition


def _is_need_do_broadcast(input_tensors):
    """
    NHWC and group_size is not 32B aligned, need do broadcast
    :param input_tensors:
    :return:
    """
    x_input = input_tensors[-1]
    sum_input = input_tensors[0]
    shape_x = te.lang.cce.util.shape_to_list(x_input.shape)
    shape_sum = te.lang.cce.util.shape_to_list(sum_input.shape)

    broadcast_aixs = []
    for i, _ in enumerate(shape_x):
        if shape_x[i] != shape_sum[i]:
            broadcast_aixs.append(i)
        else:
            if shape_x[i] == 1:
                broadcast_aixs.append(i)

    nhwc_broadcast_axis = [1, 2, 4]
    is_nhwc_format = True
    for i in nhwc_broadcast_axis:
        is_nhwc_format = is_nhwc_format and i in broadcast_aixs
    if is_nhwc_format:
        return True

    nchw_broadcast_axis = [2, 3, 4]
    is_nchw_format = True
    for i in nchw_broadcast_axis:
        is_nchw_format = is_nchw_format and i in broadcast_aixs
    if is_nchw_format:
        return True

    return False


def _gn_update_sch_do_cache(
    sch, input_tensor_dst_tensor_map,
    mid_out_tensor_list,
    mid_tensor_dst_tensor_map,
    input_broadcast_tensors,
    broadcast_not_last_axis_tensors,
    input_tensor_buffer_tensor_map,
    mid_out_tensor_read_buffer_map,
    mid_tensor_buffer_tensor_map,
    input_broadcast_tensor_buffers):
    """
    gn_update schedule do cache_read/write and compute_inline
    """
    for key in input_tensor_dst_tensor_map:
        read_buffer = sch.cache_read(key, cce.scope_ubuf,
                                     input_tensor_dst_tensor_map[key])
        input_tensor_buffer_tensor_map[key] = read_buffer

    for i in mid_out_tensor_list:
        read_buffer = sch.cache_read(i, cce.scope_ubuf,
                                     mid_tensor_dst_tensor_map[i])
        mid_out_tensor_read_buffer_map[i] = read_buffer

    for key in mid_tensor_dst_tensor_map:
        if key not in broadcast_not_last_axis_tensors:
            write_buffer = sch.cache_write(key, cce.scope_ubuf)
            mid_tensor_buffer_tensor_map[key] = write_buffer
            if key in input_broadcast_tensors:
                input_broadcast_tensor_buffers.append(write_buffer)

    for key in mid_tensor_dst_tensor_map:
        if key not in mid_out_tensor_list:
            sch[key].compute_inline()

    for tensor in mid_out_tensor_list:
        tensor_ub = mid_tensor_buffer_tensor_map[tensor]
        reuse_tensor_ub = mid_out_tensor_read_buffer_map[tensor]
        sch[tensor_ub].reused_by(reuse_tensor_ub)


def _check_gn_update_params(res, input_tensors):
    """
    check gn_update params
    """
    is_res_num_match = len(res) == 3
    if not is_res_num_match:
        raise RuntimeError("Group normalization update output nums \
                            should be 3 or 5 or 7.")

    is_input_num_match = \
        len(input_tensors) == 3 or \
        len(input_tensors) == 5
    if not is_input_num_match:
        raise RuntimeError("Group normalization update input nums \
                            should be 3 or 5 or 7.")


def _gn_update_sch_do_compute_at(
    sch, res_out, res_out_ub, mean_compute_at_axis, res_ub_outer,
    input_tensor_buffer_tensor_map, shape_x_tensor_list,
    mid_tensor_buffer_tensor_map, mid_out_tensor_list,
    mid_out_tensor_read_buffer_map, is_block_conflict):
    """
    gn_update schedule do compute_at
    """
    if is_block_conflict:
        mean_compute_at_axis, _ = sch[res_out].split(
            mean_compute_at_axis, factor=32)

    for i in input_tensor_buffer_tensor_map:
        input_tensor_buffer = input_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[input_tensor_buffer].compute_at(sch[res_out],
                                                mean_compute_at_axis)
        else:
            sch[input_tensor_buffer].compute_at(sch[res_out], res_ub_outer)

    for i in mid_tensor_buffer_tensor_map:
        mid_tensor_buffer = mid_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[mid_tensor_buffer].compute_at(sch[res_out],
                                              mean_compute_at_axis)
        else:
            sch[mid_tensor_buffer].compute_at(sch[res_out], res_ub_outer)

    for i in mid_out_tensor_list:
        if i not in shape_x_tensor_list:
            sch[i].compute_at(sch[res_out], mean_compute_at_axis)
        else:
            sch[i].compute_at(sch[res_out], res_ub_outer)
    for i in mid_out_tensor_read_buffer_map:
        mid_out_buffer = mid_out_tensor_read_buffer_map[i]
        if i not in shape_x_tensor_list:
            sch[mid_out_buffer].compute_at(sch[res_out],
                                           mean_compute_at_axis)
        else:
            sch[mid_out_buffer].compute_at(sch[res_out], res_ub_outer)

    sch[res_out_ub].compute_at(sch[res_out], res_ub_outer)

    return mean_compute_at_axis


def _gn_update_schedule_do_tiling(
    sch, shape_x, sum_input, dtype, res_out):
    """
    gn_update schedule do tiling
    """
    max_ub_count = get_max_ub_count(dtype, GN_TYPE)
    core_num = cceconf.get_soc_spec("CORE_NUM")
    batch = shape_x[0]
    group_nums = shape_x[1]
    block_split_axis = 0

    shape = te.lang.cce.util.shape_to_list(sum_input.shape)

    if batch >= core_num:
        res_block_outer, res_block_inner = sch[res_out].split(
            res_out.op.axis[0], nparts=core_num)
        block_split_inner_size = shape_x[block_split_axis] // core_num
        fused_axis = res_block_outer
        one_block_size = (shape[0] // core_num) * shape[1] * shape[2] * \
                         shape[3] * shape[4]
    elif group_nums >= core_num:
        res_block_outer, res_block_inner = sch[res_out].split(
            res_out.op.axis[1], nparts=core_num)
        block_split_axis = 1
        block_split_inner_size = shape_x[block_split_axis] // core_num
        fused_axis = res_block_outer

        reordered_axis_list = [res_block_outer,
                               res_out.op.axis[0],
                               res_block_inner,
                               res_out.op.axis[2],
                               res_out.op.axis[3],
                               res_out.op.axis[4]]
        sch[res_out].reorder(*reordered_axis_list)
        one_block_size = shape[1] // core_num * shape[2] * shape[3] * shape[4]
    elif batch*group_nums >= core_num:
        block_split_size = batch * shape_x[1] // core_num
        res_block_outer, res_block_inner = \
            sch[res_out].split(res_out.op.axis[1],
                               factor=block_split_size)
        fused_axis = sch[res_out].fuse(res_block_outer,
                                       res_out.op.axis[0])
        block_split_axis = 1
        block_split_inner_size = block_split_size
        one_block_size = shape[1] // core_num * shape[2] * shape[3] * shape[4]
    else:
        block_split_axis, block_split_size = \
            _find_split_axis(shape_x, 0, 3, core_num)
        block_split_inner_size = \
            shape_x[block_split_axis] // block_split_size
        res_block_outer, res_block_inner = \
            sch[res_out].split(res_out.op.axis[block_split_axis],
                               nparts=block_split_size)

        need_fuse_list = [res_block_outer]
        one_block_size = shape[block_split_axis] // core_num
        for i in range(block_split_axis + 1, 5, 1):
            one_block_size *= shape[i]
        for i in range(block_split_axis - 1, -1, -1):
            need_fuse_list.append(res_out.op.axis[i])
        fused_axis = need_fuse_list[0]
        for i in range(1, len(need_fuse_list)):
            fused_axis = sch[res_out].fuse(fused_axis, need_fuse_list[i])

    is_block_conflict = (one_block_size < 8)

    ub_split_axis, ub_split_inner = get_ub_tiling(shape_x, block_split_axis,
                                                  block_split_inner_size,
                                                  max_ub_count)

    split_factor = ub_split_inner
    if ub_split_axis == block_split_axis:
        if block_split_inner_size % split_factor != 0:
            while block_split_inner_size % split_factor != 0:
                split_factor -= 1

        is_need_split_next = ub_split_axis == 1 and split_factor > 1
        if is_need_split_next:
            res_ub_outer, res_ub_inner = \
                sch[res_out].split(res_out.op.axis[2], factor=shape_x[2])
        elif ub_split_axis == 0:
            split_factor = group_nums
            res_ub_outer, res_ub_inner = \
                sch[res_out].split(res_out.op.axis[1], factor=split_factor)
        else:
            res_ub_outer, res_ub_inner = \
                sch[res_out].split(res_block_inner, factor=split_factor)
    else:
        res_ub_outer, res_ub_inner = \
            sch[res_out].split(res_out.op.axis[ub_split_axis],
                               factor=split_factor)

    need_db = _in_update_need_double_buffer(shape_x, block_split_axis,
                                            block_split_inner_size,
                                            ub_split_axis,
                                            ub_split_inner)

    return fused_axis, res_ub_outer, res_ub_inner, need_db, is_block_conflict


def _gn_update_schedule_do_emit_insn(
    sch, res_out, res_out_ub, res_ub_inner,
    input_tensor_buffer_tensor_map, mid_tensor_buffer_tensor_map,
    mid_out_tensor_list, mid_out_tensor_read_buffer_map):
    """
    gn_update schedule do emit_insn
    """
    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")

    for i in mid_tensor_buffer_tensor_map:
        buffer_tensor = mid_tensor_buffer_tensor_map[i]
        insn = _get_emit_insn_map(i)
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)

    insn = _get_emit_insn_map(res_out)
    sch[res_out_ub].emit_insn(res_out_ub.op.axis[0], insn)
    sch[res_out].emit_insn(res_ub_inner, "dma_copy")

    for i in mid_out_tensor_list:
        sch[i].emit_insn(i.op.axis[0], "dma_copy")
        if i in mid_out_tensor_read_buffer_map:
            phony_read_buffer = mid_out_tensor_read_buffer_map[i]
            sch[phony_read_buffer].emit_insn(
                phony_read_buffer.op.axis[0], "phony_insn")


def gn_update_schedule(res, input_tensors):
    """
    gn update schedule
    """
    _check_gn_update_params(res, input_tensors)

    x_input = input_tensors[0]
    shape_x = te.lang.cce.util.shape_to_list(res[0].shape)

    shape_x_size = 1
    for dim in x_input.shape:
        shape_x_size = shape_x_size*dim.value

    for tmp_ten in input_tensors:
        shape_size = 1
        for dim in tmp_ten.shape:
            shape_size = shape_size*dim.value
        if shape_x_size < shape_size:
            shape_x_size = shape_size
            x_input = tmp_ten

    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}
    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}

    mid_out_tensor_list = [res[1], res[2]]

    broadcast_not_last_axis_tensors = []
    input_broadcast_tensors = []
    _gen_reversed_subgraph_list(res[0], tensor_list_map,
                                tensor_list_dst_tensor_map,
                                input_broadcast_tensors)

    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        else:
            mid_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        if tensor.op.tag.find("broadcast") != -1:
            broadcast_not_last_axis_tensors.append(tensor)

    input_broadcast_tensors = list(set(input_broadcast_tensors))
    sch = tvm.create_schedule([res[0].op])

    input_tensor_buffer_tensor_map = {}
    mid_out_tensor_read_buffer_map = {}
    mid_tensor_buffer_tensor_map = {}
    input_broadcast_tensor_buffers = []

    is_need_do_broadcast = _is_need_do_broadcast(input_tensors)
    if is_need_do_broadcast:
        broadcast_not_last_axis_tensors = []

    _gn_update_sch_do_cache(
        sch, input_tensor_dst_tensor_map,
        mid_out_tensor_list,
        mid_tensor_dst_tensor_map,
        input_broadcast_tensors,
        broadcast_not_last_axis_tensors,
        input_tensor_buffer_tensor_map,
        mid_out_tensor_read_buffer_map,
        mid_tensor_buffer_tensor_map,
        input_broadcast_tensor_buffers)

    res_out = res[0]
    res_out_ub = sch.cache_write(res_out, cce.scope_ubuf)

    shape_x_tensor_list = []
    for i in tensor_list_map:
        tensor = tensor_list_map[i]
        shape = te.lang.cce.util.shape_to_list(tensor.shape)
        if shape == shape_x:
            shape_x_tensor_list.append(tensor)

    dtype = x_input.dtype.lower()

    fused_axis, res_ub_outer, res_ub_inner, need_db, is_block_conflict =\
        _gn_update_schedule_do_tiling(
            sch, shape_x, input_tensors[0], dtype, res_out)

    fused_axis = _gn_update_sch_do_compute_at(
        sch, res_out, res_out_ub, fused_axis, res_ub_outer,
        input_tensor_buffer_tensor_map, shape_x_tensor_list,
        mid_tensor_buffer_tensor_map, mid_out_tensor_list,
        mid_out_tensor_read_buffer_map, is_block_conflict)

    _gn_update_schedule_do_emit_insn(
        sch, res_out, res_out_ub, res_ub_inner,
        input_tensor_buffer_tensor_map, mid_tensor_buffer_tensor_map,
        mid_out_tensor_list, mid_out_tensor_read_buffer_map)

    block = tvm.thread_axis("blockIdx.x")
    sch[res_out].bind(fused_axis, block)

    if need_db:
        x_input_ub = input_tensor_buffer_tensor_map[x_input]
        sch[x_input_ub].double_buffer()

    return sch


def _is_gn_update_pattern(shape_x, shape_sum):
    broadcast_aixs = []
    for i, _ in enumerate(shape_x):
        if shape_x[i] != shape_sum[i]:
            broadcast_aixs.append(i)
        else:
            if shape_x[i] == 1:
                broadcast_aixs.append(i)

    nhwc_broadcast_axis = [1, 2, 4]
    nchw_broadcast_axis = [2, 3, 4]

    is_gn_update_nchw = True
    for i in nchw_broadcast_axis:
        is_gn_update_nchw = is_gn_update_nchw and \
                            i in broadcast_aixs

    is_gn_update_nhwc = True
    for i in nhwc_broadcast_axis:
        is_gn_update_nhwc = is_gn_update_nhwc and \
                            i in broadcast_aixs

    return is_gn_update_nchw or is_gn_update_nhwc


def _do_check_input_tensor(input_tensors):
    """
    check input tensor
    """
    is_input_num_match = \
        len(input_tensors) == 3 or \
        len(input_tensors) == 5 or \
        len(input_tensors) == 7
    if not is_input_num_match:
        raise RuntimeError("Batch normalization update input nums \
                            should be 3 or 5 or 7.")

def _in_update_get_comput_axis(shape_x, mean_compute_at_axis,
                               res_ub_outer, max_ub_count):
    shape_c0 = 16
    shape_c1 = shape_x[1]
    core_num = cceconf.get_soc_spec("CORE_NUM")

    if shape_x[0] >= core_num:
        batch_factor = (shape_x[0] + core_num - 1) // core_num
        if batch_factor * shape_c1 * shape_c0 > max_ub_count:
            mean_compute_at_axis = res_ub_outer
    elif shape_c1 * shape_c0 > max_ub_count:
        mean_compute_at_axis = res_ub_outer

    return mean_compute_at_axis

def _in_update_do_compute_at(sch, input_tensor_buffer_tensor_map,
                  shape_x_tensor_list,
                  mid_tensor_buffer_tensor_map,
                  mid_out_tensor_read_buffer_map,
                  mid_out_tensor_list, shape_x,
                  phony_out_ub, phony_out,
                  res_ub_outer, max_ub_count,
                  mean_compute_at_axis):

    mean_compute_at_axis = _in_update_get_comput_axis(shape_x,
                                                      mean_compute_at_axis,
                                                      res_ub_outer,
                                                      max_ub_count)
    for i in input_tensor_buffer_tensor_map:
        input_tensor_buffer = input_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[input_tensor_buffer].compute_at(sch[phony_out],
                                                mean_compute_at_axis)
        else:
            sch[input_tensor_buffer].compute_at(
                sch[phony_out], res_ub_outer)

    for i in mid_tensor_buffer_tensor_map:
        mid_tensor_buffer = mid_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[mid_tensor_buffer].compute_at(sch[phony_out],
                                              mean_compute_at_axis)
        else:
            sch[mid_tensor_buffer].compute_at(
                sch[phony_out], res_ub_outer)

    for i in mid_out_tensor_list:
        if i not in shape_x_tensor_list:
            sch[i].compute_at(sch[phony_out], mean_compute_at_axis)
        else:
            sch[i].compute_at(sch[phony_out], res_ub_outer)

    for i in mid_out_tensor_read_buffer_map:
        mid_out_buffer = mid_out_tensor_read_buffer_map[i]
        if i not in shape_x_tensor_list:
            sch[mid_out_buffer].compute_at(sch[phony_out],
                                           mean_compute_at_axis)
        else:
            sch[mid_out_buffer].compute_at(sch[phony_out], res_ub_outer)

    sch[phony_out_ub].compute_at(sch[phony_out], res_ub_outer)


def _in_update_do_emit_insn(sch, input_tensor_buffer_tensor_map,
                            mid_tensor_buffer_tensor_map,
                            shape_x, mid_out_tensor_list,
                            ub_split_axis, split_factor,
                            phony_out_ub, phony_out,
                            phony_tensor_list,
                            input_broadcast_tensor_buffers,
                            mid_out_tensor_read_buffer_map,
                            res_ub_inner):
    batch = shape_x[0]
    c1_size = shape_x[1]
    c0_size = shape_x[4]
    w_size = shape_x[3]

    vector_intr_with_boradcast_map = {
        "elewise_binary_mul": "vector_mul_with_broadcast",
        "elewise_binary_add": "vector_add_with_broadcast",
    }

    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].emit_insn(
            buffer_tensor.op.axis[0], "dma_copy")

    for i in mid_tensor_buffer_tensor_map:
        # after broadcast do add
        buffer_tensor = mid_tensor_buffer_tensor_map[i]
        if i not in phony_tensor_list:
            if buffer_tensor in input_broadcast_tensor_buffers:
                shape = i.shape
                shape_size = reduce(lambda i, j: i * j, shape)

                def get_insn_tensor_map(
                        shape_size, batch, c1_size, c0_size,
                        ub_split_axis, split_factor, w_size):
                    is_no_reduce = shape_size.value // \
                                   (batch * c1_size * c0_size) == 1
                    if is_no_reduce or \
                        (ub_split_axis == 3 and split_factor == 1) or \
                        (ub_split_axis == 2 and split_factor == 1 and \
                         w_size == 1):
                        insn = _get_emit_insn_map(i)
                    else:
                        if i.op.tag.find("|") != -1:
                            str_list = i.op.tag.split("|")
                            tag = str_list[0]
                        else:
                            tag = i.op.tag

                        if tag in ["elewise_binary_mul",
                                   "elewise_binary_add"]:
                            insn = vector_intr_with_boradcast_map[tag]
                        elif tag == "elewise_single_cast":
                            insn = "phony_insn"
                        else:
                            insn = _get_emit_insn_map(i)
                    return insn

                insn = get_insn_tensor_map(
                    shape_size, batch, c1_size, c0_size,
                    ub_split_axis, split_factor, w_size)
            else:
                insn = _get_emit_insn_map(i)

            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)
        else:
            sch[buffer_tensor].emit_insn(
                buffer_tensor.op.axis[0], "phony_insn")

    for i in mid_out_tensor_list:
        sch[i].emit_insn(i.op.axis[0], "dma_copy")
        if i in mid_out_tensor_read_buffer_map:
            phony_read_buffer = mid_out_tensor_read_buffer_map[i]
            sch[phony_read_buffer].emit_insn(
                phony_read_buffer.op.axis[0],
                "phony_insn")

    sch[phony_out_ub].emit_insn(phony_out_ub.op.axis[0], "phony_insn")
    sch[phony_out].emit_insn(res_ub_inner, "phony_insn")


def _in_update_get_res_ub_outer_inner(sch, phony_out, res_block_inner,
                           ub_split_axis, block_split_axis,
                           block_split_inner_size, split_factor,
                           shape_x):
    if ub_split_axis == block_split_axis:
        if block_split_inner_size % split_factor != 0:
            while block_split_inner_size % split_factor != 0:
                split_factor -= 1

        if ub_split_axis == 1 and split_factor > 1:
            # this case, C1 inner axis is not 1,
            # the scale and offset are big
            # than C0, so c1_inner_axis must be outer loop
            res_ub_outer, res_ub_inner = \
                sch[phony_out].split(
                    phony_out.op.axis[2], factor=shape_x[2])
        elif ub_split_axis == 0:
            split_factor = shape_x[1]
            res_ub_outer, res_ub_inner = \
                sch[phony_out].split(
                    phony_out.op.axis[1], factor=split_factor)
        else:
            res_ub_outer, res_ub_inner = \
                sch[phony_out].split(
                    res_block_inner, factor=split_factor)
    else:
        res_ub_outer, res_ub_inner = \
            sch[phony_out].split(phony_out.op.axis[ub_split_axis],
                                 factor=split_factor)
    return res_ub_outer, res_ub_inner, split_factor


def _in_update_res_block_outer_inner(sch, phony_out, shape_x):
    core_num = cceconf.get_soc_spec("CORE_NUM")
    batch = shape_x[0]
    c1_size = shape_x[1]
    h_size = shape_x[2]
    w_size = shape_x[3]
    c0_size = 16

    block_split_axis = 0

    if batch >= core_num:
        res_block_outer, res_block_inner = sch[phony_out].split(
            phony_out.op.axis[0], nparts=core_num)
        block_split_inner_size = shape_x[block_split_axis] // core_num
        fused_axis = res_block_outer
    elif c1_size >= core_num:
        res_block_outer, res_block_inner = sch[phony_out].split(
            phony_out.op.axis[1], nparts=core_num)
        block_split_axis = 1
        block_split_inner_size = shape_x[block_split_axis] // core_num
        fused_axis = res_block_outer

        reordered_axis_list = []
        reordered_axis_list.append(res_block_outer)
        reordered_axis_list.append(phony_out.op.axis[0])
        reordered_axis_list.append(res_block_inner)
        reordered_axis_list.append(phony_out.op.axis[2])
        reordered_axis_list.append(phony_out.op.axis[3])
        reordered_axis_list.append(phony_out.op.axis[4])
        sch[phony_out].reorder(*reordered_axis_list)
    elif batch * c1_size >= core_num:
        block_split_size = batch * shape_x[1] // core_num
        res_block_outer, res_block_inner = \
            sch[phony_out].split(phony_out.op.axis[1],
                                 factor=block_split_size)
        fused_axis = sch[phony_out].fuse(res_block_outer,
                                         phony_out.op.axis[0])
        block_split_axis = 1
        block_split_inner_size = block_split_size
    else:
        block_split_axis, block_split_size = \
            _find_split_axis(shape_x, 0, 3, core_num)
        block_split_inner_size = \
            shape_x[block_split_axis] // block_split_size
        res_block_outer, res_block_inner = \
            sch[phony_out].split(phony_out.op.axis[block_split_axis],
                                 nparts=block_split_size)

        need_fuse_list = [res_block_outer]
        for i in range(block_split_axis - 1, -1, -1):
            need_fuse_list.append(phony_out.op.axis[i])
        fused_axis = need_fuse_list[0]
        for i in range(1, len(need_fuse_list)):
            fused_axis = sch[phony_out].fuse(
                fused_axis, need_fuse_list[i])

    return res_block_outer, res_block_inner, block_split_inner_size,\
           fused_axis, block_split_axis


def in_update_schedule(res, input_tensors):
    """
    in update schedule
    """
    _do_check_input_tensor(input_tensors)

    x_input = input_tensors[-1]
    sum_input = input_tensors[0]
    shape_x = te.lang.cce.util.shape_to_list(x_input.shape)
    shape_sum = te.lang.cce.util.shape_to_list(sum_input.shape)

    is_gn_update = _is_gn_update_pattern(shape_x, shape_sum)
    log.debug("gn update schedule is %d", is_gn_update)
    if is_gn_update:
        return gn_update_schedule(res, input_tensors)

    def get_x_input(x_input, input_tensors, shape_x):
        if len(shape_x) != 5:
            raise RuntimeError("Batch normalization only support 5D format.")

        shape_x_size = 1
        for dim in x_input.shape:
            shape_x_size = shape_x_size * dim.value

        for tmp_ten in input_tensors:
            shape_size = 1
            for dim in tmp_ten.shape:
                shape_size = shape_size * dim.value
            if shape_x_size < shape_size:
                shape_x_size = shape_size
                x_input = tmp_ten
        return x_input

    x_input = get_x_input(x_input, input_tensors, shape_x)

    # res 0 is cast; res 1 is mul or add; res 2 is mul or add

    def get_phony_tensor_list(input_tensors, shape_x, res):
        if len(input_tensors) == 3 or len(input_tensors) == 5:
            phony_add_1 = res[2]
            phony_broadcast = te.lang.cce.broadcast(phony_add_1, shape_x)
            phony_tensor_list = [phony_broadcast]
        else:
            phony_add_1 = te.lang.cce.vadd(res[1], res[2])
            phony_broadcast = te.lang.cce.broadcast(phony_add_1, shape_x)
            phony_tensor_list = [phony_add_1, phony_broadcast]
        return phony_add_1, phony_broadcast, phony_tensor_list

    phony_add_1, phony_broadcast, phony_tensor_list = \
        get_phony_tensor_list(input_tensors, shape_x, res)

    phony_cast = phony_broadcast

    def do_cast_porc(phony_cast, input_tensors,
                     phony_broadcast, phony_tensor_list):
        is_cast = False

        for tensor in input_tensors:
            if tensor.dtype == "float16":
                is_cast = True

        if is_cast:
            phony_cast = te.lang.cce.cast_to(phony_broadcast, "float16")
            phony_tensor_list.append(phony_cast)
        return phony_cast, phony_tensor_list

    phony_cast, phony_tensor_list = \
        do_cast_porc(phony_cast, input_tensors,
        phony_broadcast, phony_tensor_list)
    phony_out = te.lang.cce.vadd(phony_cast, res[0])

    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}
    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}

    mid_out_tensor_list = [res[0], res[1], res[2]]

    def get_real_mid_out_tensor_list(input_tensors):
        if len(input_tensors) == 3 or len(input_tensors) == 5:
            real_mid_out_tensor_list = [res[1]]
        else:
            real_mid_out_tensor_list = []
        return real_mid_out_tensor_list

    real_mid_out_tensor_list = get_real_mid_out_tensor_list(input_tensors)

    broadcast_not_last_axis_tensors = []
    input_broadcast_tensors = []
    _gen_reversed_subgraph_list(phony_out, tensor_list_map,
                                tensor_list_dst_tensor_map,
                                input_broadcast_tensors)

    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        else:
            mid_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        if tensor.op.tag.find("broadcast") != -1:
            broadcast_not_last_axis_tensors.append(tensor)

    input_broadcast_tensors = list(set(input_broadcast_tensors))
    sch = tvm.create_schedule([phony_out.op])

    input_tensor_buffer_tensor_map = {}
    for key in input_tensor_dst_tensor_map:
        read_buffer = sch.cache_read(key, cce.scope_ubuf,
                                     input_tensor_dst_tensor_map[key])
        input_tensor_buffer_tensor_map[key] = read_buffer

    mid_out_tensor_read_buffer_map = {}
    for i in mid_out_tensor_list:
        read_buffer = sch.cache_read(i, cce.scope_ubuf,
                                     mid_tensor_dst_tensor_map[i])
        mid_out_tensor_read_buffer_map[i] = read_buffer

    mid_tensor_buffer_tensor_map = {}
    input_broadcast_tensor_buffers = []
    for key in mid_tensor_dst_tensor_map:
        if key not in broadcast_not_last_axis_tensors:
            write_buffer = sch.cache_write(key, cce.scope_ubuf)
            mid_tensor_buffer_tensor_map[key] = write_buffer
            if key in input_broadcast_tensors:
                input_broadcast_tensor_buffers.append(write_buffer)

    phony_out_ub = sch.cache_write(phony_out, cce.scope_ubuf)

    def do_compute_inline_reuse(sch, mid_tensor_dst_tensor_map,
                                mid_out_tensor_list):
        for key in mid_tensor_dst_tensor_map:
            if key not in mid_out_tensor_list:
                sch[key].compute_inline()

        for tensor in real_mid_out_tensor_list:
            tensor_ub = mid_tensor_buffer_tensor_map[tensor]
            reuse_tensor_ub = mid_out_tensor_read_buffer_map[tensor]
            sch[tensor_ub].reused_by(reuse_tensor_ub)

    do_compute_inline_reuse(sch, mid_tensor_dst_tensor_map,
                            mid_out_tensor_list)

    def do_get_shape_x_tensor_list(tensor_list_map):
        shape_x_tensor_list = []
        for i in tensor_list_map:
            tensor = tensor_list_map[i]
            shape = te.lang.cce.util.shape_to_list(tensor.shape)
            length = len(shape)
            if shape == shape_x and \
                not tensor.op.tag.find("broadcast") != -1 \
                or shape[0:length - 2] == shape_x[0:length - 2]:
                shape_x_tensor_list.append(tensor)
        return shape_x_tensor_list

    shape_x_tensor_list = do_get_shape_x_tensor_list(tensor_list_map)

    dtype = x_input.dtype.lower()
    max_ub_count = get_max_ub_count(dtype, IN_TYPE)

    res_block_outer, res_block_inner, block_split_inner_size, fused_axis, \
    block_split_axis = _in_update_res_block_outer_inner(sch,
                                                        phony_out, shape_x)

    mean_compute_at_axis = fused_axis

    ub_split_axis, ub_split_inner = get_ub_tiling(shape_x, block_split_axis,
                                                  block_split_inner_size,
                                                  max_ub_count)

    split_factor = ub_split_inner

    res_ub_outer, res_ub_inner, split_factor = \
        _in_update_get_res_ub_outer_inner(
            sch, phony_out, res_block_inner, ub_split_axis,
            block_split_axis, block_split_inner_size, split_factor,
            shape_x)

    block = tvm.thread_axis("blockIdx.x")
    sch[phony_out].bind(fused_axis, block)

    _in_update_do_compute_at(sch, input_tensor_buffer_tensor_map,
                             shape_x_tensor_list,
                             mid_tensor_buffer_tensor_map,
                             mid_out_tensor_read_buffer_map,
                             mid_out_tensor_list, shape_x,
                             phony_out_ub, phony_out,
                             res_ub_outer, max_ub_count,
                             mean_compute_at_axis)

    need_db = _in_update_need_double_buffer(shape_x, block_split_axis,
                                            block_split_inner_size,
                                            ub_split_axis,
                                            ub_split_inner)
    def do_double_buffer(need_db, sch,
                         input_tensor_buffer_tensor_map, x_input):
        if need_db:
            x_input_ub = input_tensor_buffer_tensor_map[x_input]
            sch[x_input_ub].double_buffer()

    do_double_buffer(need_db, sch,
                     input_tensor_buffer_tensor_map, x_input)

    _in_update_do_emit_insn(sch, input_tensor_buffer_tensor_map,
                            mid_tensor_buffer_tensor_map,
                            shape_x, mid_out_tensor_list,
                            ub_split_axis, split_factor,
                            phony_out_ub, phony_out,
                            phony_tensor_list,
                            input_broadcast_tensor_buffers,
                            mid_out_tensor_read_buffer_map,
                            res_ub_inner)

    return sch


def bn_update_schedule(res, input_tensors):
    """
    bn update schedule
    """
    if len(res) == 3 or len(res) == 1:
        return in_update_schedule(res, input_tensors)

    def check_input_res_num(input_tensors, res):
        is_res_num_valid = len(res) != 5 and len(res) != 6
        if is_res_num_valid:
            raise RuntimeError(
                "Batch normalization update output nums should be 5, \
                current is %d." % (len(res)))

        is_input_num_match = \
            len(input_tensors) == 5 or \
            len(input_tensors) == 7 or \
            len(input_tensors) == 8

        if not is_input_num_match:
            raise RuntimeError("Batch normalization update input nums \
                                should be 5 or 7 or 8.")
    check_input_res_num(input_tensors, res)

    is_update_v3 = False
    if len(input_tensors) == 5:
        is_update_v3 = True

    mask = None
    if len(res) == 5:
        # res_y has most input tensors, so recognize res_y by in tensors count
        cnt_0 = _get_in_tensor_cnt(res[0])
        cnt_4 = _get_in_tensor_cnt(res[4])
        if cnt_0 > cnt_4:
            # non UB fusion res order
            # res_y, mean, variance, save_mean_reduce, batch_variance
            res_y = res[0]
            mean = res[1]
            variance = res[2]
            save_mean_reduce = res[3]
            batch_variance = res[4]
        else:
            # UB fusion res order
            # mean, variance, save_mean_reduce, batch_variance, res_y
            mean = res[0]
            variance = res[1]
            save_mean_reduce = res[2]
            batch_variance = res[3]
            res_y = res[4]
    elif len(res) == 6:
        mean = res[0]
        variance = res[1]
        save_mean_reduce = res[2]
        batch_variance = res[3]
        res_y = res[4]
        mask = res[5]
    else:
        raise RuntimeError("res list size only support 5 or 6, " \
                           "current is [%d]." % len(res))

    # find input_x tensor for add
    x_input = input_tensors[0]
    shape_x_size = 1
    for dim in x_input.shape:
        shape_x_size = shape_x_size*dim.value

    for tmp_ten in input_tensors:
        shape_size = 1
        for dim in tmp_ten.shape:
            shape_size = shape_size*dim.value
        if shape_x_size < shape_size:
            shape_x_size = shape_size
            x_input = tmp_ten

    shape_x = te.lang.cce.util.shape_to_list(x_input.shape)
    if len(shape_x) != 5:
        raise RuntimeError("Batch normalization only support 5D format.")

    is_elewise_sch = is_update_v3 and _is_shape_contain_prime(shape_x)
    if is_elewise_sch:
        return None

    add_14 = mean
    add_17 = variance
    mul_0 = save_mean_reduce
    mul_11 = batch_variance

    phony_add_1 = te.lang.cce.vadd(add_14, add_17)
    phony_broadcast = te.lang.cce.broadcast(phony_add_1, shape_x)
    phony_cast = phony_broadcast
    is_cast = False

    for tensor in input_tensors:
        if tensor.dtype == "float16":
            is_cast = True

    phony_tensor_list = [phony_add_1, phony_broadcast]
    if is_cast:
        phony_cast = te.lang.cce.cast_to(phony_broadcast, "float16")
        phony_tensor_list.append(phony_cast)

    if mask is not None:
        phony_mask_cast = te.lang.cce.cast_to(mask, "float16")
        phony_mask_reduce = te.lang.cce.reduce_min(phony_mask_cast, -1,
                                                   keepdims=True)
        phony_mask_brc = te.lang.cce.broadcast(phony_mask_reduce, shape_x)

        phony_add_y = te.lang.cce.vadd(phony_cast, phony_mask_brc)

        phony_out = te.lang.cce.vadd(phony_add_y, res_y)

        phony_tensor_list = phony_tensor_list + \
                            [phony_mask_cast, phony_mask_reduce, \
                             phony_mask_brc, phony_add_y]
    else:
        phony_out = te.lang.cce.vadd(phony_cast, res_y)

    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}
    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}
    # for config output address same with input by index (2,3)
    if mask is not None:
        mid_out_tensor_list = [mul_0, mul_11, add_14, add_17, res_y, mask]
    else:
        mid_out_tensor_list = [mul_0, mul_11, add_14, add_17, res_y]
    real_mid_out_tensor_list = [mul_0, mul_11]
    broadcast_not_last_axis_tensors = []
    input_broadcast_tensors = []

    _gen_reversed_subgraph_list(phony_out, tensor_list_map,
                                tensor_list_dst_tensor_map,
                                input_broadcast_tensors)

    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        else:
            mid_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        if tensor.op.tag.find("broadcast") != -1:
            broadcast_not_last_axis_tensors.append(tensor)

    input_broadcast_tensors = list(set(input_broadcast_tensors))

    sch = tvm.create_schedule([phony_out.op])

    input_tensor_buffer_tensor_map = {}
    for key in input_tensor_dst_tensor_map:
        read_buffer = sch.cache_read(key, cce.scope_ubuf,
                                     input_tensor_dst_tensor_map[key])
        input_tensor_buffer_tensor_map[key] = read_buffer

    mid_out_tensor_read_buffer_map = {}
    for i in mid_out_tensor_list:
        read_buffer = sch.cache_read(i, cce.scope_ubuf,
                                     mid_tensor_dst_tensor_map[i])
        mid_out_tensor_read_buffer_map[i] = read_buffer

    mid_tensor_buffer_tensor_map = {}
    input_broadcast_tensor_buffers = []
    for key in mid_tensor_dst_tensor_map:
        if key not in broadcast_not_last_axis_tensors:
            write_buffer = sch.cache_write(key, cce.scope_ubuf)
            mid_tensor_buffer_tensor_map[key] = write_buffer
            if key in input_broadcast_tensors:
                input_broadcast_tensor_buffers.append(write_buffer)

    phony_out_ub = sch.cache_write(phony_out, cce.scope_ubuf)

    for key in mid_tensor_dst_tensor_map:
        if key not in mid_out_tensor_list:
            sch[key].compute_inline()

    for tensor in real_mid_out_tensor_list:
        tensor_ub = mid_tensor_buffer_tensor_map[tensor]
        reuse_tensor_ub = mid_out_tensor_read_buffer_map[tensor]
        sch[tensor_ub].reused_by(reuse_tensor_ub)

    shape_x_tensor_list = []
    for i in tensor_list_map:
        tensor = tensor_list_map[i]
        shape = te.lang.cce.util.shape_to_list(tensor.shape)
        length = len(shape)
        # need to check mask shape [****,2]
        if shape == shape_x and not tensor.op.tag.find("broadcast") != -1 \
            or shape[0:length-2] == shape_x[0:length-2]:
            shape_x_tensor_list.append(tensor)

    dtype = x_input.dtype.lower()
    max_ub_count = get_max_ub_count(dtype, BN_TYPE)

    is_model_para_case = _check_is_model_para_case(shape_x, max_ub_count)

    is_model_parallel_sch = not is_update_v3 and \
                            mask is None and is_model_para_case
    if is_model_parallel_sch:
        sch_list = [sch]
        return bn_update_schedule_model_parallel(
            shape_x, sch_list, phony_out, phony_out_ub,
            max_ub_count, x_input,
            input_tensor_buffer_tensor_map,
            mid_tensor_buffer_tensor_map,
            mid_out_tensor_list,
            shape_x_tensor_list,
            mid_out_tensor_read_buffer_map,
            phony_tensor_list,
            input_broadcast_tensor_buffers)

    core_num = cceconf.get_soc_spec("CORE_NUM")
    batch = shape_x[0]
    c1_size = shape_x[1]
    h_size = shape_x[2]
    w_size = shape_x[3]
    c0_size = 16

    is_param_buffer_reuse = True

    is_can_use_conditional_exec = False
    cut_mode = None
    if is_param_buffer_reuse:
        is_c1_match = _check_is_c1_match(c1_size, core_num)
        if is_c1_match:
            if batch >= core_num:
                is_can_use_conditional_exec = True
                cut_mode = "cut_batch"
            elif batch*c1_size >= core_num and\
                batch*c1_size % core_num == 0 and\
                core_num % batch == 0:
                is_can_use_conditional_exec = True
                cut_mode = "cut_batch_c1"
            elif batch*c1_size < core_num:
                is_can_use_conditional_exec = True
                cut_mode = "cut_batch_c1_hw"
    if mask is not None and is_model_para_case:
        is_can_use_conditional_exec = True
        cut_mode = "cut_batch"

    size_one_core_threshold = 512

    is_general_sch = is_update_v3 or \
                     ((not is_param_buffer_reuse or \
                     is_can_use_conditional_exec) and \
                     h_size*w_size*c0_size > size_one_core_threshold)
    block_split_size = 1
    if is_general_sch:
        block_split_axis = 0
        core_num = cceconf.get_soc_spec("CORE_NUM")
        if batch >= core_num:
            res_block_outer, res_block_inner = sch[phony_out].split(
                phony_out.op.axis[0], nparts=core_num)
            block_split_inner_size = shape_x[block_split_axis] // core_num
            fused_axis = res_block_outer
        elif c1_size >= core_num:
            res_block_outer, res_block_inner = sch[phony_out].split(
                phony_out.op.axis[1], nparts=core_num)
            block_split_axis = 1
            block_split_inner_size = shape_x[block_split_axis] // core_num
            fused_axis = res_block_outer

            reordered_axis_list = []
            reordered_axis_list.append(res_block_outer)
            reordered_axis_list.append(phony_out.op.axis[0])
            reordered_axis_list.append(res_block_inner)
            reordered_axis_list.append(phony_out.op.axis[2])
            reordered_axis_list.append(phony_out.op.axis[3])
            reordered_axis_list.append(phony_out.op.axis[4])
            sch[phony_out].reorder(*reordered_axis_list)
        elif batch*c1_size >= core_num:
            block_split_size = batch * shape_x[1] // core_num
            res_block_outer, res_block_inner = \
                sch[phony_out].split(phony_out.op.axis[1],
                                     factor=block_split_size)
            fused_axis = sch[phony_out].fuse(res_block_outer,
                                             phony_out.op.axis[0])
            block_split_axis = 1
            block_split_inner_size = block_split_size
        else:
            block_split_axis, block_split_size = \
                _find_split_axis(shape_x, 0, 3, core_num)
            block_split_inner_size = \
                shape_x[block_split_axis] // block_split_size
            res_block_outer, res_block_inner = \
                sch[phony_out].split(phony_out.op.axis[block_split_axis],
                                     nparts=block_split_size)

            need_fuse_list = [res_block_outer]
            for i in range(block_split_axis - 1, -1, -1):
                need_fuse_list.append(phony_out.op.axis[i])
            fused_axis = need_fuse_list[0]
            for i in range(1, len(need_fuse_list)):
                fused_axis = sch[phony_out].fuse(fused_axis, need_fuse_list[i])
    else:
        cut_mode = None
        block_split_axis = 1
        if c1_size < core_num:
            res_block_outer, res_block_inner = sch[phony_out].split(
                phony_out.op.axis[1], factor=1)
            block_split_inner_size = 1
        else:
            res_block_outer, res_block_inner = sch[phony_out].split(
                phony_out.op.axis[1], nparts=core_num)
            block_split_inner_size = shape_x[block_split_axis] // core_num

        block_split_axis = 1
        fused_axis = res_block_outer

        reordered_axis_list = []
        reordered_axis_list.append(res_block_outer)
        reordered_axis_list.append(phony_out.op.axis[0])
        reordered_axis_list.append(res_block_inner)
        reordered_axis_list.append(phony_out.op.axis[2])
        reordered_axis_list.append(phony_out.op.axis[3])
        reordered_axis_list.append(phony_out.op.axis[4])
        sch[phony_out].reorder(*reordered_axis_list)

    mean_compute_at_axis = fused_axis

    ub_split_axis, ub_split_inner = get_ub_tiling(shape_x, block_split_axis,
                                                  block_split_inner_size,
                                                  max_ub_count)

    split_factor = ub_split_inner
    if ub_split_axis == block_split_axis:
        if block_split_inner_size % split_factor != 0:
            while block_split_inner_size % split_factor != 0:
                split_factor -= 1

        if ub_split_axis == 1 and split_factor > 1:
            # this case, C1 inner axis is not 1, the scale and offset are big
            # than C0, so c1_inner_axis must be outer loop
            res_ub_outer, res_ub_inner = \
                sch[phony_out].split(phony_out.op.axis[2], factor=shape_x[2])
        elif ub_split_axis == 0:
            split_factor = c1_size
            res_ub_outer, res_ub_inner = \
                sch[phony_out].split(phony_out.op.axis[1], factor=split_factor)
        else:
            res_ub_outer, res_ub_inner = \
                sch[phony_out].split(res_block_inner, factor=split_factor)
    else:
        res_ub_outer, res_ub_inner = \
            sch[phony_out].split(phony_out.op.axis[ub_split_axis],
                                 factor=split_factor)

    is_general_db = is_update_v3 or not is_param_buffer_reuse
    if is_general_db:
        need_db = _in_update_need_double_buffer(shape_x, block_split_axis,
                                                block_split_inner_size,
                                                ub_split_axis,
                                                ub_split_inner)
    else:
        need_db = _need_double_buffer_for_param_buffer_reuse(
            shape_x, block_split_axis,
            block_split_inner_size,
            ub_split_axis,
            ub_split_inner)

    shape_c0 = 16
    shape_c1 = shape_x[1]
    if shape_c1*shape_c0 > max_ub_count:
        mean_compute_at_axis = res_ub_outer

    block = tvm.thread_axis("blockIdx.x")
    sch[phony_out].bind(fused_axis, block)

    for i in input_tensor_buffer_tensor_map:
        input_tensor_buffer = input_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[input_tensor_buffer].compute_at(sch[phony_out],
                                                mean_compute_at_axis)
        else:
            sch[input_tensor_buffer].compute_at(sch[phony_out], res_ub_outer)

    for i in mid_tensor_buffer_tensor_map:
        mid_tensor_buffer = mid_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[mid_tensor_buffer].compute_at(sch[phony_out],
                                              mean_compute_at_axis)
        else:
            sch[mid_tensor_buffer].compute_at(sch[phony_out], res_ub_outer)

    for i in mid_out_tensor_list:
        if i not in shape_x_tensor_list:
            sch[i].compute_at(sch[phony_out], mean_compute_at_axis)
        else:
            sch[i].compute_at(sch[phony_out], res_ub_outer)

    for i in mid_out_tensor_read_buffer_map:
        mid_out_buffer = mid_out_tensor_read_buffer_map[i]
        if i not in shape_x_tensor_list:
            sch[mid_out_buffer].compute_at(sch[phony_out],
                                           mean_compute_at_axis)
        else:
            sch[mid_out_buffer].compute_at(sch[phony_out], res_ub_outer)

    sch[phony_out_ub].compute_at(sch[phony_out], res_ub_outer)

    if need_db:
        x_input_ub = input_tensor_buffer_tensor_map[x_input]
        sch[x_input_ub].double_buffer()

    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")

    batch = shape_x[0]
    c1_size = shape_x[1]
    c0_size = shape_x[4]
    w_size = shape_x[3]

    vector_intr_with_boradcast_map = {
        "elewise_binary_mul": "vector_mul_with_broadcast",
        "elewise_binary_add": "vector_add_with_broadcast",
    }

    for i in mid_tensor_buffer_tensor_map:
        buffer_tensor = mid_tensor_buffer_tensor_map[i]
        if i not in phony_tensor_list:
            if buffer_tensor in input_broadcast_tensor_buffers:
                shape = i.shape
                shape_size = reduce(lambda i, j: i * j, shape)
                # pylint: disable=too-many-boolean-expressions
                is_match = \
                    shape_size.value // (batch*c1_size*c0_size) == 1 or \
                    (ub_split_axis == 3 and split_factor == 1) or \
                    (ub_split_axis == 2 and split_factor == 1 and w_size == 1)
                if is_match:
                    insn = _get_emit_insn_map(i)
                else:
                    if i.op.tag.find("|") != -1:
                        str_list = i.op.tag.split("|")
                        tag = str_list[0]
                    else:
                        tag = i.op.tag

                    if tag in ["elewise_binary_mul", "elewise_binary_add"]:
                        insn = vector_intr_with_boradcast_map[tag]
                    elif tag == "elewise_single_cast":
                        insn = "phony_insn"
                    else:
                        insn = _get_emit_insn_map(i)
            else:
                insn = _get_emit_insn_map(i)

            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)
        else:
            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0],
                                         "phony_insn")
    index = 0

    for i in mid_out_tensor_list:
        sch[i].emit_insn(i.op.axis[0], "dma_copy")

        is_need_add_condition = not is_update_v3 and \
                                is_can_use_conditional_exec
        if is_need_add_condition:
            if index in (2, 3):
                condition = \
                    _get_update_condition(
                        cut_mode, block, core_num, shape_x,
                        block_split_axis, block_split_size)

                if condition:
                    sch[i].set_store_predicate(condition)
            index += 1

        if i in mid_out_tensor_read_buffer_map:
            phony_read_buffer = mid_out_tensor_read_buffer_map[i]
            sch[phony_read_buffer].emit_insn(phony_read_buffer.op.axis[0],
                                             "phony_insn")

    sch[phony_out_ub].emit_insn(phony_out_ub.op.axis[0], "phony_insn")
    sch[phony_out].emit_insn(res_ub_inner, "phony_insn")

    return sch


def _get_emit_insn_map(tensor):
    insn_map = {"elewise_single_cast": "vector_conv",
                "elewise_single_VS_max": "vector_maxs",
                "elewise_single_VS_min": "vector_mins",
                "elewise_single_log": "vector_ln",
                "elewise_single_exp": "vector_exp",
                "elewise_single_rec": "vector_rec",
                "elewise_single_relu": "vector_relu",
                "elewise_single_abs": "vector_abs",
                "elewise_single_not": "vector_not",
                "elewise_single_sqrt": "vector_sqrt",
                "elewise_single_rsqrt": "vector_rsqrt",
                "elewise_binary_mul": "vector_mul",
                "elewise_single_VS_mul": "vector_muls",
                "elewise_binary_div": "vector_div",
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
                "broadcast_for_tensor": "unified_broadcast",
                "elewise_multiple_mla": "vector_multiple",
                "elewise_multiple_madd": "vector_multiple",
                "elewise_multiple_maddrelu": "vector_multiple",
                "elewise_multiple_sel": "vector_select_bool",
                "emit_insn_elewise_binary_cmp": "elewise_binary_cmp",
                "elewise_binary_sub": "vector_sub"}
    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn
