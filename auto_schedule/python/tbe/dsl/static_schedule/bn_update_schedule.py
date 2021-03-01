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
batch_normalization_forward_training_update
"""
# pylint: import-error, unused-import, ungrouped-imports
from __future__ import absolute_import
from __future__ import division
import functools
from math import sqrt

import te.lang.cce
from tbe import tvm
from te import platform as cceconf
from te.platform.cce_conf import CceProductParams as pver
import te.platform.cce_params as cce
from .util import get_nearest_factor
from .util import DTYPE_WIDTH_MAP
from te.platform import log

MAX_SHAPE_NUM = 10000000
BN_TYPE = 0
IN_TYPE = 1
GN_TYPE = 2

FORMAT_NCHW = "NCHW"
FORMAT_NHWC = "NHWC"


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
        total_width_other = 7
        total_width_cloud = 3
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


def bn_update_schedule_model_parallel(  # pylint: disable=R0912, R0913, R0914, R0915
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
                shape_size = functools.reduce(lambda i, j: i * j, shape)
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
                          shape,  # pylint: disable=too-many-statements
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
        if c1_size < core_num:
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


def _gn_update_block_more_than_max_ub_do_compute_at(
        sch, res_out, res_ub_outer, input_tensor_buffer_tensor_map,
        mid_tensor_buffer_tensor_map, mid_out_tensor_list,
        mid_out_tensor_read_buffer_map):

    for i in input_tensor_buffer_tensor_map:
        input_tensor_buffer = input_tensor_buffer_tensor_map[i]
        sch[input_tensor_buffer].compute_at(sch[res_out], res_ub_outer)

    for i in mid_tensor_buffer_tensor_map:
        mid_tensor_buffer = mid_tensor_buffer_tensor_map[i]
        sch[mid_tensor_buffer].compute_at(sch[res_out], res_ub_outer)

    for i in mid_out_tensor_list:
        sch[i].compute_at(sch[res_out], res_ub_outer)
    for i in mid_out_tensor_read_buffer_map:
        mid_out_buffer = mid_out_tensor_read_buffer_map[i]
        sch[mid_out_buffer].compute_at(sch[res_out], res_ub_outer)


def _gn_update_sch_norm_do_compute_at(
        sch, res_out, mean_compute_at_axis, res_ub_outer,
        input_tensor_buffer_tensor_map, shape_x_tensor_list,
        mid_tensor_buffer_tensor_map, mid_out_tensor_list,
        mid_out_tensor_read_buffer_map,
        is_block_conflict, shape_x, group_nums, format_input):
    need_buffer_tile_tensors = []

    for i in input_tensor_buffer_tensor_map:
        input_tensor_buffer = input_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[input_tensor_buffer].compute_at(sch[res_out],
                                                mean_compute_at_axis)
            need_buffer_tile_tensors.append(input_tensor_buffer)
        else:
            sch[input_tensor_buffer].compute_at(sch[res_out], res_ub_outer)

    for i in mid_tensor_buffer_tensor_map:
        mid_tensor_buffer = mid_tensor_buffer_tensor_map[i]
        if i not in shape_x_tensor_list:
            sch[mid_tensor_buffer].compute_at(sch[res_out],
                                              mean_compute_at_axis)
            need_buffer_tile_tensors.append(mid_tensor_buffer)
        else:
            sch[mid_tensor_buffer].compute_at(sch[res_out], res_ub_outer)

    for i in mid_out_tensor_list:
        if i not in shape_x_tensor_list:
            sch[i].compute_at(sch[res_out], mean_compute_at_axis)
            need_buffer_tile_tensors.append(i)
        else:
            sch[i].compute_at(sch[res_out], res_ub_outer)

    for i in mid_out_tensor_read_buffer_map:
        mid_out_buffer = mid_out_tensor_read_buffer_map[i]
        if i not in shape_x_tensor_list:
            sch[mid_out_buffer].compute_at(sch[res_out],
                                           mean_compute_at_axis)
            need_buffer_tile_tensors.append(mid_out_buffer)
        else:
            sch[mid_out_buffer].compute_at(sch[res_out], res_ub_outer)

    def _do_buffer_tile():
        if is_block_conflict:
            tile_values = []
            batch_size = shape_x[0]
            # need do buffer_tile
            if format_input == FORMAT_NCHW:
                tile_values = \
                    ([0, batch_size], [0, group_nums],
                    [None, None], [None, None], [None, None])
            else:
                tile_values = \
                    ([0, batch_size], [None, None], [None, None],
                    [0, group_nums], [None, None])

            for tensor in need_buffer_tile_tensors:
                sch[tensor].buffer_tile(*tile_values)

    _do_buffer_tile()


def _gn_update_sch_do_compute_at(
        sch, res_out, res_out_ub, mean_compute_at_axis, res_ub_outer,
        input_tensor_buffer_tensor_map, shape_x_tensor_list,
        mid_tensor_buffer_tensor_map, mid_out_tensor_list,
        mid_out_tensor_read_buffer_map, one_block_more_than_max_ub,
        is_block_conflict, shape_x, group_nums, format_input):
    """
    gn_update schedule do compute_at
    """
    if one_block_more_than_max_ub is True:
        _gn_update_block_more_than_max_ub_do_compute_at(
            sch, res_out, res_ub_outer,
            input_tensor_buffer_tensor_map,
            mid_tensor_buffer_tensor_map, mid_out_tensor_list,
            mid_out_tensor_read_buffer_map)
    else:
        _gn_update_sch_norm_do_compute_at(
            sch, res_out, mean_compute_at_axis, res_ub_outer,
            input_tensor_buffer_tensor_map, shape_x_tensor_list,
            mid_tensor_buffer_tensor_map, mid_out_tensor_list,
            mid_out_tensor_read_buffer_map,
            is_block_conflict, shape_x, group_nums, format_input)

    sch[res_out_ub].compute_at(sch[res_out], res_ub_outer)


def _gn_update_ub_do_tiling(res_out, ub_split_inner, ub_split_axis,
                            block_split_axis, block_split_inner_size,
                            res_block_inner, max_ub_count, one_block_size,
                            group_nums, sch, shape_x):
    one_block_more_than_max_ub = False
    split_factor = ub_split_inner
    is_change_split_axis = False
    if ub_split_axis == block_split_axis:
        if block_split_inner_size % split_factor != 0:
            while block_split_inner_size % split_factor != 0:
                split_factor -= 1

        is_need_split_next = ub_split_axis == 1 and split_factor > 1
        if is_need_split_next:
            res_ub_outer, res_ub_inner = \
                sch[res_out].split(res_out.op.axis[2], factor=shape_x[2])
            is_change_split_axis = True
        elif ub_split_axis == 0:
            if one_block_size > max_ub_count:
                res_ub_outer, res_ub_inner = \
                    sch[res_out].split(res_block_inner, factor=split_factor)
                one_block_more_than_max_ub = True
            else:
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
    return one_block_more_than_max_ub, res_ub_outer,\
           res_ub_inner, is_change_split_axis


def _get_gn_update_group_nums(shape_x, shape_sum):
    """
    get gn_update group_nums
    """
    format_input = FORMAT_NCHW
    if len(shape_x) == len(shape_sum):
        if shape_x[1] == shape_sum[1] and shape_sum[3] == 1:
            group_nums = shape_x[1]
        elif shape_x[3] == shape_sum[3] and shape_sum[1] == 1:
            group_nums = shape_x[3]
            format_input = FORMAT_NHWC
    else:
        if shape_x[1] == shape_sum[1]:
            group_nums = shape_x[1]
        elif shape_x[3] == shape_sum[1]:
            group_nums = shape_x[3]
            format_input = FORMAT_NHWC

    return group_nums, format_input


def _check_is_block_conflict(block_axis, block_factor,
                             ub_axis, ub_factor, group_nums,
                             format_input, dtype):
    def _check_block_axis_zero():
        if ub_axis == block_axis:
            if ub_factor * group_nums * \
                    DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                return True
        elif ub_axis > block_axis:
            if block_factor * group_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                return True
        return False

    is_block_conflict = False
    if block_axis == 0:
        is_block_conflict = _check_block_axis_zero()
    else:
        if format_input == FORMAT_NCHW:
            if block_axis == 1:
                if block_factor * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                    is_block_conflict = True
            elif block_axis >= 2:
                is_block_conflict = True
        else:
            if block_axis in [1, 2]:
                if group_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                    is_block_conflict = True
            elif block_axis == 3:
                if block_factor * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                    is_block_conflict = True

    return is_block_conflict


def _recal_block_tiling(
        shape_x, block_axis, block_factor, max_ub_count,
        group_nums, format_input, dtype,
        core_num):
    """
    when block conflict, modify block tiling
    [N, G1, G0, H, W]  cut G0/H/W
    [N, H, W, G1, G0]  cut H/W
    """
    n_size = shape_x[0]
    one_core_data_threshold = 32

    def _find_cut_mid_axis_tiling():
        is_find_tiling = False
        split_axis = None
        block_inner = None
        if format_input == FORMAT_NCHW:
            temp_size = 1
            h_axis_index = 2
            for i in range(h_axis_index, len(shape_x)):
                if temp_size * shape_x[i] < core_num:
                    temp_size *= shape_x[i]
                    continue

                split_axis = i
                tmp = (core_num + temp_size - 1) // temp_size
                block_inner = (shape_x[i] + tmp - 1) // tmp

                one_core_size = block_inner
                for j in range(i + 1, len(shape_x), 1):
                    one_core_size *= shape_x[j]

                if one_core_size < one_core_data_threshold:
                    one_core_size = one_core_size // block_inner
                    if one_core_size * shape_x[i] > one_core_data_threshold:
                        for k in range(block_inner + 1, shape_x[i] + 1):
                            if one_core_size  * k < one_core_data_threshold:
                                continue
                            block_inner = k
                            break
                    else:
                        return None, None, is_find_tiling

                is_find_tiling = True
                break
        else:
            h_axis_index = 1
            num_group_axis_index = 3
            temp_size = 1
            for i in range(h_axis_index, num_group_axis_index):
                if temp_size * shape_x[i] < core_num:
                    temp_size *= shape_x[i]
                    continue

                split_axis = i
                tmp = core_num // temp_size
                block_inner = (shape_x[i] + tmp - 1) // tmp

                one_core_size = block_inner
                for j in range(i + 1, len(shape_x), 1):
                    one_core_size *= shape_x[j]

                if one_core_size < one_core_data_threshold:
                    return None, None, False

                is_find_tiling = True
                break

        return split_axis, block_inner, is_find_tiling

    new_block_axis, new_block_factor, is_find = \
        _find_cut_mid_axis_tiling()

    if is_find:
        return new_block_axis, new_block_factor, True

    new_block_axis = block_axis
    new_block_factor = block_factor

    def _get_new_block_factor():
        new_block_axis = 1
        new_block_factor = group_nums
        is_find_new_factor = False
        for i in range(block_factor + 1, group_nums, 1):
            if group_nums % i != 0:
                continue
            if i * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                continue
            new_block_factor = i
            is_find_new_factor = True
            break

        if not is_find_new_factor:
            new_block_axis = 0
            new_block_factor = n_size
            # split aixs 0
            for i in range(1, n_size, 1):
                if n_size % i != 0:
                    continue
                if i * group_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                    continue
                new_block_factor = i

        return new_block_axis, new_block_factor

    if block_axis == 0:
        new_block_factor = n_size
        for i in range(block_factor + 1, n_size, 1):
            if n_size % i != 0:
                continue
            if i * group_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                continue
            new_block_factor = i
            break
    else:
        if format_input == FORMAT_NCHW:
            if block_axis >= 1:
                new_block_axis, new_block_factor = _get_new_block_factor()
        else:
            if block_axis in [1, 2]:
                new_block_axis = 0
                new_block_factor = n_size

                for i in range(1, n_size, 1):
                    if i * group_nums * DTYPE_WIDTH_MAP[dtype] * 2 < 32:
                        continue
                    new_block_factor = i
                    break
            elif block_axis == 3:
                new_block_axis, new_block_factor = _get_new_block_factor()

    return new_block_axis, new_block_factor, False


def _cal_gn_update_tiling(shape_x, group_nums, format_input, max_ub_count):
    """
    calculate gn_update tiling
    """
    core_num = cceconf.get_soc_spec("CORE_NUM")

    one_core_data_threshold = 128

    def _get_block_tiling():
        block_inner = 1
        tmp_size = 1
        find_block_tiling = False
        for i, dim in enumerate(shape_x):
            tmp_size = tmp_size*dim
            if tmp_size < core_num:
                continue
            if tmp_size == core_num:
                block_split_axis = i
                block_inner = 1
                break

            tmp_size = tmp_size // dim
            for j in range(dim, 0, -1):
                if tmp_size*j > core_num:
                    continue
                block_split_axis = i
                block_inner = (dim + j - 1) // j

                remain_size = 1
                if len(shape_x[block_split_axis + 1:]) > 0:
                    remain_size = functools.reduce(lambda m, n: m * n,
                                         shape_x[block_split_axis + 1:])

                remain_size = remain_size * block_inner

                if remain_size < one_core_data_threshold:
                    remain_size = remain_size // block_inner
                    k = 0
                    for k in range(j, 0, -1):
                        if dim % k != 0:
                            continue
                        if remain_size*(dim // k) < one_core_data_threshold:
                            continue
                        block_inner = dim // k
                        break
                    if k == 0:
                        block_split_axis = 0 if block_split_axis == 0 \
                            else block_split_axis - 1
                        block_inner = 1

                find_block_tiling = True
                break

            if find_block_tiling:
                break

        return block_split_axis, block_inner

    def _get_ub_tiling(block_split_axis, block_inner):
        ub_split_axis = len(shape_x) - 1
        ub_inner = shape_x[-1]

        tmp_size = 1
        i = len(shape_x) - 1
        is_find_ub_tiling = False
        for i in range(len(shape_x) - 1, block_split_axis, -1):
            if tmp_size*shape_x[i] < max_ub_count:
                tmp_size *= shape_x[i]
                continue
            for j in range(shape_x[i], 0, -1):
                if j*tmp_size > max_ub_count:
                    continue
                is_find_ub_tiling = True
                ub_split_axis = i
                ub_inner = j
                break
            if is_find_ub_tiling:
                break

        if not is_find_ub_tiling:
            ub_split_axis = block_split_axis + 1

            for j in range(block_inner, 0, -1):
                if block_inner % j != 0:
                    continue
                if j*tmp_size > max_ub_count:
                    continue

                ub_split_axis = block_split_axis
                ub_inner = j
                break

        return ub_split_axis, ub_inner

    def _is_prime(num):
        for i in range(2, int(sqrt(num) + 1)):
            if num % i == 0:
                return False
        return True

    block_axis, block_factor = _get_block_tiling()

    ub_axis, ub_factor = _get_ub_tiling(block_axis, block_factor)

    dtype_mean = "float32"
    is_mean_output_block_conflict = \
        ub_axis == block_axis and ub_axis == 0 and ub_factor == 1 and \
        group_nums * DTYPE_WIDTH_MAP[dtype_mean] * 2 < 32 and \
        _is_prime(block_factor) and block_factor < shape_x[block_axis]
    if is_mean_output_block_conflict:
        block_factor += 1
        ub_axis, ub_factor = _get_ub_tiling(block_axis, block_factor)

    is_block_conflict = \
        _check_is_block_conflict(block_axis, block_factor, ub_axis,
                                 ub_factor, group_nums,
                                 format_input, dtype_mean)

    is_need_reorder = False

    return block_axis, block_factor, ub_axis, \
           ub_factor, is_block_conflict, is_need_reorder


def _gn_update_schedule_do_tiling(
        sch, shape_x, group_nums, format_input, dtype, res_out):
    """
    gn_update schedule do tiling
    """
    max_ub_count = get_max_ub_count(dtype, GN_TYPE)

    block_axis, block_factor, ub_axis, \
        ub_factor, is_block_conflict, is_need_reorder = \
        _cal_gn_update_tiling(shape_x, group_nums, format_input, max_ub_count)

    res_block_outer, res_block_inner = sch[res_out].split(
        res_out.op.axis[block_axis], factor=block_factor)

    def _do_axis_fuse():
        new_fused_axis = res_block_outer
        if is_need_reorder:
            if format_input == FORMAT_NCHW:
                h_axis_index = 2
                if block_axis > h_axis_index:
                    fuse_axis_list = res_out.op.axis[h_axis_index:block_axis]
                    fuse_axis_list.append(res_block_outer)
                    new_fused_axis = sch[res_out].fuse(*fuse_axis_list)
            else:
                h_axis_index = 1
                if block_axis > h_axis_index:
                    fuse_axis_list = \
                        [res_out.op.axis[h_axis_index], res_block_outer]
                    new_fused_axis = sch[res_out].fuse(*fuse_axis_list)
        else:
            if block_axis != 0:
                fuse_axis_list = res_out.op.axis[:block_axis]
                fuse_axis_list.append(res_block_outer)
                new_fused_axis = sch[res_out].fuse(*fuse_axis_list)

        return new_fused_axis

    fused_axis = _do_axis_fuse()

    one_block_size = block_factor
    if len(shape_x[block_axis + 1:]) > 0:
        one_block_size *= \
            functools.reduce(lambda m, n: m * n, shape_x[block_axis + 1:])

    one_block_more_than_max_ub, res_ub_outer,\
    res_ub_inner, is_change_split_axis = \
        _gn_update_ub_do_tiling(res_out, ub_factor, ub_axis,
                                block_axis, block_factor,
                                res_block_inner, max_ub_count, one_block_size,
                                group_nums, sch, shape_x)

    need_db = True

    bind_block_axis = fused_axis
    mean_compute_at_axis = fused_axis

    def _do_reorder():
        reorder_axis_list = [fused_axis, ]
        if format_input == FORMAT_NCHW:
            h_axis_index = 2
        else:
            h_axis_index = 1

        reorder_axis_list += res_out.op.axis[:h_axis_index]
        if block_axis == ub_axis:
            if is_change_split_axis:
                reorder_axis_list.append(res_block_inner)
                reorder_axis_list += \
                    res_out.op.axis[block_axis + 1: ub_axis + 1]
                reorder_axis_list.append(res_ub_outer)
                reorder_axis_list.append(res_ub_inner)
                reorder_axis_list += res_out.op.axis[ub_axis + 2:]
            else:
                reorder_axis_list.append(res_ub_outer)
                reorder_axis_list.append(res_ub_inner)
                reorder_axis_list += res_out.op.axis[block_axis + 1:]
        else:
            reorder_axis_list.append(res_block_inner)
            reorder_axis_list += res_out.op.axis[block_axis + 1: ub_axis]
            reorder_axis_list.append(res_ub_outer)
            reorder_axis_list.append(res_ub_inner)
            reorder_axis_list += res_out.op.axis[ub_axis + 1:]

        sch[res_out].reorder(*reorder_axis_list)

    if is_need_reorder:
        _do_reorder()
    else:
        if is_block_conflict:
            mean_compute_at_axis, bind_block_axis = sch[res_out].split(
                fused_axis, nparts=1)

    return mean_compute_at_axis, bind_block_axis, res_ub_outer, \
        res_ub_inner, need_db, one_block_more_than_max_ub, is_block_conflict


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

    sum_input = input_tensors[0]
    shape_sum = te.lang.cce.util.shape_to_list(sum_input.shape)

    group_nums, format_input = \
        _get_gn_update_group_nums(shape_x, shape_sum)

    mean_compute_at_axis, block_axis, res_ub_outer, \
        res_ub_inner, need_db, one_block_more_than_max_ub, \
        is_block_conflict = \
        _gn_update_schedule_do_tiling(
            sch, shape_x, group_nums, format_input, dtype, res_out)

    _gn_update_sch_do_compute_at(
        sch, res_out, res_out_ub, mean_compute_at_axis, res_ub_outer,
        input_tensor_buffer_tensor_map, shape_x_tensor_list,
        mid_tensor_buffer_tensor_map, mid_out_tensor_list,
        mid_out_tensor_read_buffer_map, one_block_more_than_max_ub,
        is_block_conflict, shape_x, group_nums, format_input)

    _gn_update_schedule_do_emit_insn(
        sch, res_out, res_out_ub, res_ub_inner,
        input_tensor_buffer_tensor_map, mid_tensor_buffer_tensor_map,
        mid_out_tensor_list, mid_out_tensor_read_buffer_map)

    block = tvm.thread_axis("blockIdx.x")
    sch[res_out].bind(block_axis, block)

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
        "elewise_binary_div": "vector_div_with_broadcast",
        "elewise_binary_sub": "vector_sub_with_broadcast"
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
                shape_size = functools.reduce(lambda m, n: m * n, shape)

                def get_insn_tensor_map(
                        shape_size, batch, c1_size, c0_size,
                        ub_split_axis, split_factor, w_size):
                    is_no_reduce = shape_size.value // \
                        (batch * c1_size * c0_size) == 1
                    if is_no_reduce or \
                        (ub_split_axis == 3 and split_factor == 1) or \
                        (ub_split_axis == 2 and split_factor == 1 and
                         w_size == 1):
                        insn = _get_emit_insn_map(i)
                    else:
                        if i.op.tag.find("|") != -1:
                            str_list = i.op.tag.split("|")
                            tag = str_list[0]
                        else:
                            tag = i.op.tag

                        if tag in ["elewise_binary_mul",
                                   "elewise_binary_add",
                                   "elewise_binary_div",
                                   "elewise_binary_sub"]:
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


    def _check_is_in_infer(res):
        """
        check whether is in_infer pattern
        """
        stack = [res[0]]
        visited_list = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor.op.tag == "elewise_single_rsqrt":
                    return True

                if in_tensor not in visited_list:
                    stack.append(in_tensor)
        return False

    is_in_infer_pattern = _check_is_in_infer(res)

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
        if (len(input_tensors) == 3 or len(input_tensors) == 5) and \
            not is_in_infer_pattern:
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
        if (len(input_tensors) == 3 or len(input_tensors) == 5) and \
            not is_in_infer_pattern:
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
        raise RuntimeError("res list size only support 5 or 6, "
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
            [phony_mask_cast, phony_mask_reduce,
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
        ((not is_param_buffer_reuse or
          is_can_use_conditional_exec) and
         h_size * w_size * c0_size > size_one_core_threshold)
    block_split_size = 1
    special_need_condition = False
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
    elif batch == 24 and c1_size < 24 <= core_num:
        cut_mode = "cut_batch"
        special_need_condition = True
        block_split_axis = 0
        res_block_outer, res_block_inner = sch[phony_out].split(phony_out.op.axis[0], nparts=24)
        block_split_inner_size = shape_x[block_split_axis] // 24
        fused_axis = res_block_outer
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
                shape_size = functools.reduce(lambda i, j: i * j, shape)
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

        is_need_add_condition = not is_update_v3 and (is_can_use_conditional_exec or special_need_condition)
        if is_need_add_condition:
            if index in (2, 3):
                if special_need_condition:
                    condition = \
                        _get_update_condition(
                            cut_mode, block, 24, shape_x,
                            block_split_axis, block_split_size)
                else:
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
