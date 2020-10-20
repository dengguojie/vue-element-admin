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
batch_normalization_forward_training_reduce
"""
from __future__ import absolute_import
from __future__ import division
from functools import reduce as functools_reduce
from te import tvm
import te.platform.cce_params as cce
from te import platform as cceconf
from te.platform import log
from .util import get_emit_insn_map
from .util import DTYPE_WIDTH_MAP


def get_max_ub_count(cast_dtype, reduce_len):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    # div 2 for align to fp16
    total_size = cceconf.get_soc_spec("UB_SIZE") // 2
    dtype_size = DTYPE_WIDTH_MAP.get(cast_dtype)
    total_size = total_size // dtype_size
    # div 2 for double buffer
    total_size //= 2
    reduce_width = 5.0 + 3.0/reduce_len
    elewise_width = 10.0/reduce_len
    if reduce_len >= 2:
        total_width = reduce_width
    else:
        total_width = elewise_width

    if not total_width:
        raise RuntimeError("Can not calculate with no compute")
    align_to = 128

    max_bound = total_width * align_to
    max_ub_count = int(total_size // max_bound * align_to)

    return max_ub_count


def _get_block_tiling(shape, one_core_data_threadhold):
    """
    find block tiling
    """
    core_num = cceconf.get_soc_spec("CORE_NUM")
    block_axis = 0
    block_factor = 1

    dim = shape[0]
    tmp_size = shape[0]
    if tmp_size <= core_num:
        block_axis = 0
        block_factor = 1
    else:
        tmp_size = tmp_size // dim
        for j in range(dim, 0, -1):
            if dim % j != 0:
                continue
            if tmp_size*j > core_num:
                continue
            block_axis = 0
            block_factor = dim // j

            remain_size = functools_reduce(lambda x, y: x * y,
                                           shape[block_axis + 1:])

            remain_size = remain_size * block_factor

            if remain_size < one_core_data_threadhold:
                remain_size = remain_size // block_factor
                k = 0
                for k in range(j, 0, -1):
                    if dim % k != 0:
                        continue
                    if remain_size*(dim // k) < one_core_data_threadhold:
                        continue
                    block_factor = dim // k
                    break
                if k == 1:
                    block_axis = 0 if block_axis == 0 else block_axis - 1
                    block_factor = 1
            break

    return block_axis, block_factor


def _get_ub_tiling(shape, block_axis, block_factor, max_ub_count):
    """
    find ub tiling
    """
    tmp_size = 1
    find_ub_tiling = False
    ub_axis = block_axis
    ub_factor = block_factor

    for i in range(len(shape) - 1, block_axis, -1):
        tmp_size = tmp_size*shape[i]
        if tmp_size < max_ub_count:
            continue
        if tmp_size == max_ub_count:
            ub_axis = i
            ub_factor = shape[i]
            break
        dim = shape[i]
        tmp_size = tmp_size // dim
        for j in range(dim, 0, -1):
            if dim % j != 0:
                continue
            if tmp_size*j > max_ub_count:
                continue
            ub_axis = i
            ub_factor = j
            find_ub_tiling = True
            break
        if find_ub_tiling:
            break

    if not find_ub_tiling:
        ub_axis = block_axis
        block_inner = block_factor
        for j in range(block_inner, 0, -1):
            if block_inner % j != 0:
                continue
            if tmp_size*j > max_ub_count:
                continue
            ub_factor = j
            break
    if len(shape) == 2 and ub_axis == 0 and ub_factor < 16:
        ub_axis = 1
        ub_factor = shape[-1]

    return ub_axis, ub_factor


def _get_tiling(cast_dtype, x_shape, is_reduce_output):
    """
    get tiling
    """
    max_ub_count = get_max_ub_count(cast_dtype, x_shape[1])
    one_core_data_threadhold = 1024

    total_size = 1
    for i in x_shape:
        total_size *= i
    # if data size too small, only use one core
    if total_size < one_core_data_threadhold:
        return 0, x_shape[0], 0, x_shape[0]

    block_axis, block_factor = \
        _get_block_tiling(x_shape, one_core_data_threadhold)

    if not is_reduce_output:
        one_core_res_nums = block_factor
        if len(x_shape) >= 3:
            for i in range(2, len(x_shape)):
                one_core_res_nums *= x_shape[i]

        if one_core_res_nums*DTYPE_WIDTH_MAP[cast_dtype]*2 < 32:
            block_axis = 0

            block_factor = x_shape[0]
            if cast_dtype == "float32":
                min_factor = 8
            else:
                min_factor = 16

            for i in range(min_factor, x_shape[0] + 1):
                if x_shape[0] % i == 0:
                    block_factor = i
                break

    remain_size = 1
    for i in range(len(x_shape) - 1, block_axis, -1):
        remain_size = remain_size*x_shape[i]

    remain_size = remain_size*block_factor
    if remain_size < one_core_data_threadhold:
        ub_axis = block_axis
        ub_factor = block_factor
        return block_axis, block_factor, ub_axis, ub_factor

    ub_axis, ub_factor = \
        _get_ub_tiling(x_shape, block_axis, block_factor, max_ub_count)

    return block_axis, block_factor, ub_axis, ub_factor


def _get_block_inner_tiling(cast_dtype, x_shape, block_factor):
    """
    some case, need cut block inner
    """
    max_ub_count = get_max_ub_count(cast_dtype, x_shape[1])

    block_inner_factor = block_factor
    tmp_size = 1
    if len(x_shape) == 3:
        tmp_size = tmp_size * x_shape[-1]

    block_inner = block_factor
    for j in range(block_inner, 0, -1):
        if block_inner % j != 0:
            continue
        if tmp_size*j > max_ub_count:
            continue
        block_inner_factor = j
        break
    return block_inner_factor


def _cut_no_reduce_axis_sch_do_cache(
        sch, res_tensor, is_reduce_output,
        input_tensor_dst_tensor_map,
        mid_tensor_dst_tensor_map,
        input_tensor_buffer_tensor_map,
        mid_tensor_buffer_tensor_map,
        double_buffer_tensors):
    """
    _cut_no_reduce_axis_schedule do
    cache_read/write and compute_inlie
    """
    # cache_read/write
    for key in input_tensor_dst_tensor_map:
        read_buffer = sch.cache_read(key, cce.scope_ubuf,
                                     input_tensor_dst_tensor_map[key])
        input_tensor_buffer_tensor_map[key] = read_buffer

        double_buffer_tensors.append(read_buffer)

    for key in mid_tensor_dst_tensor_map:
        write_buffer = sch.cache_write(key, cce.scope_ubuf)
        mid_tensor_buffer_tensor_map[key] = write_buffer

    if not is_reduce_output:
        write_buffer = sch.cache_write(res_tensor, cce.scope_ubuf)
        mid_tensor_buffer_tensor_map[res_tensor] = write_buffer

    for key in mid_tensor_dst_tensor_map:
        sch[key].compute_inline()


def _cut_no_reduce_axis_sch_do_tiling(
        sch, res_tensor, is_reduce_output, shape_x,
        block_axis, block_factor, ub_factor):
    """
    _cut_no_reduce_axis_schedule do tiling
    """
    barrier_tensor = res_tensor
    res_tensor_gm = None
    res_tensor_rf_ub = None
    if is_reduce_output:
        res_outer, _ = sch[res_tensor].split(
            res_tensor.op.reduce_axis[0], factor=block_factor)
        res_tensor_rf = sch.rfactor(res_tensor, res_outer)
        res_tensor_gm = sch.cache_write(res_tensor, "global")
        res_tensor_rf_ub = sch.cache_write(res_tensor_rf, cce.scope_ubuf)
        sch[res_tensor_rf].compute_inline()
        barrier_tensor = res_tensor_rf_ub
        compute_at_axis = res_tensor_rf_ub.op.axis[0]
        res_emit_insn_axis = res_tensor_rf_ub.op.axis[1]
        bind_block_axis = res_tensor_gm.op.reduce_axis[0]
        bind_block_tensor = res_tensor_gm
        if ub_factor != 1:
            rf_ub_outer, rf_ub_inner = \
                sch[res_tensor_rf_ub].split(
                    res_tensor_rf_ub.op.reduce_axis[len(shape_x)-2],
                    factor=ub_factor)
            if len(shape_x) == 3:
                sch[res_tensor_rf_ub].reorder(
                    res_tensor_rf_ub.op.axis[0],
                    rf_ub_outer, rf_ub_inner,
                    res_tensor_rf_ub.op.reduce_axis[0]
                )
            compute_at_axis = rf_ub_outer
            res_emit_insn_axis = rf_ub_inner
    else:
        # split compute_at
        res_block_outer, res_block_inner = \
            sch[res_tensor].split(sch[res_tensor].op.axis[block_axis],
                                  factor=block_factor)
        bind_block_axis = res_block_outer
        bind_block_tensor = res_tensor
        res_ub_outer, res_ub_inner = \
            sch[res_tensor].split(res_block_inner, factor=ub_factor)

        compute_at_axis = res_ub_outer
        res_emit_insn_axis = res_ub_inner

    return barrier_tensor, compute_at_axis, res_emit_insn_axis, \
        res_tensor_gm, res_tensor_rf_ub, \
        bind_block_tensor, bind_block_axis


def _cut_no_reduce_axis_sch_do_compute_at(
        sch, barrier_tensor, compute_at_axis,
        input_tensor_buffer_tensor_map,
        mid_tensor_buffer_tensor_map):
    """
    cut_no_reduce_axis_schedule do compute_at
    """
    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].compute_at(sch[barrier_tensor], compute_at_axis)

    for i in mid_tensor_buffer_tensor_map:
        buffer_tensor = mid_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].compute_at(sch[barrier_tensor], compute_at_axis)


def _cut_no_reduce_axis_sch_do_emit_insn(
        sch, is_reduce_output, res_tensor,
        res_tensor_rf_ub, res_tensor_gm,
        res_emit_insn_axis,
        input_tensor_buffer_tensor_map,
        mid_tensor_buffer_tensor_map):
    """
    cut_no_reduce_axis_schedule do emit_insn
    """
    # emit_insn
    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")

    if is_reduce_output:
        sch[res_tensor_rf_ub].emit_insn(res_emit_insn_axis, "vector_reduce_sum")
        sch[res_tensor_gm].emit_insn(res_tensor_gm.op.axis[0], "dma_copy")
        sch[res_tensor].emit_insn(sch[res_tensor].op.axis[0], "phony_insn")
    else:
        sch[res_tensor].emit_insn(res_emit_insn_axis, "dma_copy")

    for i in mid_tensor_buffer_tensor_map:
        buffer_tensor = mid_tensor_buffer_tensor_map[i]
        insn = get_emit_insn_map(i)
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)


def cosine_embedding_loss_schedule_cut_no_reduce_axis(
        res, tensor_list_dst_tensor_map,
        reduce_tensor_map, is_reduce_output,
        tiling_params):
    """
    cosine_embedding_loss schedule
    """
    res_tensor = res[0]

    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}

    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        else:
            mid_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
    sch = tvm.create_schedule(res_tensor.op)

    double_buffer_tensors = []
    input_tensor_buffer_tensor_map = {}
    mid_tensor_buffer_tensor_map = {}

    _cut_no_reduce_axis_sch_do_cache(
        sch, res_tensor, is_reduce_output,
        input_tensor_dst_tensor_map,
        mid_tensor_dst_tensor_map,
        input_tensor_buffer_tensor_map,
        mid_tensor_buffer_tensor_map,
        double_buffer_tensors)

    reduce_input_tensor = list(reduce_tensor_map.keys())[0].op.input_tensors[0]
    shape_x = [i.value for i in reduce_input_tensor.shape]

    # tiling
    block_axis, block_factor, _, ub_factor = \
        tiling_params

    barrier_tensor, compute_at_axis, res_emit_insn_axis, \
        res_tensor_gm, res_tensor_rf_ub, \
        bind_block_tensor, bind_block_axis = \
        _cut_no_reduce_axis_sch_do_tiling(
            sch, res_tensor, is_reduce_output, shape_x,
            block_axis, block_factor, ub_factor)

    _cut_no_reduce_axis_sch_do_compute_at(
        sch, barrier_tensor, compute_at_axis,
        input_tensor_buffer_tensor_map,
        mid_tensor_buffer_tensor_map)

    _cut_no_reduce_axis_sch_do_emit_insn(
        sch, is_reduce_output, res_tensor,
        res_tensor_rf_ub, res_tensor_gm,
        res_emit_insn_axis,
        input_tensor_buffer_tensor_map,
        mid_tensor_buffer_tensor_map)

    if is_reduce_output:
        res[0] = res_tensor_gm

    # double buffer
    for tensor in double_buffer_tensors:
        sch[tensor].double_buffer()

    # bind
    block = tvm.thread_axis("blockIdx.x")
    sch[bind_block_tensor].bind(bind_block_axis, block)
    return sch, []


def _get_node_list(out_tensor, tensor_list):
    """
    traverse tensors by Depth-First-Search
    get out_tensor's all input tensors
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
                tensor_list.append(in_tensor)


def _cut_notfirst_axis_sch_do_workspace_tiling(
        sch, workspace_node_list, ws_ub_axis, ws_ub_factor,
        workspace_input_tensor_buffer_map,
        workspace_tensor_input_map,
        workspace_node_buffer_map,
        workspace_mid_tensor_buffer_map,
        workspace_node_emit_axis_map):
    """
    _cut_notfirst_axis_schedule do workspace tiling
    """
    for tensor in workspace_node_list:
        tensor_ub_outer, tensor_ub_inner = \
            sch[tensor].split(sch[tensor].op.axis[ws_ub_axis],
                              factor=ws_ub_factor)

        workspace_node_emit_axis_map[tensor] = tensor_ub_inner

        buffer_tensor = workspace_input_tensor_buffer_map[tensor]
        sch[buffer_tensor].compute_at(sch[tensor], tensor_ub_outer)
        for i in workspace_tensor_input_map[tensor]:
            if not isinstance(i.op, tvm.tensor.PlaceholderOp):
                buffer_tensor = workspace_mid_tensor_buffer_map[i]
                sch[buffer_tensor].compute_at(sch[tensor], tensor_ub_outer)

        buffer_tensor = workspace_node_buffer_map[tensor]
        sch[buffer_tensor].compute_at(sch[tensor], tensor_ub_outer)


def _cut_notfirst_axis_sch_do_reduce_tiling(
        sch, shape_x, reduce_ub_factor, ub_axis, last_axis_ub_factor,
        reduce_tensor_map, mid_tensor_buffer_map,
        reduce_tensor_input_map,
        reduce_tensor_placeholder_map,
        reduce_tensor_placeholder_buffer_map,
        reduce_tensor_buffer_emit_axis_map):
    """
    _cut_notfirst_axis_schedule do reduce tiling
    """
    # the tensor before reduce node compute_at to reduce node
    for reduce_tensor in reduce_tensor_map:
        reduce_buffer = mid_tensor_buffer_map[reduce_tensor]

        reduce_ub_outer, reduce_ub_inner = \
            sch[reduce_buffer].split(sch[reduce_buffer].op.reduce_axis[0],
                                     factor=reduce_ub_factor)

        reorder_axis_list = []
        if len(shape_x) >= 3:
            split_axis = ub_axis - 1
            if ub_axis == 1:
                split_axis = 1
            last_axis_ub_outer, last_axis_ub_inner = \
                sch[reduce_buffer].split(sch[reduce_buffer].op.axis[split_axis],
                                         factor=last_axis_ub_factor)

            reorder_axis_list.append(reduce_buffer.op.axis[0])
            reorder_axis_list += reduce_buffer.op.axis[1:split_axis]
            reorder_axis_list.append(last_axis_ub_outer)
            reorder_axis_list.append(reduce_ub_outer)
            reorder_axis_list.append(reduce_ub_inner)
            reorder_axis_list.append(last_axis_ub_inner)
            reorder_axis_list += reduce_buffer.op.axis[split_axis + 1:]
        else:
            reorder_axis_list.append(reduce_buffer.op.axis[0])
            reorder_axis_list.append(reduce_ub_outer)
            reorder_axis_list.append(reduce_ub_inner)
            reorder_axis_list += reduce_buffer.op.axis[1:]

        sch[reduce_buffer].reorder(*reorder_axis_list)

        reduce_tensor_buffer_emit_axis_map[reduce_buffer] = reduce_ub_inner

        for buffer_tensor in reduce_tensor_placeholder_buffer_map[reduce_tensor]:
            sch[buffer_tensor].compute_at(sch[reduce_buffer], reduce_ub_outer)

        for tensor in reduce_tensor_input_map[reduce_tensor]:
            if tensor not in reduce_tensor_placeholder_map[reduce_tensor]:
                buffer_tensor = mid_tensor_buffer_map[tensor]
                sch[buffer_tensor].compute_at(sch[reduce_buffer], reduce_ub_outer)


def _cut_notfirst_axis_sch_do_tiling(
        sch, res_tensor, is_reduce_output, shape_x,
        block_axis, block_factor, ub_axis, ub_factor,
        block_inner_factor, workspace_node_list,
        workspace_input_tensor_buffer_map, workspace_tensor_input_map,
        workspace_mid_tensor_buffer_map, workspace_node_buffer_map,
        reduce_tensor_map, mid_tensor_buffer_map,
        reduce_tensor_placeholder_map,
        reduce_tensor_placeholder_buffer_map, reduce_tensor_input_map):
    """
    cut_notfirst_axis_schedule do tiling
    """
    barrier_tensor = res_tensor
    res_tensor_gm = None
    res_tensor_rf_ub = None
    res_ub_outer_size = 1
    if is_reduce_output:
        res_outer, _ = sch[res_tensor].split(
            res_tensor.op.reduce_axis[0], factor=block_factor)

        res_tensor_rf = sch.rfactor(res_tensor, res_outer)

        res_tensor_gm = sch.cache_write(res_tensor, "global")
        res_tensor_rf_ub = sch.cache_write(res_tensor_rf, cce.scope_ubuf)
        sch[res_tensor_rf].compute_inline()

        barrier_tensor = res_tensor_rf_ub
        bind_block_axis = res_tensor_gm.op.reduce_axis[0]
        bind_block_tensor = res_tensor_gm

        sch[res_tensor_gm].reorder(
            *(res_tensor_gm.op.reduce_axis[:] +
              res_tensor_gm.op.axis[:]))

        rf_ub_reduce_axis = [res_tensor_rf_ub.op.reduce_axis[-1]] + \
            res_tensor_rf_ub.op.reduce_axis[0:-1]

        res_block_inner_outer, res_block_inner_inner = \
            sch[res_tensor_rf_ub].split(rf_ub_reduce_axis[0],
                                        factor=block_inner_factor)

        reorder_axis_list = []
        if len(shape_x) >= 3:
            split_axis = ub_axis - 1
            res_ub_factor = ub_factor
            res_ub_outer_size = \
                (shape_x[ub_axis] + ub_factor - 1) // ub_factor
            if ub_axis == 1:
                split_axis = 1
                res_ub_factor = shape_x[2]
                res_ub_outer_size = block_factor

            rf_ub_outer, rf_ub_inner = \
                sch[res_tensor_rf_ub].split(
                    rf_ub_reduce_axis[split_axis],
                    factor=res_ub_factor)

            reorder_axis_list.append(res_tensor_rf_ub.op.axis[0])
            reorder_axis_list.append(res_block_inner_outer)
            reorder_axis_list.append(res_block_inner_inner)
            reorder_axis_list += rf_ub_reduce_axis[1:split_axis]
            reorder_axis_list.append(rf_ub_outer)
            reorder_axis_list.append(rf_ub_inner)
            reorder_axis_list += rf_ub_reduce_axis[split_axis + 1:]
            sch[res_tensor_rf_ub].reorder(*reorder_axis_list)
            compute_at_axis = rf_ub_outer
            res_emit_insn_axis = rf_ub_inner
        else:
            reorder_axis_list.append(res_tensor_rf_ub.op.axis[0])
            reorder_axis_list.append(res_block_inner_outer)
            reorder_axis_list.append(res_block_inner_inner)
            reorder_axis_list += res_tensor_rf_ub.op.reduce_axis[0:-1]

            sch[res_tensor_rf_ub].reorder(*reorder_axis_list)
            compute_at_axis = res_block_inner_outer
            res_emit_insn_axis = res_block_inner_inner
            res_ub_outer_size = \
                (shape_x[block_axis] + block_factor - 1) // block_factor\
                // block_inner_factor
    else:
        # split compute_at
        res_block_outer, res_block_inner = \
            sch[res_tensor].split(sch[res_tensor].op.axis[block_axis],
                                  factor=block_factor)
        res_block_inner_outer, res_block_inner_inner = \
            sch[res_tensor].split(res_block_inner, factor=block_inner_factor)

        if len(shape_x) >= 3:
            split_axis = ub_axis - 1
            res_ub_factor = ub_factor
            if ub_axis == 1:
                split_axis = 1
                res_ub_factor = shape_x[2]

            res_ub_outer, res_ub_inner = \
                sch[res_tensor].split(
                    sch[res_tensor].op.axis[split_axis],
                    factor=res_ub_factor)
            compute_at_axis = res_ub_outer
            res_emit_insn_axis = res_ub_inner
        else:
            compute_at_axis = res_block_inner_outer
            res_emit_insn_axis = res_block_inner_inner

        bind_block_axis = res_block_outer
        bind_block_tensor = res_tensor

    if ub_axis < block_axis:
        raise RuntimeError("Invalid tiling!")

    if ub_axis >= 2:
        reduce_ub_factor = 1
        last_axis_ub_factor = ub_factor
    elif ub_axis == 1:
        reduce_ub_factor = ub_factor
        last_axis_ub_factor = 1
        if len(shape_x) > 2:
            last_axis_ub_factor = shape_x[2]
    else:
        raise RuntimeError("Invalid Tiling!")

    workspace_node_emit_axis_map = {}

    _cut_notfirst_axis_sch_do_workspace_tiling(
        sch, workspace_node_list, ub_axis, ub_factor,
        workspace_input_tensor_buffer_map,
        workspace_tensor_input_map,
        workspace_node_buffer_map,
        workspace_mid_tensor_buffer_map,
        workspace_node_emit_axis_map)

    reduce_tensor_buffer_emit_axis_map = {}

    _cut_notfirst_axis_sch_do_reduce_tiling(
        sch, shape_x, reduce_ub_factor, ub_axis, last_axis_ub_factor,
        reduce_tensor_map, mid_tensor_buffer_map,
        reduce_tensor_input_map,
        reduce_tensor_placeholder_map,
        reduce_tensor_placeholder_buffer_map,
        reduce_tensor_buffer_emit_axis_map)

    return barrier_tensor, compute_at_axis, res_emit_insn_axis, \
        res_tensor_gm, res_tensor_rf_ub, res_ub_outer_size, \
        bind_block_tensor, bind_block_axis, \
        workspace_node_emit_axis_map, \
        reduce_tensor_buffer_emit_axis_map


def _cut_notfirst_axis_sch_do_inline(
        sch, res_tensor,
        workspace_mid_tensor_buffer_map,
        mid_tensor_buffer_map):
    """
    _cut_notfirst_axis_schedule do cache_read/write
    """
    for key in workspace_mid_tensor_buffer_map:
        sch[key].compute_inline()

    for key in mid_tensor_buffer_map:
        if key != res_tensor:
            sch[key].compute_inline()


def _cut_notfirst_axis_sch_do_cache_read(
        sch, wk_tensor_placeholder_map,
        input_tensor_dst_tensor_map,
        reduce_tensor_placeholder_list,
        wk_tensor_placeholder_list,
        reduce_tensor_placeholder_map,
        tensor_list_dst_tensor_map,
        reduce_tensor_input_map):
    """
    _cut_notfirst_axis_schedule do cache_read
    """
    double_buffer_tensors = []
    workspace_input_tensor_buffer_map = {}
    for tensor in wk_tensor_placeholder_map:
        input_tensor = wk_tensor_placeholder_map[tensor]
        read_buffer = sch.cache_read(
            input_tensor, cce.scope_ubuf,
            input_tensor_dst_tensor_map[input_tensor])
        workspace_input_tensor_buffer_map[tensor] = read_buffer
        double_buffer_tensors.append(read_buffer)

    # cache_read/write
    input_tensor_buffer_tensor_map = {}
    for key in input_tensor_dst_tensor_map:
        if key not in reduce_tensor_placeholder_list +\
                wk_tensor_placeholder_list:
            read_buffer = sch.cache_read(
                key, cce.scope_ubuf,
                input_tensor_dst_tensor_map[key])
            input_tensor_buffer_tensor_map[key] = read_buffer
            double_buffer_tensors.append(read_buffer)

    # reduce_tensor's placeholder cache_read twice
    reduce_tensor_placeholder_buffer_map = {}
    for key in reduce_tensor_placeholder_map:
        placeholder_list = reduce_tensor_placeholder_map[key]
        reduce_tensor_placeholder_buffer_map[key] = []
        for tensor in placeholder_list:
            dst_tensor = []
            for i in tensor_list_dst_tensor_map[tensor]:
                if i in reduce_tensor_input_map[key]:
                    dst_tensor.append(i)
            read_buffer = sch.cache_read(tensor, cce.scope_ubuf,
                                         dst_tensor)
            reduce_tensor_placeholder_buffer_map[key].append(read_buffer)

            double_buffer_tensors.append(read_buffer)

    return double_buffer_tensors, workspace_input_tensor_buffer_map, \
        input_tensor_buffer_tensor_map, reduce_tensor_placeholder_buffer_map


def _cut_notfirst_axis_sch_do_cache_write(
        sch, res_tensor, is_reduce_output,
        workspace_node_list,
        workspace_tensor_input_list,
        mid_tensor_dst_tensor_map):
    """
    cut_notfirst_axis_schedule do cache_write
    """
    workspace_node_buffer_map = {}
    for key in workspace_node_list:
        write_buffer = sch.cache_write(key, cce.scope_ubuf)
        workspace_node_buffer_map[key] = write_buffer

    workspace_mid_tensor_buffer_map = {}
    for key in workspace_tensor_input_list:
        write_buffer = sch.cache_write(key, cce.scope_ubuf)
        workspace_mid_tensor_buffer_map[key] = write_buffer

    mid_tensor_buffer_map = {}
    for key in mid_tensor_dst_tensor_map:
        if key not in workspace_tensor_input_list + workspace_node_list:
            write_buffer = sch.cache_write(key, cce.scope_ubuf)
            mid_tensor_buffer_map[key] = write_buffer

    if not is_reduce_output:
        write_buffer = sch.cache_write(res_tensor, cce.scope_ubuf)
        mid_tensor_buffer_map[res_tensor] = write_buffer

    return workspace_node_buffer_map, workspace_mid_tensor_buffer_map, \
        mid_tensor_buffer_map


def _cut_notfirst_axis_sch_do_compute_at(
        sch, barrier_tensor, compute_at_axis,
        is_reduce_output, res_tensor_gm, res_tensor_rf_ub,
        workspace_node_list, input_tensor_buffer_tensor_map,
        reduce_tensor_input_list, reduce_tensor_map,
        mid_tensor_buffer_map):
    """
    cut_notfirst_axis_schedule do compute_at
    """
    for i in workspace_node_list:
        sch[i].compute_at(sch[barrier_tensor], compute_at_axis)

    for i in input_tensor_buffer_tensor_map:
        if i in reduce_tensor_input_list:
            continue
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].compute_at(sch[barrier_tensor], compute_at_axis)

    for i in reduce_tensor_map:
        if i in workspace_node_list:
            continue
        buffer_tensor = mid_tensor_buffer_map[i]
        sch[buffer_tensor].compute_at(sch[barrier_tensor], compute_at_axis)

    for i in mid_tensor_buffer_map:
        if i in reduce_tensor_input_list:
            continue
        buffer_tensor = mid_tensor_buffer_map[i]
        sch[buffer_tensor].compute_at(sch[barrier_tensor], compute_at_axis)

    if is_reduce_output:
        sch[res_tensor_rf_ub].compute_at(sch[res_tensor_gm],
                                         res_tensor_gm.op.reduce_axis[0])


def _cut_notfirst_axis_sch_do_emit_insn(
        sch, res, res_tensor, is_reduce_output, res_ub_outer_size,
        res_tensor_rf_ub, res_tensor_gm, res_emit_insn_axis,
        workspace_tensor_input_map, workspace_input_tensor_buffer_map,
        workspace_mid_tensor_buffer_map,
        workspace_node_list, workspace_node_emit_axis_map,
        input_tensor_buffer_tensor_map,
        reduce_tensor_placeholder_buffer_map,
        workspace_node_buffer_map, mid_tensor_buffer_map,
        reduce_tensor_map, reduce_tensor_buffer_emit_axis_map):
    """
    cut_notfirst_axis_schedule do emit_insn
    """
    # emit_insn
    for i in workspace_tensor_input_map:
        buffer_tensor = workspace_input_tensor_buffer_map[i]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")

    for i in workspace_mid_tensor_buffer_map:
        buffer_tensor = workspace_mid_tensor_buffer_map[i]
        insn = get_emit_insn_map(i)
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)

    for i in workspace_node_list:
        emit_axis = workspace_node_emit_axis_map[i]
        sch[i].emit_insn(emit_axis, "dma_copy")

    for i in input_tensor_buffer_tensor_map:
        buffer_tensor = input_tensor_buffer_tensor_map[i]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")

    for i in reduce_tensor_placeholder_buffer_map:
        for buffer_tensor in reduce_tensor_placeholder_buffer_map[i]:
            sch[buffer_tensor].emit_insn(
                buffer_tensor.op.axis[0], "dma_copy")

    for i in workspace_node_buffer_map:
        buffer_tensor = workspace_node_buffer_map[i]
        insn = get_emit_insn_map(i)
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)

    for i in mid_tensor_buffer_map:
        if i not in reduce_tensor_map:
            buffer_tensor = mid_tensor_buffer_map[i]
            insn = get_emit_insn_map(i)
            sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)

    for reduce_tensor in reduce_tensor_map:
        reduce_buffer = mid_tensor_buffer_map[reduce_tensor]
        emit_axis = reduce_tensor_buffer_emit_axis_map[reduce_buffer]
        sch[reduce_buffer].emit_insn(emit_axis, "vector_reduce_sum")

    if is_reduce_output:
        if res_ub_outer_size == 1:
            sch[res_tensor_rf_ub].emit_insn(res_emit_insn_axis, "vector_reduce_sum")
        else:
            # use vector_reduce_sum, res will be error
            sch[res_tensor_rf_ub].emit_insn(
                res_emit_insn_axis,
                "reduce_last_axis_reduce_sum")

        sch[res_tensor_gm].emit_insn(res_tensor_gm.op.axis[0], "dma_copy")
        sch[res_tensor].emit_insn(sch[res_tensor].op.axis[0], "phony_insn")

        res[0] = res_tensor_gm
    else:
        sch[res_tensor].emit_insn(res_emit_insn_axis, "dma_copy")


def _get_workspace(reduce_tensor_map):
    """
    get workspace node list
    """
    workspace_node_list = []

    for reduce_tensor in reduce_tensor_map:
        tensor_mul = reduce_tensor.op.input_tensors[0]
        # get broadcast or cast tensor, the tensor as workspace node
        if len(tensor_mul.op.input_tensors) == 2:
            for i in tensor_mul.op.input_tensors:
                if isinstance(i.op, tvm.tensor.PlaceholderOp):
                    continue
                if "broadcast" in i.op.tag:
                    workspace_node_list.append(i)
                elif "cast" in i.op.tag:
                    workspace_node_list.append(i)
            break

    return workspace_node_list


def _get_workspace_input_info(
        workspace_node_list,
        workspace_tensor_input_map,
        workspace_tensor_placeholder_map,
        workspace_tensor_placeholder_list,
        workspace_tensor_input_list):
    """
    get workspace node input tensor info
    """
    # get input node and tensor of workspace node
    for tensor in workspace_node_list:
        tensor_list_before_workspace = []
        _get_node_list(tensor, tensor_list_before_workspace)
        workspace_tensor_input_map[tensor] = \
            tensor_list_before_workspace

        for i in tensor_list_before_workspace:
            if isinstance(i.op, tvm.tensor.PlaceholderOp):
                workspace_tensor_placeholder_map[tensor] = i
                workspace_tensor_placeholder_list.append(i)
            else:
                workspace_tensor_input_list.append(i)


def _get_reduce_input_info_with_workspace(
        reduce_tensor_map,
        workspace_node_list,
        workspace_tensor_input_list,
        workspace_tensor_placeholder_list):
    """
    get reduce node input tensor info with workspace
    """
    reduce_tensor_input_map = {}
    reduce_tensor_placeholder_map = {}
    reduce_tensor_placeholder_list = []
    reduce_tensor_input_list = []

    for reduce_tensor in reduce_tensor_map:
        tensor_list_before_reduce = []
        _get_node_list(reduce_tensor, tensor_list_before_reduce)
        tensor_list_before_reduce = \
            [i for i in tensor_list_before_reduce
             if i not in workspace_tensor_input_list +
             workspace_tensor_placeholder_list]

        reduce_tensor_input_map[reduce_tensor] = \
            tensor_list_before_reduce

        reduce_tensor_placeholder_map[reduce_tensor] = []
        for tensor in tensor_list_before_reduce:
            if tensor in workspace_tensor_input_list + workspace_node_list:
                reduce_tensor_placeholder_map[reduce_tensor].append(tensor)
                if tensor not in reduce_tensor_placeholder_list:
                    reduce_tensor_placeholder_list.append(tensor)
            elif isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                if tensor not in workspace_tensor_placeholder_list:
                    reduce_tensor_placeholder_map[reduce_tensor].append(tensor)
                    if tensor not in reduce_tensor_placeholder_list:
                        reduce_tensor_placeholder_list.append(tensor)
            else:
                if tensor not in reduce_tensor_input_list:
                    reduce_tensor_input_list.append(tensor)

    return reduce_tensor_input_map, reduce_tensor_placeholder_map, \
        reduce_tensor_placeholder_list, reduce_tensor_input_list


def _get_reduce_input_info(
        reduce_tensor_map,
        is_need_workspace,
        workspace_node_list,
        workspace_tensor_input_list,
        workspace_tensor_placeholder_list):
    """
    get reduce node input tensor info
    """
    reduce_tensor_input_map = {}
    reduce_tensor_placeholder_map = {}
    reduce_tensor_placeholder_list = []
    reduce_tensor_input_list = []

    # get input node and tensor of reduce node
    if is_need_workspace:
        # if the workspace exist, reduce node's
        # input node will be workspace node
        reduce_tensor_input_map, reduce_tensor_placeholder_map, \
            reduce_tensor_placeholder_list, reduce_tensor_input_list = \
            _get_reduce_input_info_with_workspace(
                reduce_tensor_map, workspace_node_list,
                workspace_tensor_input_list,
                workspace_tensor_placeholder_list)
    else:
        for reduce_tensor in reduce_tensor_map:
            tensor_list_before_reduce = []
            _get_node_list(reduce_tensor, tensor_list_before_reduce)
            reduce_tensor_input_map[reduce_tensor] = \
                tensor_list_before_reduce

            reduce_tensor_placeholder_map[reduce_tensor] = []
            for tensor in tensor_list_before_reduce:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    reduce_tensor_placeholder_map[reduce_tensor].append(tensor)
                    if tensor not in reduce_tensor_placeholder_list:
                        reduce_tensor_placeholder_list.append(tensor)
                else:
                    if tensor not in reduce_tensor_input_list:
                        reduce_tensor_input_list.append(tensor)

    return reduce_tensor_input_map, reduce_tensor_placeholder_map, \
        reduce_tensor_placeholder_list, reduce_tensor_input_list


def cosine_embedding_loss_schedule_cut_notfirst_axis(
        res, tensor_list_dst_tensor_map,
        reduce_tensor_map, is_reduce_output,
        tiling_params):
    """
    Cosine_embedding_loss schedule for cut not first axis.
    Because the reduce axis is big, it is need to cut reduce axis,
    so the tensor that before reduce node need compute at to reduce node,
    the input node need cache_read twice, than can compute_at to different
    reduce node.
    When input_data1 and input_data2, it will use workspace.
    And the workspace node will cache_read twice.
    """
    res_tensor = res[0]

    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}
    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        else:
            mid_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]

    sch = tvm.create_schedule(res_tensor.op)

    is_need_workspace = False
    workspace_node_list = _get_workspace(reduce_tensor_map)

    workspace_tensor_input_map = {}
    workspace_tensor_input_list = []
    workspace_tensor_placeholder_list = []
    workspace_tensor_placeholder_map = {}

    if workspace_node_list:
        is_need_workspace = True
        _get_workspace_input_info(
            workspace_node_list,
            workspace_tensor_input_map,
            workspace_tensor_placeholder_map,
            workspace_tensor_placeholder_list,
            workspace_tensor_input_list)

    reduce_tensor_input_map, reduce_tensor_placeholder_map, \
        reduce_tensor_placeholder_list, reduce_tensor_input_list = \
        _get_reduce_input_info(
            reduce_tensor_map,
            is_need_workspace,
            workspace_node_list,
            workspace_tensor_input_list,
            workspace_tensor_placeholder_list)

    double_buffer_tensors, workspace_input_tensor_buffer_map, \
        input_tensor_buffer_tensor_map, \
        reduce_placeholder_buffer_map = \
        _cut_notfirst_axis_sch_do_cache_read(
            sch, workspace_tensor_placeholder_map,
            input_tensor_dst_tensor_map,
            reduce_tensor_placeholder_list,
            workspace_tensor_placeholder_list,
            reduce_tensor_placeholder_map,
            tensor_list_dst_tensor_map,
            reduce_tensor_input_map
        )

    workspace_node_buffer_map, workspace_mid_tensor_buffer_map, \
        mid_tensor_buffer_map = \
        _cut_notfirst_axis_sch_do_cache_write(
            sch, res_tensor, is_reduce_output,
            workspace_node_list,
            workspace_tensor_input_list,
            mid_tensor_dst_tensor_map
        )

    _cut_notfirst_axis_sch_do_inline(
        sch, res_tensor,
        workspace_mid_tensor_buffer_map,
        mid_tensor_buffer_map)

    reduce_input_tensor = list(reduce_tensor_map.keys())[0].op.input_tensors[0]
    shape_x = [i.value for i in reduce_input_tensor.shape]
    cast_dtype = reduce_input_tensor.dtype.lower()

    block_axis, block_factor, ub_axis, ub_factor = tiling_params

    block_inner_factor = \
        _get_block_inner_tiling(cast_dtype, shape_x, block_factor)

    barrier_tensor, compute_at_axis, res_emit_insn_axis, \
        res_tensor_gm, res_tensor_rf_ub, res_ub_outer_size, \
        bind_block_tensor, bind_block_axis, \
        workspace_node_emit_axis_map, \
        reduce_tensor_buffer_emit_axis_map = \
        _cut_notfirst_axis_sch_do_tiling(
            sch, res_tensor, is_reduce_output, shape_x,
            block_axis, block_factor, ub_axis, ub_factor,
            block_inner_factor, workspace_node_list,
            workspace_input_tensor_buffer_map, workspace_tensor_input_map,
            workspace_mid_tensor_buffer_map, workspace_node_buffer_map,
            reduce_tensor_map, mid_tensor_buffer_map, reduce_tensor_placeholder_map,
            reduce_placeholder_buffer_map, reduce_tensor_input_map
        )

    _cut_notfirst_axis_sch_do_compute_at(
        sch, barrier_tensor, compute_at_axis,
        is_reduce_output, res_tensor_gm, res_tensor_rf_ub,
        workspace_node_list, input_tensor_buffer_tensor_map,
        reduce_tensor_input_list, reduce_tensor_map,
        mid_tensor_buffer_map
    )

    _cut_notfirst_axis_sch_do_emit_insn(
        sch, res, res_tensor, is_reduce_output, res_ub_outer_size,
        res_tensor_rf_ub, res_tensor_gm, res_emit_insn_axis,
        workspace_tensor_input_map, workspace_input_tensor_buffer_map,
        workspace_mid_tensor_buffer_map,
        workspace_node_list, workspace_node_emit_axis_map,
        input_tensor_buffer_tensor_map,
        reduce_placeholder_buffer_map,
        workspace_node_buffer_map, mid_tensor_buffer_map,
        reduce_tensor_map, reduce_tensor_buffer_emit_axis_map
    )

    # double buffer
    for tensor in double_buffer_tensors:
        sch[tensor].double_buffer()

    # bind
    block = tvm.thread_axis("blockIdx.x")
    sch[bind_block_tensor].bind(bind_block_axis, block)

    sch.cce_special = dict()
    sch.cce_special["tensor_list"] = workspace_node_list

    return sch, workspace_node_list


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
                                reduce_tensor_map):
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
                if in_tensor.op.tag.find("reduce") != -1:
                    _map_apend(reduce_tensor_map, in_tensor, cur_tensor)

            _map_apend(tensor_list_dst_tensor_map, in_tensor, cur_tensor)


def _check_sch_support(dtype, shape_x, tiling_parmas):
    """
    check this sch whether support
    """
    dtype_byte_size = 4

    block_axis, block_inner, ub_axis, ub_inner = tiling_parmas

    reduce_outer_size = 1
    no_reduce_nums = 1
    if len(shape_x) == 2:
        # one time dma num is ub_inner*shape_x[1], is not 32B aligned,
        # need use scalar do align, inefficiency
        if ub_inner > 1 and shape_x[1]*dtype_byte_size % 32 != 0:
            return False
    else:
        if ub_axis == 0:
            for i in range(2, len(shape_x)):
                no_reduce_nums *= shape_x[i]
        elif ub_axis == 1:
            reduce_outer_size = shape_x[ub_axis] // ub_inner
            for i in range(2, len(shape_x)):
                no_reduce_nums *= shape_x[i]
        else:
            no_reduce_nums = ub_inner
            for i in range(ub_axis + 1, len(shape_x)):
                no_reduce_nums *= shape_x[i]

    one_core_reduce_nums = ub_inner
    tail_nums = shape_x[ub_axis] % ub_inner
    if ub_axis == 1:
        one_core_reduce_nums = 1
        tail_nums = 1

    for i in range(ub_axis + 1, len(shape_x)):
        one_core_reduce_nums *= shape_x[i]
        tail_nums *= shape_x[i]

    is_not_aligned = \
        reduce_outer_size == 1 and \
        no_reduce_nums*dtype_byte_size % 32 != 0
    if is_not_aligned:
        return False

    is_not_aligned = \
        reduce_outer_size > 1 and \
        (one_core_reduce_nums*dtype_byte_size % 32 != 0 or
         tail_nums*dtype_byte_size % 32 != 0)
    if is_not_aligned:
        return False

    return True


def cosine_embedding_loss_schedule(res, input_list):
    """
    cosine_embedding_loss schedule
    """
    log.debug("start cosine_embedding_loss_schedule")
    if len(input_list) != 3:
        raise RuntimeError("CosineEmbeddingLoss input nums should be 3.")

    if len(res) != 1:
        raise RuntimeError("CosineEmbeddingLoss input nums should be 1.")

    result_tensor = res[0]

    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}
    reduce_tensor_map = {}

    _gen_reversed_subgraph_list(result_tensor, tensor_list_map,
                                tensor_list_dst_tensor_map,
                                reduce_tensor_map)

    is_reduce_output = False
    if result_tensor.op.tag == "reduce_sum":
        is_reduce_output = True

    reduce_input_tensor = \
        list(reduce_tensor_map.keys())[0].op.input_tensors[0]
    shape_x = [i.value for i in reduce_input_tensor.shape]

    dtype = result_tensor.dtype.lower()
    dtype_size = DTYPE_WIDTH_MAP.get(dtype)

    block_axis, block_inner, ub_axis, ub_inner = \
        _get_tiling(dtype, shape_x, is_reduce_output)
    log.debug("cosine_embedding_loss_schedule tiling, \
              block_axis=%d, block_inner=%d, ub_axis=%d, ub_inner=%d",
              block_axis, block_inner, ub_axis, ub_inner)

    tiling_params = [block_axis, block_inner, ub_axis, ub_inner]

    is_support = _check_sch_support(dtype, shape_x, tiling_params)
    if not is_support:
        log.warn("Cosine_embedding_loss_schedule can't process!")
        return None, []

    if block_axis == 0 and ub_axis == 0:
        # cut first axis
        log.debug("cosine_embedding_loss_schedule cut_no_reduce_axis")
        return cosine_embedding_loss_schedule_cut_no_reduce_axis(
            res, tensor_list_dst_tensor_map,
            reduce_tensor_map, is_reduce_output,
            tiling_params)

    # cut not first axis
    log.debug("cosine_embedding_loss_schedule cut_notfirst_axis")
    return cosine_embedding_loss_schedule_cut_notfirst_axis(
        res, tensor_list_dst_tensor_map,
        reduce_tensor_map, is_reduce_output,
        tiling_params)
