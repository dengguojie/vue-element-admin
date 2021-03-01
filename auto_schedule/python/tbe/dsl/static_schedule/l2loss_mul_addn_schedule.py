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
l2loss_mull_addn_schedule
"""
from __future__ import absolute_import
import functools

import te
from tbe import tvm
from te import platform as cce
from .util import get_emit_insn_map
from .util import gen_reversed_subgraph_list
from .util import DTYPE_WIDTH_MAP


def get_max_ub_count(dtype):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    # div 2 for align to fp16
    total_size = cce.get_soc_spec("UB_SIZE") // 2
    dtype_size = DTYPE_WIDTH_MAP.get(dtype)
    total_size = total_size // dtype_size
    # two input, not need to do double buffer
    total_width = 4.0001

    align_to = 32
    max_bound = total_width * align_to
    max_ub_count = int(total_size // max_bound * align_to)

    return max_ub_count


def _ceil(num, div):
    """
    find the ceil of (num / div)
    """
    return (num + div - 1) // div


def _find_factor(numble, nparts, is_align_shape):
    """
    find the factor of number
    :return:
    """
    if is_align_shape:
        return _ceil(numble, nparts)
    for i in range(nparts, 0, -1):
        if numble % i == 0:
            return numble // i
    return numble


def _get_last_emit_tiling(shape, min_emit_size, is_align_shape):
    """
    find the min emit axis and factor
    """
    last_emit_axis = 0
    last_emit_factor = shape[0]
    tmp_size = 1
    for i in range(len(shape) - 1, -1, -1):
        tmp_size *= shape[i]
        if tmp_size >= min_emit_size:
            last_emit_axis = i
            pre_size = tmp_size // shape[i]
            cur_min_factor = _ceil(min_emit_size, pre_size)
            cur_max_nparts = _ceil(shape[i], cur_min_factor)
            last_emit_factor = _find_factor(shape[i], cur_max_nparts, is_align_shape)
            break
    return last_emit_axis, last_emit_factor


def _get_block_tiling(shape, last_emit_axis, last_emit_factor, is_align_shape):
    """
    find block tiling
    """
    core_num = cce.get_soc_spec("CORE_NUM")
    block_split_axis = 0
    block_factor = 1
    tmp_size = 1

    for i, dim in enumerate(shape):
        tmp_size *= dim
        if i == last_emit_axis or tmp_size >= core_num:
            block_split_axis = i
            cur_max_nparts = core_num // (tmp_size // dim)
            block_factor = _find_factor(dim, cur_max_nparts, is_align_shape)
            if i == last_emit_axis:
                block_factor = max(last_emit_factor, block_factor)
            break
    return block_split_axis, block_factor


def _get_ub_tiling(shape, block_split_axis, block_factor, last_emit_axis, max_ub_count, is_align_shape):
    """
    find ub tiling
    """
    ub_split_axis = block_split_axis
    ub_factor = block_factor
    shape[block_split_axis] = block_factor

    tmp_size = 1
    for i in range(len(shape) - 1, block_split_axis - 1, -1):
        tmp_size *= shape[i]
        if tmp_size < max_ub_count:
            continue
        ub_split_axis = i
        pre_tmp_size = tmp_size // shape[i]
        ub_factor = max_ub_count // pre_tmp_size
        if ub_split_axis < last_emit_axis:
            is_align_shape = True
        if is_align_shape:
            nparts = _ceil(shape[i], ub_factor)
            ub_factor = _ceil(shape[i], nparts)
            return ub_split_axis, ub_factor
        for j in range(ub_factor, 0, -1):
            if shape[i] % j == 0:
                ub_factor = j
                break
        return ub_split_axis, ub_factor
    return ub_split_axis, ub_factor


def _get_tiling(shape, dtype):
    """
    ubtiling for l2loss + mul + addn fusion
    :param one_core_data_cnt: one_core_data_cnt
    :param dtype: data type
    :return: ubtiling factor
    """
    dtype_size = cce.get_bit_len(dtype) // 8
    min_emit_size = cce.VECTOR_INST_BLOCK_WIDTH // cce.VECTOR_INST_BLOCK_NUM // dtype_size
    max_ub_count = get_max_ub_count(dtype)
    # 1024 is a empirical value
    one_core_data_threadhold = 1024 * min_emit_size
    max_ub_count = min(max_ub_count, one_core_data_threadhold)
    is_align_shape = True if shape[-1] % min_emit_size == 0 else False
    total_size = 1
    for i in shape:
        total_size *= i
    if total_size <= max_ub_count:
        return [0, shape[0], 0, shape[0]]

    if len(shape) == 1:
        block_nums = _ceil(shape[0], max_ub_count)
        block_split_axis = 0
        ub_split_axis = 0
        core_num = cce.get_soc_spec("CORE_NUM")
        block_factor = _ceil(block_nums, core_num) * max_ub_count
        ub_factor = min(block_factor, max_ub_count)
        return [block_split_axis, block_factor, ub_split_axis, ub_factor]

    last_emit_axis, last_emit_factor = _get_last_emit_tiling(shape, min_emit_size, is_align_shape)
    block_split_axis, block_factor = _get_block_tiling(shape, last_emit_axis, last_emit_factor, is_align_shape)
    ub_split_axis, ub_factor = _get_ub_tiling(shape, block_split_axis, block_factor, last_emit_axis, max_ub_count,
                                              is_align_shape)

    return [block_split_axis, block_factor, ub_split_axis, ub_factor]


def _check_params(res, input_tensors):
    """
    check params
    """
    if len(res) != 2:
        raise RuntimeError("L2loss mul addn output nums should be 2!")

    if len(input_tensors) != 3:
        raise RuntimeError("L2loss mul addn input nums should be 3!")


def _do_emit_insn(sch_list, cache_read_buffer_list, mid_out_tensor_list, mid_out_tensor_read_buffer_map,
                  cache_write_buffer_map, phony_tensor):
    # pylint: too-many-arguments
    sch = sch_list[0]
    for tensor_u in cache_read_buffer_list:
        sch[tensor_u].emit_insn(tensor_u.op.axis[0], 'dma_copy')

    for tensor_u in mid_out_tensor_list:
        sch[tensor_u].emit_insn(tensor_u.op.axis[0], 'dma_copy')

    for tensor_u in mid_out_tensor_read_buffer_map:
        buf = mid_out_tensor_read_buffer_map[tensor_u]
        sch[buf].emit_insn(buf.op.axis[0], 'phony_insn')

    for tensor in cache_write_buffer_map:
        buf = cache_write_buffer_map[tensor]
        if tensor in phony_tensor:
            sch[buf].emit_insn(buf.op.axis[0], 'phony_insn')
        else:
            emit_insn_pragma = get_emit_insn_map(buf)
            sch[buf].emit_insn(buf.op.axis[0], emit_insn_pragma)
    sch_list[0] = sch


def _do_compute_at(sch_list, cache_read_buffer_list, mid_out_tensor_list, mid_out_tensor_read_buffer_map,
                   cache_write_buffer_map, compute_at_tensor, compute_at_axis):
    sch = sch_list[0]
    for i in cache_read_buffer_list:
        sch[i].compute_at(sch[compute_at_tensor], compute_at_axis)

    for i in mid_out_tensor_list:
        sch[i].compute_at(sch[compute_at_tensor], compute_at_axis)

    for i in mid_out_tensor_read_buffer_map:
        buf = mid_out_tensor_read_buffer_map[i]
        sch[buf].compute_at(sch[compute_at_tensor], compute_at_axis)

    for i in cache_write_buffer_map:
        buf = cache_write_buffer_map[i]
        sch[buf].compute_at(sch[compute_at_tensor], compute_at_axis)
    sch_list[0] = sch


def l2loss_mul_addn_schedule(res, input_tensors):
    """
    l2loss + mul + addn fusion schedule for float32 and dim cnt equal to 1
    :param res: res tensor
    :param input_tensors: input tensors
    :return: sch
    """
    _check_params(res, input_tensors)

    res_add = res[0]
    res_l2l0ss = res[1]

    dtype = res_add.dtype
    if dtype != "float32":
        raise RuntimeError("L2loss mul addn only support float32 input!")

    mul_3 = res_l2l0ss.op.input_tensors[0]

    phony_mul = te.lang.cce.vmuls(res_add, 0.0)
    phony_add = te.lang.cce.vadd(phony_mul, mul_3)
    axis = [i for i in range(len(res_add.shape))]
    new_res = te.lang.cce.sum(phony_add, axis=axis, keepdims=False)

    shape_add = te.lang.cce.util.shape_to_list(res_add.shape)

    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}
    mid_out_tensor_list = [
        res_add,
    ]
    phony_tensor = [phony_mul, phony_add]

    gen_reversed_subgraph_list(new_res, tensor_list_map, tensor_list_dst_tensor_map)

    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}
    cache_read_tensor_list = []
    cache_write_tensor_list = []
    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[tensor]
            cache_read_tensor_list.append(tensor)
        else:
            mid_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[tensor]
            cache_write_tensor_list.append(tensor)

    sch = tvm.create_schedule([new_res.op])

    block_split_axis, block_factor, ub_split_axis, ub_factor = _get_tiling(shape_add, dtype)

    if ub_split_axis < block_split_axis:
        raise RuntimeError("Invalid tiling!")

    res_block_outer, _ = sch[new_res].split(new_res.op.reduce_axis[block_split_axis], block_factor)

    fuse_axis_list = []
    for i in range(block_split_axis):
        fuse_axis_list.append(new_res.op.reduce_axis[i])
    fuse_axis_list.append(res_block_outer)
    fused_axis = sch[new_res].fuse(*fuse_axis_list)
    res_ub_rf = sch.rfactor(new_res, fused_axis)

    # ---------cache read/write--------------
    cache_read_buffer_list = []
    for tensor in cache_read_tensor_list:
        cache_read_buffer_list.append(sch.cache_read(tensor, cce.scope_ubuf, input_tensor_dst_tensor_map[tensor]))

    mid_out_tensor_read_buffer_map = {}
    for i in mid_out_tensor_list:
        read_buffer = sch.cache_read(i, cce.scope_ubuf, mid_tensor_dst_tensor_map[i])
        mid_out_tensor_read_buffer_map[i] = read_buffer

    cache_write_buffer_list = []
    cache_write_buffer_map = {}
    for tensor in cache_write_tensor_list:
        buf = sch.cache_write(tensor, cce.scope_ubuf)
        cache_write_buffer_list.append(buf)
        cache_write_buffer_map[tensor] = buf

    new_res_global = sch.cache_write(new_res, cce.scope_gm)

    sch[res_ub_rf].set_scope(cce.scope_ubuf)

    # ---------compute inline----------------
    for tensor in cache_write_tensor_list:
        if tensor not in mid_out_tensor_list:
            sch[tensor].compute_inline()

    # reuse buffer of mul_3 and phony_add
    tensor_ub = cache_write_buffer_map[mul_3]
    reuse_tensor_ub = cache_write_buffer_map[phony_add]
    sch[tensor_ub].reused_by(reuse_tensor_ub)

    reorder_axis_list = []
    if ub_split_axis == block_split_axis:
        ub_outer, ub_inner = sch[res_ub_rf].split(res_ub_rf.op.reduce_axis[-1], factor=ub_factor)
        reorder_axis_list += res_ub_rf.op.axis
        reorder_axis_list.append(ub_outer)
        reorder_axis_list.append(ub_inner)
        reorder_axis_list += res_ub_rf.op.reduce_axis[0:-1]
        sch[res_ub_rf].reorder(*reorder_axis_list)
    else:
        ub_outer, ub_inner = \
            sch[res_ub_rf].split(res_ub_rf.op.reduce_axis[ub_split_axis - 1],
                                 factor=ub_factor)
        reorder_axis_list += res_ub_rf.op.axis
        reorder_axis_list.append(res_ub_rf.op.reduce_axis[-1])
        reorder_axis_list += res_ub_rf.op.reduce_axis[0:ub_split_axis - 1]
        reorder_axis_list.append(ub_outer)
        reorder_axis_list.append(ub_inner)
        reorder_axis_list += res_ub_rf.op.reduce_axis[ub_split_axis:-1]
        sch[res_ub_rf].reorder(*reorder_axis_list)

    reorder_axis_list = []
    reorder_axis_list.append(new_res_global.op.reduce_axis[0])
    reorder_axis_list += new_res_global.op.axis
    sch[new_res_global].reorder(*reorder_axis_list)

    compute_at_axis = ub_outer

    sch_list = [sch]
    _do_compute_at(sch_list, cache_read_buffer_list, mid_out_tensor_list, mid_out_tensor_read_buffer_map,
                   cache_write_buffer_map, res_ub_rf, compute_at_axis)
    sch = sch_list[0]

    sch[res_ub_rf].compute_at(sch[new_res_global], new_res_global.op.reduce_axis[0])

    res[0] = res_add
    res[1] = new_res_global

    sch_list = [sch]
    _do_emit_insn(sch_list, cache_read_buffer_list, mid_out_tensor_list, mid_out_tensor_read_buffer_map,
                  cache_write_buffer_map, phony_tensor)
    sch = sch_list[0]

    sch[res_ub_rf].emit_insn(ub_inner, "reduce_last_axis_reduce_sum")
    sch[new_res_global].emit_insn(new_res_global.op.axis[0], "dma_copy")
    sch[new_res].emit_insn(sch[new_res].op.axis[0], "phony_insn")

    # ------------------bind----------------------
    block = tvm.thread_axis("blockIdx.x")
    sch[new_res_global].bind(new_res_global.op.reduce_axis[0], block)

    return sch
