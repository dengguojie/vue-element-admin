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
ascend_dequant
"""
from functools import reduce as function_reduce

from tbe import tvm
from tbe.common.utils import shape_to_list
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform import scope_ubuf
from tbe.common.platform import scope_cc


def _get_tensor_map(res, tensor_map):
    """
    get the compute tensors

    Parameters
    ----------
    res: the placeholder of result
    tensor_map: the compute tensors

    Returns
    -------
    None
    """
    stack = [res]
    visited_list = []
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                tensor_map[in_tensor.name] = in_tensor
    if "x" in tensor_map:
        tensor_map.pop("x")
    if "deq_scale" in tensor_map:
        tensor_map.pop("deq_scale")


def _tilling_axis(shape, dtype_size, tensor_num):
    """
    get the split axis and factor by ub size

    Parameters
    ----------
    shape: the shape of input
    dtype_size: the dtype size
    tensor_num: the number of tensor size

    Returns
    -------
    split_axis and split_factor
    """
    shape_new = list(shape).copy()
    shape_new[2] = (shape_new[2] + 15) // 16 * 16
    total_ele = get_soc_spec("UB_SIZE") // \
        dtype_size // tensor_num // 2
    block_num = get_soc_spec("CORE_NUM")
    val_cnt = 1
    index_cnt = 0
    for i in range(0, len(shape_new) - 1):
        val_cnt = val_cnt * shape_new[i]
        index_cnt = i
        if val_cnt >= block_num:
            break

    block_size = val_cnt // block_num * \
        function_reduce(lambda x, y: x * y, shape_new[index_cnt + 1:])
    if 256 <= block_size <= total_ele:
        total_ele = block_size
    total_ele = total_ele // 256 * 256
    split_axis = 0
    split_factor = 1
    size = function_reduce(lambda x, y: x * y, shape_new[1:])
    for index, _ in enumerate(shape_new):
        ele_cnt = function_reduce(lambda x, y: x * y, shape_new[index:])
        if ele_cnt <= total_ele:
            split_axis = index - 1
            split_factor = total_ele // ele_cnt
            break
    if split_axis < 0 or (split_axis == 0 and size <= total_ele):
        split_axis = 0
        split_factor = 1
    return split_axis, split_factor


def _get_fuse_info(sch, res, res_split_shape, split_info):
    """
    get the fuse info

    Parameters
    ----------
    sch: the schedule
    res: the placeholder of result
    res_split_shape: the output shape
    split_info: split_axis and split_factor

    Returns
    -------
    fused_value, fused_list, axis_outer_num
    """
    split_axis = split_info[0]
    split_factor = split_info[1]
    if res_split_shape[split_axis] % split_factor > 0:
        axis_outer_num = res_split_shape[split_axis] // split_factor + 1
    else:
        axis_outer_num = res_split_shape[split_axis] // split_factor
    origin_list = [res_split_shape[i] for i in range(split_axis)]
    fused_value = 1
    for _, item in enumerate(origin_list):
        fused_value *= item
    fused_list = [sch[res].op.axis[i] for i in range(split_axis)]
    return fused_value, fused_list, axis_outer_num


def _set_buffer_scope(sch, tensor_map):
    """
    set the scope for tensors

    Parameters
    ----------
    sch: the schedule
    tensor_map: the compute tensors

    Returns
    -------
    None
    """
    for key, value in tensor_map.items():
        if key == "x_l0c":
            sch[value].set_scope(scope_cc)
        else:
            sch[value].set_scope(scope_ubuf)


def _bind_fuse(fused_value, fused_list, axis_outer_num, sch, res,
               axis_outer, out_shape):
    """
    bind the fused axis.
    """
    core_num = get_soc_spec("CORE_NUM")
    bind_axis = axis_outer
    if fused_list:
        if fused_value * axis_outer_num <= core_num:
            fused_list.append(axis_outer)
            bind_axis = sch[res].fuse(*fused_list)
            axis_outer = bind_axis
        elif fused_value < core_num:
            num = core_num // fused_value
            thread_outer, axis_outer = sch[res].split(axis_outer,
                                                      nparts=num)
            fused_list.append(thread_outer)
            bind_axis = sch[res].fuse(*fused_list)
        else:
            val_cnt = 1
            index = 0
            for i in range(len(fused_list)):
                val_cnt = val_cnt * out_shape[i]
                if val_cnt >= core_num:
                    index = i
                    break
            num = core_num // (val_cnt // out_shape[index])
            thread_outer, _ = sch[res].split(res.op.axis[index], nparts=num)
            new_fused_list = fused_list[:index]
            new_fused_list.append(thread_outer)
            bind_axis = sch[res].fuse(*new_fused_list)
    sch[res].bind(bind_axis, tvm.thread_axis("blockIdx.x"))
    return axis_outer


def _bind_core(out_shape, sch, res, tensor_map):
    """
    bind multi-core

    Parameters
    ----------
    out_shape: the output shape
    sch: the schedule
    res: the placeholder of result
    tensor_map: the compute tensors

    Returns
    -------
    axis_outer, axis_inner
    """
    core_num = get_soc_spec("CORE_NUM")
    split_axis, split_factor = _tilling_axis(out_shape, 4, 2)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                            factor=split_factor)
    fused_value, fused_list, axis_outer_num = _get_fuse_info(
        sch, res, out_shape, (split_axis, split_factor))
    bind_axis = 0
    can_bind = False
    for i in range(split_axis):
        if out_shape[i] >= core_num:
            bind_axis = i
            can_bind = True
            break
    if can_bind:
        thread_outer, _ = sch[res].split(res.op.axis[bind_axis],
                                         nparts=core_num)
        sch[res].bind(thread_outer, tvm.thread_axis("blockIdx.x"))
    elif axis_outer_num >= core_num:
        thread_outer, axis_outer = sch[res].split(axis_outer,
                                                  nparts=core_num)
        sch[res].bind(thread_outer, tvm.thread_axis("blockIdx.x"))
    else:
        axis_outer = _bind_fuse(fused_value, fused_list, axis_outer_num, sch,
                                res, axis_outer, out_shape)
    sch[tensor_map.get("x_ub")].double_buffer()
    sch[tensor_map.get("deq_ub")].double_buffer()
    return axis_outer, axis_inner


def _set_buffer_emit_insn(sch, res, tensor_map, axis_inner):
    """
    instruction mapping

    Parameters
    ----------
    sch: the schedule
    res: the placeholder of result
    tensor_map: the compute tensors
    axis_inner: the inner axis

    Returns
    -------
    None
    """
    sch[tensor_map.get("x_ub")].emit_insn(
        sch[tensor_map.get("x_ub")].op.axis[0], 'dma_copy')
    sch[tensor_map.get("deq_ub")].emit_insn(
        sch[tensor_map.get("deq_ub")].op.axis[0], 'dma_copy')
    sch[tensor_map.get("x_l0c")].emit_insn(
        sch[tensor_map.get("x_l0c")].op.axis[0], 'dma_copy')
    sch[res].emit_insn(axis_inner, 'dma_copy')
    for key, value in tensor_map.items():
        if key in ["x_ub", "deq_ub"]:
            pass
        elif key == "x_l0c":
            sch[value].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
        elif key == "dequant_to_fp16":
            sch[value].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
            if get_soc_spec("SOC_VERSION") in ("Ascend710",
                                                       "Ascend610",
                                                       "Ascend615",
                                                       "Hi3796CV300CS",
                                                       "SD3403"):
                if res.op.attrs['is_scalar'].value == 1:
                    sch[value].emit_insn(sch[value].op.axis[0], 'dma_copy')
                else:
                    sch[value].emit_insn(sch[value].op.axis[2], 'dma_copy')
            else:
                if res.op.attrs['is_scalar'].value == 1:
                    sch[value].pragma(value.op.axis[0], 'deq_scale', 'scalar')
                else:
                    sch[value].pragma(value.op.axis[2], 'deq_scale', 'vector')
        else:
            sch[value].emit_insn(
                sch[value].op.axis[0], 'vector_auto')


def ascend_dequant_schedule(res, input_tensors):
    """
    the schedule processes of dequant

    Parameters
    ----------
    res: the placeholder of result
    input_tensors: the placeholder of input

    Returns
    -------
    the result of schedule
    """
    sch = tvm.create_schedule(res.op)
    tensor_map = {}
    _get_tensor_map(res, tensor_map)
    out_shape = shape_to_list(res.shape)
    _set_buffer_scope(sch, tensor_map)
    axis_outer, axis_inner = _bind_core(out_shape, sch, res, tensor_map)

    for _, value in tensor_map.items():
        sch[value].compute_at(sch[res], axis_outer)

    _set_buffer_emit_insn(sch, res, tensor_map, axis_inner)
    return sch
