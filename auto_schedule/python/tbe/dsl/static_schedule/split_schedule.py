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
split schedule
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
from functools import reduce as functools_reduce

from tbe import tvm
from tbe.common.platform import scope_ubuf
from tbe.dsl.instrinsic.cce_intrin import get_bit_len
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.rl_bank import bank_manager
from tbe.common.rl_bank import rl_bank


# pylint: disable=locally-disabled,too-many-locals
def _tile_axis(shape, dtype, split_dim):
    """Calculate the tile parameters.

    Parameters
    ----------
    shape: list or tuple
        shape of tensor.
    dtype: str
        dtype of tensor.
    split_dim: int
        the dimension along which to split.

    Returns
    -------
    tile_axis: int
        the target axis that is used for tile the tensor.
    tile_factor: int
        the factor used when tile the target axis.
    """
    ub_size = get_soc_spec("UB_SIZE") - 1024
    dtype_size = get_bit_len(dtype) // 8
    total_cnt = ub_size // dtype_size
    block_cnt = 32 // dtype_size
    split_cnt = functools_reduce(lambda x, y: x * y, shape[split_dim:])

    tile_shape = []
    for dim in shape:
        tile_shape.append(dim)

    if split_cnt % block_cnt != 0 and split_dim != 0:
        last_ele = math.ceil(shape[-1] / block_cnt) * block_cnt
        tile_shape[-1] = int(last_ele)

    tile_axis = 0
    tile_factor = 1
    for i, _ in enumerate(tile_shape):
        ele_cnt = functools_reduce(lambda x, y: x * y, tile_shape[i:])
        if ele_cnt <= total_cnt:
            tile_axis = i - 1
            tile_factor = total_cnt // ele_cnt
            break

    if tile_shape[-1] > total_cnt:
        tile_axis = len(tile_shape) - 1
        tile_factor = total_cnt

    if tile_axis < 0:
        tile_axis = 0
        tile_factor = tile_shape[0]

    return tile_axis, tile_factor


def _check_align(shape_list, block_cnt, split_dim):
    """Check if the output is aligned.

    Parameters
    ----------
    shape_list: list
        the list of shapes.
    block_cnt: int
        the element count of one block.
    split_dim: int
        the dimension along which to split.

    Returns
    -------
    divide_flag: bool
        whether the outputs are equally divided.
    align_flag: bool
        whether the outputs are aligned.
    """
    divide_flag = True
    for i, _ in enumerate(shape_list):
        if shape_list[i][split_dim] != shape_list[0][split_dim]:
            divide_flag = False
            break

    align_flag = True
    for i, _ in enumerate(shape_list):
        split_ele = functools_reduce(lambda x, y: x * y, shape_list[i][split_dim:])
        if split_ele % block_cnt != 0:
            align_flag = False
            break

    return divide_flag, align_flag


def do_split_schedule(divide_flag, split_dim, align_flag, shape_list, i, dtype, sch, res, res_op,  # pylint: disable=too-many-arguments
                      block_idx, tensor_list, block_cnt):  # pylint: disable=too-many-arguments
    """
    do_split_schedule
    :param divide_flag:
    :param split_dim:
    :param align_flag:
    :param shape_list:
    :param i:
    :param dtype:
    :param sch:
    :param res:
    :param res_op:
    :param block_idx:
    :param tensor_list:
    :param block_cnt:
    :return:
    """
    if divide_flag and (split_dim == 0 or align_flag):
        tile_axis, tile_factor = _tile_axis(shape_list[i], dtype, split_dim)
        axis_outer, axis_inner = sch[res[i]].split(res_op[i].axis[tile_axis], factor=tile_factor)
        if tile_axis == 0:
            sch[res[i]].bind(axis_outer, block_idx)
        else:
            sch[res[i]].bind(res[i].op.axis[0], block_idx)
    elif not divide_flag and split_dim == 0:
        tile_axis, tile_factor = _tile_axis(shape_list[i], dtype, split_dim)
        axis_outer, axis_inner = sch[res[i]].split(res_op[i].axis[tile_axis], factor=tile_factor)
    elif not divide_flag and align_flag:
        tile_axis, tile_factor = _tile_axis(shape_list[i], dtype, split_dim)
        axis_outer, axis_inner = sch[res[i]].split(
            res_op[i].axis[split_dim],
            factor=shape_list[i][split_dim]) if tile_axis < split_dim else sch[res[i]].split(
                res_op[i].axis[tile_axis], factor=tile_factor)
        sch[res[i]].bind(res[i].op.axis[0], block_idx)
    else:
        tile_axis, tile_factor = _tile_axis(shape_list[i], dtype, split_dim)
        axis_outer, axis_inner = sch[res[i]].split(res_op[i].axis[tile_axis], factor=tile_factor)
        sch[tensor_list[i]].storage_align(tensor_list[i].op.axis[split_dim - 1], block_cnt, 0)

    sch[tensor_list[i]].compute_at(sch[res[i]], axis_outer)
    sch[tensor_list[i]].emit_insn(tensor_list[i].op.axis[tile_axis], "dma_copy")
    sch[res[i]].emit_insn(axis_inner, "dma_copy")


def split_schedule_com(data, split_dim, shape_list, tensor_list):
    """Create split schedule.

    Parameters
    ----------
    data: TVM tensor
        input tensor.
    split_dim: int
        the dimension along which to split.
    shape_list: list
        the list of output shapes.
    tensor_list: list
        the list of output tensors, tensor type is TVM tensor.

    Returns
    -------
    sch: schedule.Schedule
        The created schedule.
    build_list: list
        the list of input and output tensors, tensor type is TVM tensor.
    """
    res = []
    data_ub = None
    shape_ub = None
    for i, _ in enumerate(shape_list):
        data_ub = tensor_list[i]
        shape_ub = shape_list[i]
        # pylint: disable=locally-disabled,unnecessary-lambda
        data_gm = tvm.compute(shape_ub,
                              lambda *index: data_ub(*index),
                              name='res' + str(i),
                              tag='split_com|schedule_' + str(i))
        res.append(data_gm)
    # for RL tune getting res
    bank_manager.set_op_res(res)

    res_op = []
    build_list = [data]
    for data_gm in res:
        res_op.append(data_gm.op)
        build_list.append(data_gm)

    _, sch = rl_bank.query_rl_bank(res)
    if sch:
        return sch, build_list

    sch = tvm.create_schedule(res_op)

    for tensor in tensor_list:
        sch[tensor].set_scope(scope_ubuf)

    dtype = data.dtype
    dtype_size = get_bit_len(dtype) // 8
    block_cnt = 32 // dtype_size
    block_idx = tvm.thread_axis('blockIdx.x')
    divide_flag, align_flag = _check_align(shape_list, block_cnt, split_dim)

    for i, _ in enumerate(shape_list):
        do_split_schedule(divide_flag, split_dim, align_flag, shape_list, i, dtype, sch, res,
                          res_op, block_idx, tensor_list, block_cnt)

    return sch, build_list
