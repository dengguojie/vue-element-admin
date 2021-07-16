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
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.compute.dilation_compute import calc_minimum_ub
from tbe.dsl.compute.dilation_compute import calc_space_of_ub
from tbe.dsl.compute.dilation_compute import get_first_axis_need_dilate


def _get_tiling_factor(shape, dilations, dtype):
    """
    get tiling axis and factor
    :param shape: list or tuple
    :param dilations: list or tuple
    :param dtype: str
    :return: axis to split and factor
    """
    first_dilate_axis = get_first_axis_need_dilate(dilations)
    minimum_ub = calc_minimum_ub(shape, dilations, dtype)
    ub_size = tbe_platform_info.get_soc_spec("UB_SIZE")
    if minimum_ub > ub_size:
        args_dict = {
            "errCode": "E60038",
            "desc": "input is too large, the minimum space may exceed UB_Buffer"
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    axis_exceed_ub = -1
    for i in range(first_dilate_axis):
        size = calc_space_of_ub(shape, dilations, i, dtype)
        if size > ub_size:
            axis_exceed_ub = i
        else:
            break
    if axis_exceed_ub == -1:
        return -1, -1
    max_no_exceed_size = calc_space_of_ub(shape, dilations, axis_exceed_ub + 1, dtype)
    factor = ub_size // max_no_exceed_size
    return axis_exceed_ub, factor


def _get_buffer_align_param(dilations, dtype):
    """
    get_buffer_align_param
    :param dilations: list
    :param dtype: str,  data type
    :return: buffer_align params
    """
    res = [(1, 1)] * (len(dilations) - 1)
    align_num = tbe_platform.BLOCK_REDUCE
    if dtype == "int8":
        align_num = tbe_platform.BLOCK_REDUCE_INT8
    res.append((1, align_num))
    return res


def _get_all_tensors(res):
    """
    get all tensors in schedule
    :param res: tvm.tensor
    :return: dict of all tensors
    """
    all_tensor = dict()
    all_tensor["res"] = res

    def get(tensor):
        """
        get all input tensors
        :param tensor: tvm.tensor
        """
        tensor_list = tensor.op.input_tensors
        for each_tensor in tensor_list:
            if each_tensor.op.name not in all_tensor:
                all_tensor[each_tensor.op.name] = each_tensor
                get(each_tensor)

    get(res)
    return all_tensor


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
    all_tensors = _get_all_tensors(res)
    x = all_tensors.get("x")
    ub_x = all_tensors.get("ub_x")
    padding_default = all_tensors.get("padding_default")
    ub_dilated_x = all_tensors.get("ub_dilated_x")
    dilated_x = all_tensors.get("res")
    dilations = [i.value for i in x.op.attrs["dilations"]]
    dtype = x.op.attrs["dtype"].value

    # set scope and buffer_align
    sch[ub_x].buffer_align(*_get_buffer_align_param(dilations, dtype))
    sch[ub_dilated_x].buffer_align(*_get_buffer_align_param(dilations, dtype))
    sch[ub_x].set_scope(tbe_platform_info.scope_ubuf)
    sch[padding_default].set_scope(tbe_platform_info.scope_ubuf)
    sch[ub_dilated_x].set_scope(tbe_platform_info.scope_ubuf)

    # get split params, split and compute_at
    shape_ub_x = [i.value for i in ub_x.shape]
    split_axis, split_factor = _get_tiling_factor(shape_ub_x, dilations, dtype)
    if split_axis == -1:
        dilate_x_out = dilated_x.op.axis[0]
        dilate_x_in = dilated_x.op.axis[1]
    else:
        dilate_x_out, dilate_x_in = sch[dilated_x].split(dilated_x.op.axis[split_axis], split_factor)
    sch[padding_default].compute_inline()
    sch[ub_x].compute_at(sch[dilated_x], dilate_x_out)
    sch[ub_dilated_x].compute_at(sch[dilated_x], dilate_x_out)

    # emit_insn
    sch[padding_default].emit_insn(padding_default.op.axis[0], "dma_copy")
    sch[ub_x].emit_insn(ub_x.op.axis[0], "dma_copy")
    ub_dilate_x_emit_str = "vector_muls"
    if dtype == "int8":
        ub_dilate_x_emit_str = "dma_copy"
    sch[ub_dilated_x].emit_insn(ub_dilated_x.op.axis[0], ub_dilate_x_emit_str)
    sch[dilated_x].emit_insn(dilate_x_in, "dma_copy")
    return True
