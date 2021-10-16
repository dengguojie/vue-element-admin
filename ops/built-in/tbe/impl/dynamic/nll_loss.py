#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

nll_loss
"""

from impl.util import util_common
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl import common_util


class Constant:
    """
    The class for constant
    """
    OP_TYPE = "nll_loss"
    # ub size count
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # aicore count
    CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    # byte count one block
    BLOCK_BYTE_COUNT = 32
    # repeat up limit for mte
    REPEAT_LIMIT = 255
    # max int64 value
    MAX_INT64_VALUE = 2 ** 64 - 1
    # parameters for moving tiling data
    TILING_CTRL_PARAM = ("int64", 64, 4)


# pylint: disable=unused-argument
def check_supported(x, target, weight, y, total_weight, reduction="mean", ignore_index=-100, kernel_name="nll_loss"):
    """
    check nllloss supported

    Parameters
    ----------
    x : dict
        shape and dtype of input x, the length of shape should be two or one.
    target : dict
        shape and dtype of input target, the length of shape only support one.
    weight : dict
        shape and dtype of input weight, the length of shape only support one.
    y:dict
        shape and dtype of output y.
        it's a tensor with shape(minibatch, ) when reduction == 'none' and
        the input is 2D. Otherwise, the output is a scalar.
    total_weight:
        shape and dtype of output total_weight, should be same type as weight.
        the output is scalar.
    reduction: str
        default value is "mean"
    ignore_index: int
        default value is -100
    kernel_name : str
        kernel name, default value is "nll_loss"

    Returns
    -------
    (is_supported, description)
    """
    x_shape = x.get("ori_shape")

    if util_common.is_unknown([x, target, weight]):
        return True, ""

    if _dynamic_static_union(x_shape, reduction):
        return True, ""

    return False, ""


def _dynamic_static_union(shape, reduction):
    """
    for dynamic and static union fully verified
    """
    white_list_dict = {"none": [],
                       "sum": []}

    if reduction not in white_list_dict:
        return False

    x_shape = list(shape)
    if x_shape in white_list_dict[reduction]:
        return True

    return False


# pylint: disable=invalid-name
def _ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


def _get_element_cnt_one_block(dtype):
    """
    get element count in a block
    """
    byte_len = common_util.get_data_size(dtype)
    element_cnt = Constant.BLOCK_BYTE_COUNT // byte_len

    return element_cnt


def _get_max_element_in_ub(dtype, ub_part):
    """
    get the up limit elements in UB
    """
    byte_len = common_util.get_data_size(dtype)

    ub_upper_limit = (Constant.UB_SIZE - 2 * 1024) // ub_part
    element_size = ub_upper_limit // byte_len

    return element_size


def _get_reduce_sum_ub_work_space_size(dtype, ub_size):
    """
    get reduce sum ub work space size
    """
    data_one_block = _get_element_cnt_one_block(dtype)
    repeat_data_num = 8 * data_one_block
    repeat_num = _ceil_div(ub_size, repeat_data_num)

    # 32byte align
    if repeat_num % data_one_block > 0:
        repeat_num = _ceil_div(repeat_num, data_one_block) * data_one_block

    return repeat_num


def _check_input_params(x_dtype, target_dtype, weight_dtype, y_dtype, total_weight_dtype, reduction):
    """
    check whether the input parameters is valid or not
    """
    para_check.check_dtype(x_dtype, ("float32",), param_name="x")
    para_check.check_dtype(target_dtype, ("int32",), param_name="target")
    para_check.check_dtype(weight_dtype, ("float32",), param_name="weight")
    para_check.check_dtype(y_dtype, ("float32",), param_name="y")
    para_check.check_dtype(total_weight_dtype, ("float32",), param_name="total_weight")

    if reduction not in ("none", "sum", "mean"):
        reduction_rule = "reduction should be in range ('none', 'sum', 'mean')"
        error_manager_vector.raise_err_check_params_rules(Constant.OP_TYPE, reduction_rule, "reduction", reduction)


def scalar_vector_func(tik_inst, vec_func, dst, src, scalar, data_len, data_type):
    """
    do scalar vector operator (vmuls)
    """
    data_one_block = _get_element_cnt_one_block(data_type)
    repeat_data_num = 8 * data_one_block
    repeat = tik_inst.Scalar("int64", "scalar_vector_func_repeat")
    repeat.set_as(data_len // repeat_data_num)
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT

    with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
        with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
            offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            vec_func(repeat_data_num, dst[offset], src[offset], scalar, Constant.REPEAT_LIMIT, 1, 1, 8, 8)
        repeat.set_as(repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT)
    with tik_inst.if_scope(repeat > 0):
        offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
        vec_func(repeat_data_num, dst[offset], src[offset], scalar, repeat, 1, 1, 8, 8)
    with tik_inst.if_scope(repeat_tail > 0):
        offset = (loop_repeat_cnt * Constant.REPEAT_LIMIT + repeat) * repeat_data_num
        vec_func(repeat_tail, dst[offset], src[offset], scalar, 1, 1, 1, 8, 8)


def vector_func(tik_inst, vec_func, dst, src1, src2, data_len, data_type):
    """
    do vector operator (vmul)
    """
    data_one_block = _get_element_cnt_one_block(data_type)
    repeat_data_num = 8 * data_one_block
    repeat = tik_inst.Scalar("int64", "vector_func_repeat")
    repeat.set_as(data_len // repeat_data_num)
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT

    with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
        with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
            offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            vec_func(repeat_data_num, dst[offset], src1[offset], src2[offset], Constant.REPEAT_LIMIT, 1, 1, 1, 8, 8, 8)
        repeat.set_as(repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT)
    with tik_inst.if_scope(repeat > 0):
        offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
        vec_func(repeat_data_num, dst[offset], src1[offset], src2[offset], repeat, 1, 1, 1, 8, 8, 8)
    with tik_inst.if_scope(repeat_tail > 0):
        offset = (loop_repeat_cnt * Constant.REPEAT_LIMIT + repeat) * repeat_data_num
        vec_func(repeat_tail, dst[offset], src1[offset], src2[offset], 1, 1, 1, 1, 8, 8, 8)


def reduce_sum_compute(tik_inst, dst, src, work_space, data_len, data_type):
    """
    do reduce sum vector operator(vcadd,vadd)
    """
    data_one_block = _get_element_cnt_one_block(data_type)
    repeat_data_num = 8 * data_one_block
    repeat = tik_inst.Scalar("int32", "reduce_sum_compute_repeat")
    repeat.set_as(data_len // repeat_data_num)
    repeat_tail = data_len % repeat_data_num

    with tik_inst.if_scope(repeat_tail == 0):
        tik_inst.vec_reduce_add(repeat_data_num, dst, src, work_space, repeat, 8)
    with tik_inst.else_scope():
        offset = repeat * repeat_data_num
        tik_inst.vcadd(repeat_tail, dst, src[offset], 1, 1, 1, 8)
        with tik_inst.if_scope(repeat > 0):
            offset = repeat_data_num
            tik_inst.vec_reduce_add(repeat_data_num, dst[offset], src, work_space, repeat, 8)
            tik_inst.vadd(1, dst, dst, dst[offset], 1, 1, 1, 1, 8, 8, 8)


def _init_tiling_params(tik_inst, tiling_reg_list):
    """
    init tiling parameters
    """
    # tiling mode
    tiling_reg_list[0] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "tiling_mode")
    # need core num
    tiling_reg_list[1] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "need_core_num")
    # n size
    tiling_reg_list[2] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "n_size")
    # c size
    tiling_reg_list[3] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "c_size")
    # per core size
    tiling_reg_list[4] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "per_core_size")
    # per core loop count
    tiling_reg_list[5] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "per_core_loop_count")
    # per core left size
    tiling_reg_list[6] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "per_core_left_size")
    # last core size
    tiling_reg_list[7] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "last_core_size")
    # last core loop count
    tiling_reg_list[8] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "last_core_loop_count")
    # last core left size
    tiling_reg_list[9] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "last_core_left_size")
    # x size
    tiling_reg_list[10] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "x_size")
    # target size
    tiling_reg_list[11] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "target_size")
    # weight size
    tiling_reg_list[12] = tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "weight_size")

    return tiling_reg_list


def _get_tiling_params(tik_inst, ub_tiling, tiling_reg_list):
    """
    get tiling parameters
    """
    _init_tiling_params(tik_inst, tiling_reg_list)
    # get tiling mode
    tiling_reg_list[0].set_as(ub_tiling[0])
    # get need core num
    tiling_reg_list[1].set_as(ub_tiling[1])
    # get n size
    tiling_reg_list[2].set_as(ub_tiling[2])
    # get c size
    tiling_reg_list[3].set_as(ub_tiling[3])
    # get per core size
    tiling_reg_list[4].set_as(ub_tiling[4])
    # get per core loop count
    tiling_reg_list[5].set_as(ub_tiling[5])
    # get per core left size
    tiling_reg_list[6].set_as(ub_tiling[6])
    # get last core size
    tiling_reg_list[7].set_as(ub_tiling[7])
    # get last core loop count
    tiling_reg_list[8].set_as(ub_tiling[8])
    # get last core left size
    tiling_reg_list[9].set_as(ub_tiling[9])
    # get x size
    tiling_reg_list[10].set_as(ub_tiling[10])
    # get target size
    tiling_reg_list[11].set_as(ub_tiling[11])
    # get weight size
    tiling_reg_list[12].set_as(ub_tiling[12])


def _init_scalar_params(block_idx, scalar_params_list, tiling_data_list):
    """
    init scalar params
    """
    _, c_size, per_core_size, core_loop_cnt, core_x_loop_size, core_target_loop_size, core_left_size, core_x_offset, \
        core_target_offset, core_x_left_offset, core_target_left_offset, _, _, _, _ = scalar_params_list

    size, loop_cnt, left_size = tiling_data_list
    target_loop_size = (size - left_size) // loop_cnt
    x_loop_size = target_loop_size * c_size

    def _get_core_target_offset():
        """
        get target offset
        """
        return block_idx * per_core_size

    def _get_core_x_offset():
        """
        get x offset
        """
        return _get_core_target_offset() * c_size

    def _get_core_x_left_offset():
        """
        get x left offset
        """
        return _get_core_x_offset() + (size - left_size) * c_size

    def _get_core_target_left_offset():
        """
        get target left offset
        """
        return _get_core_target_offset() + size - left_size

    core_loop_cnt.set_as(loop_cnt)
    core_x_loop_size.set_as(x_loop_size)
    core_target_loop_size.set_as(target_loop_size)
    core_left_size.set_as(left_size)
    core_x_offset.set_as(_get_core_x_offset())
    core_target_offset.set_as(_get_core_target_offset())
    core_x_left_offset.set_as(_get_core_x_left_offset())
    core_target_left_offset.set_as(_get_core_target_left_offset())


def _get_scalar_params(tik_inst, block_idx, tiling_reg_list, target_dtype):
    """
    get scalar params
    """
    need_core_num = tiling_reg_list[1]
    c_size = tiling_reg_list[3]
    per_core_size = tiling_reg_list[4]
    per_core_loop_cnt = tiling_reg_list[5]
    per_core_left_size = tiling_reg_list[6]
    last_core_size = tiling_reg_list[7]
    last_core_loop_cnt = tiling_reg_list[8]
    last_core_left_size = tiling_reg_list[9]
    x_size = tiling_reg_list[10]
    target_size = tiling_reg_list[11]
    weight_size = tiling_reg_list[12]

    core_loop_cnt = tik_inst.Scalar("int64", "core_loop_cnt")
    core_x_loop_size = tik_inst.Scalar("int64", "core_x_loop_size")
    core_target_loop_size = tik_inst.Scalar("int64", "core_target_loop_size")
    core_left_size = tik_inst.Scalar("int64", "core_left_size")
    core_x_offset = tik_inst.Scalar("int64", "core_x_offset")
    core_target_offset = tik_inst.Scalar("int64", "core_target_offset")
    core_x_left_offset = tik_inst.Scalar("int64", "core_x_left_offset")
    core_target_left_offset = tik_inst.Scalar("int64", "core_target_left_offset")
    target_index = tik_inst.Scalar(target_dtype, "target_index")

    scalar_params = [need_core_num, c_size, per_core_size, core_loop_cnt, core_x_loop_size, core_target_loop_size,
                     core_left_size, core_x_offset, core_target_offset, core_x_left_offset,
                     core_target_left_offset, target_index, x_size, target_size, weight_size]

    # last core
    with tik_inst.if_scope(block_idx == need_core_num - 1):
        tiling_params = [last_core_size, last_core_loop_cnt, last_core_left_size]
        _init_scalar_params(block_idx, scalar_params, tiling_params)
    with tik_inst.else_scope():
        tiling_params = [per_core_size, per_core_loop_cnt, per_core_left_size]
        _init_scalar_params(block_idx, scalar_params, tiling_params)

    return scalar_params


def _process_valid_data(tik_inst, data_x, data_weight, ub_x, ub_weight, ub_valid_x, ub_valid_weight, ub_work_space,
                        target_loop_size, reduction):
    """
    process valid data
    """
    # valid_x mul -1
    scalar_vector_func(tik_inst, tik_inst.vmuls, ub_valid_x, ub_valid_x, -1, target_loop_size,
                       data_x.dtype)

    # valid_x mul valid_weight
    vector_func(tik_inst, tik_inst.vmul, ub_valid_x, ub_valid_x, ub_valid_weight, target_loop_size,
                data_x.dtype)

    # get sum of valid_weight
    if reduction == "sum":
        reduce_sum_compute(tik_inst, ub_x, ub_valid_x, ub_work_space, target_loop_size,
                           data_x.dtype)
        reduce_sum_compute(tik_inst, ub_weight, ub_valid_weight, ub_work_space, target_loop_size,
                           data_weight.dtype)


def _data_move_in_x(tik_inst, data_x, ub_x, x_offset, x_loop_size):
    """
    do move x from out to ub
    """
    x_data_one_block = _get_element_cnt_one_block(data_x.dtype)
    tik_inst.data_move(ub_x, data_x[x_offset], 0, 1, _ceil_div(x_loop_size, x_data_one_block), 0, 0)


def _data_move_in_weight(tik_inst, data_weight, ub_weight, weight_loop_size):
    """
    do move weight from out to ub
    """
    weight_data_one_block = _get_element_cnt_one_block(data_weight.dtype)
    tik_inst.data_move(ub_weight, data_weight, 0, 1, _ceil_div(weight_loop_size, weight_data_one_block), 0, 0)


def _data_move_in_target(tik_inst, data_target, ub_target, target_offset, target_loop_size):
    """
    do move target from out to ub
    """
    target_data_one_block = _get_element_cnt_one_block(data_target.dtype)
    tik_inst.data_move(ub_target, data_target[target_offset], 0, 1,
                       _ceil_div(target_loop_size, target_data_one_block), 0, 0)


def _data_move_out_output(tik_inst, data_x, data_y, data_total_weight, ub_x, ub_weight, ub_valid_x, reduction,
                          target_offset, target_loop_size):
    """
    do move y and total_weight from ub to out
    """
    x_data_one_block = _get_element_cnt_one_block(data_x.dtype)

    if reduction == "sum":
        # move out sum valid_y and sum valid_weight
        tik_inst.set_atomic_add(1)
        tik_inst.data_move(data_y, ub_x, 0, 1, 1, 0, 0)
        tik_inst.data_move(data_total_weight, ub_weight, 0, 1, 1, 0, 0)
        tik_inst.set_atomic_add(0)
    else:
        # move out valid_x
        tik_inst.data_move(data_y[target_offset], ub_valid_x, 0, 1,
                           _ceil_div(target_loop_size, x_data_one_block), 0, 0)


def _normal_weight_nll_loss(tik_inst, block_idx, scalar_params, trans_params):
    """
    normal weight nllloss
    """
    need_core_num, c_size, _, core_loop_cnt, core_x_loop_size, core_target_loop_size, core_left_size, core_x_offset, \
        core_target_offset, core_x_left_offset, core_target_left_offset, target_index, x_size, target_size, \
        weight_size = scalar_params

    data_x, data_target, data_weight, data_y, data_total_weight, reduction = trans_params

    ub_zero_x, ub_x, ub_target, ub_zero_weight, ub_weight, ub_valid_x, ub_valid_weight, ub_work_space = \
        _init_normal_weight_ub(tik_inst, data_x, data_target, data_weight, x_size, target_size,
                               weight_size, reduction)

    with tik_inst.if_scope(block_idx < need_core_num):
        def _normal_weight(core_info, core_left_info):
            """
            detail process for normal weight
            """
            def _process_in_normal_weight(loop_cnt, x_loop_size, target_loop_size, x_offset, target_offset):
                """
                process data in normal weight
                """
                x_data_one_block = _get_element_cnt_one_block(data_x.dtype)
                weight_data_one_block = _get_element_cnt_one_block(data_weight.dtype)
                tik_inst.vector_dup(x_data_one_block, ub_zero_x, 0, 1, 1, 8)
                tik_inst.vector_dup(weight_data_one_block, ub_zero_weight, 0, 1, 1, 8)

                # every core one time copy weight to ub
                if reduction == "none":
                    _data_move_in_weight(tik_inst, data_weight, ub_weight, c_size)

                # every loop n is multiples of 32byte, if not process in one core
                with tik_inst.for_range(0, loop_cnt) as lp_cnt:
                    x_offset = x_offset + lp_cnt * x_loop_size
                    _data_move_in_x(tik_inst, data_x, ub_x, x_offset, x_loop_size)

                    target_offset = target_offset + lp_cnt * target_loop_size
                    _data_move_in_target(tik_inst, data_target, ub_target, target_offset, target_loop_size)

                    if reduction == "sum":
                        _data_move_in_weight(tik_inst, data_weight, ub_weight, c_size)

                    # get valid index from target
                    with tik_inst.for_range(0, target_loop_size) as n_lp_cnt:
                        target_index.set_as(ub_target[n_lp_cnt])

                        # process ignore index target
                        with tik_inst.if_scope(tik.any(target_index < 0, target_index >= c_size)):
                            target_index.set_as(-1)

                        # set valid x and weight
                        ub_valid_x[n_lp_cnt] = ub_x[n_lp_cnt * c_size + target_index]
                        ub_valid_weight[n_lp_cnt] = ub_weight[target_index]

                    # process valid data
                    _process_valid_data(tik_inst, data_x, data_weight, ub_x, ub_weight, ub_valid_x, ub_valid_weight,
                                        ub_work_space, target_loop_size, reduction)

                    # move out output
                    _data_move_out_output(tik_inst, data_x, data_y, data_total_weight, ub_x, ub_weight, ub_valid_x,
                                          reduction, target_offset, target_loop_size)

            def _process_left_data_in_normal_weight(left_size, x_left_offset, target_left_offset):
                """
                process left data in normal weight
                """
                with tik_inst.if_scope(left_size > 0):
                    _process_in_normal_weight(1, left_size * c_size, left_size, x_left_offset, target_left_offset)

            _process_in_normal_weight(*core_info)
            _process_left_data_in_normal_weight(*core_left_info)

        per_core_info = (core_loop_cnt, core_x_loop_size, core_target_loop_size,
                         core_x_offset, core_target_offset)
        per_core_left_info = (core_left_size, core_x_left_offset, core_target_left_offset)
        _normal_weight(per_core_info, per_core_left_info)


def _large_weight_nll_loss(tik_inst, block_idx, scalar_params, trans_params):
    """
    large weight nllloss
    """
    data_x, data_target, data_weight, data_y, data_total_weight, reduction = trans_params

    need_core_num, c_size, _, core_loop_cnt, core_x_loop_size, core_target_loop_size, core_left_size, core_x_offset, \
        core_target_offset, core_x_left_offset, core_target_left_offset, target_index, x_size, target_size, \
        weight_size = scalar_params

    ub_x, ub_target, ub_weight, ub_valid_x, ub_valid_weight, ub_work_space = \
        _init_large_weight_ub(tik_inst, data_x, data_target, data_weight, x_size, target_size,
                              weight_size, reduction)

    with tik_inst.if_scope(block_idx < need_core_num):
        def _large_weight(core_info, core_left_info):
            """
            detail process for large weight
            """

            def _process_in_large_weight(loop_cnt, x_loop_size, target_loop_size, x_offset, target_offset):
                """
                process data in large weight
                """
                x_data_one_block = _get_element_cnt_one_block(data_x.dtype)
                weight_data_one_block = _get_element_cnt_one_block(data_weight.dtype)

                # every loop n is multiples of 32byte, if not process in one core
                with tik_inst.for_range(0, loop_cnt) as lp_cnt:
                    target_offset = target_offset + lp_cnt * target_loop_size
                    _data_move_in_target(tik_inst, data_target, ub_target, target_offset, target_loop_size)

                    with tik_inst.for_range(0, target_loop_size) as n_lp_cnt:
                        target_index.set_as(ub_target[n_lp_cnt])

                        # process ignore index target
                        with tik_inst.if_scope(tik.any(target_index < 0, target_index >= c_size)):
                            tik_inst.vector_dup(x_data_one_block, ub_x, 0, 1, 1, 8)
                            tik_inst.vector_dup(weight_data_one_block, ub_weight, 0, 1, 1, 8)
                        with tik_inst.else_scope():
                            x_offset = x_offset + lp_cnt * x_loop_size + n_lp_cnt * c_size + target_index
                            tik_inst.data_move(ub_x, data_x[x_offset], 0, 1, 1, 0, 0)
                            tik_inst.data_move(ub_weight, data_weight[target_index], 0, 1, 1, 0, 0)

                        ub_valid_x[n_lp_cnt] = ub_x[0]
                        ub_valid_weight[n_lp_cnt] = ub_weight[0]

                    # process valid data
                    _process_valid_data(tik_inst, data_x, data_weight, ub_x, ub_weight, ub_valid_x, ub_valid_weight,
                                        ub_work_space, target_loop_size, reduction)

                    # move out output
                    _data_move_out_output(tik_inst, data_x, data_y, data_total_weight, ub_x, ub_weight, ub_valid_x,
                                          reduction, target_offset, target_loop_size)

            def _process_left_data_in_large_weight(left_size, x_left_offset, target_left_offset):
                """
                process left data in large weight
                """
                with tik_inst.if_scope(left_size > 0):
                    _process_in_large_weight(1, left_size * c_size, left_size, x_left_offset, target_left_offset)

            _process_in_large_weight(*core_info)
            _process_left_data_in_large_weight(*core_left_info)

        per_core_info = (core_loop_cnt, core_x_loop_size, core_target_loop_size, core_x_offset, core_target_offset)
        per_core_left_info = (core_left_size, core_x_left_offset, core_target_left_offset)
        _large_weight(per_core_info, per_core_left_info)


def _get_reduce_sum_ub_work_space(tik_inst, dtype, data_len):
    """
    get reduce sum ub work space
    """
    # use temp ub size
    ub_work_space_size = _get_reduce_sum_ub_work_space_size(dtype, data_len)
    ub_work_space = tik_inst.Tensor(dtype, (ub_work_space_size,), tik.scope_ubuf, "ub_work_space")
    return ub_work_space


def _init_normal_weight_ub(tik_inst, data_x, data_target, data_weight, x_size, target_size, weight_size, reduction):
    """
    init normal weight ub
    """
    ub_all_x = tik_inst.Tensor(data_x.dtype, (x_size + 8,), tik.scope_ubuf, "ub_all_x")
    ub_zero_x = ub_all_x[0:8]
    ub_x = ub_all_x[8:]
    ub_target = tik_inst.Tensor(data_target.dtype, (target_size,), tik.scope_ubuf, "ub_target")
    ub_all_weight = tik_inst.Tensor(data_weight.dtype, (weight_size + 8,), tik.scope_ubuf, "ub_all_weight")
    ub_zero_weight = ub_all_weight[0:8]
    ub_weight = ub_all_weight[8:]
    ub_valid_x = tik_inst.Tensor(data_x.dtype, (target_size,), tik.scope_ubuf, "ub_valid_x")
    ub_valid_weight = tik_inst.Tensor(data_weight.dtype, (target_size,), tik.scope_ubuf, "ub_valid_weight")
    ub_work_space = None
    if reduction == "sum":
        ub_work_space = _get_reduce_sum_ub_work_space(tik_inst, data_x.dtype, target_size)

    return [ub_zero_x, ub_x, ub_target, ub_zero_weight, ub_weight, ub_valid_x, ub_valid_weight, ub_work_space]


def _init_large_weight_ub(tik_inst, data_x, data_target, data_weight, x_size, target_size, weight_size, reduction):
    """
    init large weight ub
    """
    ub_x = tik_inst.Tensor(data_x.dtype, (x_size,), tik.scope_ubuf, "ub_x")
    ub_target = tik_inst.Tensor(data_target.dtype, (target_size,), tik.scope_ubuf, "ub_target")
    ub_weight = tik_inst.Tensor(data_weight.dtype, (weight_size,), tik.scope_ubuf, "ub_weight")
    ub_valid_x = tik_inst.Tensor(data_x.dtype, (target_size,), tik.scope_ubuf, "ub_valid_x")
    ub_valid_weight = tik_inst.Tensor(data_weight.dtype, (target_size,), tik.scope_ubuf, "ub_valid_weight")
    ub_work_space = None
    if reduction == "sum":
        ub_work_space = _get_reduce_sum_ub_work_space(tik_inst, data_x.dtype, target_size)

    return [ub_x, ub_target, ub_weight, ub_valid_x, ub_valid_weight, ub_work_space]


def nll_loss_compute(tik_inst, tensor_list, reduction):
    """
    do nll_loss
    """
    data_x, data_target, data_weight, data_y, data_total_weight, data_tiling = tensor_list

    with tik_inst.for_range(0, Constant.CORE_NUM, block_num=Constant.CORE_NUM) as block_idx:
        ub_tiling = tik_inst.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],), tik.scope_ubuf, "ub_tiling")
        tik_inst.data_move(ub_tiling, data_tiling, 0, 1, Constant.TILING_CTRL_PARAM[1] // Constant.TILING_CTRL_PARAM[2], 0, 0)
        tiling_reg_list = [None] * Constant.TILING_CTRL_PARAM[1]
        _get_tiling_params(tik_inst, ub_tiling, tiling_reg_list)
        scalar_params = _get_scalar_params(tik_inst, block_idx, tiling_reg_list, data_target.dtype)
        tiling_mode = tiling_reg_list[0]

        trans_params = [data_x, data_target, data_weight, data_y, data_total_weight, reduction]
        with tik_inst.if_scope(tiling_mode == 1):
            with tik_inst.new_stmt_scope():
                _normal_weight_nll_loss(tik_inst, block_idx, scalar_params, trans_params)
        with tik_inst.if_scope(tiling_mode == 2):
            with tik_inst.new_stmt_scope():
                _large_weight_nll_loss(tik_inst, block_idx, scalar_params, trans_params)


@register_operator("NLLLoss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def nll_loss(x, target, weight, y, total_weight, reduction="mean", ignore_index=-100, kernel_name="nll_loss"):
    """
    do nllloss by reduction and ignore_index attribute

    Parameters
    ----------
    x : dict
        shape and dtype of input x, the length of shape should be two or one.
    target : dict
        shape and dtype of input target, the length of shape only support one.
    weight : dict
        shape and dtype of input weight, the length of shape only support one.
    y:dict
        shape and dtype of output y.
        it's a tensor with shape(minibatch, ) when reduction == 'none' and
        the input is 2D. Otherwise, the output is a scalar.
    total_weight:
        shape and dtype of output total_weight, should be same type as weight.
        the output is scalar.
    reduction: str
        default value is "mean"
    ignore_index: int
        default value is -100
    kernel_name : str
        kernel name, default value is "nll_loss"

    Returns
    -------
    compile info
    """
    x_dtype = x.get("dtype").lower()
    target_dtype = target.get("dtype").lower()
    weight_dtype = weight.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    total_weight_dtype = total_weight.get("dtype").lower()
    _check_input_params(x_dtype, target_dtype, weight_dtype, y_dtype, total_weight_dtype, reduction)

    tik_inst = tik.Tik()
    data_x = tik_inst.Tensor(x_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_x")
    data_target = tik_inst.Tensor(target_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_target")
    data_weight = tik_inst.Tensor(weight_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_weight")
    data_tiling = tik_inst.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],), tik.scope_gm, "data_tiling")
    if reduction == "none":
        data_y = tik_inst.Tensor(y_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_y")
        data_total_weight = tik_inst.Tensor(total_weight_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_total_weight")
    else:
        data_y = tik_inst.Tensor(y_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_y", is_atomic_add=True)
        data_total_weight = tik_inst.Tensor(total_weight_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_total_weight",
                                            is_atomic_add=True)

    tensor_list = [data_x, data_target, data_weight, data_y, data_total_weight, data_tiling]

    nll_loss_compute(tik_inst, tensor_list, reduction)

    ub_size = _get_max_element_in_ub(x_dtype, 1)
    tbe_context.get_context().add_compile_info("vars", {"ub_size": ub_size, "core_num": Constant.CORE_NUM,
                                                        "reduction": reduction, "ignore_index": ignore_index})

    opt_config = {"enable_const_fold": True}
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[data_x, data_target, data_weight], outputs=[data_y, data_total_weight],
                      flowtable=[data_tiling], config=opt_config)

    return {"compile_info": tbe_context.get_context().get_compile_info()}
