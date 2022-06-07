#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved
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

from te import tik
import tbe.common.platform as tbe_platform
from tbe.common.utils import para_check


class Cst:
    """
    The class for Constant.
    """
    LEN = 2
    DIM_H2 = 4
    DIM_H1 = 3
    DIM_W2 = 5
    DIM_W1 = 2
    DIM_C = 1
    DIM_N = 0
    FULL_LINE = 8
    MAX_REPEAT = 255
    VEC_DUMP_SHAPE = 128
    VEC_MASK_FP16 = 128
    VEC_MASK_FP32 = 64
    VEC_MASK = 128
    BLOCK = 16


def cal_level(dividend):
    """
    cal_level
    """
    cnt = 0
    while dividend % Cst.LEN == 0:
        dividend //= Cst.LEN
        cnt += 1
    return cnt, dividend - 1


def gen_triu_mask(params_list, data_b):
    """
    gen_triu_mask
    """
    w_dim, line, input_shape, _, tik_instance = params_list
    tri_mask_ub = tik_instance.Tensor('float16',
                                      (w_dim, w_dim),
                                      name="tri_mask_ub",
                                      scope=tbe_platform.scope_ubuf)
    if w_dim <= Cst.VEC_MASK:
        tik_instance.vec_dup(w_dim, tri_mask_ub, tik_instance.Scalar(init_value=1, dtype="float16"),
                             w_dim, w_dim // Cst.BLOCK)
        with tik_instance.for_range(0, w_dim) as j:
            tik_instance.vec_dup(j + 1, tri_mask_ub[j * w_dim], tik_instance.Scalar(init_value=0, dtype="float16"),
                                 1, 0)
    else:
        repeat = w_dim * w_dim // Cst.VEC_MASK // Cst.MAX_REPEAT
        remain = w_dim * w_dim // Cst.VEC_MASK % Cst.MAX_REPEAT
        with tik_instance.for_range(0, repeat) as j:
            tik_instance.vec_dup(Cst.VEC_MASK, tri_mask_ub[Cst.VEC_MASK * Cst.MAX_REPEAT * j],
                                 tik_instance.Scalar(init_value=1, dtype="float16"),
                                 Cst.MAX_REPEAT, Cst.VEC_MASK // Cst.BLOCK)
        with tik_instance.if_scope(remain > 0):
            tik_instance.vec_dup(Cst.VEC_MASK, tri_mask_ub[Cst.VEC_MASK * Cst.MAX_REPEAT * repeat],
                                 tik_instance.Scalar(init_value=1, dtype="float16"),
                                 remain, Cst.VEC_MASK // Cst.BLOCK)
        with tik_instance.for_range(0, w_dim) as j:
            repeat = (j + 1) // Cst.VEC_MASK
            remain = (j + 1) % Cst.VEC_MASK
            with tik_instance.if_scope(repeat > 0):
                tik_instance.vec_dup(Cst.VEC_MASK, tri_mask_ub[j * w_dim],
                                     tik_instance.Scalar(init_value=0, dtype="float16"),
                                     repeat, Cst.VEC_MASK // Cst.BLOCK)
            with tik_instance.if_scope(remain > 0):
                tik_instance.vec_dup(remain, tri_mask_ub[j * w_dim + repeat * Cst.VEC_MASK],
                                     tik_instance.Scalar(init_value=0, dtype="float16"),
                                     1, 0)
    with tik_instance.for_range(0, w_dim // Cst.BLOCK) as j:
        tik_instance.data_move(data_b[w_dim * Cst.BLOCK * j], tri_mask_ub[Cst.BLOCK * j], 0, w_dim,
                               1, w_dim // Cst.BLOCK - 1, 0)


# 'pylint: disable=too-many-locals,too-many-statements
def data_move_in(offset, offset_mask, tensor_dtype, params_list, mov_list, fixed_triu_mask):
    """
    data_move_in
    """
    w_dim, line, input_shape, _, tik_instance = params_list
    tensor_input_x1, mask_input, ub_1, ub_3, ub_mask, ub_mask_fp16 = mov_list
    with tik_instance.if_scope(tensor_dtype == 'float16'):
        tik_instance.data_move(ub_1, tensor_input_x1[offset], 0, input_shape[Cst.DIM_W1],
                               Cst.BLOCK * line, (input_shape[Cst.DIM_H1] - line) * Cst.BLOCK, 0)
    with tik_instance.else_scope():
        tik_instance.data_move(ub_3, tensor_input_x1[offset], 0, input_shape[Cst.DIM_W1],
                               Cst.BLOCK * line * 2, (input_shape[Cst.DIM_H1] - line) * Cst.BLOCK * 2, 0)
        tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_1, ub_3,
                           line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32, 1, 1, 4, 8)
    if fixed_triu_mask:
        tik_instance.data_move(ub_mask_fp16, mask_input[offset_mask], 0, input_shape[Cst.DIM_W1], Cst.BLOCK * line,
                               (input_shape[Cst.DIM_H1] - line) * Cst.BLOCK, 0)
    else:
        tik_instance.data_move(ub_mask, mask_input[offset_mask], 0, input_shape[Cst.DIM_W1],
                               Cst.BLOCK * line // Cst.LEN,
                               (input_shape[Cst.DIM_H1] - line) * Cst.BLOCK // Cst.LEN, 0)
        tik_instance.vconv(Cst.VEC_MASK_FP16, "", ub_mask_fp16, ub_mask,
                           line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP16, 1, 1, 8, 4)


def data_move_in_w_large(offset, offset_mask, tensor_dtype, params_list, mov_list, fixed_triu_mask):
    """
    data_move_in_w_large
    """

    w_dim, line, input_shape, _, tik_instance = params_list
    times = line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32
    repeat_times = times // Cst.MAX_REPEAT
    remain = times % Cst.MAX_REPEAT
    tensor_input_x1, mask_input, ub_1, ub_3, ub_mask, ub_mask_fp16 = mov_list
    with tik_instance.if_scope(tensor_dtype == 'float16'):
        tik_instance.data_move(ub_1, tensor_input_x1[offset], 0, input_shape[Cst.DIM_W1],
                               Cst.BLOCK * line, (input_shape[Cst.DIM_H1] - line) * Cst.BLOCK, 0)
    with tik_instance.else_scope():
        tik_instance.data_move(ub_3, tensor_input_x1[offset], 0, input_shape[Cst.DIM_W1],
                               Cst.BLOCK * line * 2, (input_shape[Cst.DIM_H1] - line) * Cst.BLOCK * 2, 0)
        with tik_instance.for_range(0, repeat_times) as i:
            tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_1[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i], Cst.MAX_REPEAT, 1, 1, 4, 8)
        tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_1[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_times],
                           ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_times], remain, 1, 1, 4, 8)
    if fixed_triu_mask:
        tik_instance.data_move(ub_mask_fp16, mask_input[offset_mask], 0, input_shape[Cst.DIM_W1], Cst.BLOCK * line,
                               (input_shape[Cst.DIM_H1] - line) * Cst.BLOCK, 0)
    else:
        tik_instance.data_move(ub_mask, mask_input[offset_mask], 0, input_shape[Cst.DIM_W1],
                               Cst.BLOCK * line // Cst.LEN,
                               (input_shape[Cst.DIM_H1] - line) * Cst.BLOCK // Cst.LEN, 0)
        tik_instance.vconv(Cst.VEC_MASK_FP16, "", ub_mask_fp16, ub_mask,
                           line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP16, 1, 1, 8, 4)


def smooth_by_argmax(params_list, reduce_max_list):
    """
    smooth_by_argmax
    """
    w_dim, line, input_shape, _, tik_instance = params_list
    ub_1, ub_2, ub_reducemax, ub_broadcast, ub_dup = reduce_max_list
    cnt, remain = cal_level(input_shape[Cst.DIM_W1])
    time = tik_instance.Scalar("int32", name='time', init_value=Cst.LEN)
    tik_instance.vmax(Cst.VEC_MASK_FP16, ub_2, ub_1,
                      ub_1[w_dim * Cst.BLOCK * line // time],
                      w_dim * Cst.BLOCK * line // time // Cst.VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
    with tik_instance.for_range(1, cnt) as j:
        time.set_as(time * Cst.LEN)
        tik_instance.vmax(Cst.VEC_MASK_FP16, ub_2, ub_2,
                          ub_2[w_dim * Cst.BLOCK * line // time],
                          w_dim * Cst.BLOCK * line // time // Cst.VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
    with tik_instance.if_scope(remain > 0):
        with tik_instance.for_range(1, remain + 1) as j:
            tik_instance.vmax(Cst.VEC_MASK_FP16, ub_2[Cst.BLOCK * Cst.BLOCK * line * (remain - j)],
                              ub_2[Cst.BLOCK * Cst.BLOCK * line * (remain - j)],
                              ub_2[Cst.BLOCK * Cst.BLOCK * line * (remain - j + 1)],
                              Cst.BLOCK * Cst.BLOCK * line // Cst.VEC_MASK_FP16, 1, 1, 1, 8, 8, 8)
    tik_instance.vcgmax(Cst.VEC_MASK_FP16, ub_reducemax, ub_2, Cst.LEN * line, 1, 1, 8)

    tik_instance.vector_dup(Cst.VEC_DUMP_SHAPE, ub_dup, tik_instance.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)
    ub_reducemax_int16 = ub_reducemax.reinterpret_cast_to("uint16")
    with tik_instance.for_range(0, line) as j:
        tik_instance.vor(Cst.BLOCK, ub_broadcast[Cst.BLOCK * Cst.BLOCK * j],
                         ub_reducemax_int16[Cst.BLOCK * j], ub_dup, Cst.BLOCK,
                         1, 1, 0, 1, 0, 0)
    with tik_instance.for_range(0, line) as j:
        tik_instance.vtranspose(ub_broadcast[Cst.BLOCK * Cst.BLOCK * j], ub_broadcast[Cst.BLOCK * Cst.BLOCK * j])
    ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

    with tik_instance.for_range(0, line * Cst.BLOCK * Cst.BLOCK // Cst.VEC_MASK) as idx:
        tik_instance.vsub(Cst.VEC_MASK, ub_2[idx * Cst.VEC_MASK], ub_1[idx * Cst.VEC_MASK],
                          ub_broadcast_fp16[idx * Cst.VEC_MASK],
                          input_shape[Cst.DIM_W1], 1, 1, 1, line * Cst.BLOCK, line * Cst.BLOCK, 0)


def create_gm_tensor(tik_instance, tensor_shape, mask_shape, tensor_dtype):
    """
    create_gm_tensor
    """
    tensor_input_x1 = tik_instance.Tensor(tensor_dtype,
                                          tensor_shape,
                                          name="tensor_input_x1",
                                          scope=tbe_platform.scope_gm)
    tensor_mask = tik_instance.Tensor('bool',
                                      mask_shape,
                                      name="tensor_mask",
                                      scope=tbe_platform.scope_gm)

    tensor_output_y1 = tik_instance.Tensor('float16',
                                           tensor_shape,
                                           name="tensor_output_y1",
                                           scope=tbe_platform.scope_gm)
    gm_tensor_tuple = (tensor_input_x1, tensor_mask, tensor_output_y1)
    return gm_tensor_tuple


def do_exp_w_large(params_list, exp_list):
    """
    do_exp_w_large
    """
    w_dim, line, _, _, tik_instance = params_list
    ub_1, ub_2, ub_cast = exp_list
    repeat_time = line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32
    cnt = repeat_time // Cst.MAX_REPEAT
    remain = repeat_time % Cst.MAX_REPEAT
    with tik_instance.for_range(0, cnt) as i:
        tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_cast[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32 * i],
                           ub_2[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32 * i], Cst.MAX_REPEAT, 1, 1, 8, 4)
    tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_cast[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32 * cnt],
                       ub_2[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32 * cnt], remain, 1, 1, 8, 4)

    with tik_instance.for_range(0, cnt) as i:
        tik_instance.vexp(Cst.VEC_MASK_FP32, ub_cast[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32 * i],
                          ub_cast[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32 * i], Cst.MAX_REPEAT, 1, 1, 8, 8)
    tik_instance.vexp(Cst.VEC_MASK_FP32, ub_cast[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32 * cnt],
                      ub_cast[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32 * cnt], remain, 1, 1, 8, 8)


def do_exp(params_list, exp_list):
    """
    do_exp
    """
    w_dim, line, _, _, tik_instance = params_list
    ub_1, ub_2, ub_cast = exp_list
    repeat_time = line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32
    remain = repeat_time % Cst.MAX_REPEAT
    tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_cast, ub_2, remain, 1, 1, 8, 4)
    tik_instance.vexp(Cst.VEC_MASK_FP32, ub_cast, ub_cast, remain, 1, 1, 8, 8)


def calc_softmax(params_list, reduce_sum_and_div_list):
    """
    calc_softmax
    """
    w_dim, line, input_shape, _, tik_instance = params_list
    ub_cast, ub_mask_fp16, ub_reduceadd, ub_reduceadd_high_preci, \
    work_tensor_ub, ub_broadcast, ub_dup, ub_3, ub_dup_fp32, ub_1 = \
        reduce_sum_and_div_list
    time = tik_instance.Scalar("int32", name='time', init_value=1)
    cnt, remain = cal_level(input_shape[Cst.DIM_W1])
    repeat_remain = (line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32) % Cst.MAX_REPEAT
    tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_3, ub_mask_fp16, repeat_remain, 1, 1, 8, 4)
    tik_instance.vmul(Cst.VEC_MASK_FP32, ub_cast, ub_3, ub_cast, repeat_remain, 1, 1, 1, 8, 8, 8)
    tik_instance.vmuls(Cst.VEC_MASK_FP32, ub_3, ub_cast, tik_instance.Scalar(init_value=1, dtype="float32"),
                       repeat_remain, 1, 1, 8, 8)
    tik_instance.vadds(Cst.VEC_MASK_FP32, ub_cast, ub_cast, tik_instance.Scalar(init_value=1e-16, dtype="float32"),
                       repeat_remain, 1, 1, 8, 8)
    # reduce_add
    with tik_instance.for_range(0, cnt) as j:
        time.set_as(time * Cst.LEN)
        tik_instance.vadd(Cst.VEC_MASK_FP32, ub_cast, ub_cast,
                          ub_cast[w_dim * Cst.BLOCK * line // time],
                          w_dim * Cst.BLOCK * line // time // Cst.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    with tik_instance.if_scope(remain > 0):
        with tik_instance.for_range(1, remain + 1) as j:
            tik_instance.vadd(Cst.VEC_MASK_FP32, ub_cast[Cst.BLOCK * Cst.BLOCK * line * (remain - j)],
                              ub_cast[Cst.BLOCK * Cst.BLOCK * line * (remain - j)],
                              ub_cast[Cst.BLOCK * Cst.BLOCK * line * (remain - j + 1)],
                              Cst.BLOCK * Cst.BLOCK * line // Cst.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    tik_instance.vcadd(Cst.BLOCK, ub_reduceadd, ub_cast, Cst.BLOCK * line, 1, 1, 2)
    # vrec
    tik_instance.vec_rec_high_preci(line * Cst.BLOCK, ub_reduceadd_high_preci[0], ub_reduceadd[0],
                                    work_tensor_ub[0:], 1, 4, 4)
    #
    with tik_instance.for_range(0, line * input_shape[Cst.DIM_H2] / 8) as j:
        with tik_instance.for_range(0, 8) as k:
            tik_instance.vector_dup(16, ub_dup_fp32[j * 128 + 16 * k],
                                    tik_instance.Scalar(
                                        init_value=ub_reduceadd_high_preci[j * 8 + k], dtype="float32"),
                                    1, 1, 8)

    with tik_instance.for_range(0, line * Cst.BLOCK * Cst.BLOCK // Cst.VEC_MASK_FP32) as idx:
        tik_instance.vmul(Cst.VEC_MASK_FP32, ub_3[idx * Cst.VEC_MASK_FP32], ub_3[idx * Cst.VEC_MASK_FP32],
                          ub_dup_fp32[idx * Cst.VEC_MASK_FP32],
                          input_shape[Cst.DIM_W1], 1, 1, 1, line * Cst.BLOCK * 2, line * Cst.BLOCK * 2, 0)
    repeat_remain = (w_dim // 2) % Cst.MAX_REPEAT
    tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_1, ub_3, repeat_remain, 1, 1, 4, 8)


def masked_fill_w_large(params_list, reduce_sum_list):
    """
    masked_fill_w_large
    """
    w_dim, line, input_shape, _, tik_instance = params_list
    ub_cast, ub_mask_fp16, ub_reduceadd, ub_reduceadd_high_preci, \
    work_tensor_ub, ub_broadcast, ub_dup, ub_3, ub_dup_fp32, ub_1 = reduce_sum_list
    repeat_time = (line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32) // Cst.MAX_REPEAT
    repeat_remain = (line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32) % Cst.MAX_REPEAT
    with tik_instance.if_scope(repeat_time > 0):
        with tik_instance.for_range(0, repeat_time) as i:
            tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               ub_mask_fp16[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i], Cst.MAX_REPEAT, 1, 1, 8, 4)
    tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time],
                       ub_mask_fp16[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time], repeat_remain, 1, 1, 8, 4)
    with tik_instance.if_scope(repeat_time > 0):
        with tik_instance.for_range(0, repeat_time) as i:
            tik_instance.vmul(Cst.VEC_MASK_FP32, ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                              ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                              ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i], Cst.MAX_REPEAT,
                              1, 1, 1, 8, 8, 8)
    tik_instance.vmul(Cst.VEC_MASK_FP32, ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time],
                      ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time],
                      ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time],
                      repeat_remain, 1, 1, 1, 8, 8, 8)
    with tik_instance.if_scope(repeat_time > 0):
        with tik_instance.for_range(0, repeat_time) as i:
            tik_instance.vmuls(Cst.VEC_MASK_FP32, ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               tik_instance.Scalar(init_value=1, dtype="float32"), Cst.MAX_REPEAT, 1, 1, 8, 8)
    tik_instance.vmuls(Cst.VEC_MASK_FP32, ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time],
                       ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time],
                       tik_instance.Scalar(init_value=1, dtype="float32"), repeat_remain, 1, 1, 8, 8)
    with tik_instance.if_scope(repeat_time > 0):
        with tik_instance.for_range(0, repeat_time) as i:
            tik_instance.vadds(Cst.VEC_MASK_FP32, ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               tik_instance.Scalar(init_value=1e-16, dtype="float32"), Cst.MAX_REPEAT, 1, 1, 8, 8)
    tik_instance.vadds(Cst.VEC_MASK_FP32, ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time],
                       ub_cast[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time],
                       tik_instance.Scalar(init_value=1e-16, dtype="float32"), repeat_remain, 1, 1, 8, 8)


def calc_softmax_w_large(params_list, reduce_sum_list):
    """
    calc_softmax_w_large
    """
    w_dim, line, input_shape, _, tik_instance = params_list
    ub_cast, ub_mask_fp16, ub_reduceadd, ub_reduceadd_high_preci, \
    work_tensor_ub, ub_broadcast, ub_dup, ub_3, ub_dup_fp32, ub_1 = reduce_sum_list

    cnt, remain = cal_level(input_shape[Cst.DIM_W1])
    time = tik_instance.Scalar("int32", name='time', init_value=1)
    masked_fill_w_large(params_list, reduce_sum_list)
    # reduce_add
    with tik_instance.for_range(0, cnt) as j:
        time.set_as(time * Cst.LEN)
        tik_instance.vadd(Cst.VEC_MASK_FP32, ub_cast, ub_cast, ub_cast[w_dim * Cst.BLOCK * line // time],
                          w_dim * Cst.BLOCK * line // time // Cst.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    with tik_instance.if_scope(remain > 0):
        with tik_instance.for_range(1, remain + 1) as j:
            tik_instance.vadd(Cst.VEC_MASK_FP32, ub_cast[Cst.BLOCK * Cst.BLOCK * line * (remain - j)],
                              ub_cast[Cst.BLOCK * Cst.BLOCK * line * (remain - j)],
                              ub_cast[Cst.BLOCK * Cst.BLOCK * line * (remain - j + 1)],
                              Cst.BLOCK * Cst.BLOCK * line // Cst.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    tik_instance.vcadd(Cst.BLOCK, ub_reduceadd,
                       ub_cast, Cst.BLOCK * line, 1, 1, 2)
    # vrec
    tik_instance.vec_rec_high_preci(line * Cst.BLOCK, ub_reduceadd_high_preci[0],
                                    ub_reduceadd[0], work_tensor_ub[0:], 1, 4, 4)
    #
    with tik_instance.for_range(0, line * input_shape[Cst.DIM_H2] / 8) as j:
        with tik_instance.for_range(0, 8) as k:
            tik_instance.vector_dup(16, ub_dup_fp32[j * 128 + 16 * k],
                                    tik_instance.Scalar(
                                        init_value=ub_reduceadd_high_preci[j * 8 + k], dtype="float32"),
                                    1, 1, 8)
    with tik_instance.for_range(0, line * Cst.BLOCK * Cst.BLOCK // Cst.VEC_MASK_FP32) as idx:
        tik_instance.vmul(Cst.VEC_MASK_FP32, ub_3[idx * Cst.VEC_MASK_FP32], ub_3[idx * Cst.VEC_MASK_FP32],
                          ub_dup_fp32[idx * Cst.VEC_MASK_FP32], input_shape[Cst.DIM_W1],
                          1, 1, 1, line * Cst.BLOCK * 2,
                          line * Cst.BLOCK * 2, 0)
    repeat_time = (w_dim // 2) // Cst.MAX_REPEAT
    repeat_remain = (w_dim // 2) % Cst.MAX_REPEAT
    with tik_instance.if_scope(repeat_time > 0):
        with tik_instance.for_range(0, repeat_time) as i:
            tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_1[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i], Cst.MAX_REPEAT, 1, 1, 4, 8)
    tik_instance.vconv(Cst.VEC_MASK_FP32, "", ub_1[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time],
                       ub_3[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_time], repeat_remain, 1, 1, 4, 8)


def move_data_out(offset, params_list, ub_1, output1):
    """
    move_data_out
    """
    w_dim, line, input_shape, _, tik_instance = params_list
    tik_instance.data_move(output1[offset], ub_1, 0, input_shape[Cst.DIM_W1],
                           Cst.BLOCK * line, 0, (input_shape[Cst.DIM_H1] - line) * Cst.BLOCK)


def cal_prarms_list(input_tensor, mask_tensor, fixed_triu_mask):
    """
    cal_prarms_list
    """
    input_shape = input_tensor.get("shape")
    mask_shape = mask_tensor.get("shape")
    tik_instance = tik.Tik(tik.Dprofile(), disable_debug=True)
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    target = input_shape[Cst.DIM_N] * input_shape[Cst.DIM_C]
    remain_aicore_num = target
    iter_num = target // aicore_num + 1
    if fixed_triu_mask:
        broad_ratio = input_shape[Cst.DIM_C] * input_shape[Cst.DIM_N]
    else:
        broad_ratio = input_shape[Cst.DIM_C] * input_shape[Cst.DIM_N] // mask_shape[Cst.DIM_C] // mask_shape[Cst.DIM_N]
    ele_per_batch = input_shape[Cst.DIM_W1] * input_shape[Cst.DIM_H1] * input_shape[Cst.DIM_H2] * input_shape[
        Cst.DIM_W2]
    ele_per_core = ele_per_batch
    w_dim = input_shape[Cst.DIM_W1] * input_shape[Cst.DIM_W2]
    line = 2
    ranges = input_shape[Cst.DIM_H1] // line
    shape = (input_shape[Cst.DIM_W1], line * Cst.BLOCK, input_shape[Cst.DIM_W2])
    return [w_dim, line, input_shape, mask_shape, tik_instance], \
           [aicore_num, remain_aicore_num, iter_num, broad_ratio, ele_per_core, ele_per_batch, ranges, shape]


def create_ub(tik_instance, shape, line):
    """
    create_ub
    """
    ub_1 = tik_instance.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_1")
    ub_2 = tik_instance.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="ub_2")
    ub_cast = tik_instance.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="ub_cast")
    ub_3 = tik_instance.Tensor("float32", shape, scope=tbe_platform.scope_ubuf, name="ub_3")

    ub_reducemax = tik_instance.Tensor("float16", (line * Cst.BLOCK,),
                                       scope=tbe_platform.scope_ubuf, name="ub_reducemax")
    ub_reduceadd = tik_instance.Tensor("float32", (line * Cst.BLOCK,),
                                       scope=tbe_platform.scope_ubuf, name="ub_reduceadd")
    ub_reduceadd_high_preci = tik_instance.Tensor("float32", (line * Cst.BLOCK,),
                                                  scope=tbe_platform.scope_ubuf,
                                                  name="ub_reduceadd_high_preci")
    work_tensor_ub = tik_instance.Tensor("float32", (2 * line * Cst.BLOCK,), scope=tbe_platform.scope_ubuf,
                                         name="work_tensor_ub")
    ub_dup = tik_instance.Tensor("uint16", (Cst.VEC_DUMP_SHAPE,), scope=tbe_platform.scope_ubuf,
                                 name="ub_dup")
    ub_dup_fp32 = tik_instance.Tensor("float32", (32, 16), scope=tbe_platform.scope_ubuf,
                                      name="ub_dup_fp32")
    ub_broadcast = tik_instance.Tensor("uint16", (line * Cst.BLOCK * Cst.BLOCK,), scope=tbe_platform.scope_ubuf,
                                       name="ub_broadcast")

    ub_mask_fp16 = tik_instance.Tensor("float16", shape, scope=tbe_platform.scope_ubuf,
                                       name="ub_mask_fp16")
    ub_mask = tik_instance.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="ub_mask")
    offset = tik_instance.Scalar("int32", name="offset")
    offset_mask = tik_instance.Scalar("int32", name="offset_mask")

    return [ub_1, ub_2, ub_cast, ub_3, ub_reducemax, ub_reduceadd,
            ub_reduceadd_high_preci, work_tensor_ub, ub_dup, ub_dup_fp32,
            ub_broadcast, ub_mask_fp16, ub_mask, offset, offset_mask]


def scaled_masked_softmax_compute(params_list, core_attr_list, gm_tensors, scale, z, core_index, tensor_dtype):
    """
    implement scaled masked softmax algorithm in ub

    Parameters
    ----------
    params_list : List
        useful params list
    core_attr_list : List
        core attrs list
    gm_tensors : List
        gm tensors list
    scale : float
        input scale
    z : int
        iteration index
    core_index : int
        core index
    tensor_dtype : str
        dtype of input tensor
    Returns
    -------
    None
    """
    w_dim, line, tensor_shape, mask_shape, tik_instance = params_list
    aicore_num, remain_aicore_num, iter_num, broad_ratio, \
    ele_per_core, ele_per_batch, batch_range, shape = core_attr_list
    ub_1, ub_2, ub_cast, ub_3, ub_reducemax, ub_reduceadd, \
    ub_reduceadd_high_preci, work_tensor_ub, ub_dup, ub_dup_fp32, \
    ub_broadcast, ub_mask_fp16, ub_mask, offset, offset_mask = create_ub(tik_instance, shape, line)
    tensor_input_x1, data_b, output1, fixed_triu_mask = gm_tensors

    with tik_instance.if_scope(z * aicore_num + core_index < remain_aicore_num):
        with tik_instance.for_range(0, batch_range) as i:
            offset_mask.set_as(ele_per_core * ((z * aicore_num + core_index) // broad_ratio)
                               + i * line * Cst.BLOCK * Cst.BLOCK)
            offset.set_as(z * ele_per_core * aicore_num + core_index * ele_per_core + i * line * Cst.BLOCK * Cst.BLOCK)
            with tik_instance.if_scope(w_dim >= 512):
                data_move_in_w_large(offset, offset_mask, tensor_dtype, params_list,
                                     [tensor_input_x1, data_b, ub_1, ub_3, ub_mask, ub_mask_fp16], fixed_triu_mask)
                tik_instance.vmuls(Cst.VEC_MASK, ub_1, ub_1, tik_instance.Scalar(init_value=scale, dtype="float16"),
                                   w_dim // 4, 1, 1, 8, 8)
                tik_instance.vmuls(Cst.VEC_MASK, ub_mask_fp16, ub_mask_fp16,
                                   tik_instance.Scalar(init_value=-1., dtype="float16"),
                                   w_dim // 4, 1, 1, 8, 8)
                tik_instance.vadds(Cst.VEC_MASK, ub_mask_fp16, ub_mask_fp16,
                                   tik_instance.Scalar(init_value=1., dtype="float16"),
                                   w_dim // 4, 1, 1, 8, 8)
                smooth_by_argmax(params_list, [ub_1, ub_2, ub_reducemax, ub_broadcast, ub_dup])
                do_exp_w_large(params_list, [ub_1, ub_2, ub_cast])
                calc_softmax_w_large(params_list, [ub_cast, ub_mask_fp16, ub_reduceadd, ub_reduceadd_high_preci,
                                                   work_tensor_ub, ub_broadcast, ub_dup, ub_3, ub_dup_fp32, ub_1])
            with tik_instance.else_scope():
                data_move_in(offset, offset_mask, tensor_dtype, params_list,
                             [tensor_input_x1, data_b, ub_1, ub_3, ub_mask, ub_mask_fp16], fixed_triu_mask)
                tik_instance.vmuls(Cst.VEC_MASK, ub_1, ub_1, tik_instance.Scalar(init_value=scale, dtype="float16"),
                                   w_dim // 4, 1, 1, 8, 8)
                tik_instance.vmuls(Cst.VEC_MASK, ub_mask_fp16, ub_mask_fp16,
                                   tik_instance.Scalar(init_value=-1., dtype="float16"),
                                   w_dim // 4, 1, 1, 8, 8)
                tik_instance.vadds(Cst.VEC_MASK, ub_mask_fp16, ub_mask_fp16,
                                   tik_instance.Scalar(init_value=1., dtype="float16"),
                                   w_dim // 4, 1, 1, 8, 8)
                smooth_by_argmax(params_list, [ub_1, ub_2, ub_reducemax, ub_broadcast, ub_dup])
                do_exp(params_list, [ub_1, ub_2, ub_cast])
                calc_softmax(params_list,
                             [ub_cast, ub_mask_fp16, ub_reduceadd, ub_reduceadd_high_preci, work_tensor_ub,
                              ub_broadcast, ub_dup, ub_3, ub_dup_fp32, ub_1])
            move_data_out(offset, params_list, ub_1, output1)


# 'pylint: disable=unused-argument,too-many-arguments
# 'pylint: disable=too-many-locals,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scaled_masked_softmax(x, mask, y, scale=1.0, fixed_triu_mask=False,
                          kernel_name="scaled_masked_softmax"):
    """
    Algorithm:
        mask = torch.triu(mask.shape, diagonal=1) if fixed_triu_mask else mask
        y = torch.softmax((x * scale).masked_fill(mask, -inf), dim=-1) 

    Parameters
    ----------
    x : dict
        shape and dtype of input tensor, the shape must be 6D in format Fractal_NZ.
    mask : dict
        shape and dtype of mask, the shape must be broadcastble with x.
    y : dict
        shape and dtype of output, the shape must be same as x.
    scale : float
        a float scalar scaling the input tensor x
    fixed_triu_mask : bool
        if true: the mask is a fixed upper triangle mask
        if false: the mask is input mask
    kernel_name : str
        kernel name, default value is "scaled_masked_softmax"

    Returns
    -------
    None
    """
    tensor_dtype = x.get("dtype").lower()
    params_list, core_attr_list = cal_prarms_list(x, mask, fixed_triu_mask)
    w_dim, line, tensor_shape, mask_shape, tik_instance = params_list
    aicore_num, remain_aicore_num, iter_num, broad_ratio, \
    ele_per_core, ele_per_batch, batch_range, shape = core_attr_list
    tensor_input_x1, mask_input, output1 = \
        create_gm_tensor(tik_instance, tensor_shape, mask_shape, tensor_dtype)
    if fixed_triu_mask:
        data_b = mask_input.reinterpret_cast_to('float16')
        gen_triu_mask(params_list, data_b)
        gm_tensors = [tensor_input_x1, data_b, output1, fixed_triu_mask]
    else:
        data_b = mask_input.reinterpret_cast_to('uint8')
        gm_tensors = [tensor_input_x1, data_b, output1, fixed_triu_mask]
    with tik_instance.for_range(0, iter_num) as z:
        with tik_instance.for_range(0, aicore_num, block_num=aicore_num) as core_index:
            scaled_masked_softmax_compute(params_list, core_attr_list,
                                          gm_tensors, scale, z, core_index, tensor_dtype)
    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[tensor_input_x1, mask_input], outputs=[output1])
    return tik_instance