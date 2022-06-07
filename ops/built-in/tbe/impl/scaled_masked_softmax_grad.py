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

import te.platform as tbe_platform
from te import tik
from te.utils import para_check


class Cst:
    """
    The class for Constant.
    """
    SHAPE_SIZE_LIMIT = 1 << 30
    BLOCK = 16
    VEC_MASK = 128
    VEC_MASK_FP32 = 64
    VEC_DUMP_SHAPE = 32
    MAX_REPEAT = 255
    FULL_LINE = 8
    DIM_N = 0
    DIM_C = 1
    DIM_W1 = 2
    DIM_W2 = 5
    DIM_H1 = 3
    DIM_H2 = 4
    LEN = 2


def cal_level(dividend):
    """
    cal_level
    """
    cnt = 0
    while dividend % Cst.LEN == 0:
        dividend //= Cst.LEN
        cnt += 1
    return cnt, dividend - 1


# 'pylint: disable=too-many-locals,too-many-statements
def data_move_in(offset, offset_mask, tensor_dtype, para_lis, mov_lis, fixed_triu_mask):
    """
    data_move_in
    """
    w_dim, line, grad_shape, _, tik_inst = para_lis
    grad_ub, input_ub_fp32, grad_gm, softmax_ub, softmax_ub_fp32, y_gm, mask_ub, mask_gm = mov_lis
    times = line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32
    repeat_times = times // Cst.MAX_REPEAT
    remain_times = times % Cst.MAX_REPEAT

    with tik_inst.if_scope(tensor_dtype == 'float16'):
        tik_inst.data_move(grad_ub, grad_gm[offset], 0, grad_shape[Cst.DIM_W1],
                           Cst.BLOCK * line, (grad_shape[Cst.DIM_H1] - line) * Cst.BLOCK, 0)
        tik_inst.data_move(softmax_ub, y_gm[offset], 0, grad_shape[Cst.DIM_W1],
                           Cst.BLOCK * line, (grad_shape[Cst.DIM_H1] - line) * Cst.BLOCK, 0)
    with tik_inst.else_scope():
        tik_inst.data_move(input_ub_fp32, grad_gm[offset], 0, grad_shape[Cst.DIM_W1],
                           Cst.BLOCK * line * 2, (grad_shape[Cst.DIM_H1] - line) * Cst.BLOCK * 2, 0)
        with tik_inst.if_scope(repeat_times > 0):
            with tik_inst.for_range(0, repeat_times) as i:
                tik_inst.vconv(Cst.VEC_MASK_FP32, "", grad_ub[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               input_ub_fp32[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i], Cst.MAX_REPEAT, 1, 1, 4, 8)
        tik_inst.vconv(Cst.VEC_MASK_FP32, "", grad_ub[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_times],
                       input_ub_fp32[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_times], remain_times, 1, 1, 4, 8)
        tik_inst.data_move(softmax_ub_fp32, y_gm[offset], 0, grad_shape[Cst.DIM_W1],
                           Cst.BLOCK * line * 2, (grad_shape[Cst.DIM_H1] - line) * Cst.BLOCK * 2, 0)
        with tik_inst.if_scope(repeat_times > 0):
            with tik_inst.for_range(0, repeat_times) as i:
                tik_inst.vconv(Cst.VEC_MASK_FP32, "", softmax_ub[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               softmax_ub_fp32[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * i],
                               Cst.MAX_REPEAT, 1, 1, 4, 8)
        tik_inst.vconv(Cst.VEC_MASK_FP32, "", softmax_ub[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_times],
                       softmax_ub_fp32[Cst.VEC_MASK_FP32 * Cst.MAX_REPEAT * repeat_times], remain_times, 1, 1, 4, 8)
    if not fixed_triu_mask:
        tik_inst.data_move(mask_ub, mask_gm[offset_mask], 0, grad_shape[Cst.DIM_W1],
                           Cst.BLOCK * line // Cst.LEN, (grad_shape[Cst.DIM_H1] - line) * Cst.BLOCK // Cst.LEN, 0)


def calc_product(para_lis, output_ub_fp32, grad_ub, softmax_ub_fp32, softmax_ub):
    """
    calc_product
    """
    w_dim, line, _, _, tik_inst = para_lis
    if line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32 > Cst.MAX_REPEAT:
        tik_inst.vconv(Cst.VEC_MASK_FP32, "", output_ub_fp32, grad_ub, Cst.MAX_REPEAT, 1, 1, 8, 4)
        tik_inst.vconv(Cst.VEC_MASK_FP32, "", output_ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32],
                       grad_ub[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32], 1, 1, 1, 8, 4)
        tik_inst.vconv(Cst.VEC_MASK_FP32, "", softmax_ub_fp32, softmax_ub, Cst.MAX_REPEAT, 1, 1, 8, 4)
        tik_inst.vconv(Cst.VEC_MASK_FP32, "", softmax_ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32],
                       softmax_ub[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32], 1, 1, 1, 8, 4)
        tik_inst.vec_mul(Cst.VEC_MASK_FP32, output_ub_fp32, softmax_ub_fp32, output_ub_fp32, Cst.MAX_REPEAT,
                         8, 8, 8)
        tik_inst.vec_mul(Cst.VEC_MASK_FP32, output_ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32],
                         softmax_ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32],
                         output_ub_fp32[Cst.MAX_REPEAT * Cst.VEC_MASK_FP32], 1, 8, 8, 8)
    else:
        tik_inst.vconv(Cst.VEC_MASK_FP32, "", output_ub_fp32, grad_ub, line * Cst.BLOCK * w_dim
                       // Cst.VEC_MASK_FP32, 1, 1, 8, 4)
        tik_inst.vconv(Cst.VEC_MASK_FP32, "", softmax_ub_fp32, softmax_ub, line * Cst.BLOCK * w_dim
                       // Cst.VEC_MASK_FP32, 1, 1, 8, 4)
        tik_inst.vec_mul(Cst.VEC_MASK_FP32, output_ub_fp32, softmax_ub_fp32, output_ub_fp32,
                         line * Cst.BLOCK * w_dim // Cst.VEC_MASK_FP32, 8, 8, 8)


def calc_reducesum(para_lis, output_ub_fp32, ub_reduce_add, ub_reduceadd_fp16, ub_broadcast, ub_dup):
    """
    calc_reducesum
    """
    w_dim, line, grad_shape, _, tik_inst = para_lis
    cnt, remain = cal_level(grad_shape[Cst.DIM_W1])
    time = tik_inst.Scalar("int32", name='time', init_value=1)
    with tik_inst.for_range(0, cnt) as j:
        time.set_as(time * Cst.LEN)
        tik_inst.vadd(Cst.VEC_MASK_FP32, output_ub_fp32, output_ub_fp32,
                      output_ub_fp32[w_dim * Cst.BLOCK * line // time],
                      w_dim * Cst.BLOCK * line // time // Cst.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    with tik_inst.if_scope(remain > 0):
        with tik_inst.for_range(1, remain + 1) as j:
            tik_inst.vadd(Cst.VEC_MASK_FP32, output_ub_fp32[Cst.BLOCK * Cst.BLOCK * line * (remain - j)],
                          output_ub_fp32[Cst.BLOCK * Cst.BLOCK * line * (remain - j)],
                          output_ub_fp32[Cst.BLOCK * Cst.BLOCK * line * (remain - j + 1)],
                          Cst.BLOCK * Cst.BLOCK * line // Cst.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
    tik_inst.vcadd(Cst.BLOCK, ub_reduce_add, output_ub_fp32, Cst.BLOCK * line, 1, 1, 2)
    tik_inst.vconv(Cst.BLOCK * line, "", ub_reduceadd_fp16, ub_reduce_add, 1, 1, 1, 0, 0)
    tik_inst.vector_dup(Cst.VEC_DUMP_SHAPE, ub_dup, tik_inst.Scalar(init_value=0, dtype="int16"), 1, 1, 8)
    ub_reduceadd_int16 = ub_reduceadd_fp16.reinterpret_cast_to("int16")
    with tik_inst.for_range(0, line) as j:
        tik_inst.vor(Cst.BLOCK, ub_broadcast[Cst.BLOCK * Cst.BLOCK * j], ub_reduceadd_int16[Cst.BLOCK * j], ub_dup,
                     Cst.BLOCK,
                     1, 1, 0, 1, 0, 0)
    with tik_inst.for_range(0, line) as j:
        tik_inst.vtranspose(ub_broadcast[Cst.BLOCK * Cst.BLOCK * j], ub_broadcast[Cst.BLOCK * Cst.BLOCK * j])


def calc_softmax_grad_and_masked_fill(offset, para_lis, ub_broadcast, grad_ub, mask_ub, softmax_ub, y_gm, ex_params):
    """
    calc_softmax_grad_and_masked_fill
    """
    w_dim, line, grad_shape, _, tik_inst = para_lis
    scale, fixed_triu_mask, mask_gm, offset_mask = ex_params
    ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")
    with tik_inst.for_range(0, line * Cst.BLOCK * Cst.BLOCK // Cst.VEC_MASK) as idx:
        tik_inst.vsub(Cst.VEC_MASK, grad_ub[idx * Cst.VEC_MASK], grad_ub[idx * Cst.VEC_MASK],
                      ub_broadcast_fp16[idx * Cst.VEC_MASK],
                      grad_shape[Cst.DIM_W1], 1, 1, 1, line * Cst.BLOCK, line * Cst.BLOCK, 0)
    tik_inst.vec_mul(Cst.VEC_MASK, grad_ub, grad_ub, softmax_ub,
                     line * Cst.BLOCK * w_dim // Cst.VEC_MASK, 8, 8, 8)
    if fixed_triu_mask:
        tik_inst.data_move(softmax_ub, mask_gm[offset_mask], 0, grad_shape[Cst.DIM_W1],
                           Cst.BLOCK * line, (grad_shape[Cst.DIM_H1] - line) * Cst.BLOCK, 0)
    else:
        tik_inst.vconv(Cst.VEC_MASK, '', softmax_ub, mask_ub,
                       line * Cst.BLOCK * w_dim // Cst.VEC_MASK, 1, 1, 8, 4)
    tik_inst.vmuls(Cst.VEC_MASK, softmax_ub, softmax_ub,
                   tik_inst.Scalar(init_value=-1., dtype="float16"),
                   line * Cst.BLOCK * w_dim // Cst.VEC_MASK, 1, 1, 8, 8)
    tik_inst.vadds(Cst.VEC_MASK, softmax_ub, softmax_ub,
                   tik_inst.Scalar(init_value=1., dtype="float16"),
                   line * Cst.BLOCK * w_dim // Cst.VEC_MASK, 1, 1, 8, 8)
    tik_inst.vec_mul(Cst.VEC_MASK, grad_ub, grad_ub, softmax_ub,
                     line * Cst.BLOCK * w_dim // Cst.VEC_MASK, 8, 8, 8)
    tik_inst.vec_muls(Cst.VEC_MASK, grad_ub, grad_ub,
                      tik_inst.Scalar(init_value=scale, dtype="float16"),
                      line * Cst.BLOCK * w_dim // Cst.VEC_MASK, 8, 8)
    tik_inst.data_move(y_gm[offset], grad_ub, 0, grad_shape[Cst.DIM_W1],
                       Cst.BLOCK * line, 0, (grad_shape[Cst.DIM_H1] - line) * Cst.BLOCK)


def paras_check(y_grad, mask):
    """
    paras_check
    """
    para_check.check_dtype_rule(y_grad.get('dtype').lower(), ("float16", 'float32'), param_name='y_grad')
    para_check.check_dtype_rule(mask.get('dtype').lower(), ("uint8", 'bool', 'int8'), param_name='mask')
    para_check.check_shape_rule(y_grad.get('shape'), max_shape_num=Cst.SHAPE_SIZE_LIMIT)
    para_check.check_shape_rule(mask.get('shape'), max_shape_num=Cst.SHAPE_SIZE_LIMIT)


def cal_prarms_list(y_grad, mask, fixed_triu_mask):
    """
    cal_prarms_list
    """
    grad_shape = y_grad.get("shape")
    mask_shape = mask.get("shape")
    tik_inst = tik.Tik(tik.Dprofile(), disable_debug=True)
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    target = grad_shape[Cst.DIM_N] * grad_shape[Cst.DIM_C]
    remain_aicore_num = target
    iter_num = target // aicore_num + 1
    if fixed_triu_mask:
        broad_ratio = grad_shape[Cst.DIM_C] * grad_shape[Cst.DIM_N]
    else:
        broad_ratio = grad_shape[Cst.DIM_C] * grad_shape[Cst.DIM_N] // mask_shape[Cst.DIM_C] // mask_shape[Cst.DIM_N]
    ele_per_core = grad_shape[Cst.DIM_W1] * grad_shape[Cst.DIM_H1] * grad_shape[Cst.DIM_H2] * grad_shape[Cst.DIM_W2]
    ele_per_batch = ele_per_core
    w_dim = grad_shape[Cst.DIM_W1] * grad_shape[Cst.DIM_W2]
    line = 2
    ranges = grad_shape[Cst.DIM_H1] // line
    shape = (grad_shape[Cst.DIM_W1], line * Cst.BLOCK, grad_shape[Cst.DIM_W2])
    params_list = [w_dim, line, grad_shape, mask_shape, tik_inst]
    core_attr_list = [aicore_num, remain_aicore_num, iter_num, broad_ratio,
                      ele_per_core, ele_per_batch, ranges, shape]
    return params_list, core_attr_list


def create_ub(tik_inst, shape, line):
    """
    create_ub
    """
    offset_mask = tik_inst.Scalar("int32", name="offset_mask")
    grad_ub = tik_inst.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="grad_ub")
    mask_ub = tik_inst.Tensor("uint8", shape, scope=tbe_platform.scope_ubuf, name="mask_ub")
    softmax_ub = tik_inst.Tensor("float16", shape, scope=tbe_platform.scope_ubuf, name="softmax_ub")
    softmax_ub_fp32 = tik_inst.Tensor("float32", shape, scope=tbe_platform.scope_ubuf,
                                      name="softmax_ub_fp32")
    output_ub_fp32 = tik_inst.Tensor("float32", shape, scope=tbe_platform.scope_ubuf,
                                     name="output_ub_fp32")
    ub_reduce_add = tik_inst.Tensor("float32", (line * Cst.BLOCK,), tik.scope_ubuf, "work_tensor_ub")
    ub_reduceadd_fp16 = tik_inst.Tensor("float16", (line * Cst.BLOCK,), tik.scope_ubuf, "dst_ub")
    ub_dup = tik_inst.Tensor("int16", (Cst.VEC_DUMP_SHAPE,), scope=tbe_platform.scope_ubuf, name="ub_dup")
    ub_broadcast = tik_inst.Tensor("int16", (line * Cst.BLOCK, Cst.BLOCK), scope=tbe_platform.scope_ubuf,
                                   name="ub_broadcast")
    offset = tik_inst.Scalar("int32", name="offset")
    return [offset_mask, grad_ub, mask_ub, softmax_ub, softmax_ub_fp32,
            output_ub_fp32, ub_reduce_add, ub_reduceadd_fp16,
            ub_dup, ub_broadcast, offset]


def gen_triu_mask(params_list, data_mask_gm):
    """
    gen_triu_mask
    """
    w_dim, line, grad_shape, _, tik_inst = params_list
    tri_mask_ub = tik_inst.Tensor('float16',
                                  (w_dim, w_dim),
                                  name="tri_mask_ub",
                                  scope=tbe_platform.scope_ubuf)
    if w_dim <= Cst.VEC_MASK:
        tik_inst.vec_dup(w_dim, tri_mask_ub, tik_inst.Scalar(init_value=1, dtype="float16"),
                         w_dim, w_dim // Cst.BLOCK)
        with tik_inst.for_range(0, w_dim) as j:
            tik_inst.vec_dup(j + 1, tri_mask_ub[j * w_dim], tik_inst.Scalar(init_value=0, dtype="float16"),
                             1, 0)
    else:
        repeat = w_dim * w_dim // Cst.VEC_MASK // Cst.MAX_REPEAT
        remain = w_dim * w_dim // Cst.VEC_MASK % Cst.MAX_REPEAT
        with tik_inst.for_range(0, repeat) as j:
            tik_inst.vec_dup(Cst.VEC_MASK, tri_mask_ub[Cst.VEC_MASK * Cst.MAX_REPEAT * j],
                             tik_inst.Scalar(init_value=1, dtype="float16"),
                             Cst.MAX_REPEAT, Cst.VEC_MASK // Cst.BLOCK)
        with tik_inst.if_scope(remain > 0):
            tik_inst.vec_dup(Cst.VEC_MASK, tri_mask_ub[Cst.VEC_MASK * Cst.MAX_REPEAT * repeat],
                             tik_inst.Scalar(init_value=1, dtype="float16"),
                             remain, Cst.VEC_MASK // Cst.BLOCK)
        with tik_inst.for_range(0, w_dim) as j:
            repeat = (j + 1) // Cst.VEC_MASK
            remain = (j + 1) % Cst.VEC_MASK
            with tik_inst.if_scope(repeat > 0):
                tik_inst.vec_dup(Cst.VEC_MASK, tri_mask_ub[j * w_dim],
                                 tik_inst.Scalar(init_value=0, dtype="float16"),
                                 repeat, Cst.VEC_MASK // Cst.BLOCK)
            with tik_inst.if_scope(remain > 0):
                tik_inst.vec_dup(remain, tri_mask_ub[j * w_dim + repeat * Cst.VEC_MASK],
                                 tik_inst.Scalar(init_value=0, dtype="float16"),
                                 1, 0)
    with tik_inst.for_range(0, w_dim // Cst.BLOCK) as j:
        tik_inst.data_move(data_mask_gm[w_dim * Cst.BLOCK * j], tri_mask_ub[Cst.BLOCK * j], 0, w_dim,
                           1, w_dim // Cst.BLOCK - 1, 0)


def scaled_masked_softmax_grad_compute(params_lis, core_attr_lis, gm_list, scale, grad_dtype, z, core_index):
    """
    implement scaled masked softmax backward algorithm in ub

    Parameters
    ----------
    params_list : List
        useful params list
    core_attr_list : List
        core attrs list
    gm_list : List
        gm tensors list
    scale : float
        input scale
    z : int
        iteration index
    core_index : int
        core index
    grad_dtype : str
        dtype of input grad tensor
    Returns
    -------
    None
    """
    w_dim, line, grad_shape, mask_shape, tik_inst = params_lis
    aicore_num, remain_aicore_num, iter_num, broad_ratio, ele_per_core, \
    ele_per_batch, batch_range, shape = core_attr_lis
    y_grad_gm, y_gm, data_mask_gm, x_grad_gm, fixed_triu_mask = gm_list
    offset_mask, grad_ub, mask_ub, softmax_ub, softmax_ub_fp32, \
    output_ub_fp32, ub_reduce_add, ub_reduceadd_fp16, \
    ub_dup, ub_broadcast, offset = create_ub(tik_inst, shape, line)

    with tik_inst.if_scope(z * aicore_num + core_index < remain_aicore_num):
        with tik_inst.for_range(0, batch_range) as i:
            offset.set_as(
                z * ele_per_core * aicore_num + core_index * ele_per_core + i * line * Cst.BLOCK * Cst.BLOCK)
            offset_mask.set_as(
                ele_per_core * ((z * aicore_num + core_index) // broad_ratio) + i * line * Cst.BLOCK * Cst.BLOCK)
            move_list = [grad_ub, output_ub_fp32, y_grad_gm, softmax_ub, softmax_ub_fp32,
                         y_gm, mask_ub, data_mask_gm]
            data_move_in(offset, offset_mask, grad_dtype, params_lis, move_list, fixed_triu_mask)
            calc_product(params_lis, output_ub_fp32, grad_ub, softmax_ub_fp32, softmax_ub)
            calc_reducesum(params_lis, output_ub_fp32, ub_reduce_add, ub_reduceadd_fp16,
                           ub_broadcast, ub_dup)
            calc_softmax_grad_and_masked_fill(offset, params_lis, ub_broadcast, grad_ub, mask_ub, softmax_ub,
                                              x_grad_gm, [scale, fixed_triu_mask, data_mask_gm, offset_mask])


# 'pylint: disable=unused-argument,too-many-arguments, disable=too-many-locals,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def scaled_masked_softmax_grad(y_grad, y, mask,
                               x_grad, scale, fixed_triu_mask=False,
                               kernel_name="scaled_masked_softmax_grad"):
    """
    Algorithm:
        mask = torch.triu(mask.shape, diagonal=1) if fixed_triu_mask else mask
        x_grad = (y_grad - (y_grad * y).sum(-1).unsqueeze(-1)) * y
        x_grad = (x_grad * scale).masked_fill(mask, 0)

    Parameters
    ----------
    y_grad : dict
        shape and dtype of input grad tensor, the shape must be 6D in format Fractal_NZ.
    y : dict
        shape and dtype of forward output tensor, the shape must be same as y_grad.
    mask : dict
        shape and dtype of mask, the shape must be broadcastble with y_grad.
    x_grad : dict
        shape and dtype of output grad tensor, the shape must be same as y_grad.
    scale : float
        a float scalar scaling the input_grad. 
    fixed_triu_mask : bool
        if true: the mask is a fixed upper triangle mask
        if false: the mask is input mask
    kernel_name : str
        kernel name, default value is "scaled_masked_softmax_grad"

    Returns
    -------
    None
    """
    paras_check(y_grad, mask)
    grad_dtype = y_grad.get("dtype").lower()
    mask_dtype = mask.get("dtype").lower()
    softmax_output_dtype = y.get("dtype").lower()
    params_lis, core_attr_lis = cal_prarms_list(y_grad, mask, fixed_triu_mask)
    w_dim, line, grad_shape, mask_shape, tik_inst = params_lis
    aicore_num, remain_aicore_num, iter_num, broad_ratio, ele_per_core, \
    ele_per_batch, batch_range, shape = core_attr_lis
    y_grad_gm = tik_inst.Tensor(grad_dtype, grad_shape, name="y_grad_gm", scope=tbe_platform.scope_gm)
    mask_gm = tik_inst.Tensor('bool', mask_shape, name="mask_gm", scope=tbe_platform.scope_gm)
    y_gm = tik_inst.Tensor(softmax_output_dtype, grad_shape, name="y_gm", scope=tbe_platform.scope_gm)
    x_grad_gm = tik_inst.Tensor('float16', grad_shape, name="x_grad_gm", scope=tbe_platform.scope_gm)
    if fixed_triu_mask:
        data_mask_gm = mask_gm.reinterpret_cast_to('float16')
        gen_triu_mask(params_lis, data_mask_gm)
    else:
        data_mask_gm = mask_gm.reinterpret_cast_to('uint8')
    with tik_inst.for_range(0, iter_num) as z:
        with tik_inst.for_range(0, aicore_num, block_num=aicore_num) as core_index:
            scaled_masked_softmax_grad_compute(params_lis, core_attr_lis,
                                               [y_grad_gm, y_gm, data_mask_gm, x_grad_gm, fixed_triu_mask],
                                               scale, grad_dtype, z, core_index)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[y_grad_gm, y_gm, mask_gm], outputs=[x_grad_gm, ])
    return tik_inst