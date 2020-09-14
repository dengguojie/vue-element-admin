#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licenses.
"""

import math
from te import platform as cce
from te import tvm

ASCEND_QUANT_TAG = "quant"
POOLING3D_TAG_PREFIX = "pooling3d_"
ASCEND_ANTI_QUANT_TAG = "anti_quant"
C0_DIMENSION_DATA_SIZE = 32
BLOCK_SIZE = 16
MAX_BUILD_ROUND_FOR_RECALC_UB = 8


# 'pylint: disable=invalid-name,
def _build_context(res):
    pool_params = res.op.attrs["pooling_params"]
    p_n, p_c1 = pool_params["batch_size"].value, pool_params["c1_value"].value
    p_d, p_h, p_w = pool_params["in_size_d"].value, \
                    pool_params["in_size_h"].value, \
                    pool_params["in_size_w"].value
    o_d, o_h, o_w = pool_params["out_size_d"].value, \
                    pool_params["out_size_h"].value, \
                    pool_params["out_size_w"].value
    k_d, k_h, k_w = pool_params["window_d"].value, \
                    pool_params["window_h"].value, \
                    pool_params["window_w"].value
    s_d, s_h, s_w = pool_params["stride_d"].value, \
                    pool_params["stride_h"].value, \
                    pool_params["stride_w"].value

    context = {}
    context["align_axis"] = pool_params["align_axis"].value
    context["fast_path"] = pool_params["fast_path"].value
    context["n"], context["c1"] = p_n, p_c1
    context["d"], context["h"], context["w"] = p_d, p_h, p_w
    context["do"], context["ho"], context["wo"] = o_d, o_h, o_w
    context["kd"], context["kh"], context["kw"] = k_d, k_h, k_w
    context["sd"], context["sh"], context["sw"] = s_d, s_h, s_w
    context["round"] = _get_build_round(res)
    return context


# 'pylint: disable=too-many-locals,too-many-statements
def pooling3d_max_grad_grad_schedule(res, sch_list):
    """
    :params:
    :res: result tensor
    :sch_list: schedule list
    :return: True
    """
    sch = sch_list[0]
    context = _build_context(res)
    pool_tensors = _crawl_pool_tensor(res)
    _set_scope(sch, pool_tensors.values(), cce.scope_ubuf)

    def _fast_path_schedule():
        tx_grad_grad_c = pool_tensors["tx_grad_grad_c"]
        tx_grad_grad = pool_tensors["tx_grad_grad"]
        fast_path_res = pool_tensors["fast_path_res"]
        sch[tx_grad_grad_c].emit_insn(tx_grad_grad_c.op.axis[0], 'dma_copy', _split_select())
        sch[tx_grad_grad].emit_insn(tx_grad_grad.op.axis[0], "phony_insn")
        sch[fast_path_res].emit_insn(fast_path_res.op.axis[0], 'vector_auto')
        sch[res].emit_insn(d_in, 'dma_copy')
        _mem_reuse(sch, [tx_grad_grad], tx_grad_grad_c)
        return True


    def _split_for_dma_copy():
        ft_d = context["kd"]
        ft_h = context["kh"]
        ft_w = context["kw"]

        sch[tx_orig_in_ext].split(tx_orig_in_ext.op.axis[2], factor=ft_d)
        sch[tx_grad_grad_ext].split(tx_grad_grad_ext.op.axis[2], factor=ft_d)
        sch[tx_max_broadcasted].split(tx_max_broadcasted.op.axis[2], factor=ft_d)
        sch[tx_orig_out_ext].split(tx_orig_out_ext.op.axis[2], factor=ft_d)
        sch[tx_decrease_kernel_ext].split(tx_decrease_kernel_ext.op.axis[2], factor=ft_d)

        sch[tx_orig_in_ext].split(tx_orig_in_ext.op.axis[3], factor=ft_h)
        sch[tx_grad_grad_ext].split(tx_grad_grad_ext.op.axis[3], factor=ft_h)
        sch[tx_max_broadcasted].split(tx_max_broadcasted.op.axis[3], factor=ft_h)
        sch[tx_orig_out_ext].split(tx_orig_out_ext.op.axis[3], factor=ft_h)
        sch[tx_decrease_kernel_ext].split(tx_decrease_kernel_ext.op.axis[3], factor=ft_h)

        sch[tx_orig_in_ext].split(tx_orig_in_ext.op.axis[4], factor=ft_w)
        sch[tx_grad_grad_ext].split(tx_grad_grad_ext.op.axis[4], factor=ft_w)
        sch[tx_max_broadcasted].split(tx_max_broadcasted.op.axis[4], factor=ft_w)
        sch[tx_orig_out_ext].split(tx_orig_out_ext.op.axis[4], factor=ft_w)
        sch[tx_decrease_kernel_ext].split(tx_decrease_kernel_ext.op.axis[4], factor=ft_w)


    def _init_only_once():
        conditions = []
        conditions.append(fuse_i.equal(0))
        conditions.append(d_out.equal(0))
        conditions.append(h_out.equal(0))
        conditions.append(w_out.equal(0))
        sch[tx_all_zero].set_store_predicate(conditions)
        sch[tx_all_zero].mem_unique()
        sch[tx_decrease_kernel_ext].set_store_predicate(conditions)
        sch[tx_decrease_kernel_ext].mem_unique()


    def _split_select():
        split_select = {"split_select": 1}
        #if kernel gt than 8, pass will compile failure with this flag.
        if context["kd"] >= 7 or context["kh"] >= 7 or context["kw"] >= 7:
            return

        #if kernel is 5 and stride is 1, pass will compile failure with this flag.
        if context["kd"] >= 5 and context["kh"] >= 5 and context["kw"] >= 5 and\
           context["sd"] <= 2 and context["sh"] <= 2 and context["sw"] <= 2:
            return

        return split_select


    def _emit_insn():
        sch[tx_orig_in_c].emit_insn(tx_orig_in_c.op.axis[0], 'dma_copy', _split_select())
        sch[tx_orig_in_ft].emit_insn(tx_orig_in_ft.op.axis[0], 'vector_dup', _split_select())
        sch[tx_orig_in_bk].emit_insn(tx_orig_in_bk.op.axis[0], 'vector_dup', _split_select())
        sch[tx_orig_in_t].emit_insn(tx_orig_in_t.op.axis[0], 'vector_dup', _split_select())
        sch[tx_orig_in_b].emit_insn(tx_orig_in_b.op.axis[0], 'vector_dup', _split_select())
        sch[tx_orig_in_l].emit_insn(tx_orig_in_l.op.axis[0], 'vector_dup', _split_select())
        sch[tx_orig_in_r].emit_insn(tx_orig_in_r.op.axis[0], 'vector_dup', _split_select())
        sch[tx_orig_in].emit_insn(tx_orig_in.op.axis[0], "phony_insn")
        sch[tx_orig_out].emit_insn(tx_orig_out.op.axis[0], 'dma_copy')
        sch[tx_grad_grad_c].emit_insn(tx_grad_grad_c.op.axis[0], 'dma_copy', _split_select())
        sch[tx_grad_grad].emit_insn(tx_grad_grad.op.axis[0], "phony_insn")
        sch[tx_decrease_kernel].emit_insn(tx_decrease_kernel.op.axis[0], 'dma_copy')
        sch[tx_orig_in_ext].emit_insn(tx_orig_in_ext.op.axis[0], 'dma_copy')
        sch[tx_orig_out_ext].emit_insn(tx_orig_out_ext.op.axis[0], 'dma_copy')
        sch[tx_grad_grad_ext].emit_insn(tx_grad_grad_ext.op.axis[0], 'dma_copy')
        sch[tx_decrease_kernel_ext].emit_insn(tx_decrease_kernel_ext.op.axis[0], 'dma_copy')
        sch[tx_mask].emit_insn(tx_mask.op.axis[0], 'vector_auto')
        sch[tx_all_zero].emit_insn(tx_all_zero.op.axis[0], 'vector_dup')
        sch[tx_decrease_sparse_matrix].emit_insn(tx_decrease_sparse_matrix.op.axis[0], 'vector_select_bool')
        sch[tx_max_broadcasted].emit_insn(tx_max_broadcasted.op.axis[0], 'dma_copy')
        sch[tx_mask_no_dup].emit_insn(tx_mask_no_dup.op.axis[0], 'vector_auto')
        sch[tx_grad_by_mask].emit_insn(tx_grad_by_mask.op.axis[0], 'vector_select_bool')
        sch[res].emit_insn(d_in, 'dma_copy')

        for tensor in pool_tensors.values():
            if tensor.op.tag == "reduce_max":
                sch[tensor].emit_insn(tensor.op.axis[0], 'vector_auto')

        for tensor in pool_tensors.values():
            if tensor.op.tag == "reduce_grad_max":
                sch[tensor].emit_insn(tensor.op.axis[0], 'vector_auto')


    def _buff_align():
        align_num = _calc_align_num(context, d_factor, h_factor, w_factor)
        _buff_align_8(context, sch, align_num, tx_mask)
        _buff_align_8(context, sch, align_num, tx_decrease_sparse_matrix)
        _buff_align_8(context, sch, align_num, tx_grad_by_mask)
        _buff_align_8(context, sch, align_num, tx_mask_no_dup)

    d_factor, h_factor, w_factor = _calc_d_h_w_factor(context)
    d_factor = d_factor if d_factor > 0 else 1
    h_factor = h_factor if h_factor > 0 else 1
    w_factor = w_factor if w_factor > 0 else 1
    d_out, d_in = _split_d_axis(res, sch, d_factor)
    h_out, h_in = _split_h_axis(res, sch, h_factor)
    w_out, w_in = _split_w_axis(res, sch, w_factor)
    sch[res].reorder(res.op.axis[0], res.op.axis[2], d_out, h_out, w_out, d_in, h_in, w_in, res.op.axis[5])
    fuse = sch[res].fuse(res.op.axis[0], res.op.axis[2])
    fuse_o, fuse_i = sch[res].split(fuse, nparts=cce.get_soc_spec("CORE_NUM"))
    thread_block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(fuse_o, thread_block)
    for tensor in pool_tensors.values():
        if _need_to_compute_at(tensor):
            sch[tensor].compute_at(sch[res], w_out)

    if context["fast_path"] == "True":
        return _fast_path_schedule()

    tx_orig_in_c = pool_tensors["tx_orig_in_c"]
    tx_orig_in_ft = pool_tensors["tx_orig_in_ft"]
    tx_orig_in_bk = pool_tensors["tx_orig_in_bk"]
    tx_orig_in_t = pool_tensors["tx_orig_in_t"]
    tx_orig_in_b = pool_tensors["tx_orig_in_b"]
    tx_orig_in_l = pool_tensors["tx_orig_in_l"]
    tx_orig_in_r = pool_tensors["tx_orig_in_r"]
    tx_orig_in = pool_tensors["tx_orig_in"]
    tx_grad_grad_c = pool_tensors["tx_grad_grad_c"]
    tx_grad_grad = pool_tensors["tx_grad_grad"]
    tx_orig_out = pool_tensors["tx_orig_out"]
    tx_decrease_kernel = pool_tensors["tx_decrease_kernel"]
    tx_grad_grad_ext = pool_tensors["tx_grad_grad_ext"]
    tx_orig_in_ext = pool_tensors["tx_orig_in_ext"]
    tx_orig_out_ext = pool_tensors["tx_orig_out_ext"]
    tx_decrease_kernel_ext = pool_tensors["tx_decrease_kernel_ext"]
    tx_max_broadcasted = pool_tensors["tx_max_broadcasted"]
    tx_mask = pool_tensors["tx_mask"]
    tx_all_zero = pool_tensors["tx_all_zero"]
    tx_decrease_sparse_matrix = pool_tensors["tx_decrease_sparse_matrix"]
    tx_mask_no_dup = pool_tensors["tx_mask_no_dup"]
    tx_grad_by_mask = pool_tensors["tx_grad_by_mask"]

    _mem_reuse(sch,
               [tx_orig_in_ft,
                tx_orig_in_bk,
                tx_orig_in_t,
                tx_orig_in_b,
                tx_orig_in_l,
                tx_orig_in_r,
                tx_orig_in],
               tx_orig_in_c)

    _mem_reuse(sch, [tx_grad_grad], tx_grad_grad_c)

    _split_for_dma_copy()
    _init_only_once()
    _emit_insn()
    _buff_align()

    return True


def _need_to_compute_at(tensor):
    if tensor.op.name == "tx_decrease_kernel":
        return False
    return True


def _buff_align_8(context, sch, align_num, tensor):
    if context["align_axis"] == "axis_w":
        sch[tensor].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, align_num), (1, BLOCK_SIZE))
    elif context["align_axis"] == "axis_h":
        sch[tensor].buffer_align((1, 1), (1, 1), (1, 1), (1, align_num), (1, 1), (1, BLOCK_SIZE))
    elif context["align_axis"] == "axis_d":
        sch[tensor].buffer_align((1, 1), (1, 1), (1, align_num), (1, 1), (1, 1), (1, BLOCK_SIZE))


def _calc_process_per_window_ub_size(context):
    k_d, k_h, k_w = context["kd"], context["kh"], context["kw"]
    s_d, s_h, s_w = context["sd"], context["sh"], context["sw"]
    dd = k_d if k_d > s_d else s_d
    hh = k_h if k_h > s_h else s_h
    ww = k_w if k_w > s_w else s_w
    '''
    tx_orig_in     +  tx_orig_in_ext
    tx_orig_out    +  tx_orig_out_ext
    tx_grad_grad   +  tx_grad_grad_ext
    tx_decrease_kernel +  tx_decrease_kernel_ext
    tx_decrease_sparse_matrix +  tx_max_broadcasted
    tx_all_zero
    tx_mask_no_dup
    rd_x/rh_x/rw_x
    tx_grad_rdx/tx_grad_rhx/tx_grad_rwx
    tx_res
    '''
    factor = 16
    if k_d == 1 and k_h == 1 and k_w == 1:
        factor = s_d * s_d * s_w + 2
    return dd * hh * ww * C0_DIMENSION_DATA_SIZE * factor


def _calc_window_numbers_per_batch(context):
    ub_size = cce.get_soc_spec("UB_SIZE")
    ub_size = ub_size - int(ub_size * 0.1 * context["round"])
    return ub_size // _calc_process_per_window_ub_size(context)


def _calc_d_h_w_factor(context):
    #if kernel is gt 5, one kernel each time has better perf
    if context["kd"] >= 5 and context["kh"] >= 5 and context["kw"] >= 5:
        return 1, 1, 1
    df, hf, wf = _special_case(context)
    if df != 0 or hf != 0 or wf != 0:
        return df, hf, wf
    if context["align_axis"] == "axis_w":
        return _calc_w_h_d_factor_impl(context)
    elif context["align_axis"] == "axis_h":
        return _calc_h_w_d_factor_impl(context)
    elif context["align_axis"] == "axis_d":
        return _calc_d_w_h_factor_impl(context)


def _special_case(context):
    special_case = [
                    [13, 8, 11, 3, 3, 3, 4, 4, 4, 3, 2, 1],
                    [8, 8, 5, 3, 3, 3, 3, 3, 3, 3, 1, 1]
                   ]
    for item in special_case:
        if (item[0] == context["d"] and item[1] == context["h"] and item[2] == context["w"] and\
            item[3] == context["kd"] and item[4] == context["kh"] and item[5] == context["kw"] and\
            item[6] == context["sd"] and item[7] == context["sh"] and item[8] == context["sw"]):
            return item[9], item[10], item[11]
    return 0, 0, 0


def _calc_w_h_d_factor_impl(context):
    o_d, o_h, o_w = context["do"], context["ho"], context["wo"]
    w_factor = h_factor = d_factor = 1
    num = _calc_window_numbers_per_batch(context)

    if o_w >= num:
        w_factor = num
        num = 1
    else:
        w_factor = o_w
        num = num // w_factor

    if o_h >= num:
        h_factor = num
        num = 1
    else:
        h_factor = o_h
        num = num // h_factor

    if o_d >= num:
        d_factor = num
    else:
        d_factor = o_d

    return d_factor, h_factor, w_factor


def _calc_h_w_d_factor_impl(context):
    o_d, o_h, o_w = context["do"], context["ho"], context["wo"]
    w_factor = h_factor = d_factor = 1
    num = _calc_window_numbers_per_batch(context)

    if o_h >= num:
        h_factor = num
        num = 1
    else:
        h_factor = o_h
        num = num // h_factor

    if o_w >= num:
        w_factor = num
        num = 1
    else:
        w_factor = o_w
        num = num // w_factor

    if o_d >= num:
        d_factor = num
    else:
        d_factor = o_d

    return d_factor, h_factor, w_factor


def _calc_d_w_h_factor_impl(context):
    o_d, o_h, o_w = context["do"], context["ho"], context["wo"]
    w_factor = h_factor = d_factor = 1
    num = _calc_window_numbers_per_batch(context)

    if o_d >= num:
        d_factor = num
        num = 1
    else:
        d_factor = o_d
        num = num // d_factor

    if o_w >= num:
        w_factor = num
        num = 1
    else:
        w_factor = o_w
        num = num // w_factor

    if o_h >= num:
        h_factor = num
    else:
        h_factor = o_h

    return d_factor, h_factor, w_factor


#' N  C1o/C1i  D  H  W  C0    =>   N  C1o/C1i  Do/Di  H  W  C0
#'             ^                                 ^
#'             |                                 |
def _split_d_axis(res, sch, d_factor):
    return sch[res].split(res.op.axis[1], factor=d_factor)


# 'N  C1o/C1i  Do/Di  H  W  C0  =>  N  C1o/C1i  Do/Di  Ho/Hi  W  C0
# '                   ^                                  ^
# '                   |                                  |
def _split_h_axis(res, sch, h_factor):
    return sch[res].split(res.op.axis[3], factor=h_factor)


# 'N  C1o/C1i  Do/Di  Ho/Hi  W  C0  =>  N  C1o/C1i  Do/Di  Ho/Hi  Wo/Wi  C0
# '                          ^                                      ^
# '                          |                                      |
def _split_w_axis(res, sch, w_factor):
    return sch[res].split(res.op.axis[4], factor=w_factor)


def _set_scope(sch, tensors, scope):
    for tensor in tensors:
        sch[tensor].set_scope(scope)


def _is_placeholder(tensor):
    return isinstance(tensor.op, tvm.tensor.PlaceholderOp)


def _crawl_pool_tensor(res):
    tensors = {}
    queue = [res]
    visited = []
    while queue:
        head = queue.pop(0)
        for tensor in head.op.input_tensors:
            if tensor in visited or _is_placeholder(tensor) or \
                    tensor.op.tag == ASCEND_ANTI_QUANT_TAG:
                continue
            tensors[tensor.op.name] = tensor
            visited.append(tensor)
            queue.append(tensor)
    return tensors


def _get_build_round(res):
    #UB size can not be calculated accurately, so try MAX_BUILD_ROUND_FOR_RECALC_UB times at most
    for i in range(MAX_BUILD_ROUND_FOR_RECALC_UB, 0, -1):
        try:
            if "recalc_ub_round_"+str(i) in res.op.attrs:
                return i
        except Exception as e:
            continue
    return 0

def _mem_reuse(sch, tensors, target_tensor):
    for tensor in tensors:
        sch[target_tensor].reused_by(tensor)


def _calc_align_num(context, d_factor, h_factor, w_factor):
    if context["align_axis"] == "axis_w":
        if _aligin_up_8(w_factor * context["kw"]) % 16 == 0:
            return 8
        else:
            return 16
    elif context["align_axis"] == "axis_h":
        if (_aligin_up_8(h_factor * context["kh"]) * w_factor * context["kw"]) % 16 == 0:
            return 8
        else:
            return 16
    elif context["align_axis"] == "axis_d":
        if (_aligin_up_8(d_factor * context["kd"]) * h_factor * context["kh"] * w_factor * context["kw"]) % 16 == 0:
            return 8
        else:
            return 16


def _aligin_up_8(n):
    if n % 8 == 0:
        return n
    return n + (8 - n % 8)

