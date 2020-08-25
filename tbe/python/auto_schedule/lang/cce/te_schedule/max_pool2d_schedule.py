#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

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

from te import platform as cce
from te import tvm
from te.platform import get_soc_spec

# define the quantize tensor name
CAST_F16_NAME = "cast_f16_ub"
INPUT_NAME = "input_ub"
VMULS_REFORM_NAME = "reform_by_vmuls"
SQRT_NAME = "scale_sqrt_ub"
OFFSET_NAME = "offset_ub"
CAST_I8_NAME = "cast_i8_ub"
VADDS_REFORM_NAME = "reform_by_vadds"

SIZE_OF_FP16 = 2
C0 = 16
SIZE_OF_FP32 = 4
ASCEND_QUANT_TAG = "quant"
POOLING2D_TAG_PREFIX = "pooling2d_"
ASCEND_ANTI_QUANT_TAG = "anti_quant"


def _is_placeholder(tensor):
    return isinstance(tensor.op, tvm.tensor.PlaceholderOp)


def _crawl_dequant_tensor(res):
    tensors = {}
    queue = [res]
    visited = []
    while queue:
        head = queue.pop(0)
        for tensor in head.op.input_tensors:
            if tensor in visited or _is_placeholder(tensor):
                continue
            tensors[tensor.op.name] = tensor
            visited.append(tensor)
            queue.append(tensor)
    return tensors


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


def _crawl_quant_tensor(res):
    tensors = {}
    queue = [res]
    visited = []
    while queue:
        head = queue.pop(0)
        for tensor in head.op.input_tensors:
            if tensor in visited or _is_placeholder(tensor) or \
                    tensor.op.tag.startswith(POOLING2D_TAG_PREFIX):
                continue
            tensors[tensor.op.name] = tensor
            visited.append(tensor)
            queue.append(tensor)
    return tensors


def _set_scope(sch, tensors, scope):
    for tensor in tensors:
        sch[tensor].set_scope(scope)


def _mem_reuse(sch, tensors, target_tensor):
    for tensor in tensors:
        sch[target_tensor].reused_by(tensor)


def _set_db(sch, tensors):
    for tensor in tensors:
        sch[tensor].double_buffer()


def _set_preload(sch, tensors):
    for tensor in tensors:
        sch[tensor].preload()


def _compute_at(sch, tensors, target_tensor, axis):
    for tensor in tensors:
        sch[tensor].compute_at(sch[target_tensor], axis)


def _evaluate_bind_core(cherry_num, factor):
    core_n = get_soc_spec("CORE_NUM")
    pizza_n = (cherry_num + factor - 1) // factor
    round_n = (pizza_n + core_n - 1) // core_n
    return round_n * factor


def _find_core_factor(cherry_num, core_n):
    core_gap = cherry_num * core_n
    factor = 1

    for i in range(cherry_num, 0, -1):
        i_gap = _evaluate_bind_core(cherry_num, i)
        if i_gap < core_gap:
            core_gap = i_gap
            factor = i

    return factor


def _tiling(context):
    fused_dequant = context["fused_dequant"]
    fused_quant = context["fused_quant"]
    enabler_c1_bind_core = context["enabler_c1_bind_core"]
    # pool shape
    p_c1, p_w = context["c1"], context["w"]
    o_h, o_w = context["ho"], context["wo"]
    k_h, k_w = context["kh"], context["kw"]
    s_h, s_w = context["sh"], context["sw"]
    ub_size0 = get_soc_spec("UB_SIZE")
    core_n = get_soc_spec("CORE_NUM")
    factors = {}

    def _check_c1_factor(c1_factor):
        if fused_dequant and c1_factor % 2 != 0:
            return False
        if fused_quant and c1_factor != p_c1 and c1_factor % 2 != 0:
            return False
        return True

    def _get_min_c1_factor():
        if fused_dequant:
            return 2
        if fused_quant and p_c1 != 1:
            return 2
        return 1

    def _get_need_size(c1_factor, ho_factor):
        h_factor = (ho_factor - 1) * s_h + k_h
        w_p = (o_w - 1) * s_h + k_h

        ub_1 = c1_factor * h_factor * w_p * 16 * 2
        rw_1 = c1_factor * h_factor * o_w * 16 * 2
        rh_1 = c1_factor * ho_factor * o_w * 16 * 2
        need_size = ub_1 + rw_1 * 2 + rh_1 * 2

        if fused_dequant:
            dequant_ub_1 = (c1_factor // 2) * h_factor * w_p * 32
            need_size = need_size + ub_1 + dequant_ub_1
        if fused_quant:
            need_size = need_size + rh_1
        return need_size

    def _get_need_size_cut_w(c1_factor, ho_factor, wo_factor):
        h_factor = (ho_factor - 1) * s_h + k_h
        w_factor = (wo_factor - 1) * s_w + k_w

        ub_1 = c1_factor * h_factor * w_factor * 16 * 2
        rw_1 = c1_factor * h_factor * wo_factor * 16 * 2
        rh_1 = c1_factor * ho_factor * wo_factor * 16 * 2
        need_size = ub_1 + rw_1 * 2 + rh_1 * 2

        if fused_dequant:
            dequant_ub_1 = (c1_factor // 2) * h_factor * w_factor * 32
            need_size = need_size + ub_1 + dequant_ub_1
        if fused_quant:
            need_size = need_size + rh_1
        return need_size

    def _try_tiling(ub_size):
        core_gap = p_c1 * core_n
        find = False
        for c1_factor in range(p_c1, 0, -1):
            if not _check_c1_factor(c1_factor) or \
                    _get_need_size_cut_w(c1_factor, o_h, o_w) > ub_size:
                continue
            find = True
            if not enabler_c1_bind_core:
                factors["c1_factor"] = c1_factor
                factors["ho_factor"] = o_h
                factors["wo_factor"] = o_w
                break
            i_gap = _evaluate_bind_core(p_c1, c1_factor)
            if i_gap < core_gap:
                core_gap = i_gap
                factors["c1_factor"] = c1_factor
                factors["ho_factor"] = o_h
                factors["wo_factor"] = o_w

        if find:
            return True

        c1_factor = _get_min_c1_factor()
        for ho_factor in range(o_h, 0, -1):
            if _get_need_size_cut_w(c1_factor, ho_factor, o_w) > ub_size:
                continue
            factors["c1_factor"] = c1_factor
            factors["ho_factor"] = ho_factor
            factors["wo_factor"] = o_w
            return True

        c1_factor = _get_min_c1_factor()
        ho_factor = 1
        for wo_factor in range(o_w, 0, -1):
            if _get_need_size_cut_w(c1_factor, ho_factor, wo_factor) > ub_size:
                continue
            factors["c1_factor"] = c1_factor
            factors["ho_factor"] = ho_factor
            factors["wo_factor"] = wo_factor
            return True

        return False

    if _try_tiling(ub_size0 // 2):
        return True, factors["c1_factor"], factors["ho_factor"], factors[
            "wo_factor"]
    if _try_tiling(ub_size0):
        return False, factors["c1_factor"], factors["ho_factor"], factors[
            "wo_factor"]
    raise RuntimeError("Cannot find tiling, kw and kh is too big!")


def _set_round_emit_insn(round_mode):
    """
    Obtains the conv instruction by the round mode attr

    Parameters
    ----------
    round_mode: the attr of round mode

    Returns
    -------
    instruction
    """
    if get_soc_spec("SOC_VERSION") == "Ascend310":
        # mini
        emit_insn_str = "vector_conv"
    else:
        if round_mode == "Round":
            emit_insn_str = "vector_conv_round"
        elif round_mode == "Ceil":
            emit_insn_str = "vector_conv_ceil"
        elif round_mode == "Floor":
            emit_insn_str = "vector_conv_floor"
        elif round_mode == "Trunc":
            emit_insn_str = "vector_conv_trunc"
        else:
            emit_insn_str = "vector_conv"
    return emit_insn_str


def _schedule_quant(res, sch, tensors, context):
    # scope
    _set_scope(sch, tensors.values(), cce.scope_ubuf)

    # compute optimize
    _compute_at(sch, tensors.values(), res, context["Woo"])
    for tensor in tensors.values():
        if tensor.op.name in (VADDS_REFORM_NAME, VMULS_REFORM_NAME):
            sch[tensor].split(tensor.op.axis[4], factor=16)

    def _get_tensor(name):
        return tensors.get(name, None)

    def _emit_insn(_tensor, insn):
        if _tensor is not None:
            sch[_tensor].emit_insn(_tensor.op.axis[0], insn)

    # emit insn
    _emit_insn(_get_tensor(CAST_F16_NAME), "vector_conv")
    _emit_insn(_get_tensor(OFFSET_NAME), "vector_adds")
    _emit_insn(_get_tensor(SQRT_NAME), "vector_muls")
    _emit_insn(_get_tensor(VMULS_REFORM_NAME), "vector_muls")
    _emit_insn(_get_tensor(VADDS_REFORM_NAME), "vector_adds")
    _emit_insn(_get_tensor(CAST_I8_NAME),
               _set_round_emit_insn(res.op.attrs["round_mode"]))
    _emit_insn(_get_tensor(INPUT_NAME), "dma_copy")


def _schedule_pool(res, sch, tensors, context):
    # tensor
    tx_ub_t, tx_ub_b = tensors["tx_ub_t"], tensors["tx_ub_b"]
    tx_ub_l, tx_ub_r = tensors["tx_ub_l"], tensors["tx_ub_r"]
    tx_ub_c, tx_ub = tensors["tx_ub_c"], tensors["tx_ub"]

    # scope
    _set_scope(sch, tensors.values(), cce.scope_ubuf)

    # db
    enabler_db = context["enabler_db"]
    fused_dequant = context["fused_dequant"]
    if enabler_db and not fused_dequant:
        sch[tx_ub_c].preload()
        sch[tx_ub_c].double_buffer()
        sch[tx_ub_t].double_buffer()
        sch[tx_ub_b].double_buffer()
        sch[tx_ub_l].double_buffer()
        sch[tx_ub_r].double_buffer()
        sch[tx_ub].double_buffer()

    # memory reuse
    _mem_reuse(sch, [tx_ub_t, tx_ub_b, tx_ub_l, tx_ub_r, tx_ub], tx_ub_c)
    sch[tx_ub].reused_by(reuse_data=True)

    # compute optimize
    if context["fused_quant"]:
        _compute_at(sch, tensors.values(), res, res.op.axis[0])
    else:
        _compute_at(sch, tensors.values(), res, context["Woo"])

    # emit insn
    split_select = {"split_select": 1}
    sch[tx_ub_c].emit_insn(tx_ub_c.op.axis[2], "dma_copy", split_select)
    sch[tx_ub_t].emit_insn(tx_ub_t.op.axis[2], "vector_dup", split_select)
    sch[tx_ub_b].emit_insn(tx_ub_b.op.axis[2], "vector_dup", split_select)
    sch[tx_ub_l].emit_insn(tx_ub_l.op.axis[2], "vector_dup", split_select)
    sch[tx_ub_r].emit_insn(tx_ub_r.op.axis[2], "vector_dup", split_select)
    sch[tx_ub].emit_insn(tx_ub.op.axis[2], "phony_insn")

    for tensor in tensors.values():
        if tensor.op.tag == "reduce_max":
            sch[tensor].emit_insn(tensor.op.axis[2], "vector_auto")


def _schedule_dequant(res, sch, tensors, context):
    # tensor
    input_ub = tensors["input_ub"]

    # scope
    _set_scope(sch, tensors.values(), cce.scope_ubuf)

    # db
    if context["enabler_db"]:
        sch[input_ub].preload()
        sch[input_ub].double_buffer()

    # compute optimize
    _compute_at(sch, tensors.values(), res, res.op.axis[0])
    reform_by_vmuls = tensors["reform_by_vmuls"]
    sch[reform_by_vmuls].split(reform_by_vmuls.op.axis[1], factor=2)

    # emit insn
    for tensor in tensors.values():
        if tensor.op.name == "input_ub":
            sch[tensor].emit_insn(tensor.op.axis[0], 'dma_copy')
        else:
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_auto')


def schedule(res, sch_list):
    """
    schedule for generic vector template
    :param res:
    :param sch_list:
    :return:
    """
    sch = sch_list[0]
    context = {}

    # quant params
    fused_quant = res.op.tag == ASCEND_QUANT_TAG
    context["fused_quant"] = fused_quant
    if fused_quant:
        quant_res = res
        quant_tensors = _crawl_quant_tensor(quant_res)
        pool_res = quant_tensors["input_ub"].op.input_tensors[0]
    else:
        pool_res = res

    # dequant params
    pool_tensors = _crawl_pool_tensor(pool_res)
    pool_x = pool_tensors["tx_ub_c"].op.input_tensors[0]
    fused_dequant = pool_x.op.tag == ASCEND_ANTI_QUANT_TAG
    context["fused_dequant"] = fused_dequant
    if fused_dequant:
        dequant_res = pool_x
        dequant_tensors = _crawl_dequant_tensor(dequant_res)

    # pool params
    pool_params = pool_res.op.attrs["pooling_params"]
    p_n, p_c1 = pool_params["batch_size"].value, pool_params["c1_value"].value
    p_h, p_w = pool_params["in_size_h"].value, pool_params["in_size_w"].value
    o_h, o_w = pool_params["out_size_h"].value, pool_params["out_size_w"].value
    k_h, k_w = pool_params["window_h"].value, pool_params["window_w"].value
    s_h, s_w = pool_params["stride_h"].value, pool_params["stride_w"].value
    context["n"], context["c1"] = p_n, p_c1
    context["h"], context["w"] = p_h, p_w
    context["ho"], context["wo"] = o_h, o_w
    context["kh"], context["kw"] = k_h, k_w
    context["sh"], context["sw"] = s_h, s_w

    # bind core select: n, c1
    core_n = get_soc_spec("CORE_NUM")
    c1_core = (p_c1 + 1) // 2 if fused_quant else p_c1
    enabler_c1_bind_core = p_n < core_n and p_n < c1_core
    context["enabler_c1_bind_core"] = enabler_c1_bind_core

    # tiling
    enabler_db, c1_factor, ho_factor, wo_factor = _tiling(context)
    if fused_quant:
        c1_factor = (c1_factor + 1) // 2

    context["enabler_db"] = enabler_db

    res1o, res1i = sch[res].split(res.op.axis[1], factor=c1_factor)
    hoo, hoi = sch[res].split(res.op.axis[2], factor=ho_factor)
    woo, woi = sch[res].split(res.op.axis[3], factor=wo_factor)

    context["Hoo"] = hoo
    context["Hoi"] = hoi
    context["Woo"] = woo
    context["Woi"] = woi

    # bind core
    if enabler_c1_bind_core:
        c1o = (p_c1 + c1_factor - 1) // c1_factor
        c1o_factor = _find_core_factor(c1o, core_n)
        res1oo, res1oi = sch[res].split(res1o, factor=c1o_factor)
        block_axis = res1oo
        sch[res].reorder(res1oo, res1oi, res.op.axis[0], hoo, woo, res1i,
                         hoi, woi, res.op.axis[4])
    else:
        n_factor = _find_core_factor(p_n, core_n)
        res0o, re0i = sch[res].split(res.op.axis[0], factor=n_factor)
        block_axis = res0o
        sch[res].reorder(res0o, re0i, res1o, hoo, woo, res1i, hoi,
                         woi, res.op.axis[4])
    thread_block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(block_axis, thread_block)

    # schedule quant
    if fused_quant:
        _schedule_quant(quant_res, sch, quant_tensors, context)
        _set_scope(sch, [pool_res], cce.scope_ubuf)
        sch[pool_res].compute_at(sch[res], woo)
        tx_rh_name = "tx_rh" + str(k_h - 1)
        sch[pool_res].reused_by(pool_tensors[tx_rh_name])
        sch[pool_res].emit_insn(pool_res.op.axis[1], "phony_insn")

    # schedule dequant
    if fused_dequant:
        _schedule_dequant(dequant_res, sch, dequant_tensors, context)
        _set_scope(sch, [dequant_res], cce.scope_ubuf)
        sch[dequant_res].compute_at(sch[res], woo)
        sch[dequant_res].reused_by(dequant_tensors["reform_by_vmuls"])
        sch[dequant_res].emit_insn(dequant_res.op.axis[1], "phony_insn")

    # schedule pool
    _schedule_pool(pool_res, sch, pool_tensors, context)

    sch[res].emit_insn(res1i, "dma_copy")

    return True
