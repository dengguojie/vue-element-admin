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
from te.platform import cce_emitinsn_params as cce_params
from te.platform import get_soc_spec
from te.platform.cce_conf import CceProductParams as pver

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
STRIDED_WRITE_TAG = "strided_write"

# define the type of L1 fusion
DEFAULT_VALUE = -1
L1_DEPTH_FUSION = 0
L1_BREADTH_FUSION = 1

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


def _crawl_input_tensor(res):
    queue = [res]
    visited = []
    while queue:
        head = queue.pop(0)
        for tensor in head.op.input_tensors:
            if len(tensor.op.input_tensors) == 1 and \
                    _is_placeholder(tensor.op.input_tensors[0]):
                return tensor
            if tensor in visited:
                continue
            visited.append(tensor)
            queue.append(tensor)
    return None


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


def _evaluate_bind_core(core_n, cherry_num, factor):
    pizza_n = (cherry_num + factor - 1) // factor
    round_n = (pizza_n + core_n - 1) // core_n
    return round_n * factor


def _find_core_factor(cherry_num, core_n):
    core_gap = cherry_num * core_n
    factor = 1

    for i in range(cherry_num, 0, -1):
        i_gap = _evaluate_bind_core(core_n, cherry_num, i)
        if i_gap < core_gap:
            core_gap = i_gap
            factor = i

    return factor


def _tiling(context, core_n):
    fused_dequant = context["fused_dequant"]
    fused_quant = context["fused_quant"]
    bind_core_axis = context["bind_core_axis"]
    # pool shape
    p_c1, p_w = context["c1"], context["w"]
    o_h, o_w = context["ho"], context["wo"]
    k_h, k_w = context["kh"], context["kw"]
    s_h, s_w = context["sh"], context["sw"]
    ub_size0 = get_soc_spec("UB_SIZE")
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

    def _get_need_size(c1_factor, ho_factor, wo_factor):
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

    def _try_c1_tiling(ub_size):
        c1_core_gap = p_c1 * core_n
        find = False
        for c1_factor in range(p_c1, 0, -1):
            if not _check_c1_factor(c1_factor) or \
                    _get_need_size(c1_factor, o_h, o_w) > ub_size:
                continue
            find = True
            if bind_core_axis != "c1_axis" or core_n == 1:
                factors["c1_factor"] = c1_factor
                factors["ho_factor"] = o_h
                factors["wo_factor"] = o_w
                break
            i_gap = _evaluate_bind_core(core_n, p_c1, c1_factor)
            if i_gap < c1_core_gap:
                c1_core_gap = i_gap
                factors["c1_factor"] = c1_factor
                factors["ho_factor"] = o_h
                factors["wo_factor"] = o_w
        return find

    def _try_ho_tiling(ub_size):
        c1_factor = _get_min_c1_factor()
        ho_core_gap = o_h * core_n
        find = False
        for ho_factor in range(o_h, 0, -1):
            if _get_need_size(c1_factor, ho_factor, o_w) > ub_size:
                continue
            find = True
            if bind_core_axis != "ho_axis" or core_n == 1:
                factors["c1_factor"] = c1_factor
                factors["ho_factor"] = ho_factor
                factors["wo_factor"] = o_w
                break
            i_gap = _evaluate_bind_core(core_n, o_h, ho_factor)
            if i_gap < ho_core_gap:
                ho_core_gap = i_gap
                factors["c1_factor"] = c1_factor
                factors["ho_factor"] = ho_factor
                factors["wo_factor"] = o_w
        return find

    def _try_wo_tiling(ub_size):
        c1_factor = _get_min_c1_factor()
        ho_factor = 1
        find = False
        for wo_factor in range(o_w, 0, -1):
            if _get_need_size(c1_factor, ho_factor, wo_factor) > ub_size:
                continue
            factors["c1_factor"] = c1_factor
            factors["ho_factor"] = ho_factor
            factors["wo_factor"] = wo_factor
            find = True
            break
        return find

    def _try_tiling(ub_size):
        find_c1 = _try_c1_tiling(ub_size)
        if find_c1 and bind_core_axis != "ho_axis":
            return True

        find_ho = _try_ho_tiling(ub_size)
        if find_ho:
            return True

        # cut w does not support db now, for getting the same
        # performance compares to the old template
        if ub_size == ub_size0 // 2:
            return False

        find_wo = _try_wo_tiling(ub_size)
        if find_wo:
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
    if pver().is_mini_version():
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


def _schedule_quant(res, sch, tensors, context, round_mode):
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
               _set_round_emit_insn(round_mode))
    _emit_insn(_get_tensor(INPUT_NAME), "dma_copy")


def _schedule_pool(res, sch, tensors, context):
    # tensor
    tx_ub_t, tx_ub_b = tensors["tx_ub_t"], tensors["tx_ub_b"]
    tx_ub_l, tx_ub_r = tensors["tx_ub_l"], tensors["tx_ub_r"]
    tx_ub_c, tx_ub = tensors["tx_ub_c"], tensors["tx_ub"]
    tx_avg = tensors["tx_avg"]
    # scope
    _set_scope(sch, tensors.values(), cce.scope_ubuf)

    # db
    enabler_db = context["enabler_db"]
    fused_dequant = context["fused_dequant"]
    is_l1fusion = context["is_l1fusion"]

    # dequant/l1_fusion can't support double buffer ddr->l1 stage
    if fused_dequant or is_l1fusion:
        enabler_db = False

    if enabler_db:
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
        if tensor.op.tag == "reduce_sum":
            sch[tensor].emit_insn(tensor.op.axis[2], "vector_auto")

    sch[tx_avg].emit_insn(tx_avg.op.axis[2], "vector_muls", split_select)

def _schedule_dequant(res, sch, tensors, context):
    # tensor
    input_ub = tensors["input_ub"]

    sch[input_ub].buffer_align((1, 1),
                               (1, 1),
                               (1, 1),
                               (1, 1),
                               (1, 32))
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


def _get_max_axis(n_axis, c1_axis, ho_axis):
    max_axis = "n_axis"
    if n_axis >= c1_axis and n_axis >= ho_axis:
        max_axis = "n_axis"
    elif c1_axis > n_axis and c1_axis >= ho_axis:
        max_axis = "c1_axis"
    elif ho_axis > n_axis and ho_axis > c1_axis:
        max_axis = "ho_axis"

    return max_axis


def _choose_bind_core_axis(core_n, n_axis, c1_axis, ho_axis):
    if n_axis >= core_n or core_n == 1:
        bind_core_axis = "n_axis"
    elif c1_axis >= core_n:
        bind_core_axis = "c1_axis"
    elif ho_axis >= core_n:
        bind_core_axis = "ho_axis"
    else:
        bind_core_axis = _get_max_axis(n_axis, c1_axis, ho_axis)
    return bind_core_axis


def _set_pragma_for_cache_read_mode(stride_less_than_kernel, is_split_hw,
                                    stage, first_axis):
    """
    set pragma on the first axis for cache read mode

    Parameters
    ----------
    stride_less_than_kernel: bool if stride less than kernel is True

    is_split_hw: bool if split h or w is True

    stage: a Stage represents schedule for one operation

    first_axis: axis to set flag

    Returns
    -------
    """
    cache_read_mode = 0 if (stride_less_than_kernel and is_split_hw) else 1
    stage.pragma(first_axis, "json_info_cache_read_mode", cache_read_mode)


def schedule(res, sch_list):
    """
    schedule for generic vector template
    :param res:
    :param sch_list:
    :return:
    """
    sch = sch_list[0]
    context = {}

    fused_select_write = res.op.name.find("write_select") >= 0
    # fused strided_write
    fused_strided_write = res.op.tag == STRIDED_WRITE_TAG
    # place holder
    context["fused_select_write"] = fused_select_write
    context["fused_strided_write"] = fused_strided_write

    def _preprocess_fusion():
        res_select_or_strided_write = None
        quant_res = None
        quant_tensors = None
        hwc0 = None

        if fused_select_write or fused_strided_write:
            before_res = res.op.input_tensors[0]
            fused_ascend_quant = before_res.op.tag == ASCEND_QUANT_TAG
            context["fused_quant"] = fused_ascend_quant
        else:
            fused_ascend_quant = res.op.tag == ASCEND_QUANT_TAG
            context["fused_quant"] = fused_ascend_quant

        if (fused_select_write and fused_ascend_quant) or \
                (fused_strided_write and fused_ascend_quant):
            res_select_or_strided_write = res
            quant_res = res_select_or_strided_write.op.input_tensors[0]

            quant_tensors = _crawl_quant_tensor(res)
            pool_res = quant_tensors["input_ub"].op.input_tensors[0]
        elif fused_select_write or fused_strided_write:
            res_select_or_strided_write = res
            pool_res = res_select_or_strided_write.op.input_tensors[0]
        elif fused_ascend_quant:
            quant_res = res
            quant_tensors = _crawl_quant_tensor(res)
            pool_res = quant_tensors["input_ub"].op.input_tensors[0]
        else:
            pool_res = res

        res_list = (res_select_or_strided_write, fused_ascend_quant,
                    quant_res, quant_tensors, pool_res, hwc0)
        return res_list

    res_select_or_strided_write, fused_quant, \
        quant_res, quant_tensors, pool_res, hwc0 = \
            _preprocess_fusion()

    def _get_l1_fusion_params(pooling2d_res):
        fusion_params_map = pooling2d_res.op.attrs['fusion_params']
        fusion_params = {}
        if fusion_params_map:
            for key, value in fusion_params_map.items():
                if hasattr(value, "value"):
                    fusion_params[key] = value.value
                else:
                    fusion_params[key] = value

        # fused_op, l1 fusion info should get from fused_op output
        # so there is revise out l1 flag
        is_fused_compute = fusion_params.get("is_fused_compute")
        if is_fused_compute:
            revise_out_l1_flag = res.op.attrs["addr_type"].value == 1 \
                if "addr_type" in res.op.attrs else False
            fusion_params["out_l1_flag"] = revise_out_l1_flag
        return fusion_params

    fusion_params = _get_l1_fusion_params(pool_res)
    cce_params.cceEmitParamsIns.insert_params(fusion_params)

    l1_fusion_type = fusion_params.get("l1_fusion_type", DEFAULT_VALUE)
    in_select_read_flag = fusion_params.get("in_select_read_flag")
    is_l1fusion = l1_fusion_type in (L1_DEPTH_FUSION, L1_BREADTH_FUSION)
    context["is_l1fusion"] = is_l1fusion
    in_l1_flag = fusion_params.get("in_l1_flag", False)
    out_l1_flag = fusion_params.get("out_l1_flag", False)
    device_core_num = 1 if is_l1fusion else \
        get_soc_spec("CORE_NUM")

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

    # bind core select: n, c1, ho
    c1_core = (p_c1 + 1) // 2 if fused_quant else p_c1
    bind_core_axis = _choose_bind_core_axis(device_core_num, p_n, c1_core, o_h)
    context["bind_core_axis"] = bind_core_axis

    tensor_in_ub = _crawl_input_tensor(pool_res)

    def _l1_fusion_set_scope():
        if in_l1_flag:
            tensor_in = tensor_in_ub.op.input_tensors[0]
            if l1_fusion_type == L1_BREADTH_FUSION:
                sch[tensor_in].set_scope(cce.scope_cbuf)
            else:
                sch[tensor_in].set_scope(cce.scope_cbuf_fusion)

        if is_l1fusion and out_l1_flag:
            sch[res].set_scope(cce.scope_cbuf_fusion)

    _l1_fusion_set_scope()

    round_mode_temp = None
    # select_write and stirded_write compute inline
    is_select_or_stride_write_quant = (fused_select_write and fused_quant) or \
                                      (fused_strided_write and fused_quant)
    is_select_or_stride_write = fused_select_write or fused_strided_write
    if is_select_or_stride_write_quant:
        # get quant round_mode
        round_mode_temp = quant_res.op.attrs["round_mode"]
        sch[quant_res].compute_inline()
        quant_res = res_select_or_strided_write
    elif is_select_or_stride_write:
        sch[pool_res].compute_inline()
        pool_res = res_select_or_strided_write
    # tiling
    enabler_db, c1_factor, ho_factor, wo_factor = \
        _tiling(context, device_core_num)
    if fused_quant:
        c1_factor = (c1_factor + 1) // 2

    context["enabler_db"] = enabler_db

    res1o, res1i = sch[res].split(res.op.axis[1], factor=c1_factor)
    res2o, res2i = sch[res].split(res.op.axis[2], factor=ho_factor)
    res3o, res3i = sch[res].split(res.op.axis[3], factor=wo_factor)

    context["Hoo"] = res2o
    context["Hoi"] = res2i
    context["Woo"] = res3o
    context["Woi"] = res3i

    res_c1_outer_value = (p_c1 + c1_factor - 1) // c1_factor
    is_no_bind = is_l1fusion or \
        (p_n == 1 and res_c1_outer_value == 1 and o_h == 1)

    # bind core
    def _bind_core():
        if bind_core_axis == "c1_axis":
            c1o = (p_c1 + c1_factor - 1) // c1_factor
            c1o_factor = _find_core_factor(c1o, device_core_num)
            res1oo, res1oi = sch[res].split(res1o, factor=c1o_factor)
            block_axis = res1oo
            block_tag = "c1_block_tag"
            sch[res].reorder(res1oo, res1oi, res.op.axis[0], res2o, res3o,
                             res1i, res2i, res3i, res.op.axis[4])
            axis_first = res1oi
        elif bind_core_axis == "ho_axis":
            hoo = (o_h + ho_factor - 1) // ho_factor
            hoo_factor = _find_core_factor(hoo, device_core_num)
            res2oo, res2oi = sch[res].split(res2o, factor=hoo_factor)
            block_axis = res2oo
            block_tag = "ho_block_tag"
            sch[res].reorder(res2oo, res2oi, res.op.axis[0], res1o, res3o,
                             res1i, res2i, res3i, res.op.axis[4])
            axis_first = res2oi
        else:
            n_factor = 1 if is_no_bind else \
                _find_core_factor(p_n, device_core_num)
            block_tag = None if is_no_bind else "batch_block_tag"
            res0o, re0i = sch[res].split(res.op.axis[0], factor=n_factor)
            block_axis = res0o
            sch[res].reorder(res0o, re0i, res1o, res2o, res3o, res1i, res2i,
                             res3i, res.op.axis[4])
            axis_first = re0i
        if block_tag is not None:
            thread_block = tvm.thread_axis("blockIdx.x")
            sch[res].bind(block_axis, thread_block)
        return axis_first

    first_axis = _bind_core()

    # schedule quant or dequant
    def _schedule_quant_or_dequant():
        # schedule quant
        if fused_quant:
            # To get "round_mode" in fusion-pooling+quant+others
            if fused_strided_write or fused_select_write:
                round_mode = round_mode_temp
            else:
                round_mode = res.op.attrs["round_mode"]
            _schedule_quant(quant_res, sch, quant_tensors, context, round_mode)
            _set_scope(sch, [pool_res], cce.scope_ubuf)
            sch[pool_res].compute_at(sch[res], res3o)
            tx_rh_name = "tx_rh" + str(k_h - 1)
            sch[pool_res].reused_by(pool_tensors[tx_rh_name])
            sch[pool_res].emit_insn(pool_res.op.axis[1], "phony_insn")

        # schedule dequant
        if fused_dequant:
            _schedule_dequant(dequant_res, sch, dequant_tensors, context)
            _set_scope(sch, [dequant_res], cce.scope_ubuf)
            sch[dequant_res].compute_at(sch[res], res3o)
            sch[dequant_tensors["reform_by_vmuls"]].reused_by(dequant_res)
            sch[dequant_res].emit_insn(dequant_res.op.axis[1], "phony_insn")

    _schedule_quant_or_dequant()

    # use for ub fusion stride write
    def _stirde_write_for_ub_fusion():
        if fused_strided_write:
            swrite_stride = res.op.attrs["stride"].value
            _, _, swrite_h, swrite_w, swrite_c0 = list(
                i.value for i in res.shape)
            sch[res].bind_buffer(res.op.axis[0],
                swrite_stride * swrite_h * swrite_w * swrite_c0, 0)

    _stirde_write_for_ub_fusion()

    # use for L1 fusion select write
    def _select_write_for_l1_fusion():
        if fused_select_write:
            if fused_quant:
                hwc0 = res.op.attrs["HWC0"].value
                sch[res].bind_buffer(res.op.axis[1], hwc0, 0)
            else:
                hwc0 = pool_res.op.attrs["HWC0"].value
                sch[pool_res].bind_buffer(pool_res.op.axis[1], hwc0, 0)

    _select_write_for_l1_fusion()

    # L1 fusion
    if (l1_fusion_type == L1_BREADTH_FUSION) or in_select_read_flag:
        sch[tensor_in_ub].emit_insn(tensor_in_ub.op.axis[0], 'dma_copy')

    # schedule pool
    _schedule_pool(pool_res, sch, pool_tensors, context)

    sch[res].emit_insn(res1i, "dma_copy")

    stride_less_than_kernel = bool(s_h < k_h or s_w < k_w)
    is_split_hw = bool(o_h > ho_factor or o_w > wo_factor)
    _set_pragma_for_cache_read_mode(stride_less_than_kernel, is_split_hw,
                                    sch[res], first_axis)
    return True
