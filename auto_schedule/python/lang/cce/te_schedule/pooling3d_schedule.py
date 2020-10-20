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
pooling3d_schedule
"""
from te import platform as cce
from te import tvm
from te import platform as cceconf
from .pooling3d_max_grad_grad_schedule import pooling3d_max_grad_grad_schedule

ASCEND_QUANT_TAG = "quant"
POOLING3D_TAG_PREFIX = "pooling3d_"
ASCEND_ANTI_QUANT_TAG = "anti_quant"
C0_DIMENSION_DATA_SIZE_MAP = {"float16": 32, "float32": 64, "double": 128}


def pooling3d_schedule(res, sch_list):
    """
    :params:
    :res: result tensor
    :sch_list: schedule list
    :return: True
    """
    if res.op.tag == POOLING3D_TAG_PREFIX + "max":
        return _pooling3d_max_schedule(res, sch_list)

    if res.op.tag == POOLING3D_TAG_PREFIX + "max_grad_grad":
        return pooling3d_max_grad_grad_schedule(res, sch_list)

    raise RuntimeError("Not suport tag in pooling3d_schedule.")


# 'pylint: disable=too-many-locals,invalid-name,unused-argument
def _pooling3d_max_schedule(res, sch_list):

    def _split_select():
        split_select = {"split_select": 1}
        # if kernel is 7 and stride is 2, pass will compile failure with this flag.
        if context["kd"] >= 7 and context["kh"] >= 7 and context["kw"] >= 7 and\
           context["sd"] <= 2 and context["sh"] <= 2 and context["sw"] <= 2:
            return
        return split_select

    sch = sch_list[0]
    pool_tensors = _crawl_pool_tensor(res)

    # pool params
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
    context["n"], context["c1"] = p_n, p_c1
    context["d"], context["h"], context["w"] = p_d, p_h, p_w
    context["do"], context["ho"], context["wo"] = o_d, o_h, o_w
    context["kd"], context["kh"], context["kw"] = k_d, k_h, k_w
    context["sd"], context["sh"], context["sw"] = s_d, s_h, s_w
    context["dtype"] = pool_params["dtype"].value

    d_factor, h_factor, w_factor = _calc_d_h_w_factor(context)
    d_out, d_in = _split_d_axis(res, sch, d_factor)
    h_out, h_in = _split_h_axis(res, sch, h_factor)
    w_out, w_in = _split_w_axis(res, sch, w_factor)
    sch[res].reorder(res.op.axis[0], res.op.axis[2],
                     d_out, h_out, w_out, d_in, h_in, w_in, res.op.axis[5])
    fuse = sch[res].fuse(res.op.axis[0], res.op.axis[2])
    core_num = cceconf.get_soc_spec("CORE_NUM")
    fuse_o, _ = sch[res].split(fuse, nparts=core_num)
    thread_block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(fuse_o, thread_block)
    _set_scope(sch, pool_tensors.values(), cce.scope_ubuf)

    for tensor in pool_tensors.values():
        sch[tensor].compute_at(sch[res], w_out)

    tx_ub_c = pool_tensors["tx_ub_c"]
    tx_ub_ft = pool_tensors["tx_ub_ft"]
    tx_ub_bk = pool_tensors["tx_ub_bk"]
    tx_ub_t = pool_tensors["tx_ub_t"]
    tx_ub_b = pool_tensors["tx_ub_b"]
    tx_ub_l = pool_tensors["tx_ub_l"]
    tx_ub_r = pool_tensors["tx_ub_r"]
    tx_ub = pool_tensors["tx_ub"]

    _mem_reuse(sch, [tx_ub_ft, tx_ub_bk, tx_ub_t, tx_ub_b, tx_ub_l, tx_ub_r, tx_ub], tx_ub_c)

    sch[tx_ub_c].emit_insn(tx_ub_c.op.axis[0], 'dma_copy', _split_select())
    sch[tx_ub_ft].emit_insn(tx_ub_ft.op.axis[0], 'vector_dup', _split_select())
    sch[tx_ub_bk].emit_insn(tx_ub_bk.op.axis[0], 'vector_dup', _split_select())
    sch[tx_ub_t].emit_insn(tx_ub_t.op.axis[0], 'vector_dup', _split_select())
    sch[tx_ub_b].emit_insn(tx_ub_b.op.axis[0], 'vector_dup', _split_select())
    sch[tx_ub_l].emit_insn(tx_ub_l.op.axis[0], 'vector_dup', _split_select())
    sch[tx_ub_r].emit_insn(tx_ub_r.op.axis[0], 'vector_dup', _split_select())
    sch[tx_ub].emit_insn(tx_ub.op.axis[0], "phony_insn", _split_select())

    for tensor in pool_tensors.values():
        if tensor.op.tag == "reduce_max":
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_auto')

    sch[res].emit_insn(d_in, 'dma_copy')
    return True


def _calc_process_per_window_ub_size(context):
    c0_size = C0_DIMENSION_DATA_SIZE_MAP[context["dtype"]]
    k_d, k_h, k_w = context["kd"], context["kh"], context["kw"]
    s_d, s_h, s_w = context["sd"], context["sh"], context["sw"]
    dd = k_d if k_d > s_d else s_d
    hh = k_h if k_h > s_h else s_h
    ww = k_w if k_w > s_w else s_w
    fmap_size = dd * hh * ww * c0_size
    reduce_intermediate_data = (k_d * k_h * c0_size) + (k_d * c0_size)
    res_size = c0_size
    return fmap_size + reduce_intermediate_data + res_size


def _calc_window_numbers_per_batch(context):
    ub_size = cce.get_soc_spec("UB_SIZE")
    # For the reason pass may generate multi tx_rd/tx_rh/tx_rw tensors, so use half of the ub size
    ub_size = ub_size // 2
    return ub_size // _calc_process_per_window_ub_size(context)


def _calc_d_h_w_factor(context):
    o_d, o_h, o_w = context["do"], context["ho"], context["wo"]
    w_factor = h_factor = d_factor = 1
    n = _calc_window_numbers_per_batch(context)

    if o_w >= n:
        w_factor = n
        n = 1
    else:
        w_factor = o_w
        n = n // w_factor
    if o_h >= n:
        h_factor = n
        n = 1
    else:
        h_factor = o_h
        n = n // h_factor

    if o_d >= n:
        d_factor = n
    else:
        d_factor = o_d

    return d_factor, h_factor, w_factor


# calculate how many data can be processed per time, data unit is kd*kh*kw
def _calc_c1_factor(pool_params):
    return 1


# ' N  C1  D  H  W  C0    =>   N  C1o/C1i  D  H  W  C0
# '    ^                             ^
# '    |                             |
def _split_c1_axis(pool_params, res, sch):
    c1_factor = _calc_c1_factor(pool_params)
    c1_out, c1_in = sch[res].split(res.op.axis[1], factor=c1_factor)  # split C1
    return c1_out, c1_in


# ' N  C1o/C1i  D  H  W  C0    =>   N  C1o/C1i  Do/Di  H  W  C0
# '             ^                                 ^
# '             |                                 |
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


def _mem_reuse(sch, tensors, target_tensor):
    for tensor in tensors:
        sch[target_tensor].reused_by(tensor)
