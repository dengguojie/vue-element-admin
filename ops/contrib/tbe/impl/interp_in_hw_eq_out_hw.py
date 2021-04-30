# -*- coding:utf-8 -*-
from functools import reduce
from te import tvm
from te.platform import cce_params as param
from te.lang.cce.te_compute import irbuilder_api as kernel_api
from .interp_common import apply_store_buffer


def _inner_run(ib, para, core_lp_idx, ub_block, trans_station, trans_station_fp32=None):
    with ib.new_scope():
        ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
        ib.emit(tvm.call_extern(para.dtype, "copy_gm_to_ubuf", trans_station.access_ptr('w'),
                                para.inputs.access_ptr(
                                    'r',
                                    offset=para.block.var * ub_block * para.c0 +
                                            core_lp_idx * para.core_counts * ub_block * para.c0),
                                0, 1, ub_block, 0, 0))

    if para.dtype.lower() == "float32":
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
            ib.emit(tvm.call_extern(para.dtype, "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=para.block.var * ub_block * para.c0 +
                                                core_lp_idx * para.core_counts * ub_block * para.c0),
                                    trans_station.access_ptr('r'), 0, 1, ub_block, 0, 0))
    else:
        with ib.new_scope():
            ip_addr = [[trans_station_fp32, 0], [trans_station, 0]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [ub_block * 16, 8 * 8], "vconv_f162f32")
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm", para.outputs.access_ptr(
                'w',
                offset=para.block.var * ub_block * para.c0 + core_lp_idx * para.core_counts * ub_block * para.c0),
                                    trans_station_fp32.access_ptr('r'), 0, 1, ub_block * 2, 0, 0))

    return ib


# No.1 situation : input H/W == output H/W
def compute_with_in_hw_eq_out_hw(ib, para):
    # Note:
    #   1. all data move in and out directly whatever the dtype is
    #   2. brust length is certainty in limit range
    ib.scope_attr(para.block, "thread_extent", para.core_counts)
    # max ubuf block count
    ub_block = 62 * 1024 // 32
    data_block = reduce(lambda x, y: x * y, para.size_in) // para.c0
    move_loop = data_block // ub_block
    core_loop = move_loop // para.core_counts
    core_loop_tail = move_loop % para.core_counts
    tail_counts = data_block % ub_block
    trans_station = apply_store_buffer(ib, para.dtype, [ub_block * para.c0], name="trans_station")
    if para.dtype.lower() == "float16":
        trans_station_fp32 = apply_store_buffer(ib, "float32", [ub_block * para.c0], name="trans_station_fp32")
    else:
        trans_station_fp32 = None

    with ib.for_range(0, core_loop, name="core_loop") as core_lp_idx:
        _inner_run(ib, para, core_lp_idx, ub_block, trans_station, trans_station_fp32)
    if core_loop_tail > 0:
        with ib.if_scope(para.block.var < core_loop_tail):
            ib = _inner_run(ib, para, core_loop, ub_block, trans_station, trans_station_fp32)

    if tail_counts > 0:
        with ib.if_scope(para.block.var < 1):
            tail_start = move_loop * ub_block
            with ib.new_scope():
                ib.scope_attr(param.CCE_AXIS, "coproc_scope", 5)
                ib.emit(tvm.call_extern(para.dtype, "copy_gm_to_ubuf", trans_station.access_ptr('w'),
                                        para.inputs.access_ptr('r', offset=tail_start * para.c0), 0, 1, tail_counts, 0,
                                        0))

            if para.dtype.lower() == "float32":
                with ib.new_scope():
                    ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                    ib.emit(tvm.call_extern(para.dtype, "copy_ubuf_to_gm",
                                            para.outputs.access_ptr('w', offset=tail_start * para.c0),
                                            trans_station.access_ptr('r'), 0, 1, tail_counts, 0, 0))
            else:
                with ib.new_scope():
                    ip_addr = [[trans_station_fp32, 0], [trans_station, 0]]
                    kernel_api.kernel_cast_to_fuc(ib, ip_addr, [tail_counts * 16, 8 * 8], "vconv_f162f32")
                    ib.scope_attr(param.CCE_AXIS, "coproc_scope", 6)
                    ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                            para.outputs.access_ptr('w', offset=tail_start * para.c0),
                                            trans_station_fp32.access_ptr('r'), 0, 1, tail_counts * 2, 0, 0))

    return ib
