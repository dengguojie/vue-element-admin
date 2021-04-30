# -*- coding:utf-8 -*-
from functools import reduce
from te import tvm
from te.platform import cce_params as param
from te.lang.cce.te_compute import irbuilder_api as kernel_api
from .interp_common import apply_store_buffer


def _inner_run_loop(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_out, core_lp3_idx, gm_offset, loop_level4, tail_level4, offset_4 = para_list

    with ib.for_range(0, loop_level4) as i5:
        with ib.new_scope():
            _inner_loop = 2048 * para.f16_c0 // free_space_fp32
            with ib.for_range(0, _inner_loop) as inner_idx:
                ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=(gm_offset + para.block.var) * offset_4 +
                                            i5 * 2048 * para.f16_c0 + inner_idx * free_space_fp32 +
                                            core_lp3_idx * para.core_counts * offset_4),
                                        f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))

    return ib


def _inner_run_tail(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_out, core_lp3_idx, gm_offset, loop_level4, tail_level4, offset_4 = para_list

    with ib.new_scope():
        _inner_loop = tail_level4 * para.f16_c0 // free_space_fp32
        _inner_tail = tail_level4 * para.f16_c0 % free_space_fp32
        with ib.for_range(0, _inner_loop) as inner_idx:
            ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(gm_offset + para.block.var) * offset_4 +
                                        loop_level4 * 2048 * para.f16_c0 + inner_idx * free_space_fp32 +
                                        core_lp3_idx * para.core_counts * offset_4),
                                    f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
        if _inner_tail > 0:
            ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(gm_offset + para.block.var) * offset_4 +
                                        loop_level4 * 2048 * para.f16_c0 + _inner_loop * free_space_fp32 +
                                        core_lp3_idx * para.core_counts * offset_4),
                                    f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _inner_run(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_out, core_lp3_idx, gm_offset = para_list
    with ib.new_scope():
        ib.emit(
            tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_8.access_ptr('w'),
                            l1_out.access_ptr(
                                'r',
                                offset=para.block.var * 8 * para.f16_c0 +
                                core_lp3_idx * para.core_counts * 8 * para.f16_c0),
                            0, 1, 8, 0, 0))
    repeat_64 = 512 // 8
    with ib.new_scope():
        ib.emit(tvm.call_extern("float16", "vadds", f16_out.access_ptr('w'), f16_8.access_ptr('r'),
                                tvm.const(0.0, dtype="float16"), repeat_64, 1, 1, 8, 0))
    with ib.new_scope():
        ib.emit(tvm.call_extern("float16", "copy_ubuf_to_ubuf", f16_out.access_ptr('w', offset=512 * para.f16_c0),
                                f16_out.access_ptr('r'), 0, 1, 512, 0, 0))
    with ib.new_scope():
        ib.emit(tvm.call_extern("float16", "copy_ubuf_to_ubuf", f16_out.access_ptr('w', offset=1024 * para.f16_c0),
                                f16_out.access_ptr('r'), 0, 1, 1024, 0, 0))

    loop_level4 = expand_size // 2048
    tail_level4 = expand_size % 2048
    offset_4 = expand_size * para.f16_c0

    loop_tail_para = para_list + [loop_level4, tail_level4, offset_4]
    if loop_level4 > 0:
        ib = _inner_run_loop(ib, para, loop_tail_para)
    if tail_level4 > 0:
        ib = _inner_run_tail(ib, para, loop_tail_para)

    return ib


def _large_data_loop_level1_expand_size_512(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, i1, i2 = large_data_param

    # level_3 loop
    loop_level3 = 8
    with ib.for_range(0, loop_level3) as i3:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i3 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, 512, 1, 0, 8 - 1))
    # level_3 loop
    loop_level3 = 512
    core_lp3 = loop_level3 // para.core_counts
    core_lp3_tail = loop_level3 % para.core_counts
    gm_offset = (i1 * 32 + i2) * 512
    inner_param = [expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8,
                   f16_64, l1_out]

    with ib.for_range(0, core_lp3, name="core_lp3") as _core_lp3_idx:
        ib = _inner_run(ib, para, inner_param + [_core_lp3_idx, gm_offset])
    if core_lp3_tail > 0:
        with ib.if_scope(para.block.var < core_lp3_tail):
            ib = _inner_run(ib, para, inner_param + [core_lp3, gm_offset])

    return ib


def _large_data_loop_level1_expand_size_32(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, i1, i2 = large_data_param

    # level_3 loop
    loop_level3 = 8
    with ib.for_range(0, loop_level3) as i3:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i3 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, 512, 1, 0, 8 - 1))
    # level_3 loop
    loop_level3 = 512 // 8
    with ib.for_range(0, loop_level3) as i4:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_64.access_ptr('w'),
                                    l1_out.access_ptr('r', offset=i4 * 64 * para.f16_c0), 0, 1, 64, 0, 0))
        # level_4 loop
        loop_level4 = 8
        offset_l4 = expand_size * para.f16_c0
        repeat_l4 = expand_size // 8 + (1 if expand_size % 8 > 0 else 0)
        with ib.for_range(0, loop_level4) as i5:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "vadds", f16_out.access_ptr('w', offset=i5 * offset_l4),
                                        f16_64.access_ptr('r', offset=i5 * 8 * para.f16_c0),
                                        tvm.const(0.0, dtype="float16"), repeat_l4, 1, 1, 8, 0))
        offset_4 = expand_size * para.f16_c0
        expand_4 = 8 * expand_size
        with ib.new_scope():
            _inner_loop = expand_4 * para.f16_c0 // free_space_fp32
            _inner_tail = expand_4 * para.f16_c0 % free_space_fp32
            with ib.for_range(0, _inner_loop) as inner_idx:
                ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=((i1 * 32 + i2) * 512 + i4 * 8) * offset_4 +
                                        inner_idx * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
            if _inner_tail > 0:
                ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=((i1 * 32 + i2) * 512 + i4 * 8) * offset_4 +
                                        _inner_loop * free_space_fp32),
                                    f32_out.access_ptr('r'),
                                    0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _large_data_loop_level1_expand_size_0_loop_level3(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, i1, i2, loop_level3, tail_level3 = large_data_param
    with ib.for_range(0, loop_level3) as i4:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_out.access_ptr('w'),
                                    l1_out.access_ptr('r', offset=i4 * 512 * 12 * para.f16_c0), 0, 1, 512 * 12, 0, 0))
        with ib.new_scope():
            _inner_loop = 512 * 12 * para.f16_c0 // free_space_fp32
            _inner_tail = 512 * 12 * para.f16_c0 % free_space_fp32
            with ib.for_range(0, _inner_loop) as inner_idx:
                ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(((i1 * 32 + i2) * expand_size) + i4 * 12) * 512 * para.f16_c0 +
                                        inner_idx * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
            if _inner_tail > 0:
                ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(((i1 * 32 + i2) * expand_size) + i4 * 12) * 512 * para.f16_c0 +
                                        _inner_loop * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _large_data_loop_level1_expand_size_0_tail_level3(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, i1, i2, loop_level3, tail_level3 = large_data_param
    with ib.new_scope():
        ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_out.access_ptr('w'),
                                l1_out.access_ptr('r', offset=loop_level3 * 512 * 12 * para.f16_c0), 0, 1,
                                512 * tail_level3, 0, 0))
    with ib.new_scope():
        _inner_loop = 512 * tail_level3 * para.f16_c0 // free_space_fp32
        _inner_tail = 512 * tail_level3 * para.f16_c0 % free_space_fp32
        with ib.for_range(0, _inner_loop) as inner_idx:
            ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(((i1 * 32 + i2) * expand_size) + loop_level3 * 12) * 512 *
                                        para.f16_c0 + inner_idx * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
        if _inner_tail > 0:
            ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(((i1 * 32 + i2) * expand_size) + loop_level3 * 12) * 512 *
                                        para.f16_c0 + _inner_loop * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _large_data_loop_level1_expand_size_0(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, i1, i2 = large_data_param

    with ib.for_range(0, expand_size) as i3:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i3 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, 512, 1, 0, expand_size - 1))
    loop_level3 = expand_size // 12
    tail_level3 = expand_size % 12
    loop_tail_para = large_data_param + [loop_level3, tail_level3]
    if loop_level3 > 0:
        ib = _large_data_loop_level1_expand_size_0_loop_level3(ib, para, loop_tail_para)
    if tail_level3 > 0:
        ib = _large_data_loop_level1_expand_size_0_tail_level3(ib, para, loop_tail_para)

    return ib


def _large_data_loop_level1(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1 = large_data_param
    # level_1 loop
    with ib.for_range(0, loop_level1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_gm_to_cbuf", l1_in.access_ptr('w'),
                                    para.inputs.access_ptr('r', offset=i1 * l1_half * para.f16_c0),
                                    para.sid, 1, l1_half, 0, 0, para.pad_mode))
        # level_2 loop
        loop_level2 = 32
        with ib.for_range(0, loop_level2) as i2:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_in.access_ptr('w'),
                                        l1_in.access_ptr('r', offset=i2 * 512 * para.f16_c0), 0, 1, 512, 0, 0))
            # Note:
            #   1. H * W > 512 is common
            #   2. 512 >= H * W > 32
            #   3. H * W <= 32 this happen really rare

            # No.2 situation : 1. input f16 dtype : large data : 1. H * W > 512
            if expand_size > 512:
                ib = _large_data_loop_level1_expand_size_512(
                    ib, para, large_data_param + [i1, i2])
            elif expand_size > 32:
                ib = _large_data_loop_level1_expand_size_32(
                    ib, para, large_data_param + [i1, i2])
            else:
                ib = _large_data_loop_level1_expand_size_0(
                    ib, para, large_data_param + [i1, i2])

    return ib


def _large_data_tail_level1_loop_l1_expand_size_512(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, i1 = large_data_tail_param

    # level_2 loop
    loop_l2 = 8
    with ib.for_range(0, loop_l2) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i2 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, 512, 1, 0, 8 - 1))
    # level_2 loop
    loop_l2 = 512
    core_lp2 = loop_l2 // para.core_counts
    core_lp2_tail = loop_l2 % para.core_counts
    gm_offset = (loop_level1 * 32 + i1) * 512
    inner_param = [expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8,
                   f16_64, l1_out]

    with ib.for_range(0, core_lp2, name="core_lp2") as _core_lp2_idx:
        ib = _inner_run(ib, para, inner_param + [_core_lp2_idx, gm_offset])
    if core_lp2_tail > 0:
        with ib.if_scope(para.block.var < core_lp2_tail):
            ib = _inner_run(ib, para, inner_param + [core_lp2, gm_offset])

    return ib


def _large_data_tail_level1_loop_l1_expand_size_32(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, i1 = large_data_tail_param

    # No.2 situation : 1. input f16 dtype : large data : 2. 512>=H*W>32
    # level_2 loop
    loop_l2 = 8
    with ib.for_range(0, loop_l2) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i2 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, 512, 1, 0, 8 - 1))
    # level_2 loop
    loop_l2 = 512 // 8
    with ib.for_range(0, loop_l2) as i3:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_64.access_ptr('w'),
                                    l1_out.access_ptr('r', offset=i3 * 64 * para.f16_c0), 0, 1, 64, 0, 0))
        # level_3 loop
        loop_l3 = 8
        offset_l3 = expand_size * para.f16_c0
        repeat_l3 = expand_size // 8 + (1 if expand_size % 8 > 0 else 0)
        with ib.for_range(0, loop_l3) as i4:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "vadds",
                                        f16_out.access_ptr('w', offset=i4 * offset_l3),
                                        f16_64.access_ptr(
                                            'r',
                                            offset=i4 * 8 * para.f16_c0), tvm.const(0.0, dtype="float16"),
                                        repeat_l3, 1, 1, 8, 0))
        offset_4 = expand_size * para.f16_c0
        expand_4 = 8 * expand_size
        with ib.new_scope():
            _inner_loop = expand_4 * para.f16_c0 // free_space_fp32
            _inner_tail = expand_4 * para.f16_c0 % free_space_fp32
            with ib.for_range(0, _inner_loop) as inner_idx:
                ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=((loop_level1 * 32 + i1) * 512 + i3 * 8) * offset_4 +
                                            inner_idx * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
            if _inner_tail > 0:
                ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=((loop_level1 * 32 + i1) * 512 + i3 * 8) * offset_4 +
                                            _inner_loop * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _large_data_tail_level1_loop_l1_expand_size_0_tail_l2(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, i1, loop_l2, tail_l2 = large_data_tail_param
    with ib.new_scope():
        ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_out.access_ptr('w'),
                                l1_out.access_ptr('r', offset=loop_l2 * 512 * 12 * para.f16_c0),
                                0, 1, 512 * tail_l2, 0, 0))
    with ib.new_scope():
        _inner_loop = 512 * tail_l2 * para.f16_c0 // free_space_fp32
        _inner_tail = 512 * tail_l2 * para.f16_c0 % free_space_fp32
        with ib.for_range(0, _inner_loop) as inner_idx:
            ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=(((loop_level1 * 32 + i1) * expand_size) + loop_l2 * 12) * 512 *
                                    para.f16_c0 + inner_idx * free_space_fp32),
                                f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
        if _inner_tail > 0:
            ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=(((loop_level1 * 32 + i1) * expand_size) + loop_l2 * 12) * 512 *
                                           para.f16_c0 + _inner_loop * free_space_fp32),
                                f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _large_data_tail_level1_loop_l1_expand_size_0(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, i1 = large_data_tail_param

    # No.2 situation : 1. input f16 dtype : large data  3. H * W <= 32
    with ib.for_range(0, expand_size) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i2 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, 512, 1, 0, expand_size - 1))
    loop_l2 = expand_size // 12
    tail_l2 = expand_size % 12
    if loop_l2 > 0:
        with ib.for_range(0, loop_l2) as i3:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_out.access_ptr('w'),
                                        l1_out.access_ptr('r', offset=i3 * 512 * 12 * para.f16_c0),
                                        0, 1, 512 * 12, 0, 0))
            with ib.new_scope():
                _inner_loop = 512 * 12 * para.f16_c0 // free_space_fp32
                with ib.for_range(0, _inner_loop) as inner_idx:
                    ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                    kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                    ib.emit(
                        tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=(((loop_level1 * 32 + i1) * expand_size) + i3 * 12) * 512 *
                                                   para.f16_c0 + inner_idx * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))

    loop_tail_para = large_data_tail_param + [loop_l2, tail_l2]
    if tail_l2 > 0:
        ib = _large_data_tail_level1_loop_l1_expand_size_0_tail_l2(ib, para, loop_tail_para)

    return ib


def _large_data_tail_level1_loop_l1(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1 = large_data_tail_param
    with ib.for_range(0, loop_l1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_in.access_ptr('w'),
                                    l1_in.access_ptr('r', offset=i1 * 512 * para.f16_c0), 0, 1, 512, 0, 0))
        # Note:
        #   1. H * W >= 512 is common
        #   2. 512 > H * W > 32
        #   3. H * W <= 32 this happen really rare

        # No.2 situation : 1. input f16 dtype : large data : 1. H * W >= 512
        if expand_size > 512:
            ib = _large_data_tail_level1_loop_l1_expand_size_512(
                ib, para, large_data_tail_param + [i1])
        elif expand_size > 32:
            ib = _large_data_tail_level1_loop_l1_expand_size_32(
                ib, para, large_data_tail_param + [i1])
        else:
            ib = _large_data_tail_level1_loop_l1_expand_size_0(
                ib, para, large_data_tail_param + [i1])

    return ib


def _large_data_tail_level1_tail_l1_expand_size_512_loop_le2(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, \
        f16_8, f16_64, l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, loop_le2, \
        tail_le2, offset_2, i2 = large_data_tail_param

    with ib.for_range(0, loop_le2) as i3:
        with ib.new_scope():
            _inner_loop = 2048 * para.f16_c0 // free_space_fp32
            with ib.for_range(0, _inner_loop) as inner_idx:
                ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=((loop_level1 * 32 + loop_l1) * 512 + i2) * offset_2 + i3 * 2048 *
                                               para.f16_c0 + inner_idx * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
    return ib


def _large_data_tail_level1_tail_l1_expand_size_512_tail_le2(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, \
        f16_8, f16_64, l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, loop_le2, \
        tail_le2, offset_2, i2 = large_data_tail_param

    with ib.new_scope():
        _inner_loop = tail_le2 * para.f16_c0 // free_space_fp32
        _inner_tail = tail_le2 * para.f16_c0 % free_space_fp32
        with ib.for_range(0, _inner_loop) as inner_idx:
            ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=((loop_level1 * 32 + loop_l1) * 512 + i2) * offset_2 + loop_le2 * 2048 *
                                           para.f16_c0 + inner_idx * free_space_fp32),
                                f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
        if _inner_tail > 0:
            ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=((loop_level1 * 32 + loop_l1) * 512 + i2) * offset_2 + loop_le2 * 2048 *
                                           para.f16_c0 + _inner_loop * free_space_fp32),
                                f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _large_data_tail_level1_tail_l1_expand_size_512(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1 = large_data_tail_param
    # level_1 loop
    loop_le1 = 8
    with ib.for_range(0, loop_le1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i1 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, tail_l1, 1, 0, 8 - 1))
    # level_1 loop
    loop_le1 = tail_l1
    with ib.for_range(0, loop_le1) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_8.access_ptr('w'),
                                    l1_out.access_ptr('r', offset=i2 * 8 * para.f16_c0), 0, 1, 8, 0, 0))
        repeat_64 = 512 // 8
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "vadds", f16_out.access_ptr('w'), f16_8.access_ptr('r'),
                                    tvm.const(0.0, dtype="float16"), repeat_64, 1, 1, 8, 0))
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_ubuf", f16_out.access_ptr('w', offset=512 * para.f16_c0),
                                    f16_out.access_ptr('r'), 0, 1, 512, 0, 0))
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_ubuf", f16_out.access_ptr('w', offset=1024 * para.f16_c0),
                                    f16_out.access_ptr('r'), 0, 1, 1024, 0, 0))
        loop_le2 = expand_size // 2048
        tail_le2 = expand_size % 2048
        offset_2 = expand_size * para.f16_c0
        loop_tail_param = large_data_tail_param + [loop_le2, tail_le2, offset_2, i2]
        if loop_le2 > 0:
            ib = _large_data_tail_level1_tail_l1_expand_size_512_loop_le2(
                ib, para, loop_tail_param)
        if tail_le2 > 0:
            ib = _large_data_tail_level1_tail_l1_expand_size_512_tail_le2(
                ib, para, loop_tail_param)

    return ib


def _large_data_tail_level1_tail_l1_expand_size_32_loop_le1(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, loop_le1, tail_le1 = large_data_tail_param
    with ib.for_range(0, loop_le1) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_64.access_ptr('w'),
                                    l1_out.access_ptr('r', offset=i2 * 64 * para.f16_c0), 0, 1, 64, 0, 0))
        # level_2 loop
        loop_le2 = 8
        offset_le2 = expand_size * para.f16_c0
        repeat_le2 = expand_size // 8 + (1 if expand_size % 8 > 0 else 0)
        with ib.for_range(0, loop_le2) as i3:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "vadds", f16_out.access_ptr('w', offset=i3 * offset_le2),
                                        f16_64.access_ptr('r', offset=i3 * 8 * para.f16_c0),
                                        tvm.const(0.0, dtype="float16"), repeat_le2, 1, 1, 8, 0))
        offset_4 = expand_size * para.f16_c0
        expand_4 = 8 * expand_size
        with ib.new_scope():
            _inner_loop = expand_4 * para.f16_c0 // free_space_fp32
            _inner_tail = expand_4 * para.f16_c0 % free_space_fp32
            with ib.for_range(0, _inner_loop) as inner_idx:
                ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=((loop_level1 * 32 + loop_l1) * 512 + i2 * 8) * offset_4 +
                                                   inner_idx * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
            if _inner_tail > 0:
                ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=((loop_level1 * 32 + loop_l1) * 512 + i2 * 8) * offset_4 +
                                               _inner_loop * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _large_data_tail_level1_tail_l1_expand_size_32_tail_le1(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, loop_le1, tail_le1 = large_data_tail_param
    with ib.new_scope():
        ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_64.access_ptr('w'),
                                l1_out.access_ptr('r', offset=loop_le1 * 64 * para.f16_c0), 0, 1, tail_le1 * 8, 0, 0))
    # level_2 loop
    loop_le2 = tail_le1
    offset_le2 = expand_size * para.f16_c0
    repeat_le2 = expand_size // 8 + (1 if expand_size % 8 > 0 else 0)
    with ib.for_range(0, loop_le2) as i4:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "vadds", f16_out.access_ptr('w', offset=i4 * offset_le2),
                                    f16_64.access_ptr('r', offset=i4 * 8 * para.f16_c0),
                                    tvm.const(0.0, dtype="float16"), repeat_le2, 1, 1, 8, 0))
    offset_4 = expand_size * para.f16_c0
    expand_4 = tail_le1 * expand_size
    with ib.new_scope():
        _inner_loop = expand_4 * para.f16_c0 // free_space_fp32
        _inner_tail = expand_4 * para.f16_c0 % free_space_fp32
        with ib.for_range(0, _inner_loop) as inner_idx:
            ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=((loop_level1 * 32 + loop_l1) * 512 + loop_le1 * 8) * offset_4 +
                                           inner_idx * free_space_fp32),
                                f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
        if _inner_tail > 0:
            ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=((loop_level1 * 32 + loop_l1) * 512 + loop_le1 * 8) * offset_4 +
                                           _inner_loop * free_space_fp32),
                                f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _large_data_tail_level1_tail_l1_expand_size_32(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1 = large_data_tail_param
    # No.2 situation : 1. input f16 dtype : large data : 2. 512>=H*W>32
    # level_1 loop
    loop_le1 = 8
    with ib.for_range(0, loop_le1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i1 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, tail_l1, 1, 0, 8 - 1))
    # level_1 loop
    loop_le1 = tail_l1 // 8
    tail_le1 = tail_l1 % 8
    loop_tail_para = large_data_tail_param + [loop_le1, tail_le1]
    if loop_le1 > 0:
        ib = _large_data_tail_level1_tail_l1_expand_size_32_loop_le1(ib, para, loop_tail_para)
    if tail_le1 > 0:
        ib = _large_data_tail_level1_tail_l1_expand_size_32_tail_le1(ib, para, loop_tail_para)

    return ib


def _large_data_tail_level1_tail_l1_expand_size_0_loop_le2(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, loop_le2, tail_le2 = large_data_tail_param

    with ib.for_range(0, loop_le2) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_out.access_ptr('w'),
                                    l1_out.access_ptr('r', offset=i2 * 512 * 12 * para.f16_c0), 0, 1, 512 * 12, 0, 0))
        with ib.new_scope():
            _inner_loop = 512 * 12 * para.f16_c0 // free_space_fp32
            with ib.for_range(0, _inner_loop) as inner_idx:
                ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(((loop_level1 * 32 + loop_l1) * expand_size) + i2 * 12) *
                                               512 * para.f16_c0 + inner_idx * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))

    return ib


def _large_data_tail_level1_tail_l1_expand_size_0_tail_le2(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1, loop_le2, tail_le2 = large_data_tail_param

    with ib.new_scope():
        ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_out.access_ptr('w'),
                                l1_out.access_ptr('r', offset=loop_le2 * 512 * 12 * para.f16_c0),
                                0, 1, tail_le2, 0, 0))
    with ib.new_scope():
        _inner_loop = tail_le2 * para.f16_c0 // free_space_fp32
        _inner_tail = tail_le2 * para.f16_c0 % free_space_fp32
        with ib.for_range(0, _inner_loop) as inner_idx:
            ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=(((loop_level1 * 32 + loop_l1) * expand_size) + loop_le2 * 12) * 512 *
                                           para.f16_c0 + inner_idx * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
        if _inner_tail > 0:
            ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=(((loop_level1 * 32 + loop_l1) * expand_size) + loop_le2 * 12) * 512 *
                                           para.f16_c0 + _inner_loop * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _large_data_tail_level1_tail_l1(ib, para, large_data_tail_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1, loop_l1, tail_l1 = large_data_tail_param
    if tail_level1 > 512:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_in.access_ptr('w'),
                                    l1_in.access_ptr('r', offset=loop_l1 * 512 * para.f16_c0), 0, 1, tail_l1, 0, 0))
    # Note:
    #   1. H * W >= 512 is common
    #   2. 512 > H * W > 32
    #   3. H * W <= 32 this happen really rare

    # No.2 situation : 1. input f16 dtype : large data : 1. H * W >= 512
    if expand_size > 512:
        ib = _large_data_tail_level1_tail_l1_expand_size_512(
            ib, para, large_data_tail_param)
    elif expand_size > 32:
        ib = _large_data_tail_level1_tail_l1_expand_size_32(
            ib, para, large_data_tail_param)
    else:
        # No.2 situation : 1. input f16 dtype : large data  3. H * W <= 32
        with ib.for_range(0, expand_size) as i1:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i1 * para.f16_c0),
                                        f16_in.access_ptr('r'), 0, tail_l1, 1, 0, expand_size - 1))

    loop_le2 = tail_l1 * expand_size // (12 * 512)
    tail_le2 = tail_l1 * expand_size % (12 * 512)

    loop_tail_le2_param = large_data_tail_param + [loop_le2, tail_le2]
    if loop_le2 > 0:
        ib = _large_data_tail_level1_tail_l1_expand_size_0_loop_le2(
            ib, para, loop_tail_le2_param)
    if tail_le2 > 0:
        ib = _large_data_tail_level1_tail_l1_expand_size_0_tail_le2(
            ib, para, loop_tail_le2_param)

    return ib


def _large_data_tail_level1(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_half, l1_in, l1_out, loop_level1, tail_level1 = large_data_param
    # Note:
    #   1. tail_level1 larger then 512 need move to L1 first
    #   2. tail_level1 less then 512 move to ubuf directly

    if tail_level1 > 512:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float16", "copy_gm_to_cbuf", l1_in.access_ptr('w'),
                                para.inputs.access_ptr(
                                    'r',
                                    offset=loop_level1 * l1_half * para.f16_c0), para.sid, 1, tail_level1, 0, 0,
                                para.pad_mode))
    else:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float16", "copy_gm_to_ubuf",
                                f16_in.access_ptr('w'),
                                para.inputs.access_ptr(
                                    'r',
                                    offset=loop_level1 * l1_half * para.f16_c0), 0, 1, tail_level1, 0, 0))
    # level_1 loop
    loop_l1 = tail_level1 // 512
    tail_l1 = tail_level1 % 512

    large_data_tail_param = large_data_param + [loop_l1, tail_l1]
    # No.2 situation : 1. input f16 dtype : large data : tail_level1>0 : loop_l1>0
    if loop_l1 > 0:
        ib = _large_data_tail_level1_loop_l1(ib, para, large_data_tail_param)
    # No.2 situation : 1. input f16 dtype : large data : tail_level1>0 : tail_l1>0
    if tail_l1 > 0:
        ib = _large_data_tail_level1_tail_l1(ib, para, large_data_tail_param)

    return ib


def _large_data(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, \
        f16_64 = para_list
    l1_half = 512 * 32
    l1_in = apply_store_buffer(ib, "float16", [l1_half * para.f16_c0], name="l1_in", scope=param.scope_cbuf)
    l1_out = apply_store_buffer(ib, "float16", [l1_half * para.f16_c0], name="l1_out", scope=param.scope_cbuf)

    loop_level1 = expand_loop // l1_half
    tail_level1 = expand_loop % l1_half

    large_data_param = para_list + [l1_half, l1_in, l1_out, loop_level1, tail_level1]
    # No.2 situation : 1. input f16 dtype : large data : loop_level1 > 0
    if loop_level1 > 0:
        ib = _large_data_loop_level1(ib, para, large_data_param)
    # No.2 situation : 1. input f16 dtype : large data : tail_level1 > 0
    if tail_level1 > 0:
        ib = _large_data_tail_level1(ib, para, large_data_param)

    return ib


def _small_data_expand_size_512(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out,\
        f16_8, f16_64, l1_out = para_list

    # level_1 loop
    loop_lev1 = 8
    with ib.for_range(0, loop_lev1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i1 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, expand_loop, 1, 0, 8 - 1))
    # level_1 loop
    loop_lev1 = expand_loop
    core_lp1 = loop_lev1 // para.core_counts
    core_lp1_tail = loop_lev1 % para.core_counts
    gm_offset = 0
    inner_param = [expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8,
                   f16_64, l1_out]

    # level_1 loop
    with ib.for_range(0, core_lp1, name="core_lp1") as _core_lp1_idx:
        ib = _inner_run(ib, para, inner_param + [_core_lp1_idx, gm_offset])
    if core_lp1_tail > 0:
        with ib.if_scope(para.block.var < core_lp1_tail):
            ib = _inner_run(ib, para, inner_param + [core_lp1, gm_offset])

    return ib


def _small_data_expand_size_32_loop_lev1(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out,\
        f16_8, f16_64, l1_out, loop_lev1, tail_lev1 = para_list
    with ib.for_range(0, loop_lev1) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_64.access_ptr('w'),
                                    l1_out.access_ptr('r', offset=i2 * 64 * para.f16_c0), 0, 1, 64, 0, 0))
        # level_2 loop
        loop_lev2 = 8
        offset_lev2 = expand_size * para.f16_c0
        repeat_lev2 = expand_size // 8 + (1 if expand_size % 8 > 0 else 0)
        with ib.for_range(0, loop_lev2) as i3:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "vadds", f16_out.access_ptr('w', offset=i3 * offset_lev2),
                                        f16_64.access_ptr('r', offset=i3 * 8 * para.f16_c0),
                                        tvm.const(0.0, dtype="float16"), repeat_lev2, 1, 1, 8, 0))
        offset_4 = expand_size * para.f16_c0
        expand_4 = 8 * expand_size
        with ib.new_scope():
            _inner_loop = expand_4 * para.f16_c0 // free_space_fp32
            _inner_tail = expand_4 * para.f16_c0 % free_space_fp32
            with ib.for_range(0, _inner_loop) as inner_idx:
                ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=i2 * 8 * offset_4 + inner_idx * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
            if _inner_tail > 0:
                ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=i2 * 8 * offset_4 + _inner_loop * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _small_data_expand_size_32_tail_lev1(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out,\
        f16_8, f16_64, l1_out, loop_lev1, tail_lev1 = para_list
    with ib.new_scope():
        ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_64.access_ptr('w'),
                                l1_out.access_ptr('r', offset=loop_lev1 * 64 * para.f16_c0), 0, 1, tail_lev1 * 8, 0, 0))
    # level_2 loop
    loop_lev2 = tail_lev1
    offset_lev2 = expand_size * para.f16_c0
    repeat_lev2 = expand_size // 8 + (1 if expand_size % 8 > 0 else 0)
    with ib.for_range(0, loop_lev2) as i4:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "vadds", f16_out.access_ptr('w', offset=i4 * offset_lev2),
                                    f16_64.access_ptr('r', offset=i4 * 8 * para.f16_c0),
                                    tvm.const(0.0, dtype="float16"), repeat_lev2, 1, 1, 8, 0))
    offset_4 = expand_size * para.f16_c0
    expand_4 = tail_lev1 * expand_size
    with ib.new_scope():
        _inner_loop = expand_4 * para.f16_c0 // free_space_fp32
        _inner_tail = expand_4 * para.f16_c0 % free_space_fp32
        with ib.for_range(0, _inner_loop) as inner_idx:
            ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=loop_lev1 * 8 * offset_4 + inner_idx * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))
        if _inner_tail > 0:
            ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=loop_lev1 * 8 * offset_4 + _inner_loop * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _small_data_expand_size_32(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, \
       f16_8, f16_64, l1_out = para_list

    # No.2 situation : 1. input f16 dtype : small data : 2. 512>=H*W>32
    # level_1 loop
    loop_lev1 = 8
    with ib.for_range(0, loop_lev1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i1 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, expand_loop, 1, 0, 8 - 1))
    # level_1 loop
    loop_lev1 = expand_loop // 8
    tail_lev1 = expand_loop % 8
    loop_tail_para = para_list + [loop_lev1, tail_lev1]
    if loop_lev1 > 0:
        ib = _small_data_expand_size_32_loop_lev1(ib, para, loop_tail_para)
    if tail_lev1 > 0:
        ib = _small_data_expand_size_32_tail_lev1(ib, para, loop_tail_para)

    return ib


def _small_data_expand_size_0_loop_lev2(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out,\
        f16_8, f16_64, l1_out, loop_lev2, tail_lev2 = para_list
    with ib.for_range(0, loop_lev2) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_out.access_ptr('w'),
                                    l1_out.access_ptr('r', offset=i2 * 512 * 12 * para.f16_c0), 0, 1, 512 * 12, 0, 0))
        with ib.new_scope():
            _inner_loop = 512 * 12 * para.f16_c0 // free_space_fp32
            with ib.for_range(0, _inner_loop) as inner_idx:
                ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=i2 * 12 * 512 * para.f16_c0 + inner_idx * free_space_fp32),
                                        f32_out.access_ptr('r'), 0, 1, free_space_fp32 // 8, 0, 0))

    return ib


def _small_data_expand_size_0_tail_lev2(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8, f16_64, \
        l1_out, loop_lev2, tail_lev2 = para_list
    with ib.new_scope():
        ib.emit(
            tvm.call_extern("float16", "copy_cbuf_to_ubuf", f16_out.access_ptr('w'),
                            l1_out.access_ptr('r', offset=loop_lev2 * 512 * 12 * para.f16_c0),
                            0, 1, tail_lev2, 0, 0))
    with ib.new_scope():
        _inner_loop = tail_lev2 * para.f16_c0 // free_space_fp32
        _inner_tail = tail_lev2 * para.f16_c0 % free_space_fp32
        with ib.for_range(0, _inner_loop) as inner_idx:
            ip_addr = [[f32_out, 0], [f16_out, inner_idx * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [free_space_fp32, 8 * 8], "vconv_f162f32")
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=loop_lev2 * 12 * 512 * para.f16_c0 + inner_idx * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, free_space_fp32, 0, 0))
        if _inner_tail > 0:
            ip_addr = [[f32_out, 0], [f16_out, _inner_loop * free_space_fp32]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [_inner_tail, 8 * 8], "vconv_f162f32")
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=loop_lev2 * 12 * 512 * para.f16_c0 + _inner_loop * free_space_fp32),
                                    f32_out.access_ptr('r'), 0, 1, (_inner_tail + 7) // 8, 0, 0))

    return ib


def _small_data_expand_size_0(ib, para, para_list):
    expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, \
       f16_8, f16_64, l1_out = para_list

    # No.2 situation : 1. input f16 dtype : small data  3. H * W <= 16
    with ib.for_range(0, expand_size) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_ubuf_to_cbuf", l1_out.access_ptr('w', offset=i1 * para.f16_c0),
                                    f16_in.access_ptr('r'), 0, expand_loop, 1, 0, expand_size - 1))
    loop_lev2 = expand_loop * expand_size // (12 * 512)
    tail_lev2 = expand_loop * expand_size % (12 * 512)
    loop_tail_para = para_list + [loop_lev2, tail_lev2]
    if loop_lev2 > 0:
        ib = _small_data_expand_size_0_loop_lev2(ib, para, loop_tail_para)
    if tail_lev2 > 0:
        ib = _small_data_expand_size_0_tail_lev2(ib, para, loop_tail_para)

    return ib


# No.2 situation : 1. input f16 dtype
def compute_with_in_hw_eq_one_fp16(ib, para):
    expand_loop = reduce(lambda x, y: x * y, para.size_in) // para.f16_c0
    actual_loop = min(expand_loop, 512)
    free_space = 512 * 12
    free_space_fp32 = 512 * para.f16_c0
    expand_size = para.h_out * para.w_out
    ib.scope_attr(para.block, "thread_extent", para.core_counts)

    f16_in = apply_store_buffer(ib, "float16", [actual_loop * para.f16_c0], name="f16_in")
    f16_out = apply_store_buffer(ib, "float16", [free_space * para.f16_c0], name="f16_out")
    f32_out = apply_store_buffer(ib, "float32", [free_space_fp32], name="f32_out")

    f16_8 = None
    f16_64 = None
    if expand_size > 512:
        f16_8 = apply_store_buffer(ib, "float16", [8 * para.f16_c0], name="f16_8")
    elif expand_size > 32:
        f16_64 = apply_store_buffer(ib, "float16", [64 * para.f16_c0], name="f16_64")
    # Note:
    #   1. input data larger than 512 should use L1 optimize
    #   2. input small data do not need L1

    if expand_size > 32:
        ib.emit(tvm.call_extern("uint64", "set_vector_mask", tvm.const((1 << 64) - 1, dtype="uint64"),
                                tvm.const((1 << 64) - 1, dtype="uint64")))

    para_list = [expand_loop, actual_loop, free_space, free_space_fp32, expand_size, f16_in, f16_out, f32_out, f16_8,
                 f16_64]
    # No.2 situation : 1. input f16 dtype : large data
    if expand_loop > 512:
        ib = _large_data(ib, para, para_list)
    else:
        # No.2 situation : 1. input f16 dtype : small data
        l1_half = 512 * 32
        l1_out = apply_store_buffer(ib, "float16", [l1_half * para.f16_c0], name="l1_out", scope=param.scope_cbuf)
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float16", "copy_gm_to_ubuf", f16_in.access_ptr('w'), para.inputs.access_ptr('r'),
                                0, 1, expand_loop, 0, 0))
        # Note:
        #   1. H * W >= 512 is common
        #   2. 512 > H * W > 32
        #   3. H * W <= 32 this happen really rare

        small_para_list = para_list + [l1_out]
        # No.2 situation : 1. input f16 dtype : small data : 1. H * W >= 512
        if expand_size > 512:
            ib = _small_data_expand_size_512(ib, para, small_para_list)
        elif expand_size > 32:
            ib = _small_data_expand_size_32(ib, para, small_para_list)
        else:
            ib = _small_data_expand_size_0(ib, para, small_para_list)

    return ib
