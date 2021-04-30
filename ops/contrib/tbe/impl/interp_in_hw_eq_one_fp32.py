# -*- coding:utf-8 -*-
from functools import reduce
from te import tvm
from te.platform import cce_params as param
from .interp_common import apply_store_buffer


def _large_data_loop_level1_expand_size_512(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, \
            l1_half, in_l1, out_l1, f32_half, loop_level1, tail_level1, i1, i2 = large_data_param

    # level_3 loop
    loop_level3 = 4
    with ib.for_range(0, loop_level3) as i3:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                out_l1.access_ptr('w', offset=i3 * para.f16_c0),
                                f32_in.access_ptr('r'), 0, 512, 2, 0, (4 - 1) * 2))
    # level_3 loop
    loop_level3 = 512
    core_lp3 = loop_level3 // para.core_counts
    core_lp3_tail = loop_level3 % para.core_counts
    gm_offset = (i1 * 16 + i2) * 512
    inner_param = [expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, out_l1]

    # level_3 loop
    with ib.for_range(0, core_lp3, name="core_lp3") as _core_lp3_idx:
        ib = _inner_run(ib, para, inner_param + [_core_lp3_idx, gm_offset])
    if core_lp3_tail > 0:
        with ib.if_scope(para.block.var < core_lp3_tail):
            ib = _inner_run(ib, para, inner_param + [core_lp3, gm_offset])

    return ib


def _inner_run_lev3_1(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, i1, i2, core_lp3_idx = large_data_param
    with ib.new_scope():
        ib.emit(
            tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_32.access_ptr('w'),
                            out_l1.access_ptr(
                                'r',
                                offset=para.block.var * 32 * para.f32_c0 +
                                core_lp3_idx * para.core_counts * 32 * para.f32_c0),
                            0, 1, 32, 0, 0))
    # level_4 loop
    loop_level4 = 4
    offset_l4 = expand_size * para.f16_c0
    repeat_l4 = expand_size // 4 + (1 if expand_size % 4 > 0 else 0)
    with ib.for_range(0, loop_level4) as i5:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "vadds",
                                f32_out.access_ptr('w', offset=i5 * offset_l4),
                                f32_32.access_ptr(
                                    'r',
                                    offset=i5 * 4 * para.f16_c0), tvm.const(0.0, dtype="float32"),
                                repeat_l4, 1, 1, 8, 0))
    offset_4 = expand_size * para.f16_c0
    expand_4 = 4 * expand_size * 2
    with ib.new_scope():
        ib.emit(
            tvm.call_extern("float32", "copy_ubuf_to_gm",
                            para.outputs.access_ptr(
                                'w',
                                offset=((i1 * 16 + i2) * 512 + para.block.var * 4) * offset_4 +
                                core_lp3_idx * para.core_counts * 4 * offset_4),
                            f32_out.access_ptr('r'), 0, 1, expand_4, 0, 0))

    return ib


def _large_data_loop_level1_expand_size_16(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, \
            l1_half, in_l1, out_l1, f32_half, loop_level1, tail_level1, i1, i2 = large_data_param

    # No.2 situation : 1. input f32 dtype : large data : 512>=H*W>16
    # level_3 loop
    loop_level3 = 4
    with ib.for_range(0, loop_level3) as i3:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                out_l1.access_ptr('w', offset=i3 * para.f16_c0),
                                f32_in.access_ptr('r'), 0, 512, 2, 0, (4 - 1) * 2))
    # level_3 loop
    loop_level3 = 512 // 4
    core_lp3 = loop_level3 // para.core_counts
    core_lp3_tail = loop_level3 % para.core_counts

    # level_3 loop
    with ib.for_range(0, core_lp3, name="core_lp3") as _core_lp3_idx:
        ib = _inner_run_lev3_1(ib, para, large_data_param + [_core_lp3_idx])
    if core_lp3_tail > 0:
        with ib.if_scope(para.block.var < core_lp3_tail):
            ib = _inner_run_lev3_1(ib, para, large_data_param + [core_lp3])

    return ib


def _large_data_loop_level1_expand_size_0(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, \
            l1_half, in_l1, out_l1, f32_half, loop_level1, tail_level1, i1, i2 = large_data_param

    # No.2 situation : 1. input f32 dtype : large data  3. H * W <= 16
    with ib.for_range(0, expand_size) as i3:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                out_l1.access_ptr('w', offset=i3 * para.f16_c0),
                                f32_in.access_ptr('r'), 0, 512, 2, 0, (expand_size - 1) * 2))
    loop_level3 = expand_size * 2 // 12
    tail_level3 = expand_size * 2 % 12
    if loop_level3 > 0:
        with ib.for_range(0, loop_level3) as i4:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_out.access_ptr('w'),
                                    out_l1.access_ptr(
                                        'r',
                                        offset=i4 * 512 * 6 * para.f16_c0), 0, 1, 512 * 12, 0, 0))
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(((i1 * 16 + i2) * expand_size) + i4 * 6) * 512 * para.f16_c0),
                                    f32_out.access_ptr('r'), 0, 1, 512 * 12, 0, 0))
    if tail_level3 > 0:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_out.access_ptr('w'),
                                out_l1.access_ptr(
                                    'r',
                                    offset=loop_level3 * 512 * 6 * para.f16_c0), 0, 1, 512 * tail_level3, 0, 0))
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=(((i1 * 16 + i2) * expand_size) + loop_level3 * 6) * 512 * para.f16_c0),
                                f32_out.access_ptr('r'),
                                0, 1, 512 * tail_level3, 0, 0))

    return ib


def _large_data_loop_level1(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, \
        l1_half, in_l1, out_l1, f32_half, loop_level1, tail_level1 = large_data_param
    # level_1 loop
    with ib.for_range(0, loop_level1) as i1:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern(
                    "float32", "copy_gm_to_cbuf",
                    in_l1.access_ptr('w'),
                    para.inputs.access_ptr(
                        'r', offset=i1 * f32_half * para.f16_c0),
                    para.sid, 1, l1_half, 0, 0, para.pad_mode))
        # level_2 loop
        loop_level2 = 16
        with ib.for_range(0, loop_level2) as i2:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern(
                        "float32", "copy_cbuf_to_ubuf",
                        f32_in.access_ptr('w'),
                        in_l1.access_ptr(
                            'r', offset=i2 * 512 * para.f16_c0), 0,
                        1, 1024, 0, 0))
            # Note:
            #   1. H * W > 512 is common
            #   2. 512 >= H * W > 16
            #   3. H * W <= 16 this happen really rare

            # No.2 situation : 1. input f32 dtype : large data : 1. H * W > 512
            if expand_size > 512:
                ib = _large_data_loop_level1_expand_size_512(
                    ib, para, large_data_param + [i1, i2])
            elif expand_size > 16:
                ib = _large_data_loop_level1_expand_size_16(
                    ib, para, large_data_param + [i1, i2])
            else:
                ib = _large_data_loop_level1_expand_size_0(
                    ib, para, large_data_param + [i1, i2])

    return ib


def _large_data_tail_level1_loop_l1_expand_size_512(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, loop_l1, tail_l1, i1 = tail_level_param

    # level_2 loop
    loop_l2 = 4
    with ib.for_range(0, loop_l2) as i2:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                out_l1.access_ptr('w', offset=i2 * para.f16_c0),
                                f32_in.access_ptr('r'), 0, 512, 2, 0, (4 - 1) * 2))
    # level_2 loop
    loop_l2 = 512
    core_lp2 = loop_l2 // para.core_counts
    core_lp2_tail = loop_l2 % para.core_counts
    gm_offset = (loop_level1 * 16 + i1) * 512
    inner_param = [expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, out_l1]

    with ib.for_range(0, core_lp2, name="core_lp2") as _core_lp2_idx:
        ib = _inner_run(ib, para, inner_param + [_core_lp2_idx, gm_offset])
    if core_lp2_tail > 0:
        with ib.if_scope(para.block.var < core_lp2_tail):
            ib = _inner_run(ib, para, inner_param + [core_lp2, gm_offset])

    return ib


def _large_data_tail_level1_loop_l1_expand_size_16(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, loop_l1, tail_l1, i1 = tail_level_param

    # No.2 situation : 1. input f32 dtype : large data : 2. 512>=H*W>16
    # level_2 loop
    loop_l2 = 4
    with ib.for_range(0, loop_l2) as i2:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                out_l1.access_ptr('w', offset=i2 * para.f16_c0),
                                f32_in.access_ptr('r'), 0, 512, 2, 0, (4 - 1) * 2))
    # level_2 loop
    loop_l2 = 512 // 4
    with ib.for_range(0, loop_l2) as i3:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_32.access_ptr('w'),
                                out_l1.access_ptr('r', offset=i3 * 32 * para.f32_c0), 0, 1, 32, 0, 0))
        # level_3 loop
        loop_l3 = 4
        offset_l3 = expand_size * para.f16_c0
        repeat_l3 = expand_size // 4 + (1 if expand_size % 4 > 0 else 0)
        with ib.for_range(0, loop_l3) as i4:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "vadds", f32_out.access_ptr('w', offset=i4 * offset_l3),
                                    f32_32.access_ptr(
                                        'r',
                                        offset=i4 * 4 * para.f16_c0), tvm.const(0.0, dtype="float32"),
                                    repeat_l3, 1, 1, 8, 0))
        offset_4 = expand_size * para.f16_c0
        expand_4 = 4 * expand_size * 2
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=((loop_level1 * 16 + i1) * 512 + i3 * 4) * offset_4),
                                    f32_out.access_ptr('r'), 0, 1, expand_4, 0, 0))

    return ib


def _large_data_tail_level1_loop_l1_expand_size_0(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, loop_l1, tail_l1, i1 = tail_level_param

    # No.2 situation : 1. input f32 dtype : large data  3. H * W <= 16
    with ib.for_range(0, expand_size) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                    out_l1.access_ptr('w', offset=i2 * para.f16_c0),
                                    f32_in.access_ptr('r'), 0, 512, 2, 0, (expand_size - 1) * 2))
    loop_l2 = expand_size * 2 // 12
    tail_l2 = expand_size * 2 % 12
    if loop_l2 > 0:
        with ib.for_range(0, loop_l2) as i3:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_out.access_ptr('w'),
                                    out_l1.access_ptr('r', offset=i3 * 512 * 6 * para.f16_c0),
                                    0, 1, 512 * 12, 0, 0))
            with ib.new_scope():
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=(((loop_level1 * 16 + i1) * expand_size) + i3 * 6)
                                            * 512 * para.f16_c0),
                                        f32_out.access_ptr('r'), 0, 1, 512 * 12, 0, 0))
    if tail_l2 > 0:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_out.access_ptr('w'),
                                out_l1.access_ptr(
                                    'r',
                                    offset=loop_l2 * 512 * 6 * para.f16_c0), 0, 1, 512 * tail_l2, 0, 0))
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=(((loop_level1 * 16 + i1) * expand_size) + loop_l2 * 6)
                                    * 512 * para.f16_c0),
                                f32_out.access_ptr('r'), 0, 1, 512 * tail_l2, 0, 0))

    return ib


def _large_data_tail_level1_loop_l1(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, loop_l1, tail_l1 = tail_level_param
    with ib.for_range(0, loop_l1) as i1:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern(
                    "float32", "copy_cbuf_to_ubuf",
                    f32_in.access_ptr('w'),
                    in_l1.access_ptr('r', offset=i1 * 512 * para.f16_c0),
                    0, 1, 1024, 0, 0))
        # Note:
        #   1. H * W >= 512 is common
        #   2. 512 > H * W > 16
        #   3. H * W <= 16 this happen really rare

        # No.2 situation : 1. input f32 dtype : large data : 1. H * W >= 512
        if expand_size > 512:
            ib = _large_data_tail_level1_loop_l1_expand_size_512(
                ib, para, tail_level_param + [i1])
        elif expand_size > 16:
            ib = _large_data_tail_level1_loop_l1_expand_size_16(
                ib, para, tail_level_param + [i1])
        else:
            ib = _large_data_tail_level1_loop_l1_expand_size_0(
                ib, para, tail_level_param + [i1])

    return ib


def _large_data_tail_level1_tail_l1_expand_size_512_loop_tail_2(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, loop_l1, tail_l1, i2 = tail_level_param
    loop_le2 = expand_size // 2048
    tail_le2 = expand_size % 2048
    offset_2 = expand_size * para.f16_c0
    if loop_le2 > 0:
        with ib.for_range(0, loop_le2) as i3:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=((loop_level1 * 16 + loop_l1) * 512 + i2) * offset_2 +
                                        i3 * 2048 * para.f16_c0),
                                    f32_out.access_ptr('r'), 0, 1, 2048 * 2, 0, 0))
    if tail_le2 > 0:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=((loop_level1 * 16 + loop_l1) * 512 + i2) * offset_2 +
                                    loop_le2 * 2048 * para.f16_c0),
                                f32_out.access_ptr('r'), 0, 1, tail_le2 * 2, 0, 0))

    return ib


def _large_data_tail_level1_tail_l1_expand_size_512(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, loop_l1, tail_l1 = tail_level_param
    # level_1 loop
    loop_le1 = 4
    with ib.for_range(0, loop_le1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                    out_l1.access_ptr('w', offset=i1 * para.f16_c0),
                                    f32_in.access_ptr('r'), 0, tail_l1, 2, 0, (4 - 1) * 2))
    # level_1 loop
    loop_le1 = tail_l1
    with ib.for_range(0, loop_le1) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_8.access_ptr('w'),
                                    out_l1.access_ptr('r', offset=i2 * 4 * para.f16_c0), 0, 1, 8, 0, 0))
        repeat_128 = 512 // 4
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "vadds", f32_out.access_ptr('w'), f32_8.access_ptr('r'),
                                    tvm.const(0.0, dtype="float32"), repeat_128, 1, 1, 8, 0))
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_ubuf",
                                    f32_out.access_ptr('w', offset=512 * para.f16_c0),
                                    f32_out.access_ptr('r'), 0, 1, 1024, 0, 0))
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_ubuf",
                                    f32_out.access_ptr('w', offset=1024 * para.f16_c0),
                                    f32_out.access_ptr('r'), 0, 1, 2048, 0, 0))

        loop_tail_para = tail_level_param + [i2]
        ib = _large_data_tail_level1_tail_l1_expand_size_512_loop_tail_2(ib, para, loop_tail_para)

    return ib


def _large_data_tail_level1_tail_l1_expand_size_16_loop_le1(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, loop_l1, tail_l1, loop_le1, tail_le1 = tail_level_param
    with ib.for_range(0, loop_le1) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_32.access_ptr('w'),
                                    out_l1.access_ptr('r', offset=i2 * 32 * para.f32_c0), 0, 1, 32, 0, 0))
        # level_2 loop
        loop_le2 = 4
        offset_le2 = expand_size * para.f16_c0
        repeat_le2 = expand_size // 4 + (1 if expand_size % 4 > 0 else 0)
        with ib.for_range(0, loop_le2) as i3:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float32", "vadds", f32_out.access_ptr('w', offset=i3 * offset_le2),
                                        f32_32.access_ptr('r', offset=i3 * 4 * para.f16_c0),
                                        tvm.const(0.0, dtype="float32"), repeat_le2, 1, 1, 8, 0))
        offset_4 = expand_size * para.f16_c0
        expand_4 = 4 * expand_size * 2
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=((loop_level1 * 16 + loop_l1) * 512 + i2 * 4) * offset_4),
                                    f32_out.access_ptr('r'), 0, 1, expand_4, 0, 0))

    return ib


def _large_data_tail_level1_tail_l1_expand_size_16_tail_le1(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, loop_l1, tail_l1, loop_le1, tail_le1 = tail_level_param
    with ib.new_scope():
        ib.emit(tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_32.access_ptr('w'),
                                out_l1.access_ptr('r', offset=loop_le1 * 32 * para.f32_c0), 0, 1, tail_le1 * 8, 0, 0))
    # level_2 loop
    loop_le2 = tail_le1
    offset_le2 = expand_size * para.f16_c0
    repeat_le2 = expand_size // 4 + (1 if expand_size % 4 > 0 else 0)
    with ib.for_range(0, loop_le2) as i4:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "vadds", f32_out.access_ptr('w', offset=i4 * offset_le2),
                                    f32_32.access_ptr('r', offset=i4 * 4 * para.f16_c0),
                                    tvm.const(0.0, dtype="float32"), repeat_le2, 1, 1, 8, 0))
    offset_4 = expand_size * para.f16_c0
    expand_4 = tail_le1 * expand_size * 2
    with ib.new_scope():
        ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=((loop_level1 * 16 + loop_l1) * 512 + loop_le1 * 4) * offset_4),
                                f32_out.access_ptr('r'), 0, 1, expand_4, 0, 0))

    return ib


def _large_data_tail_level1_tail_l1_expand_size_16(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
            f32_half, loop_level1, tail_level1, loop_l1, tail_l1 = tail_level_param

    # No.2 situation : 1. input f32 dtype : large data : 2. 512>=H*W>16
    # level_1 loop
    loop_le1 = 4
    with ib.for_range(0, loop_le1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                    out_l1.access_ptr('w', offset=i1 * para.f16_c0),
                                    f32_in.access_ptr('r'), 0, tail_l1, 2, 0, (4 - 1) * 2))
    # level_1 loop
    loop_le1 = tail_l1 // 4
    tail_le1 = tail_l1 % 4
    loop_tail_para = tail_level_param + [loop_le1, tail_le1]
    if loop_le1 > 0:
        ib = _large_data_tail_level1_tail_l1_expand_size_16_loop_le1(ib, para, loop_tail_para)
    if tail_le1 > 0:
        ib = _large_data_tail_level1_tail_l1_expand_size_16_tail_le1(ib, para, loop_tail_para)

    return ib


def _large_data_tail_level1_tail_l1_expand_size_0(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
            f32_half, loop_level1, tail_level1, loop_l1, tail_l1 = tail_level_param

    # No.2 situation : 1. input f32 dtype : large data  3. H * W <= 16
    with ib.for_range(0, expand_size) as i1:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                out_l1.access_ptr('w', offset=i1 * para.f16_c0),
                                f32_in.access_ptr('r'), 0, tail_l1, 2, 0, (expand_size - 1) * 2))
    loop_le2 = tail_l1 * expand_size * 2 // (12 * 512)
    tail_le2 = tail_l1 * expand_size * 2 % (12 * 512)
    if loop_le2 > 0:
        with ib.for_range(0, loop_le2) as i2:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_out.access_ptr('w'),
                                    out_l1.access_ptr('r', offset=i2 * 512 * 6 * para.f16_c0), 0, 1, 512 * 12, 0, 0))
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(((loop_level1 * 16 + loop_l1) * expand_size) + i2 * 6)
                                        * 512 * para.f16_c0),
                                    f32_out.access_ptr('r'), 0, 1, 512 * 12, 0, 0))
    if tail_le2 > 0:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_out.access_ptr('w'),
                                out_l1.access_ptr('r', offset=loop_le2 * 512 * 6 * para.f16_c0),
                                0, 1, tail_le2, 0, 0))
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=(((loop_level1 * 16 + loop_l1) * expand_size) +
                                            loop_le2 * 6) * 512 * para.f16_c0),
                                f32_out.access_ptr('r'), 0, 1, tail_le2, 0, 0))

    return ib


def _large_data_tail_level1_tail_l1(ib, para, tail_level_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, l1_half, in_l1, out_l1, \
        f32_half, loop_level1, tail_level1, loop_l1, tail_l1 = tail_level_param
    if tail_level1 > 512:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern(
                    "float32", "copy_cbuf_to_ubuf",
                    f32_in.access_ptr('w'),
                    in_l1.access_ptr(
                        'r', offset=loop_l1 * 512 * para.f16_c0),
                    0, 1, tail_l1 * 2, 0, 0))
    # Note:
    #   1. H * W >= 512 is common
    #   2. 512 > H * W > 16
    #   3. H * W <= 16 this happen really rare

    # No.2 situation : 1. input f32 dtype : large data : 1. H * W >= 512
    if expand_size > 512:
        ib = _large_data_tail_level1_tail_l1_expand_size_512(ib, para, tail_level_param)
    elif expand_size > 16:
        ib = _large_data_tail_level1_tail_l1_expand_size_16(ib, para, tail_level_param)
    else:
        ib = _large_data_tail_level1_tail_l1_expand_size_0(ib, para, tail_level_param)

    return ib


def _large_data_tail_level1(ib, para, large_data_param):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, \
        l1_half, in_l1, out_l1, f32_half, loop_level1, tail_level1 = large_data_param
    if tail_level1 > 512:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern(
                    "float32", "copy_gm_to_cbuf",
                    in_l1.access_ptr('w'),
                    para.inputs.access_ptr(
                        'r',
                        offset=loop_level1 * f32_half * para.f16_c0),
                    para.sid, 1, tail_level1 * 2, 0, 0, para.pad_mode))
    else:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern(
                    "float32", "copy_gm_to_ubuf",
                    f32_in.access_ptr('w'),
                    para.inputs.access_ptr(
                        'r',
                        offset=loop_level1 * f32_half * para.f16_c0),
                    0, 1, tail_level1 * 2, 0, 0))
    # level_1 loop
    loop_l1 = tail_level1 // 512
    tail_l1 = tail_level1 % 512
    tail_level_param = large_data_param + [loop_l1, tail_l1]
    # No.2 situation : 1. input f32 dtype : large data : tail_level1>0 : loop_l1>0
    if loop_l1 > 0:
        ib = _large_data_tail_level1_loop_l1(ib, para, tail_level_param)
    # No.2 situation : 1. input f32 dtype : large data : tail_level1>0 : tail_l1>0
    if tail_l1 > 0:
        ib = _large_data_tail_level1_tail_l1(ib, para, tail_level_param)

    return ib


def _large_data(ib, para, para_list):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32 = para_list
    l1_half = 512 * 32
    in_l1 = apply_store_buffer(
        ib,
        "float32", [l1_half * para.f32_c0],
        name="in_l1",
        scope=param.scope_cbuf)
    out_l1 = apply_store_buffer(
        ib,
        "float32", [l1_half * para.f32_c0],
        name="out_l1",
        scope=param.scope_cbuf)
    f32_half = l1_half // 2
    loop_level1 = expand_loop // f32_half
    tail_level1 = expand_loop % f32_half

    large_data_param = para_list + [l1_half, in_l1, out_l1, f32_half, loop_level1, tail_level1]
    # No.2 situation : 1. input f32 dtype : large data : loop_level1 > 0
    if loop_level1 > 0:
        ib = _large_data_loop_level1(ib, para, large_data_param)
    # No.2 situation : 1. input f32 dtype : large data : tail_level1 > 0
    if tail_level1 > 0:
        ib = _large_data_tail_level1(ib, para, large_data_param)

    return ib


def _inner_run(ib, para, para_list):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, out_l1, core_lp1_idx,\
        gm_offset = para_list
    with ib.new_scope():
        ib.emit(
            tvm.call_extern("float32", "copy_cbuf_to_ubuf",
                            f32_8.access_ptr('w'),
                            out_l1.access_ptr(
                                'r',
                                offset=para.block.var * 4 * para.f16_c0 +
                                       core_lp1_idx * para.core_counts * 4 * para.f16_c0),
                            0, 1, 8, 0, 0))
    repeat_128 = 512 // 4
    with ib.new_scope():
        ib.emit(tvm.call_extern("float32", "vadds", f32_out.access_ptr('w'), f32_8.access_ptr('r'),
                                tvm.const(0.0, dtype="float32"), repeat_128, 1, 1, 8, 0))
    with ib.new_scope():
        ib.emit(tvm.call_extern("float32", "copy_ubuf_to_ubuf", f32_out.access_ptr('w', offset=512 * para.f16_c0),
                                f32_out.access_ptr('r'), 0, 1, 1024, 0, 0))
    with ib.new_scope():
        ib.emit(tvm.call_extern("float32", "copy_ubuf_to_ubuf", f32_out.access_ptr('w', offset=1024 * para.f16_c0),
                                f32_out.access_ptr('r'), 0, 1, 2048, 0, 0))
    loop_lev2 = expand_size // 2048
    tail_lev2 = expand_size % 2048
    offset_2 = expand_size * para.f16_c0
    if loop_lev2 > 0:
        with ib.for_range(0, loop_lev2) as i3:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=(gm_offset + para.block.var) * offset_2 + i3 * 2048 * para.f16_c0 +
                                               core_lp1_idx * para.core_counts * offset_2),
                                    f32_out.access_ptr('r'), 0, 1, 2048 * 2, 0, 0))
    if tail_lev2 > 0:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=(gm_offset + para.block.var) * offset_2 + loop_lev2 * 2048 * para.f16_c0 +
                                           core_lp1_idx * para.core_counts * offset_2),
                                f32_out.access_ptr('r'), 0, 1, tail_lev2 * 2, 0, 0))

    return ib


def _small_data_expand_size_512(ib, para, para_list):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, out_l1 = para_list
    # level_1 loop
    loop_lev1 = 4
    with ib.for_range(0, loop_lev1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_cbuf", out_l1.access_ptr('w', offset=i1 * para.f16_c0),
                                    f32_in.access_ptr('r'), 0, expand_loop, 2, 0, (4 - 1) * 2))
    # level_1 loop
    core_lp1 = expand_loop // para.core_counts
    core_lp1_tail = expand_loop % para.core_counts
    gm_offset = 0

    # level_1 loop
    with ib.for_range(0, core_lp1, name="core_lp1") as _core_lp1_idx:
        ib = _inner_run(ib, para, para_list + [_core_lp1_idx, gm_offset])
    if core_lp1_tail > 0:
        with ib.if_scope(para.block.var < core_lp1_tail):
            ib = _inner_run(ib, para, para_list + [core_lp1, gm_offset])

    return ib


def _small_data_expand_size_16_loop_lev1(ib, para, para_list):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, out_l1,\
        loop_lev1, tail_lev1 = para_list
    with ib.for_range(0, loop_lev1) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_32.access_ptr('w'),
                                    out_l1.access_ptr('r', offset=i2 * 32 * para.f32_c0), 0, 1, 32, 0, 0))
        # level_2 loop
        loop_lev2 = 4
        offset_lev2 = expand_size * para.f16_c0
        repeat_lev2 = expand_size // 4 + (1 if expand_size % 4 > 0 else 0)
        with ib.for_range(0, loop_lev2) as i3:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float32", "vadds", f32_out.access_ptr('w', offset=i3 * offset_lev2),
                                        f32_32.access_ptr('r', offset=i3 * 4 * para.f16_c0),
                                        tvm.const(0.0, dtype="float32"), repeat_lev2, 1, 1, 8, 0))
        offset_4 = expand_size * para.f16_c0
        expand_4 = 4 * expand_size * 2
        with ib.new_scope():
            ib.emit(
                tvm.call_extern(para.dtype, "copy_ubuf_to_gm", para.outputs.access_ptr('w', offset=i2 * 4 * offset_4),
                                f32_out.access_ptr('r'), 0, 1, expand_4, 0, 0))

    return ib


def _small_data_expand_size_16_tail_lev1(ib, para, para_list):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, out_l1,\
        loop_lev1, tail_lev1 = para_list
    with ib.new_scope():
        ib.emit(tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_32.access_ptr('w'),
                                out_l1.access_ptr('r', offset=loop_lev1 * 32 * para.f32_c0),
                                0, 1, tail_lev1 * 8, 0, 0))
    # level_2 loop
    loop_lev2 = tail_lev1
    offset_lev2 = expand_size * para.f16_c0
    repeat_lev2 = expand_size // 4 + (1 if expand_size % 4 > 0 else 0)
    with ib.for_range(0, loop_lev2) as i4:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "vadds", f32_out.access_ptr('w', offset=i4 * offset_lev2),
                                    f32_32.access_ptr('r', offset=i4 * 4 * para.f16_c0),
                                    tvm.const(0.0, dtype="float32"), repeat_lev2, 1, 1, 8, 0))
    offset_4 = expand_size * para.f16_c0
    expand_4 = tail_lev1 * expand_size * 2
    with ib.new_scope():
        ib.emit(tvm.call_extern(para.dtype, "copy_ubuf_to_gm",
                                para.outputs.access_ptr('w', offset=loop_lev1 * 4 * offset_4),
                                f32_out.access_ptr('r'), 0, 1, expand_4, 0, 0))

    return ib


def _small_data_expand_size_16(ib, para, para_list):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, out_l1 = para_list
    # No.2 situation : 1. input f32 dtype : large data : 2. 512>=H*W>16
    # level_1 loop
    loop_lev1 = 4
    with ib.for_range(0, loop_lev1) as i1:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                    out_l1.access_ptr('w', offset=i1 * para.f16_c0),
                                    f32_in.access_ptr('r'), 0, expand_loop, 2, 0, (4 - 1) * 2))
    # level_1 loop
    loop_lev1 = expand_loop // 4
    tail_lev1 = expand_loop % 4
    loop_tail_para = para_list + [loop_lev1, tail_lev1]
    if loop_lev1 > 0:
        ib = _small_data_expand_size_16_loop_lev1(ib, para, loop_tail_para)
    if tail_lev1 > 0:
        ib = _small_data_expand_size_16_tail_lev1(ib, para, loop_tail_para)

    return ib


def _small_data_expand_size_0(ib, para, para_list):
    expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32, out_l1 = para_list
    # No.2 situation : 1. input f32 dtype : large data  3. H * W <= 16
    with ib.for_range(0, expand_size) as i1:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_cbuf",
                                out_l1.access_ptr('w', offset=i1 * para.f16_c0),
                                f32_in.access_ptr('r'), 0, expand_loop, 2, 0, (expand_size - 1) * 2))
    loop_lev2 = expand_loop * expand_size * 2 // (12 * 512)
    tail_lev2 = expand_loop * expand_size * 2 % (12 * 512)
    if loop_lev2 > 0:
        with ib.for_range(0, loop_lev2) as i2:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_out.access_ptr('w'),
                                        out_l1.access_ptr('r', offset=i2 * 512 * 6 * para.f16_c0),
                                        0, 1, 512 * 12, 0, 0))
            with ib.new_scope():
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr('w', offset=i2 * 6 * 512 * para.f16_c0),
                                        f32_out.access_ptr('r'), 0, 1, 512 * 12, 0, 0))
    if tail_lev2 > 0:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_cbuf_to_ubuf", f32_out.access_ptr('w'),
                                    out_l1.access_ptr('r', offset=loop_lev2 * 512 * 6 * para.f16_c0),
                                    0, 1, tail_lev2, 0, 0))
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr('w', offset=loop_lev2 * 6 * 512 * para.f16_c0),
                                    f32_out.access_ptr('r'), 0, 1, tail_lev2, 0, 0))

    return ib


def compute_with_in_hw_eq_one_fp32(ib, para):
    # No.2 situation : 2. input f32 dtype
    expand_loop = reduce(lambda x, y: x * y, para.size_in) // para.f16_c0
    actual_loop = min(expand_loop, 512)
    free_space = 512 * 12
    expand_size = para.h_out * para.w_out
    ib.scope_attr(para.block, "thread_extent", para.core_counts)

    f32_in = apply_store_buffer(
        ib, "float32", [1024 * para.f32_c0], name="f32_in")
    f32_out = apply_store_buffer(
        ib, "float32", [free_space * para.f32_c0], name="f32_out")

    f32_8 = None
    f32_32 = None
    if expand_size > 512:
        f32_8 = apply_store_buffer(ib, "float32", [8 * para.f32_c0], name="f32_8")
    elif expand_size > 16:
        f32_32 = apply_store_buffer(ib, "float32", [32 * para.f32_c0], name="f32_32")
    # Note:
    #   1. input data larger than 512 should use L1 optimize
    #   2. input small data do not need L1

    if expand_size > 16:
        ib.emit(tvm.call_extern("uint64", "set_vector_mask",
                                tvm.const(0, dtype="uint64"), tvm.const((1 << 64) - 1, dtype="uint64")))

    para_list = [expand_loop, actual_loop, free_space, expand_size, f32_in, f32_out, f32_8, f32_32]
    # No.2 situation : 1. input f32 dtype : large data
    if expand_loop > 512:
        ib = _large_data(ib, para, para_list)
    else:
        # No.2 situation : 1. input f32 dtype : small data
        l1_half = 512 * 32
        out_l1 = apply_store_buffer(
            ib,
            "float32", [l1_half * para.f32_c0],
            name="out_l1",
            scope=param.scope_cbuf)
        with ib.new_scope():
            ib.emit(tvm.call_extern("float32", "copy_gm_to_ubuf", f32_in.access_ptr('w'),
                                    para.inputs.access_ptr('r'), 0, 1, expand_loop * 2, 0, 0))
        # Note:
        #   1. H * W >= 512 is common
        #   2. 512 > H * W > 16
        #   3. H * W <= 16 this happen really rare

        para_list = para_list + [out_l1]
        # No.2 situation : 1. input f32 dtype : large data : 1. H * W >= 512
        if expand_size > 512:
            ib = _small_data_expand_size_512(ib, para, para_list)
        elif expand_size > 16:
            ib = _small_data_expand_size_16(ib, para, para_list)
        else:
            ib = _small_data_expand_size_0(ib, para, para_list)

    return ib
