# -*- coding:utf-8 -*-
from functools import reduce
from te import tvm
from te.platform import cce_params as param
from te.lang.cce.te_compute import irbuilder_api as kernel_api
from .interp_common import LargeDataFp16Param, LargeDataFp32Param
from .interp_common import apply_store_buffer


def _fp16_large_data_loop_level1(ib, para, large_data_param, loop_level1):
    # level_1 loop
    with ib.for_range(0, loop_level1) as i1:
        # Note:
        #   1. pick gap out of range need copy many times
        #   2. pick gap in range need only multi bursts

        if large_data_param.f16_stride > large_data_param.gap_limit:
            # level_2 loop
            loop_level2 = large_data_param.l1_half
            offset_2 = large_data_param.l1_half * large_data_param.reduce_size * para.f16_c0
            with ib.for_range(0, loop_level2) as i3:
                with ib.new_scope():
                    ib.emit(tvm.call_extern("float16", "copy_gm_to_cbuf",
                                            large_data_param.in_l1.access_ptr('w', offset=i3 * para.f16_c0),
                                            para.inputs.access_ptr(
                                                'r',
                                                offset=i1 * offset_2 + i3 * large_data_param.reduce_size * para.f16_c0),
                                            para.sid, 1, 1, 0, 0, para.pad_mode))
        else:
            # level_2 loop
            loop_level2 = 4
            offset_2 = large_data_param.l1_half * large_data_param.reduce_size * para.f16_c0
            offset_3 = large_data_param.burst_limit * large_data_param.reduce_size * para.f16_c0
            with ib.for_range(0, loop_level2) as i3:
                with ib.new_scope():
                    ib.emit(tvm.call_extern("float16", "copy_gm_to_cbuf", large_data_param.in_l1.access_ptr('w'),
                                            para.inputs.access_ptr('r', offset=i1 * offset_2 + i3 * offset_3), para.sid,
                                            large_data_param.burst_limit, 1, large_data_param.f16_stride, 0,
                                            para.pad_mode))
            offset_4 = large_data_param.burst_limit * para.f16_c0
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "copy_gm_to_cbuf",
                                        large_data_param.in_l1.access_ptr('w', offset=loop_level2 * offset_4),
                                        para.inputs.access_ptr('r', offset=i1 * offset_2 + 4 * offset_3), para.sid, 4,
                                        1, large_data_param.f16_stride, 0, para.pad_mode))

        # level_2 loop
        loop_level2 = 4
        with ib.for_range(0, loop_level2) as i2:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", large_data_param.f16_out.access_ptr('w'),
                                        large_data_param.in_l1.access_ptr(
                                            'r',
                                            offset=i2 * large_data_param.f16_size * para.f16_c0),
                                        0, 1, large_data_param.f16_size, 0, 0))

            offset_4 = large_data_param.l1_half * para.f16_c0
            with ib.new_scope():
                ip_addr = [[large_data_param.out_ub_f32, 0], [large_data_param.f16_out, 0]]
                kernel_api.kernel_cast_to_fuc(ib, ip_addr, [large_data_param.f16_size * 16, 8 * 8], "vconv_f162f32")
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=i1 * offset_4 + i2 * large_data_param.f16_size * para.f16_c0),
                                        large_data_param.out_ub_f32.access_ptr('r'), 0, 1,
                                        large_data_param.f16_size * 2, 0, 0))

    return ib


def _fp16_large_data_tail_level1_loop_l1(ib, para, large_data_param, loop_level1, loop_l1):
    with ib.for_range(0, loop_l1) as i2:
        with ib.new_scope():
            ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", large_data_param.f16_out.access_ptr('w'),
                                    large_data_param.in_l1.access_ptr(
                                        'r',
                                        offset=i2 * large_data_param.f16_size * para.f16_c0),
                                    0, 1, large_data_param.f16_size, 0, 0))
        offset_3 = large_data_param.l1_half * para.f16_c0
        with ib.new_scope():
            ip_addr = [[large_data_param.out_ub_f32, 0], [large_data_param.f16_out, 0]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [large_data_param.f16_size * 16, 8 * 8], "vconv_f162f32")
            ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                    para.outputs.access_ptr(
                                        'w',
                                        offset=loop_level1 * offset_3 +
                                        i2 * large_data_param.f16_size * para.f16_c0),
                                    large_data_param.out_ub_f32.access_ptr('r'),
                                    0, 1, large_data_param.f16_size * 2, 0, 0))

    return ib


def _fp16_large_data_tail_level1_tail_l1(ib, para, large_data_param, loop_level1, loop_l1, tail_l1):
    with ib.new_scope():
        ib.emit(tvm.call_extern("float16", "copy_cbuf_to_ubuf", large_data_param.f16_out.access_ptr('w'),
                                large_data_param.in_l1.access_ptr(
                                    'r',
                                    offset=loop_l1 * large_data_param.f16_size * para.f16_c0),
                                0, 1, tail_l1, 0, 0))
    offset_4 = large_data_param.l1_half * para.f16_c0
    with ib.new_scope():
        ip_addr = [[large_data_param.out_ub_f32, 0], [large_data_param.f16_out, 0]]
        kernel_api.kernel_cast_to_fuc(ib, ip_addr, [tail_l1 * 16, 8 * 8], "vconv_f162f32")
        ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=loop_level1 * offset_4 + loop_l1 * large_data_param.f16_size * para.f16_c0),
                                large_data_param.out_ub_f32.access_ptr('r'), 0, 1, tail_l1 * 2, 0, 0))

    return ib


def _fp16_large_data_tail_level1(ib, para, large_data_param, loop_level1, tail_level1):
    # Note:
    #   1. pick gap out of range need copy many times
    #   2. pick gap in range need only one bursts
    if large_data_param.f16_stride > large_data_param.gap_limit:
        # level_1 loop
        offset_1 = large_data_param.l1_half * large_data_param.reduce_size * para.f16_c0
        with ib.for_range(0, tail_level1) as i1:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "copy_gm_to_cbuf",
                                        large_data_param.in_l1.access_ptr('w', offset=i1 * para.f16_c0),
                                        para.inputs.access_ptr(
                                            'r',
                                            offset=loop_level1 * offset_1 + i1 * large_data_param.reduce_size *
                                            para.f16_c0),
                                        para.sid, 1, 1, 0, 0, para.pad_mode))
    else:
        # level_1 loop
        loop_l1 = tail_level1 // large_data_param.burst_limit
        tail_l1 = tail_level1 % large_data_param.burst_limit
        offset_2 = large_data_param.l1_half * large_data_param.reduce_size * para.f16_c0
        offset_3 = large_data_param.burst_limit * large_data_param.reduce_size * para.f16_c0
        if loop_l1 > 0:
            with ib.for_range(0, loop_l1) as i1:
                with ib.new_scope():
                    ib.emit(tvm.call_extern("float16", "copy_gm_to_cbuf", large_data_param.in_l1.access_ptr('w'),
                                            para.inputs.access_ptr('r', offset=loop_level1 * offset_2 + i1 * offset_3),
                                            para.sid, large_data_param.burst_limit, 1, large_data_param.f16_stride, 0,
                                            para.pad_mode))
        offset_4 = large_data_param.burst_limit * para.f16_c0
        if tail_l1 > 0:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float16", "copy_gm_to_cbuf",
                                        large_data_param.in_l1.access_ptr('w', offset=loop_l1 * offset_4),
                                        para.inputs.access_ptr('r', offset=loop_level1 * offset_2 + loop_l1 * offset_3),
                                        para.sid, tail_l1, 1, large_data_param.f16_stride, 0, para.pad_mode))

    # level_1 loop
    loop_l1 = tail_level1 // large_data_param.f16_size
    tail_l1 = tail_level1 % large_data_param.f16_size
    # No.3 situation : output H/W == (1,1): input f16 dtype: tail_level1>0: loop_l1>0
    if loop_l1 > 0:
        ib = _fp16_large_data_tail_level1_loop_l1(ib, para, large_data_param, loop_level1, loop_l1)

    # No.3 situation : output H/W == (1,1): input f16 dtype: tail_level1>0: tail_l1>0
    if tail_l1 > 0:
        ib = _fp16_large_data_tail_level1_tail_l1(ib, para, large_data_param, loop_level1, loop_l1, tail_l1)

    return ib


def _fp16_large_data(ib, para, large_data_param):
    loop_level1 = large_data_param.expand_loop // large_data_param.l1_half
    tail_level1 = large_data_param.expand_loop % large_data_param.l1_half

    # No.3 situation : output H/W == (1,1): input f16 dtype: large data: loop_level1 > 0
    if loop_level1 > 0:
        ib = _fp16_large_data_loop_level1(ib, para, large_data_param, loop_level1)

    # No.3 situation : output H/W == (1,1): input f16 dtype: large data: tail_level1 > 0
    if tail_level1 > 0:
        ib = _fp16_large_data_tail_level1(ib, para, large_data_param, loop_level1, tail_level1)

    return ib


# No.3 situation : output H/W == (1,1) and dtype==fp16
def compute_with_out_hw_eq_one_fp16(ib, para):
    # Note:
    #   1. pick data from out directly
    #   2. large output data use L1 optimize
    # No.3 situation : output H/W == (1,1) : input f16 dtype
    expand_loop = reduce(lambda x, y: x * y, para.size_out) // para.f16_c0
    gap_limit = (1 << 16) - 1
    f16_stride = para.w_in * para.h_in - 1
    l1_half = 512 * 32
    f16_size = 256 * 8
    reduce_size = para.w_in * para.h_in
    burst_limit = (1 << 12) - 1
    out_ub_f32 = apply_store_buffer(ib, "float32", [f16_size * para.f16_c0], name="out_f32")
    f16_out = apply_store_buffer(ib, "float16", [f16_size * para.f16_c0], name="f16_out")
    in_l1 = apply_store_buffer(ib, "float16", [l1_half * para.f16_c0], name="in_l1", scope=param.scope_cbuf)
    # Note:
    #   1. output data larger than 512 * 8 should use L1 optimize
    #   2. output small data do not need L1

    # No.3 situation : output H/W == (1,1) : input f16 dtype : large data
    if expand_loop > f16_size:
        large_data_param = LargeDataFp16Param(expand_loop, l1_half, f16_stride, f16_size, gap_limit, reduce_size, in_l1,
                                              burst_limit, f16_out, out_ub_f32)
        ib = _fp16_large_data(ib, para, large_data_param)
    else:
        # No.3 situation : output H/W == (1,1) : input f16 dtype : small data

        # Note:
        #   1. pick gap out of range need copy many times
        #   2. pick gap in range need only one bursts

        if f16_stride > gap_limit:
            # level_1 loop
            with ib.for_range(0, expand_loop) as i1:
                with ib.new_scope():
                    ib.emit(
                        tvm.call_extern("float16", "copy_gm_to_ubuf", f16_out.access_ptr('w', offset=i1 * para.f16_c0),
                                        para.inputs.access_ptr('r', offset=i1 * reduce_size * para.f16_c0),
                                        0, 1, 1, 0, 0))
        else:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float16", "copy_gm_to_ubuf", f16_out.access_ptr('w'), para.inputs.access_ptr('r'),
                                    0, expand_loop, 1, f16_stride, 0))
        with ib.new_scope():
            ip_addr = [[out_ub_f32, 0], [f16_out, 0]]
            kernel_api.kernel_cast_to_fuc(ib, ip_addr, [expand_loop * 16, 8 * 8], "vconv_f162f32")
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm", para.outputs.access_ptr('w'), out_ub_f32.access_ptr('r'),
                                0, 1, expand_loop * 2, 0, 0))

    return ib


def _fp32_large_data_loop_level1(ib, para, large_data_param, f32_half, loop_level1):
    # level_1 loop
    with ib.for_range(0, loop_level1) as i1:
        # Note:
        #   1. pick gap out of range need copy many times
        #   2. pick gap in range need only one bursts

        if large_data_param.f32_stride > large_data_param.gap_limit:
            # level_2 loop
            loop_level2 = f32_half
            offset_2 = f32_half * large_data_param.reduce_size * para.f16_c0
            with ib.for_range(0, loop_level2) as i3:
                with ib.new_scope():
                    ib.emit(tvm.call_extern("float32", "copy_gm_to_cbuf",
                                            large_data_param.in_l1.access_ptr('w', offset=i3 * para.f16_c0),
                                            large_data_param.para.inputs.access_ptr(
                                                'r',
                                                offset=i1 * offset_2 + i3 *
                                                large_data_param.reduce_size * para.f16_c0),
                                            para.sid, 1, 2, 0, 0, para.pad_mode))
        else:
            # level_2 loop
            loop_level2 = 2
            offset_2 = f32_half * large_data_param.reduce_size * para.f16_c0
            offset_3 = large_data_param.burst_limit * large_data_param.reduce_size * para.f16_c0
            with ib.for_range(0, loop_level2) as i3:
                with ib.new_scope():
                    ib.emit(tvm.call_extern("float32", "copy_gm_to_cbuf", large_data_param.in_l1.access_ptr('w'),
                                            para.inputs.access_ptr(
                                                'r',
                                                offset=i1 * offset_2 + i3 * offset_3), para.sid,
                                            large_data_param.burst_limit, 2, large_data_param.f32_stride, 0,
                                            para.pad_mode))
            offset_4 = large_data_param.burst_limit * para.f16_c0
            with ib.new_scope():
                ib.emit(tvm.call_extern("float32", "copy_gm_to_cbuf",
                                        large_data_param.in_l1.access_ptr('w', offset=loop_level2 * offset_4),
                                        para.inputs.access_ptr('r', offset=i1 * offset_2 + 2 * offset_3), para.sid, 2,
                                        2, large_data_param.f32_stride, 0, para.pad_mode))

        # level_2 loop
        loop_level2 = 4
        with ib.for_range(0, loop_level2) as i2:
            with ib.new_scope():
                ib.emit(tvm.call_extern("float32", "copy_cbuf_to_ubuf", large_data_param.f32_in.access_ptr('w'),
                                        large_data_param.in_l1.access_ptr(
                                            'r',
                                            offset=i2 * large_data_param.f32_size * para.f32_c0),
                                        0, 1, large_data_param.f32_size, 0, 0))
            offset_4 = f32_half * para.f16_c0
            with ib.new_scope():
                ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                        para.outputs.access_ptr(
                                            'w',
                                            offset=i1 * offset_4 + i2 * large_data_param.f32_size * para.f32_c0),
                                        large_data_param.f32_out.access_ptr('r'),
                                        0, 1, large_data_param.f32_size, 0, 0))

    return ib


def _fp32_large_data_tail_level1_loop_l1(ib, para, large_data_param, f32_half, loop_level1, loop_l1):
    ib.emit(tvm.call_extern("uint64", "set_vector_mask", tvm.const(0, dtype="uint64"),
                            tvm.const((1 << 64) - 1, dtype="uint64")))
    with ib.for_range(0, loop_l1) as i2:
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_cbuf_to_ubuf", large_data_param.f32_out.access_ptr('w'),
                                large_data_param.in_l1.access_ptr(
                                    'r',
                                    offset=i2 * large_data_param.f32_size * para.f32_c0),
                                0, 1, large_data_param.f32_size, 0, 0))
        offset_3 = f32_half * para.f16_c0
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=loop_level1 * offset_3 + i2 * large_data_param.f32_size * para.f32_c0),
                                large_data_param.f32_out.access_ptr('r'), 0, 1, large_data_param.f32_size, 0, 0))

    return ib


def _fp32_large_data_tail_level1_tail_l1(ib, para, large_data_param, f32_half, loop_level1, loop_l1, tail_l1):
    with ib.new_scope():
        ib.emit(tvm.call_extern("float32", "copy_cbuf_to_ubuf", large_data_param.f32_out.access_ptr('w'),
                                large_data_param.in_l1.access_ptr(
                                    'r',
                                    offset=loop_l1 * large_data_param.f32_size * para.f32_c0),
                                0, 1, tail_l1, 0, 0))
    offset_4 = f32_half * para.f16_c0
    with ib.new_scope():
        ib.emit(tvm.call_extern("float32", "copy_ubuf_to_gm",
                                para.outputs.access_ptr(
                                    'w',
                                    offset=loop_level1 * offset_4 + loop_l1 * large_data_param.f32_size * para.f32_c0),
                                large_data_param.f32_out.access_ptr('r'), 0, 1, loop_l1, 0, 0))

    return ib


def _fp32_large_data_tail_level1(ib, para, large_data_param, f32_half, loop_level1, tail_level1):
    # Note:
    #   1. pick gap out of range need copy many times
    #   2. pick gap in range need only one bursts

    if large_data_param.f32_stride > large_data_param.gap_limit:
        # level_1 loop
        offset_1 = f32_half * large_data_param.reduce_size * para.f16_c0
        with ib.for_range(0, tail_level1) as i1:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "copy_gm_to_cbuf",
                                    large_data_param.in_l1.access_ptr('w', offset=i1 * para.f16_c0),
                                    para.inputs.access_ptr(
                                        'r',
                                        offset=loop_level1 * offset_1 +
                                        i1 * large_data_param.reduce_size * para.f16_c0),
                                    para.sid, 1, 2, 0, 0, para.pad_mode))
    else:
        # level_1 loop
        loop_l1 = tail_level1 // large_data_param.burst_limit
        tail_l1 = tail_level1 % large_data_param.burst_limit
        offset_2 = f32_half * large_data_param.reduce_size * para.f16_c0
        offset_3 = large_data_param.burst_limit * large_data_param.reduce_size * para.f16_c0
        if loop_l1 > 0:
            with ib.for_range(0, loop_l1) as i1:
                with ib.new_scope():
                    ib.emit(tvm.call_extern("float32", "copy_gm_to_cbuf", large_data_param.in_l1.access_ptr('w'),
                                            para.inputs.access_ptr('r', offset=loop_level1 * offset_2 + i1 * offset_3),
                                            para.sid, large_data_param.burst_limit, 2, large_data_param.f32_stride, 0,
                                            para.pad_mode))
        if tail_l1 > 0:
            offset_4 = large_data_param.burst_limit * para.f16_c0
            with ib.new_scope():
                ib.emit(tvm.call_extern("float32", "copy_gm_to_cbuf",
                                        large_data_param.in_l1.access_ptr('w', offset=loop_l1 * offset_4),
                                        para.inputs.access_ptr('r', offset=loop_level1 * offset_2 + loop_l1 * offset_3),
                                        para.sid, tail_l1, 2, large_data_param.f32_stride, 0, para.pad_mode))

    # level_1 loop
    loop_l1 = tail_level1 // large_data_param.f32_size
    tail_l1 = tail_level1 % large_data_param.f32_size
    # No.3 situation : output H/W == (1,1): input f32 dtype: tail_level1>0: loop_l1>0
    if loop_l1 > 0:
        ib = _fp32_large_data_tail_level1_loop_l1(ib, para, large_data_param, f32_half, loop_level1, loop_l1)
    # No.3 situation : output H/W == (1,1): input f32 dtype: tail_level1>0: tail_l1>0
    if tail_l1 > 0:
        ib = _fp32_large_data_tail_level1_tail_l1(ib, para, large_data_param, f32_half, loop_level1, loop_l1, tail_l1)

    return ib


def _fp32_large_data(ib, para, large_data_param):
    f32_half = large_data_param.l1_half // 2
    loop_level1 = large_data_param.expand_loop // f32_half
    tail_level1 = large_data_param.expand_loop % f32_half

    # No.3 situation : output H/W == (1,1): input f32 dtype: large data: loop_level1 > 0
    if loop_level1 > 0:
        ib = _fp32_large_data_loop_level1(ib, para, large_data_param, f32_half, loop_level1)

    # No.3 situation : output H/W == (1,1): input f32 dtype: large data: tail_level1 > 0
    if tail_level1 > 0:
        ib = _fp32_large_data_tail_level1(ib, para, large_data_param, f32_half, loop_level1, tail_level1)

    return ib


# No.3 situation : output H/W == (1,1) and dtype==fp32
def compute_with_out_hw_eq_one_fp32(ib, para):
    # No.3 situation : output H/W == (1,1) : input f32 dtype
    expand_loop = reduce(lambda x, y: x * y, para.size_out) // para.f16_c0
    gap_limit = (1 << 16) - 1
    f32_stride = (para.w_in * para.h_in - 1) * 2
    l1_half = 512 * 32
    f32_size = 512 * 8
    reduce_size = para.w_in * para.h_in
    burst_limit = (1 << 12) - 1

    f32_out = apply_store_buffer(ib, "float32", [f32_size * para.f32_c0], name="f32_out")
    in_l1 = apply_store_buffer(ib, "float32", [l1_half * para.f16_c0], name="in_l1", scope=param.scope_cbuf)
    f32_in = apply_store_buffer(ib, "float32", [1024 * para.f32_c0], name="f32_in")
    # Note:
    #   1. output data larger than 512 * 8 should use L1 optimize
    #   2. output small data do not need L1

    # No.3 situation : output H/W == (1,1) : input f32 dtype : large data
    if expand_loop > f32_size:
        large_data_param = LargeDataFp32Param(expand_loop, l1_half, f32_stride, f32_size, gap_limit, reduce_size, in_l1,
                                              burst_limit, f32_out, f32_in)
        ib = _fp32_large_data(ib, para, large_data_param)
    else:
        # No.3 situation : output H/W == (1,1) : input f32 dtype : small data

        # Note:
        #   1. pick gap out of range need copy many times
        #   2. pick gap in range need only one bursts

        if f32_stride > gap_limit:
            # level_1 loop
            with ib.for_range(0, expand_loop) as i1:
                with ib.new_scope():
                    ib.emit(
                        tvm.call_extern("float32", "copy_gm_to_ubuf", f32_out.access_ptr('w', offset=i1 * para.f16_c0),
                                        para.inputs.access_ptr('r', offset=i1 * reduce_size * para.f16_c0),
                                        0, 1, 2, 0, 0))
        else:
            with ib.new_scope():
                ib.emit(
                    tvm.call_extern("float32", "copy_gm_to_ubuf", f32_out.access_ptr('w'), para.inputs.access_ptr('r'),
                                    0, expand_loop, 2, f32_stride, 0))
        with ib.new_scope():
            ib.emit(
                tvm.call_extern("float32", "copy_ubuf_to_gm", para.outputs.access_ptr('w'), f32_out.access_ptr('r'),
                                0, 1, expand_loop * 2, 0, 0))

    return ib


def compute_with_out_hw_eq_one(ib, para):
    if para.dtype == "float16":
        ib = compute_with_out_hw_eq_one_fp16(ib, para)
    else:
        ib = compute_with_out_hw_eq_one_fp32(ib, para)

    return ib
