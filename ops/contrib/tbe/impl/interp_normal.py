# -*- coding:utf-8 -*-
from te import tvm
from te.platform import cce_params as param
from te.lang.cce.te_compute import irbuilder_api as kernel_api
from .interp_common import NormalSituationParam
from .interp_common import ARGS_STR_V_ALL
from .interp_common import apply_reg_buffer, apply_store_buffer


def apply_ub(ib, para):
    # apply ub for x 512
    ub_info = dict()
    l1_info = dict()

    # apply ub for x(h,w) 512  x1(h,w+1) 512
    ub_info["x0_512_x1_512_ub"] = apply_store_buffer(ib, "float32", [512 * 2 * para.f32_c0], name="x_512_x1_512")
    # apply ub for x(h+1,w) 512  x1(h+1,w+1) 512
    ub_info["x2_512_x3_512_ub"] = apply_store_buffer(ib, "float32", [512 * 2 * para.f32_c0], name="x2_512_x3_512")
    ub_info["int32_512_ub"] = apply_store_buffer(ib, "int32", [256 * para.f32_c0], name="int32_512")
    ub_info["int32_512_ub_tmp"] = apply_store_buffer(ib, "int32", [256 * para.f32_c0], name="int32_512_tmp")
    ub_info["int32_512_ub_y"] = apply_store_buffer(ib, "int32", [8 * para.f32_c0], name="int32_512_ub_y")
    ub_info["const_0"] = apply_store_buffer(ib, "float32", [8 * para.f32_c0], name="const_0")

    ub_info["x0_scale_x1_scale_512_ub"] = apply_store_buffer(ib, "float32", [512 * para.f32_c0],
                                                             name="x0_scale_x1_scale_512_ub")
    ub_info["y0_scale_ub"] = apply_store_buffer(ib, "float32", [2 * para.f32_c0], name="y0_scale_ub")

    ub_info["out_f32"] = apply_store_buffer(ib, "float32", [512 * para.f32_c0], name="out_f32")
    l1_info["l1_ypos"] = apply_store_buffer(ib, "int32", [512 * 3 * para.f32_c0], name="l1_ypos",
                                            scope=param.scope_cbuf)
    l1_info["l1_xpos"] = apply_store_buffer(ib, "int32", [512 * 4 * para.f32_c0],  # to support w_out 2048
                                            name="l1_xpos", scope=param.scope_cbuf)
    l1_info["l1_xscale"] = apply_store_buffer(ib, "float32", [2 * 512 * 4 * para.f32_c0],  # to support w_out 2048
                                              name="l1_xscale", scope=param.scope_cbuf)
    l1_info["l1_yscale"] = apply_store_buffer(ib, "float32", [2 * 512 * 3 * para.f32_c0], name="l1_yscale",
                                              scope=param.scope_cbuf)
    l1_info["l1_input"] = apply_store_buffer(ib, "float32", [2 * (para.w_in + 1) * 16], name="l1_input",
                                             scope=param.scope_cbuf)
    ub_info["ub_input"] = apply_store_buffer(ib, "float32", [2 * (para.w_in + 1) * 16], name="ub_input")
    ub_info["f16_512"] = apply_store_buffer(ib, "float16", [512 * 8], name="f16_512")
    ub_info["f16_512_1"] = apply_store_buffer(ib, "float16", [512 * 8], name="f16_512_1")
    # apply ub for x(h,w) 512  x1(h,w+1) 512
    ub_info["x0_512_x1_512_ub_1"] = apply_store_buffer(ib, "float32", [512 * 2 * para.f32_c0], name="x_512_x1_512_1")
    # apply ub for x(h+1,w) 512  x1(h+1,w+1) 512
    ub_info["x2_512_x3_512_ub_1"] = apply_store_buffer(ib, "float32", [512 * 2 * para.f32_c0], name="x2_512_x3_512_1")
    # for unfold
    ub_info["x0_512_x1_512_ub_unfold"] = apply_store_buffer(ib, "float32", [512 * 2 * para.f32_c0],
                                                            name="x0_512_x1_512_unfold")
    ub_info["x2_512_x3_512_ub_unfold"] = apply_store_buffer(ib, "float32", [512 * 2 * para.f32_c0],
                                                            name="x2_512_x3_512_unfold")

    return ib, ub_info, l1_info


def _copy_input_to_l1_fp32(para_list):
    dst, src, _dtype, cp_segment, src_offset, mid_ub, y_index, is_input_in_ub, ib, ub_info, para = para_list
    if is_input_in_ub is not True:
        _addr = [[dst, 0], [src, src_offset]]
        kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8], "copy_gm_to_cbuf")
        _addr = [[dst, cp_segment], [src, src_offset + cp_segment - 16]]
        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8], "copy_gm_to_cbuf")
        with ib.if_scope(y_index == para.size_in[2] - 1):
            _addr = [[dst, cp_segment + 16], [src, src_offset]]
            kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8], "copy_gm_to_cbuf")
            _addr = [[dst, cp_segment * 2 + 16], [src, src_offset + cp_segment - 16]]
            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8], "copy_gm_to_cbuf")
        with ib.else_scope():
            _addr = [[dst, cp_segment + 16], [src, src_offset + cp_segment]]
            kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8], "copy_gm_to_cbuf")
            _addr = [[dst, cp_segment * 2 + 16], [src, src_offset + cp_segment * 2 - 16]]
            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8], "copy_gm_to_cbuf")
    else:
        dst = ub_info["ub_input"]
        _addr = [[dst, 0], [src, src_offset]]
        kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8], "copy_gm_to_ubuf")
        _addr = [[dst, cp_segment], [src, src_offset + cp_segment - 16]]
        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8], "copy_gm_to_ubuf")
        with ib.if_scope(y_index == para.size_in[2] - 1):
            _addr = [[dst, cp_segment + 16], [src, src_offset]]
            kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8], "copy_gm_to_ubuf")
            _addr = [[dst, cp_segment * 2 + 16], [src, src_offset + cp_segment - 16]]
            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8], "copy_gm_to_ubuf")
        with ib.else_scope():
            _addr = [[dst, cp_segment + 16], [src, src_offset + cp_segment]]
            kernel_api.kernel_cp_fuc(ib, _addr, [cp_segment, 8], "copy_gm_to_ubuf")
            _addr = [[dst, cp_segment * 2 + 16], [src, src_offset + cp_segment * 2 - 16]]
            kernel_api.kernel_cp_fuc(ib, _addr, [16, 8], "copy_gm_to_ubuf")

    return ib


def _copy_input_to_l1_fp16_cp(ib, dst, src, cp_segment, fp16_ub, fp32_ub, src_offset_, des_offset_, loop, tail):
    with ib.for_range(0, loop) as loop_i:
        _addr = [[fp16_ub, 0], [src, src_offset_ + loop_i * 256 * 8]]
        kernel_api.kernel_cp_fuc(ib, _addr, [256 * 8, 16], "copy_gm_to_ubuf")
        _addr = [[fp32_ub, 0], [fp16_ub, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr, [256 * 8, 8 * 8], "vconv_f162f32")
        _addr = [[dst, des_offset_ + loop_i * 256 * 8], [fp32_ub, 0]]
        kernel_api.kernel_cp_fuc(ib, _addr, [256 * 8, 8], "copy_ubuf_to_cbuf")
    if tail != 0:
        _addr = [[fp16_ub, 0], [src, src_offset_ + loop * 256 * 8]]
        kernel_api.kernel_cp_fuc(ib, _addr, [tail, 16], "copy_gm_to_ubuf")
        _addr = [[fp32_ub, 0], [fp16_ub, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr, [tail, 8 * 8], "vconv_f162f32")
        _addr = [[dst, des_offset_ + loop * 256 * 8], [fp32_ub, 0]]
        kernel_api.kernel_cp_fuc(ib, _addr, [tail, 8], "copy_ubuf_to_cbuf")
        _addr = [[dst, cp_segment + des_offset_], [fp32_ub, tail - 16]]
        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8], "copy_ubuf_to_cbuf")
    else:
        _addr = [[dst, cp_segment + des_offset_], [fp32_ub, 256 - 16]]
        kernel_api.kernel_cp_fuc(ib, _addr, [16, 8], "copy_ubuf_to_cbuf")
    return ib


def _copy_input_to_l1_fp16(para_list):
    dst, src, _dtype, cp_segment, src_offset, mid_ub, y_index, is_input_in_ub, ib, ub_info, para = para_list
    # copy to ub
    # ub do conv fp16 tp f32
    # copy tp l1
    vconv_loop = cp_segment // (256 * 8)
    vconv_tail = cp_segment % (256 * 8)
    fp32_mid = mid_ub[0]
    fp32_mid_1 = mid_ub[1]
    fp16_mid = mid_ub[2]
    fp16_mid_1 = mid_ub[3]

    ib = _copy_input_to_l1_fp16_cp(ib, dst, src, cp_segment, fp16_mid, fp32_mid, src_offset, 0, vconv_loop, vconv_tail)

    src_offset += cp_segment
    des_offset = cp_segment + 16
    with ib.if_scope(y_index != para.size_in[2] - 1):
        ib = _copy_input_to_l1_fp16_cp(ib, dst, src, cp_segment, fp16_mid_1, fp32_mid_1, src_offset, des_offset,
                                       vconv_loop, vconv_tail)
    with ib.else_scope():
        src_offset -= cp_segment
        ib = _copy_input_to_l1_fp16_cp(ib, dst, src, cp_segment, fp16_mid_1, fp32_mid_1, src_offset, des_offset,
                                       vconv_loop, vconv_tail)

    return ib


def _copy_input_to_l1(src_offset, dst_l1_ub, dat_len, mid_ub, y_index, is_input_in_ub, ib, ub_info, para):
    dst = dst_l1_ub
    src = para.inputs
    _dtype = src.dtype
    cp_segment = dat_len // 2

    para_list = [dst, src, _dtype, cp_segment, src_offset, mid_ub, y_index, is_input_in_ub, ib, ub_info, para]
    if _dtype == "float32":
        ib = _copy_input_to_l1_fp32(para_list)
    else:
        ib = _copy_input_to_l1_fp16(para_list)

    return ib


def _cast_f32_to_int32(ib, para, _data_info, f32_ub_info, int32_ub_info, dst_f32_ub_info, _ub_info):
    _ub_int32_info = int32_ub_info
    _ub_fp32_info = f32_ub_info
    _ub_des_fp32 = dst_f32_ub_info
    _addr_info = [_ub_int32_info, _ub_fp32_info]
    if para.devices == "1980":
        _addr_info = [_ub_int32_info, _ub_fp32_info]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_f322s32f")
        _addr_info = [_ub_des_fp32, _ub_int32_info]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_s322f32")
    else:
        _ub_fp16_512 = _ub_info["f16_512"]
        _addr_info = [[_ub_fp16_512, 0], _ub_fp32_info]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_f322f16")
        _addr_info = [_ub_int32_info, [_ub_fp16_512, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_f162s32f")
        ib.emit(tvm.call_extern("float16", "set_deqscale", tvm.const(1.0, dtype="float16")))
        _addr_info = [[_ub_fp16_512, 0], _ub_int32_info]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_deq")
        _addr_info = [_ub_des_fp32, [_ub_fp16_512, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_f162f32")

        # for precise
        _ub_temp_fp32_512 = _ub_info["x0_512_x1_512_ub"]
        _ub_int32_tmp = _ub_info["int32_512_ub_tmp"]
        _addr_info = [[_ub_temp_fp32_512, 0], _ub_fp32_info, _ub_des_fp32]
        kernel_api.kernel_two_to_one_common_fuc(ib, _addr_info, _data_info, "vsub")
        _addr_info = [[_ub_fp16_512, 0], [_ub_temp_fp32_512, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_f322f16")
        _addr_info = [[_ub_int32_tmp, 0], [_ub_fp16_512, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_f162s32f")
        ib.emit(tvm.call_extern("float16", "set_deqscale", tvm.const(1.0, dtype="float16")))
        _addr_info = [[_ub_fp16_512, 0], [_ub_int32_tmp, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_deq")
        _addr_info = [[_ub_temp_fp32_512, 0], [_ub_fp16_512, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, _data_info, "vconv_f162f32")
        _addr_info = [_ub_des_fp32, _ub_des_fp32, [_ub_temp_fp32_512, 0]]
        kernel_api.kernel_two_to_one_common_fuc(ib, _addr_info, _data_info, "vadd")

        _addr_info = [_ub_int32_info, _ub_int32_info, [_ub_int32_tmp, 0]]
        kernel_api.kernel_two_to_one_common_fuc(ib, _addr_info, _data_info, "vadd")

    return ib


def _process_pos_to_l1_is_h(ib, para, _input_fp32_ub, _input_fp32_512_num, _core_id, _h_per_core):
    if para.devices == "1980":
        ib.emit(tvm.call_extern(_input_fp32_ub.dtype, "vadds", _input_fp32_ub.access_ptr('w'),
                                _input_fp32_512_num.access_ptr('r'), _core_id * tvm.const(_h_per_core, dtype="float32"),
                                32, 1, 1, 8, 8))
        _input_fp32_512_num = _input_fp32_ub
    else:
        int_reg = ib.allocate("int32", (1,), name="int_data", scope=param.scope_reg)
        int_reg[0] = _core_id
        int32_ub_8 = apply_store_buffer(ib, "int32", [8], name="int32_ub_8")
        float32_ub_8 = apply_store_buffer(ib, "float32", [64], name="float32_ub_8")
        kernel_api.kernel_vector_dup_fuc(ib, [int32_ub_8, 0], int_reg[0], [1, 64])
        ib.emit(tvm.call_extern("float16", "set_deqscale", tvm.const(1.0, dtype="float16")))
        float16_ub_8 = apply_store_buffer(ib, "float16", [128], name="float16_ub_8")
        _addr_info = [[float16_ub_8, 0], [int32_ub_8, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, [1, 64], "vconv_deq")
        _addr_info = [[float32_ub_8, 0], [float16_ub_8, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, [1, 64], "vconv_f162f32")
        kernel_api.kernel_scalar_to_one_fuc(ib, [[float32_ub_8, 0], [float32_ub_8, 0]], [1, 64], ["vmuls", _h_per_core])
        core_reg = ib.allocate("float32", (1,), name="core_reg", scope=param.scope_reg)
        ib.emit(tvm.call_extern('int32', 'pipe_barrier', ARGS_STR_V_ALL))
        core_reg[0] = float32_ub_8.vload(0, "float32")
        ib.emit(tvm.call_extern(_input_fp32_ub.dtype, "vadds", _input_fp32_ub.access_ptr('w'),
                                _input_fp32_512_num.access_ptr('r'), core_reg[0], 32, 1, 1, 8, 8))
        _input_fp32_512_num = _input_fp32_ub

    return ib


def _process_pos_to_l1(ib, para, src_ub_info, mid_ub_info, des_l1_info, para_list, loop_index, _ub_info):
    _input_fp32_ub = src_ub_info[0]
    _input_fp32_512_num = src_ub_info[1]
    _mid_ub_int32 = mid_ub_info[0]
    _mid_one_ub_fp32 = mid_ub_info[1]
    _out_l1_int = des_l1_info[0]
    _out_l1_fp32 = des_l1_info[1]
    _scale, _core_id, _is_h, _h_per_core = para_list
    if _is_h:
        ib = _process_pos_to_l1_is_h(ib, para, _input_fp32_ub, _input_fp32_512_num, _core_id, _h_per_core)
        _input_fp32_512_num = _input_fp32_ub
    ib.emit(tvm.call_extern(_input_fp32_ub.dtype, "vmuls", _input_fp32_ub.access_ptr('w'),
                            _input_fp32_512_num.access_ptr('r'), tvm.const(_scale, dtype="float32"), 32, 1, 1, 8, 8))
    data_info = [32 * 64, 64]
    # process x pos
    ib = _cast_f32_to_int32(ib, para, data_info, [_input_fp32_ub, 0], [_mid_ub_int32, 0], [_input_fp32_ub, 256 * 8],
                            _ub_info)
    _addr_info = [[_out_l1_int, loop_index * 256 * 8], [_mid_ub_int32, 0]]
    kernel_api.kernel_cp_fuc(ib, _addr_info, [256 * 8, 8], "copy_ubuf_to_cbuf")
    # x_float - x_int
    # x_index_ub[0] == x_index_ub[64*64] - x_index_ub[0]
    _addr_info = [[_input_fp32_ub, 256 * 8], [_input_fp32_ub, 0], [_input_fp32_ub, 256 * 8]]
    kernel_api.kernel_two_to_one_common_fuc(ib, _addr_info, data_info, "vsub")
    # x_index_ub[64*64] == 1 - x_index_ub[0]
    ib.emit(tvm.call_extern(_input_fp32_ub.dtype, "vsub", _input_fp32_ub.access_ptr('w', offset=0),
                            _mid_one_ub_fp32.access_ptr('r'), _input_fp32_ub.access_ptr('r', offset=256 * 8), 32, 1, 1,
                            1, 8, 0, 8))
    # copy ub tp l1
    _addr_info = [[_out_l1_fp32, loop_index * 256 * 2 * 8], [_input_fp32_ub, 0]]
    kernel_api.kernel_cp_fuc(ib, _addr_info, [256 * 2 * 8, 8], "copy_ubuf_to_cbuf")

    return ib


def calu_x_y_out_pos(ib, para, _loop_info, _scale_info, _ub_info, _l1_info, range_512, idx, h_per_core):
    _x_loop = _loop_info[0]
    _y_loop = _loop_info[1]
    x_index_ub = _ub_info["x0_scale_x1_scale_512_ub"]
    y_index_ub = _ub_info["x2_512_x3_512_ub"]
    one_ub = _ub_info["const_0"]
    int32_ub = _ub_info["int32_512_ub"]
    _scale_w, _scale_h = _scale_info
    min_loop = min(_x_loop, _y_loop)

    with ib.for_range(0, min_loop) as _loop:
        ib = _process_pos_to_l1(ib, para, [x_index_ub, range_512], [int32_ub, one_ub],
                                [_l1_info["l1_xpos"], _l1_info["l1_xscale"]], [_scale_w, idx, False, h_per_core], _loop,
                                _ub_info)
        ib = _process_pos_to_l1(ib, para, [y_index_ub, range_512], [int32_ub, one_ub],
                                [_l1_info["l1_ypos"], _l1_info["l1_yscale"]], [_scale_h, idx, True, h_per_core], _loop,
                                _ub_info)
        kernel_api.kernel_scalar_to_one_fuc(ib, [[range_512, 0], [range_512, 0]], [256 * 8, 64], ["vadds", 256.0])
    if _x_loop != _y_loop:
        if _y_loop == min_loop:
            tail_loop = (_x_loop - min_loop)
            index_ub = x_index_ub
            index_l1_int = _l1_info["l1_xpos"]
            index_l1_fp32 = _l1_info["l1_xscale"]
            tail_scale = _scale_w
            is_h = False
        else:
            tail_loop = (_y_loop - min_loop)
            index_ub = y_index_ub
            index_l1_int = _l1_info["l1_ypos"]
            index_l1_fp32 = _l1_info["l1_yscale"]
            tail_scale = _scale_h
            is_h = True
        with ib.for_range(0, tail_loop) as _loop:
            ib = _process_pos_to_l1(ib, para, [index_ub, range_512], [int32_ub, one_ub], [index_l1_int, index_l1_fp32],
                                    [tail_scale, idx, is_h, h_per_core], _loop + min_loop, _ub_info)
            kernel_api.kernel_scalar_to_one_fuc(ib, [[range_512, 0], [range_512, 0]], [256 * 8, 64], ["vadds", 256.0])
    return ib


def _data_unfold(ib, para, part_index, part_data_len, is_input_in_ub, x_index_reg, ub_info, l1_info):
    if part_index == 0:
        x0x1_ub = ub_info["x0_512_x1_512_ub_unfold"]
        x2x3_ub = ub_info["x2_512_x3_512_ub_unfold"]
        pos_int_ub = ub_info["int32_512_ub"]
        index_align_index_num = part_data_len
        scale_offset = 0
    else:
        x0x1_ub = ub_info["x0_512_x1_512_ub_1"]
        x2x3_ub = ub_info["x2_512_x3_512_ub_1"]
        pos_int_ub = ub_info["int32_512_ub"]
        index_align_index_num = part_data_len
        scale_offset = 0
    if is_input_in_ub is True:
        src_addr = ub_info["ub_input"]
        copy_cmd = "copy_ubuf_to_ubuf"
    else:
        src_addr = l1_info["l1_input"]
        copy_cmd = "copy_cbuf_to_ubuf"

    with ib.for_range(0, index_align_index_num) as _index:
        ib.emit(tvm.call_extern("int32", "reg_mov", tvm.call_extern(x_index_reg.dtype, "reg", x_index_reg),
                                pos_int_ub.access_ptr("r", offset=(scale_offset + _index) * 8), 0))
        # copy 512 to block segment
        ib.emit(tvm.call_extern(x0x1_ub.dtype, copy_cmd, x0x1_ub.access_ptr('w', offset=_index * 16),
                                src_addr.access_ptr('r', offset=x_index_reg * para.c0), 0, 2, 2, 0, 510))
        ib.emit(tvm.call_extern(x2x3_ub.dtype, copy_cmd, x2x3_ub.access_ptr('w', offset=_index * 16),
                                src_addr.access_ptr('r', offset=(x_index_reg + para.w_in + 1) * para.c0), 0, 2, 2, 0,
                                510))

    return ib


def calculate_interp_point(ib, x0x1_ub, x0x1_ub_unfold, y_scale_2_ub, x2x3_ub, x2x3_ub_unfold, x_scale_512_ub,
                           scale_offset, _repeat):
    # x0*(1-Yv)  x1*(1-Yv)
    ib.emit(tvm.call_extern(x0x1_ub.dtype, "vmul", x0x1_ub.access_ptr('w'), y_scale_2_ub.access_ptr('r', offset=0),
                            x0x1_ub_unfold.access_ptr('r'), _repeat * 2, 1, 0, 1, 8, 0, 8))

    # x2*Yv  x3*Yv
    ib.emit(tvm.call_extern(x2x3_ub.dtype, "vmul", x2x3_ub.access_ptr('w'), y_scale_2_ub.access_ptr('r', offset=8),
                            x2x3_ub_unfold.access_ptr('r'), _repeat * 2, 1, 0, 1, 8, 0, 8))
    # x0*(1-Yv)  x1*(1-Yv)
    ib.emit(tvm.call_extern(x0x1_ub.dtype, "vmul", x0x1_ub.access_ptr('w', offset=512 * 8),
                            y_scale_2_ub.access_ptr('r', offset=0), x0x1_ub_unfold.access_ptr('r', offset=512 * 8),
                            _repeat * 2, 1, 0, 1, 8, 0, 8))
    # x2*Yv  x3*Yv
    ib.emit(tvm.call_extern(x2x3_ub.dtype, "vmul", x2x3_ub.access_ptr('w', offset=512 * 8),
                            y_scale_2_ub.access_ptr('r', offset=8), x2x3_ub_unfold.access_ptr('r', offset=512 * 8),
                            _repeat * 2, 1, 0, 1, 8, 0, 8))

    # x0 == x0*(1-Yv)*(1-Xu)     offset == 0
    ib.emit(tvm.call_extern(x0x1_ub.dtype, "vmul", x0x1_ub.access_ptr('w'),
                            x_scale_512_ub.access_ptr('r', offset=scale_offset * 8), x0x1_ub.access_ptr('r'), _repeat,
                            2, 1, 2, 16, 8, 16))

    # x2 == x2*Yv*(1-Xu)    offset == 0
    ib.emit(tvm.call_extern(x2x3_ub.dtype, "vmul", x2x3_ub.access_ptr('w'),
                            x_scale_512_ub.access_ptr('r', offset=scale_offset * 8), x2x3_ub.access_ptr('r'), _repeat,
                            2, 1, 2, 16, 8, 16))
    # x0 == x0*(1-Yv)*(1-Xu)  offset == 8
    ib.emit(tvm.call_extern(x0x1_ub.dtype, "vmul", x0x1_ub.access_ptr('w', offset=8),
                            x_scale_512_ub.access_ptr('r', offset=scale_offset * 8), x0x1_ub.access_ptr('r', offset=8),
                            _repeat, 2, 1, 2, 16, 8, 16))
    # x2 == x2*Yv*(1-Xu)  offset == 8
    ib.emit(tvm.call_extern(x2x3_ub.dtype, "vmul", x2x3_ub.access_ptr('w', offset=8),
                            x_scale_512_ub.access_ptr('r', offset=scale_offset * 8), x2x3_ub.access_ptr('r', offset=8),
                            _repeat, 2, 1, 2, 16, 8, 16))
    # note: x1 == x1*(1-Yv)*Xu
    ib.emit(tvm.call_extern(x0x1_ub.dtype, "vmul", x0x1_ub.access_ptr('w', offset=512 * 8),
                            x_scale_512_ub.access_ptr('r', offset=256 * 8 + scale_offset * 8),
                            x0x1_ub.access_ptr('r', offset=512 * 8), _repeat, 2, 1, 2, 16, 8, 16))
    # note: x3 == x3Yv*Xu
    ib.emit(tvm.call_extern(x2x3_ub.dtype, "vmul", x2x3_ub.access_ptr('w', offset=512 * 8),
                            x_scale_512_ub.access_ptr('r', offset=256 * 8 + scale_offset * 8),
                            x2x3_ub.access_ptr('r', offset=512 * 8), _repeat, 2, 1, 2, 16, 8, 16))
    # x1 == x1*(1-Yv)*Xu
    ib.emit(tvm.call_extern(x0x1_ub.dtype, "vmul", x0x1_ub.access_ptr('w', offset=512 * 8 + 8),
                            x_scale_512_ub.access_ptr('r', offset=256 * 8 + scale_offset * 8),
                            x0x1_ub.access_ptr('r', offset=512 * 8 + 8), _repeat, 2, 1, 2, 16, 8, 16))
    # x3 == x3Yv*Xu
    ib.emit(tvm.call_extern(x2x3_ub.dtype, "vmul", x2x3_ub.access_ptr('w', offset=512 * 8 + 8),
                            x_scale_512_ub.access_ptr('r', offset=256 * 8 + scale_offset * 8),
                            x2x3_ub.access_ptr('r', offset=512 * 8 + 8), _repeat, 2, 1, 2, 16, 8, 16))
    # vadd x0+x1
    _addr_info = [[x0x1_ub, 0], [x0x1_ub, 0], [x0x1_ub, 512 * 8]]
    kernel_api.kernel_two_to_one_common_fuc(ib, _addr_info, [_repeat * 2 * 8 * 8, 8 * 8], "vadd")
    # vadd x2+x3
    _addr_info = [[x2x3_ub, 0], [x2x3_ub, 0], [x2x3_ub, 512 * 8]]
    kernel_api.kernel_two_to_one_common_fuc(ib, _addr_info, [_repeat * 2 * 8 * 8, 8 * 8], "vadd")

    return ib


def _process(ib, para, part_index, index_block, y_index, x_index, output_offset, ub_info, idx, h_per_core):
    index_align_blcok_1 = index_block
    index_align_blcok_2 = index_block
    if part_index == 0:
        x0x1_ub_unfold = ub_info["x0_512_x1_512_ub_unfold"]
        x2x3_ub_unfold = ub_info["x2_512_x3_512_ub_unfold"]
        x0x1_ub = ub_info["x0_512_x1_512_ub"]
        x2x3_ub = ub_info["x2_512_x3_512_ub"]
        x_scale_512_ub = ub_info["x0_scale_x1_scale_512_ub"]
        y_scale_2_ub = ub_info["y0_scale_ub"]
        fp16_ub = ub_info["f16_512"]
        index_align_index_num = index_align_blcok_1
        index_align_blcok = index_align_blcok_1
        scale_offset = 0
    else:
        x0x1_ub_unfold = ub_info["x0_512_x1_512_ub_unfold"]
        x2x3_ub_unfold = ub_info["x2_512_x3_512_ub_unfold"]
        x0x1_ub = ub_info["x0_512_x1_512_ub_1"]
        x2x3_ub = ub_info["x2_512_x3_512_ub_1"]
        x_scale_512_ub = ub_info["x0_scale_x1_scale_512_ub"]
        y_scale_2_ub = ub_info["y0_scale_ub"]
        fp16_ub = ub_info["f16_512_1"]
        index_align_index_num = index_block - index_align_blcok_1
        index_align_blcok = index_align_blcok_2
        scale_offset = index_align_blcok_1

    _repeat = (index_align_blcok // 8 + 1) if index_align_blcok % 8 != 0 else (index_align_blcok // 8 + 0)
    if index_block <= 256:
        ib = calculate_interp_point(ib, x0x1_ub, x0x1_ub_unfold, y_scale_2_ub, x2x3_ub, x2x3_ub_unfold, x_scale_512_ub,
                                    scale_offset, _repeat)

    # vadd x0+x2+x1+x3
    _addr_info = [[ub_info["out_f32"], 0], [x0x1_ub, 0], [x2x3_ub, 0]]
    kernel_api.kernel_two_to_one_common_fuc(ib, _addr_info, [_repeat * 2 * 8 * 8, 8 * 8], "vadd")
    out_ub = ub_info["out_f32"]
    out_offset = (
                idx * h_per_core + y_index
                ) * para.w_out * para.c0 + x_index * 256 * para.c0 + scale_offset * 16 + output_offset
    if para.dtype == "float161":  # do not change "float161", avoid to do fp32 to fp16
        # vconv out to fp16
        _addr_info = [[fp16_ub, 0], [out_ub, 0]]
        kernel_api.kernel_cast_to_fuc(ib, _addr_info, [_repeat * 2 * 8 * 8, 8 * 8], "vconv_f322f16")
        out_ub = fp16_ub
        # copy 512 out to gm
        ib.emit(tvm.call_extern("float16", "copy_ubuf_to_gm", para.outputs.access_ptr('w', offset=out_offset),
                                out_ub.access_ptr('r'), 0, 1, index_align_index_num, 0, 0))
    else:
        ib.emit(tvm.call_extern(para.outputs.dtype, "copy_ubuf_to_gm",
                                para.outputs.access_ptr('w', offset=out_offset), out_ub.access_ptr('r'), 0, 1,
                                index_align_index_num * 2, 0, 0))

    return ib


def compute_each_loop(ib, para, s_param, ub_info_, l1_info_, range_512_, pos_reg_, y_index_reg_, x_index_reg_, c0, idx,
                      src_nc1_input_offset_, src_nc1_output_offset_, x_segment_loop_, x_segment_tail_):
    pos_reg_[1] = tvm.const(-1.0, dtype="int32")
    # pre copy fisrt line of input
    _addr_list = [[ub_info_["int32_512_ub_y"], 0], [l1_info_["l1_ypos"], 0 * 8]]
    kernel_api.kernel_cp_fuc(ib, _addr_list, [8, 8], "copy_cbuf_to_ubuf")
    ib.emit(tvm.call_extern("int32", "reg_mov", tvm.call_extern(y_index_reg_.dtype, "reg", y_index_reg_),
                            ub_info_["int32_512_ub_y"].access_ptr("r"), 0))
    images_offset = src_nc1_input_offset_ + y_index_reg_ * para.w_in * c0
    copy_data_len = para.w_in * c0 * 2
    ib = _copy_input_to_l1(images_offset, l1_info_["l1_input"], copy_data_len,
                           [range_512_, range_512_, ub_info_["f16_512_1"], ub_info_["f16_512_1"]], y_index_reg_,
                           s_param.is_input_in_ub, ib, ub_info_, para)
    _addr_list = [[ub_info_["int32_512_ub"], 0], [l1_info_["l1_xpos"], x_segment_loop_ * 256 * 8]]
    kernel_api.kernel_cp_fuc(ib, _addr_list, [x_segment_tail_ * 8, 8], "copy_cbuf_to_ubuf")
    # copy x pos float from l1 to ub
    _addr_list = [[ub_info_["x0_scale_x1_scale_512_ub"], 0], [l1_info_["l1_xscale"], x_segment_loop_ * 512 * 8]]
    kernel_api.kernel_cp_fuc(ib, _addr_list, [512 * 8, 8], "copy_cbuf_to_ubuf")
    with ib.for_range(0, s_param.h_per_core) as y_loop:
        with ib.if_scope(idx * s_param.h_per_core + y_loop < para.h_out):
            # copy y pos int from l1 to ub
            # copy y sclace float from l1 to ub
            ib.emit(tvm.call_extern(ub_info_["y0_scale_ub"].dtype, "copy_cbuf_to_ubuf",
                                    ub_info_["y0_scale_ub"].access_ptr('w', offset=0),
                                    l1_info_["l1_yscale"].access_ptr(
                                        'r',
                                        offset=(y_loop // 256 * 512 + y_loop % 256) * 8),
                                    0, 2, 1, 255, 0))
            # if y_before_index_reg != y_index_reg_
            # copy from gm to ub or l1 and open data to x1 x2 x3 x4
            with ib.if_scope(pos_reg_[1] != y_index_reg_):
                pos_reg_[1] = y_index_reg_
                ib = _data_unfold(ib, para, 0, x_segment_tail_, s_param.is_input_in_ub, x_index_reg_, ub_info_,
                                  l1_info_)

            _addr_list = [[ub_info_["int32_512_ub_y"], 0],
                          [l1_info_["l1_ypos"], (y_loop // 256 * 256 + y_loop % 256) * 8]]
            kernel_api.kernel_cp_fuc(ib, _addr_list, [16, 8], "copy_cbuf_to_ubuf")
            # copy next unflod data if next y != cu_y
            ib.emit(tvm.call_extern("int32", "reg_mov", tvm.call_extern(y_index_reg_.dtype, "reg", y_index_reg_),
                                    ub_info_["int32_512_ub_y"].access_ptr("r", offset=8), 0))
            with ib.if_scope(tvm.all(y_index_reg_ != pos_reg_[1], y_loop < (s_param.h_per_core - 1),
                                     y_index_reg_ < para.h_in)):
                images_offset = src_nc1_input_offset_ + y_index_reg_ * para.w_in * c0
                ib = _copy_input_to_l1(images_offset, l1_info_["l1_input"], copy_data_len,
                                       [range_512_, range_512_, ub_info_["f16_512_1"], ub_info_["f16_512_1"]],
                                       y_index_reg_, s_param.is_input_in_ub, ib, ub_info_, para)
            ib = _process(ib, para, 0, x_segment_tail_, y_loop, x_segment_loop_, src_nc1_output_offset_, ub_info_, idx,
                          s_param.h_per_core)

    return ib


def compute_with_in_normal_situation(ib, para):
    s_param = NormalSituationParam(para)

    # use multi-core resource
    ib.scope_attr(para.block, "thread_extent", s_param.core_num)
    idx = para.block.var

    # Note:
    #   1. all scale calculate in ubuf and store in L1, dtype is f32
    #   2. 16 or 8 lines data move in L1 first, each line split by 1024 block at most
    #   3. 8 lines data in ubuf move out, each line split by 512 block at most
    ##########
    ib, ub_info, l1_info = apply_ub(ib, para)

    ib.emit(tvm.call_extern("float32", "vector_dup", ub_info["const_0"].access_ptr('w'),
                            tvm.const(1.0, dtype="float32"), 1, 1, 0, 8, 0))
    # No.4 normal input/output H/W : range 512 create
    range_512 = apply_store_buffer(ib, "float32", [512 * para.f32_c0], name="range_512")

    # check whether scale is int or 1/int
    # create 512 blocks from 0-511 start
    loop_level1 = 8
    c0 = 16
    ib.emit(tvm.call_extern("uint64", "set_vector_mask", tvm.const(0, dtype="uint64"),
                            tvm.const((1 << para.f32_c0) - 1, dtype="uint64")))
    ib.emit(tvm.call_extern("float32", "vector_dup", range_512.access_ptr('w', offset=8 * 0),
                            tvm.const(0.0, dtype="float32"), 1, 1, 0, 8, 0))

    with ib.for_range(0, loop_level1 - 1) as i:
        ib.emit(tvm.call_extern("float32", "vadds", range_512.access_ptr('w', offset=8 * (i + 1)),
                                range_512.access_ptr('r', offset=8 * i), tvm.const(1.0, dtype="float32"),
                                1, 1, 1, 1, 1))
    ib.emit(tvm.call_extern("uint64", "set_vector_mask", tvm.const(-1, dtype="uint64"), tvm.const(-1, dtype="uint64")))
    loop_level1 = 256 // 8
    with ib.for_range(0, loop_level1 - 1) as i:
        ib.emit(tvm.call_extern("float32", "vadds", range_512.access_ptr('w', offset=8 * 8 * (i + 1)),
                                range_512.access_ptr('r', offset=8 * 8 * i), tvm.const(8.0, dtype="float32"),
                                1, 1, 1, 8, 8))

    # create 512 blocks from 0-511 end

    # Note:
    #   1. w direction scale 512 left and right join together
    #   2. h direction scale 512 top and bottom store speartely to double lines
    # Note:
    #   1. devices 1910 f32 <-> s32 need convert to f16 first
    #   2. devices 1910 f32 <-> s32 convert directly

    # No.4 normal input/output H/W: scale calculate: input H\W == output: devices 1910
    # level_1 loop

    ib = calu_x_y_out_pos(ib, para, [s_param.loop_levelx, s_param.loop_levely], [para.scale_w, para.scale_h], ub_info,
                          l1_info, range_512, idx, s_param.h_per_core)
    pos_reg = apply_reg_buffer(ib, "int32", [7], name="pos_reg")

    # calu y edge
    nc1_input_total = para.size_in[0] * para.size_in[1]
    nc1_input_offset = para.size_in[2] * para.size_in[3] * para.size_in[4]
    nc1_output_offset = para.size_out[2] * para.size_out[3] * para.size_out[4]
    y_index_reg = pos_reg[0]
    x_index_reg = pos_reg[2]
    x_segment_loop = para.w_out // 256
    x_segment_tail = para.w_out % 256

    # nc1_input_total
    with ib.for_range(0, nc1_input_total) as nc1_loop:
        src_nc1_input_offset = nc1_loop * nc1_input_offset
        src_nc1_output_offset = nc1_loop * nc1_output_offset
        with ib.for_range(0, x_segment_loop) as x_loop:
            ib = compute_each_loop(ib, para, s_param, ub_info, l1_info, range_512, pos_reg, y_index_reg, x_index_reg,
                                   c0, idx, src_nc1_input_offset, src_nc1_output_offset, x_loop, 256)

        if x_segment_tail != 0:
            ib = compute_each_loop(ib, para, s_param, ub_info, l1_info, range_512, pos_reg, y_index_reg, x_index_reg,
                                   c0, idx, src_nc1_input_offset, src_nc1_output_offset, x_segment_loop, x_segment_tail)

    return ib
