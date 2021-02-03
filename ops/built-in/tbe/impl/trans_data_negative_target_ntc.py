# Copyright 2021 Huawei Technologies Co., Ltd
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
trans_data_negative_target_ntc
"""

from te import tik
from impl import trans_data_common_func as tdc


# used for scalar
PAD_IDX_LIST = (0, 1)
# frame up levels
FRAME_LEVEL = 2


# pylint: disable=too-many-locals, inconsistent-return-statements
def _renew_input_output_shape_format(in_shape, out_shape, in_format, out_format):
    """
    renew shape and format to adapt tiling process
    """

    in_format_upper = in_format.upper()
    out_format_upper = out_format.upper()

    if in_format_upper == "NDC1HWC0" and out_format_upper == "NCDHW":
        in_format_new = "NDCHT"
        out_format_new = "NCDH"
        axis_n, axis_c, axis_d, axis_h, axis_w = out_shape
        axis_c0 = in_shape[-1]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        in_shape_new = [axis_n] + [axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_c0]
        out_shape_new = [axis_n] + [axis_c] + [axis_d] + [axis_h * axis_w]
        new_params_ncdhw = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_ncdhw

    if in_format_upper == "NC1HWC0" and out_format_upper == "NCHW":
        in_format_new = "NCHT"
        out_format_new = "NCH"
        axis_n, axis_c, axis_h, axis_w = out_shape
        axis_c0 = in_shape[-1]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        in_shape_new = [axis_n] + [axis_c1] + [axis_h * axis_w] + [axis_c0]
        out_shape_new = [axis_n] + [axis_c] + [axis_h * axis_w]
        new_params_nchw = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_nchw

    if in_format_upper in ("FRACTAL_Z", "FRACTAL_ZN") and out_format_upper == "HWCN":
        in_format_new = "CHNT"
        out_format_new = "HCN"
        axis_h, axis_w, axis_c, axis_n = out_shape
        axis_c0 = in_shape[-1]
        axis_ni = in_shape[-2]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, axis_ni)
        in_shape_new = [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        out_shape_new = [axis_h * axis_w] + [axis_c] + [axis_n]
        new_params_hwcn = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_hwcn


# pylint: disable=too-many-statements
def _get_mc_info_negative(cr_args, c_args, cl_args):
    """
    get multiple core axis position for negative transform
    """

    src_cr_lp_cnt, src_cr_size, src_cr_lp_unit = cr_args
    src_c_lp_cnt, src_c_size, src_c_lp_unit, src_c_lp_step_in, src_c_lp_step_out = c_args
    src_cl_lp_cnt, src_cl_size, src_cl_lp_unit = cl_args

    tmp_full_lp_cnt_cr = tdc.CORE_DIM_NUM if tdc.floor_div(src_cr_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_cr = src_cr_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_cr == 0 and src_cr_size % src_cr_lp_unit > src_cr_lp_unit // 2:
        tmp_full_lp_cnt_cr += tdc.CORE_DIM_NUM
    full_lp_cnt_cr = tmp_full_lp_cnt_cr + reminder_lp_cnt_cr

    tmp_full_lp_cnt_c = tdc.CORE_DIM_NUM if tdc.floor_div(src_c_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_c = src_c_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_c == 0:
        tmp_full_lp_cnt_c += tdc.CORE_DIM_NUM
    full_lp_cnt_c = tmp_full_lp_cnt_c + reminder_lp_cnt_c

    tmp_full_lp_cnt_cl = tdc.CORE_DIM_NUM if tdc.floor_div(src_cl_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_cl = src_cl_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_cl == 0:
        tmp_full_lp_cnt_cl += tdc.CORE_DIM_NUM
    full_lp_cnt_cl = tmp_full_lp_cnt_cl + reminder_lp_cnt_cl

    lp_cnt_list = (full_lp_cnt_cl, full_lp_cnt_c, full_lp_cnt_cr)
    if lp_cnt_list.index(max(lp_cnt_list)) == 0:
        tp_200_mc_pos = 0
        tp_200_is_mc_cl = 1
        tp_200_is_mc_cr = 0
        tp_200_used_core_cnt = tdc.ceil_div(src_cl_lp_cnt, tdc.ceil_div(src_cl_lp_cnt, tdc.CORE_DIM_NUM))
        tp_200_nlc_cl_lp_cnt = tdc.ceil_div(src_cl_lp_cnt, tp_200_used_core_cnt)
        tp_200_lc_cl_lp_cnt = src_cl_lp_cnt - tp_200_nlc_cl_lp_cnt * (tp_200_used_core_cnt - 1)
        tp_200_core_step_in = 0
        tp_200_core_step_out = 0
        tp_200_nlc_cl_left = 0
        tp_200_lc_cl_left = src_cl_size % src_cl_lp_unit
        tp_200_nlc_c_lp_cnt = src_c_lp_cnt
        tp_200_lc_c_lp_cnt = src_c_lp_cnt
        tp_200_nlc_c_left = src_c_size % src_c_lp_unit
        tp_200_lc_c_left = src_c_size % src_c_lp_unit
        tp_200_nlc_cr_lp_cnt = src_cr_lp_cnt
        tp_200_lc_cr_lp_cnt = src_cr_lp_cnt
        tp_200_nlc_cr_left = src_cr_size % src_cr_lp_unit
        tp_200_lc_cr_left = src_cr_size % src_cr_lp_unit
    elif lp_cnt_list.index(max(lp_cnt_list)) == 1:
        tp_200_mc_pos = 1
        tp_200_is_mc_cl = 0
        tp_200_is_mc_cr = 0
        tp_200_used_core_cnt = tdc.ceil_div(src_c_lp_cnt, tdc.ceil_div(src_c_lp_cnt, tdc.CORE_DIM_NUM))
        tp_200_nlc_c_lp_cnt = tdc.ceil_div(src_c_lp_cnt, tp_200_used_core_cnt)
        tp_200_lc_c_lp_cnt = src_c_lp_cnt - (tp_200_used_core_cnt - 1) * tp_200_nlc_c_lp_cnt
        tp_200_nlc_c_left = 0
        tp_200_lc_c_left = src_c_size % src_c_lp_unit
        tp_200_core_step_in = tp_200_nlc_c_lp_cnt * src_c_lp_step_in
        tp_200_core_step_out = tp_200_nlc_c_lp_cnt * src_c_lp_step_out
        tp_200_nlc_cr_lp_cnt = src_cr_lp_cnt
        tp_200_lc_cr_lp_cnt = src_cr_lp_cnt
        tp_200_nlc_cr_left = src_cr_size % src_cr_lp_unit
        tp_200_lc_cr_left = src_cr_size % src_cr_lp_unit
        tp_200_nlc_cl_lp_cnt = src_cl_lp_cnt
        tp_200_lc_cl_lp_cnt = src_cl_lp_cnt
        tp_200_nlc_cl_left = src_cl_size % src_cl_lp_unit
        tp_200_lc_cl_left = src_cl_size % src_cl_lp_unit
    else:
        tp_200_mc_pos = 2
        tp_200_is_mc_cl = 0
        tp_200_is_mc_cr = 1
        tp_200_used_core_cnt = tdc.ceil_div(src_cr_lp_cnt, tdc.ceil_div(src_cr_lp_cnt, tdc.CORE_DIM_NUM))
        tp_200_nlc_cr_lp_cnt = tdc.ceil_div(src_cr_lp_cnt, tp_200_used_core_cnt)
        tp_200_lc_cr_lp_cnt = src_cr_lp_cnt - (tp_200_used_core_cnt - 1) * tp_200_nlc_cr_lp_cnt
        tp_200_nlc_cr_left = 0
        tp_200_lc_cr_left = src_cr_size % src_cr_lp_unit
        tp_200_core_step_in = 0
        tp_200_core_step_out = 0
        tp_200_nlc_c_lp_cnt = src_c_lp_cnt
        tp_200_lc_c_lp_cnt = src_c_lp_cnt
        tp_200_nlc_c_left = src_c_size % src_c_lp_unit
        tp_200_lc_c_left = src_c_size % src_c_lp_unit
        tp_200_nlc_cl_lp_cnt = src_cl_lp_cnt
        tp_200_lc_cl_lp_cnt = src_cl_lp_cnt
        tp_200_nlc_cl_left = src_cl_size % src_cl_lp_unit
        tp_200_lc_cl_left = src_cl_size % src_cl_lp_unit

    tiling_params = [tp_200_mc_pos, tp_200_is_mc_cr, tp_200_is_mc_cl] + \
                    [tp_200_used_core_cnt, tp_200_core_step_in, tp_200_core_step_out] + \
                    [tp_200_nlc_cl_lp_cnt, tp_200_nlc_c_lp_cnt, tp_200_nlc_cr_lp_cnt,
                     tp_200_nlc_cl_left, tp_200_nlc_c_left, tp_200_nlc_cr_left] + \
                    [tp_200_lc_cl_lp_cnt, tp_200_lc_c_lp_cnt, tp_200_lc_cr_lp_cnt,
                     tp_200_lc_cl_left, tp_200_lc_c_left, tp_200_lc_cr_left]

    return tiling_params


# pylint: disable=redefined-builtin, unbalanced-tuple-unpacking
def _tiling_params_negative(args):
    """
    calculate real tiling params for negative transform and last axis of target format is c
    """

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size = args
    c0_len = in_shape[-1]  # axis c0
    tp_names = locals()

    # tiling branch
    tp_200_tiling_mode = 200
    # ub layout tiling parameters
    if src_format[-2] == dst_format[-1]:  # such as NC1HWC0 -> NCHW
        tp_200_tmp_ub_offset = 0
        tp_200_src_2_dst_flag = 0
    else:  # such as C1HWNoNiC0 -> NCHW
        tp_200_tmp_ub_offset = tdc.NI_16 * c0_len
        tp_200_src_2_dst_flag = 1
    one_vnc_line_size = (ub_size - tp_200_tmp_ub_offset) // 2 // tdc.VNC_LINES // block_elem_cnt * block_elem_cnt
    tp_200_ub_offset = tp_200_tmp_ub_offset + one_vnc_line_size * tdc.VNC_LINES
    tp_200_vnc_line_size = one_vnc_line_size

    # dst axis C-RIGHT tiling parameters
    axis_src_cr_size = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:])
    tp_200_src_cr_lp_unit = one_vnc_line_size // c0_len // block_elem_cnt * block_elem_cnt
    src_cr_lp_cnt = tdc.ceil_div(axis_src_cr_size, tp_200_src_cr_lp_unit)
    tmp_dst_cr_format = dst_format[dst_format.index("C") + 1:]
    tmp_dst_cr_shape = out_shape[dst_format.index("C") + 1:]
    # count method: cr_idx/dst_rsize%size*dst_asize
    tmp_dst_cr_shape.append(1)
    for idx, chr in enumerate(reversed(tmp_dst_cr_format)):
        src_chr_pos = src_format.index(chr)
        tp_names["tp_200_cr_in_idx_" + str(idx) + "_size"] = out_shape[dst_format.index(chr)]
        tp_names["tp_200_cr_in_idx_" + str(idx) + "_dst_rsize"] = tdc.get_shape_size(tmp_dst_cr_shape[-1 - idx:])
        tp_names["tp_200_cr_in_idx_" + str(idx) + "_src_asize"] = tdc.get_shape_size(in_shape[src_chr_pos + 1:])
    # suppose there are 2 axises
    pad_axis_cnt = FRAME_LEVEL - len(tmp_dst_cr_format)
    if pad_axis_cnt:
        for _, idx in enumerate(PAD_IDX_LIST[len(tmp_dst_cr_format):]):
            tp_names["tp_200_cr_in_idx_" + str(idx) + "_size"] = 1
            tp_names["tp_200_cr_in_idx_" + str(idx) + "_dst_rsize"] = 0
            tp_names["tp_200_cr_in_idx_" + str(idx) + "_src_asize"] = 0
    tp_200_src_cr_step_out = 1

    # axis C tiling parameters
    axis_src_c_size = in_shape[src_format.index("C")]
    axis_dst_c_size = out_shape[dst_format.index("C")]
    if src_cr_lp_cnt > 1:
        tp_200_src_c_lp_unit = 1
    else:
        tp_200_src_c_lp_unit = tp_200_src_cr_lp_unit // tdc.ceil_fill(axis_src_cr_size, block_elem_cnt)
    src_c_lp_cnt = tdc.ceil_div(axis_src_c_size, tp_200_src_c_lp_unit)
    tp_200_src_c_step_in = tdc.get_shape_size(in_shape[src_format.index("C") + 1:])
    tp_200_src_c_step_out = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:])
    tp_200_src_c_lp_step_in = tp_200_src_c_lp_unit * tp_200_src_c_step_in
    tp_200_src_c_lp_step_out = tp_200_src_c_lp_unit * c0_len * tp_200_src_c_step_out
    tp_200_c_mod_c0 = axis_dst_c_size % c0_len

    # dst axis C-LEFT tiling parameters
    axis_src_cl_size = tdc.get_shape_size(out_shape[:dst_format.index("C")])
    tp_200_src_cl_lp_unit = tdc.NI_16
    tp_200_src_cl_step_out = tdc.get_shape_size(out_shape[dst_format.index("C"):])
    src_cl_lp_cnt = tdc.ceil_div(axis_src_cl_size, tp_200_src_cl_lp_unit)
    tmp_dst_cl_format = dst_format[:dst_format.index("C")]
    tmp_c_left_shape = out_shape[:dst_format.index("C")]
    tmp_c_left_shape.append(1)
    # count method: left_axis_size/dst_rsize%size*asize
    for idx, chr in enumerate(reversed(tmp_dst_cl_format)):
        tp_names["tp_200_in_idx_" + str(idx) + "_size"] = out_shape[dst_format.index(chr)]
        tp_names["tp_200_in_idx_" + str(idx) + "_dst_rsize"] = tdc.get_shape_size(tmp_c_left_shape[-1 - idx:])
        tp_names["tp_200_in_idx_" + str(idx) + "_src_asize"] = tdc.get_shape_size(in_shape[src_format.index(chr) + 1:])
    # suppose there are 2 axises
    pad_axis_cnt = FRAME_LEVEL - len(tmp_dst_cl_format)
    if pad_axis_cnt:
        for _, idx in enumerate(PAD_IDX_LIST[len(tmp_dst_cl_format):]):
            tp_names["tp_200_in_idx_" + str(idx) + "_size"] = 1
            tp_names["tp_200_in_idx_" + str(idx) + "_dst_rsize"] = 0
            tp_names["tp_200_in_idx_" + str(idx) + "_src_asize"] = 0

    # mulitple core parameters
    cr_args = src_cr_lp_cnt, axis_src_cr_size, tp_200_src_cr_lp_unit
    c_args = src_c_lp_cnt, axis_src_c_size, tp_200_src_c_lp_unit, tp_200_src_c_lp_step_in, tp_200_src_c_lp_step_out
    cl_args = src_cl_lp_cnt, axis_src_cl_size, tp_200_src_cl_lp_unit
    tp_200_mc_pos, tp_200_is_mc_cr, tp_200_is_mc_cl, tp_200_used_core_cnt, tp_200_core_step_in, tp_200_core_step_out, \
    tp_200_nlc_cl_lp_cnt, tp_200_nlc_c_lp_cnt, tp_200_nlc_cr_lp_cnt, tp_200_nlc_cl_left, tp_200_nlc_c_left, \
    tp_200_nlc_cr_left, tp_200_lc_cl_lp_cnt, tp_200_lc_c_lp_cnt, tp_200_lc_cr_lp_cnt, \
    tp_200_lc_cl_left, tp_200_lc_c_left, tp_200_lc_cr_left = _get_mc_info_negative(cr_args, c_args, cl_args)

    sub_tiling_params = [tp_200_tiling_mode, tp_200_ub_offset, tp_200_tmp_ub_offset, tp_200_mc_pos,
                         tp_200_used_core_cnt, tp_200_core_step_in, tp_200_core_step_out, tp_200_nlc_cr_lp_cnt,
                         tp_200_nlc_c_lp_cnt, tp_200_nlc_cl_lp_cnt, tp_200_nlc_cr_left, tp_200_nlc_c_left,
                         tp_200_nlc_cl_left, tp_200_lc_cr_lp_cnt, tp_200_lc_c_lp_cnt, tp_200_lc_cl_lp_cnt,
                         tp_200_lc_cr_left, tp_200_lc_c_left, tp_200_lc_cl_left, tp_200_src_cr_lp_unit,
                         tp_200_src_c_lp_unit, tp_200_src_cl_lp_unit, tp_200_src_cr_step_out, tp_200_src_c_step_in,
                         tp_200_src_c_step_out, tp_200_src_cl_step_out,
                         tp_200_src_c_lp_step_in, tp_200_src_c_lp_step_out, tp_200_c_mod_c0, tp_200_is_mc_cr,
                         tp_200_is_mc_cl, tp_200_src_2_dst_flag, tp_200_vnc_line_size,
                         tp_names["tp_200_in_idx_0_size"], tp_names["tp_200_in_idx_0_dst_rsize"],
                         tp_names["tp_200_in_idx_0_src_asize"], tp_names["tp_200_in_idx_1_size"],
                         tp_names["tp_200_in_idx_1_dst_rsize"], tp_names["tp_200_in_idx_1_src_asize"],
                         tp_names["tp_200_cr_in_idx_0_size"], tp_names["tp_200_cr_in_idx_0_dst_rsize"],
                         tp_names["tp_200_cr_in_idx_0_src_asize"], tp_names["tp_200_cr_in_idx_1_size"],
                         tp_names["tp_200_cr_in_idx_1_dst_rsize"], tp_names["tp_200_cr_in_idx_1_src_asize"]]

    return sub_tiling_params


def _get_tiling_params_func(args):
    """
    get tiling parameters function
    """

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size = args

    (in_shape_new, out_shape_new,
     in_format_new, out_format_new) = _renew_input_output_shape_format(in_shape, out_shape, src_format, dst_format)
    args_get_tp = in_shape_new, out_shape_new, in_format_new, out_format_new, block_elem_cnt, ub_size
    tiling_params = _tiling_params_negative(args_get_tp)

    return tiling_params


def _twice_vnchwconv_invert(args):
    """
    do dhc to cdh transform by twice vnchwconv
    """

    tik_inst, src_ub, tmp_ub_offset, ub_offset, c_plp_size, cr_pln_size, ele_per_block = args
    size_factor = tdc.get_dtype_factor(src_ub.dtype)
    src_ub_fp16 = src_ub.reinterpret_cast_to("float16")
    tmp_ub_offset_fp16 = tmp_ub_offset * size_factor
    ub_offset_fp16 = ub_offset * size_factor

    with tik_inst.new_stmt_scope():
        cr_pln_align_c0_size = tdc.ceil_fill(cr_pln_size, ele_per_block)
        vnc_col_len = cr_pln_align_c0_size * tdc.C0_16 * size_factor

        with tik_inst.for_range(0, c_plp_size) as c1_idx:
            c1_offset = c1_idx * vnc_col_len * tdc.VNC_LINES
            # do c1dhwc -> dhwcc1
            src_addr_list = [src_ub_fp16[tmp_ub_offset_fp16 + vnc_col_len * i + c1_offset]
                             for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [src_ub_fp16[ub_offset_fp16 + tdc.C0_16 * i + c1_offset] for i in tdc.ADDR_IDX_LIST]
            repeat_cnt = vnc_col_len // tdc.C0_16
            with tik_inst.new_stmt_scope():
                src_stride = tik_inst.Scalar(dtype="int32")
                dst_stride = tik_inst.Scalar(dtype="int32")
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(1)
                    dst_stride.set_as(16)
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

            with tik_inst.for_range(0, tdc.C0_16) as c_idx:  # do dhwcc1 -> cdhwc1
                tik_inst.data_move(src_ub_fp16[tmp_ub_offset_fp16 +
                                               c_idx * cr_pln_align_c0_size * size_factor * tdc.VNC_LINES],
                                   src_ub_fp16[ub_offset_fp16 + c_idx * size_factor * tdc.VNC_LINES + c1_offset],
                                   0, cr_pln_align_c0_size, 1 * size_factor, (tdc.C0_16 - 1) * size_factor, 0)

            # do cdhwc1 -> c1cdhw
            vnc_row_size = vnc_col_len
            src_addr_list = [src_ub_fp16[tmp_ub_offset_fp16 + tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [src_ub_fp16[ub_offset_fp16 + vnc_row_size * i + c1_offset]
                             for i in tdc.ADDR_IDX_LIST]
            repeat_cnt = vnc_row_size // tdc.C0_16
            with tik_inst.new_stmt_scope():
                src_stride = tik_inst.Scalar(dtype="int32")
                dst_stride = tik_inst.Scalar(dtype="int32")
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(16)
                    dst_stride.set_as(1)
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _update_input_offset(args, idx_args):
    """
    count input gm offset
    """

    (c1_lp_idx, src_c1_lp_step_in, core_step_in,
    cl_in_idx_0_size, cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize,
    cl_in_idx_1_size, cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize,
    cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize,
    cr_in_idx_1_size, cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize) = args
    in_cr_idx, c1_idx, c1_step_in, cl_beg_idx = idx_args

    in_offset = (c1_idx * c1_step_in + c1_lp_idx * src_c1_lp_step_in +
                 cl_beg_idx // cl_in_idx_0_dst_rsize % cl_in_idx_0_size * cl_in_idx_0_src_asize +
                 cl_beg_idx // cl_in_idx_1_dst_rsize % cl_in_idx_1_size * cl_in_idx_1_src_asize +
                 in_cr_idx // cr_in_idx_0_dst_rsize % cr_in_idx_0_size * cr_in_idx_0_src_asize +
                 in_cr_idx // cr_in_idx_1_dst_rsize % cr_in_idx_1_size * cr_in_idx_1_src_asize + core_step_in)

    return in_offset


def _update_output_offset(args):
    """
    count output gm offset
    """

    c_lp_idx, src_c_lp_step_out, core_step_out = args

    out_offset = (c_lp_idx * src_c_lp_step_out + core_step_out)

    return out_offset


# pylint: disable=unused-variable
def _copy_data_in_0(tik_args, in_offset_args):
    """
    copy data from gm to ub for transform such as nc1hwc0 -> nchw
    """

    (tik_inst, src_in_gm, src_ub, tmp_ub_offset, cr_pln_size, cr_idx_beg,
     c_plp_size, c_step_in, cl_plp_size, cl_idx_beg, ele_per_block, cr_0_size) = tik_args

    with tik_inst.new_stmt_scope():
        in_ub_offset = tik_inst.Scalar(name="in_ub_offset")
        in_gm_offset = tik_inst.Scalar(name="in_gm_offset")
        in_cr_idx = tik_inst.Scalar(name="in_cr_idx")

        with tik_inst.if_scope(cr_pln_size % ele_per_block > 0):
            cr_pln_align_c0_floor_size = cr_pln_size // ele_per_block * ele_per_block
            cr_pln_align_c0_top_size = tdc.ceil_fill(cr_pln_size, ele_per_block)
            with tik_inst.for_range(0, c_plp_size) as c1_idx:
                with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                    with tik_inst.for_range(0, cr_pln_align_c0_floor_size) as cr_idx:
                        cr_idx_1 = (cr_idx + cr_idx_beg)
                        with tik_inst.if_scope(tik.any(cr_idx_1 % cr_0_size == 0, cr_idx == 0)):
                            in_cr_idx.set_as(cr_idx_1)
                            idx_args = in_cr_idx, c1_idx, c_step_in, cl_idx + cl_idx_beg
                            in_gm_offset.set_as(_update_input_offset(in_offset_args, idx_args))
                            in_ub_offset.set_as(tdc.C0_16 *
                                                (cr_idx + (c1_idx * tdc.VNC_LINES + cl_idx) * cr_pln_align_c0_top_size))
                        with tik_inst.if_scope(tik.any((cr_idx_1 + 1) % cr_0_size == 0,
                                                       cr_idx == cr_pln_align_c0_floor_size - 1)):
                            tik_inst.data_move(src_ub[in_ub_offset], src_in_gm[in_gm_offset],
                                               0, 1, (cr_idx_1 + 1 - in_cr_idx) * tdc.C0_16 // ele_per_block, 0, 0)

            with tik_inst.for_range(0, c_plp_size) as c1_idx:  # for tail
                with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                    with tik_inst.for_range(0, ele_per_block) as cr_idx:
                        cr_idx_1 = (cr_idx + cr_idx_beg + cr_pln_size - ele_per_block)
                        with tik_inst.if_scope(tik.any(cr_idx_1 % cr_0_size == 0, cr_idx == 0)):
                            in_cr_idx.set_as(cr_idx_1)
                            idx_args = in_cr_idx, c1_idx, c_step_in, cl_idx + cl_idx_beg
                            in_gm_offset.set_as(_update_input_offset(in_offset_args, idx_args))
                            in_ub_offset.set_as(tdc.C0_16 *
                                                (cr_pln_align_c0_floor_size +
                                                 cr_idx + (c1_idx * tdc.VNC_LINES + cl_idx) * cr_pln_align_c0_top_size))
                        with tik_inst.if_scope(tik.any((cr_idx_1 + 1) % cr_0_size == 0, cr_idx == ele_per_block - 1)):
                            tik_inst.data_move(src_ub[in_ub_offset], src_in_gm[in_gm_offset],
                                               0, 1, (cr_idx_1 + 1 - in_cr_idx) * tdc.C0_16 // ele_per_block, 0, 0)

        with tik_inst.else_scope():
            with tik_inst.for_range(0, c_plp_size) as c1_idx:
                with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                    with tik_inst.for_range(0, cr_pln_size) as cr_idx:
                        cr_idx_1 = (cr_idx + cr_idx_beg)
                        with tik_inst.if_scope(tik.any(cr_idx_1 % cr_0_size == 0, cr_idx == 0)):
                            in_cr_idx.set_as(cr_idx_1)
                            idx_args = in_cr_idx, c1_idx, c_step_in, cl_idx + cl_idx_beg
                            in_gm_offset.set_as(_update_input_offset(in_offset_args, idx_args))
                            in_ub_offset.set_as(tdc.C0_16 * (cr_idx + (c1_idx * tdc.VNC_LINES + cl_idx) * cr_pln_size))
                        with tik_inst.if_scope(tik.any((cr_idx_1 + 1) % cr_0_size == 0, cr_idx == cr_pln_size - 1)):
                            tik_inst.data_move(src_ub[in_ub_offset], src_in_gm[in_gm_offset],
                                               0, 1, (cr_idx_1 + 1 - in_cr_idx) * tdc.C0_16 // ele_per_block, 0, 0)


# pylint: disable=unused-variable
def _copy_data_in_1(tik_args, in_offset_args):
    """
    copy data from gm to ub for transform such as c1hwnonic0 -> nchw
    """

    (tik_inst, src_in_gm, src_ub, tmp_ub_offset, cr_pln_size, cr_idx_beg,
     c_plp_size, c_step_in, cl_plp_size, cl_idx_beg, ele_per_block, cr_0_size) = tik_args

    with tik_inst.new_stmt_scope():
        with tik_inst.if_scope(cr_pln_size % ele_per_block > 0):
            cr_pln_align_c0_size = cr_pln_size // ele_per_block * ele_per_block
            with tik_inst.for_range(0, c_plp_size) as c1_idx:
                with tik_inst.for_range(0, cr_pln_align_c0_size) as cr_idx:
                    in_cr_idx = cr_idx + cr_idx_beg
                    idx_args = in_cr_idx, c1_idx, c_step_in, cl_idx_beg
                    in_gm_offset = _update_input_offset(in_offset_args, idx_args)
                    in_ub_offset = (tmp_ub_offset + tdc.C0_16 *
                                    (cr_idx + c1_idx * tdc.ceil_fill(cr_pln_size, ele_per_block) * tdc.VNC_LINES))
                    tik_inst.data_move(src_ub[0], src_in_gm[in_gm_offset],
                                       0, 1, cl_plp_size * tdc.C0_16 // ele_per_block, 0, 0)
                    tik_inst.data_move(src_ub[in_ub_offset], src_ub[0],
                                       0, cl_plp_size, tdc.C0_16 // ele_per_block,
                                       0, (tdc.ceil_fill(cr_pln_size, ele_per_block) - 1) * tdc.C0_16 // ele_per_block)

            with tik_inst.for_range(0, c_plp_size) as c1_idx:  # for tail
                with tik_inst.for_range(0, ele_per_block) as cr_idx:
                    in_cr_idx = cr_idx + cr_idx_beg + cr_pln_size - ele_per_block
                    idx_args = in_cr_idx, c1_idx, c_step_in, cl_idx_beg
                    in_gm_offset = _update_input_offset(in_offset_args, idx_args)
                    in_ub_offset = (tmp_ub_offset + tdc.C0_16 *
                                    (cr_idx + cr_pln_align_c0_size +
                                     c1_idx * tdc.ceil_fill(cr_pln_size, ele_per_block) * tdc.VNC_LINES))
                    tik_inst.data_move(src_ub[0], src_in_gm[in_gm_offset],
                                       0, 1, cl_plp_size * tdc.C0_16 // ele_per_block, 0, 0)
                    tik_inst.data_move(src_ub[in_ub_offset], src_ub[0],
                                       0, cl_plp_size, tdc.C0_16 // ele_per_block,
                                       0, (tdc.ceil_fill(cr_pln_size, ele_per_block) - 1) * tdc.C0_16 // ele_per_block)

        with tik_inst.else_scope():
            with tik_inst.for_range(0, c_plp_size) as c1_idx:
                with tik_inst.for_range(0, cr_pln_size) as cr_idx:
                    in_cr_idx = cr_idx + cr_idx_beg
                    idx_args = in_cr_idx, c1_idx, c_step_in, cl_idx_beg
                    in_gm_offset = _update_input_offset(in_offset_args, idx_args)
                    in_ub_offset = tdc.C0_16 * (cr_idx + c1_idx * tdc.VNC_LINES * cr_pln_size) + tmp_ub_offset
                    tik_inst.data_move(src_ub[0], src_in_gm[in_gm_offset],
                                       0, 1, cl_plp_size * tdc.C0_16 // ele_per_block, 0, 0)
                    tik_inst.data_move(src_ub[in_ub_offset], src_ub[0],
                                       0, cl_plp_size, tdc.C0_16 // ele_per_block,
                                       0, (cr_pln_size - 1) * tdc.C0_16 // ele_per_block)


# pylint: disable=unused-variable
def _copy_data_out(copy_out_args):
    """
    copy data from ub to gm
    """

    (tik_inst, dst_out_gm, out_ub, src_cr_step_out, src_c_step_out,
     src_cl_step_out, cl_beg_idx, cr_beg_idx, cr_pln_size, c_plp_size,
     cl_plp_size, ele_per_block, is_last_c1, c_mod_c0, mc_pos, block_idx, used_core_cnt) = copy_out_args

    cr_pln_align_c0_size = tdc.ceil_fill(cr_pln_size, ele_per_block)
    with tik_inst.new_stmt_scope():
        c_lp_cnt = tik_inst.Scalar(dtype="int32", name="c_lp_cnt")
        with tik_inst.for_range(0, c_plp_size) as c1_idx:
            with tik_inst.if_scope(tik.any(tik.all(c1_idx == c_plp_size - 1, is_last_c1 > 0, c_mod_c0 > 0, mc_pos != 1),
                                           tik.all(c1_idx == c_plp_size - 1, is_last_c1 > 0, c_mod_c0 > 0, mc_pos == 1,
                                                   block_idx == used_core_cnt - 1))):
                c_lp_cnt.set_as(c_mod_c0)
            with tik_inst.else_scope():
                c_lp_cnt.set_as(tdc.C0_16)

            with tik_inst.if_scope(cr_pln_size % ele_per_block > 0):
                with tik_inst.for_range(0, c_lp_cnt) as c_idx:
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        out_offset_gm = (src_c_step_out * (c_idx + c1_idx * tdc.C0_16) +
                                         (cl_idx + cl_beg_idx) * src_cl_step_out + cr_beg_idx)
                        out_offset_ub = cr_pln_align_c0_size * (c_idx + (c1_idx * tdc.VNC_LINES + cl_idx) * tdc.C0_16)
                        tik_inst.data_move(dst_out_gm[out_offset_gm], out_ub[out_offset_ub],
                                           0, 1, cr_pln_size // ele_per_block, 0, 0)
                with tik_inst.for_range(0, c_lp_cnt) as c_idx:  # for tail
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        out_offset_gm = (src_c_step_out * (c_idx + c1_idx * tdc.C0_16) + (cl_idx + cl_beg_idx) *
                                         src_cl_step_out + cr_beg_idx + cr_pln_size - ele_per_block)
                        out_offset_ub = (cr_pln_size // ele_per_block * ele_per_block + cr_pln_align_c0_size *
                                         (c_idx + (c1_idx * tdc.VNC_LINES + cl_idx) * tdc.C0_16))
                        tik_inst.data_move(dst_out_gm[out_offset_gm], out_ub[out_offset_ub], 0, 1, 1, 0, 0)

            with tik_inst.else_scope():
                with tik_inst.for_range(0, c_lp_cnt) as c_idx:
                    with tik_inst.for_range(0, cl_plp_size) as cl_idx:
                        out_offset_gm = (src_c_step_out * (c_idx + c1_idx * tdc.C0_16) +
                                         (cl_idx + cl_beg_idx) * src_cl_step_out + cr_beg_idx)
                        out_offset_ub = cr_pln_size * (c_idx + (c1_idx * tdc.VNC_LINES + cl_idx) * tdc.C0_16)
                        tik_inst.data_move(dst_out_gm[out_offset_gm], out_ub[out_offset_ub],
                                           0, 1, cr_pln_size // ele_per_block, 0, 0)


def _func_transform_200(tensor_args, tp_args):
    """
    transform function for tiling mode 200
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, ele_per_block = tensor_args
    (ub_offset, tmp_ub_offset, mc_pos, used_core, core_step_in, core_step_out, nlc_cr_lp_cnt, nlc_c_lp_cnt,
     nlc_cl_lp_cnt, nlc_cr_left, nlc_c_left, nlc_cl_left, lc_cr_lp_cnt, lc_c_lp_cnt, lc_cl_lp_cnt, lc_cr_left,
     lc_c_left, lc_cl_left, src_cr_lp_unit, src_c_lp_unit, src_cl_lp_unit, src_cr_step_out, src_c_step_in,
     src_c_step_out, src_cl_step_out, src_c_lp_step_in, src_c_lp_step_out, c_mod_c0, is_mc_cr, is_mc_cl, src_2_dst_flag,
     vnc_line_size, cl_in_idx_0_size, cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize, cl_in_idx_1_size,
     cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize, cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize,
     cr_in_idx_1_size, cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize) = tp_args

    def _inner_func(tiling_args):
        cr_lp_cnt, lb_cr_lp_cnt, cr_left, c_lp_cnt, c_left, cl_lp_cnt, lb_cl_lp_cnt, cl_left = tiling_args
        cr_pln_size = tik_inst.Scalar(name="cr_pln_size")
        c_plp_size = tik_inst.Scalar(name="c_plp_size")
        cl_plp_size = tik_inst.Scalar(name="cl_plp_size")
        is_last_c1 = tik_inst.Scalar(name="is_last_c1")
        is_cr_back = tik_inst.Scalar(name="is_cr_back")
        cr_lp_beg = is_mc_cr * block_idx * cr_lp_cnt
        cr_lp_end = cr_lp_beg + lb_cr_lp_cnt
        with tik_inst.for_range(cr_lp_beg, cr_lp_end) as cr_lp_idx:  # axis C-RIGHT
            with tik_inst.if_scope(tik.any(cr_lp_idx != cr_lp_end - 1, cr_left == 0)):
                # cr_lp_idx for last second core and lc_cr_left for last core
                with tik_inst.if_scope(tik.all(block_idx == used_core - 2, cr_lp_idx == cr_lp_end - 1,
                                               lc_cr_left > 0, lc_cr_left < ele_per_block)):
                    cr_pln_size.set_as(src_cr_lp_unit - ele_per_block)
                with tik_inst.else_scope():
                    cr_pln_size.set_as(src_cr_lp_unit)
                is_cr_back.set_as(0)

            with tik_inst.else_scope():
                with tik_inst.if_scope(tik.all(cr_left > 0, cr_left < ele_per_block)):
                    cr_pln_size.set_as(cr_left + ele_per_block)
                    is_cr_back.set_as(1)
                with tik_inst.else_scope():
                    cr_pln_size.set_as(cr_left)
                    is_cr_back.set_as(0)

            with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:  # axis C
                with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                    c_plp_size.set_as(src_c_lp_unit)
                    with tik_inst.if_scope(c_lp_idx == c_lp_cnt - 1):
                        is_last_c1.set_as(1)
                    with tik_inst.else_scope():
                        is_last_c1.set_as(0)

                with tik_inst.else_scope():
                    c_plp_size.set_as(lc_c_left)
                    is_last_c1.set_as(1)

                cl_lp_beg = is_mc_cl * block_idx * cl_lp_cnt
                cl_lp_end = cl_lp_beg + lb_cl_lp_cnt
                cr_backend = is_cr_back * ele_per_block
                with tik_inst.for_range(cl_lp_beg, cl_lp_end) as cl_lp_idx:  # axis C-LEFT
                    with tik_inst.if_scope(tik.any(cl_lp_idx != cl_lp_end - 1, cl_left == 0)):
                        cl_plp_size.set_as(src_cl_lp_unit)
                    with tik_inst.else_scope():
                        cl_plp_size.set_as(cl_left)

                    in_offset_args = (c_lp_idx, src_c_lp_step_in, block_idx * core_step_in,
                                      cl_in_idx_0_size, cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize,
                                      cl_in_idx_1_size, cl_in_idx_1_dst_rsize, cl_in_idx_1_src_asize,
                                      cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize,
                                      cr_in_idx_1_size, cr_in_idx_1_dst_rsize, cr_in_idx_1_src_asize)
                    copy_in_args = (tik_inst, src_in_gm, src_ub, tmp_ub_offset, cr_pln_size,
                                    cr_lp_idx * src_cr_lp_unit - cr_backend, c_plp_size, src_c_step_in, cl_plp_size,
                                    cl_lp_idx * src_cl_lp_unit, ele_per_block, cr_in_idx_0_size)
                    with tik_inst.if_scope(src_2_dst_flag == 0):
                        _copy_data_in_0(copy_in_args, in_offset_args)
                    with tik_inst.else_scope():
                        _copy_data_in_1(copy_in_args, in_offset_args)

                    vnc_args = (tik_inst, src_ub, tmp_ub_offset, ub_offset, c_plp_size, cr_pln_size, ele_per_block)
                    _twice_vnchwconv_invert(vnc_args)
                    out_gm_args = c_lp_idx, src_c_lp_step_out, block_idx * core_step_out
                    out_gm_offset = _update_output_offset(out_gm_args) - cr_backend
                    copy_out_args = (tik_inst, dst_out_gm[out_gm_offset], src_ub[ub_offset], src_cr_step_out,
                                     src_c_step_out, src_cl_step_out, cl_lp_idx * src_cl_lp_unit,
                                     cr_lp_idx * src_cr_lp_unit, cr_pln_size, c_plp_size, cl_plp_size,
                                     ele_per_block, is_last_c1, c_mod_c0, mc_pos, block_idx, used_core)
                    _copy_data_out(copy_out_args)

    with tik_inst.if_scope(block_idx != used_core - 1):
        nlc_args = nlc_cr_lp_cnt, nlc_cr_lp_cnt, nlc_cr_left, \
                   nlc_c_lp_cnt, nlc_c_left, nlc_cl_lp_cnt, nlc_cl_lp_cnt, nlc_cl_left
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = nlc_cr_lp_cnt, lc_cr_lp_cnt, lc_cr_left, \
                  lc_c_lp_cnt, lc_c_left, nlc_cl_lp_cnt, lc_cl_lp_cnt, lc_cl_left
        _inner_func(lc_args)


def trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name="trans_data_negative_target_ntc"):
    """
    negative transform for last dimension of target format is c

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        dtype of output should be same type as input
    src_format: str
        source data format, can be ND, NC1HWC0 etc.
    dst_format: str
        target data format, can be NHWC etc.
    groups: int
        groups count for conv case, default value is 1
    kernel_name : str
        kernel name, default value is "trans_data_negative_target_ntc"

    Returns
    -------
    None
    """

    src_format = src_format.upper()
    dst_format = dst_format.upper()
    in_shape = list(src.get("shape"))
    out_shape = list(dst.get("shape"))
    in_dtype = src.get("dtype").lower()
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1)
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)

    # get tiling parameters
    args = in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size
    tiling_params = _get_tiling_params_func(args)

    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "src_in_gm")
    dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm", is_atomic_add=True)
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tik.scope_ubuf, "total_ub")

    tiling_mode = tiling_params[0]
    used_core_cnt = tiling_params[4]
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            with tik_inst.if_scope(tiling_mode == 200):
                tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt]
                tp_args = tiling_params[1:]
                _func_transform_200(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[src_in_gm], outputs=[dst_out_gm])
