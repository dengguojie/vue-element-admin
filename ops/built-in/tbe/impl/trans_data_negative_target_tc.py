# Copyright 2020 Huawei Technologies Co., Ltd
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
trans_data_negative_target_tc
"""

from te import tik
from impl import trans_data_common_func as tdc


# used for scalar
PAD_IDX_LIST = (0, 1, 2)
# frame up levels
FRAME_LEVEL = 3


# pylint: disable=too-many-locals
def _renew_input_output_shape_format(in_shape, out_shape, in_format, out_format):
    """
    renew shape and format to adapt tiling process
    """

    in_format_upper = in_format.upper()
    out_format_upper = out_format.upper()

    if in_format_upper == "NC1HWC0" and out_format_upper == "NHWC":
        in_format_new = "NCHT"
        out_format_new = "NHC"
        axis_n, axis_c1, axis_h, axis_w, axis_c0 = in_shape
        axis_c = out_shape[-1]
        in_shape_new = [axis_n] + [axis_c1] + [axis_h * axis_w] + [axis_c0]
        out_shape_new = [axis_n] + [axis_h * axis_w] + [axis_c]

        return [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

    if in_format_upper == "FRACTAL_NZ" and out_format_upper == "ND":
        in_format_new = "HCNT"
        out_format_new = "HNC"
        if len(out_shape) == 1:
            axis_h, axis_n, axis_c = 1, 1, out_shape[0]
        elif len(out_shape) == 2:
            axis_h, axis_n, axis_c = 1, out_shape[0], out_shape[1]
        else:
            axis_h, axis_n, axis_c = tdc.get_shape_size(out_shape[:-2]), out_shape[-2], out_shape[-1]
        axis_c0 = in_shape[-1]
        axis_c1, axis_no, axis_ni = tdc.ceil_div(axis_c, axis_c0), tdc.ceil_div(axis_n, tdc.NI_16), tdc.NI_16
        in_shape_new = [axis_h] + [axis_c1] + [axis_no * axis_ni] + [axis_c0]
        out_shape_new = [axis_h] + [axis_n] + [axis_c]

        return [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

    if in_format_upper == "FRACTAL_Z_3D" and out_format_upper == "NDHWC":
        in_format_new = "DCHNT"
        out_format_new = "NDHC"
        axis_dc1hw, axis_no, axis_ni, axis_c0 = in_shape
        axis_n, axis_d, axis_h, axis_w, axis_c = out_shape
        axis_c1 = axis_dc1hw // (axis_d * axis_h * axis_w)
        in_shape_new = [axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        out_shape_new = [axis_n] + [axis_d] + [axis_h * axis_w] + [axis_c]

        return [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

    return [in_shape, out_shape] + [in_format, out_format]


# pylint: disable=too-many-statements
def _get_mc_info_negative(r2nd_args, c1_args, left_args):
    """
    get multiple core axis position for negative transform
    """

    r2nd_lp_cnt, r2nd_lp_step_in, r2nd_lp_step_out, r2nd_size, r2nd_lp_unit = r2nd_args
    c1_lp_cnt, c1_lp_step_in, c1_lp_step_out, c1_size, c1_lp_unit = c1_args
    left_lp_cnt = left_args

    tmp_full_lp_cnt_r2nd = tdc.CORE_DIM_NUM if tdc.floor_div(r2nd_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_r2nd = r2nd_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_r2nd == 0:
        tmp_full_lp_cnt_r2nd += tdc.CORE_DIM_NUM
    full_lp_cnt_r2nd = tmp_full_lp_cnt_r2nd + reminder_lp_cnt_r2nd

    tmp_full_lp_cnt_c1 = tdc.CORE_DIM_NUM if tdc.floor_div(c1_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_c1 = c1_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_c1 == 0:
        tmp_full_lp_cnt_c1 += tdc.CORE_DIM_NUM
    full_lp_cnt_c1 = tmp_full_lp_cnt_c1 + reminder_lp_cnt_c1

    tmp_full_lp_cnt_left = tdc.CORE_DIM_NUM if tdc.floor_div(left_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_left = left_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_left == 0:
        tmp_full_lp_cnt_left += tdc.CORE_DIM_NUM
    full_lp_cnt_left = tmp_full_lp_cnt_left + reminder_lp_cnt_left

    lp_cnt_list = (full_lp_cnt_left, full_lp_cnt_c1, full_lp_cnt_r2nd)
    if lp_cnt_list.index(max(lp_cnt_list)) == 0:
        tp_201_mc_flag = 1
        tp_201_used_core_cnt = tdc.ceil_div(left_lp_cnt, tdc.ceil_div(left_lp_cnt, tdc.CORE_DIM_NUM))
        tp_201_nlc_left_lp_cnt = tdc.ceil_div(left_lp_cnt, tp_201_used_core_cnt)
        tp_201_lc_left_lp_cnt = left_lp_cnt - tp_201_nlc_left_lp_cnt * (tp_201_used_core_cnt - 1)
        tp_201_core_step_in = 0
        tp_201_core_step_out = 0
        tp_201_nlc_c1_lp_cnt = c1_lp_cnt
        tp_201_lc_c1_lp_cnt = c1_lp_cnt
        tp_201_nlc_c1_left = c1_size % c1_lp_unit
        tp_201_lc_c1_left = c1_size % c1_lp_unit
        tp_201_nlc_r2nd_lp_cnt = r2nd_lp_cnt
        tp_201_lc_r2nd_lp_cnt = r2nd_lp_cnt
        tp_201_nlc_r2nd_left = r2nd_size % r2nd_lp_unit
        tp_201_lc_r2nd_left = r2nd_size % r2nd_lp_unit
    else:
        tp_201_mc_flag = 0
        tp_201_nlc_left_lp_cnt = left_lp_cnt
        tp_201_lc_left_lp_cnt = left_lp_cnt
        if lp_cnt_list.index(max(lp_cnt_list)) == 1:
            tp_201_used_core_cnt = tdc.ceil_div(c1_lp_cnt, tdc.ceil_div(c1_lp_cnt, tdc.CORE_DIM_NUM))
            tp_201_nlc_c1_lp_cnt = tdc.ceil_div(c1_lp_cnt, tp_201_used_core_cnt)
            tp_201_lc_c1_lp_cnt = c1_lp_cnt - (tp_201_used_core_cnt - 1) * tp_201_nlc_c1_lp_cnt
            tp_201_nlc_c1_left = 0
            tp_201_lc_c1_left = c1_size % c1_lp_unit
            tp_201_core_step_in = tp_201_nlc_c1_lp_cnt * c1_lp_step_in
            tp_201_core_step_out = tp_201_nlc_c1_lp_cnt * c1_lp_step_out
            tp_201_nlc_r2nd_lp_cnt = r2nd_lp_cnt
            tp_201_lc_r2nd_lp_cnt = r2nd_lp_cnt
            tp_201_nlc_r2nd_left = r2nd_size % r2nd_lp_unit
            tp_201_lc_r2nd_left = r2nd_size % r2nd_lp_unit
        else:
            tp_201_used_core_cnt = tdc.ceil_div(r2nd_lp_cnt, tdc.ceil_div(r2nd_lp_cnt, tdc.CORE_DIM_NUM))
            tp_201_nlc_r2nd_lp_cnt = tdc.ceil_div(r2nd_lp_cnt, tp_201_used_core_cnt)
            tp_201_lc_r2nd_lp_cnt = r2nd_lp_cnt - (tp_201_used_core_cnt - 1) * tp_201_nlc_r2nd_lp_cnt
            tp_201_nlc_r2nd_left = 0
            tp_201_lc_r2nd_left = r2nd_size % r2nd_lp_unit
            tp_201_core_step_in = tp_201_nlc_r2nd_lp_cnt * r2nd_lp_step_in
            tp_201_core_step_out = tp_201_nlc_r2nd_lp_cnt * r2nd_lp_step_out
            tp_201_nlc_c1_lp_cnt = c1_lp_cnt
            tp_201_lc_c1_lp_cnt = c1_lp_cnt
            tp_201_nlc_c1_left = c1_size % c1_lp_unit
            tp_201_lc_c1_left = c1_size % c1_lp_unit

    return [tp_201_mc_flag, tp_201_used_core_cnt] + \
           [tp_201_nlc_r2nd_lp_cnt, tp_201_nlc_c1_lp_cnt,
            tp_201_nlc_left_lp_cnt, tp_201_nlc_r2nd_left, tp_201_nlc_c1_left] + \
           [tp_201_lc_r2nd_lp_cnt, tp_201_lc_c1_lp_cnt, tp_201_lc_left_lp_cnt,
            tp_201_lc_r2nd_left, tp_201_lc_c1_left] + [tp_201_core_step_in, tp_201_core_step_out]


# pylint: disable=redefined-builtin, unbalanced-tuple-unpacking
def _tiling_params_negative(args):
    """
    calculate real tiling params for negative transform and last axis of target format is c
    """

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size = args
    c0_len = in_shape[-1]  # axis c0
    axis_src_c1_size = in_shape[src_format.index("C")]

    # get tiling params for using vnchwconv
    half_ub_size = ub_size // 2
    tp_201_tiling_mode = 201
    tp_201_ub_offset = half_ub_size // block_elem_cnt * block_elem_cnt

    # axis -2 tiling parameters
    axis_src_r2nd_size = out_shape[dst_format.index(src_format[-2])]
    tp_201_src_r2nd_lp_unit = tdc.NI_16
    vnc_line_size = tp_201_ub_offset // tdc.VNC_LINES
    if axis_src_c1_size < 8:  # c size is small, move in more r2nd per time
        tp_201_src_r2nd_lp_unit = vnc_line_size // c0_len // axis_src_c1_size // block_elem_cnt * block_elem_cnt
    src_r2nd_lp_cnt = tdc.ceil_div(axis_src_r2nd_size, tp_201_src_r2nd_lp_unit)
    tp_201_src_r2nd_lp_step_in = tdc.get_shape_size([tp_201_src_r2nd_lp_unit, c0_len])
    tp_201_src_r2nd_lp_step_out = tdc.get_shape_size(out_shape[dst_format.index(src_format[-2]) + 1:])

    # axis c1 tiling parameters
    tp_201_src_c1_lp_unit = axis_src_c1_size if axis_src_c1_size < 8 else 8  # in order to output 8 block per time
    src_c1_lp_cnt = tdc.ceil_div(axis_src_c1_size, tp_201_src_c1_lp_unit)
    src_c1_left = axis_src_c1_size % tp_201_src_c1_lp_unit
    tp_201_src_c1_lp_step_in = tdc.get_shape_size([tp_201_src_c1_lp_unit] + in_shape[src_format.index("C") + 1:])
    tp_201_src_c1_lp_step_out = tdc.get_shape_size([tp_201_src_c1_lp_unit, c0_len])
    tp_201_src_c1_step_in = tdc.get_shape_size(in_shape[src_format.index("C") + 1:])
    if axis_src_c1_size < tp_201_src_c1_lp_unit:
        tp_201_pl_dst_c_cnt = tp_201_src_c1_lp_unit // src_c1_left  # c count in one vnc line
    else:
        tp_201_pl_dst_c_cnt = 1
    tp_201_c_mod_c0 = out_shape[dst_format.index("C")] % c0_len

    # left axises
    tmp_left_src_format = src_format.replace(src_format[-2:], "").replace("C", "")
    tmp_left_in_shape = []
    axis_src_left_axis_size = 1
    for idx, chr in enumerate(tmp_left_src_format):
        axis_src_left_axis_size *= in_shape[src_format.index(chr)]
        tmp_left_in_shape.append(in_shape[src_format.index(chr)])

    # mulitple core parameters
    r2nd_args = src_r2nd_lp_cnt, tp_201_src_r2nd_lp_step_in, \
                tp_201_src_r2nd_lp_step_out * tp_201_src_r2nd_lp_unit, axis_src_r2nd_size, tp_201_src_r2nd_lp_unit
    c1_args = src_c1_lp_cnt, tp_201_src_c1_lp_step_in, \
              tp_201_src_c1_lp_step_out, axis_src_c1_size, tp_201_src_c1_lp_unit
    left_args = axis_src_left_axis_size
    tp_201_mc_flag, tp_201_used_core_cnt, tp_201_nlc_r2nd_lp_cnt, tp_201_nlc_c1_lp_cnt, tp_201_nlc_left_lp_cnt, \
    tp_201_nlc_r2nd_left, tp_201_nlc_c1_left, tp_201_lc_r2nd_lp_cnt, tp_201_lc_c1_lp_cnt, tp_201_lc_left_lp_cnt, \
    tp_201_lc_r2nd_left, tp_201_lc_c1_left, tp_201_core_step_in, \
    tp_201_core_step_out = _get_mc_info_negative(r2nd_args, c1_args, left_args)

    # axis left parameters
    tp_names = locals()
    tmp_left_dst_format = dst_format.replace(src_format[-2], "").replace("C", "")
    tmp_left_out_shape = []
    for idx, chr in enumerate(tmp_left_dst_format):
        tmp_left_out_shape.append(out_shape[dst_format.index(chr)])
    tmp_left_out_shape.append(1)
    # count method: left_axis_size/dst_rsize%size*asize
    for idx, chr in enumerate(reversed(tmp_left_dst_format)):
        tp_names["tp_201_in_idx_" + str(idx) + "_size"] = in_shape[src_format.index(chr)]
        tp_names["tp_201_in_idx_" + str(idx) + "_dst_rsize"] = tdc.get_shape_size(tmp_left_out_shape[-1 - idx:])
        tp_names["tp_201_in_idx_" + str(idx) + "_src_asize"] = tdc.get_shape_size(in_shape[src_format.index(chr) + 1:])
        tp_names["tp_201_out_idx_" + str(idx) + "_size"] = out_shape[dst_format.index(chr)]
        tp_names["tp_201_out_idx_" + str(idx) + "_dst_rsize"] = tdc.get_shape_size(tmp_left_out_shape[-1 - idx:])
        tp_names["tp_201_out_idx_" + str(idx) + "_dst_asize"] = tdc.get_shape_size(
            out_shape[dst_format.index(chr) + 1:])
    # suppose there are 3 axises
    pad_axis_cnt = FRAME_LEVEL - len(tmp_left_dst_format)
    if pad_axis_cnt:
        for _, idx in enumerate(PAD_IDX_LIST[len(tmp_left_dst_format):]):
            tp_names["tp_201_in_idx_" + str(idx) + "_size"] = 1
            tp_names["tp_201_in_idx_" + str(idx) + "_dst_rsize"] = 1
            tp_names["tp_201_in_idx_" + str(idx) + "_src_asize"] = 0
            tp_names["tp_201_out_idx_" + str(idx) + "_size"] = 1
            tp_names["tp_201_out_idx_" + str(idx) + "_dst_rsize"] = 1
            tp_names["tp_201_out_idx_" + str(idx) + "_dst_asize"] = 0

    # vnchwconv tiling parameters
    if src_format[-2] == dst_format[-2]:  # such as NC1HWC0 -> NHWC
        tp_201_src_2_dst_flag = 0
    else:  # such as DC1HWNoNiC0 -> NDHWC
        tp_201_src_2_dst_flag = 1

    sub_tiling_params = [tp_201_tiling_mode, tp_201_ub_offset, tp_201_mc_flag, tp_201_used_core_cnt,
                         tp_201_core_step_in, tp_201_core_step_out, tp_201_nlc_r2nd_lp_cnt, tp_201_nlc_c1_lp_cnt,
                         tp_201_nlc_left_lp_cnt, tp_201_nlc_r2nd_left, tp_201_nlc_c1_left, tp_201_lc_r2nd_lp_cnt,
                         tp_201_lc_c1_lp_cnt, tp_201_lc_left_lp_cnt, tp_201_lc_r2nd_left, tp_201_lc_c1_left,
                         tp_201_src_r2nd_lp_unit, tp_201_src_r2nd_lp_step_in, tp_201_src_r2nd_lp_step_out,
                         tp_201_src_c1_step_in, tp_201_src_c1_lp_unit, tp_201_src_c1_lp_step_in,
                         tp_201_src_c1_lp_step_out, tp_201_pl_dst_c_cnt, tp_201_c_mod_c0,
                         tp_names["tp_201_in_idx_0_size"], tp_names["tp_201_in_idx_0_dst_rsize"],
                         tp_names["tp_201_in_idx_0_src_asize"], tp_names["tp_201_in_idx_1_size"],
                         tp_names["tp_201_in_idx_1_dst_rsize"], tp_names["tp_201_in_idx_1_src_asize"],
                         tp_names["tp_201_in_idx_2_size"], tp_names["tp_201_in_idx_2_dst_rsize"],
                         tp_names["tp_201_in_idx_2_src_asize"], tp_names["tp_201_out_idx_0_size"],
                         tp_names["tp_201_out_idx_0_dst_rsize"], tp_names["tp_201_out_idx_0_dst_asize"],
                         tp_names["tp_201_out_idx_1_size"], tp_names["tp_201_out_idx_1_dst_rsize"],
                         tp_names["tp_201_out_idx_1_dst_asize"], tp_names["tp_201_out_idx_2_size"],
                         tp_names["tp_201_out_idx_2_dst_rsize"], tp_names["tp_201_out_idx_2_dst_asize"],
                         tp_201_src_2_dst_flag]

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


def _twice_vnchwconv_no_invert(args):
    """
    do ncht to nhct transform by twice vnchwconv
    """

    (tik_inst, src_ub, ub_offset, c1_lp_cnt, c1_pl_size,
     r2nd_pl_size, pln_dst_c_cnt, src_2_dst_flag, c_mod_c0, dtype_factor) = args

    src_ub_fp16 = src_ub.reinterpret_cast_to("float16")
    ub_offset = ub_offset * dtype_factor
    vnc_col_len = pln_dst_c_cnt * c1_pl_size * r2nd_pl_size * tdc.C0_16 * dtype_factor
    # do hcnt -> cnth
    src_addr_list = [src_ub_fp16[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
    dst_addr_list = [src_ub_fp16[ub_offset + tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
    repeat_cnt = pln_dst_c_cnt * c1_pl_size * r2nd_pl_size * dtype_factor
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(1)
            dst_stride.set_as(16)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

    # do cnth -> ncth
    with tik_inst.if_scope(src_2_dst_flag == 0):  # such as NC1HWC0 -> NHWC
        with tik_inst.for_range(0, pln_dst_c_cnt) as c_idx:
            dst_ub_offset = c_idx * c1_pl_size * r2nd_pl_size * tdc.C0_16 * tdc.VNC_LINES
            src_ub_offset = dst_ub_offset
            with tik_inst.for_range(0, c1_pl_size) as c1_idx:
                dst_ub_offset += c1_idx * tdc.C0_16 * dtype_factor * tdc.VNC_LINES
                src_ub_offset += c1_idx * r2nd_pl_size * tdc.C0_16 * dtype_factor * tdc.VNC_LINES
                tik_inst.data_move(src_ub_fp16[dst_ub_offset], src_ub_fp16[ub_offset + src_ub_offset], 0, r2nd_pl_size,
                                   tdc.C0_16 * dtype_factor, 0, (c1_pl_size - 1) * tdc.C0_16 * dtype_factor)
    with tik_inst.else_scope():  # such as DC1HWNoNic0 -> NDHWC
        with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx:
            dst_ub_offset = r2nd_idx * pln_dst_c_cnt * c1_pl_size * tdc.C0_16 * dtype_factor * tdc.VNC_LINES
            src_ub_offset = r2nd_idx * tdc.C0_16 * dtype_factor * tdc.VNC_LINES
            tik_inst.data_move(src_ub_fp16[dst_ub_offset], src_ub_fp16[ub_offset + src_ub_offset],
                               0, pln_dst_c_cnt * c1_pl_size, tdc.C0_16 * dtype_factor,
                               (r2nd_pl_size - 1) * tdc.C0_16 * dtype_factor, 0)

    # remove padded elements
    c_block_align_size = c1_pl_size * tdc.C0_16 * dtype_factor
    with tik_inst.if_scope(tik.all(c1_lp_cnt == 1, c_mod_c0 > 0)):
        left_c_size = ((c1_pl_size - 1) * tdc.C0_16 + c_mod_c0) * dtype_factor
        tik_inst.data_move(src_ub_fp16[left_c_size * tdc.VNC_LINES], src_ub_fp16[c_block_align_size * tdc.VNC_LINES],
                           0, pln_dst_c_cnt * r2nd_pl_size, left_c_size, c_block_align_size - left_c_size, 0)

    # do hctn -> nhct
    src_addr_list = [src_ub_fp16[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
    dst_addr_list = [src_ub_fp16[ub_offset + vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(16)
            dst_stride.set_as(1)
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _update_input_offset(args):
    """
    count input gm offset
    """

    r2nd_lp_idx, src_r2nd_lp_step_in, c1_lp_idx, src_c1_lp_step_in, \
    in_idx_0_size, in_idx_0_dst_rsize, in_idx_0_src_asize, \
    in_idx_1_size, in_idx_1_dst_rsize, in_idx_1_src_asize, \
    in_idx_2_size, in_idx_2_dst_rsize, in_idx_2_src_asize, ele_idx, core_step_in = args

    in_offset = (r2nd_lp_idx * src_r2nd_lp_step_in + c1_lp_idx * src_c1_lp_step_in +
                 ele_idx // in_idx_0_dst_rsize % in_idx_0_size * in_idx_0_src_asize +
                 ele_idx // in_idx_1_dst_rsize % in_idx_1_size * in_idx_1_src_asize +
                 ele_idx // in_idx_2_dst_rsize % in_idx_2_size * in_idx_2_src_asize + core_step_in)

    return in_offset


def _update_output_offset(args, ele_idx):
    """
    count output gm offset
    """

    r2nd_lp_idx, src_r2nd_lp_unit, src_r2nd_lp_step_out, c1_lp_idx, src_c1_lp_step_out, \
    out_idx_0_size, out_idx_0_dst_rsize, out_idx_0_dst_asize, \
    out_idx_1_size, out_idx_1_dst_rsize, out_idx_1_dst_asize, \
    out_idx_2_size, out_idx_2_dst_rsize, out_idx_2_dst_asize, core_step_out = args

    out_offset = (r2nd_lp_idx * src_r2nd_lp_unit * src_r2nd_lp_step_out + c1_lp_idx * src_c1_lp_step_out +
                  ele_idx // out_idx_0_dst_rsize % out_idx_0_size * out_idx_0_dst_asize +
                  ele_idx // out_idx_1_dst_rsize % out_idx_1_size * out_idx_1_dst_asize +
                  ele_idx // out_idx_2_dst_rsize % out_idx_2_size * out_idx_2_dst_asize + core_step_out)

    return out_offset


def _copy_data_in(args):
    """
    copy data from gm to ub
    """

    tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset, c1_step_in, c1_pl_size, r2nd_pl_size, dtype_factor = args

    with tik_inst.for_range(0, c1_pl_size) as c1_idx:
        tik_inst.data_move(src_ub[in_ub_offset + c1_idx * r2nd_pl_size * tdc.C0_16],
                           src_in_gm[in_gm_offset + c1_idx * c1_step_in], 0, 1, r2nd_pl_size * dtype_factor, 0, 0)


# pylint: disable=unused-variable
def _copy_data_out(out_offset_args, copy_out_args):
    """
    copy data from ub to gm
    """

    src_r2nd_lp_step_out = out_offset_args[2]
    (tik_inst, dst_out_gm, out_ub, pln_dst_c_cnt, out_beg_idx,
     out_end_idx, c1_pl_size, r2nd_pl_size, src_2_dst_flag, c_mod_c0, dtype_factor, ele_per_block) = copy_out_args

    vnc_line_cnt = (out_end_idx - out_beg_idx) // pln_dst_c_cnt
    left_c_cnt = (out_end_idx - out_beg_idx) % pln_dst_c_cnt
    vnc_line_ub_offset = pln_dst_c_cnt * c1_pl_size * r2nd_pl_size * tdc.C0_16

    r2nd_ub_offset = tik_inst.Scalar(name="r2nd_ub_offset")
    left_axis_ub_offset = tik_inst.Scalar(name="left_axis_ub_offset")
    with tik_inst.if_scope(src_2_dst_flag > 0):  # such as DC1HWNoNiC0 -> NDHWC, pln_dst_c_cnt is DHW
        r2nd_ub_offset.set_as(pln_dst_c_cnt * c1_pl_size * tdc.C0_16)
        left_axis_ub_offset.set_as(c1_pl_size * tdc.C0_16)
    with tik_inst.else_scope():  # such as NC1HWC0 -> NHWC, pln_dst_c_cnt is N
        r2nd_ub_offset.set_as(c1_pl_size * tdc.C0_16)
        left_axis_ub_offset.set_as(r2nd_pl_size * c1_pl_size * tdc.C0_16)

    with tik_inst.for_range(0, vnc_line_cnt) as vnc_line_idx:
        with tik_inst.for_range(0, pln_dst_c_cnt) as c_idx:
            ele_idx = vnc_line_idx * pln_dst_c_cnt + c_idx + out_beg_idx
            out_offset = _update_output_offset(out_offset_args, ele_idx)
            ub_offset = vnc_line_idx * vnc_line_ub_offset + c_idx * left_axis_ub_offset
            with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx:
                tik_inst.data_move(dst_out_gm[out_offset + r2nd_idx * src_r2nd_lp_step_out],
                                   out_ub[ub_offset + r2nd_idx * r2nd_ub_offset], 0, 1, c1_pl_size * dtype_factor, 0, 0)

    with tik_inst.if_scope(left_c_cnt > 0):
        with tik_inst.for_range(0, left_c_cnt) as c_idx_1:
            ele_idx_1 = vnc_line_cnt * pln_dst_c_cnt + c_idx_1 + out_beg_idx
            out_offset_1 = _update_output_offset(out_offset_args, ele_idx_1)
            ub_offset_1 = vnc_line_cnt * vnc_line_ub_offset + c_idx_1 * left_axis_ub_offset
            with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx_1:
                tik_inst.data_move(dst_out_gm[out_offset_1 + r2nd_idx_1 * src_r2nd_lp_step_out],
                                   out_ub[ub_offset_1 + r2nd_idx_1 * r2nd_ub_offset],
                                   0, 1, c1_pl_size * dtype_factor, 0, 0)


def _copy_data_out_c_less_8c0(out_offset_args, copy_out_args):
    """
    copy data from ub to gm
    """

    (tik_inst, dst_out_gm, out_ub, pln_dst_c_cnt, out_beg_idx, out_end_idx,
     c1_pl_size, r2nd_pl_size, src_2_dst_flag, c_mod_c0, dtype_factor, ele_per_block) = copy_out_args

    vnc_line_cnt = (out_end_idx - out_beg_idx) // pln_dst_c_cnt
    vnc_line_ub_offset = pln_dst_c_cnt * c1_pl_size * r2nd_pl_size * tdc.C0_16

    left_axis_ub_offset = tik_inst.Scalar(name="left_axis_ub_offset")
    with tik_inst.if_scope(src_2_dst_flag > 0):  # such as DC1HWNoNiC0 -> NDHWC, pln_dst_c_cnt is DHW
        left_axis_ub_offset.set_as(c1_pl_size * tdc.C0_16)
    with tik_inst.else_scope():  # such as NC1HWC0 -> NHWC, pln_dst_c_cnt is N
        left_axis_ub_offset.set_as(r2nd_pl_size * c1_pl_size * tdc.C0_16)

    with tik_inst.new_stmt_scope():
        sub_c_size = (c1_pl_size - 1) * tdc.C0_16 + c_mod_c0
        tmp_reg = tik_inst.Scalar(dtype=dst_out_gm.dtype)
        with tik_inst.for_range(0, vnc_line_cnt) as vnc_line_idx:
            with tik_inst.for_range(0, pln_dst_c_cnt) as c_idx:
                ele_idx = vnc_line_idx * pln_dst_c_cnt + c_idx + out_beg_idx
                out_offset = _update_output_offset(out_offset_args, ele_idx)
                ub_offset = vnc_line_idx * vnc_line_ub_offset + c_idx * left_axis_ub_offset
                tik_inst.data_move(dst_out_gm[out_offset],
                                   out_ub[ub_offset], 0, 1, r2nd_pl_size * sub_c_size // ele_per_block, 0, 0)

                with tik_inst.if_scope(r2nd_pl_size * sub_c_size % ele_per_block > 0):
                    with tik_inst.for_range(0, ele_per_block) as reg_idx:
                        tmp_reg.set_as(out_ub[ub_offset + r2nd_pl_size * sub_c_size - ele_per_block + reg_idx])
                        out_ub[reg_idx].set_as(tmp_reg)
                    tik_inst.data_move(dst_out_gm[out_offset + r2nd_pl_size * sub_c_size - ele_per_block],
                                       out_ub, 0, 1, 1, 0, 0)


def _func_transform_201(tensor_args, tp_args):
    """
    transform function for tiling mode 201
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, ele_per_block = tensor_args
    (ub_offset, mc_flag, used_core, core_step_in, core_step_out, nlc_r2nd_lp_cnt,
     nlc_c1_lp_cnt, nlc_left_lp_cnt, nlc_r2nd_left, nlc_c1_left, lc_r2nd_lp_cnt, lc_c1_lp_cnt,
     lc_left_lp_cnt, lc_r2nd_left, lc_c1_left, src_r2nd_lp_unit, src_r2nd_lp_step_in, src_r2nd_lp_step_out,
     src_c1_step_in, src_c1_lp_unit, src_c1_lp_step_in, src_c1_lp_step_out, pln_dst_c_cnt, c_mod_c0, in_idx_0_size,
     in_idx_0_dst_rsize, in_idx_0_src_asize, in_idx_1_size, in_idx_1_dst_rsize, in_idx_1_src_asize,
     in_idx_2_size, in_idx_2_dst_rsize, in_idx_2_src_asize, out_idx_0_size, out_idx_0_dst_rsize,
     out_idx_0_dst_asize, out_idx_1_size, out_idx_1_dst_rsize, out_idx_1_dst_asize, out_idx_2_size,
     out_idx_2_dst_rsize, out_idx_2_dst_asize, src_2_dst_flag) = tp_args
    r2nd_lp_cnt = tik_inst.Scalar(name="r2nd_lp_cnt")
    c1_lp_cnt = tik_inst.Scalar(name="c1_lp_cnt")
    c1_left = tik_inst.Scalar(name="c1_left")
    r2nd_left = tik_inst.Scalar(name="r2nd_left")
    sub_c_size = tik_inst.Scalar(name="sub_c_size")
    dtype_factor = tdc.get_dtype_factor(src_ub.dtype)

    def _inner_func(begin_idx, end_idx):
        r2nd_pl_size = tik_inst.Scalar(name="r2nd_pl_size")
        c1_pl_size = tik_inst.Scalar(name="c1_pl_size")
        is_last_c1 = tik_inst.Scalar(name="is_last_c1")
        out_beg_idx = tik_inst.Scalar(name="out_beg_idx")
        out_end_idx = tik_inst.Scalar(name="out_end_idx")
        with tik_inst.for_range(0, r2nd_lp_cnt) as r2nd_lp_idx:
            with tik_inst.if_scope(tik.any(r2nd_lp_idx != r2nd_lp_cnt - 1, r2nd_left == 0)):
                r2nd_pl_size.set_as(src_r2nd_lp_unit)
            with tik_inst.else_scope():
                r2nd_pl_size.set_as(lc_r2nd_left)

            with tik_inst.for_range(0, c1_lp_cnt) as c1_lp_idx:
                with tik_inst.if_scope(tik.any(c1_lp_idx != c1_lp_cnt - 1, c1_left == 0)):
                    c1_pl_size.set_as(src_c1_lp_unit)
                    with tik_inst.if_scope(c1_lp_idx == c1_lp_cnt - 1):
                        is_last_c1.set_as(1)
                    with tik_inst.else_scope():
                        is_last_c1.set_as(0)
                with tik_inst.else_scope():
                    c1_pl_size.set_as(lc_c1_left)
                    is_last_c1.set_as(1)

                with tik_inst.for_range(begin_idx, end_idx) as ele_idx:
                    vnc_pl_c_cnt = tdc.VNC_LINES * pln_dst_c_cnt
                    with tik_inst.if_scope((ele_idx - begin_idx) % vnc_pl_c_cnt == 0):
                        out_beg_idx.set_as(ele_idx)
                    in_offset_args = (r2nd_lp_idx, src_r2nd_lp_step_in, c1_lp_idx, src_c1_lp_step_in,
                                      in_idx_0_size, in_idx_0_dst_rsize, in_idx_0_src_asize,
                                      in_idx_1_size, in_idx_1_dst_rsize, in_idx_1_src_asize,
                                      in_idx_2_size, in_idx_2_dst_rsize, in_idx_2_src_asize, ele_idx,
                                      block_idx * core_step_in)
                    in_gm_offset = _update_input_offset(in_offset_args)
                    in_ub_offset = (ele_idx - begin_idx) % vnc_pl_c_cnt * c1_pl_size * r2nd_pl_size * tdc.C0_16
                    copy_in_args = (tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset,
                                    src_c1_step_in, c1_pl_size, r2nd_pl_size, dtype_factor)
                    _copy_data_in(copy_in_args)
                    with tik_inst.if_scope(tik.any((ele_idx - begin_idx + 1) % vnc_pl_c_cnt == 0,
                                                   ele_idx == end_idx - 1)):
                        out_end_idx.set_as(ele_idx + 1)
                        vnc_args = (tik_inst, src_ub, ub_offset, c1_lp_cnt, c1_pl_size, r2nd_pl_size, pln_dst_c_cnt,
                                    src_2_dst_flag, c_mod_c0, dtype_factor)
                        _twice_vnchwconv_no_invert(vnc_args)
                        out_gm_args = (r2nd_lp_idx, src_r2nd_lp_unit, src_r2nd_lp_step_out,
                                       c1_lp_idx, src_c1_lp_step_out,
                                       out_idx_0_size, out_idx_0_dst_rsize, out_idx_0_dst_asize,
                                       out_idx_1_size, out_idx_1_dst_rsize, out_idx_1_dst_asize,
                                       out_idx_2_size, out_idx_2_dst_rsize, out_idx_2_dst_asize,
                                       block_idx * core_step_out)
                        copy_out_args = (tik_inst, dst_out_gm, src_ub[ub_offset], pln_dst_c_cnt, out_beg_idx,
                                         out_end_idx, c1_pl_size, r2nd_pl_size, src_2_dst_flag, c_mod_c0,
                                         dtype_factor, ele_per_block)
                        with tik_inst.if_scope(c1_lp_cnt > 1):
                            _copy_data_out(out_gm_args, copy_out_args)
                        with tik_inst.else_scope():
                            _copy_data_out_c_less_8c0(out_gm_args, copy_out_args)

    with tik_inst.if_scope(block_idx != used_core - 1):
        r2nd_lp_cnt.set_as(nlc_r2nd_lp_cnt)
        c1_lp_cnt.set_as(nlc_c1_lp_cnt)
        r2nd_left.set_as(nlc_r2nd_left)
        c1_left.set_as(nlc_c1_left)
        _inner_func(mc_flag * block_idx * nlc_left_lp_cnt, (mc_flag * block_idx + 1) * nlc_left_lp_cnt)
    with tik_inst.else_scope():
        r2nd_lp_cnt.set_as(lc_r2nd_lp_cnt)
        c1_lp_cnt.set_as(lc_c1_lp_cnt)
        r2nd_left.set_as(lc_r2nd_left)
        c1_left.set_as(lc_c1_left)
        _inner_func(mc_flag * block_idx * nlc_left_lp_cnt, mc_flag * block_idx * nlc_left_lp_cnt + lc_left_lp_cnt)


def trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name="trans_data_negative_target_tc"):
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
        kernel name, default value is "trans_data_negative_target_tc"

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
    dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm")
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tik.scope_ubuf, "total_ub")

    tiling_mode = tiling_params[0]
    used_core_cnt = tiling_params[3]
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            if tiling_mode == 201:
                tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt]
                tp_args = tiling_params[1:]
                _func_transform_201(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[src_in_gm], outputs=[dst_out_gm])
