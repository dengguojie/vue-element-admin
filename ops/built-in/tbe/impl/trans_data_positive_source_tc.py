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
trans_data_positive_source_tc
"""

from te import tik
from impl import trans_data_common_func as tdc


# used for scalar
PAD_IDX_LIST = (0, 1)
# frame up levels
FRAME_LEVEL = 2


# pylint: disable=too-many-locals, inconsistent-return-statements, too-many-statements
def _renew_input_output_shape_format(in_shape, out_shape, in_format, out_format):
    """
    renew shape and format to adapt tiling process
    """

    in_format_upper = in_format.upper()
    out_format_upper = out_format.upper()

    if in_format_upper in ("NHWC", "NDHWC") and out_format_upper in ("NC1HWC0", "NDC1HWC0"):
        in_format_new = "NHC"
        out_format_new = "NCHT"
        axis_d = 1
        if len(in_shape) == 4:
            axis_n, axis_h, axis_w, axis_c = in_shape
        else:
            axis_n, axis_d, axis_h, axis_w, axis_c = in_shape
        axis_c0 = out_shape[-1]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        in_shape_new = [axis_n * axis_d] + [axis_h * axis_w] + [axis_c]
        out_shape_new = [axis_n * axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_c0]
        new_params_nc1hwc0 = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_nc1hwc0

    if in_format_upper in ("ND", "NCHW", "NHWC") and out_format_upper == "FRACTAL_NZ":
        in_format_new = "HNC"
        out_format_new = "HCNT"
        in_shape_len = len(in_shape)
        if in_shape_len == 1:
            axis_h, axis_n, axis_c = 1, 1, in_shape[0]
        elif in_shape_len == 2:
            axis_h, axis_n, axis_c = 1, in_shape[0], in_shape[1]
        else:
            axis_h, axis_n, axis_c = tdc.get_shape_size(in_shape[:-2]), in_shape[-2], in_shape[-1]
        axis_c0 = out_shape[-1]
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, tdc.NI_16)
        axis_ni = tdc.NI_16
        in_shape_new = [axis_h] + [axis_n] + [axis_c]
        out_shape_new = [axis_h] + [axis_c1] + [axis_no * axis_ni] + [axis_c0]
        new_params_nz = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_nz

    if in_format_upper == "NDHWC" and out_format_upper == "FRACTAL_Z_3D":
        in_format_new = "NDHC"
        out_format_new = "DCHNT"
        axis_c0 = out_shape[-1]
        axis_n, axis_d, axis_h, axis_w, axis_c = in_shape
        axis_c1 = tdc.ceil_div(axis_c, axis_c0)
        axis_no = tdc.ceil_div(axis_n, tdc.NI_16)
        axis_ni = tdc.NI_16
        in_shape_new = [axis_n] + [axis_d] + [axis_h * axis_w] + [axis_c]
        out_shape_new = [axis_d] + [axis_c1] + [axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        new_params_z3d = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_z3d

    if in_format_upper == "NC1HWC0" and out_format_upper in ("FRACTAL_Z", "FRACTAL_ZN"):
        in_format_new = "NDHC"
        out_format_new = "DCHNT"
        axis_d = 1
        axis_c = 1
        axis_n, axis_c1, axis_h, axis_w, axis_c0 = in_shape
        axis_no = tdc.ceil_div(axis_n, tdc.NI_16)
        axis_ni = tdc.NI_16
        in_shape_new = [axis_n] + [axis_d] + [axis_c1 * axis_h * axis_w] + [axis_c0]
        out_shape_new = [axis_d] + [axis_c] + [axis_c1 * axis_h * axis_w] + [axis_no * axis_ni] + [axis_c0]
        new_params_z3d = [in_shape_new, out_shape_new] + [in_format_new, out_format_new]

        return new_params_z3d

    return [in_shape, out_shape] + [in_format, out_format]


# pylint: disable=too-many-locals, too-many-statements, unused-variable
def _get_mc_info_positive_1010(cl_args, c_args, cr_args):
    """
    get multiple core axis position for tiling mode 1010
    """

    axis_dst_cl_size, tp_1010_dst_cl_lp_step_in, tp_1010_dst_cl_lp_step_out = cl_args
    c_lp_cnt, tp_1010_c_lp_step_in, tp_1010_c_lp_step_out, c_left = c_args
    (dst_cr_lp_cnt, tp_1010_dst_cr_lp_step_in, tp_1010_dst_cr_lp_step_out,
     pln_dst_cr_size, vnc_row_size, vnc_row_left, ll_dst_cr_left) = cr_args

    tmp_full_lp_cnt_cr = tdc.CORE_DIM_NUM if tdc.floor_div(dst_cr_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_cr = dst_cr_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_cr == 0:
        tmp_full_lp_cnt_cr += tdc.CORE_DIM_NUM
    full_lp_cnt_cr = tmp_full_lp_cnt_cr + reminder_lp_cnt_cr

    tmp_full_lp_cnt_c = tdc.CORE_DIM_NUM if tdc.floor_div(c_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_c = c_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_c == 0:
        tmp_full_lp_cnt_c += tdc.CORE_DIM_NUM
    full_lp_cnt_c = tmp_full_lp_cnt_c + reminder_lp_cnt_c

    tmp_full_lp_cnt_left = tdc.CORE_DIM_NUM if tdc.floor_div(axis_dst_cl_size, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_left = axis_dst_cl_size % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_left == 0:
        tmp_full_lp_cnt_left += tdc.CORE_DIM_NUM
    full_lp_cnt_left = tmp_full_lp_cnt_left + reminder_lp_cnt_left

    lp_cnt_list = (full_lp_cnt_left, full_lp_cnt_cr, full_lp_cnt_c)
    if lp_cnt_list.index(max(lp_cnt_list)) == 0:
        tp_1010_used_core_cnt = tdc.ceil_div(axis_dst_cl_size, tdc.ceil_div(axis_dst_cl_size, tdc.CORE_DIM_NUM))
        tp_1010_nlc_dst_cl_lp_cnt = tdc.ceil_div(axis_dst_cl_size, tp_1010_used_core_cnt)
        tp_1010_lc_dst_cl_lp_cnt = axis_dst_cl_size - tp_1010_nlc_dst_cl_lp_cnt * (tp_1010_used_core_cnt - 1)
        tp_1010_core_step_in = tp_1010_nlc_dst_cl_lp_cnt * tp_1010_dst_cl_lp_step_in
        tp_1010_core_step_out = tp_1010_nlc_dst_cl_lp_cnt * tp_1010_dst_cl_lp_step_out
        tp_1010_nlc_c_lp_cnt = c_lp_cnt
        tp_1010_lc_c_lp_cnt = c_lp_cnt
        tp_1010_nlc_c_left = c_left
        tp_1010_lc_c_left = c_left
        tp_1010_nlc_dst_cr_lp_cnt = dst_cr_lp_cnt
        tp_1010_lc_dst_cr_lp_cnt = dst_cr_lp_cnt
        tp_1010_nlc_vnc_row_left = vnc_row_left
        tp_1010_lc_vnc_row_left = vnc_row_left
        tp_1010_nlc_last_line_cr_cnt = ll_dst_cr_left
        tp_1010_lc_last_line_cr_cnt = ll_dst_cr_left
    elif lp_cnt_list.index(max(lp_cnt_list)) == 1:
        tp_1010_used_core_cnt = tdc.ceil_div(dst_cr_lp_cnt, tdc.ceil_div(dst_cr_lp_cnt, tdc.CORE_DIM_NUM))
        tp_1010_nlc_dst_cr_lp_cnt = tdc.ceil_div(dst_cr_lp_cnt, tp_1010_used_core_cnt)
        tp_1010_lc_dst_cr_lp_cnt = dst_cr_lp_cnt - tp_1010_nlc_dst_cr_lp_cnt * (tp_1010_used_core_cnt - 1)
        tp_1010_core_step_in = tp_1010_nlc_dst_cr_lp_cnt * tp_1010_dst_cr_lp_step_in
        tp_1010_core_step_out = tp_1010_nlc_dst_cr_lp_cnt * tp_1010_dst_cr_lp_step_out
        tp_1010_nlc_vnc_row_left = 0
        tp_1010_lc_vnc_row_left = vnc_row_left
        tp_1010_nlc_last_line_cr_cnt = pln_dst_cr_size
        tp_1010_lc_last_line_cr_cnt = ll_dst_cr_left
        tp_1010_nlc_c_lp_cnt = c_lp_cnt
        tp_1010_lc_c_lp_cnt = c_lp_cnt
        tp_1010_nlc_c_left = c_left
        tp_1010_lc_c_left = c_left
        tp_1010_nlc_dst_cl_lp_cnt = axis_dst_cl_size
        tp_1010_lc_dst_cl_lp_cnt = axis_dst_cl_size
    else:
        tp_1010_used_core_cnt = tdc.ceil_div(c_lp_cnt, tdc.ceil_div(c_lp_cnt, tdc.CORE_DIM_NUM))
        tp_1010_nlc_c_lp_cnt = tdc.ceil_div(c_lp_cnt, tp_1010_used_core_cnt)
        tp_1010_lc_c_lp_cnt = c_lp_cnt - tp_1010_nlc_c_lp_cnt * (tp_1010_used_core_cnt - 1)
        tp_1010_core_step_in = tp_1010_nlc_c_lp_cnt * tp_1010_c_lp_step_in
        tp_1010_core_step_out = tp_1010_nlc_c_lp_cnt * tp_1010_c_lp_step_out
        tp_1010_nlc_c_left = 0
        tp_1010_lc_c_left = c_left
        tp_1010_nlc_dst_cl_lp_cnt = axis_dst_cl_size
        tp_1010_lc_dst_cl_lp_cnt = axis_dst_cl_size
        tp_1010_nlc_dst_cr_lp_cnt = dst_cr_lp_cnt
        tp_1010_lc_dst_cr_lp_cnt = dst_cr_lp_cnt
        tp_1010_nlc_vnc_row_left = vnc_row_left
        tp_1010_lc_vnc_row_left = vnc_row_left
        tp_1010_nlc_last_line_cr_cnt = ll_dst_cr_left
        tp_1010_lc_last_line_cr_cnt = ll_dst_cr_left

    tiling_params = [tp_1010_used_core_cnt, tp_1010_core_step_in, tp_1010_core_step_out] + \
                    [tp_1010_nlc_dst_cl_lp_cnt, tp_1010_nlc_dst_cr_lp_cnt, tp_1010_nlc_vnc_row_left,
                     tp_1010_nlc_last_line_cr_cnt, tp_1010_nlc_c_lp_cnt, tp_1010_nlc_c_left] + \
                    [tp_1010_lc_dst_cl_lp_cnt, tp_1010_lc_dst_cr_lp_cnt, tp_1010_lc_vnc_row_left,
                     tp_1010_lc_last_line_cr_cnt, tp_1010_lc_c_lp_cnt, tp_1010_lc_c_left]

    return tiling_params


# pylint: disable=too-many-locals, too-many-statements
def _get_mc_info_positive_1011(r2nd_args, c_args, cl_args):
    """
    get multiple core axis position for tiling mode 1011
    """

    axis_dst_r2nd_lp_cnt, tp_1011_dst_r2nd_lp_step_in, tp_1011_dst_r2nd_lp_step_out, axis_dst_r2nd_left = r2nd_args
    c_lp_cnt, tp_1011_c_lp_step_in, tp_1011_c_lp_step_out, c_left = c_args
    axis_src_cl_lp_cnt, tp_1011_src_cl_lp_step_in, tp_1011_src_cl_lp_step_out, axis_src_cl_left = cl_args

    tmp_full_lp_cnt_r2nd = tdc.CORE_DIM_NUM if tdc.floor_div(axis_dst_r2nd_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_r2nd = axis_dst_r2nd_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_r2nd == 0:
        tmp_full_lp_cnt_r2nd += tdc.CORE_DIM_NUM
    full_lp_cnt_r2nd = tmp_full_lp_cnt_r2nd + reminder_lp_cnt_r2nd

    tmp_full_lp_cnt_c = tdc.CORE_DIM_NUM if tdc.floor_div(c_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_c = c_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_c == 0:
        tmp_full_lp_cnt_c += tdc.CORE_DIM_NUM
    full_lp_cnt_c = tmp_full_lp_cnt_c + reminder_lp_cnt_c

    tmp_full_lp_cnt_left = tdc.CORE_DIM_NUM if tdc.floor_div(axis_src_cl_lp_cnt, tdc.CORE_DIM_NUM) > 0 else 0
    reminder_lp_cnt_left = axis_src_cl_lp_cnt % tdc.CORE_DIM_NUM
    if reminder_lp_cnt_left == 0:
        tmp_full_lp_cnt_left += tdc.CORE_DIM_NUM
    full_lp_cnt_left = tmp_full_lp_cnt_left + reminder_lp_cnt_left

    lp_cnt_list = (full_lp_cnt_r2nd, full_lp_cnt_left, full_lp_cnt_c)
    if lp_cnt_list.index(max(lp_cnt_list)) == 0:
        tp_1011_mc_on_cl = 0
        tp_1011_used_core_cnt = tdc.ceil_div(axis_dst_r2nd_lp_cnt, tdc.ceil_div(axis_dst_r2nd_lp_cnt, tdc.CORE_DIM_NUM))
        tp_1011_nlc_dst_r2nd_lp_cnt = tdc.ceil_div(axis_dst_r2nd_lp_cnt, tp_1011_used_core_cnt)
        tp_1011_lc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt - tp_1011_nlc_dst_r2nd_lp_cnt * (tp_1011_used_core_cnt - 1)
        tp_1011_core_step_in = tp_1011_nlc_dst_r2nd_lp_cnt * tp_1011_dst_r2nd_lp_step_in
        tp_1011_core_step_out = tp_1011_nlc_dst_r2nd_lp_cnt * tp_1011_dst_r2nd_lp_step_out
        tp_1011_nlc_dst_r2nd_left = 0
        tp_1011_lc_dst_r2nd_left = axis_dst_r2nd_left
        tp_1011_nlc_c_lp_cnt = c_lp_cnt
        tp_1011_lc_c_lp_cnt = c_lp_cnt
        tp_1011_nlc_c_left = c_left
        tp_1011_lc_c_left = c_left
        tp_1011_nlc_src_cl_lp_cnt = axis_src_cl_lp_cnt
        tp_1011_lc_src_cl_lp_cnt = axis_src_cl_lp_cnt
        tp_1011_nlc_src_cl_left = axis_src_cl_left
        tp_1011_lc_src_cl_left = axis_src_cl_left
    elif lp_cnt_list.index(max(lp_cnt_list)) == 1:
        tp_1011_mc_on_cl = 1
        tp_1011_used_core_cnt = tdc.ceil_div(axis_src_cl_lp_cnt, tdc.ceil_div(axis_src_cl_lp_cnt, tdc.CORE_DIM_NUM))
        tp_1011_nlc_src_cl_lp_cnt = tdc.ceil_div(axis_src_cl_lp_cnt, tp_1011_used_core_cnt)
        tp_1011_lc_src_cl_lp_cnt = axis_src_cl_lp_cnt - tp_1011_nlc_src_cl_lp_cnt * (tp_1011_used_core_cnt - 1)
        tp_1011_core_step_in = tp_1011_nlc_src_cl_lp_cnt * tp_1011_src_cl_lp_step_in
        tp_1011_core_step_out = tp_1011_nlc_src_cl_lp_cnt * tp_1011_src_cl_lp_step_out
        tp_1011_nlc_src_cl_left = 0
        tp_1011_lc_src_cl_left = axis_src_cl_left
        tp_1011_nlc_c_lp_cnt = c_lp_cnt
        tp_1011_lc_c_lp_cnt = c_lp_cnt
        tp_1011_nlc_c_left = c_left
        tp_1011_lc_c_left = c_left
        tp_1011_nlc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt
        tp_1011_lc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt
        tp_1011_nlc_dst_r2nd_left = axis_dst_r2nd_left
        tp_1011_lc_dst_r2nd_left = axis_dst_r2nd_left
    else:
        tp_1011_mc_on_cl = 0
        tp_1011_used_core_cnt = tdc.ceil_div(c_lp_cnt, tdc.ceil_div(c_lp_cnt, tdc.CORE_DIM_NUM))
        tp_1011_nlc_c_lp_cnt = tdc.ceil_div(c_lp_cnt, tp_1011_used_core_cnt)
        tp_1011_lc_c_lp_cnt = c_lp_cnt - tp_1011_nlc_c_lp_cnt * (tp_1011_used_core_cnt - 1)
        tp_1011_core_step_in = tp_1011_nlc_c_lp_cnt * tp_1011_c_lp_step_in
        tp_1011_core_step_out = tp_1011_nlc_c_lp_cnt * tp_1011_c_lp_step_out
        tp_1011_nlc_c_left = 0
        tp_1011_lc_c_left = c_left
        tp_1011_nlc_src_cl_lp_cnt = axis_src_cl_lp_cnt
        tp_1011_lc_src_cl_lp_cnt = axis_src_cl_lp_cnt
        tp_1011_nlc_src_cl_left = axis_src_cl_left
        tp_1011_lc_src_cl_left = axis_src_cl_left
        tp_1011_nlc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt
        tp_1011_lc_dst_r2nd_lp_cnt = axis_dst_r2nd_lp_cnt
        tp_1011_nlc_dst_r2nd_left = axis_dst_r2nd_left
        tp_1011_lc_dst_r2nd_left = axis_dst_r2nd_left

    tiling_params = [tp_1011_mc_on_cl, tp_1011_used_core_cnt, tp_1011_core_step_in, tp_1011_core_step_out] + \
                    [tp_1011_nlc_dst_r2nd_lp_cnt, tp_1011_nlc_dst_r2nd_left, tp_1011_nlc_src_cl_lp_cnt,
                     tp_1011_nlc_src_cl_left, tp_1011_nlc_c_lp_cnt, tp_1011_nlc_c_left] + \
                    [tp_1011_lc_dst_r2nd_lp_cnt, tp_1011_lc_dst_r2nd_left, tp_1011_lc_src_cl_lp_cnt,
                     tp_1011_lc_src_cl_left, tp_1011_lc_c_lp_cnt, tp_1011_lc_c_left]

    return tiling_params


# pylint: disable=unbalanced-tuple-unpacking
def _get_tiling_params_1010(tiling_args):
    """
    get tiling parameters for tiling mode 1010
    """

    (in_shape, out_shape, src_format, dst_format, axis_c_size, one_vnc_line_size, c0_len,
     tp_101_ub_offset, tp_101_vnc_line_size, tp_101_c0_size, tp_101_c_mod_c0) = tiling_args
    tp_101_tiling_mode = 1010
    one_vnc_line_size = one_vnc_line_size // c0_len * c0_len

    # target axis c-left tiling parameters
    axis_dst_cl_size = tdc.get_shape_size(out_shape[:dst_format.index("C")])
    dst_cl_char = dst_format[dst_format.index("C") - 1]
    tp_1010_dst_cl_lp_unit = 1
    tp_1010_dst_cl_lp_step_in = tdc.get_shape_size(in_shape[src_format.index(dst_cl_char) + 1:])
    tp_1010_dst_cl_lp_step_out = tdc.get_shape_size(out_shape[dst_format.index("C"):])

    # source axis c tiling parameters
    tp_1010_c_lp_unit = axis_c_size if axis_c_size < one_vnc_line_size else one_vnc_line_size
    tp_1010_c_lp_step_in = tp_1010_c_lp_unit
    lp_c1_cnt = tdc.ceil_div(tp_1010_c_lp_unit, c0_len)
    tp_1010_c_lp_step_out = tdc.get_shape_size([lp_c1_cnt] + out_shape[dst_format.index("C") + 1:])
    tp_1010_c_step_out = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:])
    c_lp_cnt = tdc.ceil_div(axis_c_size, tp_1010_c_lp_unit)
    c_left = axis_c_size % tp_1010_c_lp_unit

    # target axis c-right tiling parameters
    axis_dst_cr_size = tdc.get_shape_size(in_shape[src_format.index(dst_format[-2]):-1])  # not include c0
    tp_1010_pln_dst_cr_size = one_vnc_line_size // tdc.ceil_fill(tp_1010_c_lp_unit, c0_len)
    tp_1010_vnc_row_size = tdc.VNC_LINES
    per_vnc_dst_cr_cnt = tp_1010_pln_dst_cr_size * tp_1010_vnc_row_size
    dst_cr_lp_cnt = tdc.ceil_div(axis_dst_cr_size, per_vnc_dst_cr_cnt)
    dst_cr_left = axis_dst_cr_size % per_vnc_dst_cr_cnt
    vnc_row_left = tdc.ceil_div(dst_cr_left, tp_1010_pln_dst_cr_size)
    tmp_dst_cr_left = dst_cr_left % tp_1010_pln_dst_cr_size
    ll_dst_cr_left = tmp_dst_cr_left if tmp_dst_cr_left > 0 else tp_1010_pln_dst_cr_size
    tp_1010_dst_cr_lp_step_in = tdc.get_shape_size(in_shape[-1:] + [per_vnc_dst_cr_cnt])
    tp_1010_dst_cr_lp_step_out = tdc.get_shape_size(out_shape[dst_format.index(src_format[-2]) + 1:] +
                                                    [per_vnc_dst_cr_cnt])
    tp_1010_dst_cr_step_in = tdc.get_shape_size(in_shape[-1:])

    cl_args = axis_dst_cl_size, tp_1010_dst_cl_lp_step_in, tp_1010_dst_cl_lp_step_out
    c_args = c_lp_cnt, tp_1010_c_lp_step_in, tp_1010_c_lp_step_out, c_left
    cr_args = (dst_cr_lp_cnt, tp_1010_dst_cr_lp_step_in, tp_1010_dst_cr_lp_step_out,
               tp_1010_pln_dst_cr_size, tp_1010_vnc_row_size, vnc_row_left, ll_dst_cr_left)
    (tp_1010_used_core_cnt, tp_1010_core_step_in, tp_1010_core_step_out, tp_1010_nlc_dst_cl_lp_cnt,
     tp_1010_nlc_dst_cr_lp_cnt, tp_1010_nlc_vnc_row_left, tp_1010_nlc_last_line_cr_cnt, tp_1010_nlc_c_lp_cnt,
     tp_1010_nlc_c_left, tp_1010_lc_dst_cl_lp_cnt, tp_1010_lc_dst_cr_lp_cnt, tp_1010_lc_vnc_row_left,
     tp_1010_lc_last_line_cr_cnt, tp_1010_lc_c_lp_cnt, tp_1010_lc_c_left) = _get_mc_info_positive_1010(cl_args,
                                                                                                       c_args, cr_args)

    sub_tiling_params = [tp_101_tiling_mode, tp_101_ub_offset, tp_1010_used_core_cnt, tp_1010_core_step_in,
                         tp_1010_core_step_out, tp_1010_dst_cl_lp_step_in, tp_1010_dst_cl_lp_step_out,
                         tp_1010_dst_cl_lp_unit, tp_1010_dst_cr_lp_step_in, tp_1010_dst_cr_lp_step_out,
                         tp_1010_dst_cr_step_in, tp_101_vnc_line_size, tp_1010_pln_dst_cr_size, tp_1010_vnc_row_size,
                         tp_1010_c_lp_step_in, tp_1010_c_lp_step_out, tp_1010_c_step_out, tp_101_c0_size,
                         tp_101_c_mod_c0, tp_1010_c_lp_unit, tp_1010_nlc_dst_cl_lp_cnt, tp_1010_nlc_dst_cr_lp_cnt,
                         tp_1010_nlc_vnc_row_left, tp_1010_nlc_last_line_cr_cnt, tp_1010_nlc_c_lp_cnt,
                         tp_1010_nlc_c_left, tp_1010_lc_dst_cl_lp_cnt, tp_1010_lc_dst_cr_lp_cnt,
                         tp_1010_lc_vnc_row_left, tp_1010_lc_last_line_cr_cnt, tp_1010_lc_c_lp_cnt, tp_1010_lc_c_left]
    pad_params = [1, 1, 0, 1, 1, 0]
    return sub_tiling_params + pad_params


# pylint: disable=unbalanced-tuple-unpacking
def _get_tiling_params_1011(tiling_args):
    """
    get tiling parameters for tiling mode 1011
    """

    (in_shape, out_shape, src_format, dst_format, axis_c_size, one_vnc_line_size,
     c0_len, tp_101_ub_offset, tp_101_vnc_line_size, tp_101_c0_size, tp_101_c_mod_c0, tp_names) = tiling_args
    tp_101_tiling_mode = 1011
    one_vnc_line_size = one_vnc_line_size // c0_len * c0_len

    # target axis -2 tiling parameters
    dst_r2nd_in_src_idx = src_format.index(dst_format[-2])
    axis_dst_r2nd_size = in_shape[dst_r2nd_in_src_idx]
    tp_1011_dst_r2nd_lp_unit = axis_dst_r2nd_size if axis_dst_r2nd_size < tdc.VNC_LINES else tdc.VNC_LINES
    axis_dst_r2nd_lp_cnt = tdc.ceil_div(axis_dst_r2nd_size, tp_1011_dst_r2nd_lp_unit)
    axis_dst_r2nd_left = axis_dst_r2nd_size % tp_1011_dst_r2nd_lp_unit
    tp_1011_dst_r2nd_lp_step_in = tdc.get_shape_size(in_shape[dst_r2nd_in_src_idx + 1:] +
                                                     [tp_1011_dst_r2nd_lp_unit])
    tp_1011_dst_r2nd_lp_step_out = tdc.get_shape_size(out_shape[-1:] + [tp_1011_dst_r2nd_lp_unit])
    tp_1011_dst_r2nd_step_in = tdc.get_shape_size(in_shape[dst_r2nd_in_src_idx + 1:])

    # source axis c tiling parameters
    tp_1011_c_lp_unit = axis_c_size if axis_c_size < one_vnc_line_size else one_vnc_line_size
    tp_1011_c_lp_step_in = tp_1011_c_lp_unit
    lp_c1_cnt = tdc.ceil_div(tp_1011_c_lp_unit, c0_len)
    tp_1011_c_lp_step_out = tdc.get_shape_size([lp_c1_cnt] + out_shape[dst_format.index("C") + 1:])
    tp_1011_c_step_out = tdc.get_shape_size(out_shape[dst_format.index("C") + 1:])
    c_lp_cnt = tdc.ceil_div(axis_c_size, tp_1011_c_lp_unit)
    c_left = axis_c_size % tp_1011_c_lp_unit

    # source axis left tiling parameters
    src_format_left = src_format.replace("C", "").replace(dst_format[-2], "")
    src_left_shape = []
    for idx, char in enumerate(src_format_left):
        src_left_shape.append(in_shape[src_format.index(char)])
    src_left_shape.append(1)
    axis_src_cl_size = tdc.get_shape_size(src_left_shape)
    pln_src_cl_cnt = one_vnc_line_size // tdc.ceil_fill(tp_1011_c_lp_unit, c0_len)
    tp_1011_src_cl_lp_unit = axis_src_cl_size if axis_src_cl_size < pln_src_cl_cnt else pln_src_cl_cnt
    axis_src_cl_lp_cnt = tdc.ceil_div(axis_src_cl_size, tp_1011_src_cl_lp_unit)
    axis_src_cl_left = axis_src_cl_size % tp_1011_src_cl_lp_unit
    tp_1011_src_cl_lp_step_in = tdc.get_shape_size(in_shape[-1:] + [tp_1011_src_cl_lp_unit])
    tp_1011_src_cl_lp_step_out = 0
    # parameters for output data
    for idx, chr_1 in enumerate(reversed(src_format_left)):
        src_chr_idx = src_format.index(chr_1)
        dst_chr_idx = dst_format.index(chr_1)
        tp_names["tp_1011_cl_out_idx_" + str(idx) + "_size"] = in_shape[src_chr_idx]
        tp_names["tp_1011_cl_out_idx_" + str(idx) + "_src_rsize"] = tdc.get_shape_size(src_left_shape[-1 - idx:])
        tp_names["tp_1011_cl_out_idx_" + str(idx) + "_dst_asize"] = tdc.get_shape_size(out_shape[dst_chr_idx + 1:])
    # suppose there are 2 axises
    pad_axis_cnt = FRAME_LEVEL - len(src_format_left)
    if pad_axis_cnt:
        for _, idx in enumerate(PAD_IDX_LIST[len(src_format_left):]):
            tp_names["tp_1011_cl_out_idx_" + str(idx) + "_size"] = 1
            tp_names["tp_1011_cl_out_idx_" + str(idx) + "_src_rsize"] = 1
            tp_names["tp_1011_cl_out_idx_" + str(idx) + "_dst_asize"] = 0

    r2nd_args = axis_dst_r2nd_lp_cnt, tp_1011_dst_r2nd_lp_step_in, tp_1011_dst_r2nd_lp_step_out, axis_dst_r2nd_left
    c_args = c_lp_cnt, tp_1011_c_lp_step_in, tp_1011_c_lp_step_out, c_left
    cl_args = axis_src_cl_lp_cnt, tp_1011_src_cl_lp_step_in, tp_1011_src_cl_lp_step_out, axis_src_cl_left
    (tp_1011_mc_on_cl, tp_1011_used_core_cnt, tp_1011_core_step_in, tp_1011_core_step_out, tp_1011_nlc_dst_r2nd_lp_cnt,
     tp_1011_nlc_dst_r2nd_left, tp_1011_nlc_src_cl_lp_cnt, tp_1011_nlc_src_cl_left, tp_1011_nlc_c_lp_cnt,
     tp_1011_nlc_c_left, tp_1011_lc_dst_r2nd_lp_cnt, tp_1011_lc_dst_r2nd_left, tp_1011_lc_src_cl_lp_cnt,
     tp_1011_lc_src_cl_left, tp_1011_lc_c_lp_cnt, tp_1011_lc_c_left) = _get_mc_info_positive_1011(r2nd_args,
                                                                                                  c_args, cl_args)

    sub_tiling_params = [tp_101_tiling_mode, tp_101_ub_offset, tp_1011_used_core_cnt, tp_1011_mc_on_cl,
                         tp_1011_core_step_in, tp_1011_core_step_out, tp_1011_dst_r2nd_lp_step_in,
                         tp_1011_dst_r2nd_lp_step_out, tp_1011_dst_r2nd_step_in, tp_1011_dst_r2nd_lp_unit,
                         tp_1011_src_cl_lp_step_in, tp_101_vnc_line_size, tp_1011_src_cl_lp_unit,
                         tp_1011_src_cl_lp_step_out, tp_1011_c_lp_step_in, tp_1011_c_lp_step_out, tp_1011_c_step_out,
                         tp_101_c0_size, tp_101_c_mod_c0, tp_1011_c_lp_unit, tp_1011_nlc_dst_r2nd_lp_cnt,
                         tp_1011_nlc_dst_r2nd_left, tp_1011_nlc_src_cl_lp_cnt, tp_1011_nlc_src_cl_left,
                         tp_1011_nlc_c_lp_cnt, tp_1011_nlc_c_left, tp_1011_lc_dst_r2nd_lp_cnt, tp_1011_lc_dst_r2nd_left,
                         tp_1011_lc_src_cl_lp_cnt, tp_1011_lc_src_cl_left, tp_1011_lc_c_lp_cnt, tp_1011_lc_c_left,
                         tp_names["tp_1011_cl_out_idx_0_size"], tp_names["tp_1011_cl_out_idx_0_src_rsize"],
                         tp_names["tp_1011_cl_out_idx_0_dst_asize"], tp_names["tp_1011_cl_out_idx_1_size"],
                         tp_names["tp_1011_cl_out_idx_1_src_rsize"], tp_names["tp_1011_cl_out_idx_1_dst_asize"]]
    return sub_tiling_params


# pylint: disable=redefined-builtin, too-many-statements, too-many-branches, unbalanced-tuple-unpacking
# pylint: disable=simplifiable-if-expression
def _tiling_params_positive(args):
    """
    calculate real tiling params for positive transform and last axis of source format is not c
    """

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size = args
    c0_len = out_shape[-1]  # axis c0
    axis_c_size = in_shape[-1]
    tp_names = locals()

    # get tiling params for using vnchwconv
    half_ub_size = ub_size // 2 if c0_len == tdc.C0_16 else ub_size // 4  # for int8 only can use half ub
    one_vnc_line_size = half_ub_size // tdc.VNC_LINES // block_elem_cnt * block_elem_cnt
    tmp_ub_offset = one_vnc_line_size * tdc.VNC_LINES
    tp_101_ub_offset = tmp_ub_offset if c0_len == tdc.C0_16 else tmp_ub_offset * 2
    tp_101_vnc_line_size = one_vnc_line_size
    tp_101_c_mod_c0 = axis_c_size % c0_len
    tp_101_c0_size = c0_len
    is_src_dst_r2nd_same = True if src_format[-2] == dst_format[-2] else False

    if is_src_dst_r2nd_same:  # such as NHWC -> NC1HWC0
        tiling_args_1010 = (in_shape, out_shape, src_format, dst_format, axis_c_size, one_vnc_line_size,
                            c0_len, tp_101_ub_offset, tp_101_vnc_line_size, tp_101_c0_size, tp_101_c_mod_c0)
        sub_tiling_params = _get_tiling_params_1010(tiling_args_1010)
    else:  # such as NDHWC -> DC1HWNoNiC0
        tiling_args_1011 = (in_shape, out_shape, src_format, dst_format, axis_c_size, one_vnc_line_size,
                            c0_len, tp_101_ub_offset, tp_101_vnc_line_size, tp_101_c0_size, tp_101_c_mod_c0, tp_names)
        sub_tiling_params = _get_tiling_params_1011(tiling_args_1011)

    return sub_tiling_params


def _get_tiling_params_func(args):
    """
    get tiling parameters function
    """

    in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size = args

    (in_shape_new, out_shape_new,
     in_format_new, out_format_new) = _renew_input_output_shape_format(in_shape, out_shape, src_format, dst_format)
    args_get_tp = in_shape_new, out_shape_new, in_format_new, out_format_new, block_elem_cnt, ub_size
    tiling_params = _tiling_params_positive(args_get_tp)

    return tiling_params


def _vnchwconv_pad_c_c0_align(args):
    """
    do ncdh to ndhc transform by twice vnchwconv
    """

    tik_inst, src_ub, ub_offset, vnc_col_size, pln_dst_cr_size, pl_c_size, c_mod_c0, c0_size, ele_per_block = args
    tensor_dtype = src_ub.dtype.lower()
    size_factor = tdc.get_dtype_factor(tensor_dtype)
    if tensor_dtype in ("float32", "int8", "uint8", "int32", "uint32"):
        src_ub_casted = src_ub.reinterpret_cast_to("float16")
    else:
        src_ub_casted = src_ub
    ub_offset_casted = ub_offset * size_factor
    vnc_col_len = vnc_col_size * size_factor

    # do hhc -> hch
    if tensor_dtype not in ("int8", "uint8"):
        src_addr_list = [src_ub_casted[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
        dst_addr_list = [src_ub_casted[ub_offset_casted + c0_size * i] for i in tdc.ADDR_IDX_LIST]
        repeat_cnt = tdc.ceil_div(pln_dst_cr_size * pl_c_size * size_factor, c0_size)
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
    else:
        with tik_inst.if_scope(pl_c_size % c0_size == 0):
            src_addr_list = [src_ub_casted[vnc_col_len // 2 * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [src_ub_casted[ub_offset // 2 + tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
            repeat_cnt = tdc.ceil_div(pln_dst_cr_size * pl_c_size, c0_size)
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
        with tik_inst.else_scope():
            src_addr_list = [src_ub[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
            repeat_cnt = tdc.ceil_div(pln_dst_cr_size * pl_c_size, c0_size)
            with tik_inst.new_stmt_scope():
                src_stride = tik_inst.Scalar()
                dst_stride = tik_inst.Scalar()
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(1)
                    dst_stride.set_as(32)
                dst_addr_list = [src_ub[ub_offset + c0_size * i] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
                dst_addr_list = [src_ub[ub_offset + tdc.C0_16 * ele_per_block + c0_size * i] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(False, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

    with tik_inst.if_scope(pl_c_size % c0_size == 0):
        tik_inst.data_move(src_ub, src_ub[ub_offset],
                           0, 1, tdc.ceil_div(pln_dst_cr_size * pl_c_size * tdc.C0_16, ele_per_block), 0, 0)
    with tik_inst.else_scope():  # pad c_mod_c0 to c0
        clean_len = pln_dst_cr_size * tdc.ceil_fill(pl_c_size, c0_size) * c0_size
        if tensor_dtype in ("int8", "uint8"):  # vector_dup does not support int8
            src_ub_int32 = src_ub.reinterpret_cast_to("int32")
            tdc.clean_ubuf(tik_inst, src_ub_int32, 0, tdc.ceil_div(clean_len, 4))
        else:
            tdc.clean_ubuf(tik_inst, src_ub, 0, clean_len)
        tik_inst.data_move(src_ub, src_ub[ub_offset], 0, pln_dst_cr_size,
                           pl_c_size * c0_size // ele_per_block, 0, (c0_size - c_mod_c0) * size_factor)


def _update_input_offset(args):
    """
    count input gm offset
    """

    (cl_lp_idx, dst_cl_lp_step_in, c_lp_idx, c_lp_step_in, cr_lp_idx, dst_cr_lp_step_in, core_step_in) = args

    in_offset = (cl_lp_idx * dst_cl_lp_step_in + c_lp_idx * c_lp_step_in +
                 cr_lp_idx * dst_cr_lp_step_in + core_step_in)

    return in_offset


def _update_output_offset(args):
    """
    count output gm offset
    """

    (cl_lp_idx, dst_cl_lp_step_out, c_lp_idx, c_lp_step_out, cr_lp_idx, dst_cr_lp_step_out, core_step_out) = args

    out_offset = (cl_lp_idx * dst_cl_lp_step_out + c_lp_idx * c_lp_step_out +
                  cr_lp_idx * dst_cr_lp_step_out + core_step_out)

    return out_offset


def _copy_data_in_1010(args):
    """
    copy data from gm to ub for tiling mode 1010
    """

    (tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset, dst_cr_step_in,
     vnc_line_cnt, ll_cr_size, pln_dst_cr_size, pl_c_size, vnc_col_size, ele_per_block) = args

    with tik_inst.for_range(0, vnc_line_cnt) as vnc_idx:
        with tik_inst.if_scope(vnc_idx != vnc_line_cnt - 1):
            tik_inst.data_move(src_ub[in_ub_offset + vnc_idx * vnc_col_size],
                               src_in_gm[in_gm_offset + vnc_idx * pln_dst_cr_size * dst_cr_step_in],
                               0, 1, tdc.ceil_div(pln_dst_cr_size * pl_c_size, ele_per_block), 0, 0)
        with tik_inst.else_scope():  # data maybe not enough for last line
            tik_inst.data_move(src_ub[in_ub_offset + vnc_idx * vnc_col_size],
                               src_in_gm[in_gm_offset + vnc_idx * pln_dst_cr_size * dst_cr_step_in],
                               0, 1, tdc.ceil_div(ll_cr_size * pl_c_size, ele_per_block), 0, 0)


def _gather_c0_for_out(args):
    """
    put data in hwc0 format for output
    """

    (tik_inst, src_ub, ub_offset, pl_c_size, c0_size, pln_dst_cr_size,
     c1_idx, src_stride, dst_stride, dst_gap, stride_value) = args

    repeat_cnt = pln_dst_cr_size
    tensor_dtype = src_ub.dtype.lower()
    size_factor = tdc.get_dtype_factor(tensor_dtype)
    ub_offset_casted = ub_offset * size_factor
    if tensor_dtype in ("float32", "int8", "uint8", "int32", "uint32"):
        src_ub_casted = src_ub.reinterpret_cast_to("float16")
    else:
        src_ub_casted = src_ub

    src_c1_step = c1_idx * c0_size * c0_size * size_factor
    if tensor_dtype not in ("int8", "uint8"):
        with tik_inst.for_range(0, size_factor) as factor_idx:
            src_factor_step = factor_idx * tdc.VNC_LINES * c0_size
            dst_factor_step = factor_idx * c0_size
            src_addr_list = [src_ub_casted[src_factor_step + src_c1_step + c0_size * i]
                             for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [src_ub_casted[ub_offset_casted + dst_factor_step + dst_gap * size_factor * i]
                             for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(tdc.ceil_fill(pl_c_size, c0_size) * size_factor)
                dst_stride.set_as(stride_value * size_factor)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

    else:
        with tik_inst.if_scope(pl_c_size % c0_size == 0):  # two int8 data in one line
            src_addr_list = [src_ub_casted[(src_c1_step // 2 + c0_size * i) // 2] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [src_ub_casted[(ub_offset_casted + dst_gap * i) // 2]
                             for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(tdc.ceil_fill(pl_c_size, c0_size) // 2)
                dst_stride.set_as(stride_value)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        with tik_inst.else_scope():  # one int8 data in one line
            dst_addr_list = [src_ub[ub_offset + dst_gap * i] for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(tdc.ceil_fill(pl_c_size, c0_size))
                dst_stride.set_as(stride_value)
            src_addr_list = [src_ub[src_c1_step + c0_size * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            src_addr_list = [src_ub[src_c1_step + tdc.VNC_LINES * c0_size + c0_size * i]
                             for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _gather_c0_with_gap_for_out(args):
    """
    put data in hwc0 format with gap for output
    """

    (tik_inst, src_ub, ub_offset, pl_c_size, c0_size, pln_dst_cr_size,
     cr_idx, src_stride, dst_stride, dst_gap, stride_value) = args

    repeat_cnt = tdc.ceil_div(pl_c_size, c0_size)
    tensor_dtype = src_ub.dtype.lower()
    size_factor = tdc.get_dtype_factor(tensor_dtype)
    ub_offset_casted = ub_offset * size_factor
    if tensor_dtype in ("float32", "int8", "uint8", "int32", "uint32"):
        src_ub_casted = src_ub.reinterpret_cast_to("float16")
    else:
        src_ub_casted = src_ub

    src_c1_step = cr_idx * tdc.ceil_fill(pl_c_size, c0_size) * c0_size * size_factor
    if tensor_dtype not in ("int8", "uint8"):
        with tik_inst.for_range(0, size_factor) as factor_idx:
            src_factor_step = factor_idx * tdc.VNC_LINES * c0_size
            dst_factor_step = factor_idx * c0_size
            src_addr_list = [src_ub_casted[src_factor_step + src_c1_step + c0_size * i]
                             for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [src_ub_casted[ub_offset_casted + dst_factor_step + dst_gap * size_factor * i]
                             for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(c0_size * size_factor)
                dst_stride.set_as(stride_value * size_factor)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

    else:
        with tik_inst.if_scope(pl_c_size % c0_size == 0):  # two int8 data in one line
            src_addr_list = [src_ub_casted[(src_c1_step // 2 + c0_size * i) // 2] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [src_ub_casted[(ub_offset_casted + dst_gap * i) // 2]
                             for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(c0_size // 2)
                dst_stride.set_as(stride_value)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        with tik_inst.else_scope():  # one int8 data in one line
            dst_addr_list = [src_ub[ub_offset + dst_gap * i] for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(c0_size)
                dst_stride.set_as(stride_value)
            src_addr_list = [src_ub[src_c1_step + c0_size * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            src_addr_list = [src_ub[src_c1_step + tdc.VNC_LINES * c0_size + c0_size * i]
                             for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


# pylint: disable=unused-variable
def _copy_data_out_1010(copy_out_args):
    """
    copy data from ub to gm for tiling mode 1010
    """

    (tik_inst, dst_out_gm, src_ub, ub_offset, c_step_out, vnc_line_cnt,
     pln_dst_cr_size, ll_cr_size, pl_c_size, c0_size, ele_per_block) = copy_out_args
    vnc_dst_col_size = pln_dst_cr_size * c0_size
    repeat_stride = pln_dst_cr_size * tdc.VNC_LINES * c0_size
    out_crc_cnt = ((vnc_line_cnt - 1) * pln_dst_cr_size + ll_cr_size) * c0_size
    dst_stride_gap = (c_step_out - out_crc_cnt) // ele_per_block

    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        c1_cnt = tdc.ceil_div(pl_c_size, c0_size)

        with tik_inst.if_scope(pln_dst_cr_size > c1_cnt):
            with tik_inst.for_range(0, c1_cnt) as c1_idx:
                gather_c0_args = (tik_inst, src_ub, ub_offset + c1_idx * repeat_stride, pl_c_size,
                                  c0_size, pln_dst_cr_size, c1_idx, src_stride, dst_stride, vnc_dst_col_size, 1)
                _gather_c0_for_out(gather_c0_args)  # transfer hc0h to hhc0
        with tik_inst.else_scope():
            with tik_inst.for_range(0, pln_dst_cr_size) as cr_idx:
                gather_c0_args = (tik_inst, src_ub, ub_offset + cr_idx * c0_size, pl_c_size, c0_size, pln_dst_cr_size,
                                  cr_idx, src_stride, dst_stride, vnc_dst_col_size, repeat_stride // c0_size)
                _gather_c0_with_gap_for_out(gather_c0_args)  # transfer hc0h to hhc0

        with tik_inst.if_scope(dst_stride_gap > tdc.STRIDE_LIMIT_MTE):
            with tik_inst.new_stmt_scope(disable_sync=True):
                with tik_inst.for_range(0, c1_cnt) as c1_idx_1:
                    tik_inst.data_move(dst_out_gm[c1_idx_1 * c_step_out], src_ub[ub_offset + c1_idx_1 * repeat_stride],
                                       0, 1, out_crc_cnt // ele_per_block, 0, 0)
        with tik_inst.else_scope():
            tik_inst.data_move(dst_out_gm[0], src_ub[ub_offset],
                               0, c1_cnt, out_crc_cnt // ele_per_block,
                               (repeat_stride - out_crc_cnt) // ele_per_block, dst_stride_gap)


def _func_transform_1010(tensor_args, tp_args):
    """
    transform function for tiling mode 100
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, ele_per_block = tensor_args
    (ub_offset, used_core_cnt, core_step_in, core_step_out, dst_cl_lp_step_in, dst_cl_lp_step_out, dst_cl_lp_unit,
     dst_cr_lp_step_in, dst_cr_lp_step_out, dst_cr_step_in, vnc_col_size, pln_dst_cr_size, vnc_row_size,
     c_lp_step_in, c_lp_step_out, c_step_out, c0_size, c_mod_c0, c_lp_unit, nlc_dst_cl_lp_cnt, nlc_dst_cr_lp_cnt,
     nlc_vnc_row_left, nlc_last_line_cr_cnt, nlc_c_lp_cnt, nlc_c_left, lc_dst_cl_lp_cnt, lc_dst_cr_lp_cnt,
     lc_vnc_row_left, lc_last_line_cr_cnt, lc_c_lp_cnt, lc_c_left) = tp_args

    def _inner_func(args):
        dst_cl_lp_cnt, dst_cr_lp_cnt, vnc_row_left, last_line_cr_cnt, c_lp_cnt, c_left = args
        pl_c_size = tik_inst.Scalar(name="pl_c_size")
        vnc_line_cnt = tik_inst.Scalar(name="vnc_line_cnt")
        ll_cr_size = tik_inst.Scalar(name="ll_cr_size")

        with tik_inst.for_range(0, dst_cl_lp_cnt) as cl_lp_idx:
            with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:
                with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                    pl_c_size.set_as(c_lp_unit)
                with tik_inst.else_scope():
                    pl_c_size.set_as(c_left)

                with tik_inst.for_range(0, dst_cr_lp_cnt) as cr_lp_idx:
                    with tik_inst.if_scope(tik.any(cr_lp_idx != dst_cr_lp_cnt - 1, vnc_row_left == 0)):
                        vnc_line_cnt.set_as(vnc_row_size)
                        ll_cr_size.set_as(pln_dst_cr_size)
                    with tik_inst.else_scope():
                        vnc_line_cnt.set_as(vnc_row_left)
                        ll_cr_size.set_as(last_line_cr_cnt)

                    in_offset_args = (cl_lp_idx, dst_cl_lp_step_in, c_lp_idx, c_lp_step_in,
                                      cr_lp_idx, dst_cr_lp_step_in, block_idx * core_step_in)
                    in_gm_offset = _update_input_offset(in_offset_args)
                    in_ub_offset = 0
                    copy_in_args = (tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset, dst_cr_step_in,
                                    vnc_line_cnt, ll_cr_size, pln_dst_cr_size, pl_c_size, vnc_col_size, ele_per_block)
                    with tik_inst.new_stmt_scope(disable_sync=True):
                        _copy_data_in_1010(copy_in_args)

                    vnc_args = (tik_inst, src_ub, ub_offset, vnc_col_size,
                                pln_dst_cr_size, pl_c_size, c_mod_c0, c0_size, ele_per_block)
                    _vnchwconv_pad_c_c0_align(vnc_args)
                    out_gm_args = (cl_lp_idx, dst_cl_lp_step_out, c_lp_idx, c_lp_step_out,
                                   cr_lp_idx, dst_cr_lp_step_out, block_idx * core_step_out)
                    out_gm_offset = _update_output_offset(out_gm_args)
                    copy_out_args = (tik_inst, dst_out_gm[out_gm_offset], src_ub, ub_offset, c_step_out, vnc_line_cnt,
                                     pln_dst_cr_size, ll_cr_size, pl_c_size, c0_size, ele_per_block)
                    _copy_data_out_1010(copy_out_args)

    with tik_inst.if_scope(block_idx != used_core_cnt - 1):
        nlc_args = (nlc_dst_cl_lp_cnt, nlc_dst_cr_lp_cnt, nlc_vnc_row_left,
                    nlc_last_line_cr_cnt, nlc_c_lp_cnt, nlc_c_left)
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = (lc_dst_cl_lp_cnt, lc_dst_cr_lp_cnt, lc_vnc_row_left, lc_last_line_cr_cnt, lc_c_lp_cnt, lc_c_left)
        _inner_func(lc_args)


def _copy_data_in_1011(args):
    """
    copy data from gm to ub for tiling mode 1011
    """

    (tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset, vnc_line_cnt,
     dst_r2nd_step_in, pln_cl_size, pl_c_size, vnc_col_size, ele_per_block) = args

    with tik_inst.for_range(0, vnc_line_cnt) as vnc_idx:
        tik_inst.data_move(src_ub[in_ub_offset + vnc_idx * vnc_col_size],
                           src_in_gm[in_gm_offset + vnc_idx * dst_r2nd_step_in],
                           0, 1, tdc.ceil_div(pln_cl_size * pl_c_size, ele_per_block), 0, 0)


def _get_cl_offset(cl_offset_args, cl_idx):
    """
    count dhw output offset
    """

    (cl_out_idx_0_size, cl_out_idx_0_src_rsize, cl_out_idx_0_dst_asize,
     cl_out_idx_1_size, cl_out_idx_1_src_rsize, cl_out_idx_1_dst_asize) = cl_offset_args

    cl_out_offset = (cl_idx // cl_out_idx_0_src_rsize % cl_out_idx_0_size * cl_out_idx_0_dst_asize +
                     cl_idx // cl_out_idx_1_src_rsize % cl_out_idx_1_size * cl_out_idx_1_dst_asize)

    return cl_out_offset


# pylint: disable=unused-variable
def _copy_data_out_1011(copy_out_args, cl_offset_args):
    """
    copy data from ub to gm for tiling mode 1011
    """

    (tik_inst, dst_out_gm, src_ub, ub_offset, c_step_out, vnc_line_cnt, cl_lp_idx, nlc_src_cl_lp_cnt,
     src_cl_lp_unit, pln_cl_size, pl_c_size, c0_size, ele_per_block, mc_on_cl, block_idx) = copy_out_args

    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        cur_cl_idx = tik_inst.Scalar()
        c1_cnt = tdc.ceil_div(pl_c_size, c0_size)
        with tik_inst.for_range(0, c1_cnt) as c1_idx:
            gather_c0_args = (tik_inst, src_ub, ub_offset, pl_c_size, c0_size,
                              pln_cl_size, c1_idx, src_stride, dst_stride, c0_size, vnc_line_cnt)
            _gather_c0_for_out(gather_c0_args)  # transfer dhc0n to dhnc0

            with tik_inst.for_range(0, pln_cl_size) as cl_idx:  # dhwnc0, move out nc0 per loop
                with tik_inst.if_scope(mc_on_cl == 1):
                    cur_cl_idx.set_as((cl_lp_idx + block_idx * nlc_src_cl_lp_cnt) * src_cl_lp_unit + cl_idx)
                with tik_inst.else_scope():
                    cur_cl_idx.set_as(cl_lp_idx * src_cl_lp_unit + cl_idx)
                cl_out_offset = _get_cl_offset(cl_offset_args, cur_cl_idx)
                tik_inst.data_move(dst_out_gm[c1_idx * c_step_out + cl_out_offset],
                                   src_ub[ub_offset + cl_idx * vnc_line_cnt * c0_size],
                                   0, 1, vnc_line_cnt * c0_size // ele_per_block, 0, 0)


def _func_transform_1011(tensor_args, tp_args):
    """
    transform function for tiling mode 100
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, ele_per_block = tensor_args
    (ub_offset, used_core_cnt, mc_on_cl, core_step_in, core_step_out, dst_r2nd_lp_step_in, dst_r2nd_lp_step_out,
     dst_r2nd_step_in, dst_r2nd_lp_unit, src_cl_lp_step_in, vnc_col_size, src_cl_lp_unit, src_cl_lp_step_out,
     c_lp_step_in, c_lp_step_out, c_step_out, c0_size, c_mod_c0, c_lp_unit, nlc_dst_r2nd_lp_cnt, nlc_dst_r2nd_left,
     nlc_src_cl_lp_cnt, nlc_src_cl_left, nlc_c_lp_cnt, nlc_c_left, lc_dst_r2nd_lp_cnt, lc_dst_r2nd_left,
     lc_src_cl_lp_cnt, lc_src_cl_left, lc_c_lp_cnt, lc_c_left, cl_out_idx_0_size, cl_out_idx_0_src_rsize,
     cl_out_idx_0_dst_asize, cl_out_idx_1_size, cl_out_idx_1_src_rsize, cl_out_idx_1_dst_asize) = tp_args

    def _inner_func(args):
        dst_r2nd_lp_cnt, dst_r2nd_left, src_cl_lp_cnt, src_cl_left, c_lp_cnt, c_left = args
        pl_c_size = tik_inst.Scalar(name="pl_c_size")
        vnc_line_cnt = tik_inst.Scalar(name="vnc_line_cnt")
        pln_cl_size = tik_inst.Scalar(name="pln_cl_size")

        with tik_inst.for_range(0, src_cl_lp_cnt) as cl_lp_idx:
            with tik_inst.if_scope(tik.any(cl_lp_idx != src_cl_lp_cnt - 1, src_cl_left == 0)):
                pln_cl_size.set_as(src_cl_lp_unit)
            with tik_inst.else_scope():
                pln_cl_size.set_as(src_cl_left)

            with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:
                with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                    pl_c_size.set_as(c_lp_unit)
                with tik_inst.else_scope():
                    pl_c_size.set_as(c_left)

                with tik_inst.for_range(0, dst_r2nd_lp_cnt) as r2nd_lp_idx:
                    with tik_inst.if_scope(tik.any(r2nd_lp_idx != dst_r2nd_lp_cnt - 1, dst_r2nd_left == 0)):
                        vnc_line_cnt.set_as(dst_r2nd_lp_unit)
                    with tik_inst.else_scope():
                        vnc_line_cnt.set_as(dst_r2nd_left)

                    in_offset_args = (cl_lp_idx, src_cl_lp_step_in, c_lp_idx, c_lp_step_in,
                                      r2nd_lp_idx, dst_r2nd_lp_step_in, block_idx * core_step_in)
                    in_gm_offset = _update_input_offset(in_offset_args)
                    in_ub_offset = 0
                    copy_in_args = (tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset, vnc_line_cnt,
                                    dst_r2nd_step_in, pln_cl_size, pl_c_size, vnc_col_size, ele_per_block)
                    with tik_inst.new_stmt_scope(disable_sync=True):
                        _copy_data_in_1011(copy_in_args)

                    vnc_args = (tik_inst, src_ub, ub_offset, vnc_col_size,
                                pln_cl_size, pl_c_size, c_mod_c0, c0_size, ele_per_block)
                    _vnchwconv_pad_c_c0_align(vnc_args)

                    out_gm_args = (cl_lp_idx, src_cl_lp_step_out, c_lp_idx, c_lp_step_out,
                                   r2nd_lp_idx, dst_r2nd_lp_step_out, block_idx * core_step_out)
                    out_gm_offset = _update_output_offset(out_gm_args)
                    copy_out_args = (tik_inst, dst_out_gm[out_gm_offset], src_ub, ub_offset, c_step_out,
                                     vnc_line_cnt, cl_lp_idx, nlc_src_cl_lp_cnt, src_cl_lp_unit, pln_cl_size,
                                     pl_c_size, c0_size, ele_per_block, mc_on_cl, block_idx)
                    cl_offset_args = (cl_out_idx_0_size, cl_out_idx_0_src_rsize, cl_out_idx_0_dst_asize,
                                      cl_out_idx_1_size, cl_out_idx_1_src_rsize, cl_out_idx_1_dst_asize)
                    _copy_data_out_1011(copy_out_args, cl_offset_args)

    with tik_inst.if_scope(block_idx != used_core_cnt - 1):
        nlc_args = (nlc_dst_r2nd_lp_cnt, nlc_dst_r2nd_left,
                    nlc_src_cl_lp_cnt, nlc_src_cl_left, nlc_c_lp_cnt, nlc_c_left)
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = (lc_dst_r2nd_lp_cnt, lc_dst_r2nd_left,
                   lc_src_cl_lp_cnt, lc_src_cl_left, lc_c_lp_cnt, lc_c_left)
        _inner_func(lc_args)


def trans_data_positive_source_tc(src, dst, src_format, dst_format, kernel_name="trans_data_positive_source_tc"):
    """
    positive transform for last dimension of source format is c

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
        kernel name, default value is "trans_data_positive_source_tc"

    Returns
    -------
    None
    """

    src_format = src_format.upper()
    dst_format = dst_format.upper()
    in_shape = list(src.get("shape"))
    out_shape = list(dst.get("shape"))
    in_dtype = src.get("dtype").lower() if src.get("dtype").lower() != "bfloat16" else "float16"
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1)
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)

    # get tiling parameters
    args = in_shape, out_shape, src_format, dst_format, block_elem_cnt, ub_size
    tiling_params = _get_tiling_params_func(args)

    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "src_in_gm")
    if src_format in ("NHWC", "NDHWC") and dst_format in ("NC1HWC0", "NDC1HWC0"):
        dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm")
    else:
        dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm", is_atomic_add=True)
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tik.scope_ubuf, "total_ub")

    tiling_mode = tiling_params[0]
    used_core_cnt = tiling_params[2]
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt]
            with tik_inst.if_scope(tiling_mode == 1010):
                tp_args = tiling_params[1:-6]
                _func_transform_1010(tensor_args, tp_args)
            with tik_inst.if_scope(tiling_mode == 1011):
                tp_args = tiling_params[1:]
                _func_transform_1011(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[src_in_gm], outputs=[dst_out_gm])
