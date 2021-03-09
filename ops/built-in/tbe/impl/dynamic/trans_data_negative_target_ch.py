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

from __future__ import absolute_import
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from .. import trans_data_common_func as tdc
from impl.util.platform_adapter import tbe_context

# used for tiling data
TILING_CTRL_PARAM = ("int64", 128)

def _get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes):
    """
    get tiling parameters function
    """

    ele_per_block = tdc.BLOCK_BYTE_SIZE // tiling_dtype_bytes
    tik_inst.data_move(tiling_ub, tiling_gm, 0, 1, TILING_CTRL_PARAM[1] // ele_per_block, 0, 0)
    for reg_idx in range(TILING_CTRL_PARAM[1]):
        tiling_params[reg_idx].set_as(tiling_ub[reg_idx])

def _twice_vnchwconv_invert(args):
    """
    do dhc to cdh transform by twice vnchwconv
    """

    tik_inst, src_ub, tmp_ub_offset, ub_offset, c_plp_size, cr_pln_size, cl_plp_size, ele_per_block = args
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
     vnc_line_size, cl_in_idx_0_size, cl_in_idx_0_dst_rsize, cl_in_idx_0_src_asize, cl_in_idx_1_size, cl_in_idx_1_dst_rsize,
     cl_in_idx_1_src_asize, cr_in_idx_0_size, cr_in_idx_0_dst_rsize, cr_in_idx_0_src_asize,
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
                                               0 < lc_cr_left, lc_cr_left < ele_per_block)):
                    cr_pln_size.set_as(src_cr_lp_unit - ele_per_block)
                with tik_inst.else_scope():
                    cr_pln_size.set_as(src_cr_lp_unit)
                is_cr_back.set_as(0)

            with tik_inst.else_scope():
                with tik_inst.if_scope(tik.all(0 < cr_left, cr_left < ele_per_block)):
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

                    vnc_args = (tik_inst, src_ub, tmp_ub_offset, ub_offset,
                                c_plp_size, cr_pln_size, cl_plp_size, ele_per_block)
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

# pylint: disable=unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_STR, para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def trans_data_negative_target_ch(src, dst, src_format, dst_format, kernel_name="trans_data_negative_target_tc"):
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
    in_dtype = src.get("dtype").lower()
    in_dtype_bytes = tdc.get_dtype_len(in_dtype)
    tiling_dtype_bytes = tdc.get_dtype_len("int64")
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1) - TILING_CTRL_PARAM[1] * tiling_dtype_bytes // in_dtype_bytes
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)

    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "src_in_gm")
    dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm")
    tiling_gm = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],), tik.scope_gm, "tiling_gm")
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tik.scope_ubuf, "total_ub")
    tiling_ub = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],), tik.scope_ubuf, "tiling_ub")
    tiling_params = [tik_inst.Scalar(TILING_CTRL_PARAM[0]) for i in range(TILING_CTRL_PARAM[1])]
    _get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes)

    tiling_mode = tiling_params[0]
    used_core_cnt = tiling_params[4]
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            with tik_inst.if_scope(tiling_mode == 200):
                tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt]
                tp_args = tiling_params[1:45]
                _func_transform_200(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[src_in_gm], outputs=[dst_out_gm], flowtable=[tiling_gm], enable_l2=False)
    tbe_context.get_context().add_compile_info("vars", {"srcFormat": src_format,
                                    "dstFormat": dst_format,
                                    "dType": in_dtype,
                                    "ubSize": ub_size,
                                    "blockDim": tdc.CORE_DIM_NUM,
                                    "inputSize": -1,
                                    "hiddenSize": -1,
                                    "group": 1})
