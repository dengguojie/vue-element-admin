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
trans_data_positive_source_ntc
"""

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tik

from impl import trans_data_common_func as tdc


# 'pylint: disable=too-few-public-methods
class Constant:
    # used for tiling data
    TILING_CTRL_PARAM = ("int64", 96)


def _get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes):
    """
    get tiling parameters function
    """

    ele_per_block = tdc.BLOCK_BYTE_SIZE // tiling_dtype_bytes
    tik_inst.data_move(tiling_ub, tiling_gm, 0, 1,
                       Constant.TILING_CTRL_PARAM[1] // ele_per_block, 0, 0)
    for reg_idx in range(Constant.TILING_CTRL_PARAM[1]):
        tiling_params[reg_idx].set_as(tiling_ub[reg_idx])


# 'pylint: disable=unused-variable, too-many-locals, too-many-statements
def _twice_vnchwconv_invert(args):
    """
    do ncdh to ndhc transform by twice vnchwconv
    """

    (tik_inst, src_ub, mc_pos, dst_ub, dst_ub_offset, vnc_col_size, plp_c_size, r1st_src_r2nd_dst_same,
     cl_lp_idx, plp_cl_size, cr_lp_cnt, plp_cr_size, c_mod_c0, c0_size, ele_per_block) = args
    tensor_dtype = src_ub.dtype.lower()
    size_factor = tdc.get_dtype_factor(tensor_dtype)
    vnc_col_len = vnc_col_size * size_factor
    dst_ub_offset_casted = dst_ub_offset * size_factor
    if tensor_dtype in ("float32", "int32", "uint32"):
        src_ub_casted = src_ub.reinterpret_cast_to("float16")
        dst_ub_casted = dst_ub.reinterpret_cast_to("float16")
    else:
        src_ub_casted = src_ub
        dst_ub_casted = dst_ub

    # do ncdh -> cdhn
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        plp_cr_block_align_size = tik_inst.Scalar(name="plp_cr_block_align_size")
        with tik_inst.if_scope(tik.all(cr_lp_cnt == 1, mc_pos != 2)):
            plp_cr_block_align_size.set_as(plp_cr_size)
        with tik_inst.else_scope():
            plp_cr_block_align_size.set_as(tdc.ceil_fill(plp_cr_size, ele_per_block))

        repeat_cnt = tdc.ceil_div(plp_c_size * plp_cr_block_align_size * size_factor, c0_size)
        if tensor_dtype not in ("int8", "uint8"):
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
                dst_stride.set_as(16)
            src_addr_list = [src_ub_casted[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub_casted[dst_ub_offset_casted + c0_size * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        else:
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
                dst_stride.set_as(32)
            src_addr_list = [src_ub[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[dst_ub_offset + c0_size * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            dst_addr_list = [dst_ub[dst_ub_offset + (tdc.C0_16 + i) * c0_size] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        # padding zero
        clean_len = plp_cr_size * tdc.ceil_fill(plp_c_size, c0_size) * c0_size
        with tik_inst.if_scope(plp_c_size < c0_size):
            if tensor_dtype in ("int8", "uint8"):
                src_ub_int32 = src_ub.reinterpret_cast_to("int32")
                tdc.clean_ubuf(tik_inst, src_ub_int32, 0, tdc.ceil_div(clean_len, 4))
            else:
                tdc.clean_ubuf(tik_inst, src_ub, 0, clean_len)
        # do cdhn -> dhcn
        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, plp_c_size) as c_idx:
                tik_inst.data_move(src_ub[c_idx * c0_size],
                                   dst_ub[dst_ub_offset + c_idx*plp_cr_block_align_size*c0_size],
                                   0, plp_cr_size, 1 * size_factor, 0, (c0_size - 1) * size_factor)

        # do dhcn -> ndhc or dhcn -> dhnc
        if tensor_dtype not in ("int8", "uint8"):
            with tik_inst.if_scope(r1st_src_r2nd_dst_same == 0):  # for NCHW -> C1HWNoNiC0
                with tik_inst.for_range(0, size_factor) as factor_idx:
                    src_addr_list = [src_ub_casted[tdc.C0_16 * (i + c0_size * factor_idx)] for i in tdc.ADDR_IDX_LIST]
                    dst_addr_list = [dst_ub_casted[dst_ub_offset_casted + (size_factor * i + factor_idx) * c0_size]
                                     for i in tdc.ADDR_IDX_LIST]
                    repeat_cnt = plp_cr_size
                    with tik_inst.if_scope(repeat_cnt == 1):
                        src_stride.set_as(0)
                        dst_stride.set_as(0)
                    with tik_inst.else_scope():
                        src_stride.set_as(16 * size_factor)
                        dst_stride.set_as(plp_cl_size * size_factor)
                    tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            with tik_inst.else_scope():  # for NCHW -> NC1HWC0
                vnc_row_size = plp_cr_size * size_factor * c0_size
                src_addr_list = [src_ub_casted[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
                dst_addr_list = [dst_ub_casted[dst_ub_offset_casted + vnc_row_size * i] for i in tdc.ADDR_IDX_LIST]
                repeat_cnt = plp_cr_size * size_factor
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(16)
                    dst_stride.set_as(1)
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        else:
            repeat_cnt = plp_cr_size
            with tik_inst.if_scope(r1st_src_r2nd_dst_same == 0):  # for NCHW -> C1HWNoNiC0
                dst_addr_list = [dst_ub[dst_ub_offset + c0_size * i] for i in tdc.ADDR_IDX_LIST]
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(32)
                    dst_stride.set_as(plp_cl_size)
                src_addr_list = [src_ub[c0_size * i] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
                src_addr_list = [src_ub[c0_size * (i + tdc.C0_16)] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            with tik_inst.else_scope():  # for NCHW -> NC1HWC0
                vnc_row_size = plp_cr_size * c0_size
                dst_addr_list = [dst_ub[dst_ub_offset + vnc_row_size * i] for i in tdc.ADDR_IDX_LIST]
                with tik_inst.if_scope(repeat_cnt == 1):
                    src_stride.set_as(0)
                    dst_stride.set_as(0)
                with tik_inst.else_scope():
                    src_stride.set_as(32)
                    dst_stride.set_as(1)
                src_addr_list = [src_ub[c0_size * i] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
                src_addr_list = [src_ub[c0_size * (i + tdc.C0_16)] for i in tdc.ADDR_IDX_LIST]
                tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


# 'pylint: disable=unused-variable, too-many-locals, too-many-statements
def _once_vnchwconv_invert(args):
    """
    do cdh to dhc transform by once vnchwconv
    """

    (tik_inst, src_ub, mc_pos, dst_ub, dst_ub_offset, vnc_col_size, plp_c_size, r1st_src_r2nd_dst_same,
     cl_lp_idx, plp_cl_size, cr_lp_cnt, plp_cr_size, c_mod_c0, c0_size, ele_per_block) = args
    tensor_dtype = src_ub.dtype.lower()
    size_factor = tdc.get_dtype_factor(tensor_dtype)
    repeat_cnt = tdc.ceil_div(plp_cr_size, c0_size)
    if tensor_dtype in ("float32", "int32", "uint32"):  # to avoid compile error
        src_ub = src_ub.reinterpret_cast_to("float16")
        dst_ub = dst_ub.reinterpret_cast_to("float16")
        if vnc_col_size % 32 > 0:  # to avoid compile error
            vnc_col_size = 32

    # do cdh -> dhc
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        dst_gap = tik_inst.Scalar()
        if tensor_dtype not in ("int8", "uint8"):
            with tik_inst.if_scope(tik.all(plp_c_size % c0_size > 0, cl_lp_idx == 0)):  # padding zero
                tdc.clean_ubuf(tik_inst, src_ub, c_mod_c0 * vnc_col_size, (c0_size - c_mod_c0) * vnc_col_size)

            with tik_inst.if_scope(r1st_src_r2nd_dst_same == 1):  # for NCHW -> NC1HWC0
                dst_gap.set_as(c0_size)
                dst_stride.set_as(16)
            with tik_inst.else_scope():  # for NCHW -> C1HWNoNiC0
                dst_gap.set_as(plp_cl_size * c0_size)
                dst_stride.set_as(plp_cl_size * 16)

            src_addr_list = [src_ub[vnc_col_size * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * size_factor * i] for i in tdc.ADDR_IDX_LIST]
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        else:
            with tik_inst.if_scope(tik.all(plp_c_size % c0_size > 0, cl_lp_idx == 0)):  # padding zero
                src_ub_int32 = src_ub.reinterpret_cast_to("int32")
                tdc.clean_ubuf(tik_inst, src_ub_int32, c_mod_c0 * vnc_col_size // 4,
                               (c0_size - c_mod_c0) * vnc_col_size // 4)

            with tik_inst.if_scope(r1st_src_r2nd_dst_same == 1):  # for NCHW -> NC1HWC0
                dst_gap.set_as(c0_size)
                dst_stride.set_as(32)
            with tik_inst.else_scope():  # for NCHW -> C1HWNoNiC0
                dst_gap.set_as(plp_cl_size * c0_size)
                dst_stride.set_as(plp_cl_size * 32)

            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)

            # for target low half
            src_addr_list = [src_ub[vnc_col_size * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * (i + tdc.C0_16)] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            # for target high half
            src_addr_offset = tdc.NI_16 * vnc_col_size
            src_addr_list = [src_ub[src_addr_offset + vnc_col_size * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            dst_addr_list = [dst_ub[dst_ub_offset + dst_gap * (i + tdc.C0_16)] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(True, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)


def _update_input_offset(args):
    """
    count input gm offset
    """

    cl_lp_idx, cl_lp_step_in, c_lp_idx, c_lp_step_in, cr_lp_idx, cr_lp_step_in, core_step_in = args

    in_offset = (cl_lp_idx * cl_lp_step_in + c_lp_idx * c_lp_step_in + cr_lp_idx * cr_lp_step_in + core_step_in)

    return in_offset


# 'pylint: disable=too-many-locals
def _copy_data_in(args):
    """
    copy data from gm to ub
    """

    (tik_inst, src_in_gm, src_ub, mc_pos, in_gm_offset, vnc_col_size, tiling_mode, plp_cl_size,
     cl_step_in, plp_c_size, c_step_in, cr_lp_cnt, plp_cr_size, ele_per_block) = args
    cr_block_align_size = tdc.ceil_fill(plp_cr_size, ele_per_block)
    cr_blocks = tdc.ceil_div(plp_cr_size, ele_per_block)
    cr_gm_stride = (c_step_in - plp_cr_size) // ele_per_block
    cr_ub_stride = (vnc_col_size - plp_cr_size) // ele_per_block

    with tik_inst.new_stmt_scope(disable_sync=True):
        with tik_inst.if_scope(tiling_mode == 1000):  # for two times vnchwconv case
            with tik_inst.if_scope(tik.all(cr_lp_cnt == 1, mc_pos != 2)):
                with tik_inst.for_range(0, plp_cl_size) as cl_idx:
                    tik_inst.data_move(src_ub[cl_idx * vnc_col_size], src_in_gm[in_gm_offset + cl_idx * cl_step_in],
                                       0, 1, tdc.ceil_div(plp_c_size * plp_cr_size, ele_per_block), 0, 0)
            with tik_inst.elif_scope(tik.all(c_step_in % ele_per_block == 0, cr_gm_stride <= tdc.STRIDE_LIMIT_MTE)):
                with tik_inst.for_range(0, plp_cl_size) as cl_idx:
                    tik_inst.data_move(src_ub[cl_idx * vnc_col_size], src_in_gm[in_gm_offset + cl_idx * cl_step_in],
                                       0, plp_c_size, cr_blocks, cr_gm_stride, 0)
            with tik_inst.else_scope():
                with tik_inst.for_range(0, plp_cl_size) as cl_idx:
                    with tik_inst.for_range(0, plp_c_size) as c_idx:
                        tik_inst.data_move(src_ub[cl_idx * vnc_col_size + c_idx * cr_block_align_size],
                                           src_in_gm[in_gm_offset + cl_idx * cl_step_in + c_idx * c_step_in],
                                           0, 1, cr_blocks, 0, 0)
        with tik_inst.else_scope():  # for one time vnchwconv case
            with tik_inst.if_scope(tik.all(c_step_in % ele_per_block == 0, cr_gm_stride <= tdc.STRIDE_LIMIT_MTE)):
                tik_inst.data_move(src_ub, src_in_gm[in_gm_offset],
                                   0, plp_c_size, cr_blocks, cr_gm_stride, cr_ub_stride)
            with tik_inst.else_scope():
                with tik_inst.for_range(0, plp_c_size) as c_idx_1:
                    tik_inst.data_move(src_ub[c_idx_1 * vnc_col_size], src_in_gm[in_gm_offset + c_idx_1 * c_step_in],
                                       0, 1, cr_blocks, 0, 0)


# 'pylint: disable=too-many-locals
def _update_out_offset_cl(args):
    """
    update c-left out offset
    """

    (cl_offset, cl_base, cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
     cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize) = args

    cl_offset.set_as(cl_base // cl_out_idx_0_dst_rsize % cl_out_idx_0_size * cl_out_idx_0_dst_asize +
                     cl_base // cl_out_idx_1_dst_rsize % cl_out_idx_1_size * cl_out_idx_1_dst_asize)


def _move_data_out(args):
    (tik_inst, dst_out_gm, out_gm_offset, dst_ub, out_ub_offset,
     loop_cnt, burst_len, gm_step, ub_step, ele_per_block) = args
    out_gm_gap = (gm_step - burst_len) // ele_per_block
    out_ub_gap = (ub_step - burst_len) // ele_per_block
    with tik_inst.if_scope(out_gm_gap <= tdc.STRIDE_LIMIT_MTE):
        tik_inst.data_move(dst_out_gm[out_gm_offset], dst_ub[out_ub_offset],
                           0, loop_cnt, burst_len // ele_per_block, out_ub_gap, out_gm_gap)
    with tik_inst.else_scope():
        with tik_inst.for_range(0, loop_cnt) as last_cl_idx:
            tik_inst.data_move(dst_out_gm[out_gm_offset + last_cl_idx*gm_step],
                               dst_ub[out_ub_offset + last_cl_idx * ub_step], 0, 1, burst_len // ele_per_block, 0, 0)


def _count_out_cr_idx(args):
    """
    get c-right index
    """

    (tik_inst, block_idx, mc_pos, cr_index, cr_idx, cr_lp_idx, nlc_cr_lp_cnt, cr_lp_unit) = args
    with tik_inst.if_scope(mc_pos == 2):
        cr_index.set_as(cr_idx + (cr_lp_idx + block_idx * nlc_cr_lp_cnt) * cr_lp_unit)
    with tik_inst.else_scope():
        cr_index.set_as(cr_idx + cr_lp_idx * cr_lp_unit)


def _update_out_offset_cr(args):
    """
    update c-right out offset
    """

    (cr_offset, cr_index, cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
     cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize) = args

    cr_offset.set_as(cr_index // cr_out_idx_0_dst_rsize % cr_out_idx_0_size * cr_out_idx_0_dst_asize +
                     cr_index // cr_out_idx_1_dst_rsize % cr_out_idx_1_size * cr_out_idx_1_dst_asize)


def _inner_move_data_out_target_two_cl(args):
    """
    move data out process for target format has two c-left dims
    """
    (tik_inst, dst_out_gm, gm_out_offset, dst_ub, ub_out_offset, last_cr_cl_size, is_last_cr_cl_nz, mid_lp_cnt,
     cur_cr_cl_size, burst_len, idx_0_dim_asize, idx_0_dim_size, idx_1_dim_asize,
     renew_idx_0_dim_size, ele_per_block) = args
    with tik_inst.if_scope(last_cr_cl_size > 0):
        last_cl_1_cr_2_args = (tik_inst, dst_out_gm, gm_out_offset, dst_ub, ub_out_offset, last_cr_cl_size,
                               burst_len, idx_0_dim_asize, burst_len, ele_per_block)
        _move_data_out(last_cl_1_cr_2_args)
        is_last_cr_cl_nz.set_as(1)
    with tik_inst.else_scope():
        is_last_cr_cl_nz.set_as(0)

    last_gm_end_offset = is_last_cr_cl_nz * ((last_cr_cl_size - idx_0_dim_size)*idx_0_dim_asize + idx_1_dim_asize)
    last_ub_end_offset = last_cr_cl_size * burst_len + ub_out_offset
    with tik_inst.if_scope(mid_lp_cnt > 0):
        with tik_inst.for_range(0, mid_lp_cnt) as mid_cl_idx:
            mid_cl_gm_offset = mid_cl_idx * idx_1_dim_asize + gm_out_offset + last_gm_end_offset
            mid_cl_ub_offset = mid_cl_idx * renew_idx_0_dim_size * burst_len + last_ub_end_offset
            mid_cl_1_cr_2_args = (tik_inst, dst_out_gm, mid_cl_gm_offset, dst_ub, mid_cl_ub_offset,
                                  renew_idx_0_dim_size, burst_len, idx_0_dim_asize, burst_len, ele_per_block)
            _move_data_out(mid_cl_1_cr_2_args)

    mid_gm_end_offset = mid_lp_cnt * idx_1_dim_asize + last_gm_end_offset + gm_out_offset
    mid_ub_end_offset = mid_lp_cnt * renew_idx_0_dim_size * burst_len + last_ub_end_offset
    with tik_inst.if_scope(cur_cr_cl_size > 0):
        cur_cl_1_cr_2_args = (tik_inst, dst_out_gm, mid_gm_end_offset, dst_ub, mid_ub_end_offset, cur_cr_cl_size,
                              burst_len, idx_0_dim_asize, burst_len, ele_per_block)
        _move_data_out(cur_cl_1_cr_2_args)


# 'pylint: disable=unused-variable
def _copy_data_out_1st_src_r2nd_dst_not_same(out_offset_args, copy_out_args):
    """
    copy data from ub to gm for source 1st and target r2nd is not same
    """

    (cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out, cr_lp_unit, nlc_cr_lp_cnt, cr_lp_idx,
     cr_lp_step_out, core_step_out, cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
     cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize) = out_offset_args
    (tik_inst, dst_out_gm, dst_ub, mc_pos, block_idx,
     cl_dims, cr_dims, plp_cl_size, plp_cr_size, cr_step_out, c0_size, ele_per_block) = copy_out_args
    offset_base = (cl_lp_idx * cl_lp_step_out + c_lp_idx * c_lp_step_out + cr_lp_idx * cr_lp_step_out + core_step_out)

    with tik_inst.new_stmt_scope(disable_sync=True):
        cr_offset = tik_inst.Scalar(name="cr_offset")
        cr_index = tik_inst.Scalar(name="cr_index")
        last_cr_size = tik_inst.Scalar(name="last_cr_size")
        mid_lp_cnt = tik_inst.Scalar(name="mid_lp_cnt")
        cur_cr_size = tik_inst.Scalar(name="cur_cr_size")
        is_last_cr_nz = tik_inst.Scalar(name="is_last_cr_nz")
        renew_cr_0_size = tik_inst.Scalar(name="renew_cr_0_size")

        # to avoid compile failed
        with tik_inst.if_scope(cr_out_idx_0_size <= tdc.REPEAT_LIMIT_MTE):
            renew_cr_0_size.set_as(cr_out_idx_0_size)
        with tik_inst.else_scope():
            renew_cr_0_size.set_as(1)

        with tik_inst.if_scope(cr_dims == 2):
            cr_idx_args = (tik_inst, block_idx, mc_pos, cr_index, 0, cr_lp_idx, nlc_cr_lp_cnt, cr_lp_unit)
            _count_out_cr_idx(cr_idx_args)
            cr_args = (cr_offset, cr_index, cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
                       cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize)
            _update_out_offset_cr(cr_args)

            split_cr_args = (tik_inst, cr_out_idx_0_size, last_cr_size,
                             mid_lp_cnt, cur_cr_size, cr_index, plp_cr_size)
            _split_dims(split_cr_args)
            two_cr_args = (tik_inst, dst_out_gm, offset_base + cr_offset, dst_ub, 0, last_cr_size,
                           is_last_cr_nz, mid_lp_cnt, cur_cr_size, plp_cl_size * c0_size,
                           cr_out_idx_0_dst_asize, cr_out_idx_0_size, cr_out_idx_1_dst_asize,
                           renew_cr_0_size, ele_per_block)
            _inner_move_data_out_target_two_cl(two_cr_args)
        with tik_inst.else_scope():
            cr_1_cl_1_args = (tik_inst, dst_out_gm, offset_base, dst_ub, 0, plp_cr_size,
                              plp_cl_size * c0_size, cr_step_out, plp_cl_size * c0_size, ele_per_block)
            _move_data_out(cr_1_cl_1_args)


def _split_dims(args):
    """
    split two dimensions into three parts
    """

    tik_inst, cr_out_idx_0_size, last_cr_size, mid_lp_cnt, cur_cr_size, cr_begin, plp_cr_size = args
    next_cr_gap = cr_out_idx_0_size - cr_begin % cr_out_idx_0_size
    with tik_inst.if_scope(next_cr_gap == cr_out_idx_0_size):
        last_cr_size.set_as(0)
    with tik_inst.else_scope():
        with tik_inst.if_scope(next_cr_gap <= plp_cr_size):
            last_cr_size.set_as(next_cr_gap)
        with tik_inst.else_scope():
            last_cr_size.set_as(plp_cr_size)
    mid_lp_cnt.set_as((plp_cr_size - last_cr_size) // cr_out_idx_0_size)
    cur_cr_size.set_as(plp_cr_size - last_cr_size - mid_lp_cnt * cr_out_idx_0_size)


# 'pylint: disable=unused-variable
def _copy_data_out_1st_src_r2nd_dst_same(out_offset_args, copy_out_args):
    """
    copy data from ub to gm for source 1st and target r2nd is same
    """

    (cl_lp_unit, nlc_cl_lp_cnt, cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out,
     cr_lp_unit, nlc_cr_lp_cnt, cr_lp_idx, cr_lp_step_out, core_step_out,
     cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
     cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize,
     cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
     cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize) = out_offset_args
    (tik_inst, dst_out_gm, dst_ub, mc_pos, block_idx, cl_dims, cr_dims, plp_cl_size,
     cl_step_out, plp_cr_size, cr_step_out, c0_size, ele_per_block) = copy_out_args
    offset_base = (cl_lp_idx * cl_lp_step_out + c_lp_idx * c_lp_step_out + cr_lp_idx * cr_lp_step_out + core_step_out)

    with tik_inst.new_stmt_scope(disable_sync=True):
        cl_offset = tik_inst.Scalar(name="cl_offset")
        cr_offset = tik_inst.Scalar(name="cr_offset")
        cr_begin = tik_inst.Scalar(name="cr_begin")
        last_cr_cl_size = tik_inst.Scalar(name="last_cr_cl_size")
        mid_lp_cnt = tik_inst.Scalar(name="mid_lp_cnt")
        cur_cr_cl_size = tik_inst.Scalar(name="cur_cr_cl_size")
        is_last_cr_cl_nz = tik_inst.Scalar(name="is_last_cr_cl_nz")
        renew_cr_0_size = tik_inst.Scalar(name="renew_cr_0_size")
        renew_cl_0_size = tik_inst.Scalar(name="renew_cl_0_size")
        cl_base = tik_inst.Scalar(name="cl_base")

        # to avoid compile failed
        with tik_inst.if_scope(cr_out_idx_0_size <= tdc.REPEAT_LIMIT_MTE):
            renew_cr_0_size.set_as(cr_out_idx_0_size)
        with tik_inst.else_scope():
            renew_cr_0_size.set_as(1)
        with tik_inst.if_scope(cl_out_idx_0_size <= tdc.REPEAT_LIMIT_MTE):
            renew_cl_0_size.set_as(cl_out_idx_0_size)
        with tik_inst.else_scope():
            renew_cl_0_size.set_as(1)

        with tik_inst.if_scope(mc_pos == 0):
            cl_base.set_as((cl_lp_idx + block_idx * nlc_cl_lp_cnt) * cl_lp_unit)
        with tik_inst.else_scope():
            cl_base.set_as(cl_lp_idx * cl_lp_unit)

        def _inner_move_data_out_srcr1st_dstr2nd_same():
            with tik_inst.if_scope(last_cr_cl_size > 0):
                last_cl_1_cr_2_args = (tik_inst, dst_out_gm, offset_base + cr_offset, dst_ub, 0, plp_cl_size,
                                       last_cr_cl_size * c0_size, cl_step_out, plp_cr_size * c0_size, ele_per_block)
                _move_data_out(last_cl_1_cr_2_args)
                is_last_cr_cl_nz.set_as(1)
            with tik_inst.else_scope():
                is_last_cr_cl_nz.set_as(0)

            last_gm_end_offset = is_last_cr_cl_nz * ((last_cr_cl_size - cr_out_idx_0_size)*cr_out_idx_0_dst_asize +
                                                     cr_out_idx_1_dst_asize)
            last_ub_end_offset = last_cr_cl_size * c0_size
            with tik_inst.if_scope(mid_lp_cnt > 0):
                with tik_inst.for_range(0, plp_cl_size) as mid_cl_idx:
                    mid_cl_gm_offset = mid_cl_idx * cl_step_out + offset_base + cr_offset + last_gm_end_offset
                    mid_cl_ub_offset = mid_cl_idx * plp_cr_size * c0_size + last_ub_end_offset
                    mid_cl_1_cr_2_args = (tik_inst, dst_out_gm, mid_cl_gm_offset, dst_ub, mid_cl_ub_offset, mid_lp_cnt,
                                          renew_cr_0_size * c0_size, cr_out_idx_1_dst_asize, renew_cr_0_size * c0_size,
                                          ele_per_block)
                    _move_data_out(mid_cl_1_cr_2_args)

            mid_gm_end_offset = mid_lp_cnt * cr_out_idx_1_dst_asize + last_gm_end_offset
            mid_ub_end_offset = mid_lp_cnt * renew_cr_0_size * c0_size + last_ub_end_offset
            with tik_inst.if_scope(cur_cr_cl_size > 0):
                cur_cl_1_cr_2_args = (tik_inst, dst_out_gm, offset_base + cr_offset + mid_gm_end_offset, dst_ub,
                                      mid_ub_end_offset, plp_cl_size, cur_cr_cl_size * c0_size, cl_step_out,
                                      plp_cr_size * c0_size, ele_per_block)
                _move_data_out(cur_cl_1_cr_2_args)

        with tik_inst.if_scope(tik.all(cl_dims == 1, cr_dims == 1)):
            cl_1_cr_1_args = (tik_inst, dst_out_gm, offset_base, dst_ub, 0, plp_cl_size,
                              plp_cr_size * c0_size, cl_step_out, plp_cr_size * c0_size, ele_per_block)
            _move_data_out(cl_1_cr_1_args)

        with tik_inst.elif_scope(tik.all(cl_dims == 2, cr_dims == 1)):
            cl_args = (cl_offset, cl_base, cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
                       cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize)
            _update_out_offset_cl(cl_args)

            split_cl_args = (tik_inst, cl_out_idx_0_size, last_cr_cl_size,
                             mid_lp_cnt, cur_cr_cl_size, cl_base, plp_cl_size)
            _split_dims(split_cl_args)
            two_cl_args = (tik_inst, dst_out_gm, offset_base + cl_offset, dst_ub, 0, last_cr_cl_size,
                           is_last_cr_cl_nz, mid_lp_cnt, cur_cr_cl_size, plp_cr_size * c0_size,
                           cl_out_idx_0_dst_asize, cl_out_idx_0_size, cl_out_idx_1_dst_asize,
                           renew_cl_0_size, ele_per_block)
            _inner_move_data_out_target_two_cl(two_cl_args)

        with tik_inst.elif_scope(tik.all(cl_dims == 1, cr_dims == 2)):
            cr_idx_args = (tik_inst, block_idx, mc_pos, cr_begin, 0, cr_lp_idx, nlc_cr_lp_cnt, cr_lp_unit)
            _count_out_cr_idx(cr_idx_args)
            cr_args = (cr_offset, cr_begin, cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
                       cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize)
            _update_out_offset_cr(cr_args)

            split_cr_args = (tik_inst, cr_out_idx_0_size, last_cr_cl_size,
                             mid_lp_cnt, cur_cr_cl_size, cr_begin, plp_cr_size)
            _split_dims(split_cr_args)
            _inner_move_data_out_srcr1st_dstr2nd_same()


# 'pylint: disable=too-many-locals
def _func_transform_100(tensor_args, tp_args):
    """
    transform function for tiling mode 100
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, dst_ub, ele_per_block, in_dtype = tensor_args
    (tiling_mode, ub_offset, mc_pos, used_core_cnt, core_step_in, core_step_out, vnc_col_size, c_mod_c0, c0_size,
     cl_dims, cr_dims, r1st_src_r2nd_dst_same, cl_step_in, cl_step_out, cl_lp_unit, cl_lp_step_in, cl_lp_step_out,
     c_step_in, c_lp_unit, c_lp_step_in, c_lp_step_out, cr_step_in, cr_step_out, cr_lp_unit, cr_lp_step_in,
     cr_lp_step_out, nlc_cl_lp_cnt, nlc_cl_left, nlc_c_lp_cnt, nlc_c_left, nlc_cr_lp_cnt, nlc_cr_left, lc_cl_lp_cnt,
     lc_cl_left, lc_c_lp_cnt, lc_c_left, lc_cr_lp_cnt, lc_cr_left,
     cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize, cl_out_idx_1_size,
     cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize, cr_out_idx_0_size, cr_out_idx_0_dst_rsize,
     cr_out_idx_0_dst_asize, cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize) = tp_args

    # 'pylint: disable=too-many-locals
    def _inner_func(args):
        cl_lp_cnt, cl_left, c_lp_cnt, c_left, cr_lp_cnt, cr_left = args
        plp_cl_size = tik_inst.Scalar(name="plp_cl_size")
        plp_c_size = tik_inst.Scalar(name="plp_c_size")
        plp_cr_size = tik_inst.Scalar(name="pln_cr_size")
        nout_lp_cnt = tik_inst.Scalar(name="nout_lp_cnt")

        with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:
            with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                plp_c_size.set_as(c_lp_unit)
            with tik_inst.else_scope():
                plp_c_size.set_as(c_left)

            with tik_inst.for_range(0, cl_lp_cnt) as cl_lp_idx:
                with tik_inst.if_scope(tik.any(cl_lp_idx != cl_lp_cnt - 1, cl_left == 0)):
                    plp_cl_size.set_as(cl_lp_unit)
                with tik_inst.else_scope():
                    plp_cl_size.set_as(cl_left)
                with tik_inst.if_scope(tik.all(tiling_mode == 1001, r1st_src_r2nd_dst_same == 0)):
                    nout_lp_cnt.set_as(plp_cl_size)  # for NCHW -> C1HWNoNic0
                with tik_inst.else_scope():
                    nout_lp_cnt.set_as(1)  # for NCHW -> NC1HWC0

                with tik_inst.for_range(0, cr_lp_cnt) as cr_lp_idx:
                    with tik_inst.if_scope(tik.any(cr_lp_idx != cr_lp_cnt - 1, cr_left == 0)):
                        plp_cr_size.set_as(cr_lp_unit)
                    with tik_inst.else_scope():
                        plp_cr_size.set_as(cr_left)

                    in_offset_args = (cl_lp_idx, cl_lp_step_in, c_lp_idx, c_lp_step_in,
                                      cr_lp_idx, cr_lp_step_in, block_idx * core_step_in)
                    in_gm_offset = _update_input_offset(in_offset_args)
                    with tik_inst.for_range(0, nout_lp_cnt) as nout_lp_idx:
                        in_gm_offset = in_gm_offset + nout_lp_idx * cl_step_in
                        copy_in_args = (tik_inst, src_in_gm, src_ub, mc_pos, in_gm_offset, vnc_col_size,
                                        tiling_mode, plp_cl_size, cl_step_in, plp_c_size, c_step_in,
                                        cr_lp_cnt, plp_cr_size, ele_per_block)
                        _copy_data_in(copy_in_args)
                        vnc_args = (tik_inst, src_ub, mc_pos, dst_ub, nout_lp_idx * c0_size, vnc_col_size,
                                    plp_c_size, r1st_src_r2nd_dst_same, cl_lp_idx, plp_cl_size,
                                    cr_lp_cnt, plp_cr_size, c_mod_c0, c0_size, ele_per_block)
                        with tik_inst.if_scope(tiling_mode == 1000):
                            _twice_vnchwconv_invert(vnc_args)
                        with tik_inst.else_scope():
                            _once_vnchwconv_invert(vnc_args)

                    with tik_inst.if_scope(r1st_src_r2nd_dst_same == 1):
                        out_gm_args = (cl_lp_unit, nlc_cl_lp_cnt, cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out,
                                       cr_lp_unit, nlc_cr_lp_cnt, cr_lp_idx, cr_lp_step_out, block_idx * core_step_out,
                                       cl_out_idx_0_size, cl_out_idx_0_dst_rsize, cl_out_idx_0_dst_asize,
                                       cl_out_idx_1_size, cl_out_idx_1_dst_rsize, cl_out_idx_1_dst_asize,
                                       cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
                                       cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize)
                        copy_out_args = (tik_inst, dst_out_gm, dst_ub, mc_pos, block_idx, cl_dims, cr_dims,
                                         plp_cl_size, cl_step_out, plp_cr_size, cr_step_out, c0_size, ele_per_block)
                        _copy_data_out_1st_src_r2nd_dst_same(out_gm_args, copy_out_args)
                    with tik_inst.else_scope():
                        out_gm_args = (cl_lp_idx, cl_lp_step_out, c_lp_idx, c_lp_step_out, cr_lp_unit,
                                       nlc_cr_lp_cnt, cr_lp_idx, cr_lp_step_out, block_idx * core_step_out,
                                       cr_out_idx_0_size, cr_out_idx_0_dst_rsize, cr_out_idx_0_dst_asize,
                                       cr_out_idx_1_size, cr_out_idx_1_dst_rsize, cr_out_idx_1_dst_asize)
                        copy_out_args = (tik_inst, dst_out_gm, dst_ub, mc_pos, block_idx, cl_dims, cr_dims,
                                         plp_cl_size, plp_cr_size, cr_step_out, c0_size, ele_per_block)
                        _copy_data_out_1st_src_r2nd_dst_not_same(out_gm_args, copy_out_args)

    with tik_inst.if_scope(block_idx != used_core_cnt - 1):
        nlc_args = nlc_cl_lp_cnt, nlc_cl_left, nlc_c_lp_cnt, nlc_c_left, nlc_cr_lp_cnt, nlc_cr_left
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = lc_cl_lp_cnt, lc_cl_left, lc_c_lp_cnt, lc_c_left, lc_cr_lp_cnt, lc_cr_left
        _inner_func(lc_args)


# 'pylint: disable=too-many-locals,unused-variable
def trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name="trans_data_positive_source_ntc"):
    """
    positive transform for last dimension of source format is not c

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        dtype of output should be same type as input
    src_format: str
        source data format, can be ND, NCHW etc.
    dst_format: str
        target data format, can be NC1HWC0, FRACTAL_Z etc.
    kernel_name : str
        kernel name, default value is "trans_data_positive_source_ntc"

    Returns
    -------
    None
    """

    src_format = src_format.upper()
    dst_format = dst_format.upper()
    in_dtype = src.get("dtype").lower() if src.get("dtype").lower() != "bfloat16" else "float16"
    in_dtype_bytes = tdc.get_dtype_len(in_dtype)
    tiling_dtype_bytes = tdc.get_dtype_len("int64")
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1, 256) - Constant.TILING_CTRL_PARAM[1] \
              * tiling_dtype_bytes // in_dtype_bytes
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)

    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "src_in_gm")
    if src_format.upper() == "NCHW" and dst_format.upper() == "NC1HWC0":
        dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm", is_atomic_add=False)
    else:
        dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm", is_atomic_add=True)
    tiling_gm = tik_inst.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],),
                                tik.scope_gm, "tiling_gm")
    half_ub = ub_size // 2
    src_ub = tik_inst.Tensor(in_dtype, (half_ub,), tik.scope_ubuf, "src_ub")
    dst_ub = tik_inst.Tensor(in_dtype, (half_ub,), tik.scope_ubuf, "dst_ub")
    tiling_ub = tik_inst.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],),
                                tik.scope_ubuf, "tiling_ub")
    tiling_params = [tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0]) for i in range(Constant.TILING_CTRL_PARAM[1])]
    _get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes)

    used_core_cnt = tiling_params[3]
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, dst_ub, block_elem_cnt, in_dtype]
            tp_args = tiling_params[0:50]
            _func_transform_100(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[src_in_gm], outputs=[dst_out_gm], flowtable=[tiling_gm], enable_l2=False,
                      config={"dynamic_tik": True, "out_of_bound_sync_check": True})
    tbe_context.get_context().add_compile_info("vars", {"ub_size": ub_size,
                                                        "block_dim": tdc.CORE_DIM_NUM,
                                                        "group": 1})
