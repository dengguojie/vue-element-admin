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
from impl.util.platform_adapter import tbe_context
from .. import trans_data_common_func as tdc


# frame up levels
FRAME_LEVEL = 2
TILING_CTRL_PARAM = ("int64", 96)
INT8_DTYPES = ("int8", "uint8")
NEED_CAST_DTYPES = ("float32", "int32", "uint32")
VNC_SUPPORT_DTYPES = ("int8", "uint8", "float16")


def _get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes):
    """
    get tiling parameters function
    """

    ele_per_block = tdc.BLOCK_BYTE_SIZE // tiling_dtype_bytes
    tik_inst.data_move(tiling_ub, tiling_gm, 0, 1, TILING_CTRL_PARAM[1] // ele_per_block, 0, 0)
    for reg_idx in range(TILING_CTRL_PARAM[1]):
        tiling_params[reg_idx].set_as(tiling_ub[reg_idx])


# pylint: disable=too-many-locals, too-many-statements
def _chtn_2_hctn_transfer(trans_args):
    """
    do chtn to hctn transpose for data in row and col line
    """

    (tik_inst, dst_ub, src_ub, tiling_mode, r2nd_pl_size, c1_pl_size, left_pl_size,
     sub_c_size, all_c_in, c0_len, dtype_factor, ele_per_block) = trans_args

    def _chtn_2_hctn_process_r2nd(left_src_offset, left_dst_offset):
        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.if_scope(all_c_in == 1):  # all c is moved in
                with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx:
                    r2nd_src_ub_offset = (r2nd_idx + left_src_offset) * c0_len * dtype_factor * ele_per_block
                    r2nd_dst_ub_offset = (r2nd_idx + left_dst_offset) * sub_c_size * dtype_factor * ele_per_block
                    tik_inst.data_move(src_ub[r2nd_dst_ub_offset], dst_ub[r2nd_src_ub_offset], 0, c1_pl_size,
                                       c0_len * dtype_factor, (r2nd_pl_size - 1) * c0_len * dtype_factor, 0)
            with tik_inst.else_scope():
                with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx:
                    r2nd_src_ub_offset = (r2nd_idx + left_src_offset) * c0_len * dtype_factor * ele_per_block
                    r2nd_dst_ub_offset = (r2nd_idx +
                                          left_dst_offset) * c1_pl_size * c0_len * dtype_factor * ele_per_block
                    tik_inst.data_move(src_ub[r2nd_dst_ub_offset], dst_ub[r2nd_src_ub_offset], 0, c1_pl_size,
                                       c0_len * dtype_factor, (r2nd_pl_size - 1) * c0_len * dtype_factor, 0)

    def _chtn_2_hctn_process_c1(left_src_offset, left_dst_offset):
        with tik_inst.if_scope(all_c_in == 1):  # all c is moved in
            with tik_inst.new_stmt_scope(disable_sync=True):
                c_mod = sub_c_size % c0_len
                data_unit = dtype_factor * ele_per_block
                with tik_inst.if_scope(c_mod > 0):
                    with tik_inst.for_range(0, c1_pl_size-1) as c1_idx:
                        c1_src_ub_offset = (c1_idx + left_src_offset) * r2nd_pl_size * c0_len * data_unit
                        c1_dst_ub_offset = (c1_idx*c0_len + left_dst_offset*sub_c_size) * data_unit
                        tik_inst.data_move(src_ub[c1_dst_ub_offset], dst_ub[c1_src_ub_offset], 0, r2nd_pl_size,
                                           c0_len * dtype_factor, 0, (sub_c_size - c0_len) * dtype_factor)
                    c1_src_ub_offset = (c1_pl_size-1 + left_src_offset) * r2nd_pl_size * c0_len * data_unit
                    c1_dst_ub_offset = ((c1_pl_size-1)*c0_len + left_dst_offset*sub_c_size) * data_unit
                    tik_inst.data_move(src_ub[c1_dst_ub_offset], dst_ub[c1_src_ub_offset],
                                       0, r2nd_pl_size, c_mod * dtype_factor, (c0_len - c_mod) * dtype_factor,
                                       (sub_c_size - c_mod) * dtype_factor)
                with tik_inst.else_scope():
                    with tik_inst.for_range(0, c1_pl_size) as c1_idx:
                        c1_src_ub_offset = (c1_idx + left_src_offset) * r2nd_pl_size * c0_len * data_unit
                        c1_dst_ub_offset = (c1_idx*c0_len + left_dst_offset*sub_c_size) * data_unit
                        tik_inst.data_move(src_ub[c1_dst_ub_offset], dst_ub[c1_src_ub_offset], 0, r2nd_pl_size,
                                           c0_len * dtype_factor, 0, (sub_c_size - c0_len) * dtype_factor)
        with tik_inst.else_scope():
            with tik_inst.new_stmt_scope(disable_sync=True):
                with tik_inst.for_range(0, c1_pl_size) as c1_idx:
                    c1_src_ub_offset = (c1_idx + left_src_offset) * r2nd_pl_size * c0_len * data_unit
                    c1_dst_ub_offset = (c1_idx + left_dst_offset*c1_pl_size) * c0_len * data_unit
                    tik_inst.data_move(src_ub[c1_dst_ub_offset], dst_ub[c1_src_ub_offset], 0, r2nd_pl_size,
                                       c0_len * dtype_factor, 0, (c1_pl_size - 1) * c0_len * dtype_factor)

    with tik_inst.if_scope(tiling_mode == 2011):
        with tik_inst.if_scope(r2nd_pl_size <= c1_pl_size):
            _chtn_2_hctn_process_r2nd(0, 0)
        with tik_inst.else_scope():
            _chtn_2_hctn_process_c1(0, 0)
    with tik_inst.else_scope():
        with tik_inst.if_scope(r2nd_pl_size <= c1_pl_size):
            with tik_inst.for_range(0, left_pl_size) as left_idx:
                _chtn_2_hctn_process_r2nd(left_idx * r2nd_pl_size * c1_pl_size, left_idx * r2nd_pl_size)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, left_pl_size) as left_idx:
                _chtn_2_hctn_process_c1(left_idx * c1_pl_size, left_idx * r2nd_pl_size)


def _twice_vnchwconv_no_invert(args):
    """
    do ncht to nhct transform by twice vnchwconv
    """

    (tik_inst, src_ub, dst_ub, left_pl_size, c1_pl_size, r2nd_pl_size, c0_len, ele_per_block, in_dtype, tiling_mode,
     all_c_in, sub_c_size, vnc_col_len, r2nd_lp_idx) = args
    dtype_factor = tdc.get_dtype_factor(in_dtype)

    with tik_inst.new_stmt_scope():
        vnc_col_data = tik_inst.Scalar(name="vnc_col_data")
        src_stride = tik_inst.Scalar()
        dst_stride = tik_inst.Scalar()
        tmp_reg = [tik_inst.Scalar(dtype=dst_ub.dtype) for i in tdc.REG_IDX_LIST[:ele_per_block]]
        with tik_inst.if_scope(tiling_mode == 2011):
            vnc_col_data.set_as(c1_pl_size * r2nd_pl_size * c0_len)
        with tik_inst.else_scope():
            vnc_col_data.set_as(left_pl_size * r2nd_pl_size * c1_pl_size * c0_len)

        if in_dtype not in INT8_DTYPES:
            src_ub_casted = src_ub.reinterpret_cast_to("float16")
            dst_ub_casted = dst_ub.reinterpret_cast_to("float16")
            # do ncht -> chtn
            src_addr_list = [src_ub_casted[vnc_col_len * dtype_factor * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub_casted[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
            repeat_cnt = vnc_col_data * dtype_factor // tdc.C0_16
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
                dst_stride.set_as(16)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        else:
            repeat_cnt = vnc_col_data // tdc.C0_32
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(1)
                dst_stride.set_as(32)
            src_addr_list = [src_ub[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[tdc.C0_32 * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            dst_addr_list = [dst_ub[tdc.C0_32 * (i + tdc.NI_16)] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, True, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        # do chtn -> hctn
        trans_args = (tik_inst, dst_ub, src_ub, tiling_mode, r2nd_pl_size, c1_pl_size, left_pl_size,
                      sub_c_size, all_c_in, c0_len, dtype_factor, ele_per_block)
        _chtn_2_hctn_transfer(trans_args)

        # do hctn -> nhct
        if in_dtype not in INT8_DTYPES:
            src_ub_casted = src_ub.reinterpret_cast_to("float16")
            dst_ub_casted = dst_ub.reinterpret_cast_to("float16")
            # do ncht -> chtn
            src_addr_list = [src_ub_casted[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub_casted[vnc_col_len * dtype_factor * i]
                             for i in tdc.ADDR_IDX_LIST]
            repeat_cnt = vnc_col_data * dtype_factor // tdc.C0_16
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(16)
                dst_stride.set_as(1)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
        else:
            repeat_cnt = vnc_col_data // tdc.C0_32
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(32)
                dst_stride.set_as(1)
            src_addr_list = [src_ub[tdc.C0_32 * i] for i in tdc.ADDR_IDX_LIST]
            dst_addr_list = [dst_ub[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)
            src_addr_list = [src_ub[tdc.C0_32 * (i + tdc.NI_16)] for i in tdc.ADDR_IDX_LIST]
            tik_inst.vnchwconv(True, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        # adjust tail block for output
        left_r2nd_c_size = left_pl_size * r2nd_pl_size * sub_c_size
        r2nd_c_size = r2nd_pl_size * sub_c_size
        with tik_inst.if_scope(tik.all(tiling_mode == 2012, left_r2nd_c_size > ele_per_block,
                                       left_r2nd_c_size % ele_per_block > 0)):
            left_r2nd_c_block_align = left_r2nd_c_size // ele_per_block * ele_per_block
            for i in tdc.REG_IDX_LIST[:ele_per_block]:
                tmp_reg[i].set_as(dst_ub[left_r2nd_c_size - ele_per_block + i])
            for i in tdc.REG_IDX_LIST[:ele_per_block]:
                dst_ub[left_r2nd_c_block_align + i:].set_as(tmp_reg[i])
        with tik_inst.if_scope(tik.all(tiling_mode == 2011, r2nd_c_size % ele_per_block > 0)):
            left_lp_cnt = tik_inst.Scalar(name="left_lp_cnt")
            with tik_inst.if_scope(r2nd_lp_idx > 0):
                left_lp_cnt.set_as(left_pl_size * r2nd_lp_idx)
            with tik_inst.else_scope():
                left_lp_cnt.set_as(left_pl_size)
            r2nd_c_block_align = r2nd_c_size // ele_per_block * ele_per_block
            with tik_inst.for_range(0, left_lp_cnt) as left_idx:
                left_ub_offset = left_idx * vnc_col_len
                for i in tdc.REG_IDX_LIST[:ele_per_block]:
                    tmp_reg[i].set_as(dst_ub[left_ub_offset + r2nd_c_size - ele_per_block + i])
                for i in tdc.REG_IDX_LIST[:ele_per_block]:
                    dst_ub[left_ub_offset + r2nd_c_block_align + i:].set_as(tmp_reg[i])


def _vor_data_move(vor_args):
    """
    reorder data by vor
    """
    (tik_inst, dst_ub_int16, src_ub_int16, zero_ub, axis_lp_cnt, src_lp_step, dst_lp_step, repeat_cnt, repeat_mod,
     repeat_full_cnt, dst_block_stride, src0_block_stride, left_src_ub_offset, left_dst_ub_offset, mod_dst_ub_offset,
     mod_src_ub_offset, rp_full_dst_offset, rp_full_src_offset, per_lp_block_cnt, dtype_factor) = vor_args
    src1_block_stride = 0
    dst_rep_stride = per_lp_block_cnt * dst_block_stride
    src0_rep_stride = per_lp_block_cnt * src0_block_stride
    src1_rep_stride = 0
    repeat_mask = per_lp_block_cnt * tdc.C0_16

    with tik_inst.if_scope(repeat_full_cnt > 0):
        with tik_inst.for_range(0, dtype_factor) as factor_idx:
            factor_offset = factor_idx * tdc.C0_16
            with tik_inst.for_range(0, axis_lp_cnt) as lp_idx:
                lp_src_ub_offset = lp_idx * src_lp_step
                lp_dst_ub_offset = lp_idx * dst_lp_step
                with tik_inst.for_range(0, repeat_full_cnt) as rp_idx:
                    src_ub_offset = lp_src_ub_offset + factor_offset + left_src_ub_offset + rp_idx*rp_full_src_offset
                    dst_ub_offset = lp_dst_ub_offset + factor_offset + left_dst_ub_offset + rp_idx*rp_full_dst_offset
                    tik_inst.vor(repeat_mask, dst_ub_int16[dst_ub_offset], src_ub_int16[src_ub_offset],
                                 zero_ub, tdc.REPEAT_LIMIT_VECT, dst_block_stride, src0_block_stride, src1_block_stride,
                                 dst_rep_stride, src0_rep_stride, src1_rep_stride)

    with tik_inst.if_scope(repeat_cnt > 0):
        with tik_inst.for_range(0, dtype_factor) as factor_idx:
            factor_offset = factor_idx * tdc.C0_16
            with tik_inst.for_range(0, axis_lp_cnt) as lp_idx:
                lp_src_ub_offset = lp_idx * src_lp_step + repeat_full_cnt*rp_full_src_offset
                lp_dst_ub_offset = lp_idx * dst_lp_step + repeat_full_cnt*rp_full_dst_offset
                src_ub_offset = lp_src_ub_offset + factor_offset + left_src_ub_offset
                dst_ub_offset = lp_dst_ub_offset + factor_offset + left_dst_ub_offset
                tik_inst.vor(repeat_mask, dst_ub_int16[dst_ub_offset], src_ub_int16[src_ub_offset],
                             zero_ub, repeat_cnt, dst_block_stride, src0_block_stride, src1_block_stride,
                             dst_rep_stride, src0_rep_stride, src1_rep_stride)

    with tik_inst.if_scope(repeat_mod > 0):
        mod_mask = repeat_mod * tdc.C0_16
        with tik_inst.for_range(0, dtype_factor) as factor_idx:
            factor_offset = factor_idx * tdc.C0_16
            with tik_inst.for_range(0, axis_lp_cnt) as lp_idx:
                lp_src_ub_offset = lp_idx * src_lp_step
                lp_dst_ub_offset = lp_idx * dst_lp_step
                src_ub_offset = lp_src_ub_offset + factor_offset + left_src_ub_offset
                dst_ub_offset = lp_dst_ub_offset + factor_offset + left_dst_ub_offset
                dst_ub_offset_mod = dst_ub_offset + mod_dst_ub_offset
                src_ub_offset_mod = src_ub_offset + mod_src_ub_offset
                tik_inst.vor(mod_mask, dst_ub_int16[dst_ub_offset_mod], src_ub_int16[src_ub_offset_mod],
                             zero_ub, 1, dst_block_stride, src0_block_stride, src1_block_stride,
                             dst_rep_stride, src0_rep_stride, src1_rep_stride)


def _update_vor_block_cnt(tik_inst, per_lp_block_cnt, blk_stride):
    """
    to get block count for per loop
    """
    max_vec_blk_cnt = 8
    with tik_inst.if_scope(tdc.REPEAT_LIMIT_VECT // blk_stride >= max_vec_blk_cnt):
        per_lp_block_cnt.set_as(max_vec_blk_cnt)
    with tik_inst.else_scope():
        per_lp_block_cnt.set_as(tdc.REPEAT_LIMIT_VECT // blk_stride)


def _ubuf_2_ubuf_convert(args):
    """
    do ncht to nhct transform by data_move
    """

    (tik_inst, src_ub, dst_ub, zero_ub, left_pl_size, c1_pl_size, r2nd_pl_size,
     c0_len, ele_per_block, sub_c_size, all_c_in, in_dtype) = args
    dtype_factor = tdc.get_dtype_factor(in_dtype)
    src_ub_int16 = src_ub.reinterpret_cast_to("int16")
    dst_ub_int16 = dst_ub.reinterpret_cast_to("int16")

    with tik_inst.new_stmt_scope(disable_sync=True):
        per_lp_block_cnt = tik_inst.Scalar(name="per_lp_block_cnt")
        left_lp_step = tik_inst.Scalar(name="left_lp_step")
        left_lp_step.set_as(r2nd_pl_size * c1_pl_size * tdc.C0_16 * dtype_factor)

        # to avoid bank conflict
        data_unit = tdc.C0_16 * dtype_factor
        with tik_inst.if_scope(tik.all(c1_pl_size % tdc.C0_16 > 0, c1_pl_size <= r2nd_pl_size)):
            repeat_cnt_r2nd = r2nd_pl_size // per_lp_block_cnt
            repeat_full_cnt_r2nd = repeat_cnt_r2nd // tdc.REPEAT_LIMIT_VECT
            repeat_full_left_r2nd = repeat_cnt_r2nd % tdc.REPEAT_LIMIT_VECT
            mod_r2nd = r2nd_pl_size % per_lp_block_cnt
            loop_src_ub_offset = r2nd_pl_size * data_unit
            loop_dst_ub_offset = data_unit
            dst_blk_stride = c1_pl_size * dtype_factor
            src0_blk_stride = 1 * dtype_factor
            mod_dst_offset = repeat_cnt_r2nd * per_lp_block_cnt * c1_pl_size * data_unit
            mod_src_offset = repeat_cnt_r2nd * per_lp_block_cnt * data_unit
            repeat_full_dst_offset = tdc.REPEAT_LIMIT_VECT * per_lp_block_cnt * c1_pl_size * data_unit
            repeat_full_src_offset = tdc.REPEAT_LIMIT_VECT * per_lp_block_cnt * data_unit
            _update_vor_block_cnt(tik_inst, per_lp_block_cnt, dst_blk_stride)
            with tik_inst.for_range(0, left_pl_size) as left_idx:
                left_src_ub_offset = left_idx * left_lp_step
                left_dst_ub_offset = left_src_ub_offset
                c1_args = (tik_inst, dst_ub_int16, src_ub_int16, zero_ub, c1_pl_size, loop_src_ub_offset,
                           loop_dst_ub_offset, repeat_full_left_r2nd, mod_r2nd, repeat_full_cnt_r2nd,
                           dst_blk_stride, src0_blk_stride, left_src_ub_offset, left_dst_ub_offset,
                           mod_dst_offset, mod_src_offset, repeat_full_dst_offset, repeat_full_src_offset,
                           per_lp_block_cnt, dtype_factor)
                _vor_data_move(c1_args)
        with tik_inst.else_scope():
            repeat_cnt_c1 = c1_pl_size // per_lp_block_cnt
            repeat_full_cnt_c1 = repeat_cnt_c1 // tdc.REPEAT_LIMIT_VECT
            repeat_full_left_c1 = repeat_cnt_c1 % tdc.REPEAT_LIMIT_VECT
            mod_c1 = c1_pl_size % per_lp_block_cnt
            loop_src_ub_offset = data_unit
            loop_dst_ub_offset = c1_pl_size * data_unit
            dst_blk_stride = 1 * dtype_factor
            src0_blk_stride = r2nd_pl_size * dtype_factor
            mod_dst_offset = repeat_cnt_c1 * per_lp_block_cnt * data_unit
            mod_src_offset = repeat_cnt_c1 * per_lp_block_cnt * r2nd_pl_size * data_unit
            repeat_full_dst_offset = tdc.REPEAT_LIMIT_VECT * per_lp_block_cnt * data_unit
            repeat_full_src_offset = tdc.REPEAT_LIMIT_VECT * per_lp_block_cnt * r2nd_pl_size * data_unit
            _update_vor_block_cnt(tik_inst, per_lp_block_cnt, src0_blk_stride)
            with tik_inst.for_range(0, left_pl_size) as left_idx:
                left_src_ub_offset = left_idx * left_lp_step
                left_dst_ub_offset = left_src_ub_offset
                r2nd_args = (tik_inst, dst_ub_int16, src_ub_int16, zero_ub, r2nd_pl_size, loop_src_ub_offset,
                             loop_dst_ub_offset, repeat_full_left_c1, mod_c1, repeat_full_cnt_c1,
                             dst_blk_stride, src0_blk_stride, left_src_ub_offset, left_dst_ub_offset,
                             mod_dst_offset, mod_src_offset, repeat_full_dst_offset, repeat_full_src_offset,
                             per_lp_block_cnt, dtype_factor)
                _vor_data_move(r2nd_args)

    # adjust tail block
    with tik_inst.if_scope(sub_c_size % ele_per_block > 0):
        tmp_reg = [tik_inst.Scalar(dtype=dst_ub.dtype) for i in tdc.REG_IDX_LIST[:ele_per_block]]
        sub_c_block_align = sub_c_size // ele_per_block * ele_per_block

        with tik_inst.if_scope(all_c_in > 0):
            r2nd_c_size = (r2nd_pl_size - 1) * c1_pl_size * c0_len
            with tik_inst.for_range(0, left_pl_size) as left_idx:
                left_dst_ub_offset = left_idx * r2nd_pl_size * c1_pl_size * c0_len
                for i in tdc.REG_IDX_LIST[:ele_per_block]:
                    tmp_reg[i].set_as(dst_ub[r2nd_c_size + sub_c_size - ele_per_block + i + left_dst_ub_offset])
                for i in tdc.REG_IDX_LIST[:ele_per_block]:
                    dst_ub[left_dst_ub_offset + r2nd_c_size + sub_c_block_align + i:].set_as(tmp_reg[i])
        with tik_inst.else_scope():
            with tik_inst.for_range(0, left_pl_size*r2nd_pl_size) as left_r2nd_idx:
                left_r2nd_dst_ub_offset = left_r2nd_idx * c1_pl_size * c0_len
                for i in tdc.REG_IDX_LIST[:ele_per_block]:
                    tmp_reg[i].set_as(dst_ub[sub_c_size - ele_per_block + i + left_r2nd_dst_ub_offset])
                for i in tdc.REG_IDX_LIST[:ele_per_block]:
                    dst_ub[left_r2nd_dst_ub_offset + sub_c_block_align + i:].set_as(tmp_reg[i])


def _update_input_offset_0(args):
    """
    count input gm offset for such as NC1HWC0 -> NHWC
    """

    (left_lp_idx, src_left_lp_step_in, left_backend, src_left_step_in, c1_lp_idx, src_c1_lp_step_in, c1_backend,
     src_c1_step_in, r2nd_lp_idx, dst_r2nd_lp_step_in, r2nd_backend, dst_r2nd_step_in, core_step_in) = args

    in_offset = (r2nd_lp_idx * dst_r2nd_lp_step_in - r2nd_backend * dst_r2nd_step_in +
                 c1_lp_idx * src_c1_lp_step_in - c1_backend * src_c1_step_in +
                 left_lp_idx * src_left_lp_step_in - left_backend * src_left_step_in + core_step_in)

    return in_offset


def _update_input_offset_1(args):
    """
    count input gm offset for such as DC1HWNoNiC0 -> NDHWC
    """

    (left_lp_idx, src_left_lp_step_in, left_backend, src_left_step_in, c1_lp_idx,
     src_c1_lp_step_in, c1_backend, src_c1_step_in, r2nd_beg, core_step_in,
     dst_r2nd_in_0_size, dst_r2nd_in_0_src_rsize, dst_r2nd_in_0_src_asize,
     dst_r2nd_in_1_size, dst_r2nd_in_1_src_rsize, dst_r2nd_in_1_src_asize) = args

    in_offset = (left_lp_idx * src_left_lp_step_in - left_backend * src_left_step_in +
                 c1_lp_idx * src_c1_lp_step_in - c1_backend * src_c1_step_in + core_step_in +
                 r2nd_beg // dst_r2nd_in_0_src_rsize % dst_r2nd_in_0_size * dst_r2nd_in_0_src_asize +
                 r2nd_beg // dst_r2nd_in_1_src_rsize % dst_r2nd_in_1_size * dst_r2nd_in_1_src_asize)

    return in_offset


def _update_output_offset(args):
    """
    count output gm offset
    """

    (r2nd_lp_idx, dst_r2nd_lp_step_out, r2nd_backend, dst_r2nd_step_out,
     c1_lp_idx, src_c1_lp_step_out, c1_backend, left_lp_idx,
     src_left_lp_step_out, left_backend, src_left_step_out, core_step_out) = args

    out_offset = (left_lp_idx * src_left_lp_step_out - left_backend * src_left_step_out +
                  c1_lp_idx * src_c1_lp_step_out - c1_backend +
                  r2nd_lp_idx * dst_r2nd_lp_step_out - r2nd_backend * dst_r2nd_step_out + core_step_out)

    return out_offset


def _copy_data_in_0(args):
    """
    copy data from gm to ub for such as NC1HWC0 -> NHWC
    """

    (tik_inst, src_in_gm, src_ub, in_gm_offset, left_pl_size, src_left_step_in,
     c1_pl_size, src_c1_step_in, r2nd_pl_size, c0_len, ele_per_block, vnc_col_len) = args

    with tik_inst.new_stmt_scope(disable_sync=True):
        c1_r2nd_gap = (src_c1_step_in - r2nd_pl_size * c0_len) // ele_per_block
        with tik_inst.if_scope(c1_r2nd_gap <= tdc.STRIDE_LIMIT_MTE):
            with tik_inst.for_range(0, left_pl_size) as left_idx:
                left_ub_offset = left_idx * vnc_col_len
                left_gm_offset = left_idx * src_left_step_in
                tik_inst.data_move(src_ub[left_ub_offset], src_in_gm[left_gm_offset + in_gm_offset],
                                   0, c1_pl_size, r2nd_pl_size * c0_len // ele_per_block, c1_r2nd_gap, 0)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, left_pl_size) as left_idx:
                left_ub_offset = left_idx * vnc_col_len
                left_gm_offset = left_idx * src_left_step_in
                with tik_inst.for_range(0, c1_pl_size) as c1_idx:
                    c1_ub_offset = c1_idx * r2nd_pl_size * c0_len
                    c1_gm_offset = c1_idx * src_c1_step_in
                    tik_inst.data_move(src_ub[c1_ub_offset + left_ub_offset],
                                       src_in_gm[c1_gm_offset + left_gm_offset + in_gm_offset],
                                       0, 1, r2nd_pl_size * c0_len // ele_per_block, 0, 0)


def _split_r2nd(args):
    """
    split reverse second dimensions into three parts when it has two dimensions
    """

    tik_inst, dst_r2nd_in_0_size, left_r2nd_size, mid_lp_cnt, right_r2nd_size, r2nd_beg, r2nd_pl_size = args
    next_r2nd_gap = dst_r2nd_in_0_size - r2nd_beg % dst_r2nd_in_0_size
    with tik_inst.if_scope(next_r2nd_gap == dst_r2nd_in_0_size):
        left_r2nd_size.set_as(0)
    with tik_inst.else_scope():
        with tik_inst.if_scope(next_r2nd_gap <= r2nd_pl_size):
            left_r2nd_size.set_as(next_r2nd_gap)
        with tik_inst.else_scope():
            left_r2nd_size.set_as(r2nd_pl_size)
    mid_lp_cnt.set_as((r2nd_pl_size - left_r2nd_size) // dst_r2nd_in_0_size)
    right_r2nd_size.set_as(r2nd_pl_size - left_r2nd_size - mid_lp_cnt * dst_r2nd_in_0_size)


def _copy_data_in_1(args):
    """
    copy data from gm to ub for such as DC1HWNoNiC0 -> NDHWC
    """

    (tik_inst, src_in_gm, src_ub, ub_offset_2011, in_gm_offset, dst_ub, zero_ub, left_pl_size, c1_pl_size,
     src_c1_step_in, r2nd_beg, r2nd_pl_size, dst_r2nd_in_0_size, dst_r2nd_in_0_src_asize,
     dst_r2nd_in_1_src_asize, c0_len, ele_per_block, vnc_col_len, in_dtype) = args

    with tik_inst.new_stmt_scope():
        left_r2nd_size = tik_inst.Scalar(name="left_r2nd_size")
        mid_lp_cnt = tik_inst.Scalar(name="mid_lp_cnt")
        right_r2nd_size = tik_inst.Scalar(name="right_r2nd_size")
        is_left_r2nd_nz = tik_inst.Scalar(name="is_left_r2nd_nz")
        renew_r2nd_size = tik_inst.Scalar(name="renew_r2nd_size")
        split_args = (tik_inst, dst_r2nd_in_0_size, left_r2nd_size, mid_lp_cnt, right_r2nd_size, r2nd_beg, r2nd_pl_size)
        _split_r2nd(split_args)
        r2nd_left_gap = (dst_r2nd_in_0_src_asize - left_pl_size * c0_len) // ele_per_block
        # to avoid compile failed
        with tik_inst.if_scope(dst_r2nd_in_0_size <= tdc.REPEAT_LIMIT_MTE):
            renew_r2nd_size.set_as(dst_r2nd_in_0_size)
        with tik_inst.else_scope():
            renew_r2nd_size.set_as(1)

        def _inner_copy_by_loop(ub_args, gm_args, r2nd_size):
            with tik_inst.for_range(0, r2nd_size) as r2nd_idx:
                r2nd_ub_offset = r2nd_idx * left_pl_size * c0_len
                r2nd_gm_offset = r2nd_idx * dst_r2nd_in_0_src_asize
                tik_inst.data_move(src_ub[r2nd_ub_offset + ub_args + ub_offset_2011],
                                   src_in_gm[r2nd_gm_offset + gm_args + in_gm_offset],
                                   0, 1, left_pl_size * c0_len // ele_per_block, 0, 0)

        def _inner_copy_by_repeat(ub_args, gm_args, r2nd_size):
            tik_inst.data_move(src_ub[ub_args + ub_offset_2011], src_in_gm[gm_args + in_gm_offset],
                               0, r2nd_size, left_pl_size * c0_len // ele_per_block, r2nd_left_gap, 0)

        def _copy_data_in_three_parts(inner_copy_func):
            with tik_inst.if_scope(left_r2nd_size > 0):
                is_left_r2nd_nz.set_as(1)
                with tik_inst.for_range(0, c1_pl_size) as c1_idx:
                    c1_ub_offset = c1_idx * r2nd_pl_size * left_pl_size * c0_len
                    c1_gm_offset = c1_idx * src_c1_step_in
                    ub_arg = c1_ub_offset
                    gm_arg = c1_gm_offset
                    inner_copy_func(ub_arg, gm_arg, left_r2nd_size)
            with tik_inst.else_scope():
                is_left_r2nd_nz.set_as(0)
            left_gm_offset = is_left_r2nd_nz * ((left_r2nd_size - dst_r2nd_in_0_size) * dst_r2nd_in_0_src_asize +
                                                dst_r2nd_in_1_src_asize)
            left_ub_offset = left_r2nd_size * left_pl_size * c0_len

            with tik_inst.if_scope(mid_lp_cnt > 0):
                with tik_inst.for_range(0, c1_pl_size) as c1_idx:
                    c1_ub_offset = c1_idx * r2nd_pl_size * left_pl_size * c0_len
                    c1_gm_offset = c1_idx * src_c1_step_in
                    with tik_inst.for_range(0, mid_lp_cnt) as mid_lp_idx:
                        r2nd1_ub_offset = mid_lp_idx * dst_r2nd_in_0_size * left_pl_size * c0_len
                        r2nd1_gm_offset = mid_lp_idx * dst_r2nd_in_1_src_asize
                        ub_arg = r2nd1_ub_offset + left_ub_offset + c1_ub_offset
                        gm_arg = r2nd1_gm_offset + left_gm_offset + c1_gm_offset
                        inner_copy_func(ub_arg, gm_arg, renew_r2nd_size)

            mid_ub_offset = mid_lp_cnt * dst_r2nd_in_0_size * left_pl_size * c0_len
            mid_gm_offset = mid_lp_cnt * dst_r2nd_in_1_src_asize
            with tik_inst.if_scope(right_r2nd_size > 0):
                with tik_inst.for_range(0, c1_pl_size) as c1_idx:
                    c1_ub_offset = c1_idx * r2nd_pl_size * left_pl_size * c0_len
                    c1_gm_offset = c1_idx * src_c1_step_in
                    ub_arg = mid_ub_offset + left_ub_offset + c1_ub_offset
                    gm_arg = mid_gm_offset + left_gm_offset + c1_gm_offset
                    inner_copy_func(ub_arg, gm_arg, right_r2nd_size)

        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.if_scope(r2nd_left_gap <= tdc.STRIDE_LIMIT_MTE):
                _copy_data_in_three_parts(_inner_copy_by_repeat)
            with tik_inst.else_scope():
                _copy_data_in_three_parts(_inner_copy_by_loop)

        # in order to move data out from same ub address
        data_size = c1_pl_size * r2nd_pl_size * left_pl_size * c0_len
        tik_inst.data_move(dst_ub, src_ub[ub_offset_2011], 0, 1, data_size // ele_per_block, 0, 0)

        # do c1hnt -> nc1ht
        with tik_inst.new_stmt_scope(disable_sync=True):
            dtype_factor = tdc.get_dtype_factor(in_dtype)
            if in_dtype in NEED_CAST_DTYPES:
                src_ub_int16 = src_ub.reinterpret_cast_to("int16")[ub_offset_2011 * 2]
            elif in_dtype in ("int8", "uint8"):
                src_ub_int16 = src_ub.reinterpret_cast_to("int16")[ub_offset_2011 // 2]
            else:
                src_ub_int16 = src_ub.reinterpret_cast_to("int16")[ub_offset_2011]
            dst_ub_int16 = dst_ub.reinterpret_cast_to("int16")
            vor_lp_block_cnt = tik_inst.Scalar(name="vor_lp_block_cnt")
            r2nd_c1_size = r2nd_pl_size * c1_pl_size
            data_unit = tdc.C0_16 * dtype_factor
            vnc_col_blk_cnt = vnc_col_len // ele_per_block
            with tik_inst.if_scope(left_pl_size <= r2nd_c1_size):
                src0_blk_stride = left_pl_size * dtype_factor
                _update_vor_block_cnt(tik_inst, vor_lp_block_cnt, src0_blk_stride)
                # src1 rep stride is too large or bank conflict
                with tik_inst.if_scope(tik.any(vor_lp_block_cnt <= 1,
                                               src0_blk_stride % tdc.C0_16 == 0)):
                    with tik_inst.for_range(0, left_pl_size) as left_idx:
                        tik_inst.data_move(src_ub[left_idx * vnc_col_len + ub_offset_2011], dst_ub[left_idx * c0_len],
                                           0, r2nd_c1_size, c0_len // ele_per_block,
                                           (left_pl_size - 1) * c0_len // ele_per_block, 0)
                with tik_inst.else_scope():
                    repeat_cnt_r2nd_c1 = r2nd_c1_size // vor_lp_block_cnt
                    repeat_full_cnt_r2nd_c1 = repeat_cnt_r2nd_c1 // tdc.REPEAT_LIMIT_VECT
                    repeat_full_left_r2nd_c1 = repeat_cnt_r2nd_c1 % tdc.REPEAT_LIMIT_VECT
                    mod_r2nd_c1 = r2nd_c1_size % vor_lp_block_cnt
                    loop_src_ub_offset = data_unit
                    loop_dst_ub_offset = vnc_col_blk_cnt * tdc.C0_16
                    dst_blk_stride = 1 * dtype_factor
                    mod_dst_offset = repeat_cnt_r2nd_c1 * vor_lp_block_cnt * data_unit
                    mod_src_offset = repeat_cnt_r2nd_c1 * vor_lp_block_cnt * left_pl_size * data_unit
                    repeat_full_dst_offset = tdc.REPEAT_LIMIT_VECT * vor_lp_block_cnt * data_unit
                    repeat_full_src_offset = tdc.REPEAT_LIMIT_VECT * vor_lp_block_cnt * left_pl_size * data_unit
                    r2nd_c1_args = (tik_inst, src_ub_int16, dst_ub_int16, zero_ub, left_pl_size, loop_src_ub_offset,
                                    loop_dst_ub_offset, repeat_full_left_r2nd_c1, mod_r2nd_c1, repeat_full_cnt_r2nd_c1,
                                    dst_blk_stride, src0_blk_stride, 0, 0, mod_dst_offset, mod_src_offset,
                                    repeat_full_dst_offset, repeat_full_src_offset, vor_lp_block_cnt, dtype_factor)
                    _vor_data_move(r2nd_c1_args)
            with tik_inst.else_scope():
                dst_blk_stride = vnc_col_blk_cnt
                _update_vor_block_cnt(tik_inst, vor_lp_block_cnt, dst_blk_stride)
                # dst rep stride is too large or bank conflict
                with tik_inst.if_scope(tik.any(vor_lp_block_cnt <= 1,
                                               dst_blk_stride % tdc.C0_16 == 0)):
                    with tik_inst.for_range(0, r2nd_c1_size) as r2nd_c1_idx:
                        tik_inst.data_move(src_ub[r2nd_c1_idx * c0_len + ub_offset_2011],
                                           dst_ub[r2nd_c1_idx * left_pl_size * c0_len],
                                           0, left_pl_size, c0_len // ele_per_block,
                                           0, (vnc_col_len - c0_len) // ele_per_block)
                with tik_inst.else_scope():
                    repeat_cnt_cl = left_pl_size // vor_lp_block_cnt
                    repeat_full_cnt_cl = repeat_cnt_cl // tdc.REPEAT_LIMIT_VECT
                    repeat_full_left_cl = repeat_cnt_cl % tdc.REPEAT_LIMIT_VECT
                    mod_cl = left_pl_size % vor_lp_block_cnt
                    loop_src_ub_offset = left_pl_size * data_unit
                    loop_dst_ub_offset = data_unit
                    src0_blk_stride = 1 * dtype_factor
                    mod_dst_offset = repeat_cnt_cl * vor_lp_block_cnt * vnc_col_blk_cnt * tdc.C0_16
                    mod_src_offset = repeat_cnt_cl * vor_lp_block_cnt * data_unit
                    repeat_full_dst_offset = tdc.REPEAT_LIMIT_VECT * vor_lp_block_cnt * vnc_col_blk_cnt * tdc.C0_16
                    repeat_full_src_offset = tdc.REPEAT_LIMIT_VECT * vor_lp_block_cnt * data_unit
                    cl_args = (tik_inst, src_ub_int16, dst_ub_int16, zero_ub, r2nd_c1_size, loop_src_ub_offset,
                               loop_dst_ub_offset, repeat_full_left_cl, mod_cl, repeat_full_cnt_cl,
                               dst_blk_stride, src0_blk_stride, 0, 0, mod_dst_offset, mod_src_offset,
                               repeat_full_dst_offset, repeat_full_src_offset, vor_lp_block_cnt, dtype_factor)
                    _vor_data_move(cl_args)


# pylint: disable=unused-variable
def _copy_data_out(copy_out_args):
    """
    copy data from ub to gm
    """

    (tik_inst, dst_out_gm, dst_ub, left_pl_size, src_left_step_out, r2nd_pl_size, dst_r2nd_step_out, c1_pl_size,
     sub_c_size, all_c_in, ele_per_block, c0_len, tiling_mode, vnc_col_len, all_r2nd_in, r2nd_lp_idx) = copy_out_args

    with tik_inst.new_stmt_scope():
        r2nd_lp_cnt = tik_inst.Scalar(name="r2nd_lp_cnt")
        with tik_inst.if_scope(r2nd_lp_idx > 0):
            r2nd_lp_cnt.set_as(r2nd_lp_idx)
        with tik_inst.else_scope():
            r2nd_lp_cnt.set_as(1)

        def _inner_copy(gm_out_offset, ub_out_offset, nburst):
            with tik_inst.if_scope(tik.all(nburst > ele_per_block, nburst % ele_per_block > 0)):
                tik_inst.data_move(dst_out_gm[gm_out_offset], dst_ub[ub_out_offset],
                                   0, 1, nburst // ele_per_block, 0, 0)
                tik_inst.data_move(dst_out_gm[nburst - ele_per_block + gm_out_offset],
                                   dst_ub[ub_out_offset + nburst//ele_per_block*ele_per_block], 0, 1, 1, 0, 0)
            with tik_inst.else_scope():
                tik_inst.data_move(dst_out_gm[gm_out_offset], dst_ub[ub_out_offset],
                                   0, 1, tdc.ceil_div(nburst, ele_per_block), 0, 0)

        with tik_inst.if_scope(tik.any(tiling_mode == 2012,
                                       tik.all(tiling_mode == 2010, all_r2nd_in == 1,
                                               all_c_in == 1, sub_c_size == c0_len))):
            burst_len = left_pl_size * r2nd_pl_size * sub_c_size
            _inner_copy(0, 0, burst_len)
        with tik_inst.elif_scope(tik.all(tik.any(tiling_mode == 2011, sub_c_size % c0_len == 0), all_c_in == 1)):
            burst_len = r2nd_pl_size * sub_c_size
            with tik_inst.new_stmt_scope(disable_sync=True):
                with tik_inst.for_range(0, left_pl_size) as left_idx:
                    left_gm_offset = left_idx * src_left_step_out
                    with tik_inst.for_range(0, r2nd_lp_cnt) as r2nd_idx:
                        left_ub_offset = (left_idx + r2nd_idx * left_pl_size) * vnc_col_len
                        r2nd_gm_offset = r2nd_idx * burst_len
                        tik_inst.data_move(dst_out_gm[left_gm_offset + r2nd_gm_offset], dst_ub[left_ub_offset],
                                           0, 1, burst_len // ele_per_block, 0, 0)
            with tik_inst.if_scope(burst_len % ele_per_block > 0):
                with tik_inst.new_stmt_scope(disable_sync=True):
                    with tik_inst.for_range(0, left_pl_size) as left_idx:
                        left_gm_offset = left_idx * src_left_step_out
                        with tik_inst.for_range(0, r2nd_lp_cnt) as r2nd_idx:
                            left_ub_offset = (left_idx + r2nd_idx * left_pl_size) * vnc_col_len
                            r2nd_gm_offset = r2nd_idx * burst_len
                            burst_len_block_align = burst_len // ele_per_block * ele_per_block
                            tik_inst.data_move(dst_out_gm[burst_len - ele_per_block + left_gm_offset + r2nd_gm_offset],
                                               dst_ub[left_ub_offset + burst_len_block_align], 0, 1, 1, 0, 0)
        with tik_inst.elif_scope(tiling_mode == 2010):
            with tik_inst.if_scope(all_c_in == 0):
                with tik_inst.new_stmt_scope(disable_sync=True):
                    with tik_inst.for_range(0, left_pl_size) as left_idx:
                        left_gm_offset = left_idx * src_left_step_out
                        left_ub_offset = left_idx * vnc_col_len
                        with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx:
                            burst_len = sub_c_size
                            out_gm_offset = r2nd_idx * dst_r2nd_step_out + left_gm_offset
                            out_ub_offset = r2nd_idx * c1_pl_size * c0_len + left_ub_offset
                            tik_inst.data_move(dst_out_gm[out_gm_offset], dst_ub[out_ub_offset],
                                               0, 1, burst_len // ele_per_block, 0, 0)
                with tik_inst.if_scope(burst_len % ele_per_block > 0):
                    with tik_inst.new_stmt_scope(disable_sync=True):
                        with tik_inst.for_range(0, left_pl_size) as left_idx:
                            left_gm_offset = left_idx * src_left_step_out
                            left_ub_offset = left_idx * vnc_col_len
                            with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx:
                                burst_len = sub_c_size
                                burst_len_block_align = burst_len // ele_per_block * ele_per_block
                                out_gm_offset = r2nd_idx * dst_r2nd_step_out + left_gm_offset
                                out_ub_offset = r2nd_idx * c1_pl_size * c0_len + left_ub_offset
                                tik_inst.data_move(dst_out_gm[burst_len - ele_per_block + out_gm_offset],
                                                   dst_ub[out_ub_offset + burst_len_block_align], 0, 1, 1, 0, 0)
            with tik_inst.else_scope():  # to deduct scalar operations
                burst_len = sub_c_size
                with tik_inst.new_stmt_scope(disable_sync=False):
                    with tik_inst.for_range(0, left_pl_size) as left_idx:
                        left_gm_offset = left_idx * src_left_step_out
                        left_ub_offset = left_idx * vnc_col_len
                        with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx:
                            out_gm_offset = r2nd_idx * dst_r2nd_step_out + left_gm_offset
                            out_ub_offset = r2nd_idx * c1_pl_size * c0_len + left_ub_offset
                            with tik_inst.if_scope(tik.any(r2nd_idx != r2nd_pl_size - 1,
                                                           burst_len % ele_per_block == 0)):
                                tik_inst.data_move(dst_out_gm[out_gm_offset], dst_ub[out_ub_offset],
                                                   0, 1, tdc.ceil_div(burst_len, ele_per_block), 0, 0)
                            with tik_inst.else_scope():
                                tik_inst.data_move(dst_out_gm[out_gm_offset], dst_ub[out_ub_offset],
                                                   0, 1, burst_len // ele_per_block, 0, 0)
                                burst_len_block_align = burst_len // ele_per_block * ele_per_block
                                tik_inst.data_move(dst_out_gm[burst_len - ele_per_block + out_gm_offset],
                                                   dst_ub[out_ub_offset + burst_len_block_align], 0, 1, 1, 0, 0)


def _get_backend_idx(args):
    """
    get backend index for each axis
    """

    (tik_inst, block_idx, mc_pos, used_core_cnt, lc_src_c1_lp_cnt, lc_src_c1_left, c1_backend_idx, src_c1_lp_cnt,
     lc_dst_r2nd_lp_cnt, lc_dst_r2nd_left, dst_r2nd_step_out, r2nd_backend_idx, dst_r2nd_lp_cnt, lc_src_left_lp_cnt,
     lc_src_left_left, r2nd_c_size, left_backend_idx, src_left_lp_cnt, src_c1_left, dst_r2nd_left, src_left_left,
     c_mod_c0, ele_per_block) = args

    with tik_inst.if_scope(tik.all(mc_pos == 1, block_idx == used_core_cnt - 2, lc_src_c1_lp_cnt == 1,
                                   lc_src_c1_left == 1, c_mod_c0 > 0, c_mod_c0 < ele_per_block)):
        c1_backend_idx.set_as(src_c1_lp_cnt - 1)
    with tik_inst.elif_scope(tik.any(tik.all(mc_pos == 1, block_idx == used_core_cnt - 1, lc_src_c1_left == 1,
                                             c_mod_c0 > 0, c_mod_c0 < ele_per_block),
                                     tik.all(mc_pos != 1, lc_src_c1_left == 1, c_mod_c0 > 0,
                                             c_mod_c0 < ele_per_block))):
        c1_backend_idx.set_as(src_c1_lp_cnt - 2)
    with tik_inst.elif_scope(src_c1_left > 0):
        c1_backend_idx.set_as(src_c1_lp_cnt - 1)
    with tik_inst.else_scope():
        c1_backend_idx.set_as(src_c1_lp_cnt)

    with tik_inst.if_scope(tik.all(mc_pos == 2, block_idx == used_core_cnt - 2, lc_dst_r2nd_lp_cnt == 1,
                                   lc_dst_r2nd_left > 0, lc_dst_r2nd_left * dst_r2nd_step_out < ele_per_block)):
        r2nd_backend_idx.set_as(dst_r2nd_lp_cnt - 1)
    with tik_inst.elif_scope(tik.any(tik.all(mc_pos == 2, block_idx == used_core_cnt - 1, lc_dst_r2nd_left > 0,
                                             lc_dst_r2nd_left * dst_r2nd_step_out < ele_per_block),
                                     tik.all(mc_pos != 2, lc_dst_r2nd_left > 0,
                                             lc_dst_r2nd_left * dst_r2nd_step_out < ele_per_block))):
        r2nd_backend_idx.set_as(dst_r2nd_lp_cnt - 2)
    with tik_inst.elif_scope(dst_r2nd_left > 0):
        r2nd_backend_idx.set_as(dst_r2nd_lp_cnt - 1)
    with tik_inst.else_scope():
        r2nd_backend_idx.set_as(dst_r2nd_lp_cnt)

    with tik_inst.if_scope(tik.all(mc_pos == 0, block_idx == used_core_cnt - 2, lc_src_left_lp_cnt == 1,
                                   lc_src_left_left > 0, lc_src_left_left * r2nd_c_size < ele_per_block)):
        left_backend_idx.set_as(src_left_lp_cnt - 1)
    with tik_inst.elif_scope(tik.any(tik.all(mc_pos == 0, block_idx == used_core_cnt - 1,
                                             lc_src_left_left > 0, lc_src_left_left * r2nd_c_size < ele_per_block),
                                     tik.all(mc_pos != 0, lc_src_left_left > 0,
                                             lc_src_left_left * r2nd_c_size < ele_per_block))):
        left_backend_idx.set_as(src_left_lp_cnt - 2)
    with tik_inst.elif_scope(src_left_left > 0):
        left_backend_idx.set_as(src_left_lp_cnt - 1)
    with tik_inst.else_scope():
        left_backend_idx.set_as(src_left_lp_cnt)


def _func_transform_201(tensor_args, tp_args):
    """
    transform function for tiling mode 201
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, dst_ub, zero_ub, ele_per_block, in_dtype = tensor_args
    (tiling_mode, ub_offset, mc_pos, used_core_cnt, srcr2nd_dstr2nd_same, c0_len, core_step_in, core_step_out,
     nlc_dst_r2nd_lp_cnt, nlc_src_c1_lp_cnt, nlc_src_left_lp_cnt, nlc_dst_r2nd_left, nlc_src_c1_left,
     nlc_src_left_left, lc_dst_r2nd_lp_cnt, lc_src_c1_lp_cnt, lc_src_left_lp_cnt, lc_dst_r2nd_left, lc_src_c1_left,
     lc_src_left_left, dst_r2nd_lp_unit, dst_r2nd_step_in, dst_r2nd_step_out, dst_r2nd_lp_step_in, dst_r2nd_lp_step_out,
     src_c1_lp_unit, all_c_in, src_c1_step_in, src_c1_step_out, src_c1_lp_step_in, src_c1_lp_step_out,
     c_mod_c0, src_left_lp_unit, src_left_step_in, src_left_step_out, src_left_lp_step_in, src_left_lp_step_out,
     dst_r2nd_in_0_size, dst_r2nd_in_0_src_rsize, dst_r2nd_in_0_src_asize, dst_r2nd_in_1_size, dst_r2nd_in_1_src_rsize,
     dst_r2nd_in_1_src_asize, dst_r2nd_dims, vnc_col_size, all_r2nd_in) = tp_args

    def _inner_func(args):
        src_left_lp_cnt, src_left_left, src_c1_lp_cnt, src_c1_left, dst_r2nd_lp_cnt, dst_r2nd_left = args
        r2nd_pl_size = tik_inst.Scalar(name="r2nd_pl_size")
        r2nd_beg = tik_inst.Scalar(name="r2nd_beg")
        c1_pl_size = tik_inst.Scalar(name="c1_pl_size")
        left_pl_size = tik_inst.Scalar(name="left_pl_size")
        sub_c_size = tik_inst.Scalar(name="sub_c_size")
        is_last_c1 = tik_inst.Scalar(name="is_last_c1")
        is_c1_back = tik_inst.Scalar(name="is_c1_back")
        is_r2nd_back = tik_inst.Scalar(name="is_r2nd_back")
        is_left_back = tik_inst.Scalar(name="is_left_back")
        vnc_col_len = tik_inst.Scalar(name="vnc_col_len")
        in_gm_offset = tik_inst.Scalar(name="in_gm_offset")
        ub_offset_2011 = tik_inst.Scalar(name="ub_offset_2011")
        r2nd_lp_idx_2011 = tik_inst.Scalar(name="r2nd_lp_idx_2011")
        cur_r2nd_lp_idx = tik_inst.Scalar(name="cur_r2nd_lp_idx")
        r2nd_vnc_cnt = tik_inst.Scalar(name="r2nd_vnc_cnt")
        c1_backend_idx = tik_inst.Scalar(name="c1_backend_idx")
        r2nd_backend_idx = tik_inst.Scalar(name="r2nd_backend_idx")
        left_backend_idx = tik_inst.Scalar(name="left_backend_idx")
        r2nd_c_size = dst_r2nd_in_0_size * dst_r2nd_in_1_size * dst_r2nd_step_out
        r2nd_vnc_cnt.set_as(tdc.VNC_LINES // src_left_lp_unit)

        backend_args = (tik_inst, block_idx, mc_pos, used_core_cnt, lc_src_c1_lp_cnt, lc_src_c1_left, c1_backend_idx,
                        src_c1_lp_cnt, lc_dst_r2nd_lp_cnt, lc_dst_r2nd_left, dst_r2nd_step_out, r2nd_backend_idx,
                        dst_r2nd_lp_cnt, lc_src_left_lp_cnt, lc_src_left_left, r2nd_c_size, left_backend_idx,
                        src_left_lp_cnt, src_c1_left, dst_r2nd_left, src_left_left, c_mod_c0, ele_per_block)
        _get_backend_idx(backend_args)

        with tik_inst.for_range(0, src_c1_lp_cnt) as c1_lp_idx:
            with tik_inst.if_scope(tik.any(c1_lp_idx != src_c1_lp_cnt - 1, src_c1_left == 0)):
                with tik_inst.if_scope(c1_lp_idx == c1_backend_idx):
                    c1_pl_size.set_as(src_c1_lp_unit - 1)
                with tik_inst.else_scope():
                    c1_pl_size.set_as(src_c1_lp_unit)
                # check last c1
                with tik_inst.if_scope(tik.any(tik.all(c1_lp_idx == src_c1_lp_cnt - 1, mc_pos != 1),
                                               tik.all(c1_lp_idx == src_c1_lp_cnt - 1, mc_pos == 1,
                                                       block_idx == used_core_cnt - 1))):
                    is_last_c1.set_as(1)
                with tik_inst.else_scope():
                    is_last_c1.set_as(0)
                is_c1_back.set_as(0)
            with tik_inst.else_scope():
                with tik_inst.if_scope(tik.all(used_core_cnt > 1, lc_src_c1_left == 1,
                                               c_mod_c0 > 0, c_mod_c0 < ele_per_block)):
                    c1_pl_size.set_as(lc_src_c1_left + 1)
                    is_c1_back.set_as(1)
                with tik_inst.else_scope():
                    c1_pl_size.set_as(lc_src_c1_left)
                    is_c1_back.set_as(0)
                is_last_c1.set_as(1)
            c1_backend = is_c1_back
            with tik_inst.if_scope(tik.all(is_last_c1 == 1, c_mod_c0 > 0)):
                sub_c_size.set_as((c1_pl_size - 1) * c0_len + c_mod_c0)
            with tik_inst.else_scope():
                sub_c_size.set_as(c1_pl_size * c0_len)

            with tik_inst.for_range(0, dst_r2nd_lp_cnt) as r2nd_lp_idx:
                with tik_inst.if_scope(tik.any(r2nd_lp_idx != dst_r2nd_lp_cnt - 1, dst_r2nd_left == 0)):
                    with tik_inst.if_scope(r2nd_lp_idx == r2nd_backend_idx):
                        r2nd_pl_size.set_as(dst_r2nd_lp_unit - ele_per_block)
                    with tik_inst.else_scope():
                        r2nd_pl_size.set_as(dst_r2nd_lp_unit)
                    is_r2nd_back.set_as(0)
                with tik_inst.else_scope():
                    with tik_inst.if_scope(tik.all(used_core_cnt > 1, lc_dst_r2nd_left * sub_c_size < ele_per_block)):
                        r2nd_pl_size.set_as(lc_dst_r2nd_left + ele_per_block)
                        is_r2nd_back.set_as(1)
                    with tik_inst.else_scope():
                        r2nd_pl_size.set_as(lc_dst_r2nd_left)
                        is_r2nd_back.set_as(0)
                r2nd_backend = is_r2nd_back * ele_per_block
                with tik_inst.if_scope(mc_pos == 2):
                    r2nd_beg.set_as((r2nd_lp_idx + block_idx * nlc_dst_r2nd_lp_cnt) * dst_r2nd_lp_unit - r2nd_backend)
                with tik_inst.else_scope():
                    r2nd_beg.set_as(r2nd_lp_idx * dst_r2nd_lp_unit - r2nd_backend)

                with tik_inst.for_range(0, src_left_lp_cnt) as left_lp_idx:
                    left_back = tdc.ceil_div(ele_per_block, r2nd_c_size) - lc_src_left_left
                    with tik_inst.if_scope(tik.any(left_lp_idx != src_left_lp_cnt - 1, src_left_left == 0)):
                        with tik_inst.if_scope(left_lp_idx == left_backend_idx):
                            left_pl_size.set_as(src_left_lp_unit - left_back)
                        with tik_inst.else_scope():
                            left_pl_size.set_as(src_left_lp_unit)
                        is_left_back.set_as(0)
                    with tik_inst.else_scope():
                        with tik_inst.if_scope(tik.all(used_core_cnt > 1, lc_src_left_left > 0,
                                                       lc_src_left_left * r2nd_c_size < ele_per_block)):
                            left_pl_size.set_as(lc_src_left_left + left_back)
                            is_left_back.set_as(1)
                        with tik_inst.else_scope():
                            left_pl_size.set_as(lc_src_left_left)
                            is_left_back.set_as(0)
                    left_backend = is_left_back * left_back

                    with tik_inst.if_scope(tiling_mode == 2011):
                        vnc_col_len.set_as(vnc_col_size)
                    with tik_inst.else_scope():
                        vnc_col_len.set_as(c1_pl_size * r2nd_pl_size * c0_len)

                    with tik_inst.if_scope(tik.all(tiling_mode == 2011, r2nd_vnc_cnt > 1,
                                                   r2nd_lp_idx < r2nd_backend_idx)):
                        ub_offset_2011.set_as(r2nd_lp_idx % r2nd_vnc_cnt * src_left_lp_unit * vnc_col_len)
                        r2nd_lp_idx_2011.set_as(r2nd_lp_idx % r2nd_vnc_cnt + 1)
                        cur_r2nd_lp_idx.set_as(r2nd_lp_idx - r2nd_lp_idx_2011 + 1)
                    with tik_inst.else_scope():
                        ub_offset_2011.set_as(0)
                        r2nd_lp_idx_2011.set_as(0)
                        cur_r2nd_lp_idx.set_as(r2nd_lp_idx)

                    with tik_inst.if_scope(srcr2nd_dstr2nd_same == 1):  # such as NC1HWC0 -> NHWC
                        in_offset_args = (left_lp_idx, src_left_lp_step_in, left_backend, src_left_step_in, c1_lp_idx,
                                          src_c1_lp_step_in, c1_backend, src_c1_step_in, r2nd_lp_idx,
                                          dst_r2nd_lp_step_in, r2nd_backend, dst_r2nd_step_in, block_idx * core_step_in)
                        in_gm_offset.set_as(_update_input_offset_0(in_offset_args))
                        copy_in_args = (tik_inst, src_in_gm, src_ub[ub_offset_2011:], in_gm_offset,
                                        left_pl_size, src_left_step_in, c1_pl_size, src_c1_step_in,
                                        r2nd_pl_size, c0_len, ele_per_block, vnc_col_len)
                        _copy_data_in_0(copy_in_args)
                    with tik_inst.else_scope():  # such as DC1HWNoNiC0 -> NDHWC
                        in_offset_args = (left_lp_idx, src_left_lp_step_in, left_backend, src_left_step_in,
                                          c1_lp_idx, src_c1_lp_step_in, c1_backend, src_c1_step_in,
                                          r2nd_beg, block_idx * core_step_in,
                                          dst_r2nd_in_0_size, dst_r2nd_in_0_src_rsize, dst_r2nd_in_0_src_asize,
                                          dst_r2nd_in_1_size, dst_r2nd_in_1_src_rsize, dst_r2nd_in_1_src_asize)
                        in_gm_offset.set_as(_update_input_offset_1(in_offset_args))
                        copy_in_args = (tik_inst, src_in_gm, src_ub, ub_offset_2011, in_gm_offset, dst_ub, zero_ub,
                                        left_pl_size, c1_pl_size, src_c1_step_in, r2nd_beg, r2nd_pl_size,
                                        dst_r2nd_in_0_size, dst_r2nd_in_0_src_asize, dst_r2nd_in_1_src_asize,
                                        c0_len, ele_per_block, vnc_col_len, in_dtype)
                        _copy_data_in_1(copy_in_args)

                    out_gm_args = (cur_r2nd_lp_idx, dst_r2nd_lp_step_out, r2nd_backend, dst_r2nd_step_out,
                                   c1_lp_idx, src_c1_lp_step_out, c1_backend * c0_len, left_lp_idx,
                                   src_left_lp_step_out, left_backend, src_left_step_out, block_idx * core_step_out)
                    out_gm_offset = _update_output_offset(out_gm_args)
                    copy_out_args = (tik_inst, dst_out_gm[out_gm_offset], dst_ub, left_pl_size, src_left_step_out,
                                     r2nd_pl_size, dst_r2nd_step_out, c1_pl_size, sub_c_size, all_c_in, ele_per_block,
                                     c0_len, tiling_mode, vnc_col_len, all_r2nd_in, r2nd_lp_idx_2011)
                    with tik_inst.if_scope(tiling_mode == 2010):  # use ubuf_2_ubuf
                        with tik_inst.if_scope(tik.all(all_c_in == 1, sub_c_size == c0_len)):
                            tik_inst.data_move(dst_ub, src_ub, 0, 1,
                                               left_pl_size * r2nd_pl_size * sub_c_size // ele_per_block, 0, 0)
                        with tik_inst.else_scope():
                            ubuf_args = (tik_inst, src_ub, dst_ub, zero_ub, left_pl_size, c1_pl_size, r2nd_pl_size,
                                         c0_len, ele_per_block, sub_c_size, all_c_in, in_dtype)
                            _ubuf_2_ubuf_convert(ubuf_args)
                    with tik_inst.elif_scope(tik.any(r2nd_lp_idx == r2nd_backend_idx - 1,
                                                     r2nd_lp_idx_2011 == r2nd_vnc_cnt, r2nd_lp_idx_2011 == 0)):
                        vnc_args = (tik_inst, src_ub, dst_ub, left_pl_size, c1_pl_size, r2nd_pl_size, c0_len,
                                    ele_per_block, in_dtype, tiling_mode, all_c_in, sub_c_size, vnc_col_len,
                                    r2nd_lp_idx_2011)
                        _twice_vnchwconv_no_invert(vnc_args)
                    with tik_inst.if_scope(tik.any(r2nd_lp_idx == r2nd_backend_idx - 1,
                                                   r2nd_lp_idx_2011 == r2nd_vnc_cnt, r2nd_lp_idx_2011 == 0)):
                        _copy_data_out(copy_out_args)

    with tik_inst.if_scope(block_idx != used_core_cnt - 1):
        nlc_args = (nlc_src_left_lp_cnt, nlc_src_left_left, nlc_src_c1_lp_cnt,
                    nlc_src_c1_left, nlc_dst_r2nd_lp_cnt, nlc_dst_r2nd_left)
        _inner_func(nlc_args)
    with tik_inst.else_scope():
        lc_args = (lc_src_left_lp_cnt, lc_src_left_left, lc_src_c1_lp_cnt,
                   lc_src_c1_left, lc_dst_r2nd_lp_cnt, lc_dst_r2nd_left)
        _inner_func(lc_args)


# pylint: disable=unused-argument
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
        source data format, can be NC1HWC0 etc.
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
    in_dtype = src.get("dtype").lower() if src.get("dtype").lower() != "bfloat16" else "float16"
    in_dtype_bytes = tdc.get_dtype_len(in_dtype)
    tiling_dtype_bytes = tdc.get_dtype_len("int64")
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1, 256) - TILING_CTRL_PARAM[1] * tiling_dtype_bytes // in_dtype_bytes
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)

    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "src_in_gm")
    dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm")
    tiling_gm = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],), tik.scope_gm, "tiling_gm")
    half_ub = ub_size // 2
    src_ub = tik_inst.Tensor(in_dtype, (half_ub,), tik.scope_ubuf, "src_ub")
    dst_ub = tik_inst.Tensor(in_dtype, (half_ub,), tik.scope_ubuf, "dst_ub")
    zero_ub = tik_inst.Tensor("int16", (tdc.MASK_128,), tik.scope_ubuf, "zero_ub")  # for vor
    tiling_ub = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],), tik.scope_ubuf, "tiling_ub")
    tiling_params = [tik_inst.Scalar(TILING_CTRL_PARAM[0]) for i in range(TILING_CTRL_PARAM[1])]
    _get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes)
    tdc.clean_ubuf(tik_inst, zero_ub, 0, tdc.MASK_128)

    used_core_cnt = tiling_params[3]
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm,
                           src_ub, dst_ub, zero_ub, block_elem_cnt, in_dtype]
            tp_args = tiling_params[0:46]
            _func_transform_201(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[src_in_gm], outputs=[dst_out_gm], flowtable=[tiling_gm], enable_l2=False,
                      config={"dynamic_tik": True, "out_of_bound_sync_check": True})
    tbe_context.get_context().add_compile_info("vars", {"ub_size": ub_size,
                                                        "block_dim": tdc.CORE_DIM_NUM,
                                                        "group": 1})
