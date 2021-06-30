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
from impl.util.platform_adapter import tik
from impl import trans_data_common_func as tdc
from impl.util.platform_adapter import tbe_context


# used for scalar
PAD_IDX_LIST = (0, 1)
# frame up levels
FRAME_LEVEL = 2
TILING_CTRL_PARAM = ("int64", 128)


def _get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes):
    """
    get tiling parameters function
    """

    ele_per_block = tdc.BLOCK_BYTE_SIZE // tiling_dtype_bytes
    tik_inst.data_move(tiling_ub, tiling_gm, 0, 1, TILING_CTRL_PARAM[1] // ele_per_block, 0, 0)
    for reg_idx in range(TILING_CTRL_PARAM[1]):
        tiling_params[reg_idx].set_as(tiling_ub[reg_idx])


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
    in_dtype = src.get("dtype").lower() if src.get("dtype").lower() != "bool" else "int8"
    in_dtype_bytes = tdc.get_dtype_len(in_dtype)
    tiling_dtype_bytes = tdc.get_dtype_len("int64")
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1) - TILING_CTRL_PARAM[1] * tiling_dtype_bytes // in_dtype_bytes
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)
    
    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "src_in_gm")
    if src_format.upper() == "NHWC" and dst_format.upper() == "NC1HWC0":
        dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm", is_atomic_add=False)
    else:
        dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm", is_atomic_add=True)
    tiling_gm = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],), tik.scope_gm, "tiling_gm")
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tik.scope_ubuf, "total_ub")
    tiling_ub = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],), tik.scope_ubuf, "tiling_ub")
    tiling_params = [tik_inst.Scalar(TILING_CTRL_PARAM[0]) for i in range(TILING_CTRL_PARAM[1])]
    _get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes)


    tiling_mode = tiling_params[0]
    used_core_cnt = tiling_params[2]
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt]
            with tik_inst.if_scope(tiling_mode == 1010):
                tp_args = tiling_params[1:32]
                _func_transform_1010(tensor_args, tp_args)
            with tik_inst.if_scope(tiling_mode == 1011):
                tp_args = tiling_params[1:38]
                _func_transform_1011(tensor_args, tp_args)

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

