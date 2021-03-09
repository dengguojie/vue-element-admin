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

def _twice_vnchwconv_invert(args):
    """
    do ncdh to ndhc transform by twice vnchwconv
    """

    tik_inst, src_ub, ub_offset, c_plp_size, cr_pln_size, is_last_c1, c_mod_c0, vnc_line_size = args
    size_factor = tdc.get_dtype_factor(src_ub.dtype)
    src_ub_fp16 = src_ub.reinterpret_cast_to("float16")
    ub_offset_fp16 = ub_offset * size_factor
    cr_pln_block_cnt = tdc.ceil_div(cr_pln_size * size_factor, tdc.C0_16)
    cr_pln_block_align_size = cr_pln_block_cnt * tdc.C0_16
    vnc_col_len = vnc_line_size * size_factor

    # do ncdh -> cdhn
    src_addr_list = [src_ub_fp16[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
    dst_addr_list = [src_ub_fp16[ub_offset_fp16 + tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
    repeat_cnt = c_plp_size * cr_pln_block_cnt
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

    with tik_inst.if_scope(is_last_c1 > 0):  # to pad c_mod_c0 to c0
        tdc.clean_ubuf(tik_inst, src_ub_fp16, ub_offset_fp16 + repeat_cnt * tdc.C0_16 * tdc.VNC_LINES,
                       (tdc.C0_16 - c_mod_c0) * cr_pln_block_cnt * tdc.C0_16 * tdc.VNC_LINES)

    with tik_inst.for_range(0, tdc.C0_16) as c_idx:  # do cdhn -> dhcn
        tik_inst.data_move(src_ub_fp16[c_idx * size_factor * tdc.VNC_LINES],
                           src_ub_fp16[ub_offset_fp16 + c_idx * cr_pln_block_align_size * tdc.VNC_LINES],
                           0, cr_pln_size, 1 * size_factor, 0, (tdc.C0_16 - 1) * size_factor)

    # do dhcn -> ndhc
    vnc_row_size = cr_pln_size * size_factor * tdc.C0_16
    src_addr_list = [src_ub_fp16[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
    dst_addr_list = [src_ub_fp16[ub_offset_fp16 + vnc_row_size * i] for i in tdc.ADDR_IDX_LIST]
    repeat_cnt = vnc_row_size // tdc.C0_16
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

    cr_lp_idx, src_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, \
    in_idx_0_size, in_idx_0_dst_rsize, in_idx_0_src_asize, \
    in_idx_1_size, in_idx_1_dst_rsize, in_idx_1_src_asize, ele_idx, core_step_in = args

    in_offset = (cr_lp_idx * src_cr_lp_step_in + c_lp_idx * src_c_lp_step_in +
                 ele_idx // in_idx_0_dst_rsize % in_idx_0_size * in_idx_0_src_asize +
                 ele_idx // in_idx_1_dst_rsize % in_idx_1_size * in_idx_1_src_asize + core_step_in)

    return in_offset


def _update_output_offset(args, ele_idx, cr_idx):
    """
    count output gm offset
    """

    c_lp_idx, src_c_lp_step_out, out_idx_0_size, out_idx_0_dst_rsize, out_idx_0_dst_asize, \
    out_idx_1_size, out_idx_1_dst_rsize, out_idx_1_dst_asize, out_cr_idx_0_size, out_cr_idx_0_dst_rsize, \
    out_cr_idx_0_dst_asize, out_cr_idx_1_size, out_cr_idx_1_dst_rsize, out_cr_idx_1_dst_asize, core_step_out = args

    out_offset = (c_lp_idx * src_c_lp_step_out +
                  ele_idx // out_idx_0_dst_rsize % out_idx_0_size * out_idx_0_dst_asize +
                  ele_idx // out_idx_1_dst_rsize % out_idx_1_size * out_idx_1_dst_asize +
                  cr_idx // out_cr_idx_0_dst_rsize % out_cr_idx_0_size * out_cr_idx_0_dst_asize +
                  cr_idx // out_cr_idx_1_dst_rsize % out_cr_idx_1_size * out_cr_idx_1_dst_asize + core_step_out)

    return out_offset


def _copy_data_in(args):
    """
    copy data from gm to ub
    """

    tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset, c_step_in, c_plp_size, cr_pln_size, ele_per_block = args
    cr_pln_block_align_size = tdc.ceil_fill(cr_pln_size, ele_per_block)
    cr_pln_block_cnt = tdc.ceil_div(cr_pln_size, ele_per_block)

    with tik_inst.for_range(0, c_plp_size) as c_idx:
        tik_inst.data_move(src_ub[in_ub_offset + c_idx * cr_pln_block_align_size],
                           src_in_gm[in_gm_offset + c_idx * c_step_in], 0, 1, cr_pln_block_cnt, 0, 0)


# pylint: disable=unused-variable
def _copy_data_out(out_offset_args, copy_out_args):
    """
    copy data from ub to gm
    """

    tik_inst, dst_out_gm, out_ub, out_beg_idx, out_end_idx, cr_beg_idx, c_plp_size, \
    cr_pln_size, src_2_dst_flag, left_axis0_size, cr_axis0_size, ele_per_block = copy_out_args

    vnc_line_cnt = (out_end_idx - out_beg_idx)
    out_ub_offset = tik_inst.Scalar(name="out_ub_offset")
    out_gm_offset = tik_inst.Scalar(name="out_gm_offset")
    out_cr_idx = tik_inst.Scalar(name="out_cr_idx")

    with tik_inst.for_range(0, vnc_line_cnt) as left_axis_idx:
        with tik_inst.for_range(0, cr_pln_size) as cr_axis_idx:
            ele_idx = left_axis_idx + out_beg_idx
            cr_idx = cr_axis_idx + cr_beg_idx
            with tik_inst.if_scope(tik.any(cr_idx % cr_axis0_size == 0, cr_axis_idx == 0)):
                out_cr_idx.set_as(cr_idx)
                out_gm_offset.set_as(_update_output_offset(out_offset_args, ele_idx, cr_idx))
                out_ub_offset.set_as(cr_axis_idx * tdc.C0_16 + left_axis_idx * cr_pln_size * tdc.C0_16)
            with tik_inst.if_scope(tik.any((cr_idx + 1) % cr_axis0_size == 0, cr_axis_idx == cr_pln_size - 1)):
                tik_inst.data_move(dst_out_gm[out_gm_offset], out_ub[out_ub_offset],
                                   0, 1, (cr_idx + 1 - out_cr_idx) * tdc.C0_16 // ele_per_block, 0, 0)


def _func_transform_100(tensor_args, tp_args):
    """
    transform function for tiling mode 100
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, ele_per_block = tensor_args
    (ub_offset, mc_flag, used_core, core_step_in, core_step_out, nlc_cr_lp_cnt,
     nlc_c_lp_cnt, nlc_left_lp_cnt, nlc_cr_left, nlc_c_left, lc_cr_lp_cnt, lc_c_lp_cnt,
     lc_left_lp_cnt, lc_cr_left, lc_c_left, src_cr_lp_unit, src_cr_lp_step_in, src_cr_lp_step_out,
     src_c_step_in, src_c_lp_unit, src_c_lp_step_in, src_c_lp_step_out, c_mod_c0, in_idx_0_size,
     in_idx_0_dst_rsize, in_idx_0_src_asize, in_idx_1_size, in_idx_1_dst_rsize, in_idx_1_src_asize,
     out_idx_0_size, out_idx_0_dst_rsize, out_idx_0_dst_asize, out_idx_1_size, out_idx_1_dst_rsize,
     out_idx_1_dst_asize, out_cr_idx_0_size, out_cr_idx_0_dst_rsize, out_cr_idx_0_dst_asize,
     out_cr_idx_1_size, out_cr_idx_1_dst_rsize, out_cr_idx_1_dst_asize, src_2_dst_flag, vnc_line_size) = tp_args
    cr_lp_cnt = tik_inst.Scalar(name="cr_lp_cnt")
    c_lp_cnt = tik_inst.Scalar(name="c_lp_cnt")
    c_left = tik_inst.Scalar(name="c_left")
    cr_left = tik_inst.Scalar(name="cr_left")

    def _inner_func(begin_idx, end_idx):
        cr_pln_size = tik_inst.Scalar(name="cr_pln_size")
        c_plp_size = tik_inst.Scalar(name="c_plp_size")
        is_last_c1 = tik_inst.Scalar(name="is_last_c1")
        cr_beg_idx = tik_inst.Scalar(name="cr_beg_idx")
        out_beg_idx = tik_inst.Scalar(name="out_beg_idx")
        out_end_idx = tik_inst.Scalar(name="out_end_idx")
        with tik_inst.for_range(0, cr_lp_cnt) as cr_lp_idx:
            with tik_inst.if_scope(tik.all(mc_flag == 0, core_step_out == 0)):  # multiple core on axis cr
                cr_beg_idx.set_as(cr_lp_idx * src_cr_lp_unit + block_idx * core_step_in)
            with tik_inst.else_scope():
                cr_beg_idx.set_as(cr_lp_idx * src_cr_lp_unit)

            with tik_inst.if_scope(tik.any(cr_lp_idx != cr_lp_cnt - 1, cr_left == 0)):
                cr_pln_size.set_as(src_cr_lp_unit)
            with tik_inst.else_scope():
                cr_pln_size.set_as(lc_cr_left)

            with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:
                with tik_inst.if_scope(tik.any(c_lp_idx != c_lp_cnt - 1, c_left == 0)):
                    c_plp_size.set_as(src_c_lp_unit)
                    is_last_c1.set_as(0)
                with tik_inst.else_scope():
                    c_plp_size.set_as(lc_c_left)
                    is_last_c1.set_as(1)

                with tik_inst.for_range(begin_idx, end_idx) as ele_idx:
                    with tik_inst.if_scope((ele_idx - begin_idx) % tdc.VNC_LINES == 0):
                        out_beg_idx.set_as(ele_idx)
                    in_offset_args = cr_lp_idx, src_cr_lp_step_in, c_lp_idx, src_c_lp_step_in, \
                                     in_idx_0_size, in_idx_0_dst_rsize, in_idx_0_src_asize, \
                                     in_idx_1_size, in_idx_1_dst_rsize, in_idx_1_src_asize, ele_idx, \
                                     block_idx * core_step_in
                    in_gm_offset = _update_input_offset(in_offset_args)
                    in_ub_offset = (ele_idx - begin_idx) % tdc.VNC_LINES * vnc_line_size
                    copy_in_args = tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset, \
                                   src_c_step_in, c_plp_size, cr_pln_size, ele_per_block
                    _copy_data_in(copy_in_args)
                    with tik_inst.if_scope(tik.any((ele_idx - begin_idx + 1) % tdc.VNC_LINES == 0,
                                                   ele_idx == end_idx - 1)):
                        out_end_idx.set_as(ele_idx + 1)
                        vnc_args = tik_inst, src_ub, ub_offset, c_plp_size, cr_pln_size, \
                                   is_last_c1, c_mod_c0, vnc_line_size
                        _twice_vnchwconv_invert(vnc_args)
                        out_gm_args = c_lp_idx, src_c_lp_step_out, \
                                      out_idx_0_size, out_idx_0_dst_rsize, out_idx_0_dst_asize, \
                                      out_idx_1_size, out_idx_1_dst_rsize, out_idx_1_dst_asize, \
                                      out_cr_idx_0_size, out_cr_idx_0_dst_rsize, out_cr_idx_0_dst_asize, \
                                      out_cr_idx_1_size, out_cr_idx_1_dst_rsize, out_cr_idx_1_dst_asize, \
                                      block_idx * core_step_out
                        copy_out_args = tik_inst, dst_out_gm, src_ub[ub_offset], out_beg_idx, out_end_idx, \
                                        cr_beg_idx, c_plp_size, cr_pln_size, src_2_dst_flag, out_idx_0_size, \
                                        out_cr_idx_0_size, ele_per_block
                        _copy_data_out(out_gm_args, copy_out_args)

    with tik_inst.if_scope(block_idx != used_core - 1):
        cr_lp_cnt.set_as(nlc_cr_lp_cnt)
        c_lp_cnt.set_as(nlc_c_lp_cnt)
        cr_left.set_as(nlc_cr_left)
        c_left.set_as(nlc_c_left)
        _inner_func(mc_flag * block_idx * nlc_left_lp_cnt, (mc_flag * block_idx + 1) * nlc_left_lp_cnt)
    with tik_inst.else_scope():
        cr_lp_cnt.set_as(lc_cr_lp_cnt)
        c_lp_cnt.set_as(lc_c_lp_cnt)
        cr_left.set_as(lc_cr_left)
        c_left.set_as(lc_c_left)
        _inner_func(mc_flag * block_idx * nlc_left_lp_cnt, mc_flag * block_idx * nlc_left_lp_cnt + lc_left_lp_cnt)


def trans_data_positive_source_nct(src, dst, src_format, dst_format, kernel_name="trans_data_positive_source_nct"):
    """
    positive transform for last dimension of target format is c

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
        kernel name, default value is "trans_data_positive_source_ntc"

    Returns
    -------
    None
    """

    src_format = src_format.upper()
    dst_format = dst_format.upper()
    in_shape = list(src.get("shape"))
    out_shape = list(dst.get("shape"))
    in_dtype = src.get("dtype").lower()
    in_dtype_bytes = tdc.get_dtype_len(in_dtype)
    tiling_dtype_bytes = tdc.get_dtype_len("int64")
    ub_size = tdc.get_max_element_in_ub(in_dtype, 1) - TILING_CTRL_PARAM[1] * tiling_dtype_bytes // in_dtype_bytes
    block_elem_cnt = tdc.BLOCK_BYTE_SIZE // tdc.get_dtype_len(in_dtype)

    tik_inst = tik.Tik()
    src_in_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "src_in_gm")
    dst_out_gm = tik_inst.Tensor(in_dtype, (tdc.MAX_INT64_VALUE,), tik.scope_gm, "dst_out_gm", is_atomic_add=True)
    tiling_gm = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],), tik.scope_gm, "tiling_gm")
    src_ub = tik_inst.Tensor(in_dtype, (ub_size,), tik.scope_ubuf, "total_ub")
    tiling_ub = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],), tik.scope_ubuf, "tiling_ub")
    tiling_params = [tik_inst.Scalar(TILING_CTRL_PARAM[0]) for i in range(TILING_CTRL_PARAM[1])]
    _get_tiling_params(tik_inst, tiling_ub, tiling_gm, tiling_params, tiling_dtype_bytes)

    tiling_mode = tiling_params[0]
    used_core_cnt = tiling_params[3]
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            with tik_inst.if_scope(tiling_mode == 100):
                tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub, block_elem_cnt]
                tp_args = tiling_params[1:44]
                _func_transform_100(tensor_args, tp_args)

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
