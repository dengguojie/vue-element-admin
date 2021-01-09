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
import te.lang.dynamic
from te import tik
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from .. import trans_data_common_func as tdc


# used for scalar
PAD_IDX_LIST = (0, 1, 2)
# frame up levels
FRAME_LEVEL = 3
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


def _twice_vnchwconv_no_invert(args):
    """
    do ncht to nhct transform by twice vnchwconv
    """

    tik_inst, src_ub, ub_offset, c1_pl_size, r2nd_pl_size, pln_dst_c_cnt, \
    src_2_dst_flag, is_last_c1 = args

    vnc_col_len = pln_dst_c_cnt * c1_pl_size * r2nd_pl_size * tdc.C0_16
    # do hcnt -> cnth
    src_addr_list = [src_ub[vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
    dst_addr_list = [src_ub[ub_offset + tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
    repeat_cnt = pln_dst_c_cnt * c1_pl_size * r2nd_pl_size
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
                dst_ub_offset += c1_idx * tdc.C0_16 * tdc.VNC_LINES
                src_ub_offset += c1_idx * r2nd_pl_size * tdc.C0_16 * tdc.VNC_LINES
                tik_inst.data_move(src_ub[dst_ub_offset], src_ub[ub_offset + src_ub_offset],
                                   0, r2nd_pl_size, tdc.C0_16, 0, (c1_pl_size - 1) * tdc.C0_16)
    with tik_inst.else_scope():  # such as DC1HWNoNic0 -> NDHWC
        with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx:
            dst_ub_offset = r2nd_idx * pln_dst_c_cnt * c1_pl_size * tdc.C0_16 * tdc.VNC_LINES
            src_ub_offset = r2nd_idx * tdc.C0_16 * tdc.VNC_LINES
            tik_inst.data_move(src_ub[dst_ub_offset], src_ub[ub_offset + src_ub_offset],
                               0, pln_dst_c_cnt * c1_pl_size, tdc.C0_16, (r2nd_pl_size - 1) * tdc.C0_16, 0)

    # do hctn -> nhct
    src_addr_list = [src_ub[tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
    dst_addr_list = [src_ub[ub_offset + vnc_col_len * i] for i in tdc.ADDR_IDX_LIST]
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

    tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset, c1_step_in, c1_pl_size, r2nd_pl_size = args

    with tik_inst.for_range(0, c1_pl_size) as c1_idx:
        tik_inst.data_move(src_ub[in_ub_offset + c1_idx * r2nd_pl_size * tdc.C0_16],
                           src_in_gm[in_gm_offset + c1_idx * c1_step_in], 0, 1, r2nd_pl_size, 0, 0)


def _copy_data_out(out_offset_args, copy_out_args):
    """
    copy data from ub to gm
    """

    src_r2nd_lp_step_out = out_offset_args[2]
    tik_inst, dst_out_gm, out_ub, pln_dst_c_cnt, \
    out_beg_idx, out_end_idx, c1_pl_size, r2nd_pl_size, src_2_dst_flag, is_last_c1, c_mod_c0, adj_addr = copy_out_args

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

    def _inner_copy(vnc_line_idx, pln_c_cnt):
        with tik_inst.for_range(0, pln_c_cnt) as c_idx:
            ele_idx = vnc_line_idx * pln_dst_c_cnt + c_idx + out_beg_idx
            out_offset = _update_output_offset(out_offset_args, ele_idx) - adj_addr
            ub_offset = vnc_line_idx * vnc_line_ub_offset + c_idx * left_axis_ub_offset
            with tik_inst.for_range(0, r2nd_pl_size) as r2nd_idx:
                with tik_inst.if_scope(tik.all(is_last_c1 > 0, c_mod_c0 > 0)):
                    tik_inst.data_move(dst_out_gm[out_offset + r2nd_idx * src_r2nd_lp_step_out],
                                       out_ub[ub_offset + r2nd_idx * r2nd_ub_offset], 0, 1, c1_pl_size - 1, 0, 0)
                    with tik_inst.new_stmt_scope():
                        tmp_reg = tik_inst.Scalar(out_ub.dtype)
                        with tik_inst.for_range(0, tdc.C0_16) as j:
                            tmp_reg.set_as(out_ub[ub_offset + r2nd_idx * r2nd_ub_offset +
                                                     (c1_pl_size - 2) * tdc.C0_16 + c_mod_c0 + j])
                            out_ub[j].set_as(tmp_reg)
                        tik_inst.data_move(dst_out_gm[out_offset + r2nd_idx * src_r2nd_lp_step_out +
                                                      (c1_pl_size - 2) * tdc.C0_16 + c_mod_c0],
                                           out_ub, 0, 1, 1, 0, 0)
                with tik_inst.else_scope():
                    tik_inst.data_move(dst_out_gm[out_offset + r2nd_idx * src_r2nd_lp_step_out],
                                       out_ub[ub_offset + r2nd_idx * r2nd_ub_offset], 0, 1, c1_pl_size, 0, 0)

    with tik_inst.for_range(0, vnc_line_cnt) as vnc_line_idx:
        _inner_copy(vnc_line_idx, pln_dst_c_cnt)
    with tik_inst.if_scope(left_c_cnt > 0):
        _inner_copy(vnc_line_cnt, left_c_cnt)


def _func_transform_201(tensor_args, tp_args):
    """
    transform function for tiling mode 201
    """

    tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub = tensor_args
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

    def _inner_func(begin_idx, end_idx):
        r2nd_pl_size = tik_inst.Scalar(name="r2nd_pl_size")
        c1_pl_size = tik_inst.Scalar(name="c1_pl_size")
        is_last_c1 = tik_inst.Scalar(name="is_last_c1")
        is_c1_left_one = tik_inst.Scalar(name="is_c1_left_one")
        out_beg_idx = tik_inst.Scalar(name="out_beg_idx")
        out_end_idx = tik_inst.Scalar(name="out_end_idx")
        with tik_inst.for_range(0, r2nd_lp_cnt) as r2nd_lp_idx:
            with tik_inst.if_scope(tik.any(r2nd_lp_idx != r2nd_lp_cnt - 1, r2nd_left == 0)):
                r2nd_pl_size.set_as(src_r2nd_lp_unit)
            with tik_inst.else_scope():
                r2nd_pl_size.set_as(lc_r2nd_left)

            with tik_inst.for_range(0, c1_lp_cnt) as c1_lp_idx:
                with tik_inst.if_scope(tik.any(c1_lp_idx != c1_lp_cnt - 1, c1_left == 0)):
                    with tik_inst.if_scope(tik.all(c1_lp_idx == c1_lp_cnt - 2, c1_left == 1)):
                        c1_pl_size.set_as(src_c1_lp_unit - 1)
                    with tik_inst.else_scope():
                        c1_pl_size.set_as(src_c1_lp_unit)

                    with tik_inst.if_scope(c1_lp_idx == c1_lp_cnt - 1):
                        is_last_c1.set_as(1)
                    with tik_inst.else_scope():
                        is_last_c1.set_as(0)
                    is_c1_left_one.set_as(0)

                with tik_inst.else_scope():
                    with tik_inst.if_scope(c1_left == 1):
                        c1_pl_size.set_as(lc_c1_left + 1)
                        is_c1_left_one.set_as(1)
                    with tik_inst.else_scope():
                        c1_pl_size.set_as(lc_c1_left)
                        is_c1_left_one.set_as(0)
                    is_last_c1.set_as(1)

                with tik_inst.for_range(begin_idx, end_idx) as ele_idx:
                    vnc_pl_c_cnt = tdc.VNC_LINES * pln_dst_c_cnt
                    with tik_inst.if_scope((ele_idx - begin_idx) % vnc_pl_c_cnt == 0):
                        out_beg_idx.set_as(ele_idx)
                    in_offset_args = r2nd_lp_idx, src_r2nd_lp_step_in, c1_lp_idx, src_c1_lp_step_in, \
                                     in_idx_0_size, in_idx_0_dst_rsize, in_idx_0_src_asize, \
                                     in_idx_1_size, in_idx_1_dst_rsize, in_idx_1_src_asize, \
                                     in_idx_2_size, in_idx_2_dst_rsize, in_idx_2_src_asize, ele_idx, \
                                     block_idx * core_step_in
                    in_gm_offset = _update_input_offset(in_offset_args) - is_c1_left_one * src_c1_step_in
                    in_ub_offset = (ele_idx - begin_idx) % vnc_pl_c_cnt * c1_pl_size * r2nd_pl_size * tdc.C0_16
                    copy_in_args = tik_inst, src_in_gm, src_ub, in_gm_offset, in_ub_offset, \
                                   src_c1_step_in, c1_pl_size, r2nd_pl_size
                    _copy_data_in(copy_in_args)
                    with tik_inst.if_scope(tik.any((ele_idx - begin_idx + 1) % vnc_pl_c_cnt == 0,
                                                   ele_idx == end_idx - 1)):
                        out_end_idx.set_as(ele_idx + 1)
                        vnc_args = tik_inst, src_ub, ub_offset, c1_pl_size, r2nd_pl_size, pln_dst_c_cnt, \
                                   src_2_dst_flag, is_last_c1
                        _twice_vnchwconv_no_invert(vnc_args)
                        out_gm_args = r2nd_lp_idx, src_r2nd_lp_unit, src_r2nd_lp_step_out, \
                                      c1_lp_idx, src_c1_lp_step_out, \
                                      out_idx_0_size, out_idx_0_dst_rsize, out_idx_0_dst_asize, \
                                      out_idx_1_size, out_idx_1_dst_rsize, out_idx_1_dst_asize, \
                                      out_idx_2_size, out_idx_2_dst_rsize, out_idx_2_dst_asize,\
                                      block_idx * core_step_out
                        copy_out_args = tik_inst, dst_out_gm, src_ub[ub_offset], pln_dst_c_cnt, \
                                        out_beg_idx, out_end_idx, c1_pl_size, r2nd_pl_size, src_2_dst_flag, \
                                        is_last_c1, c_mod_c0, is_c1_left_one * tdc.C0_16
                        _copy_data_out(out_gm_args, copy_out_args)

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


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_STR, para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
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
    used_core_cnt = tiling_params[3]
    with tik_inst.for_range(0, tdc.CORE_DIM_NUM, block_num=tdc.CORE_DIM_NUM) as block_idx:
        with tik_inst.if_scope(block_idx < used_core_cnt):
            with tik_inst.if_scope(tiling_mode == 201):
                tensor_args = [tik_inst, block_idx, src_in_gm, dst_out_gm, src_ub]
                tp_args = tiling_params[1:44]
                _func_transform_201(tensor_args, tp_args)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[src_in_gm], outputs=[dst_out_gm], flowtable=[tiling_gm], enable_l2=False)
    te.op.add_compile_info("vars", {"srcFormat": src_format,
                                    "dstFormat": dst_format,
                                    "dType": in_dtype,
                                    "ubSize": ub_size,
                                    "blockDim": tdc.CORE_DIM_NUM,
                                    "inputSize": -1,
                                    "hiddenSize": -1,
                                    "group": 1})
