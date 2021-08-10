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
trans_data_rnn
"""
from __future__ import absolute_import
import math
from impl import common_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl import constant_util as constant
from .. import trans_data_common_func as tdc
from impl.util.platform_adapter import tbe_context

# max num of tiling params
TILING_ARG_NUM = 96

def ceil_32bytes_align_count(count, dtype):
    type_size = common_util.get_data_size(dtype)
    block_count = math.ceil(count * type_size / constant.BLOCK_SIZE)
    return block_count * constant.BLOCK_SIZE // type_size

def _gm2ub(tik_instance: tik.Tik, dest: tik.Tensor, src: tik.Tensor, count):
    dtype_size = common_util.get_data_size(src.dtype)
    burst = math.ceil(count * dtype_size / constant.BLOCK_SIZE)
    tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)

def once_vnchwconv_invert_4_fp16(args):
    """
    do cn to c1nc0 transform by once vnchwconv
    """
    tik_inst, dst_gm, src_ub, ub_offset, out_level1_lp, out_level1_lp_step_ub, out_level1_lp_step_out, \
    out_level0_lp, out_level0_offset_ub, out_level0_offset_out, out_level0_repeat, \
    out_level0_burst, out_level0_src_stride, out_level0_dst_stride, row_size, data_size = args

    src_addr_list = [src_ub[row_size * i] for i in tdc.ADDR_IDX_LIST]
    dst_addr_list = [src_ub[ub_offset + tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
    repeat_cnt = tdc.ceil_div(data_size, tdc.C0_16)
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

    # move data to out
    with tik_inst.for_range(0, out_level1_lp) as level1_lp_idx:
        level1_offset_ub = level1_lp_idx * out_level1_lp_step_ub
        level1_offset_out = level1_lp_idx * out_level1_lp_step_out
        with tik_inst.for_range(0, out_level0_lp) as level0_lp_idx:
            level0_offset_ub = level1_offset_ub + level0_lp_idx * out_level0_offset_ub
            level0_offset_out = level1_offset_out + level0_lp_idx * out_level0_offset_out
            tik_inst.data_move(dst_gm[level0_offset_out], src_ub[ub_offset + level0_offset_ub],
                               0, out_level0_repeat, out_level0_burst, out_level0_src_stride, out_level0_dst_stride)

def twice_vnchwconv_no_invert(args):
    """
    do nc to c1nc0 transform by twice vnchwconv
    """
    tik_inst, dst_gm, src_ub, ub_offset, \
    out_level1_lp, out_level1_offset_out, \
    out_level0_lp, out_level0_offset_ub, out_level0_offset_out, out_level0_repeat, \
    out_level0_burst, out_level0_src_stride, out_level0_dst_stride, row_size, data_size, c_cnt, sub_c_size = args

    src_addr_list = [src_ub[row_size * i] for i in tdc.ADDR_IDX_LIST]
    dst_addr_list = [src_ub[ub_offset + tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
    repeat_cnt = tdc.ceil_div(data_size, tdc.C0_16)
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

    sub_c_size_block_align = tdc.ceil_fill(sub_c_size, tdc.C0_16)
    with tik_inst.if_scope(sub_c_size_block_align > sub_c_size):
        tdc.clean_ubuf(tik_inst, src_ub, 0, sub_c_size_block_align * c_cnt * tdc.VNC_LINES)
    tik_inst.data_move(src_ub, src_ub[ub_offset], 0, c_cnt, sub_c_size, 0, sub_c_size_block_align - sub_c_size)

    c0_cnt = out_level1_lp
    with tik_inst.for_range(0, c0_cnt) as c0_idx:
        src_addr_list = [src_ub[tdc.C0_16 * i + c0_idx * tdc.C0_16 * tdc.VNC_LINES] for i in tdc.ADDR_IDX_LIST]
        dst_addr_list = [src_ub[ub_offset + c_cnt * tdc.C0_16 * i] for i in tdc.ADDR_IDX_LIST]
        repeat_cnt = c_cnt
        with tik_inst.new_stmt_scope():
            src_stride = tik_inst.Scalar()
            dst_stride = tik_inst.Scalar()
            with tik_inst.if_scope(repeat_cnt == 1):
                src_stride.set_as(0)
                dst_stride.set_as(0)
            with tik_inst.else_scope():
                src_stride.set_as(sub_c_size_block_align)
                dst_stride.set_as(1)
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        with tik_inst.for_range(0, out_level0_lp) as n_idx:
            tik_inst.data_move(dst_gm[c0_idx * out_level1_offset_out + n_idx * out_level0_offset_out],
                               src_ub[ub_offset + n_idx * out_level0_offset_ub], 0, out_level0_repeat,
                               out_level0_burst, out_level0_src_stride, out_level0_dst_stride)

class TransData:
    def __init__(self, input, src_format, dst_format, input_size, hidden_size, groups, kernel_name):
        self.tik_instance = tik.Tik()
        self.tik_profiling = tik.Dprofile()

        self.dtype = input.get("dtype").lower()
        self.tiling_dtype = "int64"
        self.src_format = src_format
        self.dst_format = dst_format
        self.output_shape = (tdc.MAX_INT64_VALUE,)
        self.input_shape = (tdc.MAX_INT64_VALUE,)
        self.input_size = input_size
        self.hidden_size = hidden_size
        dtype_bytes = tdc.get_dtype_len(self.dtype)
        tiling_dtype_bytes = tdc.get_dtype_len(self.tiling_dtype)
        self.ub_size = tdc.get_max_element_in_ub(self.dtype, 1, 256) - TILING_ARG_NUM * tiling_dtype_bytes // dtype_bytes
        tiling_gm_size = TILING_ARG_NUM

        tiling_gm_size_align = ceil_32bytes_align_count(tiling_gm_size, self.tiling_dtype)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (tiling_gm_size_align,), name="tiling_gm", scope=tik.scope_gm)
        self.input_tensors = self.tik_instance.Tensor(self.dtype, self.input_shape, name="gm_input", scope=tik.scope_gm)
        self.output_tensor = self.tik_instance.Tensor(self.dtype, self.output_shape, name="gm_output", scope=tik.scope_gm)
        self.input_ub = self.tik_instance.Tensor(self.dtype, (self.ub_size,), tik.scope_ubuf, "total_ub")

        self.tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (tiling_gm_size_align,), name="tiling_ub", scope=tik.scope_ubuf)
        _gm2ub(self.tik_instance, self.tiling_ub, self.tiling_gm, tiling_gm_size)
        self.aicore_num = self.tik_profiling.get_aicore_num()
        self.groups =  groups
        self.kernel_name = kernel_name

    def transdata_compute(self):
        """
        build transdata op

        Returns
        -------
        None
        """
        tik_inst = self.tik_instance
        aicore_num = self.aicore_num

        in_level0_lp = tik_inst.Scalar(dtype="int32", init_value=1, name="in_level0_lp")
        in_level0_repeat = tik_inst.Scalar(dtype="int32", init_value=1, name="in_level0_repeat")
        in_level0_burst = tik_inst.Scalar(dtype="int32", init_value=1, name="in_level0_burst")
        in_level0_c_cnt = tik_inst.Scalar(dtype="int32", init_value=1, name="in_level0_c_cnt")
        in_level0_sub_c_size = tik_inst.Scalar(dtype="int32", init_value=1, name="in_level0_sub_c_size")
        in_level0_per_line_data = tik_inst.Scalar(dtype="int32", init_value=1, name="in_level0_per_line_data")
        out_level1_lp = tik_inst.Scalar()
        out_level0_lp = tik_inst.Scalar()
        out_level0_repeat = tik_inst.Scalar(dtype="int32", init_value=1, name="out_level0_repeat")
        out_level0_burst = tik_inst.Scalar(dtype="int32", init_value=1, name="out_level0_burst")

        is_last_c1_flag = tik_inst.Scalar()
        in_level0_backward = tik_inst.Scalar()
        in_level0_backward_burst = tik_inst.Scalar()
        out_level0_backward = tik_inst.Scalar()
        out_level0_backward_burst = tik_inst.Scalar()

        with tik_inst.for_range(0, aicore_num, block_num=aicore_num) as block_idx:
            shape_loop_cnt = tik_inst.Scalar(self.tiling_dtype, name="shape_loop_cnt")
            shape_loop_cnt.set_as(self.tiling_ub[0])
            params_cnt = tik_inst.Scalar(self.tiling_dtype, name="params_cnt")
            params_cnt.set_as(self.tiling_ub[1])

            with tik_inst.for_range(0, shape_loop_cnt) as shape_loop_idx:
                params_base = shape_loop_idx * params_cnt

                tiling_mode = tik_inst.Scalar(self.tiling_dtype, name="tiling_mode")
                tiling_mode.set_as(self.tiling_ub[2 + params_base])
                ub_offset = tik_inst.Scalar(self.tiling_dtype, name="ub_offset")
                ub_offset.set_as(self.tiling_ub[3 + params_base])
                core_cnt = tik_inst.Scalar(self.tiling_dtype, name="core_cnt")
                core_cnt.set_as(self.tiling_ub[4 + params_base])
                core_step_in = tik_inst.Scalar(self.tiling_dtype, name="core_step_in")
                core_step_in.set_as(self.tiling_ub[5 + params_base])
                core_step_out = tik_inst.Scalar(self.tiling_dtype, name="core_step_out")
                core_step_out.set_as(self.tiling_ub[6 + params_base])
                vnc_line_size = tik_inst.Scalar(self.tiling_dtype, name="vnc_line_size")
                vnc_line_size.set_as(self.tiling_ub[7 + params_base])

                def _common_func_tiling_mode_100(core_base):
                    """
                    the common function for tiling mode 100
                    """
                    shape_lp_in_offset = tik_inst.Scalar(self.tiling_dtype, name="shape_lp_in_offset")
                    shape_lp_in_offset.set_as(self.tiling_ub[68])
                    shape_lp_out_offset = tik_inst.Scalar(self.tiling_dtype, name="shape_lp_out_offset")
                    shape_lp_out_offset.set_as(self.tiling_ub[69])

                    in_level5_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level5_lp")
                    in_level5_lp.set_as(self.tiling_ub[34 + params_base + core_base])
                    with tik_inst.for_range(0, in_level5_lp) as level5_idx:
                        in_level5_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level5_step_in")
                        in_level5_step_in.set_as(self.tiling_ub[8 + params_base])
                        in_level5_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level5_step_out")
                        in_level5_step_out.set_as(self.tiling_ub[9 + params_base])
                        in_offset_5 = level5_idx * in_level5_step_in + block_idx * core_step_in + \
                                      shape_loop_idx * shape_lp_in_offset
                        out_offset_5 = level5_idx * in_level5_step_out + block_idx * core_step_out + \
                                       shape_loop_idx * shape_lp_out_offset

                        in_level4_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level4_lp")
                        in_level4_lp.set_as(self.tiling_ub[36 + params_base + core_base])
                        with tik_inst.for_range(0, in_level4_lp) as level4_idx:
                            in_level4_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level4_step_in")
                            in_level4_step_in.set_as(self.tiling_ub[10 + params_base])
                            in_level4_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level4_step_out")
                            in_level4_step_out.set_as(self.tiling_ub[11 + params_base])
                            in_offset_4 = in_offset_5 + level4_idx * in_level4_step_in
                            out_offset_4 = out_offset_5 + level4_idx * in_level4_step_out

                            in_level3_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level3_lp")
                            in_level3_lp.set_as(self.tiling_ub[38 + params_base + core_base])
                            with tik_inst.for_range(0, in_level3_lp) as level3_idx:
                                in_level3_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level3_step_in")
                                in_level3_step_in.set_as(self.tiling_ub[12 + params_base])
                                in_level3_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level3_step_out")
                                in_level3_step_out.set_as(self.tiling_ub[13 + params_base])
                                in_offset_3 = in_offset_4 + level3_idx * in_level3_step_in
                                out_offset_3 = out_offset_4 + level3_idx * in_level3_step_out

                                # control in-level 0 loop count and repeat of data_move
                                in_level2_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level2_lp")
                                in_level2_lp.set_as(self.tiling_ub[40 + params_base + core_base])
                                in_level2_left = tik_inst.Scalar(self.tiling_dtype, name="in_level2_left")
                                in_level2_left.set_as(self.tiling_ub[41 + params_base + core_base])
                                with tik_inst.for_range(0, in_level2_lp) as level2_idx:
                                    in_level2_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level2_step_in")
                                    in_level2_step_in.set_as(self.tiling_ub[14 + params_base])
                                    in_level2_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level2_step_out")
                                    in_level2_step_out.set_as(self.tiling_ub[15 + params_base])
                                    in_offset_2 = in_offset_3 + level2_idx * in_level2_step_in
                                    out_offset_2 = out_offset_3 + level2_idx * in_level2_step_out

                                    with tik_inst.if_scope(tik.any(level2_idx != in_level2_lp - 1, in_level2_left == 0)):
                                        in_level0_lp.set_as(self.tiling_ub[18 + params_base])
                                        in_level0_repeat.set_as(self.tiling_ub[21 + params_base])
                                    with tik_inst.else_scope():
                                        in_level0_lp.set_as(self.tiling_ub[44 + params_base + core_base])
                                        in_level0_repeat.set_as(self.tiling_ub[45 + params_base + core_base])
                                        # clean dirty data from above loops which will not be covered in last loop
                                        tdc.clean_ubuf(tik_inst, self.input_ub, in_level2_left * vnc_line_size,
                                                    (tdc.C0_16 - in_level2_left) * vnc_line_size)

                                    # control in-level 0 data_move burst and out-level 0 data_move repeat and burst
                                    in_level1_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level1_lp")
                                    in_level1_lp.set_as(self.tiling_ub[42 + params_base + core_base])
                                    in_level1_left = tik_inst.Scalar(self.tiling_dtype, name="in_level1_left")
                                    in_level1_left.set_as(self.tiling_ub[43 + params_base + core_base])

                                    with tik_inst.for_range(0, in_level1_lp) as level1_idx:
                                        in_level1_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level1_step_in")
                                        in_level1_step_in.set_as(self.tiling_ub[16 + params_base])
                                        in_level1_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level1_step_out")
                                        in_level1_step_out.set_as(self.tiling_ub[17 + params_base])
                                        in_offset_1 = in_offset_2 + level1_idx * in_level1_step_in
                                        out_offset_1 = out_offset_2 + level1_idx * in_level1_step_out

                                        with tik_inst.if_scope(tik.any(level1_idx != in_level1_lp - 1,
                                                                    in_level1_left == 0)):
                                            in_level0_burst.set_as(self.tiling_ub[22 + params_base])
                                            out_level0_lp.set_as(self.tiling_ub[27 + params_base])
                                            out_level0_repeat.set_as(self.tiling_ub[30 + params_base])
                                            out_level0_burst.set_as(self.tiling_ub[31 + params_base])
                                        with tik_inst.else_scope():
                                            in_level0_burst.set_as(self.tiling_ub[46 + params_base + core_base])
                                            out_level0_lp.set_as(self.tiling_ub[48 + params_base + core_base])
                                            out_level0_repeat.set_as(self.tiling_ub[49 + params_base + core_base])
                                            out_level0_burst.set_as(self.tiling_ub[50 + params_base + core_base])

                                        in_level0_offset_ub = tik_inst.Scalar(self.tiling_dtype, name="in_level0_offset_ub")
                                        in_level0_offset_ub.set_as(self.tiling_ub[19 + params_base])
                                        in_level0_offset_in = tik_inst.Scalar(self.tiling_dtype, name="in_level0_offset_in")
                                        in_level0_offset_in.set_as(self.tiling_ub[20 + params_base])
                                        in_level0_src_stride = tik_inst.Scalar(self.tiling_dtype, name="in_level0_src_stride")
                                        in_level0_src_stride.set_as(self.tiling_ub[23 + params_base])
                                        in_level0_dst_stride = tik_inst.Scalar(self.tiling_dtype, name="in_level0_dst_stride")
                                        in_level0_dst_stride.set_as(self.tiling_ub[24 + params_base])
                                        out_level0_offset_ub = tik_inst.Scalar(self.tiling_dtype, name="out_level0_offset_ub")
                                        out_level0_offset_ub.set_as(self.tiling_ub[28 + params_base])
                                        out_level0_offset_out = tik_inst.Scalar(self.tiling_dtype, name="out_level0_offset_out")
                                        out_level0_offset_out.set_as(self.tiling_ub[29 + params_base])
                                        out_level0_src_stride = tik_inst.Scalar(self.tiling_dtype, name="out_level0_src_stride")
                                        out_level0_src_stride.set_as(self.tiling_ub[32 + params_base])
                                        out_level0_dst_stride = tik_inst.Scalar(self.tiling_dtype, name="out_level0_dst_stride")
                                        out_level0_dst_stride.set_as(self.tiling_ub[33 + params_base])

                                        out_level1_lp_step_ub = tik_inst.Scalar(self.tiling_dtype, name="out_level1_lp_step_ub")
                                        out_level1_lp_step_ub.set_as(self.tiling_ub[25 + params_base])

                                        out_level1_lp_step_out = tik_inst.Scalar(self.tiling_dtype, name="out_level1_lp_step_out")
                                        out_level1_lp_step_out.set_as(self.tiling_ub[26 + params_base])
                                        out_level1_lp.set_as(self.tiling_ub[47 + params_base + core_base])

                                        with tik_inst.for_range(0, in_level0_lp) as level0_idx:
                                            in_offset_0 = in_offset_1 + level0_idx * in_level0_offset_in
                                            in_ub_offset = level0_idx * in_level0_offset_ub
                                            tik_inst.data_move(self.input_ub[in_ub_offset], self.input_tensors[in_offset_0],
                                                               0, in_level0_repeat, tdc.ceil_div(in_level0_burst, tdc.C0_16),
                                                               in_level0_src_stride, in_level0_dst_stride)

                                        args = (tik_inst, self.output_tensor[out_offset_1], self.input_ub, ub_offset,
                                                out_level1_lp, out_level1_lp_step_ub, out_level1_lp_step_out, out_level0_lp,
                                                out_level0_offset_ub, out_level0_offset_out, out_level0_repeat,
                                                tdc.ceil_div(out_level0_burst, tdc.C0_16), out_level0_src_stride,
                                                out_level0_dst_stride, vnc_line_size, in_level0_burst)
                                        once_vnchwconv_invert_4_fp16(args)

                def _common_func_tiling_mode_101(core_base):
                    """
                    the common function for tiling mode 101
                    """
                    in_level5_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level5_lp")
                    in_level5_lp.set_as(self.tiling_ub[37 + params_base + core_base])
                    with tik_inst.for_range(0, in_level5_lp) as level5_idx:
                        in_level5_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level5_step_in")
                        in_level5_step_in.set_as(self.tiling_ub[8 + params_base])
                        in_level5_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level5_step_out")
                        in_level5_step_out.set_as(self.tiling_ub[9 + params_base])
                        in_offset_5 = level5_idx * in_level5_step_in + block_idx * core_step_in
                        out_offset_5 = level5_idx * in_level5_step_out + block_idx * core_step_out

                        in_level4_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level4_lp")
                        in_level4_lp.set_as(self.tiling_ub[39 + params_base + core_base])
                        with tik_inst.for_range(0, in_level4_lp) as level4_idx:
                            in_level4_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level4_step_in")
                            in_level4_step_in.set_as(self.tiling_ub[10 + params_base])
                            in_level4_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level4_step_out")
                            in_level4_step_out.set_as(self.tiling_ub[11 + params_base])
                            in_offset_4 = in_offset_5 + level4_idx * in_level4_step_in
                            out_offset_4 = out_offset_5 + level4_idx * in_level4_step_out

                            in_level3_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level3_lp")
                            in_level3_lp.set_as(self.tiling_ub[41 + params_base + core_base])
                            with tik_inst.for_range(0, in_level3_lp) as level3_idx:
                                in_level3_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level3_step_in")
                                in_level3_step_in.set_as(self.tiling_ub[12 + params_base])
                                in_level3_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level3_step_out")
                                in_level3_step_out.set_as(self.tiling_ub[13 + params_base])
                                in_offset_3 = in_offset_4 + level3_idx * in_level3_step_in
                                out_offset_3 = out_offset_4 + level3_idx * in_level3_step_out

                                # control in-level 0 loop count and repeat of data_move
                                in_level2_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level2_lp")
                                in_level2_lp.set_as(self.tiling_ub[43 + params_base + core_base])
                                in_level2_left = tik_inst.Scalar(self.tiling_dtype, name="in_level2_left")
                                in_level2_left.set_as(self.tiling_ub[44 + params_base + core_base])
                                with tik_inst.for_range(0, in_level2_lp) as level2_idx:
                                    in_level2_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level2_step_in")
                                    in_level2_step_in.set_as(self.tiling_ub[14 + params_base])
                                    in_level2_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level2_step_out")
                                    in_level2_step_out.set_as(self.tiling_ub[15 + params_base])
                                    in_offset_2 = in_offset_3 + level2_idx * in_level2_step_in
                                    out_offset_2 = out_offset_3 + level2_idx * in_level2_step_out

                                    with tik_inst.if_scope(tik.any(level2_idx != in_level2_lp - 1, in_level2_left == 0)):
                                        in_level0_lp.set_as(self.tiling_ub[18 + params_base])
                                        in_level0_repeat.set_as(self.tiling_ub[24 + params_base])
                                        out_level0_burst.set_as(self.tiling_ub[34 + params_base])
                                    with tik_inst.else_scope():
                                        in_level0_lp.set_as(self.tiling_ub[47 + params_base + core_base])
                                        in_level0_repeat.set_as(self.tiling_ub[51 + params_base + core_base])
                                        out_level0_burst.set_as(self.tiling_ub[57 + params_base + core_base])
                                    
                                    # control in-level 0 data_move burst and out-level 0 data_move repeat and burst
                                    in_level1_lp = tik_inst.Scalar(self.tiling_dtype, name="in_level1_lp")
                                    in_level1_lp.set_as(self.tiling_ub[45 + params_base + core_base])
                                    in_level1_left = tik_inst.Scalar(self.tiling_dtype, name="in_level1_left")
                                    in_level1_left.set_as(self.tiling_ub[46 + params_base + core_base])
                                    with tik_inst.for_range(0, in_level1_lp) as level1_idx:
                                        in_level1_step_in = tik_inst.Scalar(self.tiling_dtype, name="in_level1_step_in")
                                        in_level1_step_in.set_as(self.tiling_ub[16 + params_base])
                                        in_level1_step_out = tik_inst.Scalar(self.tiling_dtype, name="in_level1_step_out")
                                        in_level1_step_out.set_as(self.tiling_ub[17 + params_base])
                                        in_offset_1 = in_offset_2 + level1_idx * in_level1_step_in
                                        out_offset_1 = out_offset_2 + level1_idx * in_level1_step_out

                                        with tik_inst.if_scope(tik.any(level1_idx != in_level1_lp - 1,
                                                                    in_level1_left == 0)):
                                            in_level0_burst.set_as(self.tiling_ub[25 + params_base])
                                            in_level0_c_cnt.set_as(self.tiling_ub[21 + params_base])
                                            in_level0_sub_c_size.set_as(self.tiling_ub[22 + params_base])
                                            in_level0_per_line_data.set_as(self.tiling_ub[23 + params_base])
                                            out_level1_lp.set_as(self.tiling_ub[28 + params_base])
                                            out_level0_lp.set_as(self.tiling_ub[30 + params_base])
                                            out_level0_repeat.set_as(self.tiling_ub[33 + params_base])
                                        with tik_inst.else_scope():
                                            in_level0_burst.set_as(self.tiling_ub[52 + params_base + core_base])
                                            in_level0_c_cnt.set_as(self.tiling_ub[48 + params_base])
                                            in_level0_sub_c_size.set_as(self.tiling_ub[49 + params_base])
                                            in_level0_per_line_data.set_as(self.tiling_ub[50 + params_base])
                                            out_level1_lp.set_as(self.tiling_ub[54 + params_base + core_base])
                                            out_level0_lp.set_as(self.tiling_ub[55 + params_base + core_base])
                                            out_level0_repeat.set_as(self.tiling_ub[56 + params_base + core_base])

                                        in_level0_offset_ub = tik_inst.Scalar(self.tiling_dtype, name="in_level0_offset_ub")
                                        in_level0_offset_ub.set_as(self.tiling_ub[19 + params_base])
                                        in_level0_offset_in = tik_inst.Scalar(self.tiling_dtype, name="in_level0_offset_in")
                                        in_level0_offset_in.set_as(self.tiling_ub[20 + params_base])
                                        in_level0_src_stride = tik_inst.Scalar(self.tiling_dtype, name="in_level0_src_stride")
                                        in_level0_src_stride.set_as(self.tiling_ub[26 + params_base])
                                        in_level0_dst_stride = tik_inst.Scalar(self.tiling_dtype, name="in_level0_dst_stride")
                                        in_level0_dst_stride.set_as(self.tiling_ub[27 + params_base])

                                        out_level1_offset_out = tik_inst.Scalar(self.tiling_dtype, name="out_level1_offset_out")
                                        out_level1_offset_out.set_as(self.tiling_ub[29 + params_base])
                                        out_level0_offset_ub = tik_inst.Scalar(self.tiling_dtype, name="out_level0_offset_ub")
                                        out_level0_offset_ub.set_as(self.tiling_ub[31 + params_base])
                                        out_level0_offset_out = tik_inst.Scalar(self.tiling_dtype, name="out_level0_offset_out")
                                        out_level0_offset_out.set_as(self.tiling_ub[32 + params_base])
                                        out_level0_src_stride = tik_inst.Scalar(self.tiling_dtype, name="out_level0_src_stride")
                                        out_level0_src_stride.set_as(self.tiling_ub[35 + params_base])
                                        out_level0_dst_stride = tik_inst.Scalar(self.tiling_dtype, name="out_level0_dst_stride")
                                        out_level0_dst_stride.set_as(self.tiling_ub[36 + params_base])

                                        in_level0_burst_t = tik_inst.Scalar(self.tiling_dtype, name="in_level0_burst_t")
                                        in_level0_burst_t.set_as(self.tiling_ub[53 + params_base + core_base])

                                        with tik_inst.for_range(0, in_level0_lp) as level0_idx:
                                            in_offset_0 = in_offset_1 + level0_idx * in_level0_offset_in
                                            in_ub_offset = level0_idx * in_level0_offset_ub

                                            with tik_inst.if_scope(tik.all(level2_idx == in_level2_lp - 1,
                                                                        level0_idx == in_level0_lp - 1,
                                                                        in_level0_burst_t > 0,
                                                                        level1_idx == in_level1_lp - 1)):
                                                in_level0_burst.set_as(in_level0_burst_t)
                                            tik_inst.data_move(self.input_ub[in_ub_offset], self.input_tensors[in_offset_0],
                                                               0, in_level0_repeat, tdc.ceil_div(in_level0_burst, tdc.C0_16),
                                                               in_level0_src_stride, in_level0_dst_stride)

                                        args = (tik_inst, self.output_tensor[out_offset_1], self.input_ub, ub_offset,
                                                out_level1_lp, out_level1_offset_out,
                                                out_level0_lp, out_level0_offset_ub,
                                                out_level0_offset_out, out_level0_repeat,
                                                tdc.ceil_div(out_level0_burst, tdc.C0_16), out_level0_src_stride,
                                                out_level0_dst_stride, vnc_line_size, in_level0_per_line_data,
                                                in_level0_c_cnt, in_level0_sub_c_size)
                                        twice_vnchwconv_no_invert(args)

                with tik_inst.if_scope(block_idx < core_cnt):
                    with tik_inst.if_scope(tiling_mode == 100):
                        with tik_inst.if_scope(block_idx != core_cnt - 1):
                            _common_func_tiling_mode_100(0)
                        with tik_inst.else_scope():
                            _common_func_tiling_mode_100(17)
                    
                    with tik_inst.else_scope():
                        with tik_inst.if_scope(tiling_mode == 101):
                            with tik_inst.if_scope(block_idx != core_cnt - 1):
                                _common_func_tiling_mode_101(0)
                            with tik_inst.else_scope():
                                _common_func_tiling_mode_101(21)
                        with tik_inst.else_scope():
                            pass

        # build cce
        tik_inst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_tensors], outputs=[self.output_tensor],
                          flowtable=[self.tiling_gm], enable_l2=False)      
        tbe_context.get_context().add_compile_info("vars", {"srcFormat": self.src_format,
                                        "dstFormat": self.dst_format,
                                        "dType": self.dtype,
                                        "ubSize": self.ub_size,
                                        "blockDim": self.aicore_num,
                                        "inputSize": self.input_size,
                                        "hiddenSize": self.hidden_size,
                                        "group": self.groups
                                        })
        return tik_inst

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR,
                            para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def trans_data_rnn(src, dst, src_format, dst_format,
                   input_size=0, hidden_size=0, kernel_name="trans_data_rnn", groups=1):
    """
    format transform for rnn
    """
    src_format = src_format.upper()
    dst_format = dst_format.upper()

    transdata_instance = TransData(src, src_format, dst_format, input_size, hidden_size, groups, kernel_name)
    return transdata_instance.transdata_compute()
