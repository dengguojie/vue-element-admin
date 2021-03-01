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
dynamic max pooling
"""
import te.lang.dynamic
from impl.util.platform_adapter import tik
from te import platform as tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

# max int32
MAX_INT32 = 2**31 - 1
# tiling param num
TILING_ARG_NUM = 24
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# 8 bit
EIGHT_BIT = 8
# bytes of one block
BLOCK_BYTES = 32
# c0 equal to 16 and one block can save 16 ele, for fp16
C_ZERO = 16
# repeat limit
REPEAT_LIMIT = 255
# mask length
MASK = 128
# min fp16
MIN_FP16 = -65400.0


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-statements
# pylint: disable=too-many-locals,unused-argument,attribute-defined-outside-init
class MaxPool:
    """Performs max pooling on input tensor
    """

    def __init__(self, dtype, ksize, strides, padding, kernel_name):
        """Init max pooling parameters
        """
        self.dtype = dtype
        self.ksize_h = ksize[0]
        self.ksize_w = ksize[1]
        self.strides_h = strides[0]
        self.strides_w = strides[1]
        # SAME:0, VALID:1
        self.padding = 0 if padding == "SAME" else 1
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // EIGHT_BIT
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE) // self.dtype_size
        # reduce need two buffer and open double buffer
        self.one_fourth_ub_ele = self.ub_ele // 4
        self.init_gm_tensor()

    def tiling_args(self):
        """Get runtime params from tiling
        """
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.act_core_num = self.tik_instance.Scalar("int32", name="act_core_num")
        self.act_core_num.set_as(self.tiling_ub[1])
        self.one_core_ele = self.tik_instance.Scalar("int32", name="one_core_ele")
        self.one_core_ele.set_as(self.tiling_ub[2])
        self.last_core_ele = self.tik_instance.Scalar("int32", name="last_core_ele")
        self.last_core_ele.set_as(self.tiling_ub[3])
        self.input_h = self.tik_instance.Scalar("int32", name="input_h")
        self.input_h.set_as(self.tiling_ub[4])
        self.input_w = self.tik_instance.Scalar("int32", name="input_w")
        self.input_w.set_as(self.tiling_ub[5])
        self.output_h = self.tik_instance.Scalar("int32", name="output_h")
        self.output_h.set_as(self.tiling_ub[6])
        self.output_w = self.tik_instance.Scalar("int32", name="output_w")
        self.output_w.set_as(self.tiling_ub[7])
        self.pad_h = self.tik_instance.Scalar("int32", name="pad_h")
        self.pad_h.set_as(self.tiling_ub[8])
        self.pad_w = self.tik_instance.Scalar("int32", name="pad_w")
        self.pad_w.set_as(self.tiling_ub[9])
        self.pad_t = self.tik_instance.Scalar("int32", name="pad_t")
        self.pad_t.set_as(self.tiling_ub[10])
        self.pad_b = self.tik_instance.Scalar("int32", name="pad_b")
        self.pad_b.set_as(self.tiling_ub[11])
        self.pad_l = self.tik_instance.Scalar("int32", name="pad_l")
        self.pad_l.set_as(self.tiling_ub[12])
        self.pad_r = self.tik_instance.Scalar("int32", name="pad_r")
        self.pad_r.set_as(self.tiling_ub[13])
        self.c_factor = self.tik_instance.Scalar("int32", name="c_factor")
        self.c_factor.set_as(self.tiling_ub[14])
        self.h_factor = self.tik_instance.Scalar("int32", name="h_factor")
        self.h_factor.set_as(self.tiling_ub[15])
        self.w_factor = self.tik_instance.Scalar("int32", name="w_factor")
        self.w_factor.set_as(self.tiling_ub[16])
        self.one_core_loop_num = self.tik_instance.Scalar("int32", name="one_core_loop_num")
        self.one_core_loop_num.set_as(self.tiling_ub[17])
        self.one_core_loop_left = self.tik_instance.Scalar("int32", name="one_core_loop_left")
        self.one_core_loop_left.set_as(self.tiling_ub[18])
        self.last_core_loop_num = self.tik_instance.Scalar("int32", name="last_core_loop_num")
        self.last_core_loop_num.set_as(self.tiling_ub[19])
        self.last_core_loop_left = self.tik_instance.Scalar("int32", name="last_core_loop_left")
        self.last_core_loop_left.set_as(self.tiling_ub[20])

    def init_gm_tensor(self):
        """Init gm tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="input_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="output_gm", scope=tik.scope_gm)

    def init_ub_tensor(self):
        """Init ub tensor
        """
        self.ub_a = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_a", scope=tik.scope_ubuf)
        self.ub_b = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_b", scope=tik.scope_ubuf)
        self.ub_c = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_c", scope=tik.scope_ubuf)
        self.ub_d = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_d", scope=tik.scope_ubuf)

    def init_ub_scalar_single(self):
        """Init ub scalar
        """
        # common params
        self.size = self.tik_instance.Scalar("int32", name="size")
        self.offset = self.tik_instance.Scalar("int32", name="offset")
        # first core and last core use the same scalar
        self.core_ele = self.tik_instance.Scalar("int32", name="core_ele")
        self.loop_num = self.tik_instance.Scalar("int32", name="loop_num")
        self.loop_left = self.tik_instance.Scalar("int32", name="loop_left")
        # before and after of h and w, use for h and w tiling
        self.before_h = self.tik_instance.Scalar("int32", name="before_h")
        self.after_h = self.tik_instance.Scalar("int32", name="after_h")
        self.before_w = self.tik_instance.Scalar("int32", name="before_w")
        self.after_w = self.tik_instance.Scalar("int32", name="after_w")
        self.len_h = self.tik_instance.Scalar("int32", name="len_h")
        self.len_w = self.tik_instance.Scalar("int32", name="len_w")
        # only for data_move
        self.offset_ub = self.tik_instance.Scalar("int32", name="offset_ub")
        self.nburst = self.tik_instance.Scalar("int32", name="nburst")
        self.src_stride = self.tik_instance.Scalar("int32", name="src_stride")
        self.dst_stride = self.tik_instance.Scalar("int32", name="dst_stride")
        self.burst_len_in = self.tik_instance.Scalar("int32", name="burst_len_in")
        self.burst_len_out = self.tik_instance.Scalar("int32", name="burst_len_out")
        # only for vector dup and move in
        self.size_1 = self.tik_instance.Scalar("int32", name="size_1")
        self.offset_2 = self.tik_instance.Scalar("int32", name="offset_2")
        self.size_2 = self.tik_instance.Scalar("int32", name="size_2")
        self.offset_3 = self.tik_instance.Scalar("int32", name="offset_3")
        self.repeat_3 = self.tik_instance.Scalar("int32", name="repeat_3")
        self.size_3 = self.tik_instance.Scalar("int32", name="size_3")
        # only for reduce max
        self.size_w = self.tik_instance.Scalar("int32", name="size_w")
        self.repeat_w = self.tik_instance.Scalar("int32", name="repeat_w")
        self.rep_blk_h = self.tik_instance.Scalar("int32", name="rep_blk_h")

    def init_ub_scalar_double(self):
        """Init ub scalar
        """
        # vetcor_dup and reduce_max use the same scalar
        self.size_loop = self.tik_instance.Scalar("int32", name="size_loop")
        self.size_left = self.tik_instance.Scalar("int32", name="size_left")
        self.repeat_loop = self.tik_instance.Scalar("int32", name="repeat_loop")
        self.repeat_left = self.tik_instance.Scalar("int32", name="repeat_left")
        self.size_offset = self.tik_instance.Scalar("int32", name="size_offset")
        self.repeat_offset = self.tik_instance.Scalar("int32", name="repeat_offset")
        # only for data_move
        self.offset_in = self.tik_instance.Scalar("int32", name="offset_in")
        self.offset_out = self.tik_instance.Scalar("int32", name="offset_out")
        # only for reduce max
        self.repeat_offset_src0 = self.tik_instance.Scalar("int32", name="repeat_offset_src0")
        self.repeat_offset_src1 = self.tik_instance.Scalar("int32", name="repeat_offset_src1")
        self.size_offset_src0 = self.tik_instance.Scalar("int32", name="size_offset_src0")
        self.size_offset_src1 = self.tik_instance.Scalar("int32", name="size_offset_src1")
        self.offset_dst = self.tik_instance.Scalar("int32", name="offset_dst")
        self.offset_src0 = self.tik_instance.Scalar("int32", name="offset_src0")
        self.offset_src1 = self.tik_instance.Scalar("int32", name="offset_src1")
        self.offset_h = self.tik_instance.Scalar("int32", name="offset_h")
        self.offset_w = self.tik_instance.Scalar("int32", name="offset_w")

    def select_tiling_mode(self, core_idx, core_ele, loop_num, loop_left):
        """Select tiling mode
        """
        # select tiling mode
        with self.tik_instance.if_scope(self.tiling_mode == 0):
            self.copy_only(core_idx, loop_num, loop_left)
        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.tiling_c_dim_core_nc(core_idx, core_ele)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.tiling_h_dim_core_nc(core_idx, core_ele, loop_num, loop_left)
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.tiling_w_dim_core_nc(core_idx, core_ele, loop_num, loop_left)

    def copy_only(self, core_idx, loop_num, loop_left):
        """Only execute move in and move out
        """
        # max ele of 'n*c1*h*w' can move to ub
        self.max_ele = self.ub_ele // C_ZERO
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                self.copy_only_process(core_idx, loop_idx, self.max_ele)
            with self.tik_instance.if_scope(loop_left > 0):
                self.copy_only_process(core_idx, loop_num, loop_left)

    def copy_only_process(self, core_idx, loop_idx, ele):
        """Only execute move in and move out
        """
        self.offset.set_as((core_idx * self.one_core_ele + loop_idx * self.max_ele) * C_ZERO)
        ub_tensor = self.tik_instance.Tensor(self.dtype, (self.max_ele * C_ZERO,),
                                             name="ub_tensor",
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(ub_tensor, self.input_gm[self.offset], 0, 1, ele, 0, 0)
        self.tik_instance.data_move(self.output_gm[self.offset], ub_tensor, 0, 1, ele, 0, 0)

    def vector_dup_continuous(self, src, size):
        """vector_dup continuous function, set ubuf to -65400, only support fp16
        """
        with self.tik_instance.if_scope(size > 0):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(self.size_loop // REPEAT_LIMIT)
            self.repeat_left.set_as(self.size_loop % REPEAT_LIMIT)

            with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * MASK)
                self.tik_instance.vector_dup(MASK, src[self.repeat_offset], MIN_FP16, REPEAT_LIMIT, 1, 8)
            with self.tik_instance.if_scope(self.repeat_left > 0):
                self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * MASK)
                self.tik_instance.vector_dup(MASK, src[self.repeat_offset], MIN_FP16, self.repeat_left, 1, 8)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                self.tik_instance.vector_dup(self.size_left, src[self.size_offset], MIN_FP16, 1, 1, 8)

    def vector_dup_discrete(self, src, repeat, size, dst_blk=1, dst_rep=8):
        """vector_dup discrete function, set ubuf to -65400, only support fp16, dst_rep is pad_w(<=255)
        """
        with self.tik_instance.if_scope(tik.all(size > 0, repeat > 0)):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(repeat // REPEAT_LIMIT)
            self.repeat_left.set_as(repeat % REPEAT_LIMIT)

            def _inner(src, mask_len):
                """exec repeat
                """
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.tik_instance.vector_dup(mask_len, src[self.repeat_offset], MIN_FP16, REPEAT_LIMIT, dst_blk,
                                                 dst_rep)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.tik_instance.vector_dup(mask_len, src[self.repeat_offset], MIN_FP16, self.repeat_left, dst_blk,
                                                 dst_rep)

            # exec size
            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                self.size_offset.set_as(size_loop_idx * MASK)
                _inner(src[self.size_offset:], MASK)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                _inner(src[self.size_offset:], self.size_left)

    def reduce_max_rw(self, dst, src0, src1, size, src0_blk=1, src1_blk=1):
        """reduce max for width and height, repeat from size, size is output_w * c_zero, only support fp16.
        src0_blk/src1_blk is strides_w(<=31)
        """
        if self.strides_w <= 31:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // MASK)
                self.size_left.set_as(size % MASK)
                self.repeat_loop.set_as(self.size_loop // REPEAT_LIMIT)
                self.repeat_left.set_as(self.size_loop % REPEAT_LIMIT)

                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    # dst_rep must be equal to 8, continuous emissions
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * MASK)
                    # src_rep may not equal to 8, discrete emissions, but equal to src_blk * 8
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vmax(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * MASK)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vmax(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.size_left > 0):
                    self.size_offset.set_as(self.size_loop * MASK)
                    self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                    self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                    self.tik_instance.vmax(self.size_left, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, src0_blk, src1_blk, 8, src0_blk * 8,
                                           src1_blk * 8)
        else:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // C_ZERO)
                with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                    self.size_offset.set_as(size_loop_idx * C_ZERO)
                    self.size_offset_src0.set_as(size_loop_idx * src0_blk * C_ZERO)
                    self.size_offset_src1.set_as(size_loop_idx * src1_blk * C_ZERO)
                    self.tik_instance.vmax(C_ZERO, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, 1, 1, 8, 8, 8)

    def reduce_max_rh(self, dst, src0, src1, repeat, size, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8, src1_rep=8):
        """reduce max for width and height, repeat is pad_h and output_h, size is output_w * c_zero, only support fp16.
        src0_blk/src1_blk is strides_w(<=255), dst_rep is output_w(<=255),
        src0_rep/src1_rep is pad_w or ouput_w or output_w * strides_h(<=255)
        """
        with self.tik_instance.if_scope(tik.all(size > 0, repeat > 0)):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(repeat // REPEAT_LIMIT)
            self.repeat_left.set_as(repeat % REPEAT_LIMIT)

            def _inner(dst, src0, src1, mask_len):
                """exec repeat
                """
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    # dst_rep may not equal to 8, discrete emissions
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO)
                    # src_rep may not equal to 8, discrete emissions
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vmax(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, dst_rep,
                                           src0_rep, src1_rep)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vmax(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk,
                                           dst_rep, src0_rep, src1_rep)

            # exec size
            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                # dst_blk must be equal to 1, continuous emissions
                self.size_offset.set_as(size_loop_idx * MASK)
                # src_blk may not equal to 1, discrete emissions
                self.size_offset_src0.set_as(size_loop_idx * src0_blk * MASK)
                self.size_offset_src1.set_as(size_loop_idx * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:], MASK)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:],
                       self.size_left)

    def reduce_max(self, ub_x, ub_y, repeat_p, repeat_o):
        """Reduce max
        """
        # reduce max for w
        # if call 'reduce_max_rh', blk and rep must less than or equal to 255
        # if call 'reduce_max_rw', blk must less than or equal to 31, because 'rep = blk * 8'
        if self.ksize_w == 1:
            if self.strides_w == self.ksize_w:
                self.reduce_max_rw(ub_y, ub_x, ub_x, repeat_p * self.size_w, self.strides_w, self.strides_w)
            elif self.strides_w <= 255:
                with self.tik_instance.if_scope(
                        tik.all(repeat_p >= self.repeat_w, self.output_w <= 255, self.pad_w <= 255)):
                    self.reduce_max_rh(ub_y, ub_x, ub_x, repeat_p, self.size_w, self.strides_w, self.strides_w,
                                       self.output_w, self.pad_w, self.pad_w)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_p) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                        self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src0:],
                                           self.size_w, self.strides_w, self.strides_w)
            else:
                with self.tik_instance.for_range(0, repeat_p) as h_idx:
                    self.offset_dst.set_as(h_idx * self.size_w)
                    self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                    self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src0:],
                                       self.size_w, self.strides_w, self.strides_w)
        else:
            if self.strides_w == self.ksize_w:
                self.reduce_max_rw(ub_y, ub_x, ub_x[C_ZERO:], repeat_p * self.size_w, self.strides_w, self.strides_w)
                with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                    self.offset_w.set_as((idx + 2) * C_ZERO)
                    self.reduce_max_rw(ub_y, ub_x[self.offset_w:], ub_y, repeat_p * self.size_w, self.strides_w, 1)
            elif self.strides_w <= 255:
                with self.tik_instance.if_scope(
                        tik.all(repeat_p >= self.repeat_w, self.output_w <= 255, self.pad_w <= 255)):
                    self.reduce_max_rh(ub_y, ub_x, ub_x[C_ZERO:], repeat_p, self.size_w, self.strides_w, self.strides_w,
                                       self.output_w, self.pad_w, self.pad_w)
                    with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                        self.offset_w.set_as((idx + 2) * C_ZERO)
                        self.reduce_max_rh(ub_y, ub_x[self.offset_w:], ub_y, repeat_p, self.size_w, self.strides_w, 1,
                                           self.output_w, self.pad_w, self.output_w)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_p) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                        self.offset_src1.set_as(self.offset_src0 + C_ZERO)
                        self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src1:],
                                           self.size_w, self.strides_w, self.strides_w)
                        with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                            self.offset_w.set_as(self.offset_src0 + (idx + 2) * C_ZERO)
                            self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_w:], ub_y[self.offset_dst:],
                                               self.size_w, self.strides_w, 1)
            else:
                with self.tik_instance.for_range(0, repeat_p) as h_idx:
                    self.offset_dst.set_as(h_idx * self.size_w)
                    self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                    self.offset_src1.set_as(self.offset_src0 + C_ZERO)
                    self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src1:],
                                       self.size_w, self.strides_w, self.strides_w)
                    with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                        self.offset_w.set_as(self.offset_src0 + (idx + 2) * C_ZERO)
                        self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_w:], ub_y[self.offset_dst:],
                                           self.size_w, self.strides_w, 1)
        # reduce max for h
        # if call 'reduce_max_rh', blk and rep must less than or equal to 255
        # if call 'reduce_max_rw', blk must less than or equal to 31, because 'rep = blk * 8'
        if self.ksize_h == 1:
            if self.strides_h == 1:
                self.reduce_max_rw(ub_x, ub_y, ub_y, repeat_o * self.size_w, 1, 1)
            else:
                with self.tik_instance.if_scope(
                        tik.all(repeat_o >= self.repeat_w, self.output_w <= 255, self.rep_blk_h <= 255)):
                    self.reduce_max_rh(ub_x, ub_y, ub_y, repeat_o, self.size_w, 1, 1, self.output_w, self.rep_blk_h,
                                       self.rep_blk_h)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_o) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.strides_h * self.size_w)
                        self.reduce_max_rw(ub_x[self.offset_dst:], ub_y[self.offset_src0:], ub_y[self.offset_src0:],
                                           self.size_w, 1, 1)
        else:
            if self.strides_h == 1:
                self.reduce_max_rw(ub_x, ub_y, ub_y[self.size_w], repeat_o * self.size_w, 1, 1)
                with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                    self.offset_h.set_as((idx + 2) * self.size_w)
                    self.reduce_max_rw(ub_x, ub_y[self.offset_h:], ub_x, repeat_o * self.size_w, 1, 1)
            else:
                with self.tik_instance.if_scope(
                        tik.all(repeat_o >= self.repeat_w, self.output_w <= 255, self.rep_blk_h <= 255)):
                    self.reduce_max_rh(ub_x, ub_y, ub_y[self.size_w:], repeat_o, self.size_w, 1, 1, self.output_w,
                                       self.rep_blk_h, self.rep_blk_h)
                    with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                        self.offset_h.set_as((idx + 2) * self.size_w)
                        self.reduce_max_rh(ub_x, ub_y[self.offset_h:], ub_x, repeat_o, self.size_w, 1, 1, self.output_w,
                                           self.rep_blk_h, self.output_w)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_o) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.strides_h * self.size_w)
                        self.offset_src1.set_as(self.offset_src0 + self.size_w)
                        self.reduce_max_rw(ub_x[self.offset_dst:], ub_y[self.offset_src0:], ub_y[self.offset_src1:],
                                           self.size_w, 1, 1)
                        with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                            self.offset_h.set_as(self.offset_src0 + (idx + 2) * self.size_w)
                            self.reduce_max_rw(ub_x[self.offset_dst:], ub_y[self.offset_h:], ub_x[self.offset_dst:],
                                               self.size_w, 1, 1)

    def tiling_c_dim_core_nc(self, core_idx, core_ele):
        """Tiling c1 dim
        """
        # common params
        self.size_1.set_as(self.pad_t * self.pad_w * C_ZERO + self.pad_l * C_ZERO)
        self.offset_2.set_as((self.pad_h - self.pad_b) * self.pad_w * C_ZERO - self.pad_r * C_ZERO)
        self.size_2.set_as(self.pad_b * self.pad_w * C_ZERO + self.pad_r * C_ZERO)
        self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
        self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
        self.size_w.set_as(self.output_w * C_ZERO)
        self.repeat_w.set_as(self.size_w // MASK)
        self.rep_blk_h.set_as(self.output_w * self.strides_h)
        self.burst_len_out.set_as(self.output_h * self.output_w)
        # vector_dup_discrete -> vector_dup_continuous
        self.size.set_as(self.pad_h * self.pad_w * C_ZERO)
        # other params from h dim
        with self.tik_instance.if_scope(self.pad_h <= self.input_h):
            self.nburst.set_as(self.pad_h)
        with self.tik_instance.else_scope():
            self.nburst.set_as(self.input_h)
        # other params from w dim
        with self.tik_instance.if_scope(self.pad_w <= self.input_w):
            self.repeat_3.set_as(0)
            self.burst_len_in.set_as(self.pad_w)
            self.src_stride.set_as(self.input_w - self.pad_w)
            self.dst_stride.set_as(0)
        with self.tik_instance.else_scope():
            self.repeat_3.set_as(self.pad_h - 1)
            self.burst_len_in.set_as(self.input_w)
            self.src_stride.set_as(0)
            self.dst_stride.set_as(self.pad_r + self.pad_l)

        # ub operate
        with self.tik_instance.new_stmt_scope():

            def _inner(ele_idx, ub_x, ub_y):
                self.offset_in.set_as((core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * C_ZERO)
                self.offset_out.set_as(
                    (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * C_ZERO)
                # vector dup and move in
                with self.tik_instance.if_scope(tik.all(self.pad_w > 255, self.size_3 > 0)):
                    self.vector_dup_continuous(ub_x, self.size)
                with self.tik_instance.else_scope():
                    self.vector_dup_continuous(ub_x, self.size_1)
                    self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                    self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)
                self.tik_instance.data_move(ub_x[self.size_1], self.input_gm[self.offset_in], 0, self.nburst,
                                            self.burst_len_in, self.src_stride, self.dst_stride)
                # reduce max
                self.reduce_max(ub_x, ub_y, self.pad_h, self.output_h)
                # move out
                self.tik_instance.data_move(self.output_gm[self.offset_out], ub_x, 0, 1, self.burst_len_out, 0, 0)

            # open double buffer
            self.init_ub_tensor()
            self.init_ub_scalar_double()
            with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                _inner(ele_idx * 2, self.ub_a, self.ub_b)
                _inner(ele_idx * 2 + 1, self.ub_c, self.ub_d)
            with self.tik_instance.if_scope(core_ele % 2 == 1):
                _inner(core_ele - 1, self.ub_a, self.ub_b)

    def tiling_h_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        """Tiling h dim
        """
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_idx, self.h_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_h_dim_core_nc_process(self, core_idx, core_ele, loop_idx, ele):
        """Tiling h dim process
        """
        # common params
        self.before_h.set_as(loop_idx * self.h_factor * self.strides_h)
        self.after_h.set_as((loop_idx * self.h_factor + ele - 1) * self.strides_h + self.ksize_h)
        self.len_h.set_as(self.after_h - self.before_h)
        self.size_w.set_as(self.output_w * C_ZERO)
        self.repeat_w.set_as(self.size_w // MASK)
        self.rep_blk_h.set_as(self.output_w * self.strides_h)
        self.burst_len_out.set_as(ele * self.output_w)
        # vector_dup_discrete -> vector_dup_continuous
        self.size.set_as(self.len_h * self.pad_w * C_ZERO)
        # other params form h dim
        with self.tik_instance.if_scope(self.before_h < self.pad_t):
            self.size_1.set_as((self.pad_t - self.before_h) * self.pad_w * C_ZERO + self.pad_l * C_ZERO)
            self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
            self.offset.set_as(0)
            with self.tik_instance.if_scope(self.after_h <= self.pad_h - self.pad_b):
                self.offset_2.set_as(self.len_h * self.pad_w * C_ZERO - self.pad_r * C_ZERO)
                self.size_2.set_as(self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.after_h - self.pad_t - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.after_h - self.pad_t)
            with self.tik_instance.else_scope():
                self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.pad_w * C_ZERO -
                                     self.pad_r * C_ZERO)
                self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.pad_w * C_ZERO +
                                   self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.input_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.input_h)
        with self.tik_instance.else_scope():
            self.size_1.set_as(self.pad_l * C_ZERO)
            self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
            self.offset.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO)
            with self.tik_instance.if_scope(self.after_h <= self.pad_h - self.pad_b):
                self.offset_2.set_as(self.len_h * self.pad_w * C_ZERO - self.pad_r * C_ZERO)
                self.size_2.set_as(self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.len_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.len_h)
            with self.tik_instance.else_scope():
                self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.pad_w * C_ZERO -
                                     self.pad_r * C_ZERO)
                self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.pad_w * C_ZERO +
                                   self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.pad_h - self.pad_b - self.before_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.pad_h - self.pad_b - self.before_h)
        # other params form w dim
        with self.tik_instance.if_scope(self.pad_w <= self.input_w):
            self.burst_len_in.set_as(self.pad_w)
            self.src_stride.set_as(self.input_w - self.pad_w)
            self.dst_stride.set_as(0)
        with self.tik_instance.else_scope():
            self.burst_len_in.set_as(self.input_w)
            self.src_stride.set_as(0)
            self.dst_stride.set_as(self.pad_r + self.pad_l)

        # ub operate
        with self.tik_instance.new_stmt_scope():

            def _inner(ele_idx, ub_x, ub_y):
                self.offset_in.set_as((core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * C_ZERO +
                                      self.offset)
                self.offset_out.set_as((core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w *
                                       C_ZERO + loop_idx * self.h_factor * self.output_w * C_ZERO)
                # vector dup and move in
                with self.tik_instance.if_scope(tik.all(self.pad_w > 255, self.size_3 > 0)):
                    self.vector_dup_continuous(ub_x, self.size)
                with self.tik_instance.else_scope():
                    self.vector_dup_continuous(ub_x, self.size_1)
                    self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                    self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)
                self.tik_instance.data_move(ub_x[self.size_1], self.input_gm[self.offset_in], 0, self.nburst,
                                            self.burst_len_in, self.src_stride, self.dst_stride)
                # reduce max
                self.reduce_max(ub_x, ub_y, self.len_h, ele)
                # move out
                self.tik_instance.data_move(self.output_gm[self.offset_out], ub_x, 0, 1, self.burst_len_out, 0, 0)

            # open double buffer
            self.init_ub_tensor()
            self.init_ub_scalar_double()
            with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                _inner(ele_idx * 2, self.ub_a, self.ub_b)
                _inner(ele_idx * 2 + 1, self.ub_c, self.ub_d)
            with self.tik_instance.if_scope(core_ele % 2 == 1):
                _inner(core_ele - 1, self.ub_a, self.ub_b)

    def tiling_w_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        """Tiling w dim
        """
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_w_dim_core_nc_process(core_idx, core_ele, loop_idx, self.w_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_w_dim_core_nc_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_w_dim_core_nc_process(self, core_idx, core_ele, loop_idx, ele):
        """Tiling w dim process
        """
        # common params
        self.before_w.set_as(loop_idx * self.w_factor * self.strides_w)
        self.after_w.set_as((loop_idx * self.w_factor + ele - 1) * self.strides_w + self.ksize_w)
        self.len_w.set_as(self.after_w - self.before_w)
        self.size_w.set_as(ele * C_ZERO)
        self.burst_len_out.set_as(ele)
        # vector_dup_discrete -> vector_dup_continuous
        self.size.set_as(self.ksize_h * self.len_w * C_ZERO)
        # other params from h and w dim
        with self.tik_instance.for_range(0, self.output_h) as h_idx:
            # len_h equal to ksize_h
            self.before_h.set_as(h_idx * self.h_factor * self.strides_h)
            self.after_h = self.before_h + self.ksize_h
            with self.tik_instance.if_scope(self.before_h < self.pad_t):
                # other params form h dim
                self.size_1.set_as((self.pad_t - self.before_h) * self.len_w * C_ZERO)
                self.offset_2.set_as(0)
                self.size_2.set_as(0)
                self.repeat_3.set_as(self.after_h - self.pad_t)
                self.nburst.set_as(self.after_h - self.pad_t)
                # other params form w dim
                with self.tik_instance.if_scope(self.before_w < self.pad_l):
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(self.size_1)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset.set_as(0)
                        self.offset_ub.set_as(self.size_1 + (self.pad_l - self.before_w) * C_ZERO)
                        self.burst_len_in.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset.set_as((self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(self.size_1)
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len_in.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_r - self.before_w) * C_ZERO)
                        self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * C_ZERO)
                        self.burst_len_in.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))
            with self.tik_instance.else_scope():
                # other params form h dim
                self.size_1.set_as(0)
                with self.tik_instance.if_scope(self.after_h <= self.pad_h - self.pad_b):
                    self.offset_2.set_as(0)
                    self.size_2.set_as(0)
                    self.repeat_3.set_as(self.ksize_h)
                    self.nburst.set_as(self.ksize_h)
                with self.tik_instance.else_scope():
                    self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.len_w * C_ZERO)
                    self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.len_w * C_ZERO)
                    self.repeat_3.set_as(self.pad_h - self.pad_b - self.before_h)
                    self.nburst.set_as(self.pad_h - self.pad_b - self.before_h)
                # other params form w dim
                with self.tik_instance.if_scope(self.before_w < self.pad_l):
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO)
                        self.offset_ub.set_as(self.size_3)
                        self.burst_len_in.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO +
                                       (self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(0)
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len_in.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as((self.pad_w - self.pad_r - self.before_w) * C_ZERO)
                        self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * C_ZERO)
                        self.burst_len_in.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))

            # ub operate
            with self.tik_instance.new_stmt_scope():

                def _inner(ele_idx, ub_x, ub_y):
                    self.offset_in.set_as((core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w *
                                          C_ZERO + self.offset)
                    self.offset_out.set_as((core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w *
                                           C_ZERO + h_idx * self.output_w * C_ZERO + loop_idx * self.w_factor * C_ZERO)
                    # vector dup and move in
                    with self.tik_instance.if_scope(tik.all(self.len_w > 255, self.size_3 > 0)):
                        self.vector_dup_continuous(ub_x, self.size)
                    with self.tik_instance.else_scope():
                        self.vector_dup_continuous(ub_x, self.size_1)
                        self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                        self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.len_w)
                    self.tik_instance.data_move(ub_x[self.offset_ub], self.input_gm[self.offset_in], 0, self.nburst,
                                                self.burst_len_in, self.src_stride, self.dst_stride)
                    # reduce max for w
                    # if call 'reduce_max_rw', blk must less than or equal to 31, because 'rep = blk * 8'
                    if self.ksize_w == 1:
                        with self.tik_instance.for_range(0, self.ksize_h) as k_idx:
                            self.offset_dst.set_as(k_idx * self.size_w)
                            self.offset_src0.set_as(k_idx * self.len_w * C_ZERO)
                            self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src0:],
                                               self.size_w, self.strides_w, self.strides_w)
                    else:
                        with self.tik_instance.for_range(0, self.ksize_h) as k_idx:
                            self.offset_dst.set_as(k_idx * self.size_w)
                            self.offset_src0.set_as(k_idx * self.len_w * C_ZERO)
                            self.offset_src1.set_as(self.offset_src0 + C_ZERO)
                            self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src1:],
                                               self.size_w, self.strides_w, self.strides_w)
                            with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                                self.offset_w.set_as(self.offset_src0 + (idx + 2) * C_ZERO)
                                self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_w:], ub_y[self.offset_dst:],
                                                   self.size_w, self.strides_w, 1)
                    # reduce max for h
                    # if call 'reduce_max_rw', blk must less than or equal to 31, because 'rep = blk * 8'
                    if self.ksize_h == 1:
                        self.reduce_max_rw(ub_x, ub_y, ub_y, self.size_w, 1, 1)
                    else:
                        self.reduce_max_rw(ub_x, ub_y, ub_y[self.size_w:], self.size_w, 1, 1)
                        with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                            self.offset_h.set_as((idx + 2) * self.size_w)
                            self.reduce_max_rw(ub_x, ub_y[self.offset_h:], ub_x, self.size_w, 1, 1)
                    # move out
                    self.tik_instance.data_move(self.output_gm[self.offset_out], ub_x, 0, 1, self.burst_len_out, 0, 0)

                # open double buffer
                self.init_ub_tensor()
                self.init_ub_scalar_double()
                with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                    _inner(ele_idx * 2, self.ub_a, self.ub_b)
                    _inner(ele_idx * 2 + 1, self.ub_c, self.ub_d)
                with self.tik_instance.if_scope(core_ele % 2 == 1):
                    _inner(core_ele - 1, self.ub_a, self.ub_b)

    def max_pool_compute_tiling(self):
        """Maxpool compute tiling
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_idx:
            # define tiling ub and move tiling gm to tiling ub,then get tiling args
            self.tiling_ub = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,),
                                                      name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 3, 0, 0)
            self.tiling_args()

            # call select tiling mode function
            self.init_ub_scalar_single()
            with self.tik_instance.if_scope(core_idx < self.act_core_num - 1):
                self.core_ele.set_as(self.one_core_ele)
                self.loop_num.set_as(self.one_core_loop_num)
                self.loop_left.set_as(self.one_core_loop_left)
            with self.tik_instance.if_scope(core_idx == self.act_core_num - 1):
                self.core_ele.set_as(self.last_core_ele)
                self.loop_num.set_as(self.last_core_loop_num)
                self.loop_left.set_as(self.last_core_loop_left)
            self.select_tiling_mode(core_idx, self.core_ele, self.loop_num, self.loop_left)

    def max_pool_operator(self):
        """Maxpool operator
        """
        self.max_pool_compute_tiling()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm])

        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": self.ub_ele,
                "core_num": self.core_num,
                "ksize_h": self.ksize_h,
                "ksize_w": self.ksize_w,
                "strides_h": self.strides_h,
                "strides_w": self.strides_w,
                "padding": self.padding
            })

        return self.tik_instance


@register_operator("MaxPool")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def max_pool(input_data, output_data, ksize, strides, padding, data_format="NC1HWC0", kernel_name="max_pool"):
    """Performs max pooling on input tensor.

    Parameters
    ----------
    input_data: dict
        dict of input_data, include keys(shape and dtype).
    output_data: dict
        dict of output_data, include keys(shape and dtype).
    ksize: list or tuple
        A list of `ints` that has length 4.
        The size of the window for each dimension of the input tensor.
    strides: list or tuple
        A list of `ints` that has length 4.
        The stride of the sliding window for each dimension of the input tensor.
    padding: str
        A `string` from: `"SAME", "VALID"`.The type of padding algorithm to use.
    data_format: str
        A `string` from: `"NC1HWC0", "NHWC", "NCHW"`.
    kernel_name: str
        kernel name, default value is 'max_pool'

    Returns:
    -------
    None
    """
    # get input shape, format and dtype
    input_shape = input_data.get("shape")
    input_dtype = input_data.get("dtype").lower()
    input_format = input_data.get("format")

    # check input shape, format and dtype
    para_check.check_shape(input_shape, param_name="input_data")
    para_check.check_dtype(input_dtype, ("float16",), param_name="input_data")
    if input_format not in ("NC1HWC0",):
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "input_data", "NC1HWC0", input_format)
    if len(input_shape) != 5:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "input_data length", "5", str(len(input_shape)))

    # check data_format, ksize and strides
    if len(ksize) != 4:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "kszie length", "4", str(len(ksize)))
    if len(strides) != 4:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides length", "4", str(len(strides)))
    if data_format in ("NHWC",):
        ksize_n, ksize_h, ksize_w, ksize_c = ksize[0], ksize[1], ksize[2], ksize[3]
        strides_n, strides_h, strides_w, strides_c = strides[0], strides[1], strides[2], strides[3]
    elif data_format in ("NC1HWC0", "NCHW"):
        ksize_n, ksize_h, ksize_w, ksize_c = ksize[0], ksize[2], ksize[3], ksize[1]
        strides_n, strides_h, strides_w, strides_c = strides[0], strides[2], strides[3], strides[1]
    else:
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "data_format", "NHWC,NC1HWC0,NCHW",
                                                            data_format)
    if ksize_n != 1 or ksize_c != 1:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize_n and ksize_c", "1",
                                                           str(ksize_n) + ' and ' + str(ksize_c))
    if strides_n != 1 or strides_c != 1:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides_n and strides_c", "1",
                                                           str(strides_n) + ' and ' + str(strides_c))
    if ksize_h < 1 or ksize_w < 1:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize_h and ksize_w", "greater than 0",
                                                           str(ksize_h) + ' and ' + str(ksize_w))
    if strides_h < 1 or strides_w < 1:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides_h and strides_w", "greater than 0",
                                                           str(ksize_h) + ' and ' + str(ksize_w))

    # check padding
    if padding not in ("SAME", "VALID"):
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "padding", "SAME,VALID", padding)

    # run tik
    obj = MaxPool(input_dtype, [ksize_h, ksize_w], [strides_h, strides_w], padding, kernel_name)
    obj.max_pool_operator()
