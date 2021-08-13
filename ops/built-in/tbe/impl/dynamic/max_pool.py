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
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
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
MIN_FP16 = -65504.0
# compare times in each block
BLOCK_PROCESS_NUM = 7


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-statements
# pylint: disable=too-many-locals,unused-argument,attribute-defined-outside-init,too-many-lines,too-many-public-methods
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
        self.dtype_size = tbe_platform.get_bit_len(self.dtype) // EIGHT_BIT
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE) // self.dtype_size
        # reduce need three buffer and open double buffer
        self.one_sixth_ub_ele = self.ub_ele // 6
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
        self.n_c1 = self.tik_instance.Scalar("int32", name="n_c1")
        self.n_c1.set_as(self.tiling_ub[21])

    def init_gm_tensor(self):
        """Init gm tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="input_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="output_gm", scope=tik.scope_gm)

    def init_ub_tensor(self):
        """Init ub tensor
        """
        self.ub_a = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_a", scope=tik.scope_ubuf)
        self.ub_b = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_b", scope=tik.scope_ubuf)
        self.ub_c = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_c", scope=tik.scope_ubuf)
        self.ub_d = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_d", scope=tik.scope_ubuf)
        self.ub_e = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_e", scope=tik.scope_ubuf)
        self.ub_f = self.tik_instance.Tensor(self.dtype, (self.one_sixth_ub_ele,), name="ub_f", scope=tik.scope_ubuf)

    def init_ub_scalar(self):
        """Init ub scalar
        """
        # first core and last core use the same scalar
        self.core_ele = self.tik_instance.Scalar("int32", name="core_ele")
        self.loop_num = self.tik_instance.Scalar("int32", name="loop_num")
        self.loop_left = self.tik_instance.Scalar("int32", name="loop_left")
        # only for data_move in
        self.offset_gm = self.tik_instance.Scalar("int32", name="offset_gm")
        self.offset_ub = self.tik_instance.Scalar("int32", name="offset_ub")
        self.nburst = self.tik_instance.Scalar("int32", name="nburst")
        self.burst_len = self.tik_instance.Scalar("int32", name="burst_len_in")
        self.src_stride = self.tik_instance.Scalar("int32", name="src_stride")
        self.dst_stride = self.tik_instance.Scalar("int32", name="dst_stride")
        # only for vector dup, offset_1 is zero
        self.size_1 = self.tik_instance.Scalar("int32", name="size_1")
        self.offset_2 = self.tik_instance.Scalar("int32", name="offset_2")
        self.size_2 = self.tik_instance.Scalar("int32", name="size_2")
        self.offset_3 = self.tik_instance.Scalar("int32", name="offset_3")
        self.repeat_3 = self.tik_instance.Scalar("int32", name="repeat_3")
        self.size_3 = self.tik_instance.Scalar("int32", name="size_3")
        # only for reduce max
        self.before_h = self.tik_instance.Scalar("int32", name="before_h")
        self.after_h = self.tik_instance.Scalar("int32", name="after_h")
        self.before_w = self.tik_instance.Scalar("int32", name="before_w")
        self.after_w = self.tik_instance.Scalar("int32", name="after_w")
        self.len_h = self.tik_instance.Scalar("int32", name="len_h")
        self.len_w = self.tik_instance.Scalar("int32", name="len_w")

    def copy_only(self, core_idx, loop_num, loop_left):
        """Only execute move in and move out
        """
        # max ele of 'n*c1*h*w' can move to ub
        self.max_ele = self.ub_ele // C_ZERO
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.copy_only_process(core_idx, loop_idx, self.max_ele)
        with self.tik_instance.if_scope(loop_left > 0):
            self.copy_only_process(core_idx, loop_num, loop_left)

    def copy_only_process(self, core_idx, loop_idx, ele):
        """Only execute move in and move out
        """
        offset = (core_idx * self.one_core_ele + loop_idx * self.max_ele) * C_ZERO
        ub_tensor = self.tik_instance.Tensor(self.dtype, (self.max_ele * C_ZERO,),
                                             name="ub_tensor",
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(ub_tensor, self.input_gm[offset], 0, 1, ele, 0, 0)
        self.tik_instance.data_move(self.output_gm[offset], ub_tensor, 0, 1, ele, 0, 0)

    def vector_dup_continuous(self, src, size):
        """vector_dup continuous function, set ubuf to -65400
        """
        with self.tik_instance.if_scope(size > 0):
            size_loop = size // MASK
            size_left = size % MASK
            repeat_loop = size_loop // REPEAT_LIMIT
            repeat_left = size_loop % REPEAT_LIMIT

            with self.tik_instance.if_scope(repeat_left > 0):
                repeat_offset = repeat_loop * REPEAT_LIMIT * MASK
                self.tik_instance.vector_dup(MASK, src[repeat_offset], MIN_FP16, repeat_left, 1, 8)
            with self.tik_instance.if_scope(size_left > 0):
                size_offset = size_loop * MASK
                self.tik_instance.vector_dup(size_left, src[size_offset], MIN_FP16, 1, 1, 8)

    def vector_dup_discrete(self, src, repeat, size, dst_blk=1, dst_rep=8):
        """vector_dup discrete function, set ubuf to -65400,dst_rep is pad_w or len_w(cut w)
        """
        with self.tik_instance.if_scope(size > 0):
            # dst_rep less and equal to 255
            with self.tik_instance.if_scope(dst_rep <= 255):
                size_loop = size // MASK
                size_left = size % MASK
                repeat_loop = repeat // REPEAT_LIMIT
                repeat_left = repeat % REPEAT_LIMIT

                def _inner(src, mask_len):
                    """exec repeat
                    """
                    with self.tik_instance.for_range(0, repeat_loop) as repeat_loop_idx:
                        repeat_offset = repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO
                        self.tik_instance.vector_dup(mask_len, src[repeat_offset], MIN_FP16, REPEAT_LIMIT, dst_blk,
                                                     dst_rep)
                    with self.tik_instance.if_scope(repeat_left > 0):
                        repeat_offset = repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO
                        self.tik_instance.vector_dup(mask_len, src[repeat_offset], MIN_FP16, repeat_left, dst_blk,
                                                     dst_rep)

                with self.tik_instance.for_range(0, size_loop) as size_loop_idx:
                    size_offset = size_loop_idx * MASK
                    _inner(src[size_offset:], MASK)
                with self.tik_instance.if_scope(size_left > 0):
                    size_offset = size_loop * MASK
                    _inner(src[size_offset:], size_left)
            # dst_rep greater to 255
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, repeat) as repeat_loop_idx:
                    repeat_offset = repeat_loop_idx * dst_rep * C_ZERO
                    self.vector_dup_continuous(src[repeat_offset:], size)

    def reduce_max_repeat_width(self, dst, src0, src1, size, src0_blk=1, src1_blk=1):
        """reduce max for width and height with repeat width, src0_blk/src1_blk is strides_w or one
        """
        # strides_w less and equal to 31
        if self.strides_w <= 31:
            with self.tik_instance.if_scope(size > 0):
                size_loop = size // MASK
                size_left = size % MASK
                repeat_loop = size_loop // REPEAT_LIMIT
                repeat_left = size_loop % REPEAT_LIMIT

                with self.tik_instance.if_scope(repeat_left > 0):
                    repeat_offset = repeat_loop * REPEAT_LIMIT * MASK
                    repeat_offset_src0 = repeat_loop * REPEAT_LIMIT * src0_blk * MASK
                    repeat_offset_src1 = repeat_loop * REPEAT_LIMIT * src1_blk * MASK
                    self.tik_instance.vmax(MASK, dst[repeat_offset], src0[repeat_offset_src0], src1[repeat_offset_src1],
                                           repeat_left, 1, src0_blk, src1_blk, 8, src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(size_left > 0):
                    size_offset = size_loop * MASK
                    size_offset_src0 = size_loop * src0_blk * MASK
                    size_offset_src1 = size_loop * src1_blk * MASK
                    self.tik_instance.vmax(size_left, dst[size_offset], src0[size_offset_src0], src1[size_offset_src1],
                                           1, 1, src0_blk, src1_blk, 8, src0_blk * 8, src1_blk * 8)
        # strides_w greater to 31
        else:
            with self.tik_instance.if_scope(size > 0):
                size_loop = size // C_ZERO
                with self.tik_instance.for_range(0, size_loop) as size_loop_idx:
                    size_offset = size_loop_idx * C_ZERO
                    size_offset_src0 = size_loop_idx * src0_blk * C_ZERO
                    size_offset_src1 = size_loop_idx * src1_blk * C_ZERO
                    self.tik_instance.vmax(C_ZERO, dst[size_offset], src0[size_offset_src0], src1[size_offset_src1], 1,
                                           1, 1, 1, 8, 8, 8)

    def reduce_max_repeat_width_ksize_one_width(self, ub_x, ub_y, repeat_ph, size_ow, size_pw):
        """Reduce max width with repeat width when ksize equal to one
        """
        if self.strides_w == self.ksize_w:
            self.reduce_max_repeat_width(ub_y, ub_x, ub_x, repeat_ph * size_ow, self.strides_w, self.strides_w)
        else:
            with self.tik_instance.for_range(0, repeat_ph) as p_idx:
                offset_dst = p_idx * size_ow
                offset_src = p_idx * size_pw
                self.reduce_max_repeat_width(ub_y[offset_dst:], ub_x[offset_src:], ub_x[offset_src:], size_ow,
                                             self.strides_w, self.strides_w)

    def reduce_max_repeat_width_ksize_more_width(self, ub_x, ub_y, repeat_ph, size_ow, size_pw):
        """Reduce max width with repeat width when ksize not equal to one
        """
        if self.strides_w == self.ksize_w:
            self.reduce_max_repeat_width(ub_y, ub_x, ub_x[C_ZERO:], repeat_ph * size_ow, self.strides_w, self.strides_w)
            with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                offset_w = (idx + 2) * C_ZERO
                self.reduce_max_repeat_width(ub_y, ub_x[offset_w:], ub_y, repeat_ph * size_ow, self.strides_w, 1)
        else:
            with self.tik_instance.for_range(0, repeat_ph) as p_idx:
                offset_dst = p_idx * size_ow
                offset_src0 = p_idx * size_pw
                offset_src1 = offset_src0 + C_ZERO
                self.reduce_max_repeat_width(ub_y[offset_dst:], ub_x[offset_src0:], ub_x[offset_src1:], size_ow,
                                             self.strides_w, self.strides_w)
                with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                    offset_w = offset_src0 + (idx + 2) * C_ZERO
                    self.reduce_max_repeat_width(ub_y[offset_dst:], ub_x[offset_w:], ub_y[offset_dst:], size_ow,
                                                 self.strides_w, 1)

    def reduce_max_repeat_width_ksize_one_height(self, ub_y, ub_z, repeat_oh, size_ow):
        """Reduce max height with repeat width when ksize equal to one
        """
        if self.strides_h == 1:
            self.reduce_max_repeat_width(ub_z, ub_y, ub_y, repeat_oh * size_ow, 1, 1)
        else:
            with self.tik_instance.for_range(0, repeat_oh) as o_idx:
                offset_dst = o_idx * size_ow
                offset_src = o_idx * self.strides_h * size_ow
                self.reduce_max_repeat_width(ub_z[offset_dst:], ub_y[offset_src:], ub_y[offset_src:], size_ow, 1, 1)

    def reduce_max_repeat_width_ksize_more_height(self, ub_y, ub_z, repeat_oh, size_ow):
        """Reduce max height with repeat width when ksize not equal to one
        """
        if self.strides_h == 1:
            self.reduce_max_repeat_width(ub_z, ub_y, ub_y[size_ow], repeat_oh * size_ow, 1, 1)
            with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                offset_h = (idx + 2) * size_ow
                self.reduce_max_repeat_width(ub_z, ub_y[offset_h:], ub_z, repeat_oh * size_ow, 1, 1)
        else:
            with self.tik_instance.for_range(0, repeat_oh) as o_idx:
                offset_dst = o_idx * size_ow
                offset_src0 = o_idx * self.strides_h * size_ow
                offset_src1 = offset_src0 + size_ow
                self.reduce_max_repeat_width(ub_z[offset_dst:], ub_y[offset_src0:], ub_y[offset_src1:], size_ow, 1, 1)
                with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                    offset_h = offset_src0 + (idx + 2) * size_ow
                    self.reduce_max_repeat_width(ub_z[offset_dst:], ub_y[offset_h:], ub_z[offset_dst:], size_ow, 1, 1)

    def reduce_max_repeat_height(self,
                                 dst,
                                 src0,
                                 src1,
                                 repeat,
                                 size,
                                 src0_blk=1,
                                 src1_blk=1,
                                 dst_rep=8,
                                 src0_rep=8,
                                 src1_rep=8):
        """reduce max for width and height with repeat height,
        repeat is pad_h and output_h, size is output_w * c_zero,
        src0_blk/src1_blk is strides_w(<=255), dst_rep is output_w(<=255),
        src0_rep/src1_rep is pad_w or ouput_w or output_w * strides_h(<=255)
        """
        with self.tik_instance.if_scope(size > 0):
            size_loop = size // MASK
            size_left = size % MASK
            repeat_loop = repeat // REPEAT_LIMIT
            repeat_left = repeat % REPEAT_LIMIT

            def _inner(dst, src0, src1, mask_len):
                """exec repeat
                """
                with self.tik_instance.for_range(0, repeat_loop) as repeat_loop_idx:
                    # dst_rep may not equal to 8, discrete emissions
                    repeat_offset = repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO
                    # src_rep may not equal to 8, discrete emissions
                    repeat_offset_src0 = repeat_loop_idx * REPEAT_LIMIT * src0_rep * C_ZERO
                    repeat_offset_src1 = repeat_loop_idx * REPEAT_LIMIT * src1_rep * C_ZERO
                    self.tik_instance.vmax(mask_len, dst[repeat_offset], src0[repeat_offset_src0],
                                           src1[repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, dst_rep,
                                           src0_rep, src1_rep)
                with self.tik_instance.if_scope(repeat_left > 0):
                    repeat_offset = repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO
                    repeat_offset_src0 = repeat_loop * REPEAT_LIMIT * src0_rep * C_ZERO
                    repeat_offset_src1 = repeat_loop * REPEAT_LIMIT * src1_rep * C_ZERO
                    self.tik_instance.vmax(mask_len, dst[repeat_offset], src0[repeat_offset_src0],
                                           src1[repeat_offset_src1], repeat_left, 1, src0_blk, src1_blk, dst_rep,
                                           src0_rep, src1_rep)

            # exec size
            with self.tik_instance.for_range(0, size_loop) as size_loop_idx:
                # dst_blk must be equal to 1, continuous emissions
                size_offset = size_loop_idx * MASK
                # src_blk may not equal to 1, discrete emissions
                size_offset_src0 = size_loop_idx * src0_blk * MASK
                size_offset_src1 = size_loop_idx * src1_blk * MASK
                _inner(dst[size_offset:], src0[size_offset_src0:], src1[size_offset_src1:], MASK)
            with self.tik_instance.if_scope(size_left > 0):
                size_offset = size_loop * MASK
                size_offset_src0 = size_loop * src0_blk * MASK
                size_offset_src1 = size_loop * src1_blk * MASK
                _inner(dst[size_offset:], src0[size_offset_src0:], src1[size_offset_src1:], size_left)

    def reduce_max_repeat_height_ksize_one_width(self, ub_x, ub_y, repeat_ph, size_ow):
        """Reduce max width with repeat height when ksize equal to one
        """
        if self.strides_w == self.ksize_w:
            self.reduce_max_repeat_width(ub_y, ub_x, ub_x, repeat_ph * size_ow, self.strides_w, self.strides_w)
        else:
            self.reduce_max_repeat_height(ub_y, ub_x, ub_x, repeat_ph, size_ow, self.strides_w, self.strides_w,
                                          self.output_w, self.pad_w, self.pad_w)

    def reduce_max_repeat_height_ksize_more_width(self, ub_x, ub_y, repeat_ph, size_ow):
        """Reduce max width with repeat height when ksize not equal to one
        """
        if self.strides_w == self.ksize_w:
            self.reduce_max_repeat_width(ub_y, ub_x, ub_x[C_ZERO:], repeat_ph * size_ow, self.strides_w, self.strides_w)
            with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                offset_w = (idx + 2) * C_ZERO
                self.reduce_max_repeat_width(ub_y, ub_x[offset_w:], ub_y, repeat_ph * size_ow, self.strides_w, 1)
        else:
            self.reduce_max_repeat_height(ub_y, ub_x, ub_x[C_ZERO:], repeat_ph, size_ow, self.strides_w, self.strides_w,
                                          self.output_w, self.pad_w, self.pad_w)
            with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                offset_w = (idx + 2) * C_ZERO
                self.reduce_max_repeat_height(ub_y, ub_x[offset_w:], ub_y, repeat_ph, size_ow, self.strides_w, 1,
                                              self.output_w, self.pad_w, self.output_w)

    def reduce_max_repeat_height_ksize_one_height(self, ub_y, ub_z, repeat_oh, size_ow):
        """Reduce max height with repeat height when ksize equal to one
        """
        if self.strides_h == 1:
            self.reduce_max_repeat_width(ub_z, ub_y, ub_y, repeat_oh * size_ow, 1, 1)
        else:
            self.reduce_max_repeat_height(ub_z, ub_y, ub_y, repeat_oh, size_ow, 1, 1, self.output_w,
                                          self.output_w * self.strides_h, self.output_w * self.strides_h)

    def reduce_max_repeat_height_ksize_more_height(self, ub_y, ub_z, repeat_oh, size_ow):
        """Reduce max height with repeat height when ksize not equal to one
        """
        if self.strides_h == 1:
            self.reduce_max_repeat_width(ub_z, ub_y, ub_y[size_ow], repeat_oh * size_ow, 1, 1)
            with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                offset_h = (idx + 2) * size_ow
                self.reduce_max_repeat_width(ub_z, ub_y[offset_h:], ub_z, repeat_oh * size_ow, 1, 1)
        else:
            self.reduce_max_repeat_height(ub_z, ub_y, ub_y[size_ow:], repeat_oh, size_ow, 1, 1, self.output_w,
                                          self.output_w * self.strides_h, self.output_w * self.strides_h)
            with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                offset_h = (idx + 2) * size_ow
                self.reduce_max_repeat_height(ub_z, ub_y[offset_h:], ub_z, repeat_oh, size_ow, 1, 1, self.output_w,
                                              self.output_w * self.strides_h, self.output_w)

    def tiling_c_dim_core_nc(self, core_idx, loop_num, loop_left):
        """Tiling c1 dim when core num at nc1
        """
        # move in and vector dup params
        self.size_1.set_as(self.pad_t * self.pad_w * C_ZERO + self.pad_l * C_ZERO)
        self.offset_2.set_as((self.pad_h - self.pad_b) * self.pad_w * C_ZERO - self.pad_r * C_ZERO)
        self.size_2.set_as(self.pad_b * self.pad_w * C_ZERO + self.pad_r * C_ZERO)
        self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
        self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
        # when pad and cut at same time
        with self.tik_instance.if_scope((self.pad_t > 0)&((self.pad_h - self.pad_t) < self.input_h)):    
            self.nburst.set_as(self.pad_h - self.pad_t)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.pad_h <= self.input_h):
                self.nburst.set_as(self.pad_h)
            with self.tik_instance.else_scope():
                self.nburst.set_as(self.input_h)
        with self.tik_instance.if_scope((self.pad_l > 0)&((self.pad_w - self.pad_l) < self.input_w)):
            self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_l - self.pad_r) * C_ZERO)
            self.src_stride.set_as(self.input_w - self.pad_w + self.pad_l)
            self.dst_stride.set_as(self.pad_l)
            self.burst_len.set_as(self.pad_w - self.pad_l)
            self.repeat_3.set_as(self.pad_h - 1 - self.pad_t)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.pad_w <= self.input_w):
                self.repeat_3.set_as(0)
                self.src_stride.set_as(self.input_w - self.pad_w)
                self.dst_stride.set_as(0)
                with self.tik_instance.if_scope(self.pad_w < self.input_w):
                    self.burst_len.set_as(self.pad_w)
                with self.tik_instance.else_scope():
                    self.burst_len.set_as(self.nburst * self.pad_w)
                    self.nburst.set_as(1)
            with self.tik_instance.else_scope():
                self.repeat_3.set_as(self.pad_h - 1)
                self.burst_len.set_as(self.input_w)
                self.src_stride.set_as(0)
                self.dst_stride.set_as(self.pad_r + self.pad_l)

        # run loop
        self.init_ub_tensor()
        with self.tik_instance.for_range(0, loop_num // 2) as loop_idx:
            self.tiling_c_dim_core_nc_process(self.ub_a, self.ub_b, self.ub_c, core_idx, loop_idx * 2, self.c_factor)
            self.tiling_c_dim_core_nc_process(self.ub_d, self.ub_e, self.ub_f, core_idx, loop_idx * 2 + 1,
                                              self.c_factor)
        with self.tik_instance.if_scope(loop_num % 2 == 1):
            self.tiling_c_dim_core_nc_process(self.ub_a, self.ub_b, self.ub_c, core_idx, loop_num - 1, self.c_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_c_dim_core_nc_process(self.ub_d, self.ub_e, self.ub_f, core_idx, loop_num, loop_left)

    def tiling_c_dim_core_nc_process(self, ub_x, ub_y, ub_z, core_idx, loop_idx, ele):
        """Tiling c1 dim process when core num at nc1
        """
        # vector dup and move in
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, ele) as ele_idx:
                offset_in = (core_idx * self.one_core_ele + loop_idx * self.c_factor +
                             ele_idx) * self.input_h * self.input_w * C_ZERO
                offset_a = ele_idx * self.pad_h * self.pad_w * C_ZERO
                self.tik_instance.data_move(ub_x[offset_a:][self.size_1], self.input_gm[offset_in], 0, self.nburst,
                                            self.burst_len, self.src_stride, self.dst_stride)
                self.vector_dup_continuous(ub_x[offset_a:], self.size_1)
                self.vector_dup_continuous(ub_x[offset_a:][self.offset_2:], self.size_2)
                self.vector_dup_discrete(ub_x[offset_a:][self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)

        with self.tik_instance.for_range(0, ele) as ele_idx:
            offset_a = ele_idx * self.pad_h * self.pad_w * C_ZERO
            offset_b = ele_idx * self.output_h * self.pad_w * C_ZERO
            offset_c = ele_idx * self.output_h * self.output_w * C_ZERO
            # reduce max width
            if self.ksize_w == 1:
                self.reduce_max_repeat_width_ksize_one_width(ub_x[offset_a:], ub_y[offset_b:], self.pad_h,
                                                             self.output_w * C_ZERO, self.pad_w * C_ZERO)
            else:
                self.reduce_max_repeat_width_ksize_more_width(ub_x[offset_a:], ub_y[offset_b:], self.pad_h,
                                                              self.output_w * C_ZERO, self.pad_w * C_ZERO)
            # reduce max height
            if self.ksize_h == 1:
                self.reduce_max_repeat_width_ksize_one_height(ub_y[offset_b:], ub_z[offset_c:], self.output_h,
                                                              self.output_w * C_ZERO)
            else:
                self.reduce_max_repeat_width_ksize_more_height(ub_y[offset_b:], ub_z[offset_c:], self.output_h,
                                                               self.output_w * C_ZERO)
        # move out
        offset_out = (core_idx * self.one_core_ele + loop_idx * self.c_factor) * self.output_h * self.output_w * C_ZERO
        self.tik_instance.data_move(self.output_gm[offset_out], ub_z, 0, 1, ele * self.output_h * self.output_w, 0, 0)

    def tiling_h_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        """Tiling h dim when core num at nc1
        """
        # run loop
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_idx, self.h_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_h_dim_core_nc_process(self, core_idx, core_ele, loop_idx, ele):
        """Tiling h dim process when core num at nc1
        """
        # move in and vector dup params
        self.before_h.set_as(loop_idx * self.h_factor * self.strides_h)
        self.after_h.set_as((loop_idx * self.h_factor + ele - 1) * self.strides_h + self.ksize_h)
        self.len_h.set_as(self.after_h - self.before_h)
        with self.tik_instance.if_scope(self.before_h < self.pad_t):
            self.size_1.set_as((self.pad_t - self.before_h) * self.pad_w * C_ZERO + self.pad_l * C_ZERO)
            self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
            self.offset_gm.set_as(0)
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
            self.offset_gm.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO)
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
        # when pad and cut at same time
        with self.tik_instance.if_scope((self.pad_l > 0) & ((self.pad_w - self.pad_l) < self.input_w)):
            self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_l - self.pad_r) * C_ZERO)
            self.src_stride.set_as(self.input_w - self.pad_w + self.pad_l)
            self.dst_stride.set_as(self.pad_l)
            self.burst_len.set_as(self.pad_w - self.pad_l)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.pad_w <= self.input_w):
                self.src_stride.set_as(self.input_w - self.pad_w)
                self.dst_stride.set_as(0)
                with self.tik_instance.if_scope(self.pad_w < self.input_w):
                    self.burst_len.set_as(self.pad_w)
                with self.tik_instance.else_scope():
                    self.burst_len.set_as(self.nburst * self.pad_w)
                    self.nburst.set_as(1)
            with self.tik_instance.else_scope():
                self.burst_len.set_as(self.input_w)
                self.src_stride.set_as(0)
                self.dst_stride.set_as(self.pad_r + self.pad_l)

        def _inner(ub_x, ub_y, ub_z, ele_idx):
            offset_in = (core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * C_ZERO + self.offset_gm
            offset_out = (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * C_ZERO + \
                         loop_idx * self.h_factor * self.output_w * C_ZERO
            # vector dup and move in
            with self.tik_instance.if_scope(self.before_h <= self.pad_h - self.pad_b):
                with self.tik_instance.new_stmt_scope(disable_sync=True):
                    self.tik_instance.data_move(ub_x[self.size_1], self.input_gm[offset_in], 0, self.nburst,
                                                self.burst_len, self.src_stride, self.dst_stride)
                    self.vector_dup_continuous(ub_x, self.size_1)
                    self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                    self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)
            with self.tik_instance.else_scope():
                self.vector_dup_continuous(ub_x, self.len_h * self.pad_w * C_ZERO)
            # reduce max width
            if self.ksize_w == 1:
                self.reduce_max_repeat_width_ksize_one_width(ub_x, ub_y, self.len_h, self.output_w * C_ZERO,
                                                             self.pad_w * C_ZERO)
            else:
                self.reduce_max_repeat_width_ksize_more_width(ub_x, ub_y, self.len_h, self.output_w * C_ZERO,
                                                              self.pad_w * C_ZERO)
            # reduce max height
            if self.ksize_h == 1:
                self.reduce_max_repeat_width_ksize_one_height(ub_y, ub_z, ele, self.output_w * C_ZERO)
            else:
                self.reduce_max_repeat_width_ksize_more_height(ub_y, ub_z, ele, self.output_w * C_ZERO)
            # move out
            self.tik_instance.data_move(self.output_gm[offset_out], ub_z, 0, 1, ele * self.output_w, 0, 0)

        self.init_ub_tensor()
        with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
            _inner(self.ub_a, self.ub_b, self.ub_c, ele_idx * 2)
            _inner(self.ub_d, self.ub_e, self.ub_f, ele_idx * 2 + 1)
        with self.tik_instance.if_scope(core_ele % 2 == 1):
            _inner(self.ub_a, self.ub_b, self.ub_c, core_ele - 1)

    def tiling_w_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        """Tiling w dim when core num at nc1
        """
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_w_dim_core_nc_process(core_idx, core_ele, loop_idx, self.w_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_w_dim_core_nc_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_w_dim_core_nc_process(self, core_idx, core_ele, loop_idx, ele):
        """Tiling w dim process when core num at nc1
        """
        # move in and vector dup params
        self.before_w.set_as(loop_idx * self.w_factor * self.strides_w)
        self.after_w.set_as((loop_idx * self.w_factor + ele - 1) * self.strides_w + self.ksize_w)
        self.len_w.set_as(self.after_w - self.before_w)
        with self.tik_instance.for_range(0, self.output_h) as h_idx:
            self.before_h.set_as(h_idx * self.strides_h)
            self.after_h.set_as(self.before_h + self.ksize_h)
            with self.tik_instance.if_scope(self.before_h < self.pad_t):
                self.size_1.set_as((self.pad_t - self.before_h) * self.len_w * C_ZERO)
                self.offset_2.set_as(0)
                self.size_2.set_as(0)
                self.repeat_3.set_as(self.after_h - self.pad_t)
                self.nburst.set_as(self.after_h - self.pad_t)
                with self.tik_instance.if_scope(self.after_h > self.pad_t + self.input_h):
                    self.offset_2.set_as((self.pad_t - self.before_h + self.input_h) * self.len_h * C_ZERO)
                    self.size_2.set_as((self.after_h - self.pad_t - self.input_h) * self.len_w * C_ZERO)
                    self.repeat_3.set_as(self.input_h)
                    self.nburst.set_as(self.input_h)
                with self.tik_instance.if_scope(self.before_w < self.pad_l):
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(self.size_1)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset_gm.set_as(0)
                        self.offset_ub.set_as(self.size_1 + (self.pad_l - self.before_w) * C_ZERO)
                        self.burst_len.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset_gm.set_as((self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(self.size_1)
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_r - self.before_w) * C_ZERO)
                        self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * C_ZERO)
                        self.burst_len.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))
            with self.tik_instance.else_scope():
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
                with self.tik_instance.if_scope(self.before_w < self.pad_l):
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset_gm.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO)
                        self.offset_ub.set_as(self.size_3)
                        self.burst_len.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset_gm.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO +
                                          (self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(0)
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as((self.pad_w - self.pad_r - self.before_w) * C_ZERO)
                        self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * C_ZERO)
                        self.burst_len.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))

            def _inner(ub_x, ub_y, ub_z, ele_idx):
                offset_in = (core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * C_ZERO + \
                            self.offset_gm
                offset_out = (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * C_ZERO + \
                             h_idx * self.output_w * C_ZERO + loop_idx * self.w_factor * C_ZERO
                # vector dup and move in
                with self.tik_instance.if_scope(self.before_h <= self.pad_h - self.pad_b):
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        self.tik_instance.data_move(ub_x[self.offset_ub], self.input_gm[offset_in], 0, self.nburst,
                                                    self.burst_len, self.src_stride, self.dst_stride)
                        self.vector_dup_continuous(ub_x, self.size_1)
                        self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                        self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.len_w)
                with self.tik_instance.else_scope():
                    self.vector_dup_continuous(ub_x, (self.before_h - self.after_h) * self.len_w * C_ZERO)
                # reduce max width
                if self.ksize_w == 1:
                    self.reduce_max_repeat_width_ksize_one_width(ub_x, ub_y, self.ksize_h, ele * C_ZERO,
                                                                 self.len_w * C_ZERO)
                else:
                    self.reduce_max_repeat_width_ksize_more_width(ub_x, ub_y, self.ksize_h, ele * C_ZERO,
                                                                  self.len_w * C_ZERO)
                # reduce max height
                if self.ksize_h == 1:
                    self.reduce_max_repeat_width_ksize_one_height(ub_y, ub_z, 1, ele * C_ZERO)
                else:
                    self.reduce_max_repeat_width_ksize_more_height(ub_y, ub_z, 1, ele * C_ZERO)
                # move out
                self.tik_instance.data_move(self.output_gm[offset_out], ub_z, 0, 1, ele, 0, 0)

            self.init_ub_tensor()
            with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                _inner(self.ub_a, self.ub_b, self.ub_c, ele_idx * 2)
                _inner(self.ub_d, self.ub_e, self.ub_f, ele_idx * 2 + 1)
            with self.tik_instance.if_scope(core_ele % 2 == 1):
                _inner(self.ub_a, self.ub_b, self.ub_c, core_ele - 1)

    def tiling_h_dim_core_h(self, core_idx, loop_num, loop_left):
        """Tiling h dim when core num at h
        """
        # run loop
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_h_dim_core_h_process(core_idx, loop_idx, self.h_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_h_dim_core_h_process(core_idx, loop_num, loop_left)

    def tiling_h_dim_core_h_process(self, core_idx, loop_idx, ele):
        """Tiling h dim process when core num at nc1
        """
        # move in and vector dup params
        self.before_h.set_as((core_idx * self.one_core_ele + loop_idx * self.h_factor) * self.strides_h)
        self.after_h.set_as((core_idx * self.one_core_ele + loop_idx * self.h_factor + ele - 1) * self.strides_h +
                            self.ksize_h)
        self.len_h.set_as(self.after_h - self.before_h)
        with self.tik_instance.if_scope(self.before_h < self.pad_t):
            self.size_1.set_as((self.pad_t - self.before_h) * self.pad_w * C_ZERO + self.pad_l * C_ZERO)
            self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
            self.offset_gm.set_as(0)
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
            self.offset_gm.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO)
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

        # when pad and cut at same time
        with self.tik_instance.if_scope((self.pad_l > 0) & ((self.pad_w - self.pad_l) < self.input_w)):
            self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_l - self.pad_r) * C_ZERO)
            self.src_stride.set_as(self.input_w - self.pad_w + self.pad_l)
            self.dst_stride.set_as(self.pad_l)
            self.burst_len.set_as(self.pad_w - self.pad_l)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.pad_w <= self.input_w):
                self.src_stride.set_as(self.input_w - self.pad_w)
                self.dst_stride.set_as(0)
                with self.tik_instance.if_scope(self.pad_w < self.input_w):
                    self.burst_len.set_as(self.pad_w)
                with self.tik_instance.else_scope():
                    self.burst_len.set_as(self.nburst * self.pad_w)
                    self.nburst.set_as(1)
            with self.tik_instance.else_scope():
                self.burst_len.set_as(self.input_w)
                self.src_stride.set_as(0)
                self.dst_stride.set_as(self.pad_r + self.pad_l)

        def _inner(ub_x, ub_y, ub_z, idx):
            offset_in = idx * self.input_h * self.input_w * C_ZERO + self.offset_gm
            offset_out = idx * self.output_h * self.output_w * C_ZERO + (
                core_idx * self.one_core_ele + loop_idx * self.h_factor) * self.output_w * C_ZERO
            # vector dup and move in
            with self.tik_instance.if_scope(self.before_h <= self.pad_h - self.pad_b):
                with self.tik_instance.new_stmt_scope(disable_sync=True):
                    self.tik_instance.data_move(ub_x[self.size_1], self.input_gm[offset_in], 0, self.nburst,
                                                self.burst_len, self.src_stride, self.dst_stride)
                    self.vector_dup_continuous(ub_x, self.size_1)
                    self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                    self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)
            with self.tik_instance.else_scope():
                self.vector_dup_continuous(ub_x, self.len_h * self.pad_w * C_ZERO)
            # reduce max width
            if self.ksize_w == 1:
                self.reduce_max_repeat_width_ksize_one_width(ub_x, ub_y, self.len_h, self.output_w * C_ZERO,
                                                             self.pad_w * C_ZERO)
            else:
                self.reduce_max_repeat_width_ksize_more_width(ub_x, ub_y, self.len_h, self.output_w * C_ZERO,
                                                              self.pad_w * C_ZERO)
            # reduce max height
            if self.ksize_h == 1:
                self.reduce_max_repeat_width_ksize_one_height(ub_y, ub_z, ele, self.output_w * C_ZERO)
            else:
                self.reduce_max_repeat_width_ksize_more_height(ub_y, ub_z, ele, self.output_w * C_ZERO)
            # move out
            self.tik_instance.data_move(self.output_gm[offset_out], ub_z, 0, 1, ele * self.output_w, 0, 0)

        self.init_ub_tensor()
        with self.tik_instance.for_range(0, self.n_c1 // 2) as idx:
            _inner(self.ub_a, self.ub_b, self.ub_c, idx * 2)
            _inner(self.ub_d, self.ub_e, self.ub_f, idx * 2 + 1)
        with self.tik_instance.if_scope(self.n_c1 % 2 == 1):
            _inner(self.ub_a, self.ub_b, self.ub_c, self.n_c1 - 1)

    def tiling_w_dim_core_h(self, core_idx, core_ele, loop_num, loop_left):
        """Tiling w dim when core num at nc1
        """
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_w_dim_core_h_process(core_idx, core_ele, loop_idx, self.w_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_w_dim_core_h_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_w_dim_core_h_process(self, core_idx, core_ele, loop_idx, ele):
        """Tiling w dim process when core num at nc1
        """
        # move in and vector dup params
        self.before_w.set_as(loop_idx * self.w_factor * self.strides_w)
        self.after_w.set_as((loop_idx * self.w_factor + ele - 1) * self.strides_w + self.ksize_w)
        self.len_w.set_as(self.after_w - self.before_w)
        with self.tik_instance.for_range(0, core_ele) as h_idx:
            self.before_h.set_as((core_idx * self.one_core_ele + h_idx) * self.strides_h)
            self.after_h.set_as(self.before_h + self.ksize_h)
            with self.tik_instance.if_scope(self.before_h < self.pad_t):
                self.size_1.set_as((self.pad_t - self.before_h) * self.len_w * C_ZERO)
                self.offset_2.set_as(0)
                self.size_2.set_as(0)
                self.repeat_3.set_as(self.after_h - self.pad_t)
                self.nburst.set_as(self.after_h - self.pad_t)
                with self.tik_instance.if_scope(self.after_h > self.pad_t + self.input_h):
                    self.offset_2.set_as((self.pad_t - self.before_h + self.input_h) * self.len_w * C_ZERO)
                    self.size_2.set_as((self.after_h - self.pad_t - self.input_h) * self.len_w * C_ZERO)
                    self.repeat_3.set_as(self.input_h)
                    self.nburst.set_as(self.input_h)
                with self.tik_instance.if_scope(self.before_w < self.pad_l):
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(self.size_1)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset_gm.set_as(0)
                        self.offset_ub.set_as(self.size_1 + (self.pad_l - self.before_w) * C_ZERO)
                        self.burst_len.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset_gm.set_as((self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(self.size_1)
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_r - self.before_w) * C_ZERO)
                        self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * C_ZERO)
                        self.burst_len.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))
            with self.tik_instance.else_scope():
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
                with self.tik_instance.if_scope(self.before_w < self.pad_l):
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset_gm.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO)
                        self.offset_ub.set_as(self.size_3)
                        self.burst_len.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset_gm.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO +
                                          (self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(0)
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as((self.pad_w - self.pad_r - self.before_w) * C_ZERO)
                        self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r)) * C_ZERO)
                        self.burst_len.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))

            def _inner(ub_x, ub_y, ub_z, idx):
                offset_in = idx * self.input_h * self.input_w * C_ZERO + self.offset_gm
                offset_out = idx * self.output_h * self.output_w * C_ZERO + (
                    core_idx * self.one_core_ele + h_idx) * self.output_w * C_ZERO + loop_idx * self.w_factor * C_ZERO
                # vector dup and move in
                with self.tik_instance.if_scope(self.before_h <= self.pad_h - self.pad_b):
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        self.tik_instance.data_move(ub_x[self.offset_ub], self.input_gm[offset_in], 0, self.nburst,
                                                    self.burst_len, self.src_stride, self.dst_stride)
                        self.vector_dup_continuous(ub_x, self.size_1)
                        self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                        self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.len_w)
                with self.tik_instance.else_scope():
                    self.vector_dup_continuous(ub_x, (self.before_h - self.after_h) * self.len_w * C_ZERO)
                # reduce max width
                if self.ksize_w == 1:
                    self.reduce_max_repeat_width_ksize_one_width(ub_x, ub_y, self.ksize_h, ele * C_ZERO,
                                                                 self.len_w * C_ZERO)
                else:
                    self.reduce_max_repeat_width_ksize_more_width(ub_x, ub_y, self.ksize_h, ele * C_ZERO,
                                                                  self.len_w * C_ZERO)
                # reduce max height
                if self.ksize_h == 1:
                    self.reduce_max_repeat_width_ksize_one_height(ub_y, ub_z, 1, ele * C_ZERO)
                else:
                    self.reduce_max_repeat_width_ksize_more_height(ub_y, ub_z, 1, ele * C_ZERO)
                # move out
                self.tik_instance.data_move(self.output_gm[offset_out], ub_z, 0, 1, ele, 0, 0)

            self.init_ub_tensor()
            with self.tik_instance.for_range(0, self.n_c1 // 2) as idx:
                _inner(self.ub_a, self.ub_b, self.ub_c, idx * 2)
                _inner(self.ub_d, self.ub_e, self.ub_f, idx * 2 + 1)
            with self.tik_instance.if_scope(self.n_c1 % 2 == 1):
                _inner(self.ub_a, self.ub_b, self.ub_c, self.n_c1 - 1)

    def init_ub_global(self):
        """
        init ub when tiling mode is 6 or 7
        """
        self.ub_in = self.tik_instance.Tensor(self.dtype, (self.ub_ele,), name="ub_in", scope=tik.scope_ubuf)
        self.ub_tmp = self.tik_instance.Tensor(self.dtype, (MASK,), name="ub_tmp", scope=tik.scope_ubuf)
        self.ub_out = self.tik_instance.Tensor(self.dtype, (C_ZERO,), name="ub_out", scope=tik.scope_ubuf)

    def vector_dup_ub(self):
        """
        set an initial value
        """
        self.tik_instance.vector_dup(MASK, self.ub_tmp, MIN_FP16, 1, 1, 8)
        self.tik_instance.vector_dup(C_ZERO, self.ub_out, MIN_FP16, 1, 1, 8)

    def get_hw_max(self, tmp_ub, ub_in, size):
        """
        get max from dim h and dim w
        """
        size_loop = size // MASK
        size_left = size % MASK
        repeat_loop = size_loop // REPEAT_LIMIT
        repeat_left = size_loop % REPEAT_LIMIT
        with self.tik_instance.for_range(0, repeat_loop) as repeat_loop_index:
            self.tik_instance.vmax(MASK, tmp_ub, ub_in[repeat_loop_index * MASK * REPEAT_LIMIT], tmp_ub, 255, 1, 1, 1,
                                   0, 8, 0)
        with self.tik_instance.if_scope(repeat_left > 0):
            self.tik_instance.vmax(MASK, tmp_ub, ub_in[repeat_loop * MASK * REPEAT_LIMIT], tmp_ub, repeat_left, 1, 1, 1,
                                   0, 8, 0)
        with self.tik_instance.if_scope(size_left > 0):
            self.tik_instance.vmax(size_left, tmp_ub, ub_in[size_loop * MASK], tmp_ub, 1, 1, 1, 1, 0, 8, 0)
        for i in range(BLOCK_PROCESS_NUM):
            self.tik_instance.vmax(C_ZERO, tmp_ub, tmp_ub[16 + i * 16], tmp_ub, 1, 1, 1, 1, 8, 8, 8)

    def global_ub_store_all(self, core_idx, core_ele):
        """
        tiling mode when ub can store all h w c0
        """
        self.init_ub_global()
        self.vector_dup_ub()
        with self.tik_instance.for_range(0, core_ele) as core_ele_index:
            in_gm_offset = core_idx * self.one_core_ele * self.input_h * self.input_w * C_ZERO + \
                           core_ele_index * self.input_h * self.input_w * C_ZERO
            out_gm_offset = core_idx * self.one_core_ele * C_ZERO + core_ele_index * C_ZERO
            self.tik_instance.data_move(self.ub_in, self.input_gm[in_gm_offset], 0, 1, self.input_h * self.input_w, 0,
                                        0)
            self.get_hw_max(self.ub_tmp, self.ub_in, self.input_h * self.input_w * C_ZERO)
            self.tik_instance.vmax(C_ZERO, self.ub_out, self.ub_tmp, self.ub_out, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.data_move(self.output_gm[out_gm_offset], self.ub_out, 0, 1, 1, 0, 0)
            self.vector_dup_ub()

    def global_ub_tiling_hw(self, core_idx, core_ele, loop_num, loop_left):
        """
        tiling mode when ub can not store all h w c0, tiling dim h and w
        """
        self.init_ub_global()
        self.vector_dup_ub()
        with self.tik_instance.for_range(0, core_ele) as core_ele_index:
            in_gm_offset = core_idx * self.one_core_ele * self.input_h * self.input_w * C_ZERO + \
                           core_ele_index * self.input_h * self.input_w * C_ZERO
            out_gm_offset = core_idx * self.one_core_ele * C_ZERO + core_ele_index * C_ZERO
            with self.tik_instance.for_range(0, loop_num) as loop_num_index:
                in_gm_offset_tiling_hw = in_gm_offset + loop_num_index * self.h_factor * C_ZERO
                self.tik_instance.data_move(self.ub_in, self.input_gm[in_gm_offset_tiling_hw], 0, 1, self.h_factor, 0,
                                            0)
                self.get_hw_max(self.ub_tmp, self.ub_in, self.h_factor * C_ZERO)
                self.tik_instance.vmax(C_ZERO, self.ub_out, self.ub_tmp, self.ub_out, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vector_dup(MASK, self.ub_tmp, MIN_FP16, 1, 1, 8)
            with self.tik_instance.if_scope(loop_left > 0):
                in_gm_offset_tiling_hw = in_gm_offset + loop_num * self.h_factor * C_ZERO
                self.tik_instance.data_move(self.ub_in, self.input_gm[in_gm_offset_tiling_hw], 0, 1, loop_left, 0, 0)
                self.get_hw_max(self.ub_tmp, self.ub_in, loop_left * C_ZERO)
                self.tik_instance.vmax(C_ZERO, self.ub_out, self.ub_tmp, self.ub_out, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.data_move(self.output_gm[out_gm_offset], self.ub_out, 0, 1, 1, 0, 0)
            self.vector_dup_ub()

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
            self.init_ub_scalar()
            with self.tik_instance.if_scope(core_idx <= self.act_core_num - 1):
                with self.tik_instance.if_scope(core_idx < self.act_core_num - 1):
                    self.core_ele.set_as(self.one_core_ele)
                    self.loop_num.set_as(self.one_core_loop_num)
                    self.loop_left.set_as(self.one_core_loop_left)
                with self.tik_instance.if_scope(core_idx == self.act_core_num - 1):
                    self.core_ele.set_as(self.last_core_ele)
                    self.loop_num.set_as(self.last_core_loop_num)
                    self.loop_left.set_as(self.last_core_loop_left)
                # only move in and move out
                with self.tik_instance.if_scope(self.tiling_mode == 0):
                    with self.tik_instance.new_stmt_scope():
                        self.copy_only(core_idx, self.loop_num, self.loop_left)
                # tiling c1 dim when core num at nc1
                with self.tik_instance.if_scope(self.tiling_mode == 1):
                    with self.tik_instance.new_stmt_scope():
                        self.tiling_c_dim_core_nc(core_idx, self.loop_num, self.loop_left)
                # tiling h dim when core num at nc1
                with self.tik_instance.if_scope(self.tiling_mode == 2):
                    with self.tik_instance.new_stmt_scope():
                        self.tiling_h_dim_core_nc(core_idx, self.core_ele, self.loop_num, self.loop_left)
                # tiling w dim when core num at nc1
                with self.tik_instance.if_scope(self.tiling_mode == 3):
                    with self.tik_instance.new_stmt_scope():
                        self.tiling_w_dim_core_nc(core_idx, self.core_ele, self.loop_num, self.loop_left)
                # tiling h dim when core num at h
                with self.tik_instance.if_scope(self.tiling_mode == 4):
                    with self.tik_instance.new_stmt_scope():
                        self.tiling_h_dim_core_h(core_idx, self.loop_num, self.loop_left)
                # tiling w dim when core num at h
                with self.tik_instance.if_scope(self.tiling_mode == 5):
                    with self.tik_instance.new_stmt_scope():
                        self.tiling_w_dim_core_h(core_idx, self.core_ele, self.loop_num, self.loop_left)
                # global pooling, ub can store all hwc0
                with self.tik_instance.if_scope(self.tiling_mode == 6):
                    with self.tik_instance.new_stmt_scope():
                        self.global_ub_store_all(core_idx, self.core_ele)
                # global pooling ub can not store all hwc0
                with self.tik_instance.if_scope(self.tiling_mode == 7):
                    with self.tik_instance.new_stmt_scope():
                        self.global_ub_tiling_hw(core_idx, self.core_ele, self.loop_num, self.loop_left)

    def max_pool_operator(self):
        """Maxpool operator
        """
        self.max_pool_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": self.ub_ele,
                "core_num": self.core_num,
                "ksize_h": self.ksize_h,
                "ksize_w": self.ksize_w,
                "strides_h": self.strides_h,
                "strides_w": self.strides_w,
                "padding": self.padding,
                "ceil_mode": 0,
                "pad_top": 0,
                "pad_bottom": 0,
                "pad_left": 0,
                "pad_right": 0,
                "global": 0
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
