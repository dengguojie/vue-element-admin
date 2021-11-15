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
dynamic arg_max_v2
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # 'define a scalar, value = -(2**16 - 1)'
    SCALAR_MIN_FP16 = -(2**16 - 1)
    # 'define a scalar, value = -(2**32 - 1)'
    SCALAR_MIN_FP32 = -3402823424.0
    # max set_mask_int64 value
    MAX_MASK_INT64 = 2**64 - 1
    # max segment len
    MAX_SEGMENT_LEN = 2048 * 4
    # int32 num in 8*block
    OUT_MASK = 64
    # 0101 mask value
    MASK_0_1 = 6148914691236517205
    # max int32
    MAX_INT32 = 2**31 - 1
    # tiling arg num
    TILING_ARG_NUM = 24

def _get_ceil_int(int1, int2):
    """Get ceil int
    """
    return (int1 + int2 - 1) // int2


# 'pylint: disable=unused-argument,invalid-name,useless-object-inheritance,too-many-lines,too-many-arguments
# 'pylint: disable=too-many-statements,too-many-locals,attribute-defined-outside-init
# 'pylint: disable=too-many-instance-attributes,too-many-function-args
class Argmax():
    """Function: use to store argmax schedule parameters
    """

    def __init__(self, shape_x, dtype_x, kernel_name):
        """Init Argmax base parameters
        """
        # reserved ub size
        RESERVED_UB_SIZE = 8 * 1024
        self.tik_instance = tik.Tik()
        self.dtype_x = dtype_x
        self.kernel_name = kernel_name
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = tbe_platform.get_bit_len(self.dtype_x) // 8
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE) // self.dtype_size
        self.segment = Constant.MAX_SEGMENT_LEN  # only for arg at last dim
        self.data_each_block = 8
        self.data_each_vector = 64
        if dtype_x == "float16":
            self.segment = Constant.MAX_SEGMENT_LEN * 3  # only for arg at last dim
            self.data_each_block = 16
            self.data_each_vector = 128
        self.init_gm_tensor()

    def tiling_args(self):
        """get runtime params from tiling
        """
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.first_dim_size = self.tik_instance.Scalar("int32", name="first_dim_size")
        self.first_dim_size.set_as(self.tiling_ub[1])
        self.axis_size = self.tik_instance.Scalar("int32", name="axis_size")
        self.axis_size.set_as(self.tiling_ub[2])
        self.last_dim_size = self.tik_instance.Scalar("int32", name="last_dim_size")
        self.last_dim_size.set_as(self.tiling_ub[3])
        self.act_core_num = self.tik_instance.Scalar("int32", name="act_core_num")
        self.act_core_num.set_as(self.tiling_ub[4])
        self.one_core_ele = self.tik_instance.Scalar("int32", name="one_core_ele")
        self.one_core_ele.set_as(self.tiling_ub[5])
        self.last_core_ele = self.tik_instance.Scalar("int32", name="last_core_ele")
        self.last_core_ele.set_as(self.tiling_ub[6])
        self.align_num = self.tik_instance.Scalar("int32", name="align_num")
        self.align_num.set_as(self.tiling_ub[7])
        self.axis_size_one_time = self.tik_instance.Scalar("int32", name="axis_size_one_time")
        self.axis_size_one_time.set_as(self.tiling_ub[8])
        self.loop_times = self.tik_instance.Scalar("int32", name="loop_times")
        self.loop_times.set_as(self.tiling_ub[9])
        self.tail_size = self.tik_instance.Scalar("int32", name="tail_size")
        self.tail_size.set_as(self.tiling_ub[10])
        self.one_core_segment_loop = self.tik_instance.Scalar("int32", name="one_core_segment_loop")
        self.one_core_segment_loop.set_as(self.tiling_ub[11])
        self.one_core_segment_tail = self.tik_instance.Scalar("int32", name="one_core_segment_tail")
        self.one_core_segment_tail.set_as(self.tiling_ub[12])
        self.one_core_segment_tail_data = self.tik_instance.Scalar("int32", name="one_core_segment_tail_data")
        self.one_core_segment_tail_data.set_as(self.tiling_ub[13])
        self.one_core_offset = self.tik_instance.Scalar("int32", name="one_core_offset")
        self.one_core_offset.set_as(self.tiling_ub[14])
        self.last_core_segment_loop = self.tik_instance.Scalar("int32", name="last_core_segment_loop")
        self.last_core_segment_loop.set_as(self.tiling_ub[15])
        self.last_core_segment_tail = self.tik_instance.Scalar("int32", name="last_core_segment_tail")
        self.last_core_segment_tail.set_as(self.tiling_ub[16])
        self.last_core_segment_tail_data = self.tik_instance.Scalar("int32", name="last_core_segment_tail_data")
        self.last_core_segment_tail_data.set_as(self.tiling_ub[17])
        self.last_core_offset = self.tik_instance.Scalar("int32", name="last_core_offset")
        self.last_core_offset.set_as(self.tiling_ub[18])

        # other scalar
        self.core_ele = self.tik_instance.Scalar("int32", name="core_ele")
        self.segment_loop = self.tik_instance.Scalar("int32", name="segment_loop")
        self.segment_tail = self.tik_instance.Scalar("int32", name="segment_tail")
        self.segment_tail_data = self.tik_instance.Scalar("int32", name="segment_tail_data")
        self.offset_data = self.tik_instance.Scalar("int32", name="offset_data")

    def init_gm_tensor(self):
        """Init gm tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.data_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT32,), name="data_gm", scope=tik.scope_gm)
        self.axis_gm = self.tik_instance.Tensor("int32", (1,), name="axis_gm", scope=tik.scope_gm)
        self.result_gm = self.tik_instance.Tensor("int32", (Constant.MAX_INT32,), name="result_gm", scope=tik.scope_gm)

    def do_not_last(self, segment, gm_in_offset, gm_out_offset):
        """process for a segment when arg not last dim
        """
        ub_a = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,), name="ub_a", scope=tik.scope_ubuf)
        ub_c = self.tik_instance.Tensor("int32", (Constant.MAX_SEGMENT_LEN,), name="ub_c", scope=tik.scope_ubuf)

        # move in: ub_a
        nbust_len = _get_ceil_int(segment, self.data_each_block)
        self.tik_instance.data_move(ub_a, self.data_gm[gm_in_offset], 0, 1, nbust_len, 0, 0)
        # vector_dup zeros: ub_c
        int64_num = _get_ceil_int(segment, Constant.OUT_MASK)
        self.tik_instance.vector_dup(Constant.OUT_MASK, ub_c, 0, int64_num, 1, 8)

        repeat = _get_ceil_int(segment, self.data_each_vector)
        with self.tik_instance.for_range(1, self.axis_size, thread_num=2) as axis_i:
            ub_b = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,), name="ub_b", scope=tik.scope_ubuf)
            ub_mask = self.tik_instance.Tensor("uint64", (Constant.MAX_SEGMENT_LEN // Constant.OUT_MASK,),
                                               name="ub_mask",
                                               scope=tik.scope_ubuf)
            # move in: ub_b
            self.tik_instance.data_move(ub_b, self.data_gm[gm_in_offset + axis_i * self.last_dim_size], 0, 1, nbust_len,
                                        0, 0)
            # 'vcmpv_lt: ub_a and ub_b'
            self.tik_instance.vcmpv_lt(ub_mask, ub_a, ub_b, repeat, 1, 1, 8, 8)
            # vector dup axis_i: ub_c
            with self.tik_instance.for_range(0, int64_num) as i:
                mask_l = self.tik_instance.Scalar("uint64")
                mask_l.set_as(ub_mask[i])
                with self.tik_instance.if_scope(mask_l != 0):
                    self.tik_instance.vector_dup([mask_l, mask_l], ub_c[i * Constant.OUT_MASK], axis_i, 1, 1, 8)
            # 'vmax: ub_a and ub_b'
            self.tik_instance.vmax(self.data_each_vector, ub_a, ub_a, ub_b, repeat, 1, 1, 1, 8, 8, 8)

        # move out: ub_c
        nbust_len_out = _get_ceil_int(segment, 8)
        self.tik_instance.data_move(self.result_gm[gm_out_offset], ub_c, 0, 1, nbust_len_out, 0, 0)

    def do_not_last_fp16_default(self, segment, gm_in_offset, gm_out_offset):
        """process for a segment when arg not last dim for fp 16
        """
        ub_a = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,), name="ub_a", scope=tik.scope_ubuf)
        ub_c = self.tik_instance.Tensor("int32", (Constant.MAX_SEGMENT_LEN,), name="ub_c", scope=tik.scope_ubuf)
        ub_index = self.tik_instance.Tensor(self.dtype_x, (self.data_each_vector,),
                                            name="ub_index",
                                            scope=tik.scope_ubuf)
        ub_out_fp16 = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,),
                                               name="ub_out_fp16",
                                               scope=tik.scope_ubuf)

        # move in: ub_a
        nbust_len = _get_ceil_int(segment, self.data_each_block)
        self.tik_instance.data_move(ub_a, self.data_gm[gm_in_offset], 0, 1, nbust_len, 0, 0)
        # vector dup zeros: ub_index
        self.tik_instance.vector_dup(self.data_each_vector, ub_index, 0, 1, 1, 8)
        # vector dup zeros: ub_out_fp16
        repeat = _get_ceil_int(segment, self.data_each_vector)
        self.tik_instance.vector_dup(self.data_each_vector, ub_out_fp16, 0, repeat, 1, 8)
        # vector dup zeros: ub_c
        int64_num = _get_ceil_int(segment, Constant.OUT_MASK)
        self.tik_instance.vector_dup(Constant.OUT_MASK, ub_c, 0, int64_num, 1, 8)

        with self.tik_instance.for_range(1, self.axis_size, thread_num=2) as axis_i:
            ub_b = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,), name="ub_b", scope=tik.scope_ubuf)
            ub_mask = self.tik_instance.Tensor("uint64", (Constant.MAX_SEGMENT_LEN // Constant.OUT_MASK,),
                                               name="ub_mask",
                                               scope=tik.scope_ubuf)
            # move in: ub_b
            self.tik_instance.data_move(ub_b, self.data_gm[gm_in_offset + axis_i * self.last_dim_size], 0, 1, nbust_len,
                                        0, 0)
            # index add one
            self.tik_instance.vadds(self.data_each_vector, ub_index, ub_index, 1, 1, 1, 1, 8, 8)
            # vcmpv_lt: ub_a and ub_b, vec_sel: ub_out_fp16 and ub_mask and ub_index
            with self.tik_instance.for_range(0, repeat) as i:
                offset = i * self.data_each_vector
                self.tik_instance.vcmpv_lt(ub_mask, ub_a[offset], ub_b[offset], 1, 1, 1, 8, 8)
                self.tik_instance.vec_sel(self.data_each_vector, 0, ub_out_fp16[offset], ub_mask, ub_index,
                                          ub_out_fp16[offset], 1, 8, 0, 0)
            # 'vmax: ub_a and ub_b'
            self.tik_instance.vmax(self.data_each_vector, ub_a, ub_a, ub_b, repeat, 1, 1, 1, 8, 8, 8)

        # vec_conv: ub_out_fp16->ub_c
        self.tik_instance.vec_conv(64, "round", ub_c, ub_out_fp16, int64_num, 8, 4)
        # move out: ub_c
        nbust_len_out = _get_ceil_int(segment, 8)
        self.tik_instance.data_move(self.result_gm[gm_out_offset], ub_c, 0, 1, nbust_len_out, 0, 0)

    def do_not_last_fp16_align(self, segment, gm_in_offset, gm_out_offset):
        """process for a segment when arg not last dim for fp16 align
        """
        ub_a = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,), name="ub_a", scope=tik.scope_ubuf)
        ub_c = self.tik_instance.Tensor("int32", (Constant.MAX_SEGMENT_LEN,), name="ub_c", scope=tik.scope_ubuf)
        ub_index = self.tik_instance.Tensor(self.dtype_x, (self.data_each_vector,),
                                            name="ub_index",
                                            scope=tik.scope_ubuf)
        ub_out_fp16 = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,),
                                               name="ub_out_fp16",
                                               scope=tik.scope_ubuf)

        # vector dup fp16 min: ub_a
        repeat = _get_ceil_int(segment, self.data_each_vector)
        self.tik_instance.vector_dup(self.data_each_vector, ub_a, Constant.SCALAR_MIN_FP16, repeat, 1, 8)
        # vector dup -1: ub_index
        self.tik_instance.vector_dup(self.data_each_vector, ub_index, -1, 1, 1, 8)
        # vector dup zeros: ub_out_fp16
        self.tik_instance.vector_dup(self.data_each_vector, ub_out_fp16, 0, repeat, 1, 8)
        # vector dup zeros: ub_c
        int64_num = _get_ceil_int(segment, Constant.OUT_MASK)
        self.tik_instance.vector_dup(Constant.OUT_MASK, ub_c, 0, int64_num, 1, 8)

        # tiling at axis dim
        last_align = _get_ceil_int(segment, self.data_each_vector) * self.data_each_vector
        max_axis_len = Constant.MAX_SEGMENT_LEN // last_align
        axis_loop = self.axis_size // max_axis_len
        axis_tail = self.axis_size % max_axis_len
        nbust_len = _get_ceil_int(segment, self.data_each_block)
        dst_stride = _get_ceil_int(segment, self.data_each_vector) * self.data_each_vector - segment
        dst_stride = dst_stride // self.data_each_block

        def _run_one_sigment(axis_idx, axis_len):
            ub_b = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,), name="ub_b", scope=tik.scope_ubuf)
            ub_mask = self.tik_instance.Tensor("uint64", (Constant.MAX_SEGMENT_LEN // Constant.OUT_MASK,),
                                               name="ub_mask",
                                               scope=tik.scope_ubuf)
            # move in: ub_b
            self.tik_instance.data_move(ub_b, self.data_gm[gm_in_offset + axis_idx * segment], 0, axis_len, nbust_len,
                                        0, dst_stride)
            with self.tik_instance.for_range(0, axis_len) as axis_i:
                # index add one
                self.tik_instance.vadds(self.data_each_vector, ub_index, ub_index, 1, 1, 1, 1, 8, 8)
                # vcmpv_lt: ub_a and ub_b, vec_sel: ub_out_fp16 and ub_mask and ub_index
                _axis_offset = last_align * axis_i
                with self.tik_instance.for_range(0, repeat) as i:
                    offset = i * self.data_each_vector
                    self.tik_instance.vcmpv_lt(ub_mask, ub_a[offset], ub_b[offset + _axis_offset], 1, 1, 1, 8, 8)
                    self.tik_instance.vec_sel(self.data_each_vector, 0, ub_out_fp16[offset], ub_mask, ub_index,
                                              ub_out_fp16[offset], 1, 8, 0, 0)
                # 'vmax: ub_a and ub_b'
                self.tik_instance.vmax(self.data_each_vector, ub_a, ub_a, ub_b[_axis_offset], repeat, 1, 1, 1, 8, 8, 8)

        # run loop of axis dim
        with self.tik_instance.for_range(0, axis_loop, thread_num=2) as _axis_loop:
            input_axis_offset = _axis_loop * max_axis_len
            _run_one_sigment(input_axis_offset, max_axis_len)
        # run tail of axis dim
        with self.tik_instance.if_scope(axis_tail != 0):
            _run_one_sigment(axis_loop * max_axis_len, axis_tail)

        # vec_conv: ub_out_fp16->ub_c
        self.tik_instance.vec_conv(64, "round", ub_c, ub_out_fp16, int64_num, 8, 4)
        # move out: ub_c
        nbust_len_out = _get_ceil_int(segment, 8)
        self.tik_instance.data_move(self.result_gm[gm_out_offset], ub_c, 0, 1, nbust_len_out, 0, 0)

    def compute_argmax_not_last_axis_cut_by_first_dim(self, first_idx, segment_loop, segment_tail, segment_tail_data,
                                                      offset_data, func):
        """compute when cut by first_dim
        """
        not_last_axis_fuc = func
        # run loop of last dim
        with self.tik_instance.for_range(0, segment_loop) as segm_i:
            gm_in_offset = first_idx * self.axis_size * self.last_dim_size + segm_i * Constant.MAX_SEGMENT_LEN
            gm_out_offset = first_idx * self.last_dim_size + segm_i * Constant.MAX_SEGMENT_LEN
            not_last_axis_fuc(Constant.MAX_SEGMENT_LEN, gm_in_offset, gm_out_offset)
        # run tail of last dim
        with self.tik_instance.if_scope(segment_tail != 0):
            with self.tik_instance.if_scope(tik.all(segment_tail_data % 8 != 0, segment_tail_data > 8)):
                pro_len = _get_ceil_int(segment_tail_data, 2)
                pro_len = _get_ceil_int(pro_len, 8) * 8
                offset = segment_tail_data - pro_len
                gm_in_offset = first_idx * self.axis_size * self.last_dim_size + segment_loop * Constant.MAX_SEGMENT_LEN
                gm_out_offset = first_idx * self.last_dim_size + segment_loop * Constant.MAX_SEGMENT_LEN
                not_last_axis_fuc(pro_len, gm_in_offset, gm_out_offset)
                gm_in_offset = first_idx * self.axis_size * self.last_dim_size + segment_loop * Constant.MAX_SEGMENT_LEN + offset
                gm_out_offset = first_idx * self.last_dim_size + segment_loop * Constant.MAX_SEGMENT_LEN + offset
                not_last_axis_fuc(pro_len, gm_in_offset, gm_out_offset)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(segment_tail_data % 8 == 0):
                    gm_in_offset = first_idx * self.axis_size * self.last_dim_size + segment_loop * \
                                   Constant.MAX_SEGMENT_LEN + offset_data
                    gm_out_offset = first_idx * self.last_dim_size + segment_loop * Constant.MAX_SEGMENT_LEN + offset_data
                    not_last_axis_fuc(segment_tail_data, gm_in_offset, gm_out_offset)
                with self.tik_instance.else_scope():
                    gm_in_offset = first_idx * self.axis_size * self.last_dim_size + segment_loop * Constant.MAX_SEGMENT_LEN
                    gm_out_offset = first_idx * self.last_dim_size + segment_loop * Constant.MAX_SEGMENT_LEN
                    not_last_axis_fuc(segment_tail_data, gm_in_offset, gm_out_offset)

    def compute_argmax_not_last_axis_cut_by_last_dim(self, in_offset, out_offset, segment_loop, segment_tail,
                                                     segment_tail_data, offset_data, func):
        """compute when cut by last_dim
        """
        not_last_axis_fuc = func
        # run loop of last dim
        with self.tik_instance.for_range(0, segment_loop) as segm_i:
            gm_in_offset = in_offset + Constant.MAX_SEGMENT_LEN * segm_i
            gm_out_offset = out_offset + Constant.MAX_SEGMENT_LEN * segm_i
            not_last_axis_fuc(Constant.MAX_SEGMENT_LEN, gm_in_offset, gm_out_offset)
        # run tail of last dim
        with self.tik_instance.if_scope(segment_tail != 0):
            gm_in_offset = in_offset + Constant.MAX_SEGMENT_LEN * segment_loop - offset_data
            gm_out_offset = out_offset + Constant.MAX_SEGMENT_LEN * segment_loop - offset_data
            not_last_axis_fuc(segment_tail_data, gm_in_offset, gm_out_offset)

    def compute_argmax_last_axis_fp16_less_vector(self, core_idx, segment_loop, segment_tail):
        """compute_argmax_last_axis_fp16_less_vector
        """

        self.ub_result_int32 = self.tik_instance.Tensor("int32", (Constant.MAX_SEGMENT_LEN,),
                                                        name="ub_result_int32",
                                                        scope=tik.scope_ubuf)

        def do_argmax_last_axis_fp16_less_vector(first_idx, segment_len):
            """do_argmax_last_axis_fp16_less_vector
            """
            segment_align_num = segment_len // self.align_num
            segment_align_tail = segment_len % self.align_num

            with self.tik_instance.for_range(0, self.align_num, thread_num=2) as align_idx:
                ub_data = self.tik_instance.Tensor(self.dtype_x, (self.segment,), name="ub_data", scope=tik.scope_ubuf)
                nburst = self.tik_instance.Scalar("uint16")
                nburst_len = self.tik_instance.Scalar("uint16")
                repeat_block = self.tik_instance.Scalar("uint16")
                repeat_vcmax = self.tik_instance.Scalar("uint16")
                with self.tik_instance.if_scope(align_idx < segment_align_tail):
                    nburst.set_as(segment_align_num + 1)
                with self.tik_instance.else_scope():
                    nburst.set_as(segment_align_num)
                nburst_len.set_as(_get_ceil_int(self.axis_size, self.data_each_block))
                repeat_block.set_as(nburst_len)
                repeat_vcmax.set_as(nburst)
                src_nburst_stride = _get_ceil_int(self.axis_size * self.align_num, self.data_each_block) - nburst_len
                des_nburst_stride = 0
                with self.tik_instance.if_scope(self.align_num == 1):
                    nburst_len.set_as(nburst_len * nburst)
                    nburst.set_as(1)
                    repeat_vcmax.set_as(segment_len)

                with self.tik_instance.if_scope(nburst >= 1):
                    gm_in_offset = (first_idx + align_idx) * self.axis_size
                    self.tik_instance.data_move(ub_data, self.data_gm[gm_in_offset], 0, nburst, nburst_len,
                                                src_nburst_stride, des_nburst_stride)
                    # get one axis vcmax
                    self.tik_instance.vcmax(self.axis_size, ub_data, ub_data, repeat_vcmax, 1, 1, repeat_block)
                    with self.tik_instance.for_range(0, repeat_vcmax) as _vcmax_idx:
                        last_max_index = self.tik_instance.Scalar("uint16")
                        last_max_index.set_as(ub_data[_vcmax_idx * 2 + 1])
                        self.ub_result_int32[_vcmax_idx * self.align_num + align_idx].set_as(last_max_index)

        def _run(segment_len, segment_index):
            """run function
            """
            # move in and calc
            first_idx = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            do_argmax_last_axis_fp16_less_vector(first_idx, segment_len)
            # move out
            gm_out_offset = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            out_nbust = _get_ceil_int(segment_len, 8)
            self.tik_instance.data_move(self.result_gm[gm_out_offset], self.ub_result_int32, 0, 1, out_nbust, 0, 0)

        # run loop
        with self.tik_instance.for_range(0, segment_loop) as _loop:
            _run(self.axis_size_one_time, _loop)
        # run tail
        with self.tik_instance.if_scope(segment_tail != 0):
            _run(segment_tail, segment_loop)

    def compute_argmax_last_axis_fp16_more_vector(self, core_idx, segment_loop, segment_tail):
        """compute_argmax_last_axis_fp16_more_vector
        """
        self.ub_result_int32 = self.tik_instance.Tensor("int32", (Constant.MAX_SEGMENT_LEN,),
                                                        name="ub_result_int32",
                                                        scope=tik.scope_ubuf)

        def do_argmax_last_axis_fp16_more_vector(first_idx, segment_len):
            """do_argmax_last_axis_fp16_more_vector
            """
            vector_size_repeat = _get_ceil_int(self.axis_size, self.data_each_vector)
            vector_size = vector_size_repeat * self.data_each_vector
            result_int32 = self.tik_instance.Scalar("int32")

            segment_align_num = segment_len // self.align_num
            segment_align_tail = segment_len % self.align_num

            mask_h = self.tik_instance.Scalar("int64")
            mask_l = self.tik_instance.Scalar("int64")

            def _calu_mask_by_repeat_times(_len, _mask_h, _mask_l):
                """_calu_mask_by_repeat_times
                """
                tmp = self.tik_instance.Scalar("uint32")
                with self.tik_instance.if_scope(_len >= 8):
                    _mask_h.set_as(281479271743489)
                    _mask_l.set_as(281479271743489)
                with self.tik_instance.else_scope():
                    _mask_h.set_as(0)
                    with self.tik_instance.if_scope(_len >= 4):
                        _mask_l.set_as(281479271743489)
                        with self.tik_instance.for_range(0, _len % 4) as i:
                            tmp.set_as(1)
                            with self.tik_instance.for_range(0, i):
                                tmp.set_as(65536 * tmp)
                            _mask_h.set_as(_mask_h + tmp)
                    with self.tik_instance.else_scope():
                        _mask_l.set_as(0)
                        with self.tik_instance.for_range(0, _len) as i:
                            tmp.set_as(1)
                            with self.tik_instance.for_range(0, i):
                                tmp.set_as(65536 * tmp)
                            _mask_l.set_as(_mask_l + tmp)

            def _get_tail_mask(tail_len, _mask_h, _mask_l):
                """get_tail_mask
                """
                mask = self.tik_instance.Scalar("int64", name="mask")
                mask.set_as(1)
                with self.tik_instance.if_scope(tail_len <= Constant.OUT_MASK):
                    with self.tik_instance.for_range(0, tail_len):
                        mask.set_as(2 * mask)
                    mask.set_as(mask - 1)
                    _mask_h.set_as(Constant.MAX_MASK_INT64)
                    _mask_l.set_as(Constant.MAX_MASK_INT64 - mask)
                with self.tik_instance.else_scope():
                    _mask_l.set_as(0)
                    with self.tik_instance.for_range(0, tail_len - Constant.OUT_MASK):
                        mask.set_as(2 * mask)
                    mask.set_as(mask - 1)
                    _mask_h.set_as(Constant.MAX_MASK_INT64 - mask)

            with self.tik_instance.for_range(0, self.align_num, thread_num=2) as align_idx:
                ub_data = self.tik_instance.Tensor(self.dtype_x, (self.segment,), name="ub_data", scope=tik.scope_ubuf)
                ub_second_result = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,),
                                                            name="ub_second_result",
                                                            scope=tik.scope_ubuf)
                nburst_len = _get_ceil_int(self.axis_size, self.data_each_block)
                nburst = self.tik_instance.Scalar("uint32")
                with self.tik_instance.if_scope(align_idx < segment_align_tail):
                    nburst.set_as(segment_align_num + 1)
                with self.tik_instance.else_scope():
                    nburst.set_as(segment_align_num)

                with self.tik_instance.if_scope(nburst >= 1):
                    src_nburst_stride = _get_ceil_int(self.axis_size * self.align_num,
                                                      self.data_each_block) - nburst_len
                    des_nburst_stride = vector_size_repeat * 8 - nburst_len
                    gm_in_offset = (first_idx + align_idx) * self.axis_size
                    self.tik_instance.data_move(ub_data, self.data_gm[gm_in_offset], 0, nburst, nburst_len,
                                                src_nburst_stride, des_nburst_stride)

                    tail = self.axis_size % self.data_each_vector
                    with self.tik_instance.if_scope(tail != 0):
                        _get_tail_mask(tail, mask_h, mask_l)
                        _offset = self.axis_size // self.data_each_vector
                        self.tik_instance.vector_dup([mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                                                     Constant.SCALAR_MIN_FP16, nburst, 1, vector_size_repeat * 8)
                    repeat_times = _get_ceil_int(self.axis_size, self.data_each_vector)
                    self.tik_instance.vcmax(self.data_each_vector, ub_data, ub_data, vector_size_repeat * nburst, 8 * 8,
                                            1, 8)
                    _calu_mask_by_repeat_times(repeat_times, mask_h, mask_l)
                    self.tik_instance.vcmax([mask_h, mask_l], ub_second_result, ub_data, nburst, 1, 8, repeat_times * 8)
                    with self.tik_instance.for_range(0, nburst) as out_idx:
                        second_max_index = self.tik_instance.Scalar("uint16")
                        second_max_index.set_as(ub_second_result[out_idx * 2 + 1])
                        last_max_index = self.tik_instance.Scalar("uint16")
                        last_max_index.set_as(ub_data[vector_size * out_idx + second_max_index * 8 + 1])
                        result_int32.set_as(second_max_index * 8 + last_max_index)
                        self.ub_result_int32[out_idx * self.align_num + align_idx].set_as(result_int32)

        def _run(segment_len, segment_index):
            """run function
            """
            # move in and calc
            first_idx = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            do_argmax_last_axis_fp16_more_vector(first_idx, segment_len)
            # move out
            gm_out_offset = core_idx * self.one_core_ele + self.axis_size_one_time * segment_index
            out_nbust = _get_ceil_int(segment_len, 8)
            self.tik_instance.data_move(self.result_gm[gm_out_offset], self.ub_result_int32, 0, 1, out_nbust, 0, 0)

        # run loop
        with self.tik_instance.for_range(0, segment_loop) as _loop:
            _run(self.axis_size_one_time, _loop)
        # run tail
        with self.tik_instance.if_scope(segment_tail != 0):
            _run(segment_tail, segment_loop)

    def compute_argmax_last_axis_fp16_copy_one_time(self, core_idx, segment_loop, segment_tail):
        """compute_argmax_last_axis_fp16_copy_one_time
        """
        self.ub_result_int32 = self.tik_instance.Tensor("int32", (Constant.MAX_SEGMENT_LEN,),
                                                        name="ub_result_int32",
                                                        scope=tik.scope_ubuf)

        def do_argmax_last_axis_fp16_copy_one_time(first_idx):
            """do_argmax_last_axis_fp16_copy_one_time
            """
            ub_result = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,),
                                                 name="ub_result",
                                                 scope=tik.scope_ubuf)
            ub_data = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_SEGMENT_LEN,), name="ub_data", scope=tik.scope_ubuf)
            # move in
            gm_in_offset = first_idx * self.axis_size
            self.tik_instance.data_move(ub_data, self.data_gm[gm_in_offset], 0, 1,
                                        _get_ceil_int(self.axis_size, self.data_each_block), 0, 0)

            def _calu_mask_by_one_zero(_len, _mask_h, _mask_l):
                """calu_mask_by_one_zero
                """
                _mask_h.set_as(0)
                _mask_l.set_as(0)
                tmp = self.tik_instance.Scalar("int64", name="tmp")
                tmp.set_as(1)
                with self.tik_instance.if_scope(_len > 32):
                    _mask_l.set_as(Constant.MASK_0_1)
                    with self.tik_instance.for_range(0, _len - 32) as i:
                        tmp.set_as(1)
                        with self.tik_instance.for_range(0, i):
                            tmp.set_as(tmp * 2 * 2)
                        _mask_h.set_as(_mask_h + tmp)
                with self.tik_instance.else_scope():
                    _mask_h.set_as(0)
                    with self.tik_instance.for_range(0, _len) as i:
                        tmp.set_as(1)
                        with self.tik_instance.for_range(0, i):
                            tmp.set_as(tmp * 2 * 2)
                        _mask_l.set_as(_mask_l + tmp)

            def _get_tail_mask(tail_len, _mask_h, _mask_l):
                """get_tail_mask
                """
                mask = self.tik_instance.Scalar("int64", name="mask")
                mask.set_as(1)
                with self.tik_instance.if_scope(tail_len <= Constant.OUT_MASK):
                    with self.tik_instance.for_range(0, tail_len):
                        mask.set_as(2 * mask)
                    mask.set_as(mask - 1)
                    _mask_h.set_as(Constant.MAX_MASK_INT64)
                    _mask_l.set_as(Constant.MAX_MASK_INT64 - mask)
                with self.tik_instance.else_scope():
                    _mask_l.set_as(0)
                    with self.tik_instance.for_range(0, tail_len - Constant.OUT_MASK):
                        mask.set_as(2 * mask)
                    mask.set_as(mask - 1)
                    _mask_h.set_as(Constant.MAX_MASK_INT64 - mask)

            mask_h = self.tik_instance.Scalar("int64")
            mask_l = self.tik_instance.Scalar("int64")

            # vector_dup
            tail = self.axis_size % self.data_each_vector
            with self.tik_instance.if_scope(tail != 0):
                _get_tail_mask(tail, mask_h, mask_l)
                _offset = self.axis_size // self.data_each_vector
                self.tik_instance.vector_dup([mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                                             Constant.SCALAR_MIN_FP16, 1, 1, 8)

            # vcmax
            repeat_times = _get_ceil_int(self.axis_size, self.data_each_vector)
            self.tik_instance.vcmax(self.data_each_vector, ub_result, ub_data, repeat_times, 1, 1, 8)

            # repeat_times at 2~64
            _calu_mask_by_one_zero(repeat_times, mask_h, mask_l)
            ub_second_result = self.tik_instance.Tensor(self.dtype_x, (self.data_each_vector,),
                                                        name="ub_second_result",
                                                        scope=tik.scope_ubuf)
            self.tik_instance.vcmax([mask_h, mask_l], ub_second_result, ub_result, 1, 1, 1, 8)
            second_max_index = self.tik_instance.Scalar("uint16")
            second_max_index.set_as(ub_second_result[1])
            last_max_index = self.tik_instance.Scalar("uint16")
            last_max_index.set_as(ub_result[second_max_index + 1])
            self.result_int32.set_as(second_max_index * 64 + last_max_index)

        def _run(segment_len, segment_index):
            """run function
            """
            # move in and calc
            with self.tik_instance.for_range(0, segment_len) as idx:
                first_idx = core_idx * self.one_core_ele + idx + Constant.MAX_SEGMENT_LEN * segment_index
                self.result_int32 = self.tik_instance.Scalar("int32")
                do_argmax_last_axis_fp16_copy_one_time(first_idx)
                self.ub_result_int32[idx] = self.result_int32

            # move out
            gm_out_offset = core_idx * self.one_core_ele + Constant.MAX_SEGMENT_LEN * segment_index
            out_nbust = _get_ceil_int(segment_len, 8)
            self.tik_instance.data_move(self.result_gm[gm_out_offset], self.ub_result_int32, 0, 1, out_nbust, 0, 0)

        # run loop
        with self.tik_instance.for_range(0, segment_loop) as _loop:
            _run(Constant.MAX_SEGMENT_LEN, _loop)
        # run tail
        with self.tik_instance.if_scope(segment_tail != 0):
            _run(segment_tail, segment_loop)

    def compute_argmax_last_axis_fp16(self, core_idx, segment_loop, segment_tail):
        """compute_argmax_last_axis_fp16
        """
        self.ub_result_int32 = self.tik_instance.Tensor("int32", (Constant.MAX_SEGMENT_LEN,),
                                                        name="ub_result_int32",
                                                        scope=tik.scope_ubuf)

        def do_argmax_last_axis_fp16(segment, loop, first_idx):
            """do_argmax_last_axis_fp16
            """
            segment_size = self.tik_instance.Scalar("int32")
            segment_size.set_as(segment)
            ub_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(_get_ceil_int(self.segment, self.data_each_vector) * 2, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_result",
                scope=tik.scope_ubuf)
            ub_index_int32 = self.tik_instance.Tensor("int32", (16,), name="ub_index_int32", scope=tik.scope_ubuf)
            ub_data = self.tik_instance.Tensor(
                self.dtype_x, (_get_ceil_int(self.segment, self.data_each_vector) * self.data_each_vector,),
                name="ub_data",
                scope=tik.scope_ubuf)
            # move in
            gm_in_offset = loop * self.segment + first_idx * self.axis_size
            self.tik_instance.data_move(ub_data, self.data_gm[gm_in_offset], 0, 1,
                                        _get_ceil_int(segment_size, self.data_each_block), 0, 0)

            def _calu_mask_by_one_zero(_len, _mask_h, _mask_l):
                """calu_mask_by_one_zero
                """
                _mask_h.set_as(0)
                _mask_l.set_as(0)
                tmp = self.tik_instance.Scalar("int64", name="tmp")
                tmp.set_as(1)
                with self.tik_instance.if_scope(_len > 32):
                    _mask_l.set_as(Constant.MASK_0_1)
                    with self.tik_instance.for_range(0, _len - 32) as i:
                        tmp.set_as(1)
                        with self.tik_instance.for_range(0, i):
                            tmp.set_as(tmp * 2 * 2)
                        _mask_h.set_as(_mask_h + tmp)
                with self.tik_instance.else_scope():
                    _mask_h.set_as(0)
                    with self.tik_instance.for_range(0, _len) as i:
                        tmp.set_as(1)
                        with self.tik_instance.for_range(0, i):
                            tmp.set_as(tmp * 2 * 2)
                        _mask_l.set_as(_mask_l + tmp)

            def _get_tail_mask(tail_len, _mask_h, _mask_l):
                """get_tail_mask
                """
                mask = self.tik_instance.Scalar("int64", name="mask")
                mask.set_as(1)
                with self.tik_instance.if_scope(tail_len <= Constant.OUT_MASK):
                    with self.tik_instance.for_range(0, tail_len):
                        mask.set_as(2 * mask)
                    mask.set_as(mask - 1)
                    _mask_h.set_as(Constant.MAX_MASK_INT64)
                    _mask_l.set_as(Constant.MAX_MASK_INT64 - mask)
                with self.tik_instance.else_scope():
                    _mask_l.set_as(0)
                    with self.tik_instance.for_range(0, tail_len - Constant.OUT_MASK):
                        mask.set_as(2 * mask)
                    mask.set_as(mask - 1)
                    _mask_h.set_as(Constant.MAX_MASK_INT64 - mask)

            mask_h = self.tik_instance.Scalar("int64")
            mask_l = self.tik_instance.Scalar("int64")
            # vector dup at ub_data
            tail = segment_size % self.data_each_vector
            with self.tik_instance.if_scope(tail != 0):
                _get_tail_mask(tail, mask_h, mask_l)
                _offset = segment_size // self.data_each_vector
                self.tik_instance.vector_dup([mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                                             Constant.SCALAR_MIN_FP16, 1, 1, 8)

            repeat_times = _get_ceil_int(segment_size, self.data_each_vector)
            # vcmax
            self.tik_instance.vcmax(self.data_each_vector, ub_result, ub_data, repeat_times, 1, 1, 8)

            max_index = self.tik_instance.Scalar("uint16")
            with self.tik_instance.if_scope(repeat_times >= 64):
                _repeat_times = _get_ceil_int(repeat_times, 64)
                _repeat_tail = (repeat_times * 2) % self.data_each_vector
                with self.tik_instance.if_scope(_repeat_tail != 0):
                    _get_tail_mask(_repeat_tail, mask_h, mask_l)
                    _offset = repeat_times * 2 // self.data_each_vector
                    self.tik_instance.vector_dup([mask_h, mask_l], ub_result[_offset * self.data_each_vector],
                                                 Constant.SCALAR_MIN_FP16, 1, 1, 8)
                ub_second_result = self.tik_instance.Tensor(self.dtype_x, (self.data_each_vector,),
                                                            name="ub_second_result",
                                                            scope=tik.scope_ubuf)
                self.tik_instance.vcmax([Constant.MASK_0_1, Constant.MASK_0_1], ub_second_result, ub_result, _repeat_times, 1, 1, 8)

                ub_third_result = self.tik_instance.Tensor(self.dtype_x, (self.data_each_vector,),
                                                           name="ub_third_result",
                                                           scope=tik.scope_ubuf)

                _calu_mask_by_one_zero(_repeat_times % 64, mask_h, mask_l)
                self.tik_instance.vcmax([mask_h, mask_l], ub_third_result, ub_second_result, 1, 1, 1, 8)
                third_max_index = self.tik_instance.Scalar("uint16")
                third_max_index.set_as(ub_third_result[1])
                second_max_index = self.tik_instance.Scalar("uint16")
                second_max_index.set_as(ub_second_result[third_max_index + 1])
                last_max_index = self.tik_instance.Scalar("uint16")
                last_max_index.set_as(ub_result[third_max_index * 64 + second_max_index + 1])
                max_index.set_as(third_max_index * 64 * 64 + second_max_index * 64 + last_max_index)
            with self.tik_instance.else_scope():
                _calu_mask_by_one_zero(repeat_times % 64, mask_h, mask_l)
                ub_second_result = self.tik_instance.Tensor(self.dtype_x, (self.data_each_vector,),
                                                            name="ub_second_result",
                                                            scope=tik.scope_ubuf)
                self.tik_instance.vcmax([mask_h, mask_l], ub_second_result, ub_result, 1, 1, 1, 8)
                second_max_index = self.tik_instance.Scalar("uint16")
                second_max_index.set_as(ub_second_result[1])
                last_max_index = self.tik_instance.Scalar("uint16")
                last_max_index.set_as(ub_result[second_max_index + 1])
                max_index.set_as(second_max_index * 64 + last_max_index)

            max_index_int32 = self.tik_instance.Scalar("int32")
            max_index_int32.set_as(max_index)
            ub_result_cmp = self.tik_instance.Tensor(self.dtype_x, (self.data_each_vector,),
                                                     name="ub_result_cmp",
                                                     scope=tik.scope_ubuf)
            ub_result_cmp[0].set_as(self.result_out_scalar)
            ub_result_cmp[1].set_as(ub_data[max_index_int32])
            ub_index_int32[0].set_as(self.result_int32)
            ub_index_int32[1].set_as(max_index_int32 + loop * self.segment)
            self.tik_instance.vcmax(2, ub_result_cmp, ub_result_cmp, 1, 1, 1, 8)
            max_index1 = self.tik_instance.Scalar("uint16")
            max_index1.set_as(ub_result_cmp[1])
            self.result_int32.set_as(ub_index_int32[max_index1])
            self.result_out_scalar.set_as(ub_result_cmp[0])

        def _run(segment_len, segment_index):
            """run function
            """
            with self.tik_instance.for_range(0, segment_len) as idx:
                first_idx = core_idx * self.one_core_ele + idx + Constant.MAX_SEGMENT_LEN * segment_index
                self.result_int32 = self.tik_instance.Scalar("int32")
                self.result_int32.set_as(0)
                self.result_out_scalar = self.tik_instance.Scalar(self.dtype_x)
                self.result_out_scalar.set_as(Constant.SCALAR_MIN_FP16)
                # run loop
                with self.tik_instance.for_range(0, self.loop_times, thread_num=2) as loop:
                    do_argmax_last_axis_fp16(self.segment, loop, first_idx)
                # run tail
                with self.tik_instance.if_scope(self.tail_size != 0):
                    do_argmax_last_axis_fp16(self.tail_size, self.loop_times, first_idx)
                self.ub_result_int32[idx] = self.result_int32
            # move out
            gm_out_offset = core_idx * self.one_core_ele + Constant.MAX_SEGMENT_LEN * segment_index
            out_nbust = _get_ceil_int(segment_len, 8)
            self.tik_instance.data_move(self.result_gm[gm_out_offset], self.ub_result_int32, 0, 1, out_nbust, 0, 0)

        # run loop
        with self.tik_instance.for_range(0, segment_loop) as _loop:
            _run(Constant.MAX_SEGMENT_LEN, _loop)
        # run tail
        with self.tik_instance.if_scope(segment_tail != 0):
            _run(segment_tail, segment_loop)

    def compute_argmax_last_axis_fp32(self, core_idx, segment_loop, segment_tail, is_loop=True):
        """compute_argmax_last_axis_fp32
        """

        def do_argmax_last_axis_fp32(segment_size, loop, first_idx):
            """do_argmax_last_axis_fp32
            """
            # define ub
            ub_data = self.tik_instance.Tensor(self.dtype_x, (self.segment,), name="ub_data", scope=tik.scope_ubuf)
            ub_max_64 = self.tik_instance.Tensor(self.dtype_x, (self.data_each_vector,),
                                                 name="ub_max_64",
                                                 scope=tik.scope_ubuf)
            ub_idx_64 = self.tik_instance.Tensor("int32", (self.data_each_vector,),
                                                 name="ub_idx_64",
                                                 scope=tik.scope_ubuf)
            ub_mask = self.tik_instance.Tensor("uint64", (self.segment // Constant.OUT_MASK,),
                                               name="ub_mask",
                                               scope=tik.scope_ubuf)

            # move in: ub_data
            gm_in_offset = loop * self.segment + first_idx * self.axis_size
            nbust = _get_ceil_int(segment_size, self.data_each_block)
            self.tik_instance.data_move(ub_data, self.data_gm[gm_in_offset], 0, 1, nbust, 0, 0)
            # vector dup fp32 min: ub_data(align data_each_vector)
            tail = segment_size % self.data_each_vector
            if not isinstance(tail, int) or tail != 0:
                mask = self.tik_instance.Scalar("uint64")
                mask.set_as(1)
                with self.tik_instance.for_range(0, tail):
                    mask.set_as(2 * mask)
                mask.set_as(mask - 1)
                mask_h = self.tik_instance.Scalar("uint64")
                mask_h.set_as(0)
                mask_l = self.tik_instance.Scalar("uint64")
                mask_l.set_as(Constant.MAX_MASK_INT64 - mask)
                _offset = segment_size // self.data_each_vector
                self.tik_instance.vector_dup([mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                                             Constant.SCALAR_MIN_FP32, 1, 1, 8)
            # vector dup fp32 min: ub_max_64
            self.tik_instance.vector_dup(self.data_each_vector, ub_max_64, Constant.SCALAR_MIN_FP32, 1, 1, 8)
            # vmax and vcmpv_eq
            repeat = _get_ceil_int(segment_size, self.data_each_vector)
            self.tik_instance.vmax(self.data_each_vector, ub_max_64, ub_data, ub_max_64, repeat, 1, 1, 1, 0, 8, 0)
            self.tik_instance.vcmpv_eq(ub_mask, ub_max_64, ub_data, repeat, 1, 1, 0, 8)
            # vector dup zeros and index: ub_idx_64
            self.tik_instance.vector_dup(self.data_each_vector, ub_idx_64, 0, 1, 1, 8)
            with self.tik_instance.for_range(0, repeat) as i:
                index = repeat - 1 - i
                mask_l = self.tik_instance.Scalar("uint64")
                mask_l.set_as(ub_mask[index])
                with self.tik_instance.if_scope(mask_l != 0):
                    self.tik_instance.vector_dup([mask_l, mask_l], ub_idx_64, index * self.data_each_vector, 1, 1, 8)

            # get first max value
            max_value = self.tik_instance.Scalar(self.dtype_x)
            max_index = self.tik_instance.Scalar("int32")
            max_value.set_as(ub_max_64[0])
            max_index.set_as(ub_idx_64[0])
            scalar_valid = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(segment_size > self.data_each_vector):
                scalar_valid.set_as(self.data_each_vector)
            with self.tik_instance.else_scope():
                scalar_valid.set_as(segment_size)
            with self.tik_instance.for_range(1, scalar_valid) as i:
                max_cmp_value = self.tik_instance.Scalar(self.dtype_x)
                max_cmp_index = self.tik_instance.Scalar("int32")
                max_cmp_value.set_as(ub_max_64[i])
                max_cmp_index.set_as(ub_idx_64[i])
                with self.tik_instance.if_scope(max_cmp_value > max_value):
                    max_value.set_as(ub_max_64[i])
                    max_index.set_as(max_cmp_index + i)
                with self.tik_instance.if_scope(tik.all(max_cmp_value == max_value, max_cmp_index + i < max_index)):
                    max_value.set_as(ub_max_64[i])
                    max_index.set_as(max_cmp_index + i)
            with self.tik_instance.if_scope(max_value > self.result_fp32):
                self.result_fp32.set_as(max_value)
                self.result_int32.set_as(max_index + loop * self.segment)

        def _run(segment_len, segment_index):
            """run function
            """
            with self.tik_instance.for_range(0, segment_len) as idx:
                self.result_int32 = self.tik_instance.Scalar("int32")
                self.result_int32.set_as(0)
                self.result_fp32 = self.tik_instance.Scalar(self.dtype_x)
                self.result_fp32.set_as(Constant.SCALAR_MIN_FP32)

                first_idx = core_idx * self.one_core_ele + idx + Constant.MAX_SEGMENT_LEN * segment_index
                if is_loop:
                    # run loop of last dim
                    with self.tik_instance.for_range(0, self.loop_times, thread_num=2) as loop:
                        do_argmax_last_axis_fp32(self.segment, loop, first_idx)
                    # run tail of last dim
                    with self.tik_instance.if_scope(self.tail_size != 0):
                        do_argmax_last_axis_fp32(self.tail_size, self.loop_times, first_idx)
                else:
                    # run tail of last dim
                    do_argmax_last_axis_fp32(self.tail_size, self.loop_times, first_idx)
                self.ub_result_int32[idx] = self.result_int32

            # move out
            gm_out_offset = core_idx * self.one_core_ele + Constant.MAX_SEGMENT_LEN * segment_index
            out_nbust = _get_ceil_int(segment_len, 8)
            self.tik_instance.data_move(self.result_gm[gm_out_offset], self.ub_result_int32, 0, 1, out_nbust, 0, 0)

        # define idx res ub
        self.ub_result_int32 = self.tik_instance.Tensor("int32", (Constant.MAX_SEGMENT_LEN,),
                                                        name="ub_result_int32",
                                                        scope=tik.scope_ubuf)
        # run loop of first dim
        with self.tik_instance.for_range(0, segment_loop) as _loop:
            _run(Constant.MAX_SEGMENT_LEN, _loop)
        # run tail of first dim
        with self.tik_instance.if_scope(segment_tail != 0):
            _run(segment_tail, segment_loop)

    def argmax_compute_tiling(self):
        """argmax_compute_tiling
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_idx:
            # define tiling ub and move tiling gm to tiling ub,then get tiling args
            self.tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                      name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 3, 0, 0)
            self.tiling_args()

            with self.tik_instance.if_scope(core_idx <= self.act_core_num - 1):
                with self.tik_instance.if_scope(core_idx < self.act_core_num - 1):
                    self.core_ele.set_as(self.one_core_ele)
                    self.segment_loop.set_as(self.one_core_segment_loop)
                    self.segment_tail.set_as(self.one_core_segment_tail)
                    self.segment_tail_data.set_as(self.one_core_segment_tail_data)
                    self.offset_data.set_as(self.one_core_offset)
                with self.tik_instance.else_scope():
                    self.core_ele.set_as(self.last_core_ele)
                    self.segment_loop.set_as(self.last_core_segment_loop)
                    self.segment_tail.set_as(self.last_core_segment_tail)
                    self.segment_tail_data.set_as(self.last_core_segment_tail_data)
                    self.offset_data.set_as(self.last_core_offset)

                if self.dtype_x == "float32":
                    # arg last dim:fp32:calc core at first dim and tiling at last dim
                    with self.tik_instance.if_scope(self.tiling_mode == 4):
                        with self.tik_instance.new_stmt_scope():
                            self.compute_argmax_last_axis_fp32(core_idx, self.segment_loop, self.segment_tail, True)
                    # arg last dim:fp32:calc core at first dim and tiling at last dim and no loop
                    with self.tik_instance.if_scope(self.tiling_mode == 12):
                        with self.tik_instance.new_stmt_scope():
                            self.compute_argmax_last_axis_fp32(core_idx, self.segment_loop, self.segment_tail, False)
                    # arg not last dim:fp32:calc core at first dim and tiling at last dim
                    with self.tik_instance.if_scope(self.tiling_mode == 5):
                        with self.tik_instance.new_stmt_scope():
                            with self.tik_instance.for_range(0, self.core_ele) as ele_idx:
                                first_idx = core_idx * self.one_core_ele + ele_idx
                                self.compute_argmax_not_last_axis_cut_by_first_dim(first_idx, self.segment_loop,
                                                                                   self.segment_tail,
                                                                                   self.segment_tail_data,
                                                                                   self.offset_data, self.do_not_last)
                    # arg not last dim:fp32:calc core at last dim and tiling at last dim
                    with self.tik_instance.if_scope(self.tiling_mode == 8):
                        with self.tik_instance.new_stmt_scope():
                            with self.tik_instance.for_range(0, self.first_dim_size) as ele_idx:
                                offset_in = ele_idx * self.axis_size * self.last_dim_size + core_idx * self.one_core_ele
                                offset_out = ele_idx * self.last_dim_size + core_idx * self.one_core_ele
                                self.compute_argmax_not_last_axis_cut_by_last_dim(offset_in, offset_out,
                                                                                  self.segment_loop, self.segment_tail,
                                                                                  self.segment_tail_data,
                                                                                  self.offset_data, self.do_not_last)

                if self.dtype_x == "float16":
                    # arg last dim:fp16:calc core at first dim and tiling at last dim and one time
                    with self.tik_instance.if_scope(self.tiling_mode == 0):
                        with self.tik_instance.new_stmt_scope():
                            self.compute_argmax_last_axis_fp16_copy_one_time(core_idx, self.segment_loop,
                                                                             self.segment_tail)
                    # arg last dim:fp16:calc core at first dim and tiling at first dim and more vector
                    with self.tik_instance.if_scope(self.tiling_mode == 1):
                        with self.tik_instance.new_stmt_scope():
                            self.compute_argmax_last_axis_fp16_more_vector(core_idx, self.segment_loop,
                                                                           self.segment_tail)
                    # arg last dim:fp16:calc core at first dim and tiling at first dim and less vector
                    with self.tik_instance.if_scope(self.tiling_mode == 2):
                        with self.tik_instance.new_stmt_scope():
                            self.compute_argmax_last_axis_fp16_less_vector(core_idx, self.segment_loop,
                                                                           self.segment_tail)
                    # arg last dim:fp16:calc core at first dim and tiling at last dim and defalut
                    with self.tik_instance.if_scope(self.tiling_mode == 3):
                        with self.tik_instance.new_stmt_scope():
                            self.compute_argmax_last_axis_fp16(core_idx, self.segment_loop, self.segment_tail)
                    # arg not last dim:fp16:calc core at first dim and tiling at last dim and less 2048
                    with self.tik_instance.if_scope(self.tiling_mode == 6):
                        with self.tik_instance.new_stmt_scope():
                            with self.tik_instance.for_range(0, self.core_ele) as ele_idx:
                                first_idx = core_idx * self.one_core_ele + ele_idx
                                self.compute_argmax_not_last_axis_cut_by_first_dim(first_idx, self.segment_loop,
                                                                                   self.segment_tail,
                                                                                   self.segment_tail_data,
                                                                                   self.offset_data,
                                                                                   self.do_not_last_fp16_default)
                    # arg not last dim:fp16:calc core at first dim and tiling at last dim and less 2048 and align
                    with self.tik_instance.if_scope(self.tiling_mode == 7):
                        with self.tik_instance.new_stmt_scope():
                            with self.tik_instance.for_range(0, self.core_ele) as ele_idx:
                                first_idx = core_idx * self.one_core_ele + ele_idx
                                gm_in_offset = first_idx * self.axis_size * self.last_dim_size
                                gm_out_offset = first_idx * self.last_dim_size
                                self.do_not_last_fp16_align(self.last_dim_size, gm_in_offset, gm_out_offset)
                    # arg not last dim:fp16:calc core at last dim and tiling at last dim and less 2048
                    with self.tik_instance.if_scope(self.tiling_mode == 9):
                        with self.tik_instance.new_stmt_scope():
                            with self.tik_instance.for_range(0, self.first_dim_size) as ele_idx:
                                offset_in = ele_idx * self.axis_size * self.last_dim_size + core_idx * self.one_core_ele
                                offset_out = ele_idx * self.last_dim_size + core_idx * self.one_core_ele
                                self.compute_argmax_not_last_axis_cut_by_last_dim(offset_in, offset_out,
                                                                                  self.segment_loop, self.segment_tail,
                                                                                  self.segment_tail_data,
                                                                                  self.offset_data,
                                                                                  self.do_not_last_fp16_default)
                    # arg not last dim:fp16:calc core at first dim and tiling at last dim and greater 2048
                    with self.tik_instance.if_scope(self.tiling_mode == 10):
                        with self.tik_instance.new_stmt_scope():
                            with self.tik_instance.for_range(0, self.core_ele) as ele_idx:
                                first_idx = core_idx * self.one_core_ele + ele_idx
                                self.compute_argmax_not_last_axis_cut_by_first_dim(first_idx, self.segment_loop,
                                                                                   self.segment_tail,
                                                                                   self.segment_tail_data,
                                                                                   self.offset_data, self.do_not_last)
                    # arg not last dim:fp16:calc core at last dim and tiling at last dim and greater 2048
                    with self.tik_instance.if_scope(self.tiling_mode == 11):
                        with self.tik_instance.new_stmt_scope():
                            with self.tik_instance.for_range(0, self.first_dim_size) as ele_idx:
                                offset_in = ele_idx * self.axis_size * self.last_dim_size + core_idx * self.one_core_ele
                                offset_out = ele_idx * self.last_dim_size + core_idx * self.one_core_ele
                                self.compute_argmax_not_last_axis_cut_by_last_dim(offset_in, offset_out,
                                                                                  self.segment_loop, self.segment_tail,
                                                                                  self.segment_tail_data,
                                                                                  self.offset_data, self.do_not_last)

    def argmax_compute(self):
        """argmax_compute
        """
        self.argmax_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.data_gm, self.axis_gm],
                                   outputs=[self.result_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        tbe_context.get_context().add_compile_info("vars", {
            "ub_ele": self.ub_ele,
            "core_num": self.core_num,
        })
        return self.tik_instance


@register_operator("ArgMaxV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def arg_max_v2(x, dimension, y, kernel_name="arg_max_v2"):
    """
    Generate arg_max_v2 operator use arg_max_v2

    Parameters
    ----------
    x: dict
        data of input, support "float16", "float32".
    dimension: dict
        dimension input.
    y: dict
        index of output.
    kernel_name: str
        kernel name, default value is "arg_max_v2"

    Returns
    -------
    tik_instance
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    para_check.check_shape(shape_x, param_name="x")
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_x.lower(), check_list, param_name="x")
    max_index = Argmax(shape_x, dtype_x, kernel_name)

    tik_instance = max_index.argmax_compute()

    return tik_instance
