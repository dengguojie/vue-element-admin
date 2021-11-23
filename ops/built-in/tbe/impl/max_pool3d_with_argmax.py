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
max_pool3d_with_argmax
"""
import math
import functools
from functools import partial
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from enum import unique

from te import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.load3d_common_func import img2col

@unique
class L1MoveStrategy(Enum):
    """
    L1 move strategy Enum: EMPTY, NO_NEED_TO_CUT
    """
    EMPTY = 0
    NO_NEED_TO_CUT = 1


@unique
class UbKernelStrategy(Enum):
    """
    UB kernel strategy Enum: EMPTY, WHOLE_KERNEL, KH_AND_KW, ONLY_KW
    """
    EMPTY = 0
    WHOLE_KERNEL = 1
    KH_AND_KW = 2
    ONLY_KW = 3


# 'pylint: disable=too-few-public-methods
class Core:
    """
    Calculate n, c1, d_step based on core_index
    """

    def __init__(self, core_ind, c1, d_step):
        self.ind_N = core_ind // (c1 * d_step)
        n_rest = core_ind % (c1 * d_step)
        self.ind_d = n_rest // c1
        self.ind_C1 = n_rest % c1


# 'pylint: disable=too-few-public-methods
class MaxPool3DWithArgmax(metaclass=ABCMeta):
    """
    MaxPool3DWithArgmax: compute definition of max_pool3d_with_argmax
    """

    def __init__(self, x, kernel_size, stride, kernel_name):
        self.inst = tik.Tik()

        self.kernel_name = kernel_name
        self.input_shape = x.get('shape')
        self.input_dtype = x.get('dtype')
        self.block_size = 32
        self.vector_size_bytes = 256
        self.each_elem_bytes = 2
        self.each_uint16_bytes = 2
        self.each_fp16_bytes = 2
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.ub_max_elem = self.ub_size_bytes // self.each_elem_bytes
        self.vector_size_elem = self.vector_size_bytes // self.each_elem_bytes
        self.out_mask_type = 'uint16'
        self.mask4compute = 128
        self.upper_rep_limit = 255

        # notice: though we can aligned the shape of bitmask_ub, but each repeat of vcmpv has 128bits(16B) result.
        # we have to aligned the address to 32B.
        self.upper_cmp_rep = 254

        self.N, self.input_d, self.C1, self.input_h, self.input_w, self.C0 = x.get('shape')
        _, _, self.k_d, self.k_h, self.k_w = kernel_size
        _, _, self.stride_d, self.stride_h, self.stride_w = stride

        self.d_step = (self.input_d - self.k_d) // self.stride_d + 1
        self.h_step = (self.input_h - self.k_h) // self.stride_h + 1
        self.w_step = (self.input_w - self.k_w) // self.stride_w + 1
        self.k_elem = self.k_h * self.k_w * self.k_d
        self.output_shape = (self.N, self.d_step, self.C1, self.h_step, self.w_step, self.C0)

        # first, we have to aligned the data points to uint16, not yet aligned the uint16 to block_size(32B)
        # this is approximately equal to w_step * h_step, aligned to 16.(compliment by zero)
        self.aligned_bm_line = math.ceil(self.w_step * self.h_step / 16) * 16
        self.dirty_l1_pad = math.ceil((self.aligned_bm_line - self.h_step * self.w_step) / self.w_step) * self.stride_h

        self.l1_strategy = L1MoveStrategy.EMPTY
        self.ub_kernel_stg = UbKernelStrategy.EMPTY
        self.check_load3d_support = tbe_platform.api_check_support("tik.load3dv1")

        # encapsulate the data_move function when the data is continue.
        self._move_ctn = partial(self.inst.data_move,
                                 sid=0,
                                 nburst=1,
                                 src_stride=0,
                                 dst_stride=0)

        # encapsulate the vnot function when the data is continue.
        self._vnot_ctn = partial(self.inst.vnot,
                                 src_blk_stride=1,
                                 dst_blk_stride=1,
                                 src_rep_stride=8,
                                 dst_rep_stride=8,
                                 repeat_times=1,
                                 mask=128)

        # encapsulate the vcmpv function when the data is continue.
        self._cmp_ctn = partial(self.inst.vcmpv_eq,
                                src0_blk_stride=1,
                                src1_blk_stride=1,
                                src0_rep_stride=8,
                                src1_rep_stride=8)

    # encapsulate the two-number compute function.
    # 'pylint: diable=too-many-arguments
    @staticmethod
    def _compute_two_ctn(method, dst, src0, src1, rep_times=1, mask=128):
        method(mask=mask,
               dst=dst,
               src0=src0,
               src1=src1,
               repeat_times=rep_times,
               dst_blk_stride=1,
               src0_blk_stride=1,
               src1_blk_stride=1,
               dst_rep_stride=8,
               src0_rep_stride=8,
               src1_rep_stride=8)

    def _vector_dup(self, src, src_start, shape, dup_reg):
        VECTOR_FP16_SIZE = 128
        MAX_VECTOR_REPEAT_TIME = 255

        ele_num = functools.reduce(lambda x, y: x * y, shape)
        total_repeat_time = ele_num // VECTOR_FP16_SIZE
        remain_ele = ele_num % VECTOR_FP16_SIZE
        mask_value = VECTOR_FP16_SIZE
        repeat_max_time = total_repeat_time // MAX_VECTOR_REPEAT_TIME
        remain_repeat_time = total_repeat_time % MAX_VECTOR_REPEAT_TIME

        with self.inst.for_range(0, repeat_max_time) as loop:
            self.inst.vector_dup(mask_value, src[src_start + loop * MAX_VECTOR_REPEAT_TIME * mask_value],
                                 dup_reg, MAX_VECTOR_REPEAT_TIME, 1, 8)

        if remain_repeat_time > 0:
            self.inst.vector_dup(mask_value, src[src_start + repeat_max_time * MAX_VECTOR_REPEAT_TIME * mask_value],
                                 dup_reg, remain_repeat_time, 1, 8)

        if remain_ele > 0:
            self.inst.vector_dup(remain_ele, src[src_start + repeat_max_time * MAX_VECTOR_REPEAT_TIME * mask_value +
                                                 remain_repeat_time * mask_value], dup_reg, 1, 1, 8)

    def _img2col(self, aux_l1, big_matrix_ub, l1_begin_pos, rep_times):
        if self.ub_kernel_stg is UbKernelStrategy.WHOLE_KERNEL:
            if not self.check_load3d_support:
                self._vector_dup(big_matrix_ub, 0, big_matrix_ub.shape, 0)
            with self.inst.for_range(0, self.k_d) as tmp_kd:
                # we adopt the scheme that repeat_mode = 1
                with self.inst.for_range(0, self.k_h) as tmp_kh:
                    with self.inst.for_range(0, self.k_w) as tmp_kw:
                        if self.check_load3d_support:
                            self.inst.load3dv1(
                                big_matrix_ub[tmp_kd * self.k_h * self.k_w + tmp_kh * self.k_w + tmp_kw, 0],
                                aux_l1,
                                (0, 0, 0, 0),
                                aux_l1.shape[2],
                                aux_l1.shape[3],
                                tmp_kd,
                                tmp_kw,
                                tmp_kh,
                                left_top_w=l1_begin_pos[1],
                                left_top_h=l1_begin_pos[0],
                                stride_w=self.stride_w,
                                stride_h=self.stride_h,
                                filter_w=self.k_w,
                                filter_h=self.k_h,
                                dilation_filter_w=1,
                                dilation_filter_h=1,
                                jump_offset=1,
                                repeat_mode=1,
                                repeat_time=rep_times
                            )
                        else:
                            _, _, shape_h, shape_w, _ = aux_l1.shape
                            l1_start_offset = tmp_kd * shape_h * shape_w * self.C0
                            img2col(
                                self.inst, aux_l1, big_matrix_ub, l1_start_offset,
                                (tmp_kd * self.k_h * self.k_w + tmp_kh * self.k_w + tmp_kw) * big_matrix_ub.shape[1],
                                tmp_kh, tmp_kw, l1_begin_pos[0], l1_begin_pos[1], aux_l1.shape[2], aux_l1.shape[3],
                                self.k_h, self.k_w, self.stride_h, self.stride_w, rep_times, 1, (0, 0, 0, 0))

    # 'pylint: diable=too-many-arguments
    def _calc_maxline(self, max_line_ub, big_matrix_ub, line_blk, super_line_loop, super_line_tail):
        self._move_ctn(dst=max_line_ub,
                       src=big_matrix_ub,
                       burst=line_blk)
        with self.inst.for_range(1, self.k_elem) as line_ind:
            if super_line_loop != 0:
                with self.inst.for_range(0, super_line_loop) as super_loop_ind:
                    self._compute_two_ctn(
                        method=self.inst.vmax,
                        dst=max_line_ub[super_loop_ind * self.upper_rep_limit * self.mask4compute],
                        src0=max_line_ub[super_loop_ind * self.upper_rep_limit * self.mask4compute],
                        src1=big_matrix_ub[line_ind, super_loop_ind * self.upper_rep_limit * self.mask4compute],
                        rep_times=self.upper_rep_limit
                    )
            if super_line_tail != 0:
                self._compute_two_ctn(
                    method=self.inst.vmax,
                    dst=max_line_ub[super_line_loop * self.upper_rep_limit * self.mask4compute],
                    src0=max_line_ub[super_line_loop * self.upper_rep_limit * self.mask4compute],
                    src1=big_matrix_ub[line_ind, super_line_loop * self.upper_rep_limit * self.mask4compute],
                    rep_times=super_line_tail
                )

    def _init_gm_tensor(self):
        input_gm_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                           shape=self.input_shape,
                                           name='input_tensor',
                                           scope=tik.scope_gm)
        output_gm_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                       shape=self.output_shape,
                                       name='output_tensor',
                                       scope=tik.scope_gm)
        output_gm_tensor = output_gm_tensor.reshape((self.N, self.d_step, self.C1, self.h_step * self.w_step * self.C0))

        # the output bitmask
        final_bitmask_gm_tensor = self.inst.Tensor(
            dtype=self.out_mask_type,
            # the last two dimension of the shape aim to align to 32B.
            shape=[self.N, self.d_step, self.C1 * self.k_elem, self.aligned_bm_line // 16, 16],
            name='final_bitmask',
            scope=tik.scope_gm
        )

        tensor_info = {'input': input_gm_tensor,
                       'output': output_gm_tensor,
                       'argmax': final_bitmask_gm_tensor}

        return tensor_info

    # 'pylint: diable=too-many-arguments
    def _calc_bitmask(self, bitmask_ub, big_matrix_ub, max_line_ub, super_line_loop, super_line_tail):
        with self.inst.for_range(0, self.k_elem) as line_ind:
            if super_line_loop != 0:
                with self.inst.for_range(0, super_line_loop) as super_loop_ind:
                    self._cmp_ctn(
                        dst=bitmask_ub[line_ind, super_loop_ind * self.upper_cmp_rep * self.mask4compute // 16],
                        src0=big_matrix_ub[line_ind, super_loop_ind * self.upper_cmp_rep * self.mask4compute],
                        src1=max_line_ub[super_loop_ind * self.upper_cmp_rep * self.mask4compute],
                        repeat_times=self.upper_cmp_rep
                    )
            if super_line_tail != 0:
                self._cmp_ctn(
                    dst=bitmask_ub[line_ind, super_line_loop * self.upper_cmp_rep * self.mask4compute // 16],
                    src0=big_matrix_ub[line_ind, super_line_loop * self.upper_cmp_rep * self.mask4compute],
                    src1=max_line_ub[super_line_loop * self.upper_cmp_rep * self.mask4compute],
                    repeat_times=super_line_tail
                )

    def _deduplicate_bitmask(self, mask_or_ub, mask_not_ub, bitmask_ub, data_blk, bm_loop, bm_tail):
        # first, init the mask_not and mask_or
        if bm_loop != 0:
            self._vnot_ctn(dst=mask_not_ub,
                           src=bitmask_ub,
                           repeat_times=bm_loop)
        if bm_tail != 0:
            self._vnot_ctn(dst=mask_not_ub[bm_loop * self.mask4compute],
                           src=bitmask_ub[0, bm_loop * self.mask4compute],
                           mask=bm_tail)
        self._move_ctn(dst=mask_or_ub,
                       src=bitmask_ub,
                       burst=data_blk)
        with self.inst.for_range(1, self.k_elem) as line_ind:
            if bm_loop != 0:
                self._compute_two_ctn(method=self.inst.vor,
                                      dst=mask_or_ub,
                                      src0=mask_or_ub,
                                      src1=bitmask_ub[line_ind, 0],
                                      rep_times=bm_loop)
                self._compute_two_ctn(method=self.inst.vand,
                                      dst=bitmask_ub[line_ind, 0],
                                      src0=bitmask_ub[line_ind, 0],
                                      src1=mask_not_ub,
                                      rep_times=bm_loop)
                self._vnot_ctn(dst=mask_not_ub,
                               src=mask_or_ub,
                               repeat_times=bm_loop)
            if bm_tail != 0:
                self._compute_two_ctn(method=self.inst.vor,
                                      dst=mask_or_ub[bm_loop * self.mask4compute],
                                      src0=mask_or_ub[bm_loop * self.mask4compute],
                                      src1=bitmask_ub[line_ind, bm_loop * self.mask4compute],
                                      mask=bm_tail)
                self._compute_two_ctn(method=self.inst.vand,
                                      dst=bitmask_ub[line_ind, bm_loop * self.mask4compute],
                                      src0=bitmask_ub[line_ind, bm_loop * self.mask4compute],
                                      src1=mask_not_ub[bm_loop * self.mask4compute],
                                      mask=bm_tail)
                self._vnot_ctn(dst=mask_not_ub[bm_loop * self.mask4compute],
                               src=mask_or_ub[bm_loop * self.mask4compute],
                               mask=bm_tail)

    @abstractmethod
    def run(self):
        """
        run function
        """
        pass


# 'pylint: disable=too-few-public-method
class MaxPool3DWithArgmaxWholeKernel(MaxPool3DWithArgmax):
    """
    MaxPool3DWithArgmaxWholeKernel: inherited from MaxPool3DWithArgmax
    """

    def __init__(self, x, kernel_size, stride, kernel_name):
        super().__init__(x, kernel_size, stride, kernel_name)
        # init some value for cut-ub process
        # all the tensors that occupy the ub:(X represents the size of bitmask line)
        # -------------------------------------------------------------------------------------------------------
        # name                        |  size (regardless of ub-memory)   |  hypothetical variable (NOT accuracy)
        # -------------------------------------------------------------------------------------------------------
        # bigmat_ub_T                 |  k_elem * w_step * h_step * C0    |  X * 16 * k_elem
        # bitmask_ub_T                |  k_elem * aligned_bitmask_line    |  X * k_elem
        # maxline_ub_T                |  w_step * h_step * C0             |  X * 16
        # maskor_ub_T + masknot_ub_T  |  aligned_bitmask_line * 2         |  X * 2
        # -------------------------------------------------------------------------------------------------------
        # the formula of ub tiling is: 17 * k_elem * X + 18 * X = ub_size
        self.cut_ub_val = self.ub_max_elem // (self.k_elem * 17 + 18) // 16 * 16
        self.ub_line = self.cut_ub_val * 16
        self.cut_ub_loop = self.h_step * self.w_step // self.cut_ub_val
        self.cut_ub_tail_aligned = self.aligned_bm_line - self.cut_ub_loop * self.cut_ub_val
        self.cut_ub_tail_ori = self.h_step * self.w_step % self.cut_ub_val

        self.l1_strategy = L1MoveStrategy.NO_NEED_TO_CUT
        self.ub_kernel_stg = UbKernelStrategy.WHOLE_KERNEL

        # Notice: 1. line_rep_loop probably more than 255.
        #         2. ub_line has already aligned to 256, so there is no need to consider line_rep_tail.
        self.line_rep_loop = self.ub_line // self.vector_size_elem
        self.super_line_loop = self.line_rep_loop // self.upper_rep_limit
        self.super_line_tail = self.line_rep_loop % self.upper_rep_limit
        self.bm_line_loop = self.cut_ub_val // self.vector_size_elem
        self.bm_line_tail = self.cut_ub_val % self.vector_size_elem

        self.super_line_loop_cmp = self.line_rep_loop // self.upper_cmp_rep
        self.super_line_tail_cmp = self.line_rep_loop % self.upper_cmp_rep

        # compute some values for ub-cut-tail process
        # Notice: cut_ub_tail_aligned has already aligned to 16, so there is no need to consider ubtail_ub_tail
        self.ubtail_ub_loop = self.cut_ub_tail_aligned * self.C0 // self.vector_size_elem
        self.ubtail_bm_loop = self.cut_ub_tail_aligned // self.vector_size_elem
        self.ubtail_bm_tail = self.cut_ub_tail_aligned % self.vector_size_elem
        self.ubtail_super_loop = self.ubtail_ub_loop // self.upper_rep_limit
        self.ubtail_super_tail = self.ubtail_ub_loop % self.upper_rep_limit

        self.ubtail_super_loop_cmp = self.ubtail_ub_loop // self.upper_cmp_rep
        self.ubtail_super_tail_cmp = self.ubtail_ub_loop % self.upper_cmp_rep

    # 'pylint: diable=too-many-locals
    def _cut_ub_process(self, ind_ctx, mode):
        tensor_info = ind_ctx['tensor_info']
        ind_n = ind_ctx['ind_N']
        ind_c1 = ind_ctx['ind_C1']
        ind_d = ind_ctx['ind_d']

        # init a big matrix in ub, for register the img2col.
        bigmat_ub_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                            shape=[self.k_elem, self.cut_ub_val * self.C0],
                                            name='big_matrix',
                                            scope=tik.scope_ubuf)
        # init a register to store the max line.
        maxline_ub_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                             shape=[self.cut_ub_val * self.C0, ],
                                             name='max_line',
                                             scope=tik.scope_ubuf)
        # init a uint16 bitmask, each number represents 16 bits. (16 pairs of numbers)
        bitmask_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                             shape=[self.k_elem, self.cut_ub_val],
                                             name='bitmask_ub',
                                             scope=tik.scope_ubuf)
        # init some tensors for deduplication during the inter process.
        maskor_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                            shape=[self.cut_ub_val, ],
                                            name='mask_or',
                                            scope=tik.scope_ubuf)
        masknot_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                             shape=[self.cut_ub_val, ],
                                             name='mask_not',
                                             scope=tik.scope_ubuf)

        # loop process
        if mode == 'loop':
            cut_ub_ind = ind_ctx['cut_ub_ind']
            # step1: begin to calculate img2col.
            l1_begin_h = (self.cut_ub_val * cut_ub_ind // self.w_step) * self.stride_h
            l1_begin_w = (self.cut_ub_val * cut_ub_ind % self.w_step) * self.stride_w
            l1_begin_pos = (l1_begin_h, l1_begin_w)
            super()._img2col(aux_l1=tensor_info['aux_l1'],
                             big_matrix_ub=bigmat_ub_tensor,
                             l1_begin_pos=l1_begin_pos,
                             rep_times=self.cut_ub_val // 16)

            # step2: begin to calculate max line.
            super()._calc_maxline(max_line_ub=maxline_ub_tensor,
                                  big_matrix_ub=bigmat_ub_tensor,
                                  line_blk=self.cut_ub_val,
                                  super_line_loop=self.super_line_loop,
                                  super_line_tail=self.super_line_tail)

            # step3: move the max_line_ub to gm
            self._move_ctn(dst=tensor_info['output'][ind_n, ind_d, ind_c1, cut_ub_ind * self.ub_line],
                           src=maxline_ub_tensor,
                           burst=self.ub_line * self.each_fp16_bytes // self.block_size)

            # step4: compute the bitmask
            super()._calc_bitmask(big_matrix_ub=bigmat_ub_tensor,
                                  bitmask_ub=bitmask_ub_tensor,
                                  max_line_ub=maxline_ub_tensor,
                                  super_line_loop=self.super_line_loop_cmp,
                                  super_line_tail=self.super_line_tail_cmp)

            # step5: deduplicate the bitmask, each column must have at most one '1'.
            super()._deduplicate_bitmask(mask_or_ub=maskor_ub_tensor,
                                         mask_not_ub=masknot_ub_tensor,
                                         bitmask_ub=bitmask_ub_tensor,
                                         bm_loop=self.bm_line_loop,
                                         bm_tail=self.bm_line_tail,
                                         data_blk=self.cut_ub_val // 16)

            # step6: move bitmask to gm
            self.inst.data_move(
                dst=tensor_info['argmax'][ind_n, ind_d, ind_c1 * self.k_elem, cut_ub_ind * self.cut_ub_val // 16, 0],
                src=bitmask_ub_tensor,
                sid=0,
                nburst=self.k_elem,
                burst=self.cut_ub_val * self.each_uint16_bytes // self.block_size,
                src_stride=0,
                dst_stride=(self.aligned_bm_line - self.cut_ub_val) // 16
            )
            return

        # tail process
        # step1: begin to calculate img2col
        l1_begin_h = (self.cut_ub_val * self.cut_ub_loop // self.w_step) * self.stride_h
        l1_begin_w = (self.cut_ub_val * self.cut_ub_loop % self.w_step) * self.stride_w
        l1_begin_pos = (l1_begin_h, l1_begin_w)
        super()._img2col(aux_l1=tensor_info['aux_l1'],
                         big_matrix_ub=bigmat_ub_tensor,
                         l1_begin_pos=l1_begin_pos,
                         rep_times=self.cut_ub_tail_aligned // 16)

        # step2: begin to calculate max line.
        super()._calc_maxline(max_line_ub=maxline_ub_tensor,
                              big_matrix_ub=bigmat_ub_tensor,
                              line_blk=self.cut_ub_tail_aligned,
                              super_line_loop=self.ubtail_super_loop,
                              super_line_tail=self.ubtail_super_tail)

        # step3: move the max_line_ub to gm
        self._move_ctn(dst=tensor_info['output'][ind_n, ind_d, ind_c1, self.cut_ub_loop * self.ub_line],
                       src=maxline_ub_tensor,
                       burst=self.cut_ub_tail_ori * self.C0 * self.each_fp16_bytes // self.block_size)

        # step4: compute the bitmask
        super()._calc_bitmask(big_matrix_ub=bigmat_ub_tensor,
                              bitmask_ub=bitmask_ub_tensor,
                              max_line_ub=maxline_ub_tensor,
                              super_line_loop=self.ubtail_super_loop_cmp,
                              super_line_tail=self.ubtail_super_tail_cmp)

        # step5: deduplicate the bitmask, each column must have at most one '1'.
        super()._deduplicate_bitmask(mask_or_ub=maskor_ub_tensor,
                                     mask_not_ub=masknot_ub_tensor,
                                     bitmask_ub=bitmask_ub_tensor,
                                     bm_loop=self.ubtail_bm_loop,
                                     bm_tail=self.ubtail_bm_tail,
                                     data_blk=self.cut_ub_tail_aligned // 16)

        # step6: move bitmask to gm
        self.inst.data_move(
            dst=tensor_info['argmax'][ind_n, ind_d, ind_c1 * self.k_elem, self.cut_ub_loop * self.cut_ub_val // 16, 0],
            src=bitmask_ub_tensor,
            sid=0,
            nburst=self.k_elem,
            burst=self.cut_ub_tail_aligned // 16,
            src_stride=(self.cut_ub_val - self.cut_ub_tail_aligned) // 16,
            dst_stride=(self.aligned_bm_line - self.cut_ub_tail_aligned) // 16
        )
        return

    def run(self):
        tensor_info = self._init_gm_tensor()

        if self.l1_strategy is L1MoveStrategy.NO_NEED_TO_CUT:
            l1_shape = [1, self.k_d, self.input_h + self.dirty_l1_pad, self.input_w, 16]
        else:
            l1_shape = [0, ]

        aux_l1_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                         shape=l1_shape,
                                         name='aux_l1',
                                         scope=tik.scope_cbuf)
        tensor_info['aux_l1'] = aux_l1_tensor
        core_nums = self.N * self.C1 * self.d_step

        with self.inst.for_range(0, core_nums, block_num=core_nums) as core_ind:

            core = Core(core_ind, self.C1, self.d_step)

            # if we don't need to cut l1, now is a good time to move data that all we need in this d-loop.
            if self.l1_strategy is L1MoveStrategy.NO_NEED_TO_CUT:
                if (self.C1 - 1) * self.input_h * self.input_w <= 65535:
                    self.inst.data_move(
                        dst=aux_l1_tensor[0],
                        src=tensor_info['input'][core.ind_N, core.ind_d * self.stride_d, core.ind_C1, 0, 0, 0],
                        sid=0,
                        nburst=self.k_d,
                        burst=self.input_h * self.input_w,
                        src_stride=(self.C1 - 1) * self.input_h * self.input_w,
                        dst_stride=self.dirty_l1_pad * self.input_w
                    )
                else:
                    with self.inst.for_range(0, self.k_d) as d_ind:
                        self._move_ctn(
                            dst=aux_l1_tensor[0, d_ind, 0, 0, 0],
                            src=tensor_info['input'][core.ind_N, core.ind_d * self.stride_d + d_ind,
                                                     core.ind_C1, 0, 0, 0],
                            burst=self.input_h * self.input_w
                        )

            ub_context = {"ind_N": core.ind_N,
                          "ind_C1": core.ind_C1,
                          "ind_d": core.ind_d,
                          "tensor_info": tensor_info}
            # begin to calculate cut ub-process
            if self.cut_ub_loop != 0:
                with self.inst.for_range(0, self.cut_ub_loop) as cut_ub_ind:
                    ub_context["cut_ub_ind"] = cut_ub_ind
                    self._cut_ub_process(ub_context, mode='loop')
            if self.cut_ub_tail_aligned != 0:
                self._cut_ub_process(ub_context, mode='tail')

        tensor_info['output'] = tensor_info['output'].reshape(self.output_shape)
        self.inst.BuildCCE(kernel_name=self.kernel_name, inputs=[tensor_info['input']],
                           outputs=[tensor_info['output'], tensor_info['argmax']])
        return self.inst


# 'pylint: diable=too-many-arguments,too-many-branches
def _check_param(x, ksize, strides, pads, dilation, ceil_mode, argmax_type):
    input_shape = x.get("shape")
    if x.get("dtype").lower() != "float16":
        raise RuntimeError("max_pool3d_with_argmax only support float16!")
    if len(input_shape) != 6:
        raise RuntimeError("invalid shape params, max_pool3d_with_argmax input shape must be 6D format!")
    if input_shape[-1] != 16:
        raise RuntimeError("C0 must be 16!")
    if len(ksize) != 5:
        raise RuntimeError("dimension of ksize must be 5, value of (1, 1, kernel_d, kernel_h, kernel_w)!")
    if ksize[0] != 1 or ksize[1] != 1:
        raise RuntimeError("first two dimensions of ksize should be one!")
    if ksize[3] > 255 or ksize[4] > 255:
        raise RuntimeError("current version don't support kernel_h or kernel_w > 255!")
    if len(strides) != 5:
        raise RuntimeError("dimension of stride must be 5, value of (1, 1, stride_d, stride_h, stride_w)!")
    if strides[0] != 1 or strides[1] != 1:
        raise RuntimeError("first two dimensions of strides should be one!")
    if len(pads) != 3:
        raise RuntimeError("pads' shape should be (3, 2)!")
    for pad in pads:
        if len(pad) != 2:
            raise RuntimeError("pads' shape should be (3, 2)!")
    if len(dilation) != 5:
        raise RuntimeError("dimension of dilation must be 5, value of (1, 1, dil_d, dil_h, dil_w)!")
    if dilation[0] != 1 or dilation[1] != 1:
        raise RuntimeError("first two dimensions of dilation must be one!")
    if ceil_mode:
        raise RuntimeError("current version only support ceil_mode=False!")
    if argmax_type != "bitmask":
        raise RuntimeError("current version only support bitmask argmax, please set argmax_type=bitmask!")

    # this version only support the generalization of branch MaxPool3dWithArgmaxWholeKernel,
    # that means we have to put all the kernel into ub, at each loop of img2col.
    # please refer to MaxPool3dWithArgmaxWholeKernel's init function for more details, then you will know
    # what these magic numbers mean.
    k_bound = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // 32 - 18) // 17
    if ksize[2] * ksize[3] * ksize[4] > k_bound:
        raise RuntimeError("the kernel is so big, we don't support such generalization!")
    wo = (input_shape[4] - ksize[4]) // strides[4] + 1
    ho = (input_shape[3] - ksize[3]) // strides[3] + 1
    l1_h_pad = math.ceil((math.ceil(ho * wo / 16) * 16 - ho * wo) / wo) * strides[3]

    # this version only support the strategy that with out cutting l1.
    # so we put size kernel_d * input_h * input_w into l1 at one time.
    if ksize[2] * (input_shape[3] + l1_h_pad) * input_shape[4] * 16 > tbe_platform.get_soc_spec(tbe_platform.L1_SIZE):
        raise RuntimeError("l1 is not enough, please try a smaller kernel_d, or a smaller input_h or input_w!")
    if wo == 1 and ho != 1 and tbe_platform.get_soc_spec('SOC_VERSION') != 'Ascend310':
        raise RuntimeError("current version don't support W_out = 1 in this environment!")


# 'pylint: disable=unused-argument,too-many-arguments
@register_operator("MaxPool3DWithArgmax")
def max_pool3d_with_argmax(x, y, argmax, kernel_size, strides, pads=((0, 0), (0, 0), (0, 0)),
                           dilation=(1, 1, 1, 1, 1), ceil_mode=False, data_format="NCDHW", argmax_type="bitmask",
                           kernel_name='max_pool3d_with_argmax'):
    """
    interface of max_pool3d_with_argmax.

    Parameters
    ----------
    x: the input tensor dict, include keys: shape and dtype, this version only support float16.
    y: the output tensor dict, include keys: shape and dtype, dtype is consistent with x.
    argmax: the output tensor, represent the argmax bitmask, format as NC1HWC0,
        actually, it represent (N, Do, C1*k_size, Ho*Wo//16, 16).
    kernel_size: list of kernel_size, value of (1, 1, kernel_d, kernel_h, kernel_w).
    strides: list of stride_size, value of (1, 1, stride_d, stride_h, stride_w).
    pads: list of padding, spread out the six-directions of padding.
    dilation: list of dilation, value of (1, 1, dilation_d, dilation_h, dilation_w).
    ceil_mode: reserved field, current version only support ceil_mode=False.
    data_format: the format of origin input, default value for torch is "NCDHW"
    argmax_type: reserved field, determine whether the argmax is img2col mode or torch-output mode.
        current version only support argmax_type="bitmask".
    kernel_name: max_pool3d_with_argmax

    Returns
    -------
    tik_instance
    """
    _check_param(x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type)
    obj = MaxPool3DWithArgmaxWholeKernel(x, kernel_size, strides, kernel_name)
    return obj.run()
