from te import tik
import numpy as np
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
import math


# encapsulate the data_move function when the data is continue.
def _move_ctn(dst, src, data_blknum, inst):
    inst.data_move(dst=dst,
                   src=src,
                   sid=0,
                   nburst=1,
                   burst=data_blknum,
                   src_stride=0,
                   dst_stride=0)


# encapsulate the two-number compute function.
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


# encapsulate the vnot function when the data is continue.
def _vnot_ctn(inst, dst, src, rep_times=1, mask=128):
    inst.vnot(mask=mask,
              dst=dst,
              src=src,
              repeat_times=rep_times,
              src_blk_stride=1,
              dst_blk_stride=1,
              src_rep_stride=8,
              dst_rep_stride=8)


# encapsulate the vcmpv function when the data is continue.
def _cmp_ctn(inst, dst, src0, src1, rep_times):
    inst.vcmpv_eq(dst=dst,
                  src0=src0,
                  src1=src1,
                  repeat_times=rep_times,
                  src0_blk_stride=1,
                  src1_blk_stride=1,
                  src0_rep_stride=8,
                  src1_rep_stride=8)


class MaxPool3DWithArgmax():

    def __init__(self, x, kernel_size, stride, kernel_name):
        self.kernel_name = kernel_name
        self.input_shape = x.get('shape')
        self.input_dtype = x.get('dtype')
        self.block_size = 32
        self.vector_size_bytes = 256
        self.each_elem_bytes = 2
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.ub_max_elem = self.ub_size_bytes // self.each_elem_bytes
        self.vector_size_elem = self.vector_size_bytes // self.each_elem_bytes
        self.out_mask_type = 'uint16'
        self.mask4compute = 128
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
        self.aligned_bitmask_line = math.ceil(self.w_step * self.h_step * self.C0 / 256) * 256 // 16

    def run(self):
        tik_instance = self._cut_ub()
        return tik_instance

    def _cut_ub(self):
        tik_instance = tik.Tik()

        # init some value for cut-ub process
        # all the tensors that occupy the ub:(X represents the size of bitmask line)
        # -------------------------------------------------------------------------------------------------------
        # name                        |  size                             |  hypothetical variable (NOT accuracy)
        # -------------------------------------------------------------------------------------------------------
        # bigmat_ub_T                 |  k_elem * w_step * h_step * C0    |  X * 16 * k_elem
        # bitmask_ub_T                |  k_elem * aligned_bitmask_line    |  X * k_elem
        # maxline_ub_T                |  w_step * h_step * C0             |  X * 16
        # maskor_ub_T + masknot_ub_T  |  aligned_bitmask_line * 2         |  X * 2
        # aux0_ub_T + aux1_ub_T       |  512                              |  512
        # -------------------------------------------------------------------------------------------------------
        # the formula of ub tiling is: 17 * k_elem * X + 18 * X = ub_size
        bm_line_cut_ub = (self.ub_max_elem - 512) // (self.k_elem * 17 + 18) // 16 * 16
        ub_line_cut_ub = bm_line_cut_ub * 16
        cut_ub_loop = self.h_step * self.w_step * self.C0 // ub_line_cut_ub
        cut_ub_tail = self.h_step * self.w_step * self.C0 % ub_line_cut_ub

        # Notice: line_rep_loop usually be 182, and all of 'line_loop4cmp * 2, bm_line_loop, ub_line_rep_loop,
        #       ubtail_bm_line_loop, ubtail_line_loop4cmp * 2' are smaller than that, so there is no need to
        #       consider the case that repeat times > 255
        line_rep_loop = ub_line_cut_ub // self.vector_size_elem
        line_rep_tail = ub_line_cut_ub % self.vector_size_elem
        # 256 here is to aligned the uint16's block (contains 16 uint16), and each uint16 represents 16 numbers
        line_loop4cmp = ub_line_cut_ub // 256
        bm_line_loop = bm_line_cut_ub // self.vector_size_elem
        bm_line_tail = bm_line_cut_ub % self.vector_size_elem

        # compute some values for ub-cut-tail process
        each_loop_step = ub_line_cut_ub // self.C0
        ubtail_begin_ind = cut_ub_loop * each_loop_step
        ubtail_line_rep_loop = cut_ub_tail // self.vector_size_elem
        ubtail_line_rep_tail = cut_ub_tail % self.vector_size_elem
        ubtail_line_loop4cmp = cut_ub_tail // 256
        ubtail_line_tail4cmp = cut_ub_tail % 256
        ubtail_bm_cut_ub = cut_ub_tail // 16
        ubtail_bm_line_loop = ubtail_bm_cut_ub // self.vector_size_elem
        ubtail_bm_line_tail = ubtail_bm_cut_ub % self.vector_size_elem

        input_gm_T = tik_instance.Tensor(dtype=self.input_dtype,
                                         shape=self.input_shape,
                                         name='input_tensor',
                                         scope=tik.scope_gm)
        output_gm_T = tik_instance.Tensor(dtype=self.input_dtype,
                                          shape=self.output_shape,
                                          name='output_tensor',
                                          scope=tik.scope_gm)
        output_gm_T = output_gm_T.reshape((self.N, self.d_step, self.C1, self.h_step * self.w_step * self.C0))

        # the output bitmask
        final_bitmask_gm_T = tik_instance.Tensor(
            dtype=self.out_mask_type,
            # the last two dimension of the shape aim to align to 32B.
            shape=[self.N, self.d_step, self.C1 * self.k_elem, self.aligned_bitmask_line // 16, 16],
            name='final_bitmask',
            scope=tik.scope_gm
        )
        # init a big matrix in ub, for register the img2col.
        bigmat_ub_T = tik_instance.Tensor(dtype=self.input_dtype,
                                          shape=[self.k_elem, ub_line_cut_ub],
                                          name='big_matrix',
                                          scope=tik.scope_ubuf)
        # init a register to store the max line.
        maxline_ub_T = tik_instance.Tensor(dtype=self.input_dtype,
                                           shape=[ub_line_cut_ub, ],
                                           name='max_line',
                                           scope=tik.scope_ubuf)
        # init a uint16 bitmask, each number represents 16 bits. (16 pairs of numbers)
        bitmask_ub_T = tik_instance.Tensor(dtype=self.out_mask_type,
                                           shape=[self.k_elem, bm_line_cut_ub],
                                           name='bitmask_ub',
                                           scope=tik.scope_ubuf)
        # init some tensors for deduplication during the inter process.
        maskor_ub_T = tik_instance.Tensor(dtype=self.out_mask_type,
                                          shape=[bm_line_cut_ub, ],
                                          name='mask_or',
                                          scope=tik.scope_ubuf)
        masknot_ub_T = tik_instance.Tensor(dtype=self.out_mask_type,
                                           shape=[bm_line_cut_ub, ],
                                           name='mask_not',
                                           scope=tik.scope_ubuf)

        # init some auxiliary tensors for aligned to 256 elem during vcmpv process.
        aux0_ub_T = tik_instance.Tensor(dtype=self.input_dtype,
                                        shape=[256, ],
                                        name='aux0',
                                        scope=tik.scope_ubuf)
        aux1_ub_T = tik_instance.Tensor(dtype=self.input_dtype,
                                        shape=[256, ],
                                        name='aux1',
                                        scope=tik.scope_ubuf)
        tik_instance.vec_dup(mask=self.mask4compute,
                             dst=aux0_ub_T,
                             scalar=0,
                             repeat_times=2,
                             dst_rep_stride=8)
        tik_instance.vec_dup(mask=self.mask4compute,
                             dst=aux1_ub_T,
                             scalar=0,
                             repeat_times=2,
                             dst_rep_stride=8)

        # state some scalar, aim to record the position of the inter compute process.
        cut_ub_ind_S = tik_instance.Scalar(dtype='int16', name='cut_ub_ind', init_value=0)
        move_flag_S = tik_instance.Scalar(dtype='int32', name='move_flag', init_value=0)
        ind_w_S = tik_instance.Scalar(dtype='int32', name='tail_ind_w')
        ind_h_S = tik_instance.Scalar(dtype='int32', name='tail_ind_h')

        with tik_instance.for_range(0, self.N) as loop_ind_N:
            with tik_instance.for_range(0, self.C1) as loop_ind_C1:
                with tik_instance.for_range(0, self.d_step) as loop_ind_d:
                    cut_ub_ind_S.set_as(0)
                    move_flag_S.set_as(0)
                    if cut_ub_loop != 0:
                        # begin to compute the ub-cut-loop process
                        with tik_instance.for_range(0, self.h_step) as loop_ind_h:
                            with tik_instance.for_range(0, self.w_step) as loop_ind_w:
                                # if the ub is full, compute and move out to gm.
                                with tik_instance.if_scope(move_flag_S == ub_line_cut_ub // 16):
                                    # begin to compute the bitmask each layer(each d dimension).
                                    # step 1: compute the max line of each kernel's position.
                                    move_flag_S.set_as(0)
                                    _move_ctn(dst=maxline_ub_T,
                                              src=bigmat_ub_T,
                                              data_blknum=ub_line_cut_ub * self.each_elem_bytes // self.block_size,
                                              inst=tik_instance)
                                    with tik_instance.for_range(1, self.k_elem) as line_ind:
                                        if line_rep_loop != 0:
                                            _compute_two_ctn(
                                                method=tik_instance.vmax,
                                                dst=maxline_ub_T[0],
                                                src0=maxline_ub_T,
                                                src1=bigmat_ub_T[line_ind, 0],
                                                rep_times=line_rep_loop
                                            )
                                        if line_rep_tail != 0:
                                            _compute_two_ctn(
                                                method=tik_instance.vmax,
                                                dst=maxline_ub_T[line_rep_loop * self.mask4compute],
                                                src0=maxline_ub_T[line_rep_loop * self.mask4compute],
                                                src1=bigmat_ub_T[line_ind, line_rep_loop * self.mask4compute],
                                                mask=line_rep_tail
                                            )
                                    # move the max_line_ub_T to gm
                                    _move_ctn(dst=output_gm_T[loop_ind_N, loop_ind_d, loop_ind_C1,
                                                              cut_ub_ind_S * ub_line_cut_ub],
                                              src=maxline_ub_T,
                                              data_blknum=ub_line_cut_ub * self.each_elem_bytes // self.block_size,
                                              inst=tik_instance)
                                    # step 2: compute the bitmask
                                    with tik_instance.for_range(0, self.k_elem) as line_ind:
                                        _cmp_ctn(inst=tik_instance,
                                                 dst=bitmask_ub_T[line_ind, 0],
                                                 src0=bigmat_ub_T[line_ind, 0],
                                                 src1=maxline_ub_T[0],
                                                 rep_times=line_loop4cmp * 2)
                                    # step 3: deduplicate the bitmask, each column must have at most one '1'.
                                    # first, init the masknot and maskor.
                                    if bm_line_loop != 0:
                                        _vnot_ctn(inst=tik_instance,
                                                  dst=masknot_ub_T,
                                                  src=bitmask_ub_T,
                                                  rep_times=bm_line_loop)
                                    if bm_line_tail != 0:
                                        _vnot_ctn(inst=tik_instance,
                                                  dst=masknot_ub_T[bm_line_loop * self.mask4compute],
                                                  src=bitmask_ub_T[0, bm_line_loop * self.mask4compute],
                                                  mask=bm_line_tail)
                                    _move_ctn(
                                        dst=maskor_ub_T,
                                        src=bitmask_ub_T,
                                        data_blknum=math.ceil(bm_line_cut_ub * self.each_elem_bytes / self.block_size),
                                        inst=tik_instance
                                    )
                                    with tik_instance.for_range(1, self.k_elem) as line_ind:
                                        if bm_line_loop != 0:
                                            _compute_two_ctn(method=tik_instance.vor,
                                                             dst=maskor_ub_T,
                                                             src0=maskor_ub_T,
                                                             src1=bitmask_ub_T[line_ind, 0],
                                                             rep_times=bm_line_loop)
                                            _compute_two_ctn(method=tik_instance.vand,
                                                             dst=bitmask_ub_T[line_ind, 0],
                                                             src0=bitmask_ub_T[line_ind, 0],
                                                             src1=masknot_ub_T,
                                                             rep_times=bm_line_loop)
                                            _vnot_ctn(inst=tik_instance,
                                                      dst=masknot_ub_T,
                                                      src=maskor_ub_T,
                                                      rep_times=bm_line_loop)
                                        if bm_line_tail != 0:
                                            _compute_two_ctn(
                                                method=tik_instance.vor,
                                                dst=maskor_ub_T[bm_line_loop * self.mask4compute],
                                                src0=maskor_ub_T[bm_line_loop * self.mask4compute],
                                                src1=bitmask_ub_T[line_ind, bm_line_loop * self.mask4compute],
                                                mask=bm_line_tail
                                            )
                                            _compute_two_ctn(
                                                method=tik_instance.vand,
                                                dst=bitmask_ub_T[line_ind, bm_line_loop * self.mask4compute],
                                                src0=bitmask_ub_T[line_ind, bm_line_loop * self.mask4compute],
                                                src1=masknot_ub_T[bm_line_loop * self.mask4compute],
                                                mask=bm_line_tail
                                            )
                                            _vnot_ctn(inst=tik_instance,
                                                      dst=masknot_ub_T[bm_line_loop * self.mask4compute],
                                                      src=maskor_ub_T[bm_line_loop * self.mask4compute],
                                                      mask=bm_line_tail)
                                    tik_instance.data_move(
                                        dst=final_bitmask_gm_T[loop_ind_N, loop_ind_d, loop_ind_C1 * self.k_elem,
                                                               cut_ub_ind_S * bm_line_cut_ub // 16, 0],
                                        src=bitmask_ub_T,
                                        sid=0,
                                        nburst=self.k_elem,
                                        burst=bm_line_cut_ub * self.each_elem_bytes // self.block_size,
                                        src_stride=0,
                                        dst_stride=(self.aligned_bitmask_line - bm_line_cut_ub) *
                                                   self.each_elem_bytes // self.block_size
                                    )
                                    cut_ub_ind_S.set_as(cut_ub_ind_S + 1)
                                # begin to do img2col process, this should be replaced by cube solution.
                                with tik_instance.new_stmt_scope(disable_sync=True):
                                    with tik_instance.for_range(0, self.k_d) as kd_tmp:
                                        # first, move kernel on each 2-D
                                        with tik_instance.for_range(0, self.k_h) as kh_tmp:
                                            tik_instance.data_move(
                                                dst=bigmat_ub_T[(kd_tmp * self.k_h + kh_tmp) * self.k_w,
                                                                move_flag_S * self.C0],
                                                # shape of input_gm_T: N D C1 H W C0
                                                src=input_gm_T[
                                                    loop_ind_N, loop_ind_d * self.stride_d + kd_tmp, loop_ind_C1,
                                                    loop_ind_h * self.stride_h + kh_tmp, loop_ind_w * self.stride_w, 0
                                                ],
                                                sid=0,
                                                nburst=self.k_w,
                                                burst=self.C0 * self.each_elem_bytes // self.block_size,
                                                src_stride=0,
                                                dst_stride=(ub_line_cut_ub // 16 - 1) * self.C0 *
                                                           self.each_elem_bytes // self.block_size
                                            )
                                move_flag_S.set_as(move_flag_S + 1)
                    if cut_ub_tail != 0:
                        # begin to compute the ub-cut-tail process.
                        # move tail data from gm to ub, actually it's not really full.
                        with tik_instance.for_range(ubtail_begin_ind, self.w_step * self.h_step) as tail_ind:
                            # compute the current position.
                            ind_h_S.set_as(tail_ind // self.w_step)
                            ind_w_S.set_as(tail_ind % self.w_step)
                            # img2col, will be replaced by cube solution(load3d) in future.
                            with tik_instance.new_stmt_scope(disable_sync=True):
                                with tik_instance.for_range(0, self.k_d) as kd_tmp:
                                    # first, move kernel on each 2-D
                                    with tik_instance.for_range(0, self.k_h) as kh_tmp:
                                        tik_instance.data_move(
                                            dst=bigmat_ub_T[(kd_tmp * self.k_h + kh_tmp) * self.k_w,
                                                            (tail_ind - ubtail_begin_ind) * self.C0],
                                            # shape of input_gm_T: N D C1 H W C0
                                            src=input_gm_T[
                                                loop_ind_N, loop_ind_d * self.stride_d + kd_tmp, loop_ind_C1,
                                                ind_h_S * self.stride_h + kh_tmp, ind_w_S * self.stride_w, 0
                                            ],
                                            sid=0,
                                            nburst=self.k_w,
                                            burst=self.C0 * self.each_elem_bytes // self.block_size,
                                            src_stride=0,
                                            dst_stride=(ub_line_cut_ub // 16 - 1) * self.C0 *
                                                       self.each_elem_bytes // self.block_size
                                        )
                        # begin to compute the bitmask each layer(each d dimension tail).
                        # step 1: compute the mask line of each kernel's position.
                        _move_ctn(dst=maxline_ub_T,
                                  src=bigmat_ub_T,
                                  data_blknum=cut_ub_tail * self.each_elem_bytes // self.block_size,
                                  inst=tik_instance)
                        with tik_instance.for_range(1, self.k_elem) as line_ind:
                            if ubtail_line_rep_loop != 0:
                                _compute_two_ctn(method=tik_instance.vmax,
                                                 dst=maxline_ub_T[0],
                                                 src0=maxline_ub_T[0],
                                                 src1=bigmat_ub_T[line_ind, 0],
                                                 rep_times=ubtail_line_rep_loop)
                            if ubtail_line_rep_tail != 0:
                                _compute_two_ctn(method=tik_instance.vmax,
                                                 dst=maxline_ub_T[ubtail_line_rep_loop * self.mask4compute],
                                                 src0=maxline_ub_T[ubtail_line_rep_loop * self.mask4compute],
                                                 src1=bigmat_ub_T[line_ind, ubtail_line_rep_loop * self.mask4compute],
                                                 mask=ubtail_line_rep_tail)
                        # move the maxline_ub_T to gm.
                        _move_ctn(dst=output_gm_T[loop_ind_N, loop_ind_d, loop_ind_C1, cut_ub_ind_S * ub_line_cut_ub],
                                  src=maxline_ub_T,
                                  data_blknum=cut_ub_tail * self.each_elem_bytes // self.block_size,
                                  inst=tik_instance)
                        # step 2: compute the bitmask.
                        if ubtail_line_tail4cmp != 0:
                            _move_ctn(dst=aux1_ub_T,
                                      src=maxline_ub_T[ubtail_line_loop4cmp * 256],
                                      data_blknum=ubtail_line_tail4cmp * self.each_elem_bytes // self.block_size,
                                      inst=tik_instance)
                        with tik_instance.for_range(0, self.k_elem) as line_ind:
                            if ubtail_line_loop4cmp != 0:
                                _cmp_ctn(inst=tik_instance,
                                         dst=bitmask_ub_T[line_ind, 0],
                                         src0=bigmat_ub_T[line_ind, 0],
                                         src1=maxline_ub_T,
                                         rep_times=ubtail_line_loop4cmp * 2)
                            if ubtail_line_tail4cmp != 0:
                                # use aux0 and aux1 to align tail to 256 elem for vcmpv.
                                _move_ctn(
                                    dst=aux0_ub_T,
                                    src=bigmat_ub_T[line_ind, ubtail_line_loop4cmp * 256],
                                    data_blknum=ubtail_line_tail4cmp * self.each_elem_bytes // self.block_size,
                                    inst=tik_instance
                                )
                                _cmp_ctn(inst=tik_instance,
                                         dst=bitmask_ub_T[line_ind, ubtail_line_loop4cmp * 256 // 16],
                                         src0=aux0_ub_T,
                                         src1=aux1_ub_T,
                                         rep_times=2)
                        # step 3: deduplicate the bitmask, each column must have at most one '1'.
                        # first, init the masknot and maskor.
                        if ubtail_bm_line_loop != 0:
                            _vnot_ctn(inst=tik_instance,
                                      dst=masknot_ub_T,
                                      src=bitmask_ub_T,
                                      rep_times=ubtail_bm_line_loop)
                        if ubtail_bm_line_tail != 0:
                            _vnot_ctn(inst=tik_instance,
                                      dst=masknot_ub_T[ubtail_bm_line_loop * self.mask4compute],
                                      src=bitmask_ub_T[0, ubtail_bm_line_loop * self.mask4compute],
                                      mask=ubtail_bm_line_tail)
                        _move_ctn(dst=maskor_ub_T,
                                  src=bitmask_ub_T,
                                  data_blknum=math.ceil(ubtail_bm_cut_ub * self.each_elem_bytes / self.block_size),
                                  inst=tik_instance)
                        with tik_instance.for_range(1, self.k_elem) as line_ind:
                            if ubtail_bm_line_loop != 0:
                                _compute_two_ctn(method=tik_instance.vor,
                                                 dst=maskor_ub_T,
                                                 src0=maskor_ub_T,
                                                 src1=bitmask_ub_T[line_ind, 0],
                                                 rep_times=ubtail_bm_line_loop)
                                _compute_two_ctn(method=tik_instance.vand,
                                                 dst=maskor_ub_T,
                                                 src0=maskor_ub_T,
                                                 src1=bitmask_ub_T[line_ind, 0],
                                                 rep_times=ubtail_bm_line_loop)
                                _vnot_ctn(inst=tik_instance,
                                          dst=masknot_ub_T,
                                          src=maskor_ub_T,
                                          rep_times=ubtail_bm_line_loop)
                            if ubtail_bm_line_tail != 0:
                                _compute_two_ctn(method=tik_instance.vor,
                                                 dst=maskor_ub_T[ubtail_bm_line_loop * self.mask4compute],
                                                 src0=maskor_ub_T[ubtail_bm_line_loop * self.mask4compute],
                                                 src1=bitmask_ub_T[line_ind, ubtail_bm_line_loop * self.mask4compute],
                                                 mask=ubtail_bm_line_tail)
                                _compute_two_ctn(method=tik_instance.vand,
                                                 dst=bitmask_ub_T[line_ind, ubtail_bm_line_loop * self.mask4compute],
                                                 src0=bitmask_ub_T[line_ind, ubtail_bm_line_loop * self.mask4compute],
                                                 src1=masknot_ub_T[ubtail_bm_line_loop * self.mask4compute],
                                                 mask=ubtail_bm_line_tail)
                                _vnot_ctn(inst=tik_instance,
                                          dst=masknot_ub_T[ubtail_bm_line_loop * self.mask4compute],
                                          src=maskor_ub_T[ubtail_bm_line_loop * self.mask4compute],
                                          mask=ubtail_bm_line_tail)
                        tik_instance.data_move(
                            dst=final_bitmask_gm_T[loop_ind_N, loop_ind_d, loop_ind_C1 * self.k_elem,
                                                   cut_ub_ind_S * bm_line_cut_ub // 16, 0],
                            src=bitmask_ub_T,
                            sid=0,
                            nburst=self.k_elem,
                            burst=math.ceil(ubtail_bm_cut_ub * self.each_elem_bytes / self.block_size),
                            src_stride=(bm_line_cut_ub - ubtail_bm_cut_ub) *
                                       self.each_elem_bytes // self.block_size,
                            dst_stride=0
                        )
        output_gm_T = output_gm_T.reshape(self.output_shape)
        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[input_gm_T],
                              outputs=[output_gm_T, final_bitmask_gm_T])
        return tik_instance


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
    if ceil_mode is not False:
        raise RuntimeError("current version only support ceil_mode=False!")
    if argmax_type != "bitmask":
        raise RuntimeError("current version only support bitmask argmax, please set argmax_type=bitmask!")


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
    obj = MaxPool3DWithArgmax(x, kernel_size, strides, kernel_name)
    return obj.run()

