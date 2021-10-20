# Copyright 2019 Huawei Technologies Co., Ltd
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
max_pool_grad_with_argmax_cut_one_h_v1
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl import constant_util_v1 as constant
from impl.dynamic.max_pool_grad_with_argmax_cut_w_v1 import MaxpoolGardObject


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for Constant
    """
    # size of vector calc one repeat
    ONE_REPEAT = 256
    # max repeat of vector calc
    V_MAX_REPEAT = 255
    # max num of fp16 in one repeat
    FP16_MAX = 128
    # max num of fp32 in one repeat
    FP32_MAX = 64
    # max num of fp16 mask handle one time
    MASK_MAX = 8
    BLOCK_SIZE = 32
    L1_SIZE = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


# 'pylint: disable=too-many-arguments,useless-super-delegation,super-with-arguments
# 'pylint: disable=redefined-builtin,too-many-locals,simplifiable-if-expression,too-many-branches
# 'pylint: disable=too-many-statements,literal-comparison
class MaxpoolGradCustom(MaxpoolGardObject):
    """
    parameter for max_pool_grad_with_pool
    """

    def __init__(self, grad, argmax, input_x, ksize, strides, padding, dilation, ceil_mode):
        """
        init compare and bit pack base parameters
        Parameters
        ----------
        input_x: input of maxpool, useless for maxpool gard
        grad: input of maxpoolgard or output of maxpool
        argmax:output of maxpool mask or index
        strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
        padding: pad
        Returns
        -------
        None
        """
        super(MaxpoolGradCustom, self).__init__(grad, argmax, input_x, ksize, strides, padding,
                                                dilation, ceil_mode)

    def _data_move_from_out_to_l1buf(self, output_l1, input_data, src_addr, length, dtype_size):
        repeat_time = length * dtype_size // Constant.BLOCK_SIZE
        max_repeat_time = 65535

        with self.tik_instance.if_scope(repeat_time <= max_repeat_time):
            self.tik_instance.data_move(output_l1, input_data[src_addr], constant.SID,
                                        constant.DEFAULT_NBURST, repeat_time,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        with self.tik_instance.else_scope():
            iter_times = self.tik_instance.Scalar("int32")
            iter_times.set_as(repeat_time // max_repeat_time)
            single_max_repeat_len = max_repeat_time * Constant.BLOCK_SIZE // dtype_size
            res_iter_times = repeat_time - iter_times * max_repeat_time
            with self.tik_instance.for_range(0, iter_times) as iter:
                self.tik_instance.data_move(
                    output_l1[iter * single_max_repeat_len], input_data[src_addr + iter * single_max_repeat_len],
                    constant.SID, constant.DEFAULT_NBURST, repeat_time,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            with self.tik_instance.if_scope(res_iter_times > 0):
                self.tik_instance.data_move(
                    output_l1[iter_times * single_max_repeat_len],
                    input_data[src_addr + iter_times * single_max_repeat_len],
                    constant.SID, constant.DEFAULT_NBURST, res_iter_times,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def tik_instance_cut_nc1_cut_one_h(self, block_id):
        """
        get vector instruct repeat times

        Parameters
        ----------
        kernel_name: cce kernel name, default value is "maxpoolGradWithArgmax"
        Returns
        -------
        None
        """
        dtype = self.dtype
        dtype_size = self.dtype_size

        max_mem_size0 = 8064
        max_mem_size1 = 32500
        max_mem_size2 = 32384
        max_mem_size3 = 1024
        max_mem_size4 = 8064
        max_mem_size5 = 8064

        mask_one_window = ((self.dyh * self.dyw + 15) // 16 + 1) * 16
        col2img_dyw = (self.dyw + 15) // 16 * 16

        # vector_repeat_time
        v_rep_time = col2img_dyw * self.channel * dtype_size // Constant.ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // Constant.V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % Constant.V_MAX_REPEAT

        # when every looph move data after, then dup col2img data
        v_rep_time_col = \
            (2 * self.col2img_w * self.channel * self.col2img_h * dtype_size + Constant.ONE_REPEAT - 1) // \
            Constant.ONE_REPEAT
        v_rep_cycle_col = v_rep_time_col // Constant.V_MAX_REPEAT
        v_rep_last_col = v_rep_time_col % Constant.V_MAX_REPEAT

        real_cycle = self.tik_instance.Scalar("int32")
        block_base = self.tik_instance.Scalar("int32")
        block_num = self.tik_instance.Scalar("int32")
        src_address = self.tik_instance.Scalar("int32")
        dst_address = self.tik_instance.Scalar("int32")
        nburst = self.tik_instance.Scalar("int32")
        burst_len = self.tik_instance.Scalar("int32")
        src_stride = self.tik_instance.Scalar("int32")
        dst_stride = self.tik_instance.Scalar("int32")
        dxh_address_offset = self.tik_instance.Scalar("int32")
        dxh_calcline = self.tik_instance.Scalar("int32")
        left_top_w = 0
        left_top_h = 0

        true_val = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.dyh * self.dyw * self.channel > (Constant.L1_SIZE // 2)):
            true_val.set_as(0)
        with self.tik_instance.else_scope():
            true_val.set_as(1)

        data_input_l1 = self.tik_instance.Tensor(dtype, (self.dyh * self.dyw * self.channel,), name="data_input_l12",
                                                 scope=tik.scope_cbuf, max_mem_size=524256)
        data_mask_l1 = self.tik_instance.Tensor("uint16", (self.kernel_h * self.kernel_w * mask_one_window,),
                                                name="data_mask_l12", scope=tik.scope_cbuf, max_mem_size=524256)

        with self.tik_instance.if_scope(block_id < self.block_index):
            real_cycle.set_as(self.block_cycle + 1)
            block_base.set_as(block_id * real_cycle)
        with self.tik_instance.else_scope():
            real_cycle.set_as(self.block_cycle)
            block_base.set_as(self.block_index + block_id * self.block_cycle)
        with self.tik_instance.for_range(0, real_cycle) as cycle_id:
            block_num.set_as(block_base + cycle_id)
            data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,), name="data_vsel_ub_zero2",
                                                         scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_vsel_ub_zero[0], self.data_input_origin[0], constant.SID,
                                        constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)
            # vector_dup ub every time

            dxh_address_offset.set_as(0)
            dxh_calcline.set_as(0)

            data_max_ub = self.tik_instance.Tensor(dtype, (max_mem_size0,), name="data_max_ub2",
                                                   scope=tik.scope_ubuf)  # 256
            data_vmul_ub_col2img_fp32 = \
                self.tik_instance.Tensor("float32", (max_mem_size1,),  # 24 x 16 x 9 + 64
                                         name="data_vmul_ub_col2img_fp322", scope=tik.scope_ubuf)
            data_vmul_ub_col2img_fp16 = \
                self.tik_instance.Tensor(dtype, (max_mem_size2,),  # 24 x 16 x 9 + 128
                                         name="data_vmul_ub_col2img_fp162", scope=tik.scope_ubuf)
            self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
            self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)
            self.clean_fp16_multi_repeat(data_max_ub, dtype_size)

            with self.tik_instance.if_scope(true_val == 1):
                self._data_move_from_out_to_l1buf(data_input_l1, self.data_input, block_num * self.dyh * self.dyw *
                                                  self.channel, self.dyh * self.dyw * self.channel, dtype_size)
            with self.tik_instance.for_range(0, self.dyh) as looph:  # 5
                with self.tik_instance.if_scope(true_val == 1):
                    self.tik_instance.data_move(data_max_ub, data_input_l1[looph * self.dyw * self.channel],
                                                constant.SID, constant.DEFAULT_NBURST,
                                                self.dyw * self.channel * dtype_size // Constant.BLOCK_SIZE,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        data_max_ub, self.data_input[(block_num * self.dyh + looph) * self.dyw * self.channel],
                        constant.SID, constant.DEFAULT_NBURST,
                        self.dyw * self.channel * dtype_size // Constant.BLOCK_SIZE,
                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                # mask define
                data_mask_ub = self.tik_instance.Tensor("uint16", (max_mem_size3,), name="data_mask_ub2",
                                                        scope=tik.scope_ubuf)
                data_vsel_ub = self.tik_instance.Tensor(dtype, (max_mem_size4,),
                                                        name="data_vsel_ub2", scope=tik.scope_ubuf)
                data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (max_mem_size5,),
                                                             name="data_vsel_ub_fp322", scope=tik.scope_ubuf)
                with self.tik_instance.if_scope(true_val == 1):
                    self._data_move_from_out_to_l1buf(
                        data_mask_l1, self.data_mask,
                        block_num * mask_one_window * self.kernel_w * self.kernel_h + looph * self.dyw,
                        mask_one_window * self.kernel_w * self.kernel_h, dtype_size)

                self.clean_fp16_multi_repeat(data_vsel_ub, dtype_size)
                self.clean_fp32_multi_repeat(data_vsel_ub_fp32, dtype_size)
                self.clean_fp16_multi_repeat(data_mask_ub, dtype_size)

                with self.tik_instance.for_range(0, self.kernel_h * self.kernel_w) as mask_id:  # 169
                    with self.tik_instance.if_scope(true_val == 1):
                        self.tik_instance.data_move(data_mask_ub,
                                                    data_mask_l1[mask_id * mask_one_window],
                                                    constant.SID, 1, col2img_dyw * dtype_size // Constant.BLOCK_SIZE,
                                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(data_mask_ub,
                                                    self.data_mask[
                                                        block_num * mask_one_window * self.kernel_w * self.kernel_h +
                                                        looph * self.dyw + mask_id * mask_one_window],
                                                    constant.SID, 1, col2img_dyw * dtype_size // Constant.BLOCK_SIZE,
                                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                    with self.tik_instance.if_scope(v_rep_time > 0):
                        with self.tik_instance.for_range(0, v_rep_time, thread_num=1) as cycle:
                            cmpmask = self.tik_instance.mov_tensor_to_cmpmask(data_mask_ub[cycle * Constant.MASK_MAX])
                            self.tik_instance.vsel(constant.MASK128, 0, data_vsel_ub[cycle * Constant.FP16_MAX],
                                                   cmpmask, data_max_ub[cycle * Constant.FP16_MAX],
                                                   data_vsel_ub_zero[0], constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                                                   constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                   constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT,
                                                   constant.REPEAT_STRIDE_EIGHT)

                    # fp16 to fp32
                    with self.tik_instance.if_scope(v_rep_cycle_fp32 > 0):
                        with self.tik_instance.for_range(0, v_rep_cycle_fp32, thread_num=1) as cycle:
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vsel_ub_fp32[
                                                        cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                                    data_vsel_ub[cycle * Constant.V_MAX_REPEAT * Constant.FP16_MAX],
                                                    Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_EIGHT,
                                                    constant.REPEAT_STRIDE_FOUR)
                    with self.tik_instance.if_scope(v_rep_last_fp32 != 0):
                        self.tik_instance.vconv(
                            constant.MASK64, "",
                            data_vsel_ub_fp32[v_rep_cycle_fp32 * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                            data_vsel_ub[v_rep_cycle_fp32 * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                            v_rep_last_fp32, constant.STRIDE_ONE, constant.STRIDE_ONE,
                            constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)

                    fetch_filter_w = mask_id % self.kernel_w
                    fetch_filter_h = mask_id // self.kernel_w
                    self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0], data_vsel_ub_fp32[0],
                                              (0, 0, 0, 0), self.col2img_h, self.col2img_w, fetch_filter_w,
                                              fetch_filter_h, left_top_w, left_top_h, self.stride_w, self.stride_h,
                                              self.kernel_w, self.kernel_h, 1, 1, col2img_dyw // 16)

                with self.tik_instance.if_scope(v_rep_cycle_col > 0):
                    with self.tik_instance.for_range(0, v_rep_cycle_col, thread_num=1) as cycle:
                        self.tik_instance.vconv(
                            constant.MASK64, "",
                            data_vmul_ub_col2img_fp16[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                            data_vmul_ub_col2img_fp32[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                            Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                            constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
                with self.tik_instance.if_scope(v_rep_last_col != 0):
                    self.tik_instance.vconv(
                        constant.MASK64, "",
                        data_vmul_ub_col2img_fp16[v_rep_cycle_col * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                        data_vmul_ub_col2img_fp32[v_rep_cycle_col * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                        v_rep_last_col, constant.STRIDE_ONE, constant.STRIDE_ONE,
                        constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)

                with self.tik_instance.if_scope(self.hoverlap == 0):
                    # move ub to gm
                    src_address.set_as(self.pad_left * self.channel)
                    dst_address.set_as(block_num * self.dxh * self.dxw * self.channel +
                                       (looph * self.col2img_h - self.pad_top) * self.dxw * self.channel)
                    nburst.set_as(self.col2img_h)
                    burst_len.set_as(self.offset_w)
                    src_stride.set_as(self.col2img_w - self.offset_w)
                    dst_stride.set_as(self.dxw - self.offset_w)
                    with self.tik_instance.if_scope(looph == 0):
                        src_address.set_as(src_address + self.pad_top * self.col2img_w * self.channel)
                        dst_address.set_as(block_num * self.dxh * self.dxw * self.channel)
                        nburst.set_as(nburst - self.pad_top)
                        with self.tik_instance.if_scope(looph == self.dyh - 1):
                            nburst.set_as(self.dxh)

                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(looph == self.dyh - 1):
                            nburst.set_as(self.dxh - self.col2img_h * looph + self.pad_top)

                    self.tik_instance.data_move(self.data_output[dst_address], data_vmul_ub_col2img_fp16[src_address],
                                                constant.SID, nburst, burst_len, src_stride, dst_stride)
                    with self.tik_instance.if_scope(v_rep_cycle_col > 0):
                        with self.tik_instance.for_range(0, v_rep_cycle_col, thread_num=1) as cycle:
                            self.tik_instance.vector_dup(
                                constant.MASK64,
                                data_vmul_ub_col2img_fp32[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                0, Constant.V_MAX_REPEAT, constant.STRIDE_ONE,
                                constant.REPEAT_STRIDE_EIGHT)
                    with self.tik_instance.if_scope(v_rep_last_col != 0):
                        self.tik_instance.vector_dup(
                            constant.MASK64,
                            data_vmul_ub_col2img_fp32[v_rep_cycle_col * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                            0, v_rep_last_col, constant.STRIDE_ONE,
                            constant.REPEAT_STRIDE_EIGHT)
                with self.tik_instance.else_scope():
                    src_address.set_as(self.pad_left * self.channel)
                    dst_address.set_as(block_num * self.dxh * self.dxw * self.channel + dxh_address_offset)
                    nburst.set_as(self.stride_h)
                    with self.tik_instance.if_scope(looph * self.stride_h < self.pad_top):
                        nburst.set_as((looph + 1) * self.stride_h - self.pad_top)
                        src_address.set_as(src_address +
                                           (self.pad_top - looph * self.stride_h) * self.col2img_w * self.channel)
                    with self.tik_instance.if_scope(tik.all(dxh_calcline < self.dxh, looph == self.dyh - 1)):
                        nburst.set_as(self.dxh - dxh_calcline)

                    burst_len.set_as(self.offset_w)
                    src_stride.set_as(self.col2img_w - self.offset_w)
                    dst_stride.set_as(self.dxw - self.offset_w)
                    with self.tik_instance.if_scope(nburst > 0):
                        self.tik_instance.data_move(self.data_output[dst_address],
                                                    data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, nburst,
                                                    burst_len, src_stride,
                                                    dst_stride)
                    with self.tik_instance.else_scope():
                        nburst.set_as(0)

                    dxh_address_offset.set_as(dxh_address_offset + nburst * self.dxw * self.channel)
                    dxh_calcline.set_as(dxh_calcline + nburst)

                    # dma_copy ub to ub
                    self.tik_instance.data_move(data_vmul_ub_col2img_fp32[0],
                                                data_vmul_ub_col2img_fp32[
                                                    self.stride_h * self.channel * self.col2img_w],
                                                constant.SID, self.hoverlap, 2 * self.col2img_w,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                    start_pos = self.hoverlap * self.channel * self.col2img_w
                    res_len = data_vmul_ub_col2img_fp32.shape[0] - start_pos
                    self.clear_res_fp32_ub(data_vmul_ub_col2img_fp32, start_pos, res_len, dtype_size * 2)

    def tik_instance_cut_nc1h_cut_one_h(self, block_id):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """

        dtype = self.dtype
        dtype_size = self.dtype_size

        stride_h, stride_w = self.strides[1:3]
        stridehw = stride_h * stride_w

        if stridehw == 1:
            max_mem_size0 = 16256
            max_mem_size1 = 16448
            max_mem_size2 = 16512
            max_mem_size3 = 9136
            max_mem_size4 = 16256
            max_mem_size5 = 16256
        else:
            max_mem_size0 = 5760
            max_mem_size1 = 32368
            max_mem_size2 = 32432
            max_mem_size3 = 3280
            max_mem_size4 = 5760
            max_mem_size5 = 5760

        wo_max = self.tik_instance.Scalar("int32")
        wo_max.set_as(self.wo_max)
        mask_one_window = ((self.dyh * self.dyw + 15) // 16 + 1) * 16
        mask_stride = (mask_one_window - wo_max) // 16

        # vector_repeat_time
        v_rep_time = wo_max * self.channel * dtype_size // Constant.ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // Constant.V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % Constant.V_MAX_REPEAT

        real_cycle = self.tik_instance.Scalar("int32")
        block_base = self.tik_instance.Scalar("int32")
        block_num = self.tik_instance.Scalar("int32")
        in_src_address = self.tik_instance.Scalar("int32")
        mask_address = self.tik_instance.Scalar("int32")
        block_batch = self.tik_instance.Scalar("int32")
        block_h = self.tik_instance.Scalar("int32")
        h_cycle = self.tik_instance.Scalar("int32")
        output_cuthline = self.tik_instance.Scalar("int32")
        dxh_address_offset = self.tik_instance.Scalar("int32")
        src_address = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(block_id < self.block_index):
            real_cycle.set_as(self.block_cycle + 1)
            block_base.set_as(block_id * real_cycle)
        with self.tik_instance.else_scope():
            real_cycle.set_as(self.block_cycle)
            block_base.set_as(self.block_index + block_id * self.block_cycle)
        with self.tik_instance.for_range(0, real_cycle) as cycle_id:
            block_num.set_as(block_base + cycle_id)

            data_vsel_ub_zero = self.tik_instance.Tensor(
                dtype, (128,), name="data_vsel_ub_zero3", scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_vsel_ub_zero[0], self.data_input_origin[0], constant.SID,
                                        constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)

            block_batch.set_as(block_num // self.ho_count)
            block_h.set_as(block_num % self.ho_count)
            h_cycle.set_as(self.ho_every)
            with self.tik_instance.if_scope(block_h == self.ho_count - 1):
                h_cycle.set_as(self.ho_last)

            dxh_address_offset.set_as(block_batch * self.dxh * self.dxw * self.channel)

            with self.tik_instance.if_scope(self.hoverlap is 0):
                with self.tik_instance.if_scope(block_h != 0):
                    dxh_address_offset.set_as(
                        dxh_address_offset +
                        (block_h * self.ho_every * self.kernel_h - self.pad_top) * self.dxw * self.channel)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(block_h != 0):
                    dxh_address_offset.set_as(
                        dxh_address_offset +
                        ((block_h * (self.ho_every - 1) + 1) * self.stride_h - self.pad_top) * self.dxw * self.channel)
            # vector_dup ub every time
            data_max_ub = self.tik_instance.Tensor(dtype, (max_mem_size0,), name="data_max_ub3",
                                                   scope=tik.scope_ubuf)
            self.clean_max_ub(data_max_ub, dtype)

            data_vmul_ub_col2img_fp32 = \
                self.tik_instance.Tensor("float32", (max_mem_size1,),
                                         name="data_vmul_ub_col2img_fp323", scope=tik.scope_ubuf)
            data_vmul_ub_col2img_fp16 = \
                self.tik_instance.Tensor(dtype, (max_mem_size2,),
                                         name="data_vmul_ub_col2img_fp163", scope=tik.scope_ubuf)
            self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
            self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)

            # mask define
            data_mask_ub = self.tik_instance.Tensor("uint16", (max_mem_size3,),
                                                    name="data_mask_ub3", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, h_cycle) as looph:  # 3
                # address  not  multiplex
                in_src_address.set_as(block_batch * self.dyh * self.dyw * self.channel)
                mask_address.set_as(block_batch * mask_one_window * self.kernel_w * self.kernel_h)
                with self.tik_instance.if_scope(self.hoverlap == 0):
                    in_src_address.set_as(in_src_address + (block_h * self.ho_every + looph) * self.dyw * self.channel)
                    mask_address.set_as(mask_address + (block_h * self.ho_every + looph) * self.dyw)
                with self.tik_instance.else_scope():
                    in_src_address.set_as(
                        in_src_address + (block_h * (self.ho_every - 1) + looph) * self.dyw * self.channel)
                    mask_address.set_as(mask_address + (block_h * (self.ho_every - 1) + looph) * self.dyw)

                # dy copy gm to ub, every 1 lines per repeat
                self.tik_instance.data_move(data_max_ub, self.data_input[in_src_address], constant.SID,
                                            constant.DEFAULT_NBURST,
                                            self.dyw * self.channel * dtype_size // Constant.BLOCK_SIZE,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                self.clean_fp16_multi_repeat(data_mask_ub, dtype_size)

                # mask copy gm to ub
                self.tik_instance.data_move(data_mask_ub, self.data_mask[mask_address], constant.SID,
                                            self.kernel_h * self.kernel_w, wo_max * dtype_size // Constant.BLOCK_SIZE,
                                            mask_stride, constant.STRIDE_ZERO)

                data_vsel_ub = self.tik_instance.Tensor(dtype, (max_mem_size4,),
                                                        name="data_vsel_ub3", scope=tik.scope_ubuf)
                data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (max_mem_size5,),
                                                             name="data_vsel_ub_fp323", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, self.kernel_h * self.kernel_w) as mask_id:
                    with self.tik_instance.if_scope(v_rep_time > 0):  # 336  divides to 8
                        with self.tik_instance.for_range(0, v_rep_time) as cycle:
                            cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                data_mask_ub[wo_max * mask_id + cycle * Constant.MASK_MAX])
                            self.tik_instance.vsel(
                                constant.MASK128, 0, data_vsel_ub[cycle * Constant.FP16_MAX], cmpmask,
                                data_max_ub[cycle * Constant.FP16_MAX], data_vsel_ub_zero[0], constant.REPEAT_TIME_ONCE,
                                constant.STRIDE_ONE, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_EIGHT)

                    # fp16 to fp32
                    with self.tik_instance.if_scope(v_rep_cycle_fp32 > 0):
                        with self.tik_instance.for_range(0, v_rep_cycle_fp32) as cycle:
                            self.tik_instance.vconv(
                                constant.MASK64, "",
                                data_vsel_ub_fp32[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                data_vsel_ub[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                                Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
                    with self.tik_instance.if_scope(v_rep_last_fp32 != 0):  # 336 * 16 divides to 64 equals to 84
                        self.tik_instance.vconv(
                            constant.MASK64, "",
                            data_vsel_ub_fp32[v_rep_cycle_fp32 * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                            data_vsel_ub[v_rep_cycle_fp32 * Constant.V_MAX_REPEAT * Constant.FP32_MAX], v_rep_last_fp32,
                            constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                            constant.REPEAT_STRIDE_FOUR)

                    # col2img
                    fetch_filter_w = mask_id % self.kernel_w
                    fetch_filter_h = mask_id // self.kernel_w
                    self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0], data_vsel_ub_fp32[0], (0, 0, 0, 0),
                                              self.col2img_h, self.col2img_w, fetch_filter_w, fetch_filter_h, 0,
                                              0, self.stride_w, self.stride_h, self.kernel_w, self.kernel_h, 1, 1,
                                              wo_max // 16)

                output_cuthline.set_as(self.stride_h)
                src_address.set_as(self.pad_left * self.channel)
                with self.tik_instance.if_scope(self.hoverlap == 0):
                    with self.tik_instance.if_scope(block_h == 0):
                        with self.tik_instance.if_scope(looph == 0):
                            output_cuthline.set_as(output_cuthline - self.pad_top)
                    with self.tik_instance.if_scope(block_h == self.ho_count - 1):
                        with self.tik_instance.if_scope(looph == h_cycle - 1):
                            output_cuthline.set_as(output_cuthline - self.pad_bottom)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(block_h == 0):
                        with self.tik_instance.if_scope(looph == 0):
                            output_cuthline.set_as(output_cuthline - self.pad_top)
                            src_address.set_as(src_address + self.pad_top * self.col2img_w * self.channel)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(looph == 0):
                            output_cuthline.set_as(0)
                    with self.tik_instance.if_scope(block_h == self.ho_count - 1):
                        with self.tik_instance.if_scope(looph == h_cycle - 1):
                            output_cuthline.set_as(self.kernel_h)

                # fp32 to fp16
                v_rep_time_col = \
                    2 * (self.col2img_w * self.channel * self.col2img_h + 64) * dtype_size // Constant.ONE_REPEAT
                v_rep_cycle_col = v_rep_time_col // Constant.V_MAX_REPEAT
                v_rep_last_col = v_rep_time_col % Constant.V_MAX_REPEAT
                with self.tik_instance.if_scope(v_rep_cycle_col > 0):  # 1
                    with self.tik_instance.for_range(0, v_rep_cycle_col, thread_num=1) as cycle:
                        self.tik_instance.vconv(
                            constant.MASK64, "",
                            data_vmul_ub_col2img_fp16[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                            data_vmul_ub_col2img_fp32[cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                            Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                            constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
                with self.tik_instance.if_scope(v_rep_last_col != 0):
                    self.tik_instance.vconv(
                        constant.MASK64, "",
                        data_vmul_ub_col2img_fp16[v_rep_cycle_col * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                        data_vmul_ub_col2img_fp32[v_rep_cycle_col * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                        v_rep_last_col, constant.STRIDE_ONE, constant.STRIDE_ONE,
                        constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
                with self.tik_instance.if_scope(output_cuthline != 0):
                    self.tik_instance.data_move(self.data_output[dxh_address_offset],
                                                data_vmul_ub_col2img_fp16[src_address],
                                                constant.SID, output_cuthline, self.offset_w,
                                                self.col2img_w - self.offset_w, self.dxw - self.offset_w)
                    dxh_address_offset.set_as(dxh_address_offset + output_cuthline * self.dxw * self.channel)
                with self.tik_instance.if_scope(self.hoverlap != 0):
                    with self.tik_instance.if_scope(looph + 1 != h_cycle):
                        self.tik_instance.data_move(data_vmul_ub_col2img_fp32[0],
                                                    data_vmul_ub_col2img_fp32[
                                                        self.stride_h * self.channel * self.col2img_w],
                                                    constant.SID, constant.DEFAULT_NBURST,
                                                    2 * self.hoverlap * self.col2img_w, constant.STRIDE_ZERO,
                                                    constant.STRIDE_ZERO)
                        start_pos = self.hoverlap * self.channel * self.col2img_w
                        res_len = data_vmul_ub_col2img_fp32.shape[0] - start_pos
                        self.clear_res_fp32_ub(data_vmul_ub_col2img_fp32, start_pos, res_len, dtype_size * 2)
                with self.tik_instance.else_scope():
                    self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)

    def clear_res_fp32_ub(self, data_vmul_ub_col2img_fp32, start_pos, res_len, dtype_size):
        """
        The fun just for clean ub
        """
        v_rep_clear_time = res_len * dtype_size // Constant.ONE_REPEAT
        v_rep_clear_cycle = v_rep_clear_time // Constant.V_MAX_REPEAT  # 1
        v_rep_clear_last = v_rep_clear_time % Constant.V_MAX_REPEAT  # 250
        v_res_last = \
            v_rep_clear_cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX + v_rep_clear_last * Constant.FP32_MAX
        v_res_time = res_len - v_res_last
        with self.tik_instance.if_scope(v_rep_clear_cycle > 0):
            with self.tik_instance.for_range(0, v_rep_clear_cycle, thread_num=1) as cycle:
                self.tik_instance.vector_dup(
                    constant.MASK64,
                    data_vmul_ub_col2img_fp32[start_pos + cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                    0, Constant.V_MAX_REPEAT, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        with self.tik_instance.if_scope(v_rep_clear_last != 0):
            self.tik_instance.vector_dup(
                constant.MASK64,
                data_vmul_ub_col2img_fp32[start_pos + v_rep_clear_cycle * Constant.V_MAX_REPEAT * Constant.FP32_MAX],
                0, v_rep_clear_last, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        with self.tik_instance.if_scope(v_res_time != 0):
            res_repeat_time = v_res_time * dtype_size // constant.BLOCK_SIZE
            self.tik_instance.vector_dup(
                8, data_vmul_ub_col2img_fp32[start_pos + v_res_last], 0, res_repeat_time, constant.STRIDE_ONE, 1)
