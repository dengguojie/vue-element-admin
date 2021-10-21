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
import math
from te import tik
from te.platform.cce_conf import CceProductParams
from impl import constant_util_v1 as constant
from impl.max_pool_grad_with_argmax_cut_w_v1 import MaxpoolGardObject

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
L1_SIZE = CceProductParams().getParams("L1_Buffer")


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
        repeat_time = length * dtype_size // BLOCK_SIZE
        max_repeat_time = 65535

        if repeat_time <= max_repeat_time:
            self.tik_instance.data_move(output_l1, input_data[src_addr], constant.SID,
                                        constant.DEFAULT_NBURST, repeat_time,
                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        else:
            iter_times = int(repeat_time // max_repeat_time)
            single_max_repeat_len = max_repeat_time * BLOCK_SIZE // dtype_size
            res_iter_times = repeat_time - iter_times * max_repeat_time
            with self.tik_instance.for_range(0, iter_times) as iter:
                self.tik_instance.data_move(
                    output_l1[iter * single_max_repeat_len], input_data[src_addr + iter * single_max_repeat_len],
                    constant.SID, constant.DEFAULT_NBURST, repeat_time,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
            if res_iter_times > 0:
                self.tik_instance.data_move(
                    output_l1[iter_times * single_max_repeat_len],
                    input_data[src_addr + iter_times * single_max_repeat_len],
                    constant.SID, constant.DEFAULT_NBURST, res_iter_times,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def tik_instance_cut_nc1_cut_one_h(self, kernel_name):
        """
        get vector instruct repeat times

        Parameters
        ----------
        kernel_name: cce kernel name, default value is "maxpoolGradWithArgmax"
        Returns
        -------
        None
        """
        batch, channel1, dyh, dyw, channel = self.input_gard_shape
        dxh, dxw = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        if strideh > dxh:
            strideh = dxh

        if stridew > dxw:
            stridew = dxw

        dtype = self.dtype
        dtype_size = self.dtype_size
        windowh, windoww = self.ksize[1:3]
        block = self.block
        pad_top = self.pad[0]
        pad_left = self.pad[2]

        hoverlap = self.hoverlap
        col2img_h = windowh
        if col2img_h < strideh:
            col2img_h = strideh
        col2img_dyw = (dyw + 15) // 16 * 16
        if self.woverlap == 0:
            col2img_w = col2img_dyw * stridew
        else:
            col2img_w = (col2img_dyw - 1) * stridew + windoww

        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16

        # vector_repeat_time
        v_rep_time = col2img_dyw * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % V_MAX_REPEAT

        # when every looph move data after, then dup col2img data
        v_rep_time_col = (2 * col2img_w * channel * col2img_h * dtype_size + ONE_REPEAT - 1) // ONE_REPEAT
        v_rep_cycle_col = v_rep_time_col // V_MAX_REPEAT
        v_rep_last_col = v_rep_time_col % V_MAX_REPEAT

        data_input = self.tik_instance.Tensor(dtype, self.input_gard_shape, name="data_input",
                                              scope=tik.scope_gm)
        data_mask = self.tik_instance.Tensor("uint16", (batch * channel1 * windowh * windoww * mask_one_window,),
                                             name="data_mask", scope=tik.scope_gm)
        data_output = self.tik_instance.Tensor(dtype, self.y_shape, name="data_output", scope=tik.scope_gm)

        data_input_origin = self.tik_instance.Tensor(dtype, self.y_shape, name="data_input_origin",
                                                     scope=tik.scope_gm)

        real_block, block_cycle, block_index = self.get_block_param(block)

        with self.tik_instance.for_range(0, real_block, block_num=real_block) as block_id:  # 4 x 1 x 0
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
            with self.tik_instance.if_scope(block_id < block_index):
                real_cycle.set_as(block_cycle + 1)
                block_base.set_as(block_id * real_cycle)
            with self.tik_instance.else_scope():
                real_cycle.set_as(block_cycle)
                block_base.set_as(block_index + block_id * block_cycle)

            require_l1_menmory = dyh * dyw * channel * self.dtype_size\
                                 + windowh * windoww * mask_one_window * 2
            true_val = False if require_l1_menmory > (L1_SIZE // 2) else True
            if true_val is True:
                data_input_l1 = self.tik_instance.Tensor(dtype, (dyh * dyw * channel,), name="data_input_l1",
                                                         scope=tik.scope_cbuf)
                data_mask_l1 = self.tik_instance.Tensor("uint16", (windowh * windoww * mask_one_window,),
                                                        name="data_mask_l1", scope=tik.scope_cbuf)

            with self.tik_instance.for_range(0, real_cycle) as cycle_id:
                block_num.set_as(block_base + cycle_id)
                data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,), name="data_vsel_ub_zero",
                                                             scope=tik.scope_ubuf)
                self.tik_instance.data_move(data_vsel_ub_zero[0], data_input_origin[0], constant.SID,
                                            constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)
                # vector_dup ub every time

                dxh_address_offset.set_as(0)
                dxh_calcline.set_as(0)

                data_max_ub = self.tik_instance.Tensor(dtype, (col2img_dyw * channel,), name="data_max_ub",
                                                       scope=tik.scope_ubuf)  # 256
                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32", (col2img_w * channel * col2img_h + 64,),  # 24 x 16 x 9 + 64
                                             name="data_vmul_ub_col2img_fp32", scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp16 = \
                    self.tik_instance.Tensor(dtype, (col2img_w * channel * col2img_h + 128,),  # 24 x 16 x 9 + 128
                                             name="data_vmul_ub_col2img_fp16", scope=tik.scope_ubuf)
                self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)
                self.clean_fp16_multi_repeat(data_max_ub, dtype_size)

                if true_val is True:
                    self._data_move_from_out_to_l1buf(data_input_l1, data_input, block_num * dyh * dyw * channel,
                                                      dyh * dyw * channel, dtype_size)
                with self.tik_instance.for_range(0, dyh) as looph:  # 5
                    if true_val is True:
                        self.tik_instance.data_move(data_max_ub, data_input_l1[looph * dyw * channel],
                                                    constant.SID, constant.DEFAULT_NBURST,
                                                    dyw * channel * dtype_size // BLOCK_SIZE,
                                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                    else:
                        self.tik_instance.data_move(
                            data_max_ub, data_input[(block_num * dyh + looph) * dyw * channel],
                            constant.SID, constant.DEFAULT_NBURST,
                            dyw * channel * dtype_size // BLOCK_SIZE,
                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                    # mask define
                    data_mask_ub = self.tik_instance.Tensor("uint16", (col2img_dyw,), name="data_mask_ub",
                                                            scope=tik.scope_ubuf)
                    data_vsel_ub = self.tik_instance.Tensor(dtype, (col2img_dyw * channel,),
                                                            name="data_vsel_ub", scope=tik.scope_ubuf)
                    data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (col2img_dyw * channel,),
                                                                 name="data_vsel_ub_fp32", scope=tik.scope_ubuf)
                    if true_val is True:
                        self._data_move_from_out_to_l1buf(
                            data_mask_l1, data_mask, block_num * mask_one_window * windoww * windowh + looph * dyw,
                            mask_one_window * windoww * windowh, dtype_size)

                    self.clean_fp16_multi_repeat(data_vsel_ub, dtype_size)
                    self.clean_fp32_multi_repeat(data_vsel_ub_fp32, dtype_size)
                    self.clean_fp16_multi_repeat(data_mask_ub, dtype_size)

                    with self.tik_instance.for_range(0, windowh * windoww) as mask_id:  # 169
                        if true_val is True:
                            self.tik_instance.data_move(data_mask_ub,
                                                        data_mask_l1[mask_id * mask_one_window],
                                                        constant.SID, 1, col2img_dyw * dtype_size // BLOCK_SIZE,
                                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                        else:
                            self.tik_instance.data_move(data_mask_ub,
                                                        data_mask[block_num * mask_one_window * windoww * windowh +
                                                                  looph * dyw + mask_id * mask_one_window],
                                                        constant.SID, 1, col2img_dyw * dtype_size // BLOCK_SIZE,
                                                        constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                        if v_rep_time > 0:
                            with self.tik_instance.for_range(0, v_rep_time, thread_num=1) as cycle:
                                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(data_mask_ub[cycle * MASK_MAX])
                                self.tik_instance.vsel(constant.MASK128, 0, data_vsel_ub[cycle * FP16_MAX],
                                                       cmpmask, data_max_ub[cycle * FP16_MAX], data_vsel_ub_zero[0],
                                                       constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                       constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT)

                        # fp16 to fp32
                        if v_rep_cycle_fp32 > 0:
                            with self.tik_instance.for_range(0, v_rep_cycle_fp32, thread_num=1) as cycle:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vsel_ub_fp32[cycle * V_MAX_REPEAT * FP32_MAX],
                                                        data_vsel_ub[cycle * V_MAX_REPEAT * FP16_MAX],
                                                        V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_EIGHT,
                                                        constant.REPEAT_STRIDE_FOUR)
                        if v_rep_last_fp32 != 0:
                            self.tik_instance.vconv(
                                constant.MASK64, "", data_vsel_ub_fp32[v_rep_cycle_fp32 * V_MAX_REPEAT * FP32_MAX],
                                data_vsel_ub[v_rep_cycle_fp32 * V_MAX_REPEAT * FP32_MAX],
                                v_rep_last_fp32, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)

                        fetch_filter_w = mask_id % windoww
                        fetch_filter_h = mask_id // windoww
                        self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0], data_vsel_ub_fp32[0],
                                                  (0, 0, 0, 0), col2img_h, col2img_w, fetch_filter_w,
                                                  fetch_filter_h, left_top_w, left_top_h, stridew, strideh,
                                                  windoww, windowh, 1, 1, col2img_dyw // 16)

                    if v_rep_cycle_col > 0:
                        with self.tik_instance.for_range(0, v_rep_cycle_col, thread_num=1) as cycle:
                            self.tik_instance.vconv(
                                constant.MASK64, "",
                                data_vmul_ub_col2img_fp16[cycle * V_MAX_REPEAT * FP32_MAX],
                                data_vmul_ub_col2img_fp32[cycle * V_MAX_REPEAT * FP32_MAX],
                                V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
                    if v_rep_last_col != 0:
                        self.tik_instance.vconv(
                            constant.MASK64, "",
                            data_vmul_ub_col2img_fp16[v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX],
                            data_vmul_ub_col2img_fp32[v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX],
                            v_rep_last_col, constant.STRIDE_ONE, constant.STRIDE_ONE,
                            constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)

                    if hoverlap == 0:
                        # move ub to gm
                        src_address.set_as(pad_left * channel)
                        dst_address.set_as(block_num * dxh * dxw * channel +
                                           (looph * col2img_h - pad_top) * dxw * channel)
                        nburst.set_as(col2img_h)
                        burst_len.set_as(self.offset_w)
                        src_stride.set_as(col2img_w - self.offset_w)
                        dst_stride.set_as(dxw - self.offset_w)
                        with self.tik_instance.if_scope(looph == 0):
                            src_address.set_as(src_address + pad_top * col2img_w * channel)
                            dst_address.set_as(block_num * dxh * dxw * channel)
                            nburst.set_as(nburst - pad_top)
                            with self.tik_instance.if_scope(looph == dyh - 1):
                                nburst.set_as(dxh)

                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(looph == dyh - 1):
                                nburst.set_as(dxh - col2img_h * looph + pad_top)

                        self.tik_instance.data_move(data_output[dst_address], data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, nburst, burst_len, src_stride, dst_stride)
                        if v_rep_cycle_col > 0:
                            with self.tik_instance.for_range(0, v_rep_cycle_col, thread_num=1) as cycle:
                                self.tik_instance.vector_dup(
                                    constant.MASK64,
                                    data_vmul_ub_col2img_fp32[cycle * V_MAX_REPEAT * FP32_MAX],
                                    0, V_MAX_REPEAT, constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_EIGHT)
                        if v_rep_last_col != 0:
                            self.tik_instance.vector_dup(
                                constant.MASK64,
                                data_vmul_ub_col2img_fp32[v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX],
                                0, v_rep_last_col, constant.STRIDE_ONE,
                                constant.REPEAT_STRIDE_EIGHT)
                    else:
                        src_address.set_as(pad_left * channel)
                        dst_address.set_as(block_num * dxh * dxw * channel + dxh_address_offset)
                        nburst.set_as(strideh)
                        with self.tik_instance.if_scope(looph * strideh < pad_top):
                            nburst.set_as((looph + 1) * strideh - pad_top)
                            src_address.set_as(src_address + (pad_top - looph * strideh) * col2img_w * channel)
                        with self.tik_instance.if_scope(tik.all(dxh_calcline < dxh, looph == dyh - 1)):
                            nburst.set_as(dxh - dxh_calcline)

                        burst_len.set_as(self.offset_w)
                        src_stride.set_as(col2img_w - self.offset_w)
                        dst_stride.set_as(dxw - self.offset_w)
                        with self.tik_instance.if_scope(nburst > 0):
                            self.tik_instance.data_move(data_output[dst_address],
                                                        data_vmul_ub_col2img_fp16[src_address],
                                                        constant.SID, nburst,
                                                        burst_len, src_stride,
                                                        dst_stride)
                        with self.tik_instance.else_scope():
                            nburst.set_as(0)

                        dxh_address_offset.set_as(dxh_address_offset + nburst * dxw * channel)
                        dxh_calcline.set_as(dxh_calcline + nburst)

                        # dma_copy ub to ub
                        self.tik_instance.data_move(data_vmul_ub_col2img_fp32[0],
                                                    data_vmul_ub_col2img_fp32[strideh * channel * col2img_w],
                                                    constant.SID, hoverlap, 2 * col2img_w,
                                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                        start_pos = hoverlap * channel * col2img_w
                        res_len = data_vmul_ub_col2img_fp32.shape[0] - start_pos
                        self.clear_res_fp32_ub(data_vmul_ub_col2img_fp32, start_pos, res_len, dtype_size * 2)

        self.tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(data_input_origin, data_input, data_mask),
                                   outputs=data_output, enable_l2=False)
        return self.tik_instance

    def tik_instance_cut_nc1h_cut_one_h(self, kernel_name):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        batch, channel1, dyh, dyw, channel = self.input_gard_shape
        dxh, dxw = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        if strideh > dxh:
            strideh = dxh

        if stridew > dxw:
            stridew = dxw

        dtype = self.dtype
        dtype_size = self.dtype_size
        windowh, windoww = self.ksize[1:3]
        pad_top, pad_bottom, pad_left = self.pad[0:3]
        hoverlap = self.hoverlap
        woverlap = self.woverlap

        ho_count = math.ceil(self.blocknum // (batch * channel1))
        if hoverlap == 0:
            ho_every = dyh // ho_count
            ho_last = dyh - ho_every * (ho_count - 1)
        else:
            ho_every = (dyh + ho_count - 1) // ho_count
            if ho_every == 1:
                ho_count = ho_count // 2
                ho_every = (dyh + ho_count - 1) // ho_count
            ho_last = dyh + ho_count - 1 - ho_every * (ho_count - 1)
        all_blocknum = ho_count * batch * channel1

        wo_max = math.ceil(dyw / 16) * 16
        col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + windoww
        ho_max = 1
        col2img_h = ho_max * strideh if hoverlap == 0 else (ho_max - 1) * strideh + windowh

        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16
        mask_stride = (mask_one_window - wo_max) // 16

        # vector_repeat_time
        v_rep_time = wo_max * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % V_MAX_REPEAT

        data_input = self.tik_instance.Tensor(dtype, self.input_gard_shape, name="data_input", scope=tik.scope_gm)
        data_mask = self.tik_instance.Tensor(
            "uint16", (batch * channel1 * windowh * windoww * mask_one_window,), name="data_mask", scope=tik.scope_gm)

        data_output = self.tik_instance.Tensor(dtype, self.y_shape, name="data_output", scope=tik.scope_gm)
        data_input_origin = self.tik_instance.Tensor(dtype, self.y_shape, name="data_input_origin", scope=tik.scope_gm)

        real_block, block_cycle, block_index = self.get_block_param(all_blocknum)

        with self.tik_instance.for_range(0, real_block, block_num=real_block) as block_id:
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
            with self.tik_instance.if_scope(block_id < block_index):
                real_cycle.set_as(block_cycle + 1)
                block_base.set_as(block_id * real_cycle)
            with self.tik_instance.else_scope():
                real_cycle.set_as(block_cycle)
                block_base.set_as(block_index + block_id * block_cycle)
            with self.tik_instance.for_range(0, real_cycle) as cycle_id:
                block_num.set_as(block_base + cycle_id)
                data_vsel_ub_zero = self.tik_instance.Tensor(
                    dtype, (128,), name="data_vsel_ub_zero", scope=tik.scope_ubuf)
                self.tik_instance.data_move(data_vsel_ub_zero[0], data_input_origin[0], constant.SID,
                                            constant.DEFAULT_NBURST, constant.DEFAULT_BURST_LEN,
                                            constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)

                block_batch.set_as(block_num // ho_count)
                block_h.set_as(block_num % ho_count)
                h_cycle.set_as(ho_every)
                with self.tik_instance.if_scope(block_h == ho_count - 1):
                    h_cycle.set_as(ho_last)

                dxh_address_offset.set_as(block_batch * dxh * dxw * channel)

                if hoverlap is 0:
                    with self.tik_instance.if_scope(block_h != 0):
                        dxh_address_offset.set_as(dxh_address_offset + (block_h * ho_every * windowh - pad_top) *
                                                  dxw * channel)
                else:
                    with self.tik_instance.if_scope(block_h != 0):
                        dxh_address_offset.set_as(dxh_address_offset + ((block_h * (ho_every - 1) + 1) *
                                                                        strideh - pad_top) * dxw * channel)
                # vector_dup ub every time
                data_max_ub = self.tik_instance.Tensor(dtype, (wo_max * channel,), name="data_max_ub",
                                                       scope=tik.scope_ubuf)
                self.clean_max_ub(data_max_ub, dtype)

                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32", (col2img_w * channel * col2img_h + 64,),
                                             name="data_vmul_ub_col2img_fp32", scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp16 = \
                    self.tik_instance.Tensor(dtype, (col2img_w * channel * col2img_h + 128,),
                                             name="data_vmul_ub_col2img_fp16", scope=tik.scope_ubuf)
                self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                self.clean_fp16_multi_repeat(data_vmul_ub_col2img_fp16, dtype_size)

                # mask define
                data_mask_ub = self.tik_instance.Tensor("uint16", (wo_max * windowh * windoww,),
                                                        name="data_mask_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, h_cycle) as looph:  # 3
                    # address  not  multiplex
                    in_src_address.set_as(block_batch * dyh * dyw * channel)
                    mask_address.set_as(block_batch * mask_one_window * windoww * windowh)
                    if hoverlap == 0:
                        in_src_address.set_as(in_src_address + (block_h * ho_every + looph) * dyw * channel)
                        mask_address.set_as(mask_address + (block_h * ho_every + looph) * dyw)
                    else:
                        in_src_address.set_as(in_src_address + (block_h * (ho_every - 1) + looph) * dyw * channel)
                        mask_address.set_as(mask_address + (block_h * (ho_every - 1) + looph) * dyw)

                    # dy copy gm to ub, every 1 lines per repeat
                    self.tik_instance.data_move(data_max_ub, data_input[in_src_address], constant.SID,
                                                constant.DEFAULT_NBURST, dyw * channel * dtype_size // BLOCK_SIZE,
                                                constant.STRIDE_ZERO, constant.STRIDE_ZERO)

                    self.clean_fp16_multi_repeat(data_mask_ub, dtype_size)

                    # mask copy gm to ub
                    self.tik_instance.data_move(data_mask_ub, data_mask[mask_address], constant.SID, windowh * windoww,
                                                wo_max * dtype_size // BLOCK_SIZE, mask_stride, constant.STRIDE_ZERO)

                    data_vsel_ub = self.tik_instance.Tensor(dtype, (wo_max * channel,),
                                                            name="data_vsel_ub", scope=tik.scope_ubuf)
                    data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (wo_max * channel,),
                                                                 name="data_vsel_ub_fp32", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, windowh * windoww) as mask_id:
                        if v_rep_time > 0:  # 336  divides to 8
                            with self.tik_instance.for_range(0, v_rep_time) as cycle:
                                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                    data_mask_ub[wo_max * mask_id + cycle * MASK_MAX])
                                self.tik_instance.vsel(
                                    constant.MASK128, 0, data_vsel_ub[cycle * FP16_MAX], cmpmask,
                                    data_max_ub[cycle * FP16_MAX], data_vsel_ub_zero[0], constant.REPEAT_TIME_ONCE,
                                    constant.STRIDE_ONE, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT,
                                    constant.REPEAT_STRIDE_EIGHT)

                        # fp16 to fp32
                        if v_rep_cycle_fp32 > 0:
                            with self.tik_instance.for_range(0, v_rep_cycle_fp32) as cycle:
                                self.tik_instance.vconv(
                                    constant.MASK64, "",
                                    data_vsel_ub_fp32[cycle * V_MAX_REPEAT * FP32_MAX],
                                    data_vsel_ub[cycle * V_MAX_REPEAT * FP32_MAX],
                                    V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_FOUR)
                        if v_rep_last_fp32 != 0:  # 336 * 16 divides to 64 equals to 84
                            self.tik_instance.vconv(
                                constant.MASK64, "",
                                data_vsel_ub_fp32[v_rep_cycle_fp32 * V_MAX_REPEAT * FP32_MAX],
                                data_vsel_ub[v_rep_cycle_fp32 * V_MAX_REPEAT * FP32_MAX], v_rep_last_fp32,
                                constant.STRIDE_ONE, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                                constant.REPEAT_STRIDE_FOUR)

                        # col2img
                        fetch_filter_w = mask_id % windoww
                        fetch_filter_h = mask_id // windoww
                        self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0], data_vsel_ub_fp32[0], (0, 0, 0, 0),
                                                  col2img_h, col2img_w, fetch_filter_w, fetch_filter_h, 0,
                                                  0, stridew, strideh, windoww, windowh, 1, 1, wo_max // 16)

                    output_cuthline.set_as(strideh)
                    src_address.set_as(pad_left * channel)
                    if hoverlap == 0:
                        with self.tik_instance.if_scope(block_h == 0):
                            with self.tik_instance.if_scope(looph == 0):
                                output_cuthline.set_as(output_cuthline - pad_top)
                        with self.tik_instance.if_scope(block_h == ho_count - 1):
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                output_cuthline.set_as(output_cuthline - pad_bottom)
                    else:
                        with self.tik_instance.if_scope(block_h == 0):
                            with self.tik_instance.if_scope(looph == 0):
                                output_cuthline.set_as(output_cuthline - pad_top)
                                src_address.set_as(src_address + pad_top * col2img_w * channel)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(looph == 0):
                                output_cuthline.set_as(0)
                        with self.tik_instance.if_scope(block_h == ho_count - 1):
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                output_cuthline.set_as(windowh)

                    # fp32 to fp16
                    v_rep_time_col = 2 * (col2img_w * channel * col2img_h + 64) * dtype_size // ONE_REPEAT
                    v_rep_cycle_col = v_rep_time_col // V_MAX_REPEAT
                    v_rep_last_col = v_rep_time_col % V_MAX_REPEAT
                    if v_rep_cycle_col > 0:  # 1
                        with self.tik_instance.for_range(0, v_rep_cycle_col, thread_num=1) as cycle:
                            self.tik_instance.vconv(
                                constant.MASK64, "",
                                data_vmul_ub_col2img_fp16[cycle * V_MAX_REPEAT * FP32_MAX],
                                data_vmul_ub_col2img_fp32[cycle * V_MAX_REPEAT * FP32_MAX],
                                V_MAX_REPEAT, constant.STRIDE_ONE, constant.STRIDE_ONE,
                                constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
                    if v_rep_last_col != 0:
                        self.tik_instance.vconv(
                            constant.MASK64, "",
                            data_vmul_ub_col2img_fp16[v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX],
                            data_vmul_ub_col2img_fp32[v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX],
                            v_rep_last_col, constant.STRIDE_ONE, constant.STRIDE_ONE,
                            constant.REPEAT_STRIDE_FOUR, constant.REPEAT_STRIDE_EIGHT)
                    with self.tik_instance.if_scope(output_cuthline != 0):
                        self.tik_instance.data_move(data_output[dxh_address_offset],
                                                    data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, output_cuthline, self.offset_w,
                                                    col2img_w - self.offset_w, dxw - self.offset_w)
                        dxh_address_offset.set_as(dxh_address_offset + output_cuthline * dxw * channel)
                    if hoverlap != 0:
                        with self.tik_instance.if_scope(looph + 1 != h_cycle):
                            self.tik_instance.data_move(data_vmul_ub_col2img_fp32[0],
                                                        data_vmul_ub_col2img_fp32[strideh * channel * col2img_w],
                                                        constant.SID, constant.DEFAULT_NBURST,
                                                        2 * hoverlap * col2img_w, constant.STRIDE_ZERO,
                                                        constant.STRIDE_ZERO)
                            start_pos = hoverlap * channel * col2img_w
                            res_len = data_vmul_ub_col2img_fp32.shape[0] - start_pos
                            self.clear_res_fp32_ub(data_vmul_ub_col2img_fp32, start_pos, res_len, dtype_size * 2)
                    else:
                        self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
        self.tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(data_input_origin, data_input, data_mask),
                                   outputs=data_output, enable_l2=False)

        return self.tik_instance

    def clear_res_fp32_ub(self, data_vmul_ub_col2img_fp32, start_pos, res_len, dtype_size):
        """
        The fun just for clean ub
        """
        v_rep_clear_time = res_len * dtype_size // ONE_REPEAT
        v_rep_clear_cycle = v_rep_clear_time // V_MAX_REPEAT  # 1
        v_rep_clear_last = v_rep_clear_time % V_MAX_REPEAT  # 250
        v_res_last = v_rep_clear_cycle * V_MAX_REPEAT * FP32_MAX + v_rep_clear_last * FP32_MAX
        v_res_time = res_len - v_res_last
        if v_rep_clear_cycle > 0:
            with self.tik_instance.for_range(0, v_rep_clear_cycle, thread_num=1) as cycle:
                self.tik_instance.vector_dup(
                    constant.MASK64, data_vmul_ub_col2img_fp32[start_pos + cycle * V_MAX_REPEAT * FP32_MAX],
                    0, V_MAX_REPEAT, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        if v_rep_clear_last != 0:
            self.tik_instance.vector_dup(
                constant.MASK64, data_vmul_ub_col2img_fp32[start_pos + v_rep_clear_cycle * V_MAX_REPEAT * FP32_MAX],
                0, v_rep_clear_last, constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        if v_res_time != 0:
            res_repeat_time = v_res_time * dtype_size // constant.BLOCK_SIZE
            self.tik_instance.vector_dup(
                8, data_vmul_ub_col2img_fp32[start_pos + v_res_last], 0, res_repeat_time, constant.STRIDE_ONE, 1)
