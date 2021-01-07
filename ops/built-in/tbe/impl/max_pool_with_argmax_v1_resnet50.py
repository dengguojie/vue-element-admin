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
max_pool_with_argmax_v1_resnet50
"""
import math
from te import tik
from impl import common_util_v1
from impl import constant_util_v1 as constant
from te import platform as tbe_platform

# min value of fp16
MIN_VALUE_FP16 = -65504.0
# define dilation size
DILATION = 1
# parameters for vector instruct
MASK = 128
REPEAT_2 = 2
DSTSTRIDEM0 = 1
SRC0STRIDEM0 = 1
SRC1STRIDEM0 = 1
DSTSTRIDEM1 = 8
SRC0STRIDEM1 = 8
SRC1STRIDEM1 = 8
BLOCK_H = 4
DT_INT32 = 3
DT_INT64 = 9
DIM_C0 = 16
# get available ub size
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# get available l1 size
L1_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)


# pylint: disable=missing-class-docstring,too-many-instance-attributes
# pylint: disable=too-many-arguments,no-self-use,too-many-locals,too-few-public-methods,invalid-name,unused-argument
def _ceil_div(value, factor):
    """
    caculate ceil value of div

    Parameters
    ----------
    value: dtype of int or float
        original value
    factor: dtype of int or float
        dividend value

    Returns
    -------
    value: dtype of int or float
    """
    return math.ceil(value / factor)


class MaxPoolWithArgmaxV1Resnet50:
    def __init__(self, x, ksize, strides, padding, dtype, dilation, ceil_mode, kernel_name):
        """
        init MaxPoolWithargmax parameters

        Parameters
        ----------
        x: dict
            shape and datatype
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        padding: list or tuple
            The type of padding algorithm to use.
        dtype: int
            DT_INT32 or DT_INT64
        dilation: list of tuple
        ceil_mode: str
            True or False
        kernel_name: str
        Returns
        -------
        None
        """
        self.kernel_name = kernel_name
        self.dtype = dtype
        self.input_shape = x.get("shape")
        self.input_dtype = x.get("dtype").lower()
        self.input_type_size = common_util_v1.get_data_size(self.input_dtype)
        self.tik_inst = tik.Tik()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.dilation = dilation
        self.pad_l = self.padding[0]
        self.pad_t = self.padding[2]
        self.batch_size = self.input_shape[0]
        self.c1_size = self.input_shape[1]
        self.in_size_h = self.input_shape[2]
        self.in_size_w = self.input_shape[3]
        self.c_block_size = self.input_shape[4]
        self.window_h = self.ksize[1]
        self.window_w = self.ksize[2]
        self.stride_h = self.strides[1]
        self.stride_w = self.strides[2]
        self.dilation_h = self.dilation[1]
        self.dilation_w = self.dilation[2]
        self.nc1 = self.batch_size * self.c1_size
        # scalar for load3d
        self.scalar_source_h = self.tik_inst.Scalar(dtype="int64")
        self.scalar_source_w = self.tik_inst.Scalar(dtype="int64")

        # caculate pad and output size
        self.pad, self.out_size_h, self.out_size_w = self._calc_out_size_and_pad()
        # output_shape
        self.fmap_img2col_h = self.out_size_h * self.out_size_w
        self.fmap_img2col_w = self.window_h * self.window_w
        self.fmap_img2col_h_num = _ceil_div(self.fmap_img2col_h, self.c_block_size)
        if self.input_dtype == "float16":
            self.pad_value = MIN_VALUE_FP16
        # fmap is NC1HWC0 format
        fmap_gm_shape = (self.batch_size, self.c1_size, self.in_size_h, self.in_size_w, self.c_block_size)
        output_gm_shape = (self.batch_size, self.c1_size, self.out_size_h, self.out_size_w, self.c_block_size)
        output_mask_gm_shape = \
            (self.batch_size, self.c1_size, self.fmap_img2col_w, (self.fmap_img2col_h_num + 1) * self.c_block_size)

        # input and output
        self.input_fmap_gm = self.tik_inst.Tensor(
            self.input_dtype, fmap_gm_shape, name="input_fmap_gm", scope=tik.scope_gm)
        self.output_max_gm = self.tik_inst.Tensor(
            self.input_dtype, output_gm_shape, name="output_max_gm", scope=tik.scope_gm)
        self.output_mask_gm = self.tik_inst.Tensor(
            "uint16", output_mask_gm_shape, name="output_mask_gm", scope=tik.scope_gm)

        self._tik_instance_function_init()

    def _pooling_output_shape_pad_lr(self, input_size, kernel_size, pad_l, pad_r, stride, dilation, ceil_mode):
        temp = input_size + pad_l + pad_r - dilation * (kernel_size - 1) - 1
        if ceil_mode is True:
            output_size = ((temp + (stride - 1)) // stride) + 1
        else:
            output_size = (temp // stride) + 1
        if pad_l > 0:
            # ensure that the last pooling starts inside the image
            # needed to avoid problems in ceil mode
            if (output_size - 1) * stride >= (input_size + pad_l):
                output_size = output_size - 1

        return output_size

    def _pooling_output_shape(self, input_size, kernel_size, pad, stride, dilation, ceil_mode):
        return self._pooling_output_shape_pad_lr(input_size, kernel_size, pad, pad, stride, dilation, ceil_mode)

    def _pool2d_shape_check(self, kernel_h, kernel_w, stride_h, stride_w,
                            pad_h, pad_w, dilation_h, dilation_w, output_h,
                            output_w):
        if kernel_w <= 0 or kernel_h <= 0:
            raise RuntimeError("kernel size should be greater than zero, but \
                    got ", "kH: ", kernel_h, " kW: ", kernel_w)

        if stride_h <= 0 or stride_w <= 0:
            raise RuntimeError("stride should be greater than zero, but got ", "dH= ", stride_h, "dW= ", stride_w)

        if dilation_h <= 0 or dilation_w <= 0:
            raise RuntimeError("dilation should be greater than 0, but got",
                               "dilationH= ", dilation_h, ", dilationW= ", dilation_w)

        if (kernel_w // 2) < pad_w or (kernel_h // 2) < pad_h:
            raise RuntimeError("pad should be smaller than half of kernel "
                               "size, but got", "padW=", pad_w, ", padH= ",
                               pad_h, ", kW= ", kernel_w, ", kH= ", kernel_h)

        if output_h < 1 or output_w < 1:
            raise RuntimeError("Output size is too small ", "outW= ", output_w, "outH= ", output_h)

    def _calc_out_size_and_pad(self):
        pad_h = self.padding[1]
        pad_w = self.padding[2]
        kernel_h = self.ksize[1]
        kernel_w = self.ksize[2]
        dilation_h = self.dilation[1]
        dilation_w = self.dilation[2]
        input_h = self.input_shape[2]
        input_w = self.input_shape[3]
        stride_h = self.strides[1]
        stride_w = self.strides[2]
        output_h = self._pooling_output_shape(input_h, kernel_h, pad_h, stride_h, dilation_h, self.ceil_mode)
        output_w = self._pooling_output_shape(input_w, kernel_w, pad_w, stride_w, dilation_w, self.ceil_mode)

        self._pool2d_shape_check(kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                                 dilation_h, dilation_w, output_h, output_w)

        if self.ceil_mode is False:
            pad_t = pad_h
            pad_b = pad_h
            pad_l = pad_w
            pad_r = pad_w
        else:
            pad_t = pad_h
            pad_b = pad_h + self.stride_h - 1
            pad_l = pad_w
            pad_r = pad_w + self.stride_w - 1

        pad = (pad_l, pad_r, pad_t, pad_b)

        return pad, output_h, output_w

    def _calc_max_fun(self, data_input, data_input_ub, index_w, index_h):
        """
        caculate max of data_input

        Parameters
        ----------
        data_input: input data
        data_input_ub: input data in ub
        index_w: input size in w direction
        index_h: input size in h direction

        Returns
        -------
        data_input: output tensor
        """
        self.tik_inst.vmax(
            MASK, data_input[index_h * 256], data_input[index_h * 256],
            data_input_ub[index_w * 256 + index_h * self.fmap_img2col_w * 256],
            REPEAT_2, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0, DSTSTRIDEM1,
            SRC0STRIDEM1, SRC1STRIDEM1)
        return data_input

    def _load3d_fm_to_ub(self, ub_buff, l1_buff, i_dim_w, i_dim_h):
        instance = self.tik_inst
        filter_size = self.window_h * self.window_w
        with instance.for_range(0, filter_size, thread_num=1) as loopk:
            k_dim_w = loopk % 3
            k_dim_h = loopk // 3
            instance.load3dv1(ub_buff[loopk * 4 * 56 * 16], l1_buff,
                              self.padding, self.in_size_h,
                              self.in_size_w,
                              0, k_dim_w, k_dim_h, i_dim_w, i_dim_h,
                              self.stride_w, self.stride_h,
                              self.window_w, self.window_h,
                              self.dilation_w, self.dilation_h, 1, 1, 4 * 56 // 16, 0, self.pad_value)

    def _tik_instance_function_init(self):
        ub_mask_buff_size = 8 * 1024
        dtype = self.input_dtype
        filter_size = self.window_h * self.window_w
        input_h, input_w = self.input_shape[2:4]
        self.ub_mask_buff = self.tik_inst.Tensor(
            "uint16", (ub_mask_buff_size,), name="ub_mask_buff", scope=tik.scope_ubuf)
        self.ub_mask_temp = self.tik_inst.Tensor(
            "uint16", (ub_mask_buff_size,), name="ub_mask_temp", scope=tik.scope_ubuf)
        self.ub_mask_or_buff = self.tik_inst.Tensor(
            "uint16", (ub_mask_buff_size,), name="ub_mask_or_buff", scope=tik.scope_ubuf)
        self.ub_mask_not_buff = self.tik_inst.Tensor(
            "uint16", (ub_mask_buff_size,), name="ub_mask_not_buff", scope=tik.scope_ubuf)

        ub_buff_size = BLOCK_H * self.out_size_w * DIM_C0 * filter_size
        self.buf_0 = self.tik_inst.Tensor(dtype, (ub_buff_size,), name="ub_buf_0", scope=tik.scope_ubuf)
        self.buf_1 = self.tik_inst.Tensor(dtype, (ub_buff_size,), name="ub_buf_1", scope=tik.scope_ubuf)

        self.input_idx = self.tik_inst.Scalar("uint64", name="input_idx")
        self.output_idx = self.tik_inst.Scalar("uint64", name="output_idx")
        self.l1_idx = self.tik_inst.Scalar("uint64", name="l1_idx")
        self.mask_idx = self.tik_inst.Scalar("uint64", name="mask_idx")
        self.fm_size = self.tik_inst.Scalar("uint64", init_value=0)

        l1_buff0_size = input_h * input_w * DIM_C0 + 32 * 1024
        self.l1_buff0 = self.tik_inst.Tensor(dtype, (l1_buff0_size,), name="l1_buff0", scope=tik.scope_cbuf)
        ub_max_buff_size = self.stride_h * BLOCK_H * self.out_size_w * DIM_C0
        self.ub_max_buff = self.tik_inst.Tensor(dtype, (ub_max_buff_size,), name="ub_max_buff", scope=tik.scope_ubuf)

    def _calc_ping_fm_size(self, looph, input_w):
        self.fm_size.set_as(0)
        with self.tik_inst.if_scope(looph == 0):
            self.fm_size.set_as((BLOCK_H * self.stride_h + 1) * input_w)
        with self.tik_inst.else_scope():
            self.fm_size.set_as(BLOCK_H * self.stride_h * input_w)

    def _calc_pong_fm_size(self, looph, loop_h, input_w):
        self.fm_size.set_as(0)
        with self.tik_inst.if_scope(looph == loop_h // 2 - 1):
            self.fm_size.set_as((BLOCK_H * self.stride_h - 1) * input_w)
        with self.tik_inst.else_scope():
            self.fm_size.set_as(BLOCK_H * self.stride_h * input_w)

    def _calc_output_mask(self, filter_size, repeat_1, repeat_stride, output_w):
        with self.tik_inst.for_range(2, filter_size) as idx:
            self.tik_inst.vnot(constant.MASK128, self.ub_mask_not_buff, self.ub_mask_or_buff, repeat_1, 1,
                               1, repeat_stride, repeat_stride)
            self.tik_inst.vor(constant.MASK128, self.ub_mask_or_buff, self.ub_mask_or_buff,
                              self.ub_mask_buff[idx * BLOCK_H * output_w], repeat_1, 1, 1, 1,
                              repeat_stride, repeat_stride, repeat_stride)
            self.tik_inst.vand(constant.MASK128, self.ub_mask_temp[(idx - 1) * BLOCK_H * output_w],
                               self.ub_mask_not_buff, self.ub_mask_buff[idx * BLOCK_H * output_w],
                               repeat_1, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)

    def _tik_instance_function_ping(self, looph, input_w, output_w, mask_gap,
                                    filter_size, repeat_0, repeat_1, repeat_stride, mask_one_window):
        self._calc_ping_fm_size(looph, input_w)
        self.tik_inst.data_move(self.l1_buff0[self.l1_idx], self.input_fmap_gm[self.input_idx], 0, 1,
                                self.fm_size, 0, 0)
        self._load3d_fm_to_ub(
            self.buf_0, self.l1_buff0, 0 - self.pad_l, looph * 2 * BLOCK_H * self.stride_h - self.pad_t)
        self.tik_inst.vmax(constant.MASK128, self.ub_max_buff, self.buf_0,
                           self.buf_0[BLOCK_H * output_w * DIM_C0],
                           repeat_0, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        with self.tik_inst.for_range(2, filter_size) as idx:
            self.tik_inst.vmax(constant.MASK128, self.ub_max_buff, self.ub_max_buff,
                               self.buf_0[BLOCK_H * output_w * DIM_C0 * idx],
                               repeat_0, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        output_idx_tmp = (self.output_idx + looph * 2 * BLOCK_H * output_w * DIM_C0)
        self.tik_inst.data_move(self.output_max_gm[output_idx_tmp], self.ub_max_buff, 0, 1,
                                BLOCK_H * output_w * DIM_C0 * 2 // 32, 0, 0)

        with self.tik_inst.for_range(0, filter_size) as idx:
            self.tik_inst.vcmpv_eq(self.ub_mask_buff[idx * BLOCK_H * output_w * DIM_C0 // 16],
                                   self.buf_0[idx * BLOCK_H * output_w * DIM_C0],
                                   self.ub_max_buff, repeat_0, 1, 1, repeat_stride, repeat_stride)
        self.tik_inst.vnot(
            constant.MASK128, self.ub_mask_not_buff, self.ub_mask_buff, repeat_1, 1, 1, repeat_stride,
            repeat_stride)
        self.tik_inst.vor(constant.MASK128, self.ub_mask_or_buff, self.ub_mask_buff,
                          self.ub_mask_buff[BLOCK_H * output_w],
                          repeat_1, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        self.tik_inst.vand(constant.MASK128, self.ub_mask_temp, self.ub_mask_not_buff,
                           self.ub_mask_buff[BLOCK_H * output_w],
                           repeat_1, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        self._calc_output_mask(filter_size, repeat_1, repeat_stride, output_w)
        self.tik_inst.data_move(
            self.output_mask_gm[self.mask_idx], self.ub_mask_buff, 0, 1, BLOCK_H * output_w // DIM_C0, 0, 0)
        self.tik_inst.data_move(
            self.output_mask_gm[self.mask_idx + mask_one_window], self.ub_mask_temp, 0, filter_size - 1,
            BLOCK_H * output_w // DIM_C0, 0, mask_gap
        )

        self.input_idx.set_as(self.input_idx + self.fm_size * DIM_C0)
        self.mask_idx.set_as(self.mask_idx + BLOCK_H * output_w * DIM_C0 // 16)
        self.l1_idx.set_as(self.l1_idx + self.fm_size * 16)

    def _tik_instance_function_pong(self, looph, loop_h, input_w, output_w, mask_gap, filter_size,
                                    repeat_0, repeat_1, repeat_stride, mask_one_window):
        self._calc_pong_fm_size(looph, loop_h, input_w)
        self.tik_inst.data_move(
            self.l1_buff0[self.l1_idx], self.input_fmap_gm[self.input_idx], 0, 1, self.fm_size, 0, 0)

        self._load3d_fm_to_ub(
            self.buf_1, self.l1_buff0, -self.pad_l,
            (looph * 2 + 1) * BLOCK_H * self.stride_h - self.pad_t)

        self.tik_inst.vmax(constant.MASK128, self.ub_max_buff, self.buf_1,
                           self.buf_1[BLOCK_H * output_w * DIM_C0],
                           repeat_0, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)

        with self.tik_inst.for_range(2, filter_size) as idx:
            self.tik_inst.vmax(constant.MASK128, self.ub_max_buff, self.ub_max_buff,
                               self.buf_1[BLOCK_H * output_w * DIM_C0 * idx],
                               repeat_0, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)

        output_idx_tmp = (self.output_idx + (looph * 2 + 1) * BLOCK_H * output_w * DIM_C0)
        self.tik_inst.data_move(self.output_max_gm[output_idx_tmp], self.ub_max_buff, 0, 1, BLOCK_H *
                                output_w * DIM_C0 * 2 // 32, 0, 0)

        with self.tik_inst.for_range(0, filter_size) as idx:
            self.tik_inst.vcmpv_eq(self.ub_mask_buff[idx * BLOCK_H * output_w * DIM_C0 // 16],
                                   self.buf_1[idx * BLOCK_H * output_w * DIM_C0],
                                   self.ub_max_buff, repeat_0, 1, 1, repeat_stride, repeat_stride)

        self.tik_inst.vnot(constant.MASK128, self.ub_mask_not_buff, self.ub_mask_buff, repeat_1, 1, 1,
                           repeat_stride, repeat_stride)
        self.tik_inst.vor(constant.MASK128, self.ub_mask_or_buff, self.ub_mask_buff,
                          self.ub_mask_buff[BLOCK_H * output_w],
                          repeat_1, 1, 1, 1, repeat_stride, repeat_stride, repeat_stride)
        self.tik_inst.vand(constant.MASK128, self.ub_mask_temp, self.ub_mask_not_buff,
                           self.ub_mask_buff[BLOCK_H * output_w], repeat_1, 1, 1, 1,
                           repeat_stride, repeat_stride, repeat_stride)
        self._calc_output_mask(filter_size, repeat_1, repeat_stride, output_w)

        self.tik_inst.data_move(
            self.output_mask_gm[self.mask_idx], self.ub_mask_buff, 0, 1, BLOCK_H * output_w // DIM_C0, 0, 0)
        self.tik_inst.data_move(self.output_mask_gm[self.mask_idx + mask_one_window], self.ub_mask_temp, 0,
                                filter_size - 1, BLOCK_H * output_w // DIM_C0, 0, mask_gap)

        self.input_idx.set_as(self.input_idx + self.fm_size * DIM_C0)
        self.mask_idx.set_as(self.mask_idx + BLOCK_H * output_w * DIM_C0 // 16)
        self.l1_idx.set_as(self.l1_idx + self.fm_size * 16)

    def tik_instance_function(self, kernel_name):
        """
        implementation of max_pool_with_argmax and return the tik instance
        :param kernel_name: the kernel's name
        :return: tik instance
        """
        batch_size = self.input_shape[0]
        c1_dim = self.input_shape[1]
        filter_size = self.window_h * self.window_w  # 1
        output_h = self.out_size_h
        output_w = self.out_size_w
        input_h, input_w = self.input_shape[2:4]
        loop_h = output_h // BLOCK_H   # 2
        mask_one_window = ((output_h * output_w + 15) // 16 + 1) * 16
        mask_gap_element = (mask_one_window - BLOCK_H * output_w)
        mask_gap = mask_gap_element * 2 // 32  # 3
        repeat_stride = constant.MASK128 * 2 // 32
        repeat_0 = (BLOCK_H * output_w * DIM_C0 // constant.MASK128)
        repeat_1 = math.ceil(BLOCK_H * output_w / constant.MASK128)

        with self.tik_inst.for_range(0, batch_size * c1_dim, block_num=batch_size * c1_dim) as batch_idx:
            batch = batch_idx / c1_dim
            loop = batch_idx % c1_dim
            self.input_idx.set_as(batch * c1_dim * input_h * input_w * DIM_C0 + loop * input_h * input_w * DIM_C0)
            self.output_idx.set_as(batch * c1_dim * output_h * output_w * DIM_C0 + loop * output_h * output_w * DIM_C0)
            self.l1_idx.set_as(0)
            self.mask_idx.set_as(
                batch * c1_dim * mask_one_window * filter_size + loop * mask_one_window * filter_size)

            with self.tik_inst.for_range(0, loop_h // 2) as looph:
                # ping
                self._tik_instance_function_ping(looph, input_w, output_w, mask_gap, filter_size,
                                                 repeat_0, repeat_1, repeat_stride, mask_one_window)

                # pong
                self._tik_instance_function_pong(looph, loop_h, input_w, output_w, mask_gap, filter_size,
                                                 repeat_0, repeat_1, repeat_stride, mask_one_window)

        self.tik_inst.BuildCCE(kernel_name=kernel_name, inputs=self.input_fmap_gm,
                               outputs=(self.output_max_gm, self.output_mask_gm))
        return self.tik_inst


def is_max_pool_with_argmax_param(x, ksize, strides, padding):
    """
    test if the param suitable for this module to treat
    :param x: dict of shape and dtype of the input x
    :param ksize: value of ksize
    :param strides: value of strides
    :param padding: value of padding
    :return: Bool, if the param suitable for this module to treat return True,
             if not return False
    """
    resnet50_x = {"shape": (32, 4, 112, 112, 16), "dtype": "float16"}
    resnet50_ksize = [1, 3, 3, 1]
    resnet50_strides = [1, 2, 2, 1]
    resnet50_padding = [1, 1, 1, 1]

    def is_valid_shape(resnet50shape, shape):
        if shape.get("dtype") != resnet50shape.get("dtype"):
            return False

        if len(shape.get("shape")) != len(resnet50shape.get("shape")):
            return False

        resnet50_last3dims = resnet50shape.get("shape")[2:]
        last3dims = shape.get("shape")[2:]

        return list(resnet50_last3dims) == list(last3dims)

    ksize = list(ksize)
    strides = list(strides)
    padding = list(padding)

    if (resnet50_ksize == ksize and resnet50_strides == strides and
            resnet50_padding == padding and is_valid_shape(resnet50_x, x)):
        return True

    return False


def max_pool_with_argmax_v1_resnet50(x, y, argmax, ksize, strides, pads, dtype=DT_INT32, dilation=(1, 1, 1, 1),
                                     ceil_mode=False, kernel_name="max_pool_with_argmax_v1"):
    max_pool_grad = MaxPoolWithArgmaxV1Resnet50(x, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
    return max_pool_grad.tik_instance_function(kernel_name)
