# -*- coding:utf-8 -*-
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
max_pool_with_argmaxv1
"""
from impl import constant_util
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # min value of fp16
    MIN_VALUE_FP16 = -65504.0
    TILING_NUM = 64
    # parameters for vector instruct
    TILING_MODE0 = 0
    TILING_MODE1 = 1
    TILING_MODE2 = 2
    MASK = 128
    ALIGN16 = 16
    REPEAT_2 = 2
    DSTSTRIDEM1 = 8
    SRC0STRIDEM1 = 8
    SRC1STRIDEM1 = 8
    MAX_ALLOW_UB = 253952
    DT_INT32 = 3
    DT_INT64 = 9
    SCALAR_255 = 255
    # get available ub size
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    UB_SIZE = MAX_ALLOW_UB if UB_SIZE > MAX_ALLOW_UB else UB_SIZE
    # get available l1 size
    L1_SIZE = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


# 'pylint: disable=too-many-lines,invalid-name,too-many-arguments,consider-using-in
# 'pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# 'pylint: disable=too-many-statements,no-self-use,too-few-public-methods
# 'pylint: disable=unused-argument
def check_supported(x, y, argmax, ksize, strides, pads, dtype=Constant.DT_INT32, dilation=(1, 1, 1, 1),
                    ceil_mode=False, kernel_name="max_pool_with_argmax_v1"):
    """
    check whether ai_core is supported
    """
    if ksize[1] * ksize[2] > Constant.SCALAR_255:
        reason = "ksize is too large, ksize is %s" % (str(ksize),)
        return False, reason

    return True, ""


# 'pylint: disable=too-many-lines,invalid-name,too-many-arguments,consider-using-in
# 'pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# 'pylint: disable=too-many-statements,no-self-use,too-few-public-methods
# 'pylint: disable=unused-argument
def _check_param(x, ksize, strides, padding, dtype, dilation, ceil_mode, kernel_name):
    """
    check parameters, if one is invalid, then raise error
    Parameters
    ----------
    x: dict
        shape and datatype
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: list or tuple
    kernel_name: str
    Returns
    -------
    None
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()

    if input_dtype != "float16":
        raise RuntimeError("Only support float16")

    # the format of x must be NC1HWC0
    if len(input_shape) != 5:
        raise RuntimeError("invalid shape params, input feature map must be "
                           "5D format in kernel.")
    # get shape info of feature map in NC1HWC0 format
    c0_size = input_shape[4]

    if c0_size != 16:
        raise RuntimeError("invalid featur map shape params, "
                           "C0 must be equal to 16")

    if len(ksize) != 4:
        raise RuntimeError("Invalid ksize params, ksize dim must be 4.")

    if ksize[0] != 1 or ksize[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across width/height, and other ksize "
                           "dimension should be one")
    if len(strides) != 4:
        raise RuntimeError("Invalid strides params, strides dim must be 4.")

    if strides[0] != 1 or strides[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across width/height, and other strides dimension "
                           "should be one")
    if len(padding) != 4:
        raise RuntimeError("Invalid padding params, padding dim must be 4.")

    if padding[0] != 1 or padding[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across width/height, and other padding dimension "
                           "should be one")
    if len(strides) != 4:
        raise RuntimeError("Invalid strides params, strides dim must be 4.")

    if strides[0] != 1 or strides[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across width/height, and other strides dimension "
                           "should be one")
    if len(dilation) != 4:
        raise RuntimeError("Invalid dilation params, dilation dim must be 4.")

    if dilation[0] != 1 or dilation[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across width/height, and other dilation "
                           "dimension should be one")
    if ceil_mode is not True and ceil_mode is not False:
        raise RuntimeError("MaxPoolWithArgmax only supports ceil_mode across "
                           "True/False, and other string not support!")
    if dtype != Constant.DT_INT32 and dtype != Constant.DT_INT64:
        raise RuntimeError("MaxPoolWithArgmax only supports output indices data type: "
                           "int32, int64, and other data type not support!")
    if ksize[1] * ksize[2] > 255:
        raise RuntimeError("invalid window params, kernel_h*kernel_w should be <= 255")


class MaxPoolWithargmaxPytorch:
    """
    Function: use to finish MaxPoolWithargmax main functions
    Modify : 2021-03-10
    """

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
        padding: list int
            The value of padding in all dimention, (1, padh, padw, 1).
        dilation: list int
            A parameter that controls the stride of elements in the window.
        ceil_mode: Bool
            If True, will use ceil instead of floor to compute the output
            shape
        dtype: int
            The output indices data type, only support int32 or int64.
        kernel_name: str
            The kernel's name
        Returns
        -------
        None
        """
        self.input_dtype = x.get("dtype").lower()

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
        self.core_num = tik.Dprofile("v100", "cloud").get_aicore_num()

        self.ksize = ksize
        self.strides = strides
        self.ceil_mode = ceil_mode
        self.dilation_h = dilation[1]
        self.dilation_w = dilation[2]
        self.pad_h = padding[1]
        self.pad_w = padding[2]
        self.dtype = dtype
        self.kernel_name = kernel_name

        self.c0_size = 16

        self.kernel_h = self.ksize[1]
        self.kernel_w = self.ksize[2]
        self.stride_h = self.strides[1]
        self.stride_w = self.strides[2]
        self.fmap_w = self.kernel_h * self.kernel_w
        self.pad = self._calc_out_size_and_pad()
        # scalar for load3d
        self.scalar_source_h = self.tik_instance.Scalar(dtype="int64")
        self.scalar_source_w = self.tik_instance.Scalar(dtype="int64")

        if self.input_dtype == "float16":
            self.pad_value = Constant.MIN_VALUE_FP16
        # input and output
        self.input_fmap_gm = self.tik_instance.Tensor(self.input_dtype, (constant_util.SHAPE_SIZE_LIMIT,),
                                                      name="input_fmap_gm", scope=tik.scope_gm)
        self.output_max_gm = self.tik_instance.Tensor(self.input_dtype, (constant_util.SHAPE_SIZE_LIMIT,),
                                                      name="output_max_gm", scope=tik.scope_gm)
        self.output_mask_gm = self.tik_instance.Tensor("uint16", (constant_util.SHAPE_SIZE_LIMIT,),
                                                       name="output_mask_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.mask_zero = self.tik_instance.Scalar("int64")
        # tiling params
        self.tiling_mode = None
        self.need_core_num = None
        self.nc1_per_core = None
        self.nc1_last_core = None
        self.batch_size = None
        self.c1_size = None
        self.input_h = None
        self.input_w = None
        self.input_wh = None
        self.nc1 = None
        self.output_w = None
        self.output_h = None
        self.fmap_h = None
        self.fmap_h_num = None
        self.output_wh = None
        self.mask_tmp = None
        self.cut_h_size = None
        self.cut_stride = None
        self.cut_h_num = None
        self.flag_cut_h = None
        self.cut_w_size = None
        self.cut_w_stride = None
        self.cut_w_num = None

    def get_tiling_params(self):
        """
        get runtime params from tiling
        :return: None
        """
        self.tiling_mode = self.tik_instance.Scalar("int32")
        self.need_core_num = self.tik_instance.Scalar("int32")
        self.nc1_per_core = self.tik_instance.Scalar("int32")
        self.nc1_last_core = self.tik_instance.Scalar("int32")
        self.batch_size = self.tik_instance.Scalar("int32")
        self.c1_size = self.tik_instance.Scalar("int32")
        self.input_h = self.tik_instance.Scalar("int32")
        self.input_w = self.tik_instance.Scalar("int32")
        self.input_wh = self.tik_instance.Scalar("int32")
        self.nc1 = self.tik_instance.Scalar("int32")
        self.output_w = self.tik_instance.Scalar("int32")
        self.output_h = self.tik_instance.Scalar("int32")
        self.fmap_h = self.tik_instance.Scalar("int32")
        self.fmap_h_num = self.tik_instance.Scalar("int32")
        self.output_wh = self.tik_instance.Scalar("int32")
        self.mask_tmp = self.tik_instance.Scalar("int32")
        self.cut_h_size = self.tik_instance.Scalar("int32")
        self.cut_stride = self.tik_instance.Scalar("int32")
        self.cut_h_num = self.tik_instance.Scalar("int32")
        self.flag_cut_h = self.tik_instance.Scalar("int32")
        self.cut_w_size = self.tik_instance.Scalar("int32")
        self.cut_w_stride = self.tik_instance.Scalar("int32")
        self.cut_w_num = self.tik_instance.Scalar("int32")

        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", shape=(Constant.TILING_NUM,), scope=tik.scope_ubuf,
                                                 name="tiling_ub")
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
            self.tiling_mode.set_as(tiling_ub[0])
            self.need_core_num.set_as(tiling_ub[1])
            self.nc1_per_core.set_as(tiling_ub[2])
            self.nc1_last_core.set_as(tiling_ub[3])
            self.batch_size.set_as(tiling_ub[4])
            self.c1_size.set_as(tiling_ub[5])
            self.input_h.set_as(tiling_ub[6])
            self.input_w.set_as(tiling_ub[7])
            self.input_wh.set_as(tiling_ub[8])
            self.nc1.set_as(tiling_ub[9])
            self.output_w.set_as(tiling_ub[10])
            self.output_h.set_as(tiling_ub[11])
            self.fmap_h.set_as(tiling_ub[12])
            self.fmap_h_num.set_as(tiling_ub[13])
            self.output_wh.set_as(tiling_ub[14])
            self.mask_tmp.set_as(tiling_ub[15])
            self.cut_h_size.set_as(tiling_ub[16])
            self.cut_stride.set_as(tiling_ub[17])
            self.cut_h_num.set_as(tiling_ub[18])
            self.flag_cut_h.set_as(tiling_ub[19])
            self.cut_w_size.set_as(tiling_ub[20])
            self.cut_w_stride.set_as(tiling_ub[21])
            self.cut_w_num.set_as(tiling_ub[22])

    def compute_no_cut(self, block_index):
        """
        no need cut
        :param block_index:
        :return:
        """
        per_core = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(block_index != self.need_core_num - 1):
            per_core.set_as(self.nc1_per_core)

        with self.tik_instance.else_scope():
            per_core.set_as(self.nc1_last_core)

        with self.tik_instance.for_range(0, per_core) as nc1_index:
            self._fun_no_cut(block_index, nc1_index, self.nc1_per_core)

    def compute_cut_h(self, block_index):
        """
        compute cut h
        :param block_index:
        :return:
        """
        per_core = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(block_index != self.need_core_num - 1):
            per_core.set_as(self.nc1_per_core)

        with self.tik_instance.else_scope():
            per_core.set_as(self.nc1_last_core)

        with self.tik_instance.for_range(0, per_core) as nc1_cuth_index:
            self._fun_only_cut_h(block_index, nc1_cuth_index, self.cut_h_size, self.cut_stride,
                                 self.cut_h_num, self.nc1_per_core, self.flag_cut_h)

    def compute_cut_h_w(self, block_index):
        """
        compute cut_h_w
        :param block_index:
        :return:
        """
        per_core = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(block_index != self.need_core_num - 1):
            per_core.set_as(self.nc1_per_core)

        with self.tik_instance.else_scope():
            per_core.set_as(self.nc1_last_core)

        with self.tik_instance.for_range(0, per_core) as nc1_cuth_index:
            self._fun_need_cut_h_w(block_index, nc1_cuth_index, self.cut_h_size, self.cut_stride,
                                   self.cut_h_num, self.nc1_per_core, self.flag_cut_h)

    def compute(self, kernel_name):
        """
        op compute
        :param kernel_name:
        :return:
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_index:
            self.get_tiling_params()
            with self.tik_instance.if_scope(block_index < self.need_core_num):
                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE0):
                    with self.tik_instance.new_stmt_scope():
                        self.compute_no_cut(block_index)

                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE1):
                    with self.tik_instance.new_stmt_scope():
                        self.compute_cut_h(block_index)

                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE2):
                    with self.tik_instance.new_stmt_scope():
                        self.compute_cut_h_w(block_index)

        self.tik_instance.BuildCCE(kernel_name=kernel_name, inputs=self.input_fmap_gm,
                                   outputs=(self.output_max_gm, self.output_mask_gm), flowtable=[self.tiling_gm])
        ceil_mode_int = 1 if self.ceil_mode else 0
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num, "ub_size": Constant.UB_SIZE,
                                                            "l1_size": Constant.L1_SIZE, "kernel_h": self.kernel_h,
                                                            "kernel_w": self.kernel_w, "stride_h": self.stride_h,
                                                            "stride_w": self.stride_w, "pad_h": self.pad_h,
                                                            "pad_w": self.pad_w, "dilation_h": self.dilation_h,
                                                            "dilation_w": self.dilation_w, "ceil_mode": ceil_mode_int})
        return self.tik_instance

    def _fun_no_cut(self, block_index, nc1_index, nc1_size):
        """
        funtion while no need cut H

        Parameters
        ----------
        block_index: index of block
        nc1_index: index of nc1

        Returns
        -------
        none
        """

        fmap_l1_shape = (28672, self.c0_size)
        input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype, fmap_l1_shape, name="input_fmap_l1",
                                                 scope=tik.scope_cbuf)
        fmap_shape_ub = (4096, self.c0_size)
        fmap_ub = self.tik_instance.Tensor(self.input_dtype, fmap_shape_ub, name="fmap_ub",
                                           scope=tik.scope_ubuf)
        mask_shape_ub = (256, self.c0_size)
        mask_ub = self.tik_instance.Tensor("uint16", mask_shape_ub, name="mask_ub", scope=tik.scope_ubuf)
        data_x_max = self.tik_instance.Tensor("float16", (3360, 16), name="data_x_max",
                                              scope=tik.scope_ubuf)

        zero_scalar = self.tik_instance.Scalar("int64", init_value=0)
        mask_pow = self.tik_instance.Scalar("int64", init_value=1)
        self.pow2(self.mask_tmp, mask_pow)
        self.mask_zero.set_as(2 ** 16 - mask_pow)
        # copy input fmap from gm to l1
        gm_l1_burst_len = self.input_wh
        self.tik_instance.data_move(
            input_fmap_l1, self.input_fmap_gm[(block_index * nc1_size + nc1_index) * self.input_wh * self.c0_size],
            0, 1, gm_l1_burst_len, 0, 0)

        # load3dv1, from l1 to ub, every time process 16x  kernels
        with self.tik_instance.for_range(0, self.fmap_h_num) as h_index:
            source_h = \
                (((h_index * 256 * self.fmap_w) // (16 * self.fmap_w)) // self.output_w) * self.stride_h - self.pad[2]

            source_w = \
                (((h_index * 256 * self.fmap_w) // (16 * self.fmap_w)) % self.output_w) * self.stride_w - self.pad[0]
            self.scalar_source_h.set_as(source_h)
            self.scalar_source_w.set_as(source_w)

            self.tik_instance.load3dv1(fmap_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0], self.pad,
                                       self.input_h, self.input_w, 0, 0, 0, self.scalar_source_w,
                                       self.scalar_source_h, self.stride_w, self.stride_h,
                                       self.kernel_w, self.kernel_h, self.dilation_w, self.dilation_h, 1, 0,
                                       self.fmap_w, 0, self.pad_value)

        with self.tik_instance.if_scope(self.fmap_w != 1):
            # calc max_pool and max_indices
            self._calc_max_and_mask(self.fmap_h_num, fmap_ub, data_x_max, mask_ub)

            # move max output to gm
            self.tik_instance.data_move(
                self.output_max_gm[(block_index * nc1_size + nc1_index) * self.output_wh * self.c0_size],
                data_x_max[0], 0, 1, self.fmap_h, 0, 0)

            # remove kernel size same maxval position
            self._remove_repeated_fun(mask_ub)

        with self.tik_instance.else_scope():
            # move max output to gm
            self.tik_instance.data_move(
                self.output_max_gm[(block_index * nc1_size + nc1_index) * self.output_wh * self.c0_size], fmap_ub[0],
                0, 1, self.fmap_h, 0, 0)
            self._dup_mask_fun(mask_ub, self.fmap_h_num)

        with self.tik_instance.for_range(0, self.fmap_w) as w_index:
            offset_output_mask = \
                (block_index * nc1_size + nc1_index) * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size
            with self.tik_instance.if_scope(self.mask_zero != 0):
                with self.tik_instance.if_scope(self.fmap_w != 1):
                    self.tik_instance.vector_dup(
                        [zero_scalar, self.mask_zero],
                        mask_ub[w_index * self.fmap_h_num * self.c0_size + self.fmap_h_num * 16 - 16],
                        0, 1, 1, 8)

            self.tik_instance.data_move(
                self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                mask_ub[w_index * self.fmap_h_num * self.c0_size],
                0, 1, self.fmap_h_num, 0, 0)

    def _clean_fp16_data_ub(self, input_ub, length, value):
        repeat_time = self.tik_instance.Scalar("int32", name="repeat_time")
        repeat_time.set_as(length // Constant.ALIGN16)
        max_iter_num = 255
        with self.tik_instance.if_scope(repeat_time < max_iter_num):
            self.tik_instance.vector_dup(Constant.ALIGN16, input_ub, value, repeat_time, 1, 1)
        with self.tik_instance.else_scope():
            iter_times = repeat_time // max_iter_num
            iter_max_len = max_iter_num * Constant.ALIGN16
            iter_res_time = repeat_time - iter_times * max_iter_num
            with self.tik_instance.if_scope(iter_times > 0):
                with self.tik_instance.for_range(0, iter_times) as num:
                    self.tik_instance.vector_dup(Constant.ALIGN16, input_ub[num * iter_max_len],
                                                 value, max_iter_num, 1, 1)
            with self.tik_instance.if_scope(iter_res_time > 0):
                self.tik_instance.vector_dup(Constant.ALIGN16, input_ub[iter_times * iter_max_len],
                                             value, iter_res_time, 1, 1)

    def _calc_only_cut_h_branch(self, cut_h_index, cut_h_size, cut_stride, cut_h_num, input_fmap_l1,
                                fmap_img2col_ub, fmap_img2col_cut_h, fmap_cut_h_num, nc1_num):
        """
        calc only cut H
        Parameters
        ----------
        cut_h_index: index of cuth
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        input_fmap_l1: fmag in l1
        fmap_img2col_ub: fmag in ub
        fmap_img2col_cut_h: fmag cutH
        mask_shape_ub: shape of mask
        nc1_num: num of n*c1
        Returns
        -------
        none
        """
        fmap_img2col_cut_h_num = self.tik_instance.Scalar("int32", name="fmap_img2col_cut_h_num")
        fmap_img2col_cut_h_num.set_as((fmap_img2col_cut_h + 15) // 16)
        mask_ub = self.tik_instance.Tensor("uint16", (256, 16), name="mask_ub", scope=tik.scope_ubuf)
        data_x_max = self.tik_instance.Tensor("float16", (3360, 16), name="data_x_max",
                                              scope=tik.scope_ubuf)
        len_tmp = self.tik_instance.Scalar(dtype="int32", init_value=0)
        len_tmp1 = self.tik_instance.Scalar(dtype="int32", init_value=0)
        pad_top = self.tik_instance.Scalar(dtype="int32", init_value=0)
        pad_bottom = self.tik_instance.Scalar(dtype="int32", init_value=0)
        gm_l1_burst_len_1 = self.tik_instance.Scalar(dtype="int32", init_value=0)
        gm_tem = self.tik_instance.Scalar(dtype="int32", init_value=0)
        last_tem = self.tik_instance.Scalar(dtype="int32", init_value=0)
        gm_max_burst_len = self.tik_instance.Scalar("int32")
        cut_h_tail = self.tik_instance.Scalar("int32", name="cut_h_tail")
        tmp_tail = self.tik_instance.Scalar("int32", name="tmp_tail")
        out_size_h_tail = self.tik_instance.Scalar("int32", name="out_size_h_tail")
        fmap_img2col_h_tail_num = self.tik_instance.Scalar("int32", name="fmap_img2col_h_tail_num")
        mask_zero_cut = self.tik_instance.Scalar("int64", init_value=1)
        zero_scalar = self.tik_instance.Scalar("int64", init_value=0)
        gm_l1_burst_len = self.tik_instance.Scalar("int32", name="gm_l1_burst_len")

        with self.tik_instance.new_stmt_scope():
            zero_ub = self.tik_instance.Tensor("float16", (480 * self.c0_size,),
                                               name="zero_ub", scope=tik.scope_ubuf)
            self._clean_fp16_data_ub(zero_ub, self.input_w * self.c0_size, self.pad_value)
            with self.tik_instance.for_range(0, cut_h_size) as iter_num:  # 1
                self.tik_instance.data_move(input_fmap_l1[iter_num * self.input_w * self.c0_size],
                                            zero_ub, 0, 1, self.input_w, 0, 0)

        with self.tik_instance.if_scope(cut_h_index != 0):
            with self.tik_instance.if_scope(cut_h_index != (cut_h_num - 1)):
                len_tmp.set_as(cut_h_size)
                # copy input fmap from gm to l1
                with self.tik_instance.if_scope(cut_h_size >= (self.input_h + self.pad[2] - cut_stride * cut_h_index)):
                    len_tmp.set_as(self.input_h + self.pad[2] - cut_stride * cut_h_index)
                len_tmp1.set_as(len_tmp)
                with self.tik_instance.if_scope(len_tmp >= (cut_h_size - self.pad[2] + cut_stride * cut_h_index)):
                    len_tmp1.set_as(cut_h_size - self.pad[2] + cut_stride * cut_h_index)
                gm_l1_burst_len_1.set_as(len_tmp1 * self.input_w)
                pad_top.set_as(0)
                with self.tik_instance.if_scope(self.pad[2] - cut_stride * cut_h_index > 0):
                    pad_top.set_as(self.pad[2] - cut_stride * cut_h_index)
                pad_bottom.set_as(0)
                with self.tik_instance.if_scope(
                        cut_stride * cut_h_index + cut_h_size - self.pad[2] - self.input_h > 0):
                    pad_bottom.set_as(cut_stride * cut_h_index + cut_h_size - self.pad[2] - self.input_h)
                gm_tem.set_as(0)
                with self.tik_instance.if_scope(cut_h_index * cut_stride - self.pad[2] > 0):
                    gm_tem.set_as(cut_h_index * cut_stride - self.pad[2])
                with self.tik_instance.if_scope((gm_tem + (gm_l1_burst_len_1 // self.input_w)) > self.input_h):
                    gm_l1_burst_len_1.set_as((self.input_h - gm_tem) * self.input_w)
                self.tik_instance.data_move(
                    input_fmap_l1, self.input_fmap_gm[nc1_num * self.input_h * self.input_w * self.c0_size +
                                                      gm_tem * self.input_w * self.c0_size],
                    0, 1, gm_l1_burst_len_1, 0, 0)
                with self.tik_instance.for_range(0, fmap_img2col_cut_h_num) as h_index:
                    source_h = (((h_index * 256 * self.fmap_w) // (16 * self.fmap_w)) //
                                self.output_w) * self.stride_h - pad_top
                    source_w = (((h_index * 256 * self.fmap_w) // (16 * self.fmap_w)) %
                                self.output_w) * self.stride_w - self.pad[0]
                    self.tik_instance.load3dv1(
                        fmap_img2col_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                        (self.pad[0], self.pad[1], pad_top, pad_bottom), cut_h_size - pad_top - pad_bottom,
                        self.input_w, 0, 0, 0, source_w, source_h, self.stride_w,
                        self.stride_h, self.kernel_w, self.kernel_h, 1, 1, 1, 0, self.fmap_w, 0, self.pad_value)
                with self.tik_instance.if_scope(self.fmap_w != 1):
                    self._calc_max_and_mask(fmap_img2col_cut_h_num, fmap_img2col_ub, data_x_max, mask_ub)
                    # move max output to gm
                    gm_max_burst_len.set_as(fmap_img2col_cut_h)
                    self.tik_instance.data_move(
                        self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size +
                                           cut_h_index * fmap_img2col_cut_h * self.c0_size],
                        data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h)
                with self.tik_instance.else_scope():
                    # move max output to gm
                    gm_max_burst_len.set_as(fmap_img2col_cut_h)
                    self.tik_instance.data_move(
                        self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size + cut_h_index *
                                           fmap_img2col_cut_h * self.c0_size],
                        fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._dup_mask_fun(mask_ub, fmap_cut_h_num)
                with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                    offset_output_mask = nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + \
                                         cut_h_index * fmap_img2col_cut_h
                    self.tik_instance.data_move(
                        self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                        mask_ub[w_index * fmap_img2col_cut_h_num * self.c0_size],
                        0, 1, fmap_img2col_cut_h_num, 0, 0)
            with self.tik_instance.else_scope():
                cut_h_tail.set_as(self.input_h + self.pad[2] - cut_stride * (cut_h_num - 1))
                with self.tik_instance.if_scope(cut_h_tail > cut_h_size):
                    cut_h_tail.set_as(cut_h_size)
                tmp_tail.set_as(cut_h_tail - self.kernel_h + self.stride_h + self.pad[3])
                with self.tik_instance.if_scope(tmp_tail < self.stride_h):
                    out_size_h_tail.set_as(cut_h_tail - self.kernel_h + self.stride_h + self.pad[3])
                with self.tik_instance.else_scope():
                    out_size_h_tail.set_as(tmp_tail // self.stride_h)

                fmap_img2col_h_tail = self.output_w * out_size_h_tail
                fmap_img2col_h_tail_num.set_as((fmap_img2col_h_tail + 15) // 16)
                # copy input fmap from gm to l1
                gm_l1_burst_len.set_as(cut_h_tail * self.input_w)
                gm_l1_burst_len_1.set_as(gm_l1_burst_len)

                last_tem.set_as(0)
                with self.tik_instance.if_scope(cut_h_index * cut_stride - self.pad[2] > 0):
                    last_tem.set_as(cut_h_index * cut_stride - self.pad[2])

                with self.tik_instance.if_scope((last_tem + (gm_l1_burst_len // self.input_w)) > self.input_h):
                    gm_l1_burst_len_1.set_as((self.input_h - last_tem) * self.input_w)
                self.tik_instance.data_move(
                    input_fmap_l1, self.input_fmap_gm[nc1_num * self.input_h * self.input_w * self.c0_size +
                                                      last_tem * self.input_w * self.c0_size],
                    0, 1, gm_l1_burst_len_1, 0, 0)
                pad_top.set_as(0)
                with self.tik_instance.if_scope(self.pad[2] - cut_stride * cut_h_index > 0):
                    pad_top.set_as(self.pad[2] - cut_stride * cut_h_index)
                pad_bottom.set_as(0)
                with self.tik_instance.if_scope(
                        cut_stride * cut_h_index + cut_h_size - self.pad[2] - self.input_h > 0):
                    pad_bottom.set_as(cut_stride * cut_h_index + cut_h_size - self.pad[2] - self.input_h)
                with self.tik_instance.for_range(0, fmap_img2col_h_tail_num) as h_index:
                    source_h = (((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) /
                                self.output_w) * self.stride_h - pad_top
                    source_w = ((((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % self.output_w) *
                                self.stride_w - self.pad[0])
                    self.tik_instance.load3dv1(
                        fmap_img2col_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                        (self.pad[0], self.pad[1], pad_top, pad_bottom), cut_h_tail, self.input_w, 0, 0, 0,
                        source_w, source_h, self.stride_w, self.stride_h, self.kernel_w,
                        self.kernel_h, 1, 1, 1, 0, self.fmap_w, 0, self.pad_value)
                with self.tik_instance.if_scope(self.fmap_w != 1):
                    self._calc_max_and_mask(fmap_img2col_h_tail_num, fmap_img2col_ub, data_x_max, mask_ub,
                                            fmap_img2col_cut_h_num)
                    # move max output to gm
                    with self.tik_instance.if_scope(tmp_tail < self.stride_h):
                        gm_max_burst_len.set_as(self.output_w)
                    with self.tik_instance.else_scope():
                        gm_max_burst_len.set_as(fmap_img2col_h_tail)

                    with self.tik_instance.if_scope((cut_h_index * fmap_img2col_cut_h * self.c0_size) <
                                                    (self.output_h * self.output_w * self.c0_size)):
                        self.tik_instance.data_move(
                            self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size +
                                               cut_h_index * fmap_img2col_cut_h * self.c0_size],
                            data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)

                    self._remove_repeated_fun(mask_ub, fmap_img2col_h_tail, 0, 0, fmap_img2col_cut_h)
                with self.tik_instance.else_scope():
                    # move max output to gm
                    with self.tik_instance.if_scope(tmp_tail < self.stride_h):
                        gm_max_burst_len.set_as(self.output_w)
                    with self.tik_instance.else_scope():
                        gm_max_burst_len.set_as(fmap_img2col_h_tail)

                    self.tik_instance.data_move(
                        self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size +
                                           cut_h_index * fmap_img2col_cut_h * self.c0_size],
                        fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._dup_mask_fun(mask_ub, fmap_cut_h_num)
                mask_cut = fmap_img2col_h_tail_num * 16 - fmap_img2col_h_tail
                self.pow2(mask_cut, mask_zero_cut)
                mask_zero_cut.set_as(2 ** 16 - mask_zero_cut)
                offset_output_mask = nc1_num * (
                        self.fmap_h_num + 1) * self.fmap_w * self.c0_size + cut_h_index * fmap_img2col_cut_h

                len_tmp1.set_as(cut_h_index * fmap_img2col_cut_h +
                                (self.fmap_w - 1) * (self.fmap_h_num + 1) * self.c0_size)
                len_tmp.set_as(((self.fmap_h_num + 1) * self.fmap_w * self.c0_size - len_tmp1) // self.c0_size)
                with self.tik_instance.if_scope(len_tmp > fmap_img2col_h_tail_num):
                    len_tmp.set_as(fmap_img2col_h_tail_num)

                with self.tik_instance.if_scope(len_tmp > 0):
                    with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                        with self.tik_instance.if_scope(tik.all(mask_zero_cut != 0, self.fmap_w != 1)):
                            self.tik_instance.vector_dup(
                                [zero_scalar, mask_zero_cut], mask_ub[w_index * fmap_img2col_cut_h_num * self.c0_size +
                                                                      fmap_img2col_h_tail_num * 16 - 16], 0, 1, 1, 8)
                        self.tik_instance.data_move(
                            self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                            mask_ub[w_index * fmap_img2col_cut_h_num * self.c0_size],
                            0, 1, len_tmp, 0, 0)
        with self.tik_instance.else_scope():
            # copy input fmap from gm to l1
            gm_l1_burst_len.set_as((cut_h_size - self.pad[2]) * self.input_w)
            with self.tik_instance.if_scope((cut_h_size - self.pad[2]) > self.input_h):
                gm_l1_burst_len.set_as(self.input_h * self.input_w)
            self.tik_instance.data_move(
                input_fmap_l1, self.input_fmap_gm[nc1_num * self.input_h * self.input_w * self.c0_size],
                0, 1, gm_l1_burst_len, 0, 0)
            with self.tik_instance.for_range(0, fmap_img2col_cut_h_num) as h_index:
                source_h = (((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) /
                            self.output_w) * self.stride_h - self.pad[2]
                source_w = (((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) %
                            self.output_w) * self.stride_w - self.pad[0]
                self.tik_instance.load3dv1(
                    fmap_img2col_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                    (self.pad[0], self.pad[1], self.pad[2], 0),
                    (cut_h_size - self.pad[2]), self.input_w, 0, 0, 0, source_w, source_h,
                    self.stride_w, self.stride_h, self.kernel_w, self.kernel_h, self.dilation_w, self.dilation_h,
                    1, 0, self.fmap_w, 0, self.pad_value)
            with self.tik_instance.if_scope(self.fmap_w != 1):
                self._calc_max_and_mask(fmap_img2col_cut_h_num, fmap_img2col_ub, data_x_max, mask_ub)
                # move max output to gm
                gm_max_burst_len.set_as(fmap_img2col_cut_h)
                self.tik_instance.data_move(
                    self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size],
                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h)
            with self.tik_instance.else_scope():
                # move max output to gm
                gm_max_burst_len.set_as(fmap_img2col_cut_h)
                self.tik_instance.data_move(
                    self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size],
                    fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                self._dup_mask_fun(mask_ub, fmap_cut_h_num)
            with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                offset_output_mask = \
                    nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + cut_h_index * fmap_img2col_cut_h
                self.tik_instance.data_move(
                    self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                    mask_ub[w_index * fmap_img2col_cut_h_num * self.c0_size], 0, 1, fmap_img2col_cut_h_num, 0, 0)

    def _fun_only_cut_h(self, block_index, nc1_cuth_index, cut_h_size,
                        cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h):
        """
        funtion only cut H

        Parameters
        ----------
        block_index: index of block
        nc1_cuth_index: index of nc1_cuth
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        nc1_cuth_size: size of nc1_cuth
        flag_cut_h: int

        Returns
        -------
        none
        """
        fmap_l1_shape = (28672, self.c0_size)
        input_fmap_l1 = self.tik_instance.Tensor(
            self.input_dtype, fmap_l1_shape, name="input_fmap_l1", scope=tik.scope_cbuf)
        fmap_cut_h_num = self.tik_instance.Scalar("int32", name="fmap_cut_h_num")
        fmap_cut_h = self.tik_instance.Scalar("int32", name="fmap_cut_h")
        out_size_cut_h = (cut_h_size - self.kernel_h + self.stride_h) // self.stride_h
        fmap_cut_h.set_as(self.output_w * out_size_cut_h)
        fmap_cut_h_num.set_as((fmap_cut_h + 15) // 16)
        fmap_shape_ub = (4096 * self.c0_size,)

        fmap_ub = self.tik_instance.Tensor(self.input_dtype, fmap_shape_ub, name="fmap_ub", scope=tik.scope_ubuf)
        with self.tik_instance.if_scope(flag_cut_h == 1):
            cur_h_idx = (block_index * nc1_cuth_size + nc1_cuth_index) % cut_h_num
            nc1_num = (block_index * nc1_cuth_size + nc1_cuth_index) // cut_h_num
            self._calc_only_cut_h_branch(cur_h_idx, cut_h_size, cut_stride, cut_h_num, input_fmap_l1,
                                         fmap_ub, fmap_cut_h, fmap_cut_h_num, nc1_num)
        with self.tik_instance.else_scope():
            nc1_num = block_index * nc1_cuth_size + nc1_cuth_index
            with self.tik_instance.for_range(0, cut_h_num, thread_num=1) as cur_h_idx:  # 5
                self._calc_only_cut_h_branch(cur_h_idx, cut_h_size, cut_stride, cut_h_num, input_fmap_l1,
                                             fmap_ub, fmap_cut_h, fmap_cut_h_num, nc1_num)

    def _calc_need_cut_h_w(self, nc1_num, cut_h_size, cut_h_num, cur_h_idx, cut_stride):
        """
        funtion need cut H and W while l1 not enough

        Parameters
        ----------
        nc1_num: num of n*c1
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        cur_h_idx: index of cuth

        Returns
        -------
        none
        """
        gm_l1_burst_len = self.tik_instance.Scalar("int32", name="gm_l1_burst_len")
        gm_max_burst_len = self.tik_instance.Scalar("int32")
        mv_height = self.tik_instance.Scalar("int32")
        cut_w_tail = self.tik_instance.Scalar("int32")
        fmap_tail_w = self.tik_instance.Scalar("int32")
        fmap_tail_w_num = self.tik_instance.Scalar("int32")
        fmap_cut_w_num = self.tik_instance.Scalar("int32")
        mask_zero_w = self.tik_instance.Scalar("int64", init_value=1)
        zero_w = self.tik_instance.Scalar("int64", init_value=0)

        cut_w_size, cut_w_stride, cut_w_num = self.cut_w_size, self.cut_w_stride, self.cut_w_num
        fmap_l1_shape = (28672, self.c0_size)
        input_fmap_l1 = self.tik_instance.Tensor(
            self.input_dtype, fmap_l1_shape, name="input_fmap_l1", scope=tik.scope_cbuf)
        with self.tik_instance.for_range(0, cut_w_num) as cut_w_index:
            out_size_cut_h = (cut_h_size - self.kernel_h + self.stride_h) // self.stride_h
            fmap_cut_h = self.output_w * out_size_cut_h
            out_size_cut_w = (cut_w_size - self.kernel_w + self.stride_w) // self.stride_w
            fmap_cut_w = out_size_cut_w
            fmap_cut_w_num.set_as((fmap_cut_w + 15) // 16)
            fmap_shape_ub = (4096, self.c0_size)
            fmap_ub = self.tik_instance.Tensor(self.input_dtype, fmap_shape_ub, name="fmap_ub", scope=tik.scope_ubuf)
            mask_shape_ub = (256, self.c0_size)
            mask_ub = self.tik_instance.Tensor("uint16", mask_shape_ub, name="mask_ub", scope=tik.scope_ubuf)
            data_x_max = self.tik_instance.Tensor(
                "float16", (3360, 16), name="data_x_max", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(cur_h_idx != 0):
                with self.tik_instance.if_scope(cur_h_idx != (cut_h_num - 1)):
                    # copy input fmap from gm to l1
                    gm_l1_burst_len.set_as(cut_h_size * self.input_w * self.c0_size // 16)
                    self.tik_instance.data_move(
                        input_fmap_l1,
                        self.input_fmap_gm[nc1_num * self.input_wh * self.c0_size +
                                           (cur_h_idx * cut_stride - self.pad[2]) * self.input_w * self.c0_size],
                        0, 1, gm_l1_burst_len, 0, 0)
                    with self.tik_instance.if_scope(cut_w_index != 0):
                        with self.tik_instance.if_scope(cut_w_index != (cut_w_num - 1)):
                            with self.tik_instance.for_range(0, fmap_cut_w_num) as h_index:
                                source_h = 0
                                source_w = ((((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_cut_w) *
                                            self.stride_w + cut_w_stride * cut_w_index - self.pad[0])
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                                    (self.pad[0], self.pad[1], 0, 0),
                                    cut_h_size, self.input_w, 0, 0, 0, self.scalar_source_w, self.scalar_source_h,
                                    self.stride_w, self.stride_h, self.kernel_w, self.kernel_h,
                                    self.dilation_w, self.dilation_h, 1, 0, self.fmap_w, 0, self.pad_value)
                            with self.tik_instance.if_scope(self.fmap_w != 1):
                                self._calc_max_and_mask(fmap_cut_w_num, fmap_ub, data_x_max, mask_ub)
                                # move max output to gm
                                gm_max_burst_len.set_as(fmap_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size +
                                                       cur_h_idx * self.output_w * self.c0_size +
                                                       cut_w_index * fmap_cut_w * self.c0_size],
                                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub, fmap_cut_h, fmap_cut_w)
                            with self.tik_instance.else_scope():
                                # move max output to gm
                                gm_max_burst_len.set_as(fmap_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size +
                                                       cur_h_idx * self.output_w * self.c0_size + cut_w_index *
                                                       fmap_cut_w * self.c0_size],
                                    fmap_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._dup_mask_fun(mask_ub, fmap_cut_w_num)
                            with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + \
                                    cur_h_idx * fmap_cut_h + cut_w_index * fmap_cut_w
                                self.tik_instance.data_move(
                                    self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                                    mask_ub[w_index * fmap_cut_w_num * self.c0_size], 0, 1, fmap_cut_w_num, 0, 0)
                        with self.tik_instance.else_scope():
                            cut_w_tail.set_as(self.input_w + self.pad[0] - cut_w_stride * (cut_w_num - 1))
                            with self.tik_instance.if_scope(cut_w_tail > cut_w_size):
                                cut_w_tail.set_as(cut_w_size)
                            out_size_tail_w = ((cut_w_tail - self.kernel_w + self.stride_w + self.pad[1]) //
                                               self.stride_w)
                            fmap_tail_w.set_as(out_size_tail_w)
                            fmap_tail_w_num.set_as((fmap_tail_w + 15) // 16)
                            with self.tik_instance.for_range(0, fmap_tail_w_num) as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_tail_w) * \
                                    self.stride_w + cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                                    (self.pad[0], self.pad[1], 0, 0), cut_h_size, self.input_w, 0, 0, 0,
                                    self.scalar_source_w, self.scalar_source_h, self.stride_w, self.stride_h,
                                    self.kernel_w, self.kernel_h, self.dilation_w, self.dilation_h, 1, 0,
                                    self.fmap_w, 0, self.pad_value)
                            with self.tik_instance.if_scope(self.fmap_w != 1):
                                self._calc_max_and_mask(
                                    fmap_tail_w_num, fmap_ub, data_x_max, mask_ub, fmap_cut_w_num)
                                # move max output to gm
                                gm_max_burst_len.set_as(fmap_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[nc1_num * self.output_wh * self.c0_size + cur_h_idx *
                                                       self.output_w * self.c0_size + cut_w_index *
                                                       fmap_cut_w * self.c0_size],
                                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub, fmap_cut_h, fmap_tail_w, fmap_cut_w)
                            with self.tik_instance.else_scope():
                                gm_max_burst_len.set_as(fmap_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[nc1_num * self.output_wh * self.c0_size + cur_h_idx *
                                                       self.output_w * self.c0_size + cut_w_index *
                                                       fmap_cut_w * self.c0_size],
                                    fmap_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._dup_mask_fun(mask_ub, fmap_cut_w_num)
                            with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + cur_h_idx * \
                                    fmap_cut_h + cut_w_index * fmap_cut_w
                                self.tik_instance.data_move(
                                    self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                                    mask_ub[w_index * fmap_cut_w_num * self.c0_size], 0, 1, fmap_tail_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(0, fmap_cut_w_num) as h_index:
                            source_h = 0
                            source_w = ((((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_cut_w) *
                                        self.stride_w - self.pad[0])
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                                (self.pad[0], self.pad[1], 0, 0),
                                cut_h_size, self.input_w, 0, 0, 0, self.scalar_source_w, self.scalar_source_h,
                                self.stride_w, self.stride_h, self.kernel_w, self.kernel_h, self.dilation_w,
                                self.dilation_h, 1, 0, self.fmap_w, 0, self.pad_value)
                        with self.tik_instance.if_scope(self.fmap_w != 1):
                            self._calc_max_and_mask(fmap_cut_w_num, fmap_ub, data_x_max, mask_ub)
                            # move max output to gm
                            gm_max_burst_len.set_as(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[nc1_num * self.output_wh * self.c0_size + cur_h_idx *
                                                   self.output_w * self.c0_size], data_x_max[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub, fmap_cut_h, fmap_cut_w)
                        with self.tik_instance.else_scope():
                            # move max output to gm
                            gm_max_burst_len.set_as(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[nc1_num * self.output_wh * self.c0_size + cur_h_idx *
                                                   self.output_w * self.c0_size], fmap_ub[0], 0, 1, gm_max_burst_len,
                                0, 0)
                            self._dup_mask_fun(mask_ub, fmap_cut_w_num)
                        with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + cur_h_idx * fmap_cut_h
                            self.tik_instance.data_move(
                                self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                                mask_ub[w_index * fmap_cut_w_num * self.c0_size], 0, 1, fmap_cut_w_num, 0, 0)
                with self.tik_instance.else_scope():
                    # copy input fmap from gm to l1
                    with self.tik_instance.if_scope(self.input_h - cut_stride * (cut_h_num - 1) + self.pad[2]
                                                    <= cut_h_size):
                        gm_l1_burst_len.set_as((self.input_h - cut_stride * (cut_h_num - 1) + self.pad[2]) *
                                               self.input_w * self.c0_size // 16)
                    with self.tik_instance.else_scope():
                        gm_l1_burst_len.set_as(cut_h_size * self.input_w * self.c0_size // 16)
                    self.tik_instance.data_move(
                        input_fmap_l1, self.input_fmap_gm[nc1_num * self.input_h * self.input_w * self.c0_size +
                                                          (cur_h_idx * cut_stride - self.pad[
                                                              2]) * self.input_w * self.c0_size], 0, 1,
                        gm_l1_burst_len, 0, 0)

                    cur_height = (cut_h_num - 1) * cut_stride - self.pad[2]
                    res_height = (self.input_h - cur_height)
                    mv_height.set_as(cut_h_size - self.pad[3])
                    with self.tik_instance.if_scope((cut_h_size - self.pad[3]) <= res_height):
                        mv_height.set_as(res_height)

                    with self.tik_instance.if_scope(cut_w_index != 0):
                        with self.tik_instance.if_scope(cut_w_index != (cut_w_num - 1)):
                            with self.tik_instance.for_range(0, fmap_cut_w_num) as h_index:
                                source_h = 0
                                source_w = ((((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_cut_w) *
                                            self.stride_w + cut_w_stride * cut_w_index - self.pad[0])
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                                    (self.pad[0], self.pad[1], 0, self.pad[3]), mv_height, self.input_w,
                                    0, 0, 0, self.scalar_source_w, self.scalar_source_h, self.stride_w,
                                    self.stride_h, self.kernel_w, self.kernel_h, self.dilation_w, self.dilation_h,
                                    1, 0, self.fmap_w, 0, self.pad_value)
                            with self.tik_instance.if_scope(self.fmap_w != 1):
                                self._calc_max_and_mask(fmap_cut_w_num, fmap_ub, data_x_max, mask_ub)
                                # move max output to gm
                                gm_max_burst_len.set_as(fmap_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size +
                                                       cur_h_idx * out_size_cut_h * self.output_w * self.c0_size +
                                                       cut_w_index * fmap_cut_w * self.c0_size], data_x_max[0], 0, 1,
                                    gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub, fmap_cut_h, fmap_cut_w)
                            with self.tik_instance.else_scope():
                                # move max output to gm
                                gm_max_burst_len.set_as(fmap_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size +
                                                       cur_h_idx * out_size_cut_h * self.output_w * self.c0_size +
                                                       cut_w_index * fmap_cut_w * self.c0_size],
                                    fmap_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._dup_mask_fun(mask_ub, fmap_cut_w_num)
                            with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + \
                                    cur_h_idx * fmap_cut_h + cut_w_index * fmap_cut_w
                                self.tik_instance.data_move(
                                    self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                                    mask_ub[w_index * fmap_cut_w_num * self.c0_size],
                                    0, 1, fmap_cut_w_num, 0, 0)
                        with self.tik_instance.else_scope():
                            cut_w_tail.set_as(self.input_w + self.pad[0] - cut_w_stride * (cut_w_num - 1))
                            with self.tik_instance.if_scope(cut_w_tail > cut_w_size):
                                cut_w_tail.set_as(cut_w_size)
                            out_size_tail_w = \
                                (cut_w_tail - self.kernel_w + self.stride_w + self.pad[1]) // self.stride_w
                            fmap_tail_w.set_as(out_size_tail_w)
                            fmap_tail_w_num.set_as((fmap_tail_w + 15) // 16)
                            with self.tik_instance.for_range(0, fmap_tail_w_num) as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_tail_w) * \
                                    self.stride_w + cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                                    (self.pad[0], self.pad[1], 0, self.pad[3]),
                                    mv_height, self.input_w, 0, 0, 0, self.scalar_source_w,
                                    self.scalar_source_h, self.stride_w, self.stride_h,
                                    self.kernel_w, self.kernel_h, self.dilation_w, self.dilation_h, 1, 0,
                                    self.fmap_w, 0, self.pad_value)
                            with self.tik_instance.if_scope(self.fmap_w != 1):
                                self._calc_max_and_mask(fmap_tail_w_num, fmap_ub, data_x_max, mask_ub, fmap_cut_w_num)
                                # move max output to gm
                                gm_max_burst_len.set_as(fmap_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size +
                                                       cur_h_idx * out_size_cut_h * self.output_w * self.c0_size +
                                                       cut_w_index * fmap_cut_w * self.c0_size],
                                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub, fmap_cut_h, fmap_tail_w, fmap_cut_w)
                            with self.tik_instance.else_scope():
                                # move max output to gm
                                gm_max_burst_len.set_as(fmap_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size +
                                                       cur_h_idx * out_size_cut_h * self.output_w * self.c0_size +
                                                       cut_w_index * fmap_cut_w * self.c0_size], fmap_ub[0], 0, 1,
                                    gm_max_burst_len,
                                    0, 0)
                                self._dup_mask_fun(mask_ub, fmap_cut_w_num)
                            mask_cut_w = fmap_tail_w_num * 16 - fmap_tail_w
                            self.pow2(mask_cut_w, mask_zero_w)
                            mask_zero_w.set_as(2 ** 16 - mask_zero_w)
                            with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + \
                                    cur_h_idx * fmap_cut_h + cut_w_index * fmap_cut_w
                                with self.tik_instance.if_scope(tik.all(mask_zero_w != 0, self.fmap_w != 1)):
                                    self.tik_instance.vector_dup(
                                        [zero_w, mask_zero_w],
                                        mask_ub[w_index * fmap_cut_w_num * self.c0_size + fmap_tail_w_num * 16 - 16],
                                        0, 1, 1, 8)

                                self.tik_instance.data_move(
                                    self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                                    mask_ub[w_index * fmap_cut_w_num * self.c0_size],
                                    0, 1, fmap_tail_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(0, fmap_cut_w_num) as h_index:
                            source_h = 0
                            source_w = \
                                (((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_cut_w) * \
                                self.stride_w - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_ub[h_index * 256 * self.fmap_w],
                                input_fmap_l1[0], (self.pad[0], self.pad[1], 0, self.pad[3]),
                                mv_height, self.input_w, 0, 0, 0, self.scalar_source_w, self.scalar_source_h,
                                self.stride_w, self.stride_h, self.kernel_w, self.kernel_h,
                                self.dilation_w, self.dilation_h, 1, 0, self.fmap_w, 0, self.pad_value)
                        with self.tik_instance.if_scope(self.fmap_w != 1):
                            self._calc_max_and_mask(fmap_cut_w_num, fmap_ub, data_x_max, mask_ub)
                            # move max output to gm
                            gm_max_burst_len.set_as(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[nc1_num * self.output_wh * self.c0_size + cur_h_idx *
                                                   self.output_w * self.c0_size], data_x_max[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub, fmap_cut_h, fmap_cut_w)
                        with self.tik_instance.else_scope():
                            # move max output to gm
                            gm_max_burst_len.set_as(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[nc1_num * self.output_wh * self.c0_size + cur_h_idx *
                                                   self.output_w * self.c0_size], fmap_ub[0], 0, 1, gm_max_burst_len,
                                0, 0)
                            self._dup_mask_fun(mask_ub, fmap_cut_w_num)
                        with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + cur_h_idx * fmap_cut_h
                            self.tik_instance.data_move(
                                self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                                mask_ub[w_index * fmap_cut_w_num * self.c0_size], 0, 1, fmap_cut_w_num, 0, 0)
            with self.tik_instance.else_scope():
                # copy input fmap from gm to l1
                gm_l1_burst_len.set_as((cut_h_size - self.pad[2]) * self.input_w * self.c0_size // 16)
                self.tik_instance.data_move(
                    input_fmap_l1,
                    self.input_fmap_gm[nc1_num * self.input_wh * self.c0_size], 0, 1, gm_l1_burst_len, 0, 0)
                with self.tik_instance.if_scope(cut_w_index != 0):
                    with self.tik_instance.if_scope(cut_w_index != (cut_w_num - 1)):
                        with self.tik_instance.for_range(0, fmap_cut_w_num) as h_index:
                            source_h = -self.pad[2]
                            source_w = \
                                (((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_cut_w) * \
                                self.stride_w + cut_w_stride * cut_w_index - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                                (self.pad[0], self.pad[1], self.pad[2], 0),
                                (cut_h_size - self.pad[2]), self.input_w, 0, 0, 0, self.scalar_source_w,
                                self.scalar_source_h, self.stride_w, self.stride_h, self.kernel_w, self.kernel_h,
                                self.dilation_w, self.dilation_h, 1, 0, self.fmap_w, 0, self.pad_value)
                        with self.tik_instance.if_scope(self.fmap_w != 1):
                            self._calc_max_and_mask(fmap_cut_w_num, fmap_ub, data_x_max, mask_ub)
                            # move max output to gm
                            gm_max_burst_len.set_as(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[nc1_num * self.output_wh * self.c0_size +
                                                   cut_w_index * fmap_cut_w * self.c0_size],
                                data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub, fmap_cut_h, fmap_cut_w)
                        with self.tik_instance.else_scope():
                            # move max output to gm
                            gm_max_burst_len.set_as(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[nc1_num * self.output_wh * self.c0_size +
                                                   cut_w_index * fmap_cut_w * self.c0_size],
                                fmap_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, fmap_cut_w_num)
                        with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + \
                                cur_h_idx * fmap_cut_h + cut_w_index * fmap_cut_w
                            self.tik_instance.data_move(
                                self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                                mask_ub[w_index * fmap_cut_w_num * self.c0_size], 0, 1, fmap_cut_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        cut_w_tail.set_as(self.input_w + self.pad[0] - cut_w_stride * (cut_w_num - 1))
                        with self.tik_instance.if_scope(cut_w_tail > cut_w_size):
                            cut_w_tail.set_as(cut_w_size)
                        out_size_tail_w = (cut_w_tail - self.kernel_w + self.stride_w + self.pad[1]) // self.stride_w
                        fmap_tail_w.set_as(out_size_tail_w)
                        fmap_tail_w_num.set_as((fmap_tail_w + 15) // 16)
                        with self.tik_instance.for_range(0, fmap_tail_w_num) as h_index:
                            source_h = -self.pad[2]
                            source_w = \
                                (((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_tail_w) * \
                                self.stride_w + cut_w_stride * cut_w_index - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                                (self.pad[0], self.pad[1], self.pad[2], 0), (cut_h_size - self.pad[2]),
                                self.input_w, 0, 0, 0, self.scalar_source_w, self.scalar_source_h,
                                self.stride_w, self.stride_h, self.kernel_w, self.kernel_h,
                                self.dilation_w, self.dilation_h, 1, 0, self.fmap_w, 0, self.pad_value)
                        with self.tik_instance.if_scope(self.fmap_w != 1):
                            self._calc_max_and_mask(fmap_tail_w_num, fmap_ub, data_x_max, mask_ub, fmap_cut_w_num)
                            # move max output to gm
                            gm_max_burst_len.set_as(fmap_tail_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[nc1_num * self.output_wh * self.c0_size + cut_w_index *
                                                   fmap_cut_w * self.c0_size],
                                data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub, fmap_cut_h, fmap_tail_w, fmap_cut_w)
                        with self.tik_instance.else_scope():
                            # move max output to gm
                            gm_max_burst_len.set_as(fmap_tail_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[nc1_num * self.output_wh * self.c0_size +
                                                   cut_w_index * fmap_cut_w * self.c0_size],
                                fmap_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, fmap_cut_w_num)
                        mask_cut_w = fmap_tail_w_num * 16 - fmap_tail_w
                        self.pow2(mask_cut_w, mask_zero_w)
                        mask_zero_w.set_as(2 ** 16 - mask_zero_w)
                        with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + \
                                cur_h_idx * fmap_cut_h + cut_w_index * fmap_cut_w
                            with self.tik_instance.if_scope(tik.all(mask_zero_w != 0,
                                                                    self.fmap_w != 1, cut_h_num == 1)):
                                self.tik_instance.vector_dup(
                                    [zero_w, mask_zero_w], mask_ub[w_index * fmap_cut_w_num * self.c0_size +
                                                                   fmap_tail_w_num * 16 - 16], 0, 1, 1, 8)

                            self.tik_instance.data_move(
                                self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                                mask_ub[w_index * fmap_cut_w_num * self.c0_size], 0, 1, fmap_tail_w_num, 0, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, fmap_cut_w_num) as h_index:
                        source_h = -self.pad[2]
                        source_w = ((((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_cut_w) *
                                    self.stride_w - self.pad[0])
                        self.scalar_source_h.set_as(source_h)
                        self.scalar_source_w.set_as(source_w)
                        self.tik_instance.load3dv1(
                            fmap_ub[h_index * 256 * self.fmap_w], input_fmap_l1[0],
                            (self.pad[0], self.pad[1], self.pad[2], 0), (cut_h_size - self.pad[2]),
                            self.input_w, 0, 0, 0, self.scalar_source_w, self.scalar_source_h,
                            self.stride_w, self.stride_h, self.kernel_w, self.kernel_h, self.dilation_w,
                            self.dilation_h, 1, 0, self.fmap_w, 0, self.pad_value)
                    with self.tik_instance.if_scope(self.fmap_w != 1):
                        self._calc_max_and_mask(fmap_cut_w_num, fmap_ub, data_x_max, mask_ub)
                        # move max output to gm
                        gm_max_burst_len.set_as(fmap_cut_w)
                        self.tik_instance.data_move(
                            self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size],
                            data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                        self._remove_repeated_fun(mask_ub, fmap_cut_h, fmap_cut_w)
                    with self.tik_instance.else_scope():
                        # move max output to gm
                        gm_max_burst_len.set_as(fmap_cut_w)
                        self.tik_instance.data_move(
                            self.output_max_gm[nc1_num * self.output_h * self.output_w * self.c0_size],
                            fmap_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                        self._dup_mask_fun(mask_ub, fmap_cut_w_num)
                    with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                        offset_output_mask = \
                            nc1_num * (self.fmap_h_num + 1) * self.fmap_w * self.c0_size + cur_h_idx * fmap_cut_h
                        self.tik_instance.data_move(
                            self.output_mask_gm[offset_output_mask + w_index * (self.fmap_h_num + 1) * 16],
                            mask_ub[w_index * fmap_cut_w_num * self.c0_size], 0, 1, fmap_cut_w_num, 0, 0)

    def _fun_need_cut_h_w(self, block_index, nc1_cuth_index, cut_h_size,
                          cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h):
        """
        funtion need cut H and W while l1 not enough

        Parameters
        ----------
        block_index: index of block
        nc1_cuth_index: index of nc1
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        nc1_cuth_size: cut height size
        flag_cut_h:int

        Returns
        -------
        none
        """
        with self.tik_instance.if_scope(flag_cut_h == 1):
            cur_h_idx = (block_index * nc1_cuth_size + nc1_cuth_index) % cut_h_num
            nc1_num = (block_index * nc1_cuth_size + nc1_cuth_index) // cut_h_num
            self._calc_need_cut_h_w(nc1_num, cut_h_size, cut_h_num, cur_h_idx, cut_stride)
        with self.tik_instance.else_scope():
            nc1_num = block_index * nc1_cuth_size + nc1_cuth_index
            with self.tik_instance.for_range(0, cut_h_num) as cur_h_idx:
                self._calc_need_cut_h_w(nc1_num, cut_h_size, cut_h_num, cur_h_idx, cut_stride)

    def _calc_out_size_and_pad(self):
        """
        caculate output size and padding size
        -------
        pad: include pad_t, pad_b, pad_l, pad_r
        """

        if self.ceil_mode is False:
            pad_t = self.pad_h
            pad_b = self.pad_h
            pad_l = self.pad_w
            pad_r = self.pad_w
        else:
            pad_t = self.pad_h
            pad_b = self.pad_h + self.stride_h - 1
            pad_l = self.pad_w
            pad_r = self.pad_w + self.stride_w - 1

        pad = (pad_l, pad_r, pad_t, pad_b)

        return pad

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
        self.tik_instance.vmax(
            Constant.MASK, data_input[index_h * 256], data_input[index_h * 256],
            data_input_ub[index_w * 256 + index_h * self.fmap_w * 256],
            Constant.REPEAT_2, 1, 1, 1, Constant.DSTSTRIDEM1, Constant.SRC0STRIDEM1, Constant.SRC1STRIDEM1)
        return data_input

    def _calc_max_fun_binary_search(self, data_input_ub, length):
        """
        calculate max of data_input by binary search algorithm

        Parameters
        ----------
        data_input_ub: input data in ub
        length: tensor's length

        Returns
        -------
        """
        half_time = self.tik_instance.Scalar("int32", name="half_time")
        half_time.set_as(length)
        with self.tik_instance.for_range(0, length) as loop:
            with self.tik_instance.if_scope(half_time == 1):
                pass
            with self.tik_instance.else_scope():
                half_time.set_as(half_time // 2)
                repeat_time = half_time * 256 // 128
                res_len = self.tik_instance.Scalar("int32", name="len", init_value=0)
                with self.tik_instance.if_scope(length % 2 > 0):
                    res_len.set_as(1)

                with self.tik_instance.if_scope(repeat_time > 255):
                    input_ub_0 = data_input_ub[0]
                    input_ub_1 = data_input_ub[half_time * 256]
                    iter_times = repeat_time // 255
                    iter_len = 255 * 128
                    res_iter_times = repeat_time - iter_times
                    with self.tik_instance.for_range(0, iter_times) as iter_i:
                        self.tik_instance.vmax(
                            Constant.MASK, input_ub_0[iter_i * iter_len], input_ub_0[iter_i * iter_len],
                            input_ub_1[iter_i * iter_len],
                            255, 1, 1, 1, Constant.DSTSTRIDEM1, Constant.SRC0STRIDEM1, Constant.SRC1STRIDEM1)
                    self.tik_instance.vmax(
                        Constant.MASK, input_ub_0[iter_times * iter_len], input_ub_0[iter_times * iter_len],
                        input_ub_1[iter_times * iter_len],
                        res_iter_times, 1, 1, 1, Constant.DSTSTRIDEM1, Constant.SRC0STRIDEM1, Constant.SRC1STRIDEM1
                    )

                with self.tik_instance.else_scope():
                    self.tik_instance.vmax(
                        Constant.MASK, data_input_ub[0], data_input_ub[0], data_input_ub[half_time * 256],
                        repeat_time, 1, 1, 1, Constant.DSTSTRIDEM1, Constant.SRC0STRIDEM1, Constant.SRC1STRIDEM1)

                with self.tik_instance.if_scope(res_len > 0):
                    self.tik_instance.vmax(
                        Constant.MASK, data_input_ub[0], data_input_ub[0], data_input_ub[half_time * 2 * 256],
                        Constant.REPEAT_2, 1, 1, 1, Constant.DSTSTRIDEM1, Constant.SRC0STRIDEM1, Constant.SRC1STRIDEM1)

    def _calc_mask_fun(self, data_input_max, data_input_ub, index_w, index_h, fmap_h_num, mask_ub):
        """
        caculate mask of data_input_max

        Parameters
        ----------
        data_input_max: max value in input data
        data_input_ub: input data in ub
        index_w: index of w, along to kernel, 3x3
        index_h: index of h, alogn to output, 6x6
        fmap_h_num: num of fmap in h
        mask_ub: mask in ub, 3 x 3 x 3 x 16

        Returns
        -------
        mask_ub: mask in ub
        """
        self.tik_instance.vcmpv_eq(mask_ub[index_w * fmap_h_num * 16 + index_h * 16],
                                   data_input_ub[index_w * 256 + index_h * self.fmap_w * 256],
                                   data_input_max[index_h * 256], Constant.REPEAT_2,
                                   1, 1, Constant.SRC0STRIDEM1, Constant.SRC1STRIDEM1)
        return mask_ub

    def _calc_max_and_mask(self, fmap_h_num, fmap_ub, data_x_max, mask_ub, fmap_cut_w_num=0, fmap_h_tail_num=0):
        """
        caculate max and mask of data_input

        Parameters
        ----------
        exampel for: 1x12x12x16
        fmap_h_num: num of fmap_h, 3
        fmap_ub: fmap in ub, 48x3x3x16
        data_x_max: max value in input data, 48x16
        mask_ub: mask in ub, 3x3x3x16
        fmap_cut_w_num: cut number of w, default as 0
        fmap_h_tail_num: num of h tail, default as 0

        Returns
        -------
        data_input_ub: output tensor
        """
        scalar_repeat_times = self.tik_instance.Scalar("int32")
        repeat_times = self.tik_instance.Scalar("int32")
        scalar_repeat_times.set_as(fmap_h_num * 2)
        repeat_times.set_as((scalar_repeat_times + 253) // 254)
        # dup output max with a given value
        with self.tik_instance.if_scope(scalar_repeat_times > 255):
            with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                with self.tik_instance.if_scope(repeat_index != (repeat_times - 1)):
                    self.tik_instance.vector_dup(
                        Constant.MASK, data_x_max[repeat_index * 254 * 128], Constant.MIN_VALUE_FP16, 254, 1,
                        Constant.SRC0STRIDEM1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(
                        Constant.MASK, data_x_max[repeat_index * 254 * 128], Constant.MIN_VALUE_FP16,
                        (scalar_repeat_times - repeat_index * 254), 1, Constant.SRC0STRIDEM1)
        with self.tik_instance.else_scope():
            self.tik_instance.vector_dup(Constant.MASK, data_x_max, Constant.MIN_VALUE_FP16, scalar_repeat_times, 1,
                                         Constant.SRC0STRIDEM1)
        with self.tik_instance.new_stmt_scope():
            feature_map_l1 = self.tik_instance.Tensor(
                self.input_dtype, (256 * 256,), name="feature_map_l1", scope=tik.scope_cbuf)

            self.tik_instance.data_move(feature_map_l1, fmap_ub, 0, 1, fmap_h_num * self.fmap_w * 16, 0, 0)
            with self.tik_instance.for_range(0, fmap_h_num) as index_h:  # 2
                # the first 128
                feature_map_w = self.fmap_w
                self._calc_max_fun_binary_search(fmap_ub[index_h * self.fmap_w * 256:], feature_map_w)
                self.tik_instance.data_move(data_x_max[index_h * 256], fmap_ub[index_h * self.fmap_w * 256],
                                            0, 1, 256 // 16, 0, 0)
            self.tik_instance.data_move(fmap_ub, feature_map_l1, 0, 1, fmap_h_num * self.fmap_w * 16, 0, 0)

        # calc mask indices
        with self.tik_instance.for_range(0, self.fmap_w) as index_w:
            with self.tik_instance.for_range(0, fmap_h_num) as index_h:
                with self.tik_instance.if_scope(fmap_cut_w_num == 0):
                    with self.tik_instance.if_scope(fmap_h_tail_num == 0):
                        mask_ub = self._calc_mask_fun(data_x_max, fmap_ub, index_w, index_h, fmap_h_num, mask_ub)
                    with self.tik_instance.else_scope():
                        mask_ub = self._calc_mask_fun(data_x_max, fmap_ub, index_w, index_h, fmap_h_tail_num, mask_ub)
                with self.tik_instance.else_scope():
                    mask_ub = self._calc_mask_fun(data_x_max, fmap_ub, index_w, index_h, fmap_cut_w_num, mask_ub)

    def _remove_repeated_fun(self, mask_ub, fmap_cut_h=0, fmap_cut_w=0, fmap_tail_w=0, fmap_tail_h=0):
        """
        caculate max and mask of data_input

        Parameters
        ----------
        mask_ub: mask in ub
        fmap_cut_h: size of fmap_cut_h
        fmap_cut_w: size of fmap_cut_w
        fmap_tail_w: size of fmap_tail_w
        fmap_tail_h: size of tail_h

        Returns
        -------
        data_input_ub: output tensor
        """
        fmap_h_num = self.tik_instance.Scalar("int32", name="h_num")
        fmap_tail_num = self.tik_instance.Scalar("int32", name="tail_num")

        with self.tik_instance.if_scope(fmap_cut_h != 0):
            with self.tik_instance.if_scope(fmap_cut_w != 0):
                fmap_h_num.set_as((fmap_cut_w + 15) // 16)
            with self.tik_instance.else_scope():
                fmap_h_num.set_as((fmap_cut_h + 15) // 16)
        with self.tik_instance.else_scope():
            fmap_h_num.set_as((self.fmap_h + 15) // 16)

        mask_or_shape_ub = (240, 16)
        mask_or = self.tik_instance.Tensor("uint16", mask_or_shape_ub, name="mask_or", scope=tik.scope_ubuf)
        mask_not = self.tik_instance.Tensor("uint16", mask_or_shape_ub, name="mask_not", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.fmap_w) as index_w:
            with self.tik_instance.if_scope(index_w > 0):
                with self.tik_instance.if_scope(fmap_tail_w == 0):
                    with self.tik_instance.if_scope(fmap_tail_h == 0):
                        self.tik_instance.vor(
                            16, mask_or[0], mask_ub[index_w * fmap_h_num * 16], mask_or[0], fmap_h_num, 1,
                            1, 1, 1, 1, 1)
                        self.tik_instance.vand(
                            16, mask_ub[index_w * fmap_h_num * 16], mask_not[0], mask_ub[index_w * fmap_h_num * 16],
                            fmap_h_num, 1, 1, 1, 1, 1,
                            1)
                    with self.tik_instance.else_scope():
                        fmap_tail_num.set_as((fmap_tail_h + 15) // 16)
                        self.tik_instance.vor(
                            16, mask_or[0], mask_ub[index_w * fmap_tail_num * 16], mask_or[0], fmap_h_num, 1,
                            1, 1, 1, 1, 1)
                        self.tik_instance.vand(
                            16, mask_ub[index_w * fmap_tail_num * 16], mask_not[0],
                            mask_ub[index_w * fmap_tail_num * 16], fmap_h_num, 1, 1,
                            1, 1, 1, 1)
                with self.tik_instance.else_scope():
                    fmap_tail_num.set_as((fmap_tail_w + 15) // 16)
                    self.tik_instance.vor(
                        16, mask_or[0], mask_ub[index_w * fmap_tail_num * 16], mask_or[0], fmap_h_num, 1,
                        1, 1, 1, 1, 1)
                    self.tik_instance.vand(
                        16, mask_ub[index_w * fmap_tail_num * 16], mask_not[0], mask_ub[index_w * fmap_tail_num * 16],
                        fmap_h_num, 1, 1, 1, 1, 1, 1)
                self.tik_instance.vnot(16, mask_not[0], mask_or[0], fmap_h_num, 1,
                                       1, 1, 1)
            with self.tik_instance.else_scope():
                self.tik_instance.vnot(16, mask_not[0], mask_ub[0], fmap_h_num, 1,
                                       1, 1, 1)
                self.tik_instance.data_move(mask_or[0], mask_ub[0], 0, 1, fmap_h_num, 0, 0)

    def _dup_mask_fun(self, mask_ub, scalar_repeat_times):
        """
         caculate max and mask of data_input

         Parameters
         ----------
         mask_ub: mask in ub
         mask_shape_ub: shape of mask_ub

         Returns
         -------
         none
         """
        repeat_times = self.tik_instance.Scalar("int32", name="repeat_times")
        repeat_times.set_as((scalar_repeat_times + 239) // 240)
        # dup 8*blocks init 1 into a buffer:
        with self.tik_instance.if_scope(scalar_repeat_times > 240):
            with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                with self.tik_instance.if_scope(repeat_index != (repeat_times - 1)):
                    self.tik_instance.vector_dup(
                        Constant.MASK, mask_ub[repeat_index * 240 * 16], 65535, 30, 1, Constant.SRC0STRIDEM1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(
                        16, mask_ub[repeat_index * 240 * 16], 65535, (scalar_repeat_times - repeat_index * 240),
                        1, 1)
        with self.tik_instance.else_scope():
            self.tik_instance.vector_dup(16, mask_ub, 65535, scalar_repeat_times, 1, 1)

    def pow2(self, scalar0, scalar1):
        """
        impl of pow
        :param scalar0:
        :param scalar1:
        :return:
        """
        with self.tik_instance.for_range(0, 16 - scalar0) as idx:
            scalar1.set_as(scalar1 * 2)


# 'pylint: disable=unused-argument
@register_operator("MaxPoolWithArgmaxV1")
def max_pool_with_argmax_v1(x, y, argmax, ksize, strides, pads, dtype=Constant.DT_INT32, dilation=(1, 1, 1, 1),
                            ceil_mode=False, kernel_name="max_pool_with_argmax_v1"):
    """
    implementation of max_pool_with_argmax for pytorch and return the \
    tik instance
    :param x: dict of shape and dtype of the input x
    :param y: dict of shape and dtype of the output y
    :param argmax: dict of shape and dtype of the output argmax
    :param ksize: the size of the window to take a max over
    :param strides: the stride of the window
    :param pads: implicit zero padding to be added on both sides
    :param dilation: a parameter that controls the stride of elements \
                     in the window
    :param ceil_mode: when True, will use ceil instead of floor to compute \
                      the output shape
    :param dtype: input data type, only support int32 or int64
    :param kernel_name: the kernel's name
    :return: tik_instance
    """
    _check_param(x, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
    max_pool_grad = MaxPoolWithargmaxPytorch(x, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
    return max_pool_grad.compute(kernel_name)
