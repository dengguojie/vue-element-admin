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
deformable_offsets
"""
from functools import reduce as functools_reduce
from impl import common_util
from impl import constant_util as constant
from impl.util import util_tik_comm_func
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    BLOCK_BYTES_SIZE = 32
    VECTOR_BYTES_SIZE = 256
    # const parameters used for float16 and uint16
    FP16_SIZE = 2
    FP16_MASK = 128
    FP16_RATIO = 1
    # const parameters used for float32 and int32
    FP32_SIZE = 4
    FP32_MASK = 64
    FP32_RATIO = 2
    BLOCK_SIZE = 32
    # Mask size for clip operation.
    CAL_MASK_SIZE = 64
    F16_CAL_MASK_SIZE = 128


def _ceil_div(value, block):
    """
    upper division
    """
    return (value + block - 1) // block


def _ceil_align(value, block):
    """
    upper align
    """
    return _ceil_div(value, block) * block


def get_params(dtype):
    """
    Get params according to given dtype.

    Parameters:
    ----------
    dtype, data type.

    Returns:
    -------
    None.
    """
    if dtype in ("float16", "uint16"):
        size = Constant.FP16_SIZE
        mask = Constant.FP16_MASK
        ratio = Constant.FP16_RATIO
    elif dtype in ("float32", "int32"):
        size = Constant.FP32_SIZE
        mask = Constant.FP32_MASK
        ratio = Constant.FP32_RATIO
    return size, mask, ratio


# 'pylint: disable=unused-argument,unused-variable
# 'pylint: disable=too-many-arguments,too-many-locals,too-many-return-statements
def check_supported(x,
                    offsets,
                    helper,
                    y,
                    strides,
                    pads,
                    ksize,
                    dialations=(1, 1, 1, 1),
                    data_format="NHWC",
                    deformable_groups=1,
                    modulated=True,
                    kernel_name="deformable_offsets"):
    """
    check whether ai_core is supported
    """
    check_list = ["float32", "float16"]
    format_list = ["NCHW", "NHWC"]
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    x_format = x.get("format")
    if not modulated:
        reason = "modulated is False"
        return False, reason
    if x_dtype not in check_list:
        reason = "dtype is x is not supported, x_dtype is %s,supported list is %s" % (x_dtype, check_list)
        return False, reason
    if len(x_shape) != 4:
        reason = "len of x_shape is not 4, x_shape is %s" % (str(x_shape),)
        return False, reason
    if (x_format not in format_list) or (data_format not in format_list):
        reason = "x_format is not [NHWC, NCHW] or data_format is not [NHWC, NCHW]"
        return False, reason
    if x_shape[3] % deformable_groups != 0:
        reason = "x_shape[3][%s] is not a multiple of deformable_groups[%s] != 0 " % (
            str(x_shape[3]), str(deformable_groups))
        return False, reason
    group_c = x_shape[3] // deformable_groups
    if group_c % 8 != 0 or deformable_groups != 1:
        reason = "group_c[%s] is not multiple of 8, or deformable_groups is not 1" % (str(group_c),)
        return False, reason
    return True, ""


# 'pylint: disable=too-many-instance-attributes,too-many-arguments
class DeformableOffsets(object):
    """
    initialize some properties
    """
    # 'pylint: disable=too-many-statements
    def __init__(self, x, offsets, helper, y, strides, pads, ksize,
                 data_format="NHWC", deformable_groups=1, modulated=True,
                 kernel_name="deformable_offsets"):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.x_shape = x.get("shape")
        self.x_dtype = x.get("dtype").lower()
        self.offsets_shape = offsets.get("shape")
        self.offsets_dtype = offsets.get("dtype").lower()
        self.y_shape = y.get("shape")
        self.y_dtype = y.get("dtype").lower()
        self.helper_shape = helper.get("shape")
        self.helper_dtype = helper.get("dtype").lower()
        self.kernel_name = kernel_name
        self.strides = strides
        self.pads = pads
        self.data_format = data_format
        self.deformable_groups = deformable_groups
        self.check_param()

        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.ai_core_num = tik.Dprofile().get_aicore_num()
        self.dtype_bytes_size = common_util.get_data_size(self.x_dtype)
        self.data_in_one_block = Constant.BLOCK_BYTES_SIZE // self.dtype_bytes_size
        self.data_in_one_vector = Constant.VECTOR_BYTES_SIZE // self.dtype_bytes_size
        max_ub_elem = (self.total_ub - 21480) // self.dtype_bytes_size
        self.dim_offsets_n = self.offsets_shape[0]
        self.dim_offsets_h = self.offsets_shape[1]
        self.dim_offsets_w = self.offsets_shape[2]
        self.dim_kh = ksize[0]
        self.dim_kw = ksize[1]
        self.dim_group = self.offsets_shape[3] // 3 // ksize[0] // ksize[1]
        self.dim_h_in = self.x_shape[1]
        self.dim_w_in = self.x_shape[2]
        self.dim_c = self.x_shape[3]
        self.dim_group_c = self.dim_c // self.dim_group
        self.elem_num_offsets_filter = self.dim_group * self.dim_kh * self.dim_kw * 3
        self.elem_num_aligned = _ceil_align(self.elem_num_offsets_filter, self.data_in_one_block)
        self.x_len = functools_reduce(lambda x_ele, y_ele: x_ele * y_ele, self.x_shape)
        self.offsets_len = functools_reduce(lambda x_ele, y_ele: x_ele * y_ele, self.offsets_shape)
        self.helper_len = self.offsets_len // self.dim_offsets_n
        self.y_len = self.offsets_len * self.dim_group_c // 3
        self.thread_num = 1
        self.x_gm = self.tik_instance.Tensor(self.x_dtype, [self.x_len, ], name="x_gm", scope=tbe_platform.scope_gm)
        self.offsets_gm = self.tik_instance.Tensor(
            self.offsets_dtype, [self.offsets_len, ], name="offsets_gm", scope=tbe_platform.scope_gm)
        self.helper_gm = self.tik_instance.Tensor(
            self.helper_dtype, [self.helper_len, ], name="helper_gm", scope=tbe_platform.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.y_dtype, [self.y_len], name="y_gm", scope=tbe_platform.scope_gm)
        self.scalar_const_pos1 = self.tik_instance.Scalar(dtype=self.x_dtype, name="const_pos1", init_value=1.0)
        self.ub_seg_size = max_ub_elem // (self.dim_c * self.dim_kw) // self.thread_num
        self.loop_seg = self.dim_offsets_w // self.ub_seg_size
        self.ub_seg_res = self.dim_offsets_w % self.ub_seg_size
        self.cmp_flag = tbe_platform.api_check_support("tik.vec_cmpv_ge", dtype="float32")
        self.conv_num = min(self.dim_group_c, Constant.CAL_MASK_SIZE)        

        self.ub_out = None
        self.ub_offsets_ori = None
        self.ub_offsets_floor = None
        self.ub_offsets_ceil = None
        self.ub_offsets_floor_f16 = None
        self.ub_offsets_ceil_f16 = None
        self.ub_offsets_int32_ceil = None
        self.ub_offsets_int32_floor = None
        self.ub_offsets_sub_floor = None
        self.ub_offsets_ceil_sub = None
        self.ub_helper = None
        self.ub_limit_h = None
        self.ub_limit_w = None
        self.ub_limit_0 = None
        self.sel_mask1 = None
        self.sel_mask2 = None
        self.sel_mask_ceil_x = None
        self.sel_mask_ceil_y = None
        self.sel_mask_floor_x = None
        self.sel_mask_floor_y = None

    def check_param(self):
        """
        Check if the shape and dtype of input be right.

        Parameters:
        ----------
        None.(Get from class member.)

        Returns:
        -------
        None.
        Error will report when exception happened.
        """
        check_list = ["float32", "float16"]
        para_check.check_shape(self.x_shape, min_rank=4, max_rank=4, param_name="x")
        para_check.check_shape(self.offsets_shape, min_rank=4, max_rank=4, param_name="offsets")
        para_check.check_dtype(self.x_dtype, check_list, param_name="x")
        if len(self.strides) != 4:
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, "the length of strides should be 4",
                                                              "strides", self.strides)
        if len(self.pads) != 4:
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, "the length of pads should be 4",
                                                              "pads", self.pads)

    def vector_dup(self, ub_to_dup, process_num, const_val, dtype="float32"):
        """
        Duplicate data to a tensor.

        Parameters:
        ----------
        ub_to_dup: Dst tensor to store the result.
        process_num: The length to operate.
        const_val: A scalar value.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in ub_to_dup.
        """
        size, mask, _ = get_params(dtype)
        repeat_times = process_num // mask
        if repeat_times != 0:
            self.tik_instance.vec_dup(mask, ub_to_dup[0],
                                      const_val, repeat_times, 8)
        tail_elem = process_num % mask
        if tail_elem != 0:
            self.tik_instance.vec_dup(tail_elem, ub_to_dup[repeat_times * mask],
                                      const_val, 1, tail_elem // (Constant.BLOCK_SIZE // size))

    def vector_adds(self, dst, src, const, process_num, dtype="float32"):
        """
        Vector operator, Add a scalar value with a tensor.

        Parameters:
        ----------
        dst: Dst tensor to store the result.
        process_num: The length to operate.
        src: src tensor.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in dst.
        """
        size, mask, _ = get_params(dtype)
        repeat_times = process_num // mask
        if repeat_times != 0:
            self.tik_instance.vadds(mask, dst[0], src[0], const,
                                    repeat_times, 1, 1, 8, 8)
        tail_elem = process_num % mask
        if tail_elem != 0:
            self.tik_instance.vadds(tail_elem,
                                    dst[mask * repeat_times],
                                    src[mask * repeat_times],
                                    const,
                                    1, 1, 1,
                                    tail_elem // (Constant.BLOCK_SIZE // size),
                                    tail_elem // (Constant.BLOCK_SIZE // size))

    def vector_add(self, dst, src1, src2, process_num, dtype="float32"):
        """
        Vector operator, Add two tensors elemwise.

        Parameters:
        ----------
        dst: Dst tensor to store the result.
        process_num: The length to operate.
        src1 & src2: src tensor.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in dst.
        """
        size, mask, _ = get_params(dtype)
        repeat_times = process_num // mask
        if repeat_times != 0:
            self.tik_instance.vadd(mask, dst[0], src1[0], src2[0],
                                   repeat_times, 1, 1, 1, 8, 8, 8)
        tail_elem = process_num % mask
        if tail_elem != 0:
            self.tik_instance.vadd(tail_elem,
                                   dst[mask * repeat_times],
                                   src1[mask * repeat_times],
                                   src2[mask * repeat_times],
                                   1, 1, 1, 1,
                                   tail_elem // (Constant.BLOCK_SIZE // size),
                                   tail_elem // (Constant.BLOCK_SIZE // size),
                                   tail_elem // (Constant.BLOCK_SIZE // size))

    def vector_sub(self, dst, src1, src2, process_num, dtype="float32"):
        """
        Vector operator, Sub one tensors by another elemwise.

        Parameters:
        ----------
        dst: Dst tensor to store the result.
        process_num: The length to operate.
        src1 & src2: src tensor.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in dst.
        """
        size, mask, _ = get_params(dtype)
        repeat_times = process_num // mask
        if repeat_times != 0:
            self.tik_instance.vsub(mask, dst[0], src1[0], src2[0],
                                   repeat_times, 1, 1, 1, 8, 8, 8)
        tail_elem = process_num % mask
        if tail_elem != 0:
            self.tik_instance.vsub(tail_elem,
                                   dst[mask * repeat_times],
                                   src1[mask * repeat_times],
                                   src2[mask * repeat_times],
                                   1, 1, 1, 1,
                                   tail_elem // (Constant.BLOCK_SIZE // size),
                                   tail_elem // (Constant.BLOCK_SIZE // size),
                                   tail_elem // (Constant.BLOCK_SIZE // size))

    def vector_mul(self, dst, src1, src2, process_num, dtype="float32"):
        """
        Vector operator, Mul one tensors by another elemwise.

        Parameters:
        ----------
        dst: Dst tensor to store the result.
        process_num: The length to operate.
        src1 & src2: src tensor.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in dst.
        """
        size, mask, _ = get_params(dtype)
        repeat_times = process_num // mask
        if repeat_times != 0:
            self.tik_instance.vmul(mask, dst[0], src1[0], src2[0],
                                   repeat_times, 1, 1, 1, 8, 8, 8)
        tail_elem = process_num % mask
        if tail_elem != 0:
            self.tik_instance.vmul(tail_elem,
                                   dst[mask * repeat_times],
                                   src1[mask * repeat_times],
                                   src2[mask * repeat_times],
                                   1, 1, 1, 1,
                                   tail_elem // (Constant.BLOCK_SIZE // size),
                                   tail_elem // (Constant.BLOCK_SIZE // size),
                                   tail_elem // (Constant.BLOCK_SIZE // size))

    def vector_muls(self, dst, src, const, process_num, dtype="float32"):
        """
        Vector operator, Mul each elem in src tensor with a scalar value const.

        Parameters:
        ----------
        dst: Dst tensor to store the result.
        const: Scalar value.
        process_num: The length to operate.
        src: src tensor.
        dtype: Data type.

        Returns:
        -------
        None.
        The result will be stored in dst.
        """
        size, mask, _ = get_params(dtype)
        repeat_times = process_num // mask
        if repeat_times != 0:
            self.tik_instance.vmuls(mask, dst[0], src[0], const,
                                    repeat_times, 1, 1, 8, 8)
        tail_elem = process_num % mask
        if tail_elem != 0:
            self.tik_instance.vmuls(tail_elem,
                                    dst[mask * repeat_times],
                                    src[mask * repeat_times],
                                    const,
                                    1, 1, 1,
                                    tail_elem // (Constant.BLOCK_SIZE // size),
                                    tail_elem // (Constant.BLOCK_SIZE // size))

    def vector_conv_int322fp16(self, dst_ub_fp16, src_ub_int, process_num):
        """
        Vector operator, convert int32 to fp16, when not support fp32.
        """
        size, mask, _ = get_params("int32")
        repeat_times = process_num // mask
        if repeat_times != 0:
            self.tik_instance.vec_conv(mask, '', dst_ub_fp16[0], src_ub_int[0],
                                       repeat_times, 4, 8, deqscale=1.0)
        tail_elem = process_num % mask
        if tail_elem != 0:
            self.tik_instance.vec_conv(tail_elem, '',
                                       dst_ub_fp16[mask * repeat_times],
                                       src_ub_int[mask * repeat_times],
                                       1, 4, 8, deqscale=1.0)

    def clip_tensor(self, ub_x_tensor, ub_y_tensor, process_num, dtype="float32"):
        """
        To avoid of overflow, Clip the offset tensor.
        -------
        Parameters:
        ----------
        ub_x_tensor: Will be clipped in the range of [0, W_IN - 1].
        ub_y_tensor: Will be clipped in the range of [0, H_IN - 1].
        process_num: The length to operate.
        """
        size, mask, _ = get_params(dtype)
        repeat_times = process_num // mask
        tail_elem = process_num % mask
        if repeat_times != 0:
            self.tik_instance.vmax(mask,
                                   ub_y_tensor[0],
                                   ub_y_tensor[0],
                                   self.ub_limit_0,
                                   repeat_times,
                                   1, 1, 1, 8, 8, 0)
            self.tik_instance.vmin(mask,
                                   ub_y_tensor[0],
                                   ub_y_tensor[0],
                                   self.ub_limit_h,
                                   repeat_times,
                                   1, 1, 1, 8, 8, 0)

            self.tik_instance.vmax(mask,
                                   ub_x_tensor[0],
                                   ub_x_tensor[0],
                                   self.ub_limit_0,
                                   repeat_times,
                                   1, 1, 1, 8, 8, 0)
            self.tik_instance.vmin(mask,
                                   ub_x_tensor[0],
                                   ub_x_tensor[0],
                                   self.ub_limit_w,
                                   repeat_times,
                                   1, 1, 1, 8, 8, 0)
        if tail_elem != 0:
            self.tik_instance.vmax(tail_elem,
                                   ub_y_tensor[mask * repeat_times],
                                   ub_y_tensor[mask * repeat_times],
                                   self.ub_limit_0,
                                   1, 1, 1, 1, 8, 8, 0)
            self.tik_instance.vmin(tail_elem,
                                   ub_y_tensor[mask * repeat_times],
                                   ub_y_tensor[mask * repeat_times],
                                   self.ub_limit_h,
                                   1, 1, 1, 1, 8, 8, 0)

            self.tik_instance.vmax(tail_elem,
                                   ub_x_tensor[mask * repeat_times],
                                   ub_x_tensor[mask * repeat_times],
                                   self.ub_limit_0,
                                   1, 1, 1, 1, 8, 8, 0)
            self.tik_instance.vmin(tail_elem,
                                   ub_x_tensor[mask * repeat_times],
                                   ub_x_tensor[mask * repeat_times],
                                   self.ub_limit_w,
                                   1, 1, 1, 1, 8, 8, 0)

    # 'pylint: disable=too-many-locals
    def get_x(self, ub_x, offset_list, process_num, sel_mask):
        """
        Get input x values by given index.
        -------
        Parameters:
        ----------
        ub_x: Result will be stored in ub_x.
        offset_list: offsets index.
        process_num: The length to operate.
        sel_mask: Decide if ub_x is 0 or x.
                  when offset_list is valid, sel_mask will be all 1.
                  and if offset_list is invalid , sel_mask will be all 0.
        """
        dim_n = offset_list[0]
        dim_h = offset_list[1]
        dim_w = offset_list[2]
        dim_g = offset_list[3]
        size, mask, _ = get_params(self.x_dtype)
        repeat_times = process_num // mask
        tail_elem = process_num % mask
        x_offset = ((dim_n * self.dim_h_in + dim_h) * self.dim_w_in + \
                    dim_w * self.dim_group + dim_g) * self.dim_c
        burst_len = process_num // self.data_in_one_block
        self.tik_instance.data_move(ub_x[0],
                                    self.x_gm[x_offset],
                                    constant.SID,
                                    constant.DEFAULT_NBURST,
                                    burst_len,
                                    constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)
        plat_flag = tbe_platform.api_check_support("tik.vsel", dtype="float32")
        if (not plat_flag) and (self.x_dtype == "float32"):
            size, mask, _ = get_params("float16")
            repeat_times = process_num // mask
            tail_elem = process_num % mask
            ub_x_local = self.tik_instance.Tensor(
                "float16", (process_num,), scope=tbe_platform.scope_ubuf, name="ub_x_local")
            util_tik_comm_func.tik_func_vconv(
                self.tik_instance, ub_x_local, ub_x, process_num)
            if repeat_times != 0:
                cmp_mask = self.tik_instance.mov_tensor_to_cmpmask(sel_mask[0])
                self.tik_instance.vsel(mask, 0, ub_x_local[0], cmp_mask, ub_x_local[0],
                                       self.ub_limit_0,
                                       repeat_times, 1, 1, 1, 8, 8, 0)
            if tail_elem != 0:
                cmp_mask = self.tik_instance.mov_tensor_to_cmpmask(sel_mask[mask * repeat_times])
                self.tik_instance.vsel(tail_elem, 0,
                                       ub_x_local[mask * repeat_times],
                                       cmp_mask,
                                       ub_x_local[mask * repeat_times],
                                       self.ub_limit_0,
                                       1, 1, 1, 1, 8, 8, 0)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, ub_x, ub_x_local, process_num)
        else:
            if repeat_times != 0:
                cmp_mask = self.tik_instance.mov_tensor_to_cmpmask(sel_mask[0])
                self.tik_instance.vsel(mask, 0, ub_x[0], cmp_mask, ub_x[0],
                                       self.ub_limit_0,
                                       repeat_times, 1, 1, 1, 8, 8, 0)
            if tail_elem != 0:
                cmp_mask = self.tik_instance.mov_tensor_to_cmpmask(sel_mask[mask * repeat_times])
                self.tik_instance.vsel(tail_elem, 0,
                                       ub_x[mask * repeat_times],
                                       cmp_mask,
                                       ub_x[mask * repeat_times],
                                       self.ub_limit_0,
                                       1, 1, 1, 1, 8, 8, 0)

    def alloc_global_ub(self):
        """
        Allocate some global ub tensors.
        """
        vmax_support = tbe_platform.api_check_support("tik.vmax", dtype="float32")
        vsel_support = tbe_platform.api_check_support("tik.vsel", dtype="float32")
        if (vsel_support and vmax_support) and (self.x_dtype == "float32"):
            self.ub_limit_h = self.tik_instance.Tensor(
                "float32", (Constant.CAL_MASK_SIZE,), name="ub_limit_h", scope=tbe_platform.scope_ubuf)
            self.ub_limit_w = self.tik_instance.Tensor(
                "float32", (Constant.CAL_MASK_SIZE,), name="ub_limit_w", scope=tbe_platform.scope_ubuf)
            self.ub_limit_0 = self.tik_instance.Tensor(
                "float32", (Constant.CAL_MASK_SIZE,), name="ub_limit_0", scope=tbe_platform.scope_ubuf)
            self.vector_dup(self.ub_limit_h, Constant.CAL_MASK_SIZE, self.dim_h_in - 1, dtype="float32")
            self.vector_dup(self.ub_limit_w, Constant.CAL_MASK_SIZE, self.dim_w_in - 1, dtype="float32")
            self.vector_dup(self.ub_limit_0, Constant.CAL_MASK_SIZE, 0, dtype="float32")
        else:
            self.ub_limit_h = self.tik_instance.Tensor(
                "float16", (Constant.F16_CAL_MASK_SIZE,), name="ub_limit_h", scope=tbe_platform.scope_ubuf)
            self.ub_limit_w = self.tik_instance.Tensor(
                "float16", (Constant.F16_CAL_MASK_SIZE,), name="ub_limit_w", scope=tbe_platform.scope_ubuf)
            self.ub_limit_0 = self.tik_instance.Tensor(
                "float16", (Constant.F16_CAL_MASK_SIZE,), name="ub_limit_0", scope=tbe_platform.scope_ubuf)
            self.vector_dup(self.ub_limit_h, Constant.F16_CAL_MASK_SIZE, self.dim_h_in - 1, dtype="float16")
            self.vector_dup(self.ub_limit_w, Constant.F16_CAL_MASK_SIZE, self.dim_w_in - 1, dtype="float16")
            self.vector_dup(self.ub_limit_0, Constant.F16_CAL_MASK_SIZE, 0, dtype="float16")
            self.ub_offsets_ceil_f16 = self.tik_instance.Tensor(
                "float16", (Constant.F16_CAL_MASK_SIZE,), name="ub_offsets_ceil_f16", scope=tbe_platform.scope_ubuf)
            self.ub_offsets_floor_f16 = self.tik_instance.Tensor(
                "float16", (Constant.F16_CAL_MASK_SIZE,), name="ub_offsets_floor_f16", scope=tbe_platform.scope_ubuf)
        self.sel_mask1 = self.tik_instance.Tensor(
            "uint16", (Constant.F16_CAL_MASK_SIZE,), name="sel_mask1", scope=tbe_platform.scope_ubuf)
        self.sel_mask2 = self.tik_instance.Tensor(
            "uint16", (Constant.F16_CAL_MASK_SIZE,), name="sel_mask2", scope=tbe_platform.scope_ubuf)
        self.sel_mask_ceil_x = self.tik_instance.Tensor(
            "uint16", (Constant.F16_CAL_MASK_SIZE,), name="sel_mask_ceil_x", scope=tbe_platform.scope_ubuf)
        self.sel_mask_ceil_y = self.tik_instance.Tensor(
            "uint16", (Constant.F16_CAL_MASK_SIZE,), name="sel_mask_ceil_y", scope=tbe_platform.scope_ubuf)
        self.sel_mask_floor_x = self.tik_instance.Tensor(
            "uint16", (Constant.F16_CAL_MASK_SIZE,), name="sel_mask_floor_x", scope=tbe_platform.scope_ubuf)
        self.sel_mask_floor_y = self.tik_instance.Tensor(
            "uint16", (Constant.F16_CAL_MASK_SIZE,), name="sel_mask_floor_y", scope=tbe_platform.scope_ubuf)

    def deformable_offset_compute(self):
        """
        The main comute func.
        """
        total_wc_num = self.dim_offsets_n * self.dim_offsets_h * self.dim_kh
        wc_per_core = _ceil_div(total_wc_num, self.ai_core_num)
        core_used = _ceil_div(total_wc_num, wc_per_core)
        last_wc_start = wc_per_core * (core_used - 1)
        wc_last = total_wc_num - last_wc_start
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_index:
            self.alloc_global_ub()
            th_n = self.thread_num
            with self.tik_instance.if_scope(core_index < (core_used - 1)):
                if wc_per_core < 2:
                    th_n = 1
                with self.tik_instance.for_range(0, wc_per_core, thread_num=th_n) as wc_index:
                    sw_start = core_index * wc_per_core + wc_index
                    offsets_w_start = sw_start // self.dim_kh
                    kh_idx = sw_start % self.dim_kh
                    self.compute_one_w(kh_idx, offsets_w_start)
            with self.tik_instance.else_scope():
                if wc_last < 2:
                    th_n = 1
                with self.tik_instance.for_range(0, wc_last, thread_num=th_n) as last_wc_index:
                    last_sw_start = (core_used - 1) * wc_per_core + last_wc_index
                    last_w_start = last_sw_start // self.dim_kh
                    kh_idx = last_sw_start % self.dim_kh
                    self.compute_one_w(kh_idx, last_w_start)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm, self.offsets_gm, self.helper_gm],
                                   outputs=[self.y_gm])

    def compute_one_w(self, kh_idx, offsets_w_start):
        """
        Compute one w dim unit.
        """
        out_gm_start = offsets_w_start * self.dim_kh + kh_idx
        helper_w_start = offsets_w_start % self.dim_offsets_h
        n_idx = offsets_w_start // self.dim_offsets_h
        self.ub_offsets_ori = self.tik_instance.Tensor(self.x_dtype,
                                                       (self.elem_num_aligned,),
                                                       scope=tbe_platform.scope_ubuf,
                                                       name="ub_offsets_ori")
        self.ub_offsets_ceil = self.tik_instance.Tensor(self.x_dtype,
                                                        (self.elem_num_aligned,),
                                                        scope=tbe_platform.scope_ubuf,
                                                        name="ub_offsets_ceil")
        self.ub_offsets_floor = self.tik_instance.Tensor(self.x_dtype,
                                                         (self.elem_num_aligned,),
                                                         scope=tbe_platform.scope_ubuf,
                                                         name="ub_offsets_floor")
        self.ub_offsets_sub_floor = self.tik_instance.Tensor(self.x_dtype,
                                                             (self.elem_num_aligned,),
                                                             scope=tbe_platform.scope_ubuf,
                                                             name="ub_offsets_sub_floor")
        self.ub_offsets_ceil_sub = self.tik_instance.Tensor(self.x_dtype,
                                                            (self.elem_num_aligned,),
                                                            scope=tbe_platform.scope_ubuf,
                                                            name="ub_offsets_ceil_sub")
        self.ub_offsets_int32_ceil = self.tik_instance.Tensor("int32",
                                                              (self.elem_num_aligned,),
                                                              scope=tbe_platform.scope_ubuf,
                                                              name="ub_offsets_int32_ceil")
        self.ub_offsets_int32_floor = self.tik_instance.Tensor("int32",
                                                               (self.elem_num_aligned,),
                                                               scope=tbe_platform.scope_ubuf,
                                                               name="ub_offsets_int32_floor")
        self.ub_helper = self.tik_instance.Tensor(self.x_dtype,
                                                  (self.elem_num_aligned,),
                                                  scope=tbe_platform.scope_ubuf,
                                                  name="ub_helper")
        self.ub_out = self.tik_instance.Tensor(self.x_dtype,
                                               (self.ub_seg_size * self.dim_kw * self.dim_c,),
                                               scope=tbe_platform.scope_cbuf,
                                               name="ub_out")
        if self.loop_seg != 0:
            with self.tik_instance.for_range(0, self.loop_seg) as ub_w_idx:
                offsets_f_start = offsets_w_start * self.dim_offsets_w + ub_w_idx * self.ub_seg_size
                helper_f_start = helper_w_start * self.dim_offsets_w + ub_w_idx * self.ub_seg_size
                with self.tik_instance.new_stmt_scope():
                    self.compute_one_filter(
                        n_idx, offsets_f_start, helper_f_start, kh_idx, self.ub_seg_size, self.ub_out)
                with self.tik_instance.new_stmt_scope():
                    out_gm_addr = (out_gm_start * self.dim_offsets_w + \
                                   ub_w_idx * self.ub_seg_size) * self.dim_c * self.dim_kw
                    bur_len = _ceil_div(self.ub_seg_size * self.dim_kw * self.dim_c, self.data_in_one_block)

                    ub_out_temp = self.tik_instance.Tensor(self.x_dtype,
                                                           (self.ub_seg_size * self.dim_kw * self.dim_c,),
                                                           scope=tbe_platform.scope_ubuf,
                                                           name="ub_out_temp")
                    self.tik_instance.data_move(ub_out_temp[0],
                                                self.ub_out[0],
                                                constant.SID,
                                                constant.DEFAULT_NBURST,
                                                bur_len,
                                                constant.STRIDE_ZERO,
                                                constant.STRIDE_ZERO)
                    self.tik_instance.data_move(self.y_gm[out_gm_addr],
                                                ub_out_temp[0],
                                                constant.SID,
                                                constant.DEFAULT_NBURST,
                                                bur_len,
                                                constant.STRIDE_ZERO,
                                                constant.STRIDE_ZERO)
        if self.ub_seg_res != 0:
            offsets_f_start = offsets_w_start * self.dim_offsets_w + self.loop_seg * self.ub_seg_size
            helper_f_start = helper_w_start * self.dim_offsets_w + self.loop_seg * self.ub_seg_size
            with self.tik_instance.new_stmt_scope():
                self.compute_one_filter(n_idx, offsets_f_start, helper_f_start, kh_idx, self.ub_seg_res, self.ub_out)
            with self.tik_instance.new_stmt_scope():
                out_gm_addr = (out_gm_start * self.dim_offsets_w + \
                               self.loop_seg * self.ub_seg_size) * self.dim_c * self.dim_kw
                bur_len = _ceil_div(self.ub_seg_res * self.dim_kw * self.dim_c, self.data_in_one_block)
                temp_ub_out = self.tik_instance.Tensor(self.x_dtype,
                                                       (self.ub_seg_size * self.dim_kw * self.dim_c,),
                                                       scope=tbe_platform.scope_ubuf,
                                                       name="temp_ub_out")
                self.tik_instance.data_move(temp_ub_out[0],
                                            self.ub_out[0],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            bur_len,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                self.tik_instance.data_move(self.y_gm[out_gm_addr],
                                            temp_ub_out[0],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            bur_len,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

    # 'pylint: disable=too-many-locals,too-many-statements
    def compute_one_filter(self, n_idx, f_start, hlp_start, kh_idx, out_c_num, ub_out):
        """
        Compute each w unit.
        """
        with self.tik_instance.for_range(0, out_c_num) as cur_w_index:
            offsets_1wc_start = (f_start + cur_w_index) * self.elem_num_offsets_filter
            helper_1wc_start = (hlp_start + cur_w_index) * self.elem_num_offsets_filter
            self.load_offsets(offsets_1wc_start, helper_1wc_start, self.elem_num_aligned)
            ub_offsets_x_ceil_int32 = self.tik_instance.Tensor(
                "int32", (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_offsets_x_ceil_int32")
            ub_offsets_y_ceil_int32 = self.tik_instance.Tensor(
                "int32", (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_offsets_y_ceil_int32")
            ub_offsets_x_floor_int32 = self.tik_instance.Tensor(
                "int32", (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_offsets_x_floor_int32")
            ub_offsets_y_floor_int32 = self.tik_instance.Tensor(
                "int32", (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_offsets_y_floor_int32")
            ub_weight_lt = self.tik_instance.Tensor(
                self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_weight_lt")
            ub_weight_lb = self.tik_instance.Tensor(
                self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_weight_lb")
            ub_weight_rt = self.tik_instance.Tensor(
                self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_weight_rt")
            ub_weight_rb = self.tik_instance.Tensor(
                self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_weight_rb")

            scalar_idx_lth_int = self.tik_instance.Scalar(dtype="int32")
            scalar_idx_rth_int = scalar_idx_lth_int
            scalar_idx_ltw_int = self.tik_instance.Scalar(dtype="int32")
            scalar_idx_lbw_int = scalar_idx_ltw_int
            scalar_idx_lbh_int = self.tik_instance.Scalar(dtype="int32")
            scalar_idx_rbh_int = scalar_idx_lbh_int
            scalar_idx_rtw_int = self.tik_instance.Scalar(dtype="int32")
            scalar_idx_rbw_int = scalar_idx_rtw_int

            ceil_sub_x = self.tik_instance.Tensor(
                self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ceil_sub_x")
            ceil_sub_y = self.tik_instance.Tensor(
                self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ceil_sub_y")
            sub_floor_x = self.tik_instance.Tensor(
                self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="sub_floor_x")
            sub_floor_y = self.tik_instance.Tensor(
                self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="sub_floor_y")
            scalar_tmp_f32 = self.tik_instance.Scalar(dtype=self.x_dtype)
            ub_offset_s = self.tik_instance.Tensor(
                self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_offset_s")

            with self.tik_instance.for_range(0, self.dim_kw) as kw_idx:
                out_ub_start = cur_w_index * self.dim_kw + kw_idx
                with self.tik_instance.for_range(0, self.dim_group) as group_idx:
                    scalar_tmp_f32.set_as(
                        self.ub_offsets_ori[(2 * self.dim_group + group_idx) * self.dim_kh * self.dim_kw + \
                                            kh_idx * self.dim_kw + kw_idx])
                    self.vector_dup(ub_offset_s, self.dim_group_c, scalar_tmp_f32, self.x_dtype)

                    if (self.x_dtype == "float16") or (not self.cmp_flag):
                        ub_offset_x_ceil_f16 = self.tik_instance.Tensor("float16",
                                                                        (Constant.F16_CAL_MASK_SIZE,),
                                                                        scope=tbe_platform.scope_ubuf,
                                                                        name="ub_offset_x_ceil_f16")
                        ub_offset_y_ceil_f16 = self.tik_instance.Tensor("float16",
                                                                        (Constant.F16_CAL_MASK_SIZE,),
                                                                        scope=tbe_platform.scope_ubuf,
                                                                        name="ub_offset_y_ceil_f16")
                        ub_offset_x_floor_f16 = self.tik_instance.Tensor("float16",
                                                                         (Constant.F16_CAL_MASK_SIZE,),
                                                                         scope=tbe_platform.scope_ubuf,
                                                                         name="ub_offset_x_floor_f16")
                        ub_offset_y_floor_f16 = self.tik_instance.Tensor("float16",
                                                                         (Constant.F16_CAL_MASK_SIZE,),
                                                                         scope=tbe_platform.scope_ubuf,
                                                                         name="ub_offset_y_floor_f16")
                        scalar_tmp_f16 = self.tik_instance.Scalar(dtype="float16")
                        scalar_tmp_f16.set_as(
                            self.ub_offsets_ceil_f16[group_idx * self.dim_kh * self.dim_kw + \
                                                     kh_idx * self.dim_kw + kw_idx])
                        self.vector_dup(ub_offset_x_ceil_f16, Constant.F16_CAL_MASK_SIZE,
                                        scalar_tmp_f16, dtype="float16")
                        scalar_tmp_f16.set_as(
                            self.ub_offsets_ceil_f16[(self.dim_group + group_idx) * self.dim_kh * self.dim_kw + \
                                                     kh_idx * self.dim_kw + kw_idx])
                        self.vector_dup(ub_offset_y_ceil_f16, Constant.F16_CAL_MASK_SIZE,
                                        scalar_tmp_f16, dtype="float16")
                        scalar_tmp_f16.set_as(
                            self.ub_offsets_floor_f16[group_idx * self.dim_kh * self.dim_kw + \
                                                      kh_idx * self.dim_kw + kw_idx])
                        self.vector_dup(ub_offset_x_floor_f16, Constant.F16_CAL_MASK_SIZE,
                                        scalar_tmp_f16, dtype="float16")
                        scalar_tmp_f16.set_as(
                            self.ub_offsets_floor_f16[(self.dim_group + group_idx) * self.dim_kh * self.dim_kw + \
                                                      kh_idx * self.dim_kw + kw_idx])
                        self.vector_dup(ub_offset_y_floor_f16, Constant.F16_CAL_MASK_SIZE,
                                        scalar_tmp_f16, dtype="float16")

                        self.tik_instance.vec_cmpv_ge(self.sel_mask1,
                                                      ub_offset_y_ceil_f16,
                                                      self.ub_limit_0,
                                                      1, 0, 0)
                        self.tik_instance.vec_cmpv_le(self.sel_mask2,
                                                      ub_offset_y_ceil_f16,
                                                      self.ub_limit_h,
                                                      1, 0, 0)
                        self.tik_instance.vand(Constant.F16_CAL_MASK_SIZE,
                                               self.sel_mask_ceil_y,
                                               self.sel_mask1,
                                               self.sel_mask2,
                                               1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vec_cmpv_ge(self.sel_mask1,
                                                      ub_offset_y_floor_f16,
                                                      self.ub_limit_0,
                                                      1, 0, 0)
                        self.tik_instance.vec_cmpv_le(self.sel_mask2,
                                                      ub_offset_y_floor_f16,
                                                      self.ub_limit_h,
                                                      1, 0, 0)
                        self.tik_instance.vand(Constant.F16_CAL_MASK_SIZE,
                                               self.sel_mask_floor_y,
                                               self.sel_mask1,
                                               self.sel_mask2,
                                               1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vec_cmpv_ge(self.sel_mask1,
                                                      ub_offset_x_ceil_f16,
                                                      self.ub_limit_0,
                                                      1, 0, 0)
                        self.tik_instance.vec_cmpv_le(self.sel_mask2,
                                                      ub_offset_x_ceil_f16,
                                                      self.ub_limit_w,
                                                      1, 0, 0)
                        self.tik_instance.vand(Constant.F16_CAL_MASK_SIZE,
                                               self.sel_mask_ceil_x,
                                               self.sel_mask1,
                                               self.sel_mask2,
                                               1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vec_cmpv_ge(self.sel_mask1,
                                                      ub_offset_x_floor_f16,
                                                      self.ub_limit_0,
                                                      1, 0, 0)
                        self.tik_instance.vec_cmpv_le(self.sel_mask2,
                                                      ub_offset_x_floor_f16,
                                                      self.ub_limit_w,
                                                      1, 0, 0)
                        self.tik_instance.vand(Constant.F16_CAL_MASK_SIZE,
                                               self.sel_mask_floor_x,
                                               self.sel_mask1,
                                               self.sel_mask2,
                                               1, 1, 1, 1, 8, 8, 8)
                        self.clip_tensor(ub_offset_x_ceil_f16,
                                         ub_offset_y_ceil_f16,
                                         Constant.F16_CAL_MASK_SIZE,
                                         dtype="float16")
                        self.clip_tensor(ub_offset_x_floor_f16,
                                         ub_offset_y_floor_f16,
                                         Constant.F16_CAL_MASK_SIZE,
                                         dtype="float16")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance,
                                                          ub_offsets_x_ceil_int32,
                                                          ub_offset_x_ceil_f16,
                                                          self.conv_num,
                                                          mode="ceil")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance,
                                                          ub_offsets_x_floor_int32,
                                                          ub_offset_x_floor_f16,
                                                          self.conv_num,
                                                          mode="ceil")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance,
                                                          ub_offsets_y_ceil_int32,
                                                          ub_offset_y_ceil_f16,
                                                          self.conv_num,
                                                          mode="ceil")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance,
                                                          ub_offsets_y_floor_int32,
                                                          ub_offset_y_floor_f16,
                                                          self.conv_num,
                                                          mode="ceil")
                    else:
                        ub_offset_x_ceil_f32 = self.tik_instance.Tensor("float32",
                                                                        (Constant.CAL_MASK_SIZE,),
                                                                        scope=tbe_platform.scope_ubuf,
                                                                        name="ub_offset_x_ceil_f32")
                        ub_offset_y_ceil_f32 = self.tik_instance.Tensor("float32",
                                                                        (Constant.CAL_MASK_SIZE,),
                                                                        scope=tbe_platform.scope_ubuf,
                                                                        name="ub_offset_y_ceil_f32")
                        ub_offset_x_floor_f32 = self.tik_instance.Tensor("float32",
                                                                         (Constant.CAL_MASK_SIZE,),
                                                                         scope=tbe_platform.scope_ubuf,
                                                                         name="ub_offset_x_floor_f32")
                        ub_offset_y_floor_f32 = self.tik_instance.Tensor("float32",
                                                                         (Constant.CAL_MASK_SIZE,),
                                                                         scope=tbe_platform.scope_ubuf,
                                                                         name="ub_offset_y_floor_f32")
                        scalar_tmp_f32.set_as(
                            self.ub_offsets_ceil[group_idx * self.dim_kh * self.dim_kw + \
                                                 kh_idx * self.dim_kw + kw_idx])
                        self.vector_dup(ub_offset_x_ceil_f32, Constant.CAL_MASK_SIZE, scalar_tmp_f32)
                        scalar_tmp_f32.set_as(
                            self.ub_offsets_ceil[(self.dim_group + group_idx) * self.dim_kh * self.dim_kw + \
                                                 kh_idx * self.dim_kw + kw_idx])
                        self.vector_dup(ub_offset_y_ceil_f32, Constant.CAL_MASK_SIZE, scalar_tmp_f32)
                        scalar_tmp_f32.set_as(
                            self.ub_offsets_floor[group_idx * self.dim_kh * self.dim_kw + \
                                                  kh_idx * self.dim_kw + kw_idx])
                        self.vector_dup(ub_offset_x_floor_f32, Constant.CAL_MASK_SIZE, scalar_tmp_f32)
                        scalar_tmp_f32.set_as(
                            self.ub_offsets_floor[(self.dim_group + group_idx) * self.dim_kh * self.dim_kw + \
                                                  kh_idx * self.dim_kw + kw_idx])
                        self.vector_dup(ub_offset_y_floor_f32, Constant.CAL_MASK_SIZE, scalar_tmp_f32)

                        self.tik_instance.vec_cmpv_ge(self.sel_mask1,
                                                      ub_offset_y_ceil_f32,
                                                      self.ub_limit_0,
                                                      1, 0, 0)
                        self.tik_instance.vec_cmpv_le(self.sel_mask2,
                                                      ub_offset_y_ceil_f32,
                                                      self.ub_limit_h,
                                                      1, 0, 0)
                        self.tik_instance.vand(Constant.CAL_MASK_SIZE,
                                               self.sel_mask_ceil_y,
                                               self.sel_mask1,
                                               self.sel_mask2,
                                               1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vec_cmpv_ge(self.sel_mask1,
                                                      ub_offset_y_floor_f32,
                                                      self.ub_limit_0,
                                                      1, 0, 0)
                        self.tik_instance.vec_cmpv_le(self.sel_mask2,
                                                      ub_offset_y_floor_f32,
                                                      self.ub_limit_h,
                                                      1, 0, 0)
                        self.tik_instance.vand(Constant.CAL_MASK_SIZE,
                                               self.sel_mask_floor_y,
                                               self.sel_mask1,
                                               self.sel_mask2,
                                               1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vec_cmpv_ge(self.sel_mask1,
                                                      ub_offset_x_ceil_f32,
                                                      self.ub_limit_0,
                                                      1, 0, 0)
                        self.tik_instance.vec_cmpv_le(self.sel_mask2,
                                                      ub_offset_x_ceil_f32,
                                                      self.ub_limit_w,
                                                      1, 0, 0)
                        self.tik_instance.vand(Constant.CAL_MASK_SIZE,
                                               self.sel_mask_ceil_x,
                                               self.sel_mask1,
                                               self.sel_mask2,
                                               1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vec_cmpv_ge(self.sel_mask1,
                                                      ub_offset_x_floor_f32,
                                                      self.ub_limit_0,
                                                      1, 0, 0)
                        self.tik_instance.vec_cmpv_le(self.sel_mask2,
                                                      ub_offset_x_floor_f32,
                                                      self.ub_limit_w,
                                                      1, 0, 0)
                        self.tik_instance.vand(Constant.CAL_MASK_SIZE,
                                               self.sel_mask_floor_x,
                                               self.sel_mask1,
                                               self.sel_mask2,
                                               1, 1, 1, 1, 8, 8, 8)
                        self.clip_tensor(ub_offset_x_ceil_f32,
                                         ub_offset_y_ceil_f32,
                                         Constant.CAL_MASK_SIZE)
                        self.clip_tensor(ub_offset_x_floor_f32,
                                         ub_offset_y_floor_f32,
                                         Constant.CAL_MASK_SIZE)
                        util_tik_comm_func.tik_func_vconv(self.tik_instance,
                                                          ub_offsets_x_ceil_int32,
                                                          ub_offset_x_ceil_f32,
                                                          self.conv_num,
                                                          mode="ceil")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance,
                                                          ub_offsets_x_floor_int32,
                                                          ub_offset_x_floor_f32,
                                                          self.conv_num,
                                                          mode="ceil")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance,
                                                          ub_offsets_y_ceil_int32,
                                                          ub_offset_y_ceil_f32,
                                                          self.conv_num,
                                                          mode="ceil")
                        util_tik_comm_func.tik_func_vconv(self.tik_instance,
                                                          ub_offsets_y_floor_int32,
                                                          ub_offset_y_floor_f32,
                                                          self.conv_num,
                                                          mode="ceil")
                    scalar_idx_lth_int.set_as(ub_offsets_y_floor_int32[0])
                    scalar_idx_ltw_int.set_as(ub_offsets_x_floor_int32[0])
                    scalar_idx_lbh_int.set_as(ub_offsets_y_ceil_int32[0])
                    scalar_idx_rtw_int.set_as(ub_offsets_x_ceil_int32[0])
                    ub_lt_x = self.tik_instance.Tensor(
                        self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_lt_x")
                    ub_lb_x = self.tik_instance.Tensor(
                        self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_lb_x")
                    ub_rt_x = self.tik_instance.Tensor(
                        self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_rt_x")
                    ub_rb_x = self.tik_instance.Tensor(
                        self.x_dtype, (self.dim_group_c,), scope=tbe_platform.scope_ubuf, name="ub_rb_x")
                    self.tik_instance.vand(Constant.CAL_MASK_SIZE,
                                           self.sel_mask1,
                                           self.sel_mask_floor_y,
                                           self.sel_mask_floor_x,
                                           1, 1, 1, 1, 8, 8, 8)
                    self.get_x(ub_lt_x, [n_idx, scalar_idx_lth_int, scalar_idx_ltw_int, group_idx],
                               self.dim_group_c, self.sel_mask1)
                    self.tik_instance.vand(Constant.CAL_MASK_SIZE,
                                           self.sel_mask1,
                                           self.sel_mask_ceil_y,
                                           self.sel_mask_floor_x,
                                           1, 1, 1, 1, 8, 8, 8)
                    self.get_x(ub_lb_x, [n_idx, scalar_idx_lbh_int, scalar_idx_lbw_int, group_idx],
                               self.dim_group_c, self.sel_mask1)
                    self.tik_instance.vand(Constant.CAL_MASK_SIZE,
                                           self.sel_mask1,
                                           self.sel_mask_floor_y,
                                           self.sel_mask_ceil_x,
                                           1, 1, 1, 1, 8, 8, 8)
                    self.get_x(ub_rt_x, [n_idx, scalar_idx_rth_int, scalar_idx_rtw_int, group_idx],
                               self.dim_group_c, self.sel_mask1)
                    self.tik_instance.vand(Constant.CAL_MASK_SIZE,
                                           self.sel_mask1,
                                           self.sel_mask_ceil_y,
                                           self.sel_mask_ceil_x,
                                           1, 1, 1, 1, 8, 8, 8)
                    self.get_x(ub_rb_x, [n_idx, scalar_idx_rbh_int, scalar_idx_rbw_int, group_idx],
                               self.dim_group_c, self.sel_mask1)
                    scalar_tmp_f32.set_as(
                        self.ub_offsets_ceil_sub[group_idx * self.dim_kh * self.dim_kw + \
                                                 kh_idx * self.dim_kw + kw_idx])
                    self.vector_dup(ceil_sub_x, self.dim_group_c, scalar_tmp_f32, self.x_dtype)
                    scalar_tmp_f32.set_as(
                        self.ub_offsets_ceil_sub[(self.dim_group + group_idx) * self.dim_kh * self.dim_kw + \
                                                 kh_idx * self.dim_kw + kw_idx])
                    self.vector_dup(ceil_sub_y, self.dim_group_c, scalar_tmp_f32, self.x_dtype)
                    scalar_tmp_f32.set_as(
                        self.ub_offsets_sub_floor[group_idx * self.dim_kh * self.dim_kw + \
                                                  kh_idx * self.dim_kw + kw_idx])
                    self.vector_dup(sub_floor_x, self.dim_group_c, scalar_tmp_f32, self.x_dtype)
                    scalar_tmp_f32.set_as(
                        self.ub_offsets_sub_floor[(self.dim_group + group_idx) * self.dim_kh * self.dim_kw + \
                                                  kh_idx * self.dim_kw + kw_idx])
                    self.vector_dup(sub_floor_y, self.dim_group_c, scalar_tmp_f32, self.x_dtype)
                    self.vector_mul(ub_weight_lt, sub_floor_y, sub_floor_x, self.dim_group_c, self.x_dtype)
                    self.vector_mul(ub_weight_lb, sub_floor_x, ceil_sub_y, self.dim_group_c, self.x_dtype)
                    self.vector_mul(ub_weight_rt, sub_floor_y, ceil_sub_x, self.dim_group_c, self.x_dtype)
                    self.vector_mul(ub_weight_rb, ceil_sub_x, ceil_sub_y, self.dim_group_c, self.x_dtype)
                    self.vector_mul(ub_lt_x, ub_lt_x, ub_weight_lt, self.dim_group_c, self.x_dtype)
                    self.vector_mul(ub_lb_x, ub_lb_x, ub_weight_lb, self.dim_group_c, self.x_dtype)
                    self.vector_mul(ub_rt_x, ub_rt_x, ub_weight_rt, self.dim_group_c, self.x_dtype)
                    self.vector_mul(ub_rb_x, ub_rb_x, ub_weight_rb, self.dim_group_c, self.x_dtype)
                    self.vector_add(ub_lt_x, ub_lt_x, ub_lb_x, self.dim_group_c, self.x_dtype)
                    self.vector_add(ub_lt_x, ub_lt_x, ub_rt_x, self.dim_group_c, self.x_dtype)
                    self.vector_add(ub_lt_x, ub_lt_x, ub_rb_x, self.dim_group_c, self.x_dtype)
                    self.vector_mul(ub_lt_x, ub_lt_x, ub_offset_s, self.dim_group_c, self.x_dtype)
                    burst_len_out = _ceil_div(self.dim_group_c, self.data_in_one_block)
                    out_ub_addr = (out_ub_start * self.dim_group + group_idx) * self.dim_group_c
                    self.tik_instance.data_move(ub_out[out_ub_addr],
                                                ub_lt_x[0],
                                                constant.SID,
                                                1, burst_len_out, 0, 0)

    def load_offsets(self, offsets_start, helper_start, load_num):
        """
        Load offsets and helper from gm tensor and prepare for ceil and floor offsets tensor.
        -------
        Parameters:
        ----------
        offsets_start: The index w of offsets to load.
        helper_start: The index w of helper to load.
        load_num: The length will be loaded.
        """
        burst_len = _ceil_div(load_num, self.data_in_one_block)
        self.tik_instance.data_move(self.ub_offsets_ori[0],
                                    self.offsets_gm[offsets_start],
                                    constant.SID,
                                    1, burst_len, 0, 0)
        self.tik_instance.data_move(self.ub_helper[0],
                                    self.helper_gm[helper_start],
                                    constant.SID,
                                    1, burst_len, 0, 0)
        self.vector_add(self.ub_offsets_ori, self.ub_offsets_ori, self.ub_helper, load_num)
        util_tik_comm_func.tik_func_vconv(self.tik_instance, self.ub_offsets_int32_ceil, self.ub_offsets_ori,
                                          load_num, mode="ceil")
        util_tik_comm_func.tik_func_vconv(self.tik_instance, self.ub_offsets_int32_floor, self.ub_offsets_ori,
                                          load_num, mode="floor")
        util_tik_comm_func.tik_func_vconv(self.tik_instance, self.ub_offsets_ceil,
                                          self.ub_offsets_int32_ceil, load_num)
        util_tik_comm_func.tik_func_vconv(self.tik_instance, self.ub_offsets_floor,
                                          self.ub_offsets_int32_floor, load_num)
        if (self.x_dtype == "float16") or (not self.cmp_flag):
            util_tik_comm_func.tik_func_vconv(self.tik_instance, self.ub_offsets_ceil_f16,
                                              self.ub_offsets_int32_ceil, load_num)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, self.ub_offsets_floor_f16,
                                              self.ub_offsets_int32_floor, load_num)
        self.vector_sub(self.ub_offsets_ceil_sub, self.ub_offsets_ori, self.ub_offsets_ceil, load_num)
        self.vector_adds(self.ub_offsets_ceil_sub, self.ub_offsets_ceil_sub, self.scalar_const_pos1, load_num)
        self.vector_sub(self.ub_offsets_sub_floor, self.ub_offsets_floor, self.ub_offsets_ori, load_num)
        self.vector_adds(self.ub_offsets_sub_floor, self.ub_offsets_sub_floor, self.scalar_const_pos1, load_num)

@register_operator("DeformableOffsets")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def deformable_offsets(x,
                       offsets,
                       helper,
                       y,
                       strides,
                       pads,
                       ksize,
                       dialations=(1, 1, 1, 1),
                       data_format="NHWC",
                       deformable_groups=1,
                       modulated=True,
                       kernel_name="deformable_offsets"):
    """
    Computes the deformed convolution output with the expected input
    Parameters:
    ----------
    x: A Tensor of type float16,float32
    offsets: A Tensor of type float16,float32.Deformation offset parameter.
    Required Attributes:
    strides: A tuple/list of 4 integers.The stride of the sliding window for
             height and width for H/W dimension.
    pads: A tuple/list of 4 integers.Padding added to H/W dimension
          of the input.
    ksize: A tuple/list of 2 integers.kernel size.
    Attributes:
    dilations: A tuple/list of 4 integers, The dilation factor for each dimension
               of input.  Defaults to [1, 1, 1, 1]
    data_format: An optional string from: "NCHW", "NHWC". Defaults to "NCHW". Specify the data format of the input x.
    deformable_groups: Specify the c-axis grouping number of input x.
    modulated: Specify version of DeformableConv2D, true means v2, false means v1
    Outputs:
    y: A Tensor. A Tensor of type float16, float32.
    """
    dfm_inst = DeformableOffsets(x, offsets, helper, y,
                                 strides, pads, ksize, dialations,
                                 data_format, deformable_groups,
                                 kernel_name)
    dfm_inst.deformable_offset_compute()
