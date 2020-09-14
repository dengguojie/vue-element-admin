"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

crop_and_resize
"""
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *


# define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
SCALAR_MIN_FP32 = 2**(-126)
# define a scalar, value = 2**(50)
SCALAR_MUL_FP32 = 2**50
# define a scalar, value = 2**(26)
SCALAR_MUL2_FP32 = 2**26


def check_supported(x, boxes, box_index, y, crop_size, extrapolation_value,
                    method, kernel_name="crop_and_resize"):
    """To check whether the AICORE operator can support the length of w/h or not
    """
    input_shape = x.get("ori_shape")
    input_type = x.get("dtype")
    input_format = x.get("ori_format")
    output_h, output_w = crop_size
    boxes_shape = boxes.get("ori_shape")
    boxes_num = boxes_shape[0]
    if method not in ("bilinear",):
        return False

    if boxes_num <= 50 or boxes_num > 3500:
        # boxes_num is more, the performance is more xxcellent than aicpu
        return False

    if input_type in ("float32", "float16",):
        # c0 // num in one block
        copy_block = 2
    else:
        return False

    if len(input_shape) != 4:
        # shape must be [N, H, W, C] or [N, C, H, W]
        return False

    if input_format in ("NHWC",):
        input_c = input_shape[3]
        input_h = input_shape[1]
        input_w = input_shape[2]
    elif input_format in ("NCHW",):
        input_c = input_shape[1]
        input_h = input_shape[2]
        input_w = input_shape[3]
    else:
        # format must be NHWC or NCHW
        return False

    if input_c < 512:
        return False

    if input_h * input_w * copy_block > 5000 or output_h * output_w * copy_block > 5000:
        return False

    if max(output_h, output_w) > 16:
        # tmp limit for fasterrcnn
        return False

    return True


def ub_offset(ub):
    """get ub offset
    when ub.shape is 1D tensor offset = 0
    when ub.shape is not 1D tensor change offset = 1D
    ex:
       ub.shape = [2,2,2]
       ub1 = ub[1,:,:]
       ub_offset(ub1) = 2*2 = 4 for ub
    """
    ub_shape = ub.shape
    if len(ub_shape) in (0, 1):
        return 0

    return ub.offset


def tik_func_vcomple(tik_instance, function, out_dst, src0, src1, copy_num,
                     dst_blk=1, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8,
                     src1_rep=8):
    """tik_func_vcomple
    """
    do_dtype = out_dst.dtype
    if do_dtype in ("float16",):
        block_num = 16
    else:
        block_num = 8
    vector_num = block_num*8
    repeat_time = copy_num // vector_num
    repeat_tail = copy_num % vector_num
    tik_fun = None
    ori_offset_dst = ub_offset(out_dst)
    ori_offset_src0 = ub_offset(src0)
    ori_offset_src1 = ub_offset(src1)
    if function == "vmin":
        tik_fun = tik_instance.vmin
    elif function == "vmax":
        tik_fun = tik_instance.vmax
    elif function == "vmul":
        tik_fun = tik_instance.vmul
    elif function == "vadd":
        tik_fun = tik_instance.vadd
    elif function == "vsub":
        tik_fun = tik_instance.vsub

    while repeat_time > 255:
        tik_fun(vector_num,
                out_dst[ori_offset_dst],
                src0[ori_offset_src0],
                src1[ori_offset_src1],
                255,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)
        repeat_time = repeat_time - 255
        ori_offset_dst = ori_offset_dst + 255 * block_num * dst_rep
        ori_offset_src0 = ori_offset_src0 + 255 * block_num * src0_rep
        ori_offset_src1 = ori_offset_src1 + 255 * block_num * src1_rep

    if repeat_time > 0:
        tik_fun(vector_num,
                out_dst[ori_offset_dst],
                src0[ori_offset_src0],
                src1[ori_offset_src1],
                repeat_time,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)
        ori_offset_dst = ori_offset_dst + repeat_time * block_num * dst_rep
        ori_offset_src0 = ori_offset_src0 + repeat_time * block_num * src0_rep
        ori_offset_src1 = ori_offset_src1 + repeat_time * block_num * src1_rep

    if repeat_tail > 0:
        tik_fun(repeat_tail,
                out_dst[ori_offset_dst],
                src0[ori_offset_src0],
                src1[ori_offset_src1],
                1,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)


def tik_func_vmuls(tik_instance, dst_ub, src_ub, value, do_len):
    """tik_func_vmuls
    """
    vmuls_type = dst_ub.dtype
    vector_num = 128
    if vmuls_type in ("float16",):
        vector_num = 128
    elif vmuls_type in ("float32", "int32"):
        vector_num = 64
    repeat = do_len // vector_num
    repeat_tail = do_len % vector_num
    dst_offset = ub_offset(dst_ub)
    src_offset = ub_offset(src_ub)
    while repeat > 255:
        tik_instance.vmuls(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                           255, 1, 1, 8, 8)
        repeat = repeat - 255
        dst_offset = dst_offset + vector_num * 255
        src_offset = src_offset + vector_num * 255
    if repeat > 0:
        tik_instance.vmuls(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                           repeat, 1, 1, 8, 8)
        dst_offset = dst_offset + vector_num * repeat
        src_offset = src_offset + vector_num * repeat
    if repeat_tail > 0:
        tik_instance.vmuls(repeat_tail, dst_ub[dst_offset], src_ub[src_offset], value,
                           1, 1, 1, 8, 8)


def tik_func_vconv(tik_instance, dst_ub, src_ub, do_len, mode=""):
    """tik_func_vconv
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype

    def do_vconv(dst_repeat_stride, src_repeat_stride, deq_scale=None, block_num=64):
        ori_dst_offset = ub_offset(dst_ub)
        ori_src_offset = ub_offset(src_ub)
        repeat = do_len // block_num
        repeat_tail = do_len % block_num
        while repeat > 255:
            tik_instance.vconv(block_num, mode, tmp_fp16_ub[ori_dst_offset], src_ub[ori_src_offset],
                               255, 1, 1, dst_repeat_stride, src_repeat_stride, deqscale=deq_scale)
            repeat = repeat - 255
            ori_dst_offset = ori_dst_offset + block_num*255
            ori_src_offset = ori_src_offset + block_num*255
        if repeat > 0:
            tik_instance.vconv(64, mode, tmp_fp16_ub[ori_dst_offset], src_ub[ori_src_offset],
                               repeat, 1, 1, dst_repeat_stride, src_repeat_stride, deqscale=deq_scale)
            ori_dst_offset = ori_dst_offset + block_num*repeat
            ori_src_offset = ori_src_offset + block_num*repeat
        if repeat_tail > 0:
            tik_instance.vconv(repeat_tail, mode, dst_ub[ori_dst_offset], src_ub[ori_src_offset],
                               1, 1, 1, dst_repeat_stride, src_repeat_stride, deqscale=deq_scale)

    if src_dtype in ("float32",) and dst_dtype in ("int32",):
        cast_flag = tbe_platform.cce_conf.api_check_support("tik.vconv", "f322s32r")
        if not cast_flag:
            with tik_instance.new_stmt_scope():
                tmp_fp16_ub = tik_instance.Tensor(
                    "float16", (((do_len + 15) // 16) * 16,),
                    name="tmp_fp16_ub", scope=tik.scope_ubuf)
                tik_func_vconv(tik_instance, tmp_fp16_ub, src_ub, do_len)
                tik_func_vconv(tik_instance, dst_ub, tmp_fp16_ub, do_len, mode)
        else:
            do_vconv(8, 8)

    elif src_dtype in ("float32",) and dst_dtype in ("float16",):
        do_vconv(4, 8)

    elif src_dtype in ("float16",) and dst_dtype in ("int32",):
        do_vconv(8, 4)

    elif src_dtype in ("int32",) and dst_dtype in ("float16",):
        do_vconv(4, 8, 1.0)

    elif src_dtype in ("float16",) and dst_dtype in ("float32",):
        do_vconv(8, 4)

    elif src_dtype in ("int32",) and dst_dtype in ("float32",):
        cast_flag = tbe_platform.cce_conf.api_check_support("tik.vconv", "s322f32")
        if not cast_flag:
            with tik_instance.new_stmt_scope():
                tmp_fp16_ub = tik_instance.Tensor(
                    "float16", (((do_len + 15) // 16) * 16,),
                    name="tmp_fp16_ub", scope=tik.scope_ubuf)
                tik_func_vconv(tik_instance, tmp_fp16_ub, src_ub, do_len)
                tik_func_vconv(tik_instance, dst_ub, tmp_fp16_ub, do_len)
        else:
            do_vconv(8, 8)


class CropAndResize:
    """
    Function: use to store CropAndResize base parameters
    Modify : 2020-8-4
    """
    def __init__(self,
                 x,
                 boxes,
                 box_index,
                 crop_size,
                 y,
                 extrapolation_value,
                 method):
        """
        Init CropAndResize base parameters

        Returns
        -------
        None
        """
        self.image_shape = x.get("shape")
        self.image_type = x.get("dtype")
        self.boxes_shape = boxes.get("shape")
        self.boxes_index_shape = box_index.get("shape")
        self.boxes_index_type = box_index.get("dtype")
        self.crop_size = crop_size
        self.extrapolation_value = extrapolation_value
        self.method = method
        self.output_shape = y.get("shape")
        self.output_type = y.get("dtype")

        # init tik_instance
        self.tik_instance = tik.Tik()
        self.aicore_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.input_gm_list = []
        self.output_gm_list = []

        # parsing input
        self.crop_height, self.crop_width = crop_size
        self.batch_size, self.image_c1, self.image_height, self.image_width, self.image_c0 = self.image_shape
        self.num_boxes, _ = self.boxes_shape
        if self.image_type in ("float32",):
            self.block_num = 8
        else:
            self.block_num = 16

        self.index_ub = None
        self.height_mask_list = None
        self.width_mask_list = None

    def get_tik_instance(self):
        """get_tik_instance
        """
        return self.tik_instance

    def apply_mem(self, shape, name, mem_type=tik.scope_ubuf, dtype=None):
        """tik.scope_ubuf / tik.scope_cbuf / tik.scope_gm
        """
        if dtype is None:
            dtype = self.image_type

        return self.tik_instance.Tensor(dtype, shape, name=name, scope=mem_type)

    def init_gm_mem(self):
        """init tik gm mem
        """
        # init gm input
        image_gm = self.apply_mem(self.image_shape, "image_gm", tik.scope_gm)
        boxes_gm = self.apply_mem(self.boxes_shape, "boxes_gm", tik.scope_gm)
        boxes_index_gm = self.apply_mem(self.boxes_index_shape, "boxes_index_gm", tik.scope_gm, self.boxes_index_type)

        self.input_gm_list = [image_gm, boxes_gm, boxes_index_gm]

        # init gm output
        y_gm = self.apply_mem(self.output_shape, "y_gm", tik.scope_gm, self.output_type)
        self.output_gm_list = [y_gm]

    def init_ub_scalar(self):
        """gen two uint64 scalar for vector mask
        one is 0001000100010001000100010001000100010001000100010001000100010001
        another is 0010001000100010001000100010001000100010001000100010001000100010
        """
        # for vector mask
        height_mask1_scalar = self.tik_instance.Scalar(dtype="uint64")
        width_mask1_scalar = self.tik_instance.Scalar(dtype="uint64")
        height_scale_mask = int("0001"*16, 2)
        width_scale_mask = int("0010"*16, 2)
        if self.block_num == 8:
            zero_scalar = self.tik_instance.Scalar(dtype="uint64")
            zero_scalar.set_as(0)
            height_mask1_scalar.set_as(height_scale_mask)
            width_mask1_scalar.set_as(width_scale_mask)
            self.height_mask_list = [zero_scalar, height_mask1_scalar]
            self.width_mask_list = [zero_scalar, width_mask1_scalar]
        else:
            height_mask1_scalar.set_as(height_scale_mask)
            width_mask1_scalar.set_as(width_scale_mask)
            self.height_mask_list = [height_mask1_scalar, height_mask1_scalar]
            self.width_mask_list = [width_mask1_scalar, width_mask1_scalar]

    def build_tik_instance(self, kernel_name_value):
        """build tik instance
        """
        self.tik_instance.BuildCCE(kernel_name=kernel_name_value,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   output_files_path=None,
                                   enable_l2=False)

        return self.tik_instance

    def get_core_cut(self):
        """calc the core used num and boxes num for per core
        """
        boxes_num_per_core = get_ceil_int(self.num_boxes, self.aicore_num)
        core_used = get_ceil_int(self.num_boxes, boxes_num_per_core)
        boxes_num_last_core = self.num_boxes - boxes_num_per_core * (core_used - 1)

        return core_used, boxes_num_per_core, boxes_num_last_core


def get_ceil_int(int1, int2):
    """get cel for input1 and input2
    """
    if int1 == 0:
        return 1
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result

    return _result + 1


def fill_index_in_ub(tik_instance, idx_ub, idx_num):
    """fill 0,1,2  .... (idx_num -1) in idx_ub
    when the idx_num is less than 16, fill it one by one
    when the type is not int32, will fill in int32 ub and cast to idx_ub dtype
    when the type is int32, will fill in int32 one by one
    """
    # when the idx_num is less than 16, fill it one by one
    idx_ub_type = idx_ub.dtype
    if idx_num <= 16:
        _idx_scalar = tik_instance.Scalar(dtype=idx_ub.dtype)
        for _idx in range(idx_num):
            _idx_scalar.set_as(_idx)
            idx_ub[_idx].set_as(_idx_scalar)
    # when the type is not int32, will fill in int32 ub and cast to idx_ub dtype
    elif idx_ub_type not in ("int32",):
        with tik_instance.new_stmt_scope():
            idx_ub_int32 = tik_instance.Tensor("int32", (idx_num, ),
                                               name="idx_ub_int32", scope=tik.scope_ubuf)
            with tik_instance.for_range(0, idx_num) as _idx:
                idx_ub_int32[_idx].set_as(_idx)
        # cast to idx_ub
        tik_func_vconv(tik_instance, idx_ub, idx_ub_int32, idx_num)
    else:
        with tik_instance.for_range(0, idx_num) as _idx:
            idx_ub[_idx].set_as(_idx)


def do_crop_and_resize_compute_one_core(box_num_sigment, obj, box_num_offset):
    """do crop and resize in one core
        step 1 read boxes from boxes and calc h_top_index/h_bottom_index/h_lerp/w_left_index/w_right_index/w_lerp
        step 2 read input_batch_num from box_index
        step 3 copy 4 data(Total C(C1*C0)) in ub
               use use input_batch_num/h_top_index/h_bottom_index/w_left_index/w_right_index
        step 4 calcu the out
               top = top_left + (top_right - top_left) * x_lerp
               bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
               out = top + (bottom - top) * y_lerp;

    Parameters:
    ----------
    box_num_sigment : int.
        the crop boxes num for one core
    obj : class.
        crop_and_resize par object
    box_num_offset: int
        copy boxes offset

    Returns
    -------
    None
    """
    tik_instance = obj.get_tik_instance()
    # get float32 index ub
    index_ub = obj.index_ub
    men_len = get_ceil_int(box_num_sigment*4, obj.block_num*8) * obj.block_num*8

    # apply ub mem for index
    boxes_ub_small = obj.apply_mem((men_len,), "boxes_ub_h1", tik.scope_ubuf)
    boxes_ub_big = obj.apply_mem((men_len,), "boxes_ub_h2", tik.scope_ubuf)
    boxes_ub_scale = obj.apply_mem((men_len,), "boxes_ub_scale", tik.scope_ubuf)
    copy_burst_len = get_ceil_int(box_num_sigment*4, obj.block_num)

    # copy boxes in boxes_ub_small
    tik_instance.data_move(boxes_ub_small, obj.input_gm_list[1][box_num_offset*4],
                           0, 1, copy_burst_len, 0, 0)
    copy_burst_len = get_ceil_int(box_num_sigment*4 - 2, obj.block_num)
    # copy boxes[2] in boxes_ub_small
    tik_instance.data_move(boxes_ub_big, obj.input_gm_list[1][box_num_offset*4 + 2],
                           0, 1, copy_burst_len, 0, 0)
    # calc boxes[2] - boxes  means y2 - y1 and x2 - x1
    tik_func_vcomple(tik_instance, "vsub", boxes_ub_scale, boxes_ub_big, boxes_ub_small, men_len)

    # calc resize scale for h and w
    # to get scale_h: scale * (image_height - 1) / (crop_height - 1)
    repeat_time = get_ceil_int(box_num_sigment*4, obj.block_num*8)
    tik_instance.vmuls([obj.height_mask_list[0], obj.height_mask_list[1]],
                       boxes_ub_scale, boxes_ub_scale, (obj.image_height - 1) / (obj.crop_height - 1),
                       repeat_time, 1, 1, 8, 8)
    # to get scale_w:  scale * (image_width - 1) / (crop_width - 1)
    tik_instance.vmuls(obj.width_mask_list,
                       boxes_ub_scale, boxes_ub_scale, (obj.image_width - 1) / (obj.crop_width - 1),
                       repeat_time, 1, 1, 8, 8)
    # to get h_small: h_small * (image_height - 1)
    tik_instance.vmuls(obj.height_mask_list,
                       boxes_ub_small, boxes_ub_small, obj.image_height - 1,
                       repeat_time, 1, 1, 8, 8)
    # to get w_small: w_small * (image_width - 1)
    tik_instance.vmuls(obj.width_mask_list,
                       boxes_ub_small, boxes_ub_small, obj.image_width - 1,
                       repeat_time, 1, 1, 8, 8)

    # box_index process for one sigment
    box_index_ub = obj.apply_mem((get_ceil_int(box_num_sigment, obj.block_num)*obj.block_num,),
                                 "box_index_ub", tik.scope_ubuf, "int32")
    copy_burst_len = get_ceil_int(box_num_sigment, 8)
    tik_instance.data_move(box_index_ub, obj.input_gm_list[2][box_num_offset],
                           0, 1, copy_burst_len, 0, 0)

    with tik_instance.for_range(0, box_num_sigment) as _box_idx:
        _out_batch_idx = _box_idx + box_num_offset
        scaler_h_small = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        scaler_w_small = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        scaler_h_scale = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        scaler_w_scale = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        # read scale for h and w
        scaler_h_small.set_as(boxes_ub_small[_box_idx*4])
        scaler_w_small.set_as(boxes_ub_small[_box_idx*4 + 1])
        scaler_h_scale.set_as(boxes_ub_scale[_box_idx*4])
        scaler_w_scale.set_as(boxes_ub_scale[_box_idx*4 + 1])

        input_boxes_in_h = obj.apply_mem((get_ceil_int(obj.crop_height, 16) * 16,),
                                         "input_boxes_in_h", tik.scope_ubuf)
        tik_instance.vmuls(obj.crop_height,
                           input_boxes_in_h, index_ub, scaler_h_scale,
                           1, 1, 1, 8, 8)
        input_boxes_in_w = obj.apply_mem((get_ceil_int(obj.crop_width, 16) * 16,),
                                         "input_boxes_in_w", tik.scope_ubuf)
        tik_instance.vmuls(obj.crop_width,
                           input_boxes_in_w, index_ub, scaler_w_scale,
                           1, 1, 1, 8, 8)
        tik_instance.vadds(obj.crop_height,
                           input_boxes_in_h, input_boxes_in_h, scaler_h_small,
                           1, 1, 1, 8, 8)
        tik_instance.vadds(obj.crop_width,
                           input_boxes_in_w, input_boxes_in_w, scaler_w_small,
                           1, 1, 1, 8, 8)

        h_top_index = \
            obj.apply_mem((get_ceil_int(obj.crop_height, 16) * 16,), "h_top_index", tik.scope_ubuf, "int32")
        w_left_index = \
            obj.apply_mem((get_ceil_int(obj.crop_width, 16) * 16,), "w_left_index", tik.scope_ubuf, "int32")

        tik_func_vconv(tik_instance, h_top_index, input_boxes_in_h, obj.crop_height, mode="floor")
        tik_func_vconv(tik_instance, w_left_index, input_boxes_in_w, obj.crop_width, mode="floor")
        with tik_instance.new_stmt_scope():
            tmp_float_ub_0 = obj.apply_mem((get_ceil_int(obj.crop_height, 16) * 16,),
                                           "tmp_float_ub_0", tik.scope_ubuf)
            # h_top_index vconv from int32 to float32
            tik_func_vconv(tik_instance, tmp_float_ub_0, h_top_index, obj.crop_height)
            # do: h_lerp = input_boxes_in_h - tmp_float_ub
            tik_func_vcomple(tik_instance, "vsub", input_boxes_in_h,
                             input_boxes_in_h, tmp_float_ub_0, obj.crop_height)
            tmp_float_ub_1 = obj.apply_mem((get_ceil_int(obj.crop_height, 16) * 16,),
                                           "tmp_float_ub_1", tik.scope_ubuf)
            # h_top_index vconv from int32 to float32
            tik_func_vconv(tik_instance, tmp_float_ub_1, w_left_index, obj.crop_width)
            # do: w_lerp = input_boxes_in_h - tmp_float_ub
            tik_func_vcomple(tik_instance, "vsub", input_boxes_in_w,
                             input_boxes_in_w, tmp_float_ub_1, obj.crop_width)

        # when the product not support f322s32, will cast to fp16 and to int32, get error
        # when f32 value is 1.99998, cast int32 is 2, this step will process the error
        # step 1 int32 cast to fp32_new   2.0
        # step 2 int32_sub_fp32_value = f32_old - fp32_new
        # step 3 int32_sub_fp32_value = 0 when int32_sub_fp32_value >= 0
        #        int32_sub_fp32_value = 1 when int32_sub_fp32_value < 0
        # step 4 int32 - int32_sub_fp32_value
        support_cast = tbe_platform.cce_conf.api_check_support("tik.vconv", "f322s32r")
        if not support_cast:
            with tik_instance.new_stmt_scope():
                zero_ub = obj.apply_mem((obj.block_num*8,), "zero_ub", tik.scope_ubuf)
                fp32_min_ub = obj.apply_mem((obj.block_num*8,), "fp32_min_ub", tik.scope_ubuf)
                tmp_h_ub = obj.apply_mem(input_boxes_in_h.shape, "tmp_h_ub", tik.scope_ubuf)
                tmp_w_ub = obj.apply_mem(input_boxes_in_w.shape, "tmp_w_ub", tik.scope_ubuf)
                tik_instance.vmuls(obj.block_num*8, zero_ub, zero_ub, 0.0,
                                   1, 1, 1, 8, 8)
                tik_instance.vector_dup(obj.block_num*8, fp32_min_ub, SCALAR_MIN_FP32,
                                        1, 1, 1)
                tik_func_vmuls(tik_instance, tmp_h_ub,
                               input_boxes_in_h, -1.0, obj.crop_height)
                tik_func_vmuls(tik_instance, tmp_w_ub,
                               input_boxes_in_w, -1.0, obj.crop_width)

                tik_func_vcomple(tik_instance, "vmax", tmp_h_ub,
                                 zero_ub, tmp_h_ub, obj.crop_height, src0_rep=0)
                tik_func_vcomple(tik_instance, "vmax", tmp_w_ub,
                                 zero_ub, tmp_w_ub, obj.crop_width,  src0_rep=0)
                tik_func_vcomple(tik_instance, "vmin", tmp_h_ub,
                                 fp32_min_ub, tmp_h_ub, obj.crop_height, src0_rep=0)
                tik_func_vcomple(tik_instance, "vmin", tmp_w_ub,
                                 fp32_min_ub, tmp_w_ub, obj.crop_width,  src0_rep=0)
                tik_func_vmuls(tik_instance, tmp_h_ub,
                               tmp_h_ub, SCALAR_MUL_FP32, obj.crop_height)
                tik_func_vmuls(tik_instance, tmp_w_ub,
                               tmp_w_ub, SCALAR_MUL_FP32, obj.crop_width)
                tik_func_vmuls(tik_instance, tmp_h_ub,
                               tmp_h_ub, SCALAR_MUL_FP32, obj.crop_height)
                tik_func_vmuls(tik_instance, tmp_w_ub,
                               tmp_w_ub, SCALAR_MUL_FP32, obj.crop_width)
                tik_func_vmuls(tik_instance, tmp_h_ub,
                               tmp_h_ub, SCALAR_MUL2_FP32, obj.crop_height)
                tik_func_vmuls(tik_instance, tmp_w_ub,
                               tmp_w_ub, SCALAR_MUL2_FP32, obj.crop_width)

                tik_func_vcomple(tik_instance, "vadd", input_boxes_in_h,
                                 input_boxes_in_h, tmp_h_ub, obj.crop_height)
                tik_func_vcomple(tik_instance, "vadd", input_boxes_in_w,
                                 input_boxes_in_w, tmp_w_ub, obj.crop_width)
                tmp_h_ub_int = obj.apply_mem(input_boxes_in_h.shape, "tmp_h_ub_int", tik.scope_ubuf, "int32")
                tmp_w_ub_int = obj.apply_mem(input_boxes_in_w.shape, "tmp_w_ub_int", tik.scope_ubuf, "int32")
                tik_func_vconv(tik_instance, tmp_h_ub_int, tmp_h_ub, obj.crop_height, mode="round")
                tik_func_vconv(tik_instance, tmp_w_ub_int, tmp_w_ub, obj.crop_width, mode="round")
                tik_func_vcomple(tik_instance, "vsub", h_top_index,
                                 h_top_index, tmp_h_ub_int, obj.crop_height)
                tik_func_vcomple(tik_instance, "vsub", w_left_index,
                                 w_left_index, tmp_w_ub_int, obj.crop_width)

        # read input batch index and calc input offset
        input_batch_offset = tik_instance.Scalar(dtype="int32")
        input_batch_offset.set_as(box_index_ub[_box_idx])
        input_batch_offset.set_as(input_batch_offset*obj.image_c1*obj.image_c0*obj.image_height*obj.image_width)
        input_h_offset = tik_instance.Scalar(dtype="int32")
        input_w_offset = tik_instance.Scalar(dtype="int32")
        h_lerp = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        w_lerp = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        with tik_instance.for_range(0, obj.crop_height) as _crop_height_idx:
            input_h_offset.set_as(h_top_index[_crop_height_idx])
            input_h_offset.set_as(input_h_offset * obj.image_width * obj.image_c0)
            h_lerp.set_as(input_boxes_in_h[_crop_height_idx])
            with tik_instance.for_range(0, obj.crop_width) as _crop_width_idx:
                input_w_offset.set_as(w_left_index[_crop_width_idx])
                input_w_offset.set_as(input_w_offset * obj.image_c0)
                w_lerp.set_as(input_boxes_in_w[_crop_width_idx])
                image_gm = obj.input_gm_list[0]
                output_gm = obj.output_gm_list[0]
                # copy all C data in ub
                c0_block_num = obj.image_c0 // obj.block_num
                with tik_instance.new_stmt_scope():
                    h0_w_ub = obj.apply_mem((obj.image_c1*obj.image_c0*2,), "h0_w_ub", tik.scope_ubuf)
                    h1_w_ub = obj.apply_mem((obj.image_c1*obj.image_c0*2,), "h1_w_ub", tik.scope_ubuf)
                    tik_instance.data_move(h0_w_ub,
                                           image_gm[input_batch_offset + input_h_offset + input_w_offset],
                                           0, obj.image_c1, c0_block_num,
                                           obj.image_height*obj.image_width*c0_block_num - c0_block_num, 0)
                    tik_instance.data_move(h0_w_ub[obj.image_c1*obj.image_c0],
                                           image_gm[input_batch_offset + input_h_offset + input_w_offset
                                                    + obj.image_c0],
                                           0, obj.image_c1, c0_block_num,
                                           obj.image_height*obj.image_width*c0_block_num - c0_block_num, 0)
                    tik_instance.data_move(h1_w_ub,
                                           image_gm[input_batch_offset + input_h_offset + input_w_offset
                                                    + obj.image_width * obj.image_c0],
                                           0, obj.image_c1, c0_block_num,
                                           obj.image_height*obj.image_width*c0_block_num - c0_block_num, 0)
                    tik_instance.data_move(h1_w_ub[obj.image_c1*obj.image_c0],
                                           image_gm[input_batch_offset + input_h_offset + input_w_offset
                                                    + obj.image_c0 + obj.image_width * obj.image_c0],
                                           0, obj.image_c1, c0_block_num,
                                           obj.image_height*obj.image_width*c0_block_num - c0_block_num, 0)
                    tik_func_vcomple(tik_instance, "vsub", h1_w_ub,
                                     h1_w_ub, h0_w_ub,
                                     obj.image_c1*obj.image_c0*2)

                    tik_func_vmuls(tik_instance, h1_w_ub,
                                   h1_w_ub, h_lerp, obj.image_c1*obj.image_c0*2)

                    tik_func_vcomple(tik_instance, "vadd", h0_w_ub,
                                     h1_w_ub, h0_w_ub,
                                     obj.image_c1*obj.image_c0*2)

                    tik_func_vcomple(tik_instance, "vsub", h0_w_ub[obj.image_c1*obj.image_c0:],
                                     h0_w_ub[obj.image_c1*obj.image_c0:], h0_w_ub,
                                     obj.image_c1*obj.image_c0)

                    tik_func_vmuls(tik_instance, h0_w_ub[obj.image_c1*obj.image_c0:],
                                   h0_w_ub[obj.image_c1*obj.image_c0:],
                                   w_lerp, obj.image_c1*obj.image_c0)

                    tik_func_vcomple(tik_instance, "vadd", h0_w_ub,
                                     h0_w_ub[obj.image_c1*obj.image_c0:],
                                     h0_w_ub,
                                     obj.image_c1*obj.image_c0)
                    output_offset = \
                        _out_batch_idx*obj.image_c1*obj.crop_width*obj.crop_height*obj.image_c0 \
                        + _crop_height_idx*obj.crop_width*obj.image_c0 + _crop_width_idx*obj.image_c0
                    tik_instance.data_move(output_gm[output_offset],
                                           h0_w_ub, 0, obj.image_c1, c0_block_num,
                                           0, obj.crop_height*obj.crop_width*c0_block_num - c0_block_num)


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_FLOAT,
                 REQUIRED_ATTR_STR, KERNEL_NAME)
def crop_and_resize(x, boxes, box_index, y, crop_size, extrapolation_value,
                    method, kernel_name="crop_and_resize"):
    """
    do crop_and_resize

    Parameters:
    ----------
    x : dict.
        dict info of images value, must include the keys(shape and dtype).
        and shape will be 5HD
    boxes : dict.
        dict info of boxes, a 2D Tensor of type float32 with shape (num_anchors, 4).
        "num_anchors" indicates num of boxes.
        the shape value "4" refers to "y1", "x1", "y2", and "x2".
    box_index : dict.
        dict info of box_index, a 1D Tensor of type int32 with shape (num_anchors).
        "num_anchors" indicates num of boxes. the value indicates the image index for each boxes
    y : dict.
        A 5HD Tensor with shape (batch,).
        specifying output crop image
    crop_size : list.
        A required attribute of type list int, specifying the output image size.
    extrapolation_value : float.
        A required attribute of type float32, specifying the extrapolation_value
    method : string.
        A required attribute of type string, specifying the resize type.
    kernel_name : str.
        cce kernel name, default value is "crop_and_resize"

    Returns
    -------
    tik_instance
    """
    # init object for crop_and_resize
    crop_and_resize_obj = CropAndResize(x, boxes, box_index, crop_size, y, extrapolation_value, method)
    # init gm
    tik_instance = crop_and_resize_obj.get_tik_instance()
    core_used, num_per_core, num_last_core = crop_and_resize_obj.get_core_cut()
    crop_and_resize_obj.init_gm_mem()
    with tik_instance.for_range(0, core_used, block_num=core_used) as _core_idx:
        crop_and_resize_obj.init_ub_scalar()
        max_idx = max(crop_and_resize_obj.crop_height, crop_and_resize_obj.crop_width)
        max_idx = get_ceil_int(max_idx, 16) * 16
        crop_and_resize_obj.index_ub = crop_and_resize_obj.apply_mem([max_idx],
                                                                     "tmp_index_ub", tik.scope_ubuf, "float32")
        fill_index_in_ub(tik_instance, crop_and_resize_obj.index_ub, max_idx)
        if num_per_core == num_last_core or core_used == 1:
            do_crop_and_resize_compute_one_core(num_per_core, crop_and_resize_obj, num_per_core*_core_idx)
        else:
            with tik_instance.if_scope(_core_idx < core_used - 1):
                do_crop_and_resize_compute_one_core(num_per_core, crop_and_resize_obj, num_per_core*_core_idx)
            with tik_instance.else_scope():
                do_crop_and_resize_compute_one_core(num_last_core, crop_and_resize_obj, num_per_core*_core_idx)

    # build tik instance use kernel_name
    crop_and_resize_obj.build_tik_instance(kernel_name)

    return tik_instance

