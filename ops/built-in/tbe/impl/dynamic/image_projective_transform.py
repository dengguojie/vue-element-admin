"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

image_projective_transform
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_common import ceil_div_scalar as ceil_div
from impl import constant_util


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # tiling param num
    TILING_ARG_NUM = 20
    # 8 bit
    EIGHT_BIT = 8
    # divide ub to five part
    ONE_FIFTH_UB = 5
    # divide ub to seven part
    ONE_SEVENTH_UB = 7
    # divide ub to eight part
    ONE_EIGHTH_UB = 8
    # divide ub to nine part
    ONE_NINETH_UB = 9
    # divide ub to four part
    ONE_FOURTH_UB = 4


# 'pylint: disable=too-many-lines,too-many-public-methods,too-many-instance-attributes,too-many-arguments
# 'pylint: disable=too-many-locals, too-many-statements
# 'pylint: disable=attribute-defined-outside-init
class ImageProjectiveTransform:
    """
    Class for Dynamic shape operator ImageProjectiveTransform
    """
    def __init__(self, images_dtype, transform_dtype, interpolation, fill_mode, kernel_name):
        # check interpolation
        if interpolation not in ("NEAREST", "BILINEAR"):
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "interpolation", "NEAREST,BILINEAR",
                                                               interpolation)

        # check fill mode
        if fill_mode not in ("REFLECT", "WRAP", "CONSTANT", "NEAREST"):
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "fill_mode",
                                                               "REFLECT,WRAP,CONSTANT,NEAREST",
                                                               fill_mode)
        self.tik_instance = tik.Tik()
        self.dtype = images_dtype
        self.trans_dtype = transform_dtype
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = tbe_platform.get_bit_len(self.dtype) // Constant.EIGHT_BIT
        self.trans_dtype_size = 4
        self.block_byte_size = 32
        self.all_ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.kernel_name = kernel_name
        self.repeat_size = 256
        self.exist_fill_value_n = False
        self.tiling_gm = None
        self.input_gm = None
        self.transform_gm = None
        self.output_gm = None
        self.output_shape_gm = None
        self.fill_value_gm = None
        self.ub_size_trans = (self.all_ub_size - self.block_byte_size) // Constant.ONE_SEVENTH_UB
        self.one_eighth_ub_fp32 = self.ub_size_trans // (Constant.ONE_EIGHTH_UB * self.trans_dtype_size)
        self.one_eighth_ub_dtype = self.ub_size_trans // (Constant.ONE_NINETH_UB * self.dtype_size)
        self.one_fourth_ub_dtype = self.ub_size_trans // (Constant.ONE_FOURTH_UB * self.dtype_size)
        self.one_seventh_ub_fp32 = self.ub_size_trans // (Constant.ONE_SEVENTH_UB * self.trans_dtype_size)
        self.one_fifth_ub_dtype = self.ub_size_trans // (Constant.ONE_FIFTH_UB * self.dtype_size)
        # NEAREST:0 BILINEAR:1
        self.interpolation = 0 if interpolation == "NEAREST" else 1
        # CONSTANT:0 REFLECT:1 WRAP:2 NEAREST:3
        if fill_mode == "CONSTANT":
            self.fill_mode = 0
        elif fill_mode == "REFLECT":
            self.fill_mode = 1
        elif fill_mode == "WRAP":
            self.fill_mode = 2
        elif fill_mode == "NEAREST":
            self.fill_mode = 3
        self.core_ele = self.tik_instance.Scalar("int32", name="core_ele")
        self.input_x_float = self.tik_instance.Scalar("float32", name="input_x_float")
        self.input_x_int = self.tik_instance.Scalar("int32", name="input_x_int")
        self.input_y_float = self.tik_instance.Scalar("float32", name="input_y_float")
        self.input_y_int = self.tik_instance.Scalar("int32", name="input_y_int")
        self.zero = self.tik_instance.Scalar(images_dtype, name="zero", init_value=0.0)
        self.img_num = self.tik_instance.Scalar("int32", name="img_num")
        self.float_one = self.tik_instance.Scalar("float32", name="float_one", init_value=1.0)
        self.trans_a0 = self.tik_instance.Scalar("float32", name="trans_a0")
        self.trans_a1 = self.tik_instance.Scalar("float32", name="trans_a1")
        self.trans_a2 = self.tik_instance.Scalar("float32", name="trans_a2")
        self.trans_b0 = self.tik_instance.Scalar("float32", name="trans_b0")
        self.trans_b1 = self.tik_instance.Scalar("float32", name="trans_b1")
        self.trans_b2 = self.tik_instance.Scalar("float32", name="trans_b2")
        self.trans_c0 = self.tik_instance.Scalar("float32", name="trans_c0")
        self.trans_c1 = self.tik_instance.Scalar("float32", name="trans_c1")
        self.value_xyfloor = self.tik_instance.Scalar(self.dtype, name="value_xyfloor")
        self.value_y_xceil = self.tik_instance.Scalar(self.dtype, name="value_y_xceil")
        self.value_x_yceil = self.tik_instance.Scalar(self.dtype, name="value_x_yceil")
        self.value_xyceil = self.tik_instance.Scalar(self.dtype, name="value_xyceil")
        self.offset_xyfloor = self.tik_instance.Scalar("int32", name="offset_xyfloor")
        self.offset_y_xceil = self.tik_instance.Scalar("int32", name="offset_y_xceil")
        self.offset_x_yceil = self.tik_instance.Scalar("int32", name="offset_x_yceil")
        self.offset_xyceil = self.tik_instance.Scalar("int32", name="offset_xyceil")
        self.flag_xyfloor = self.tik_instance.Scalar("int32", name="flag_xyfloor")
        self.flag_y_xceil = self.tik_instance.Scalar("int32", name="flag_y_xceil")
        self.flag_x_yceil = self.tik_instance.Scalar("int32", name="flag_x_yceil")
        self.flag_xyceil = self.tik_instance.Scalar("int32", name="flag_xyceil")
        self.img_cal_repeat = self.tik_instance.Scalar("int32", name="img_cal_repeat")
        self.img_cal_left = self.tik_instance.Scalar("int32", name="img_cal_left")
        self.img_cal_left_num = self.tik_instance.Scalar("int32", name="img_cal_left_num")
        self.ub_c_repeat = self.tik_instance.Scalar("int32", name="ub_c_repeat")
        self.ub_c_left = self.tik_instance.Scalar("int32", name="ub_c_left")
        self.ub_c_left_num = self.tik_instance.Scalar("int32", name="ub_c_left_num")
        self.block_size = self.tik_instance.Scalar("int32", name="block_size")
        self.mask = self.tik_instance.Scalar("int32", name="mask")
        self.cal_repeat = self.tik_instance.Scalar("int32", name="cal_repeat")
        self.fill_val = self.tik_instance.Scalar(self.dtype, name="fill_val", init_value=0)

    def img_compute(self):
        """
        excute V2 operator
        """
        self._init_gm_tensor()
        inputs = [self.input_gm, self.transform_gm, self.output_shape_gm]
        if self.exist_fill_value_n is True:
            self.fill_value_gm = self.tik_instance.Tensor(self.dtype, (8,), name="fill_value_gm", scope=tik.scope_gm)
            inputs.append(self.fill_value_gm)
        
        self._run()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": self.one_seventh_ub_fp32,
                "core_num": self.aicore_num,
                "trans_dtype_size": self.trans_dtype_size,
                "block_byte_size": self.block_byte_size
            })
        
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                inputs=inputs,
                                outputs=[self.output_gm],
                                flowtable=[self.tiling_gm],
                                config=opt_config)

    def image_projective_compute(self):
        """
        excute operator
        """
        self._init_gm_tensor()
        self._run()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}

        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": self.one_seventh_ub_fp32,
                "core_num": self.aicore_num,
                "trans_dtype_size": self.trans_dtype_size,
                "block_byte_size": self.block_byte_size
            })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                inputs=[self.input_gm, self.transform_gm, self.output_shape_gm],
                                outputs=[self.output_gm],
                                flowtable=[self.tiling_gm],
                                config=opt_config)

    def _run(self):
        """
        execute_tilling, copy tiling and read
        """
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as core_idx:
            self._set_tiling_param()
            self.block_size.set_as(self.block_byte_size / self.dtype_size)
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.if_scope(self.exist_fill_value_n is True):
                    self.ub_fill = self.tik_instance.Tensor(self.dtype, (8,), name="ub_fill", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(self.ub_fill, self.fill_value_gm, 0, 1, 1, 0, 0)
                    self.fill_val.set_as(self.ub_fill[0])

            with self.tik_instance.if_scope(core_idx < self.act_core_num):
                # images_b less than core num and transform_b is equal to 1 and h*w less than ub
                with self.tik_instance.if_scope(self.tiling_mode == 0):
                    with self.tik_instance.new_stmt_scope():
                        self._move_one_transform()
                        self._one_core_one_image_copy(core_idx)

                # images_b less than core num and transform_b is equal to images_b and h*w less than ub
                with self.tik_instance.if_scope(self.tiling_mode == 1):
                    with self.tik_instance.new_stmt_scope():
                        self._move_transforms(core_idx)
                        self._one_core_one_image_copy(core_idx)

                # images_b more than core num and transform_b is equal to 1 and h*w less than ub
                with self.tik_instance.if_scope(self.tiling_mode == 2):
                    with self.tik_instance.new_stmt_scope():
                        with self.tik_instance.if_scope(core_idx < self.imgnum_cal_left):
                            self.img_num.set_as(self.imgnum_cal_ceil_repeat)
                        with self.tik_instance.else_scope():
                            self.img_num.set_as(self.imgnum_cal_repeat)
                        self._gt_corenum_copy(core_idx)

                # images_b more than core num and transform_b is equal to images_b and h*w less than ub
                with self.tik_instance.if_scope(self.tiling_mode == 3):
                    with self.tik_instance.new_stmt_scope():
                        with self.tik_instance.if_scope(core_idx < self.imgnum_cal_left):
                            self.img_num.set_as(self.imgnum_cal_ceil_repeat)
                        with self.tik_instance.else_scope():
                            self.img_num.set_as(self.imgnum_cal_repeat)
                        self._gt_corenum_copy_transforms(core_idx)

                # images_b more than core num and transform_b is equal to 1 and h*w more than ub
                with self.tik_instance.if_scope(self.tiling_mode == 4):
                    with self.tik_instance.new_stmt_scope():
                        self._move_one_transform()
                        self._gt_ub_copy_one(core_idx)

                # images_b more than core num and transform_b is equal to images_b and h*w more than ub
                with self.tik_instance.if_scope(self.tiling_mode == 5):
                    with self.tik_instance.new_stmt_scope():
                        self._move_transforms(core_idx)
                        self._gt_ub_copy_one(core_idx)

                # images_b less than core num and transform_b is equal to 1 and h*w more than ub
                with self.tik_instance.if_scope(self.tiling_mode == 6):
                    with self.tik_instance.new_stmt_scope():
                        with self.tik_instance.if_scope(core_idx < self.imgnum_cal_left):
                            self.img_num.set_as(self.imgnum_cal_ceil_repeat)
                        with self.tik_instance.else_scope():
                            self.img_num.set_as(self.imgnum_cal_repeat)
                        self._gt_ub_copy(core_idx)

                # images_b less than core num and transform_b is equal to images_b and h*w more than ub
                with self.tik_instance.if_scope(self.tiling_mode == 7):
                    with self.tik_instance.new_stmt_scope():
                        with self.tik_instance.if_scope(core_idx < self.imgnum_cal_left):
                            self.img_num.set_as(self.imgnum_cal_ceil_repeat)
                        with self.tik_instance.else_scope():
                            self.img_num.set_as(self.imgnum_cal_repeat)
                        self._gt_ub_copy_transforms(core_idx)

                # images_b more than core num and transform_b is equal to one
                with self.tik_instance.if_scope(self.tiling_mode == 8):
                    with self.tik_instance.new_stmt_scope():
                        self._move_one_transform()
                        self._one_block_for_one_transform()

                # images_b more than core num and transform_b is equal to images_b
                with self.tik_instance.if_scope(self.tiling_mode == 9):
                    with self.tik_instance.new_stmt_scope():
                        self._one_block_for_transforms()

    def _init_gm_tensor(self):
        """
        Init gm tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.dtype, (constant_util.SHAPE_SIZE_LIMIT,),
                                                 name="input_gm",
                                                 scope=tik.scope_gm)
        self.transform_gm = self.tik_instance.Tensor(self.trans_dtype, (constant_util.SHAPE_SIZE_LIMIT,),
                                                     name="transform_gm",
                                                     scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, (constant_util.SHAPE_SIZE_LIMIT,),
                                                  name="output_gm",
                                                  scope=tik.scope_gm)
        self.output_shape_gm = self.tik_instance.Tensor("int32", (2,), name="output_shape_gm", scope=tik.scope_gm)

    def _init_xy_ub_tensor(self, ub_size):
        """
        Init x y coords tensor
        """
        self.ub_input_x = self.tik_instance.Tensor("float32", (ub_size,), name="ub_input_x", scope=tik.scope_ubuf)
        self.ub_input_y = self.tik_instance.Tensor("float32", (ub_size,), name="ub_input_y", scope=tik.scope_ubuf)

    def _init_ub_tensor(self, ub_size):
        """
        Init ub tensor
        """
        self.ub_images = self.tik_instance.Tensor(self.dtype, (ub_size,), name="ub_images", scope=tik.scope_ubuf)
        self.ub_output = self.tik_instance.Tensor(self.dtype, (ub_size,), name="ub_output", scope=tik.scope_ubuf)
        self.ub_aligned = self.tik_instance.Tensor(self.dtype, (ub_size,), name="ub_aligned", scope=tik.scope_ubuf)

    def _move_tiling_to_ub(self):
        """
        set tiling numger to ub
        """
        tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                             name="tiling_ub",
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, ceil_div(Constant.TILING_ARG_NUM, 8), 0, 0)

        self.tiling_mode.set_as(tiling_ub[0])
        self.act_core_num.set_as(tiling_ub[1])
        self.input_b.set_as(tiling_ub[2])
        self.input_h.set_as(tiling_ub[3])
        self.input_w.set_as(tiling_ub[4])
        self.input_c.set_as(tiling_ub[5])
        self.input_size.set_as(tiling_ub[6])
        self.output_h.set_as(tiling_ub[7])
        self.output_w.set_as(tiling_ub[8])
        self.ub_height.set_as(tiling_ub[9])
        self.ub_repeat_time.set_as(tiling_ub[10])
        self.ub_repeat_left.set_as(tiling_ub[11])
        self.imgnum_cal_repeat.set_as(tiling_ub[12])
        self.imgnum_cal_ceil_repeat.set_as(tiling_ub[13])
        self.imgnum_cal_left.set_as(tiling_ub[14])

    def _set_tiling_param(self):
        """
        _set_tiling_param
        """
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.act_core_num = self.tik_instance.Scalar("int32", name="act_core_num")
        self.input_b = self.tik_instance.Scalar("int32", name="input_b")
        self.input_h = self.tik_instance.Scalar("int32", name="input_h")
        self.input_w = self.tik_instance.Scalar("int32", name="input_w")
        self.input_c = self.tik_instance.Scalar("int32", name="input_c")
        self.output_h = self.tik_instance.Scalar("int32", name="output_h")
        self.output_w = self.tik_instance.Scalar("int32", name="output_w")
        self.ub_height = self.tik_instance.Scalar("int32", name="ub_height")
        self.ub_repeat_time = self.tik_instance.Scalar("int32", name="ub_repeat_time")
        self.ub_repeat_left = self.tik_instance.Scalar("int32", name="ub_repeat_left")
        self.input_size = self.tik_instance.Scalar("int32", name="input_size")
        self.imgnum_cal_repeat = self.tik_instance.Scalar("int32", name="imgnum_cal_repeat")
        self.imgnum_cal_ceil_repeat = self.tik_instance.Scalar("int32", name="imgnum_cal_ceil_repeat")
        self.imgnum_cal_left = self.tik_instance.Scalar("int32", name="imgnum_cal_left")

        with self.tik_instance.new_stmt_scope():
            self._move_tiling_to_ub()

    def _cal_coords(self, ub_size, h_start, h_end, rep, height_rep):
        """
        create two tensor by x and y loop
        calculate transform data with output x and y coords to get input x and y coords
        """
        with self.tik_instance.new_stmt_scope():
            # init some temporary tensor for cal coords
            ub_output_x = self.tik_instance.Tensor("float32", (ub_size,), name="ub_output_x", scope=tik.scope_ubuf)
            ub_output_y = self.tik_instance.Tensor("float32", (ub_size,), name="ub_output_y", scope=tik.scope_ubuf)
            projection_ub = self.tik_instance.Tensor("float32", (ub_size,), name="projection_ub", scope=tik.scope_ubuf)
            ub_tmp = self.tik_instance.Tensor("float32", (ub_size,), name="ub_tmp", scope=tik.scope_ubuf)
            ub_tmp_y = self.tik_instance.Tensor("float32", (ub_size,), name="ub_tmp_y", scope=tik.scope_ubuf)
            rep_idx = self.tik_instance.Scalar("int32", name="rep_idx")
            coord_cal_repeat = self.tik_instance.Scalar("int32", name="coord_cal_repeat")
            coord_cal_ceil_repeat = self.tik_instance.Scalar("int32", name="coord_cal_ceil_repeat")
            coord_cal_left = self.tik_instance.Scalar("int32", name="coord_cal_left")

            # ub could store h*w ele
            with self.tik_instance.if_scope(rep == -1):
                rep_idx.set_as(0)
            # h*w more than ub
            with self.tik_instance.else_scope():
                rep_idx.set_as(rep)

            # a tensor with output y coords from zero to height
            width = self.output_w
            with self.tik_instance.for_range(h_start, h_end) as loop_h_idx:
                # a tensor with output x coords from zero to width
                with self.tik_instance.for_range(0, width) as loop_w_idx:
                    offset_o = (loop_h_idx - rep_idx * height_rep) * width + loop_w_idx
                    ub_output_y[offset_o].set_as(loop_h_idx * 1.0)
                    ub_output_x[offset_o].set_as(loop_w_idx * 1.0)

            self.mask.set_as(self.repeat_size / self.trans_dtype_size)

            with self.tik_instance.if_scope(self.ub_repeat_time == 0):
                coord_cal_repeat.set_as((self.output_h * self.output_w) / self.mask)
                coord_cal_ceil_repeat.set_as(coord_cal_repeat + 1)
                coord_cal_left.set_as((self.output_h * self.output_w) % self.mask)
            with self.tik_instance.else_scope():
                coord_cal_repeat.set_as((self.ub_height * self.output_w) / self.mask)
                coord_cal_ceil_repeat.set_as(coord_cal_repeat + 1)
                coord_cal_left.set_as((self.ub_height * self.output_w) % self.mask)

            with self.tik_instance.if_scope(coord_cal_left == 0):
                self.cal_repeat.set_as(coord_cal_repeat)
            with self.tik_instance.elif_scope(coord_cal_repeat == 0):
                self.cal_repeat.set_as(1)
                self.mask.set_as(coord_cal_left)
            with self.tik_instance.else_scope():
                self.cal_repeat.set_as(coord_cal_ceil_repeat)

            # cal projection `c0 * output_x + c1 * output_y + 1.f`
            self.tik_instance.vec_muls(self.mask, self.ub_input_x, ub_output_x, self.trans_c0, self.cal_repeat, 8, 8)
            self.tik_instance.vec_muls(self.mask, self.ub_input_y, ub_output_y, self.trans_c1, self.cal_repeat, 8, 8)
            self.tik_instance.vec_add(self.mask, self.ub_input_x, self.ub_input_x, self.ub_input_y, self.cal_repeat, 8,
                                      8, 8)
            self.tik_instance.vec_adds(self.mask, projection_ub, self.ub_input_x, self.float_one, self.cal_repeat, 8,
                                      8)

            # cal input_x `(a0 * output_x + a1 * output_y + a2) / projection`
            self.tik_instance.vec_muls(self.mask, ub_tmp, ub_output_x, self.trans_a0, self.cal_repeat, 8, 8)
            self.tik_instance.vec_muls(self.mask, self.ub_input_y, ub_output_y, self.trans_a1, self.cal_repeat, 8, 8)
            self.tik_instance.vec_add(self.mask, self.ub_input_x, ub_tmp, self.ub_input_y, self.cal_repeat, 8, 8, 8)
            self.tik_instance.vec_adds(self.mask, ub_tmp, self.ub_input_x, self.trans_a2, self.cal_repeat, 8, 8)
            self.tik_instance.vdiv(self.mask, self.ub_input_x, ub_tmp, projection_ub, self.cal_repeat, 1, 1, 1, 8, 8,
                                   8)

            # cal input_y `(b0 * output_x + b1 * output_y + b3) / projection`
            self.tik_instance.vec_muls(self.mask, self.ub_input_y, ub_output_y, self.trans_b1, self.cal_repeat, 8, 8)
            self.tik_instance.vec_muls(self.mask, ub_tmp_y, ub_output_x, self.trans_b0, self.cal_repeat, 8, 8)
            self.tik_instance.vec_add(self.mask, self.ub_input_y, ub_tmp_y, self.ub_input_y, self.cal_repeat, 8, 8, 8)
            self.tik_instance.vec_adds(self.mask, ub_tmp_y, self.ub_input_y, self.trans_b2, self.cal_repeat, 8, 8)
            self.tik_instance.vdiv(self.mask, self.ub_input_y, ub_tmp_y, projection_ub, self.cal_repeat, 1, 1, 1, 8,
                                   8, 8)
        with self.tik_instance.new_stmt_scope():
            self._map_coord()

    def _cal_coords_one_block(self, h_start, h_end, rep, height_rep):
        """
        create two tensor by x and y loop
        calculate transform data with output x and y coords to get input x and y coords
        """
        with self.tik_instance.new_stmt_scope():
            # init some temporary tensor for cal coords
            ub_output_x = self.tik_instance.Tensor("float32", (8,), name="ub_output_x", scope=tik.scope_ubuf)
            ub_output_y = self.tik_instance.Tensor("float32", (8,), name="ub_output_y", scope=tik.scope_ubuf)
            projection_ub = self.tik_instance.Tensor("float32", (8,), name="projection_ub", scope=tik.scope_ubuf)
            ub_input_y_tmp = self.tik_instance.Tensor("float32", (8,), name="ub_input_y_tmp", scope=tik.scope_ubuf)
            ub_input_x_tmp = self.tik_instance.Tensor("float32", (8,), name="ub_input_x_tmp", scope=tik.scope_ubuf)
            ub_tmp = self.tik_instance.Tensor("float32", (8,), name="ub_tmp", scope=tik.scope_ubuf)
            ub_tmp_y = self.tik_instance.Tensor("float32", (8,), name="ub_tmp_y", scope=tik.scope_ubuf)
            rep_idx = self.tik_instance.Scalar("int32", name="rep_idx")
            coord_cal_repeat = self.tik_instance.Scalar("int32", name="coord_cal_repeat")
            coord_cal_ceil_repeat = self.tik_instance.Scalar("int32", name="coord_cal_ceil_repeat")
            coord_cal_left = self.tik_instance.Scalar("int32", name="coord_cal_left")

            # ub could store h*w ele
            with self.tik_instance.if_scope(rep == -1):
                rep_idx.set_as(0)
            # h*w more than ub
            with self.tik_instance.else_scope():
                rep_idx.set_as(rep)

            # a tensor with output y coords from zero to height
            width = self.output_w
            with self.tik_instance.for_range(h_start, h_end) as loop_h_idx:
                # a tensor with output x coords from zero to width
                with self.tik_instance.for_range(0, width) as loop_w_idx:
                    offset_o = (loop_h_idx - rep_idx * height_rep) * width + loop_w_idx
                    ub_output_y[offset_o].set_as(loop_h_idx * 1.0)
                    ub_output_x[offset_o].set_as(loop_w_idx * 1.0)

            self.mask.set_as(self.repeat_size / self.trans_dtype_size)

            coord_cal_repeat.set_as(self.input_b * Constant.EIGHT_BIT / self.mask)
            coord_cal_ceil_repeat.set_as(coord_cal_repeat + 1)
            coord_cal_left.set_as(self.input_b * Constant.EIGHT_BIT % self.mask)

            with self.tik_instance.if_scope(coord_cal_left == 0):
                self.cal_repeat.set_as(coord_cal_repeat)
            with self.tik_instance.elif_scope(coord_cal_repeat == 0):
                self.cal_repeat.set_as(1)
            with self.tik_instance.else_scope():
                self.cal_repeat.set_as(coord_cal_ceil_repeat)

            with self.tik_instance.for_range(0, self.input_b) as img_idx:
                with self.tik_instance.new_stmt_scope():
                    self._move_transforms(img_idx)
                offset_s = img_idx * Constant.EIGHT_BIT
                offset_e = Constant.EIGHT_BIT + img_idx * Constant.EIGHT_BIT
                # cal projection `c0 * output_x + c1 * output_y + 1.f`
                self.tik_instance.vec_muls(self.mask, self.ub_input_x[offset_s:offset_e], ub_output_x, self.trans_c0,
                                           1, 8, 8)
                self.tik_instance.vec_muls(self.mask, self.ub_input_y[offset_s:offset_e], ub_output_y, self.trans_c1,
                                           1, 8, 8)
                self.tik_instance.vec_add(self.mask, ub_input_x_tmp, self.ub_input_x[offset_s:offset_e],
                                          self.ub_input_y[offset_s:offset_e], 1, 8, 8, 8)
                self.tik_instance.vec_adds(self.mask, projection_ub, ub_input_x_tmp, self.float_one, 1, 8, 8)

                # cal input_x `(a0 * output_x + a1 * output_y + a2) / projection`
                self.tik_instance.vec_muls(self.mask, ub_tmp, ub_output_x, self.trans_a0, 1, 8, 8)
                self.tik_instance.vec_muls(self.mask, self.ub_input_y[offset_s:offset_e], ub_output_y, self.trans_a1,
                                           1, 8, 8)
                self.tik_instance.vec_add(self.mask, self.ub_input_x[offset_s:offset_e], ub_tmp,
                                          self.ub_input_y[offset_s:offset_e], 1, 8, 8, 8)
                self.tik_instance.vec_adds(self.mask, ub_tmp, self.ub_input_x[offset_s:offset_e], self.trans_a2,
                                           1, 8, 8)
                self.tik_instance.vdiv(self.mask, self.ub_input_x[offset_s:offset_e], ub_tmp, projection_ub,
                                       1, 1, 1, 1, 8, 8, 8)

                # cal input_y `(b0 * output_x + b1 * output_y + b3) / projection`
                self.tik_instance.vec_muls(self.mask, self.ub_input_y[offset_s:offset_e], ub_output_y, self.trans_b1,
                                           1, 8, 8)
                self.tik_instance.vec_muls(self.mask, ub_tmp_y, ub_output_x, self.trans_b0, 1, 8, 8)
                self.tik_instance.vec_add(self.mask, ub_input_y_tmp,
                                          self.ub_input_y[offset_s:offset_e], ub_tmp_y, 1, 8, 8, 8)
                self.tik_instance.vec_adds(self.mask, ub_tmp_y, ub_input_y_tmp, self.trans_b2, 1, 8, 8)
                self.tik_instance.vdiv(self.mask, self.ub_input_y[offset_s:offset_e], ub_tmp_y, projection_ub,
                                       1, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.new_stmt_scope():
            self._map_coord()

    def _map_coord(self):
        """
        do fill mode select
        """
        with self.tik_instance.if_scope(self.fill_mode == 1):
            with self.tik_instance.for_range(0, self.cal_repeat) as repeat_idx:
                self._map_coord_reflect(self.ub_input_x, self.input_w, repeat_idx)
                self._map_coord_reflect(self.ub_input_y, self.input_h, repeat_idx)
        with self.tik_instance.elif_scope(self.fill_mode == 2):
            with self.tik_instance.for_range(0, self.cal_repeat) as repeat_idx:
                self._map_coord_wrap(self.ub_input_x, self.input_w, repeat_idx)
                self._map_coord_wrap(self.ub_input_y, self.input_h, repeat_idx)
        with self.tik_instance.elif_scope(self.fill_mode == 3):
            with self.tik_instance.for_range(0, self.cal_repeat) as repeat_idx:
                self._map_coord_nearest(self.ub_input_x, self.input_w, repeat_idx)
                self._map_coord_nearest(self.ub_input_y, self.input_h, repeat_idx)

    def _map_coord_reflect(self, ub_input, len_hw, loop_repeat):
        """
        cal the map coord for reflect fill mode
        """
        ub_sz2 = self.tik_instance.Tensor("float32", (64,), name="ub_sz2", scope=tik.scope_ubuf)
        ub_one = self.tik_instance.Tensor("float32", (64,), name="ub_one", scope=tik.scope_ubuf)
        ub_neg_input = self.tik_instance.Tensor("float32", (64,), name="ub_neg_input", scope=tik.scope_ubuf)
        ub_input_div_sz2 = self.tik_instance.Tensor("float32", (64,), name="ub_input_div_sz2", scope=tik.scope_ubuf)
        ub_div_mul_sz2 = self.tik_instance.Tensor("float32", (64,), name="ub_div_mul_sz2", scope=tik.scope_ubuf)
        ub_input_add_one = self.tik_instance.Tensor("float32", (64,), name="ub_input_add_one", scope=tik.scope_ubuf)
        ub_div_one = self.tik_instance.Tensor("float32", (64,), name="ub_div_one", scope=tik.scope_ubuf)
        ub_h_cast = self.tik_instance.Tensor("float32", (64,), name="ub_h_cast", scope=tik.scope_ubuf)
        ub_dup_one = self.tik_instance.Tensor("float32", (64,), name="ub_dup_one", scope=tik.scope_ubuf)
        ub_cmp = self.tik_instance.Tensor("float32", (64,), name="ub_cmp", scope=tik.scope_ubuf)
        ub_int = self.tik_instance.Tensor("int32", (64,), name="ub_int", scope=tik.scope_ubuf)
        ub_tmp = self.tik_instance.Tensor("float32", (64,), name="ub_tmp", scope=tik.scope_ubuf)
        ub_sel = self.tik_instance.Tensor("uint16", (64,), name="ub_sel", scope=tik.scope_ubuf)
        ub_compare = self.tik_instance.Tensor("float32", (64,), name="ub_compare", scope=tik.scope_ubuf)
        len_sub_one = self.tik_instance.Scalar("float32", name="len_sub_one")
        neg_len = self.tik_instance.Scalar("float32", name="neg_len")
        len_float = self.tik_instance.Scalar("float32", name="len_float")
        sz2_r = self.tik_instance.Scalar("float32", name="sz2_r")

        len_sub_one.set_as(len_hw - 1)
        neg_len.set_as(len_hw * -1)
        len_float.set_as(len_hw)
        sz2_r.set_as(len_hw * 2)
    
        offset = loop_repeat * self.mask
        self.tik_instance.data_move(ub_tmp, ub_input[offset], 0, 1, 8, 0, 0)

        # select case len <= 1
        with self.tik_instance.if_scope(len_hw <= 1):
            self.tik_instance.vec_dup(self.mask, ub_input[offset], 0, 1, 8)

        with self.tik_instance.else_scope():
            # calculate `sz2 * (int)(-coord / sz2) + coord`
            self.tik_instance.vec_dup(self.mask, ub_sz2, sz2_r, 1, 8)
            self.tik_instance.vec_muls(self.mask, ub_neg_input, ub_tmp, -1, 1, 8, 8)
            self.tik_instance.vdiv(self.mask, ub_input_div_sz2, ub_neg_input, ub_sz2, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.h_cast(ub_int, ub_input_div_sz2, "to-zero")
            self.tik_instance.h_cast(ub_div_mul_sz2, ub_int, "none")
            self.tik_instance.vec_mul(self.mask, ub_input_div_sz2, ub_div_mul_sz2, ub_sz2, 1, 8, 8, 8)
            self.tik_instance.vec_add(self.mask, ub_input_div_sz2, ub_input_div_sz2, ub_tmp, 1, 8, 8, 8)

            # case2 ub_neg_input in_coord + sz2(with)
            self.tik_instance.vec_add(self.mask, ub_neg_input, ub_input_div_sz2, ub_sz2, 1, 8, 8, 8)

            # case3 ub_div_mul_sz2 -in_coord - 1(with)
            self.tik_instance.vec_muls(self.mask, ub_sz2, ub_input_div_sz2, -1, 1, 8, 8)
            self.tik_instance.vec_dup(self.mask, ub_one, 1, 1, 8)
            self.tik_instance.vec_sub(self.mask, ub_div_mul_sz2, ub_sz2, ub_one, 1, 8, 8, 8)

            # case 2 ub_sz2 -in_coord - 1(no)
            self.tik_instance.vec_muls(self.mask, ub_sz2, ub_tmp, -1, 1, 8, 8)
            self.tik_instance.vec_sub(self.mask, ub_sz2, ub_sz2, ub_one, 1, 8, 8, 8)

            # case 1 ub_input_add_one in_coord + sz2(no)
            self.tik_instance.vec_dup(self.mask, ub_one, sz2_r, 1, 8)
            self.tik_instance.vec_add(self.mask, ub_input_add_one, ub_one, ub_tmp, 1, 8, 8, 8)

            # calculate `coord - sz2 * (int)(coord / sz2)`
            self.tik_instance.vdiv(self.mask, ub_div_one, ub_tmp, ub_one, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.h_cast(ub_int, ub_div_one, "to-zero")
            self.tik_instance.h_cast(ub_h_cast, ub_int, "none")
            self.tik_instance.vec_mul(self.mask, ub_h_cast, ub_h_cast, ub_one, 1, 8, 8, 8)
            self.tik_instance.vec_sub(self.mask, ub_div_one, ub_tmp, ub_h_cast, 1, 8, 8, 8)

            # case 6 ub_h_cast (sz2 - coord - 1)
            self.tik_instance.vec_sub(self.mask, ub_h_cast, ub_one, ub_div_one, 1, 8, 8, 8)
            self.tik_instance.vec_dup(self.mask, ub_dup_one, 1, 1, 8)
            self.tik_instance.vec_sub(self.mask, ub_h_cast, ub_h_cast, ub_dup_one, 1, 8, 8, 8)

            # select in_coord < sz2? ub_input_add_one
            self.tik_instance.vec_dup(self.mask, ub_compare, neg_len, 1, 8)
            self.tik_instance.vec_cmpv_lt(ub_sel, ub_input_div_sz2, ub_compare, 1, 8, 8)
            self.tik_instance.vec_sel(self.mask, 0, ub_cmp, ub_sel, ub_neg_input, ub_div_mul_sz2, 1, 8, 8, 8)

            self.tik_instance.vec_cmpv_lt(ub_sel, ub_tmp, ub_compare, 1, 8, 8)
            self.tik_instance.vec_sel(self.mask, 0, ub_neg_input, ub_sel, ub_input_add_one, ub_sz2, 1, 8, 8, 8)

            self.tik_instance.vec_cmpv_lt(ub_sel, ub_tmp, ub_one, 1, 8, 8)
            self.tik_instance.vec_sel(self.mask, 0, ub_input_add_one, ub_sel, ub_cmp, ub_neg_input, 1, 8, 8, 8)

            # select in_coord >= len ? ub_div_mul_sz2
            self.tik_instance.vec_dup(self.mask, ub_compare, len_float, 1, 8)
            self.tik_instance.vec_cmpv_ge(ub_sel, ub_div_one, ub_compare, 1, 8, 8)
            self.tik_instance.vec_sel(self.mask, 0, ub_div_mul_sz2, ub_sel, ub_h_cast, ub_div_one, 1, 8, 8, 8)

            # select ub_cmp
            self.tik_instance.vec_dup(self.mask, ub_compare, len_sub_one, 1, 8)
            self.tik_instance.vec_cmpv_gt(ub_sel, ub_tmp, ub_compare, 1, 8, 8)
            self.tik_instance.vec_sel(self.mask, 0, ub_cmp, ub_sel, ub_div_mul_sz2, ub_tmp, 1, 8, 8, 8)

            # select ub_neg_input
            self.tik_instance.vec_dup(self.mask, ub_compare, 0, 1, 8)
            self.tik_instance.vec_cmpv_lt(ub_sel, ub_tmp, ub_compare, 1, 8, 8)
            self.tik_instance.vec_sel(self.mask, 0, ub_neg_input, ub_sel, ub_input_add_one, ub_cmp, 1, 8, 8, 8)

            # compare 1e-15 with ub_neg_input to select max result ub_sz2
            self.tik_instance.vec_dup(self.mask, ub_one, 1e-15, 1, 8)
            self.tik_instance.vmax(self.mask, ub_sz2, ub_one, ub_neg_input, 1, 1, 1, 1, 8, 8, 8)

            # compare len-1 with ub_neg_input to select min result ub_input_div_sz2
            self.tik_instance.vec_dup(self.mask, ub_one, len_sub_one, 1, 8)
            self.tik_instance.vmin(self.mask, ub_input_div_sz2, ub_one, ub_sz2, 1, 1, 1, 1, 8, 8, 8)

            # move this part result to ub_input
            self.tik_instance.data_move(ub_input[offset], ub_input_div_sz2, 0, 1, 8, 0, 0)

    def _map_coord_wrap(self, ub_input, len_hw, loop_repeat):
        """
        cal the map coord for wrap fill mode
        """
        ub_sz2 = self.tik_instance.Tensor("float32", (64,), name="ub_sz2", scope=tik.scope_ubuf)
        ub_one = self.tik_instance.Tensor("float32", (64,), name="ub_one", scope=tik.scope_ubuf)
        ub_neg_input = self.tik_instance.Tensor("float32", (64,), name="ub_neg_input", scope=tik.scope_ubuf)
        ub_input_div_sz2 = self.tik_instance.Tensor("float32", (64,), name="ub_input_div_sz2", scope=tik.scope_ubuf)
        ub_div_mul_sz2 = self.tik_instance.Tensor("float32", (64,), name="ub_div_mul_sz2", scope=tik.scope_ubuf)
        ub_int = self.tik_instance.Tensor("int32", (64,), name="ub_int", scope=tik.scope_ubuf)
        ub_tmp = self.tik_instance.Tensor("float32", (64,), name="ub_tmp", scope=tik.scope_ubuf)
        ub_sel = self.tik_instance.Tensor("uint16", (64,), name="ub_sel", scope=tik.scope_ubuf)
        ub_compare = self.tik_instance.Tensor("float32", (64,), name="ub_compare", scope=tik.scope_ubuf)
        len_sub_one = self.tik_instance.Scalar("float32", name="len_sub_one")
        len_float = self.tik_instance.Scalar("float32", name="len_float")
        sz2_w = self.tik_instance.Scalar("float32", name="sz2_w")

        len_sub_one.set_as(len_hw - 1)
        len_float.set_as(len_hw)
        sz2_w.set_as(len_hw - 1)
    
        offset = loop_repeat * self.mask
        self.tik_instance.data_move(ub_tmp, ub_input[offset], 0, 1, 8, 0, 0)

        # select case len <= 1
        with self.tik_instance.if_scope(len_hw <= 1):
            self.tik_instance.vec_dup(self.mask, ub_input[offset], 0, 1, 8)

        with self.tik_instance.else_scope():
            # calculate `in_coord + len  * (int)(-coord / sz2) + 1`
            # ub_neg_input
            self.tik_instance.vec_dup(self.mask, ub_one, sz2_w, 1, 8)
            self.tik_instance.vec_dup(self.mask, ub_div_mul_sz2, len_float, 1, 8)
            self.tik_instance.vec_muls(self.mask, ub_sz2, ub_tmp,  -1, 1, 8, 8)
            self.tik_instance.vdiv(self.mask, ub_neg_input, ub_sz2, ub_one, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.h_cast(ub_int, ub_neg_input, "to-zero")
            self.tik_instance.h_cast(ub_sz2, ub_int, "none")
            self.tik_instance.vec_adds(self.mask, ub_input_div_sz2, ub_sz2, 1, 1, 8, 8)
            self.tik_instance.vec_mul(self.mask, ub_input_div_sz2, ub_input_div_sz2, ub_div_mul_sz2, 1, 8, 8, 8)
            self.tik_instance.vec_add(self.mask, ub_neg_input, ub_tmp, ub_input_div_sz2, 1, 8, 8, 8)

            # calculate `in_coord - len  * (int)(-coord / sz2)`
            self.tik_instance.vdiv(self.mask, ub_sz2, ub_tmp, ub_one, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.h_cast(ub_int, ub_sz2, "to-zero")
            self.tik_instance.h_cast(ub_sz2, ub_int, "none")
            self.tik_instance.vec_mul(self.mask, ub_sz2, ub_sz2, ub_div_mul_sz2, 1, 8, 8, 8)
            self.tik_instance.vec_sub(self.mask, ub_one, ub_tmp, ub_sz2, 1, 8, 8, 8)

            # select ub_sz2
            self.tik_instance.vec_dup(self.mask, ub_compare, len_sub_one, 1, 8)
            self.tik_instance.vec_cmpv_gt(ub_sel, ub_tmp, ub_compare, 1, 8, 8)
            self.tik_instance.vec_sel(self.mask, 0, ub_sz2, ub_sel, ub_one, ub_tmp, 1, 8, 8, 8)

            # select ub_one
            self.tik_instance.vec_dup(self.mask, ub_input_div_sz2, 0, 1, 8)
            self.tik_instance.vec_cmpv_lt(ub_sel, ub_tmp, ub_input_div_sz2, 1, 8, 8)
            self.tik_instance.vec_sel(self.mask, 0, ub_one, ub_sel, ub_neg_input, ub_sz2, 1, 8, 8, 8)

            # compare 1e-15 with ub_neg_input to select max result ub_sz2
            self.tik_instance.vec_dup(self.mask, ub_neg_input, 1e-15, 1, 8)
            self.tik_instance.vmax(self.mask, ub_sz2, ub_neg_input, ub_one, 1, 1, 1, 1, 8, 8, 8)

            # compare len-1 with ub_neg_input to select min result ub_input_div_sz2
            self.tik_instance.vmin(self.mask, ub_input_div_sz2, ub_compare, ub_sz2, 1, 1, 1, 1, 8, 8, 8)

            # move this part result to ub_input
            self.tik_instance.data_move(ub_input[offset], ub_input_div_sz2, 0, 1, 8, 0, 0)

    def _map_coord_nearest(self, ub_input, len_hw, loop_repeat):
        """
        cal the map coord for nearest fill mode
        """
        ub_sz2 = self.tik_instance.Tensor("float32", (64,), name="ub_sz2", scope=tik.scope_ubuf)
        ub_one = self.tik_instance.Tensor("float32", (64,), name="ub_one", scope=tik.scope_ubuf)
        ub_neg_input = self.tik_instance.Tensor("float32", (64,), name="ub_neg_input", scope=tik.scope_ubuf)
        ub_tmp = self.tik_instance.Tensor("float32", (64,), name="ub_tmp", scope=tik.scope_ubuf)
        ub_compare = self.tik_instance.Tensor("float32", (64,), name="ub_compare", scope=tik.scope_ubuf)
        len_sub_one = self.tik_instance.Scalar("float32", name="len_sub_one")
        len_sub_one.set_as(len_hw - 1)
    
        offset = loop_repeat * self.mask
        self.tik_instance.data_move(ub_tmp, ub_input[offset], 0, 1, 8, 0, 0)
        # compare 1e-15 with ub_neg_input to select max result ub_sz2
        self.tik_instance.vec_dup(self.mask, ub_neg_input, 1e-15, 1, 8)
        self.tik_instance.vmax(self.mask, ub_sz2, ub_neg_input, ub_tmp, 1, 1, 1, 1, 8, 8, 8)

        # compare len-1 with ub_neg_input to select min result ub_one
        self.tik_instance.vec_dup(self.mask, ub_compare, len_sub_one, 1, 8)
        self.tik_instance.vmin(self.mask, ub_one, ub_compare, ub_sz2, 1, 1, 1, 1, 8, 8, 8)

        # move this part result to ub_input
        self.tik_instance.data_move(ub_input[offset], ub_one, 0, 1, 8, 0, 0)

    def _get_div_round_int(self, float1, float2):
        """
        Get Round Int
        """
        result = self.tik_instance.Scalar("float32", name="_result")
        result.set_as(float1 // float2)
        return result

    def _read_with_fill_value(self, input_x, input_y, input_width, input_height):
        """
        do the selection for coord
        """
        out_of_image = self.tik_instance.Scalar("int32", name="out_of_image")
        with self.tik_instance.if_scope(tik.all(input_x >= 0, input_x < input_width)):
            with self.tik_instance.if_scope(tik.all(input_y >= 0, input_y < input_height)):
                out_of_image.set_as(1)
        with self.tik_instance.else_scope():
            out_of_image.set_as(0)

        return out_of_image

    def _set_xy_value_xyfloor(self):
        """
        set coord xfloor yfloor value
        """
        with self.tik_instance.if_scope(self.flag_xyfloor == 0):
            self.value_xyfloor.set_as(self.fill_val)
        with self.tik_instance.else_scope():
            self.value_xyfloor.set_as(self.ub_images[self.offset_xyfloor])

    def _set_xy_value_yfloor(self):
        """
        set coord xceil yfloor value
        """
        with self.tik_instance.if_scope(self.flag_y_xceil == 0):
            self.value_y_xceil.set_as(self.fill_val)
        with self.tik_instance.else_scope():
            self.value_y_xceil.set_as(self.ub_images[self.offset_y_xceil])

    def _set_xy_value_xfloor(self):
        """
        set coord xfloor yceil value
        """
        with self.tik_instance.if_scope(self.flag_x_yceil == 0):
            self.value_x_yceil.set_as(self.fill_val)
        with self.tik_instance.else_scope():
            self.value_x_yceil.set_as(self.ub_images[self.offset_x_yceil])

    def _set_xy_value_xyceil(self):
        """
        set coord xceil yceil value
        """
        with self.tik_instance.if_scope(self.flag_xyceil == 0):
            self.value_xyceil.set_as(self.fill_val)
        with self.tik_instance.else_scope():
            self.value_xyceil.set_as(self.ub_images[self.offset_xyceil])

    def _set_xy_value_gm_xyfloor(self, core_offset, int_y, int_x, loop_c_idx):
        """
        set coord xfloor yfloor value
        """
        with self.tik_instance.if_scope(self.flag_xyfloor == 0):
            self.value_xyfloor.set_as(self.fill_val)
        with self.tik_instance.else_scope():
            offset_img = core_offset + int_y * self.input_w * self.input_c + int_x * self.input_c + loop_c_idx
            self.tik_instance.data_move(self.ub_images, self.input_gm[offset_img], 0, 1, 1, 0, 0)
            self.value_xyfloor.set_as(self.ub_images[0])

    def _set_xy_value_gm_yfloor(self, core_offset, int_y, int_x, loop_c_idx):
        """
        set coord xceil yfloor value
        """
        with self.tik_instance.if_scope(self.flag_y_xceil == 0):
            self.value_y_xceil.set_as(self.fill_val)
        with self.tik_instance.else_scope():
            offset_img = core_offset + int_y * self.input_w * self.input_c + int_x * self.input_c + loop_c_idx
            self.tik_instance.data_move(self.ub_images, self.input_gm[offset_img], 0, 1, 1, 0, 0)
            self.value_y_xceil.set_as(self.ub_images[0])

    def _set_xy_value_gm_xfloor(self, core_offset, int_y, int_x, loop_c_idx):
        """
        set coord xfloor yceil value
        """
        with self.tik_instance.if_scope(self.flag_x_yceil == 0):
            self.value_x_yceil.set_as(self.fill_val)
        with self.tik_instance.else_scope():
            offset_img = core_offset + int_y * self.input_w * self.input_c + int_x * self.input_c + loop_c_idx
            self.tik_instance.data_move(self.ub_images, self.input_gm[offset_img], 0, 1, 1, 0, 0)
            self.value_x_yceil.set_as(self.ub_images[0])

    def _set_xy_value_gm_xyceil(self, core_offset, int_y, int_x, loop_c_idx):
        """
        set coord xceil yceil value
        """
        with self.tik_instance.if_scope(self.flag_xyceil == 0):
            self.value_xyceil.set_as(self.fill_val)
        with self.tik_instance.else_scope():
            offset_img = core_offset + int_y * self.input_w * self.input_c + int_x * self.input_c + loop_c_idx
            self.tik_instance.data_move(self.ub_images, self.input_gm[offset_img], 0, 1, 1, 0, 0)
            self.value_xyceil.set_as(self.ub_images[0])

    def _bilinear_interpolation(self, core_idx, core_ele, rep_idx, loop_c_idx):
        """
        cal coord value for bilinear interpolation
        """
        # coord_y and coord_x from float to int by floor
        int_yfloor = self.tik_instance.Scalar("int32", name="int_yfloor")
        int_xfloor = self.tik_instance.Scalar("int32", name="int_xfloor")
        floor_y = self.tik_instance.Scalar("float32", name="floor_y")
        floor_x = self.tik_instance.Scalar("float32", name="floor_x")
        int_yceil = self.tik_instance.Scalar("int32", name="int_yceil")
        int_xceil = self.tik_instance.Scalar("int32", name="int_xceil")
        float_yceil = self.tik_instance.Scalar("float32", name="float_yceil")
        float_xceil = self.tik_instance.Scalar("float32", name="float_xceil")
        value_yfloor = self.tik_instance.Scalar("float32", name="value_yfloor")
        value_yceil = self.tik_instance.Scalar("float32", name="value_yceil")
        value_bilinear = self.tik_instance.Scalar("float32", name="value_bilinear")

        self.tik_instance.scalar_conv('floor', int_yfloor, self.input_y_float)
        self.tik_instance.scalar_conv('floor', int_xfloor, self.input_x_float)

        # int coord_y and int coord_x conv to float
        self.tik_instance.scalar_conv('none', floor_y, int_yfloor)
        self.tik_instance.scalar_conv('none', floor_x, int_xfloor)

        # create int y_ceil and int x_ceil
        int_yceil.set_as(int_yfloor + 1)
        int_xceil.set_as(int_xfloor + 1)
        self.tik_instance.scalar_conv('none', float_yceil, int_yceil)
        self.tik_instance.scalar_conv('none', float_xceil, int_xceil)

        self.offset_xyfloor.set_as(int_yfloor * self.input_w * self.input_c + int_xfloor * self.input_c + loop_c_idx)
        self.offset_y_xceil.set_as(int_yfloor * self.input_w * self.input_c + int_xceil * self.input_c + loop_c_idx)
        self.offset_x_yceil.set_as(int_yceil * self.input_w * self.input_c + int_xfloor * self.input_c + loop_c_idx)
        self.offset_xyceil.set_as(int_yceil * self.input_w * self.input_c + int_xceil * self.input_c + loop_c_idx)

        self.flag_xyfloor = self._read_with_fill_value(int_xfloor, int_yfloor, self.input_w, self.input_h)
        self.flag_y_xceil = self._read_with_fill_value(int_xceil, int_yfloor, self.input_w, self.input_h)
        self.flag_x_yceil = self._read_with_fill_value(int_xfloor, int_yceil, self.input_w, self.input_h)
        self.flag_xyceil = self._read_with_fill_value(int_xceil, int_yceil, self.input_w, self.input_h)

        with self.tik_instance.if_scope(rep_idx == -1):
            self._set_xy_value_xyfloor()
            self._set_xy_value_yfloor()
            self._set_xy_value_xfloor()
            self._set_xy_value_xyceil()

        with self.tik_instance.else_scope():
            self._set_xy_value_gm_xyfloor(core_idx * core_ele, int_yfloor, int_xfloor, loop_c_idx)
            self._set_xy_value_gm_yfloor(core_idx * core_ele, int_yfloor, int_xceil, loop_c_idx)
            self._set_xy_value_gm_xfloor(core_idx * core_ele, int_yceil, int_xfloor, loop_c_idx)
            self._set_xy_value_gm_xyceil(core_idx * core_ele, int_yceil, int_xceil, loop_c_idx)

        value_yfloor.set_as((float_xceil - self.input_x_float) * self.value_xyfloor +
                            (self.input_x_float - floor_x) * self.value_y_xceil)
        value_yceil.set_as((float_xceil - self.input_x_float) * self.value_x_yceil +
                           (self.input_x_float - floor_x) * self.value_xyceil)
        value_bilinear.set_as((float_yceil - self.input_y_float) * value_yfloor +
                              (self.input_y_float - floor_y) * value_yceil)

        return value_bilinear

    def _copy_only_process(self, core_idx, core_ele, mode, block_flag):
        """
        Only execute case that images_b is less than aicore number
        """
        with self.tik_instance.for_range(0, self.output_h) as loop_h_idx:
            with self.tik_instance.for_range(0, self.output_w) as loop_w_idx:
                # do fill mode select
                offset_o = loop_h_idx * self.output_w + loop_w_idx + (block_flag * core_idx * Constant.EIGHT_BIT)
                self.input_x_float.set_as(self.ub_input_x[offset_o])
                self.input_y_float.set_as(self.ub_input_y[offset_o])
                offset_mode = (mode * core_idx * self.output_h * self.output_w * self.input_c)
                offset_out = offset_mode + loop_h_idx * self.output_w * self.input_c + loop_w_idx * self.input_c
                # do interpolation select
                with self.tik_instance.if_scope(self.interpolation == 0):
                    self.tik_instance.scalar_conv('round', self.input_x_int, self.input_x_float)
                    self.tik_instance.scalar_conv('round', self.input_y_int, self.input_y_float)
                    offset_in = self.input_y_int * self.input_w * self.input_c + self.input_x_int * self.input_c
                    with self.tik_instance.if_scope(tik.all(self.input_x_int >= 0, self.input_x_int < self.input_w)):
                        with self.tik_instance.if_scope(tik.all(self.input_y_int >= 0,
                                                                self.input_y_int < self.input_h)):
                            with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                                self.ub_output[offset_out + loop_c_idx].set_as(self.ub_images[offset_in + loop_c_idx])
                        with self.tik_instance.else_scope():
                            with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                                self.ub_output[offset_out + loop_c_idx].set_as(self.fill_val)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                            self.ub_output[offset_out + loop_c_idx].set_as(self.fill_val)

                with self.tik_instance.else_scope():
                    rep_idx = -1
                    with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                        value_bilinear = self._bilinear_interpolation(core_idx, core_ele, rep_idx, loop_c_idx)
                        self.ub_output[offset_out + loop_c_idx].set_as(value_bilinear)

    def _copy_only_process_ubs(self, img_idx, core_ele, rep_idx, ub_height):
        """
        Only execute case that h*w is more than ub
        """
        with self.tik_instance.for_range(0, ub_height) as loop_h_idx:
            with self.tik_instance.for_range(0, self.output_w) as loop_w_idx:
                # do fill mode select
                offset_o = loop_h_idx * self.output_w + loop_w_idx
                self.input_x_float.set_as(self.ub_input_x[offset_o])
                self.input_y_float.set_as(self.ub_input_y[offset_o])
                block_ele = self.block_byte_size / self.dtype_size
                self.ub_c_repeat.set_as(self.input_c / block_ele)
                offset_out = loop_w_idx * self.input_c
                # do interpolation select
                with self.tik_instance.if_scope(self.interpolation == 0):
                    self.tik_instance.scalar_conv('round', self.input_x_int, self.input_x_float)
                    self.tik_instance.scalar_conv('round', self.input_y_int, self.input_y_float)
                    with self.tik_instance.if_scope(tik.all(self.input_x_int >= 0, self.input_x_int < self.input_w)):
                        with self.tik_instance.if_scope(tik.all(self.input_y_int >= 0,
                                                                self.input_y_int < self.input_h)):
                            offset_h = self.input_y_int * self.input_w * self.input_c
                            offset_img = img_idx * core_ele + offset_h + self.input_x_int * self.input_c
                            self.tik_instance.data_move(self.ub_images, self.input_gm[offset_img], 0,
                                                        self.ub_c_repeat + 1, 1, 0, 0)
                            with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                                self.ub_output[offset_out + loop_c_idx].set_as(self.ub_images[loop_c_idx])
                        with self.tik_instance.else_scope():
                            with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                                self.ub_output[offset_out + loop_c_idx].set_as(self.fill_val)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                            self.ub_output[offset_out + loop_c_idx].set_as(self.fill_val)

                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                        value_bilinear = self._bilinear_interpolation(img_idx, core_ele, rep_idx, loop_c_idx)
                        self.ub_output[offset_out + loop_c_idx].set_as(value_bilinear)

                # move one w*c to output_gm
                self.ub_c_repeat.set_as(self.input_c * self.output_w / block_ele)
                self.ub_c_left.set_as(self.input_c * self.output_w % block_ele)
                self.ub_c_left_num.set_as(block_ele -  self.input_c * self.output_w % block_ele)

                core_ele = self.output_h * self.output_w * self.input_c
                offset_in = self.ub_c_repeat * block_ele
                offset_images = img_idx * core_ele + (loop_h_idx + rep_idx *
                                                      self.ub_height ) * self.output_w * self.input_c
                offset_out = offset_images + offset_in - self.ub_c_left_num

                with self.tik_instance.if_scope(self.ub_c_left == 0):
                    with self.tik_instance.if_scope(self.ub_c_repeat > 0):
                        self.tik_instance.data_move(self.output_gm[offset_images], self.ub_output, 0, self.ub_c_repeat,
                                                    1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.ub_aligned, self.ub_output, 0, self.ub_c_repeat, 1, 0, 0)

                    with self.tik_instance.for_range(0, self.ub_c_left_num) as dup_idx:
                        rec_idx = self.ub_c_left_num - dup_idx
                        self.ub_aligned[offset_in + dup_idx].set_as(self.ub_output[offset_in - rec_idx])

                    with self.tik_instance.for_range(0, block_ele - self.ub_c_left_num) as data_idx:
                        self.ub_aligned[offset_in + self.ub_c_left_num + data_idx].set_as(
                            self.ub_output[offset_in + data_idx])

                    self.tik_instance.data_move(self.output_gm[offset_images], self.ub_aligned, 0, self.ub_c_repeat, 1,
                                                0, 0)
                    self.tik_instance.data_move(self.output_gm[offset_out], self.ub_aligned[offset_in], 0, 1, 1, 0, 0)

    def _move_image_to_ub(self, offset, repeat, left):
        """
        move image form gm to ub
        """
        with self.tik_instance.if_scope(left == 0):
            self.tik_instance.data_move(self.ub_images, self.input_gm[offset], 0, repeat, 1, 0, 0)

        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.ub_images, self.input_gm[offset], 0, repeat + 1, 1, 0, 0)

    def _move_ub_image_to_gm(self, left, repeat, left_num, offset_img):
        """
        move image form ub to output gm
        """
        block_ele = self.block_byte_size / self.dtype_size
        offset_in = self.img_cal_repeat * (self.block_byte_size / self.dtype_size)
        offset_out = offset_img + offset_in - left

        with self.tik_instance.if_scope(left_num == 0):
            with self.tik_instance.if_scope(repeat > 0):
                self.tik_instance.data_move(self.output_gm[offset_img], self.ub_output, 0, repeat, 1, 0, 0)

        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(repeat > 0):
                self.tik_instance.data_move(self.ub_aligned, self.ub_output, 0, repeat, 1, 0, 0)

                with self.tik_instance.for_range(0, left) as dup_idx:
                    rec_idx = left - dup_idx
                    self.ub_aligned[offset_in + dup_idx].set_as(self.ub_output[offset_in - rec_idx])

                with self.tik_instance.for_range(0, block_ele - left) as data_idx:
                    self.ub_aligned[offset_in + left + data_idx].set_as(self.ub_output[offset_in + data_idx])

                self.tik_instance.data_move(self.output_gm[offset_img], self.ub_aligned, 0, repeat, 1, 0, 0)
                self.tik_instance.data_move(self.output_gm[offset_out], self.ub_aligned[offset_in], 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.output_gm[offset_img], self.ub_output, 0, 1, 1, 0, 0)

    def _cal_img_param(self):
        """
        calculate parameters about images
        """
        self.img_cal_repeat.set_as((self.input_h * self.input_w * self.input_c) / self.block_size)
        self.img_cal_left.set_as(self.block_size - (self.input_h * self.input_w * self.input_c) % self.block_size)
        self.img_cal_left_num.set_as((self.input_h * self.input_w * self.input_c) % self.block_size)

    def _cal_gm_img_param(self):
        """
        calculate parameters about images
        """
        self.img_cal_repeat.set_as((self.output_h * self.output_w * self.input_c) / self.block_size)
        self.img_cal_left.set_as(self.block_size - (self.output_h * self.output_w * self.input_c) % self.block_size)
        self.img_cal_left_num.set_as((self.output_h * self.output_w * self.input_c) % self.block_size)

    def _one_core_one_image_copy(self, core_idx):
        """
        ub could store one picture size
        """
        core_ele = self.input_size
        self._init_xy_ub_tensor(self.one_seventh_ub_fp32)
        self._cal_coords(self.one_seventh_ub_fp32, 0, self.output_h, -1, 0)
        self._init_ub_tensor(self.one_fifth_ub_dtype)

        self._cal_img_param()
        offset_images = core_idx * core_ele

        self._move_image_to_ub(offset_images, self.img_cal_repeat, self.img_cal_left_num)
        offset_image = core_idx * self.output_h * self.output_w * self.input_c

        self._cal_gm_img_param()
        self._copy_only_process(core_idx, core_ele, 0, 0)
        self._move_ub_image_to_gm(self.img_cal_left, self.img_cal_repeat, self.img_cal_left_num, offset_image)

    def _gt_corenum_copy(self, core_idx):
        """
        images number is more than aicore number
        """
        with self.tik_instance.for_range(0, self.img_num) as img_num_idx:
            img_idx = core_idx + img_num_idx * 32
            core_ele = self.input_size
            self._init_xy_ub_tensor(self.one_seventh_ub_fp32)
            self._move_one_transform()
            self._cal_coords(self.one_seventh_ub_fp32, 0, self.output_h, -1, 0)
            self._init_ub_tensor(self.one_fifth_ub_dtype)

            self._cal_img_param()
            offset_images = img_idx * core_ele
            self._move_image_to_ub(offset_images, self.img_cal_repeat, self.img_cal_left_num)

            self._cal_gm_img_param()
            offset_image = img_idx * self.output_h * self.output_w * self.input_c

            self._copy_only_process(img_idx, core_ele, 0, 0)
            self._move_ub_image_to_gm(self.img_cal_left, self.img_cal_repeat, self.img_cal_left_num, offset_image)

    def _gt_corenum_copy_transforms(self, core_idx):
        """
        images number is more than aicore number, and transforms_b is equal to images_b
        """
        with self.tik_instance.for_range(0, self.img_num) as img_num_idx:
            img_idx = core_idx + img_num_idx * 32
            core_ele = self.input_size
            self._init_xy_ub_tensor(self.one_seventh_ub_fp32)
            self._move_transforms(img_idx)
            self._cal_coords(self.one_seventh_ub_fp32, 0, self.output_h, -1, 0)
            self._init_ub_tensor(self.one_fifth_ub_dtype)

            self._cal_img_param()
            offset_images = img_idx * core_ele
            self._move_image_to_ub(offset_images, self.img_cal_repeat, self.img_cal_left_num)

            self._cal_gm_img_param()
            offset_image = img_idx * self.output_h * self.output_w * self.input_c

            self._copy_only_process(img_idx, core_ele, 0, 0)
            self._move_ub_image_to_gm(self.img_cal_left, self.img_cal_repeat, self.img_cal_left_num, offset_image)

    def _gt_ub_copy(self, core_idx):
        """
        To store images h*w need several times.
        transforms_b is one, and images_b is more than core num
        """
        with self.tik_instance.for_range(0, self.img_num) as img_num_idx:
            img_idx = core_idx + img_num_idx * 32
            core_ele = self.input_size
            self._init_xy_ub_tensor(self.one_seventh_ub_fp32)
            self._move_one_transform()

            with self.tik_instance.for_range(0, self.ub_repeat_time + 1) as rep_idx:
                h_start = 0 + rep_idx * self.ub_height
                with self.tik_instance.if_scope(rep_idx == self.ub_repeat_time):
                    h_end = self.ub_repeat_left + rep_idx * self.ub_height
                    self._cal_coords(self.one_seventh_ub_fp32, h_start, h_end, rep_idx, self.ub_height)
                with self.tik_instance.else_scope():
                    h_end = self.ub_height + rep_idx * self.ub_height
                    self._cal_coords(self.one_seventh_ub_fp32, h_start, h_end, rep_idx, self.ub_height)

                self._init_ub_tensor(self.one_fifth_ub_dtype)
                with self.tik_instance.if_scope(rep_idx == self.ub_repeat_time):
                    self._copy_only_process_ubs(img_idx, core_ele, rep_idx, self.ub_repeat_left)
                with self.tik_instance.else_scope():
                    self._copy_only_process_ubs(img_idx, core_ele, rep_idx, self.ub_height)

    def _gt_ub_copy_transforms(self, core_idx):
        """
        To store images h*w need several times.
        transforms_b is images_b, and images_b is more than core num
        """
        with self.tik_instance.for_range(0, self.img_num) as img_num_idx:
            img_idx = core_idx + img_num_idx * 32
            core_ele = self.input_size
            self._init_xy_ub_tensor(self.one_seventh_ub_fp32)
            self._move_transforms(img_idx)

            with self.tik_instance.for_range(0, self.ub_repeat_time + 1) as rep_idx:
                h_start = 0 + rep_idx * self.ub_height
                with self.tik_instance.if_scope(rep_idx == self.ub_repeat_time):
                    h_end = self.ub_repeat_left + rep_idx * self.ub_height
                    self._cal_coords(self.one_seventh_ub_fp32, h_start, h_end, rep_idx, self.ub_height)
                with self.tik_instance.else_scope():
                    h_end = self.ub_height + rep_idx * self.ub_height
                    self._cal_coords(self.one_seventh_ub_fp32, h_start, h_end, rep_idx, self.ub_height)

                self._init_ub_tensor(self.one_fifth_ub_dtype)
                with self.tik_instance.if_scope(rep_idx == self.ub_repeat_time):
                    self._copy_only_process_ubs(img_idx, core_ele, rep_idx, self.ub_repeat_left)
                with self.tik_instance.else_scope():
                    self._copy_only_process_ubs(img_idx, core_ele, rep_idx, self.ub_height)

    def _gt_ub_copy_one(self, core_idx):
        """
        To store images h*w need several times.
        transforms_b is one, and images_b is less than core num
        """
        core_ele = self.input_size
        self._init_xy_ub_tensor(self.one_seventh_ub_fp32)

        with self.tik_instance.for_range(0, self.ub_repeat_time + 1) as rep_idx:
            h_start = 0 + rep_idx * self.ub_height
            with self.tik_instance.if_scope(rep_idx == self.ub_repeat_time):
                h_end = self.ub_repeat_left + rep_idx * self.ub_height
                self._cal_coords(self.one_seventh_ub_fp32, h_start, h_end, rep_idx, self.ub_height)
            with self.tik_instance.else_scope():
                h_end = self.ub_height + rep_idx * self.ub_height
                self._cal_coords(self.one_seventh_ub_fp32, h_start, h_end, rep_idx, self.ub_height)

            self._init_ub_tensor(self.one_fifth_ub_dtype)
            with self.tik_instance.if_scope(rep_idx == self.ub_repeat_time):
                self._copy_only_process_ubs(core_idx, core_ele, rep_idx, self.ub_repeat_left)
            with self.tik_instance.else_scope():
                self._copy_only_process_ubs(core_idx, core_ele, rep_idx, self.ub_height)

    def _one_block_for_one_transform(self):
        """
        one core to deal with case "h*w*c is less than 1 block"
        """
        core_ele = self.input_size

        self._init_xy_ub_tensor(self.one_seventh_ub_fp32)
        self._cal_coords(self.one_seventh_ub_fp32, 0, self.output_h, -1, 0)
        self._init_ub_tensor(self.one_fifth_ub_dtype)

        self._cal_img_param()
        with self.tik_instance.for_range(0, self.input_b) as img_idx:
            offset_images = img_idx * core_ele
            self._move_image_to_ub(offset_images, self.img_cal_repeat, self.img_cal_left_num)
            self._copy_only_process(img_idx, core_ele, 1, 0)

        self.img_cal_repeat.set_as((self.input_b * self.output_h * self.output_w * self.input_c) / self.block_size)
        self.img_cal_left.set_as(self.block_size -
                                 (self.input_b * self.output_h * self.output_w * self.input_c) % self.block_size)
        self.img_cal_left_num.set_as((self.input_b * self.output_h * self.output_w * self.input_c) % self.block_size)

        self._move_ub_image_to_gm(self.img_cal_left, self.img_cal_repeat, self.img_cal_left_num, 0)

    def _one_block_for_transforms(self):
        """
        one core to deal with case "h*w*c is less than 1 block"
        images_b is equal to transform_b
        """
        core_ele = self.input_size

        self._init_xy_ub_tensor(self.one_eighth_ub_fp32)
        self._cal_coords_one_block(0, self.output_h, -1, 0)
        self._init_ub_tensor(self.one_fifth_ub_dtype)

        self._cal_img_param()
        with self.tik_instance.for_range(0, self.input_b) as img_idx:
            offset_images = img_idx * core_ele
            self._move_image_to_ub(offset_images, self.img_cal_repeat, self.img_cal_left_num)
            self._copy_only_process(img_idx, core_ele, 1, 1)

        self.img_cal_repeat.set_as((self.input_b * self.output_h * self.output_w * self.input_c) / self.block_size)
        self.img_cal_left.set_as(self.block_size -
                                 (self.input_b * self.output_h * self.output_w * self.input_c) % self.block_size)
        self.img_cal_left_num.set_as((self.input_b * self.output_h * self.output_w * self.input_c) % self.block_size)

        self._move_ub_image_to_gm(self.img_cal_left, self.img_cal_repeat, self.img_cal_left_num, 0)

    def _move_transforms(self, core_idx):
        """
        transform_b is equal to images_b
        """
        with self.tik_instance.new_stmt_scope():
            # move transform params from gm to ub
            self.ub_transforms = self.tik_instance.Tensor(self.trans_dtype, (8,),
                                                        name="ub_transforms",
                                                        scope=tik.scope_ubuf)
            offset_trans = core_idx * 8
            self.tik_instance.data_move(self.ub_transforms, self.transform_gm[offset_trans], 0, 1, 1, 0, 0)

            # scalar for store transforms data
            self.trans_a0.set_as(self.ub_transforms[0])
            self.trans_a1.set_as(self.ub_transforms[1])
            self.trans_a2.set_as(self.ub_transforms[2])
            self.trans_b0.set_as(self.ub_transforms[3])
            self.trans_b1.set_as(self.ub_transforms[4])
            self.trans_b2.set_as(self.ub_transforms[5])
            self.trans_c0.set_as(self.ub_transforms[6])
            self.trans_c1.set_as(self.ub_transforms[7])

    def _move_one_transform(self):
        """
        transform_b is equal to one
        """
        with self.tik_instance.new_stmt_scope():
            # move transform params from gm to ub
            self.ub_one_transform = self.tik_instance.Tensor(self.trans_dtype, (8,),
                                                            name="ub_one_transform",
                                                            scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.ub_one_transform, self.transform_gm, 0, 1, 1, 0, 0)

            # scalar for store transforms data
            self.trans_a0.set_as(self.ub_one_transform[0])
            self.trans_a1.set_as(self.ub_one_transform[1])
            self.trans_a2.set_as(self.ub_one_transform[2])
            self.trans_b0.set_as(self.ub_one_transform[3])
            self.trans_b1.set_as(self.ub_one_transform[4])
            self.trans_b2.set_as(self.ub_one_transform[5])
            self.trans_c0.set_as(self.ub_one_transform[6])
            self.trans_c1.set_as(self.ub_one_transform[7])


# 'pylint: disable=unused-argument
@register_operator("ImageProjectiveTransform")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def image_projective_transform(images,
                               transforms,
                               output_shape,
                               transformed_image,
                               interpolation,
                               fill_mode="CONSTANT",
                               kernel_name="image_projective_transform"):
    """
    Generate arg_min operator use arg_min

    Parameters
    ----------
    images: dict
        data of input, support "float16", "float32", "uint8", "int32".
    transforms: dict
        3 x 3 projective transformation matrix, support "float32".
    output_shape: dict
        shape of output, support "int32".
    interpolation: str
        interpolation method, support "NEAREST" or "BILINEAR".
    fill_mode: str
        An optional string, Default is "CONSTANT", support "REFLECT", "WRAP", "NEAREST" or "CONSTANT".
    y: dict
        index of output.
    kernel_name: str
        kernel name, default value is "image_projective_transform"

    Returns
    -------
    tik_instance
    """
    images_dtype = images.get("dtype").lower()
    transforms_dtype = transforms.get("dtype").lower()
    output_shape_dtype = output_shape.get("dtype").lower()

    # check input shape, format and dtype
    para_check.check_dtype(images_dtype, ("float16", "float32", "uint8", "int32"), param_name="images")
    para_check.check_dtype(transforms_dtype, ("float32",), param_name="transforms")
    para_check.check_dtype(output_shape_dtype, ("int32",), param_name="output_shape")

    obj = ImageProjectiveTransform(images_dtype, transforms_dtype, interpolation, fill_mode, kernel_name)

    return obj.image_projective_compute()
