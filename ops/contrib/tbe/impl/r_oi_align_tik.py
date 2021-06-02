# -*- coding:utf-8 -*-
from . import get_version
import numpy as np

tik, TBE_VERSION = get_version.get_tbe_version()


class ROIAlignTIK(object):
    def __init__(self, feature_map_shape, roi_shape, ground_size, spatial_scale, pooled_w_reciprocal,
                 pooled_h_reciprocal, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.kernel_name = kernel_name
        self.roi_shape = roi_shape
        self.feature_map_shape = feature_map_shape
        self.n_roi, self.c1_roi, self.h_roi, self.w_roi = roi_shape[0], roi_shape[1], roi_shape[2], roi_shape[3]
        self.align_roi = ((roi_shape[0] + 15) // 16) * 16

        self.n_feature_map, self.c1_feature_map, self.h_feature_map, self.w_feature_map = \
            feature_map_shape[0], feature_map_shape[1], feature_map_shape[2], feature_map_shape[3]

        if ground_size <= 0 or ground_size > self.h_feature_map or ground_size > self.w_feature_map:
            raise RuntimeError("ground_size used to set pooled_h and pooled_w, \
                make sure larger than 0 and less than feature_map's height and width")

        self.pooled_h, self.pooled_w = ground_size, ground_size

        # when roi num equals to 304, pooled_h equals to pooled_w equals to 8,
        # will use 242.22KB ubuf space and 988KB L1 space
        if self.n_roi * self.pooled_h * self.pooled_w >= 304 * 8 * 8:
            raise RuntimeError("the size of roi data is too large, since avaiable ub and L1 buffer size is limited")

        self.spatial_scale = spatial_scale
        self.pooled_w_reciprocal, self.pooled_h_reciprocal = pooled_w_reciprocal, pooled_h_reciprocal

        self.shape_align = ((ground_size * ground_size + 15) // 16) * 16
        # refer to 4080 // 16 = 255, repeat times lesser than 255
        if self.shape_align >= 4080:
            raise RuntimeError("ground_size too larger ,must small than 63!")

        # memory size should be limited on platform 310
        # avaiable ub buffer size is 253852 Bytes(248KB),
        # avaiable L1 buffer size is 1048576 Bytes(1024KB == 1MB)
        self.roi_gm = self.tik_instance.Tensor("float16", (self.n_roi, self.c1_roi, self.h_roi, self.w_roi, 16),
                                               name="roi_gm", scope=tik.scope_gm)
        self.feature_map_gm = self.tik_instance.Tensor("float16", (self.n_feature_map, self.c1_feature_map,
                                                                   self.h_feature_map, self.w_feature_map, 16),
                                                       name="feature_map_gm", scope=tik.scope_gm)
        self.roi_align_gm = self.tik_instance.Tensor("float16", (self.n_roi, self.c1_feature_map, self.pooled_h,
                                                                 self.pooled_h, 16), name="roi_align_output_gm",
                                                     scope=tik.scope_gm)

        print("pooled_h : ", self.pooled_h)
        print("pooled_w : ", self.pooled_w)
        print("spatial_scale : ", self.spatial_scale)
        print("pooled_w_reciprocal : ", self.pooled_w_reciprocal)
        print("pooled_h_reciprocal : ", self.pooled_h_reciprocal)
        print("self.n_feature_map : ", self.n_feature_map)
        print("self.c1_feature_map : ", self.c1_feature_map)
        print("self.h_feature_map : ", self.h_feature_map)
        print("self.w_feature_map : ", self.w_feature_map)

    def floor_f32toi32(self, ub_ret, ub_in, data_len):
        with self.tik_instance.new_stmt_scope():
            repeat_time = data_len // 16
            ub_functmp_f16 = self.tik_instance.Tensor("float16", (data_len,), name="ub_functmp_f16",
                                                      scope=tik.scope_ubuf)
            ub_functmp_f32 = self.tik_instance.Tensor("float32", (data_len,), name="ub_functmp_f32",
                                                      scope=tik.scope_ubuf)
            ub_functmp2_f32 = self.tik_instance.Tensor("float32", (data_len,), name="ub_functmp2_f32",
                                                       scope=tik.scope_ubuf)
            ub_functmp_i32 = self.tik_instance.Tensor("int32", (data_len,), name="ub_functmp_i32",
                                                      scope=tik.scope_ubuf)
            ub_16_f32_val0 = self.tik_instance.Tensor("float32", (data_len,), name="ub_16_f32_val0",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(16, ub_16_f32_val0, 0.0, repeat_time, 1, 2)

            self.tik_instance.vadds(16, ub_functmp2_f32, ub_in, 0.5, repeat_time, 1, 1, 2, 2)
            self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp2_f32, repeat_time, 1, 1, 1, 2)
            self.tik_instance.vconv(16, "floor", ub_functmp_i32, ub_functmp_f16, repeat_time, 1, 1, 2, 1)
            self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp_i32, repeat_time, 1, 1, 1, 2, 1.0)
            self.tik_instance.vconv(16, "none", ub_functmp2_f32, ub_functmp_f16, repeat_time, 1, 1, 2, 1)
            self.tik_instance.vsub(16, ub_functmp2_f32, ub_in, ub_functmp2_f32, repeat_time, 1, 1, 1, 2, 2, 2)

            self.tik_instance.vmin(16, ub_functmp2_f32, ub_functmp2_f32, ub_16_f32_val0, repeat_time,
                                   1, 1, 1, 2, 2, 2)
            self.tik_instance.vrec(16, ub_functmp_f32, ub_functmp2_f32, repeat_time, 1, 1, 2, 2)
            self.tik_instance.vabs(16, ub_functmp_f32, ub_functmp_f32, repeat_time, 1, 1, 2, 2)
            self.tik_instance.vmul(16, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, repeat_time,
                                   1, 1, 1, 2, 2, 2)
            self.tik_instance.vrec(16, ub_functmp_f32, ub_functmp2_f32, repeat_time, 1, 1, 2, 2)
            self.tik_instance.vabs(16, ub_functmp_f32, ub_functmp_f32, repeat_time, 1, 1, 2, 2)
            self.tik_instance.vmul(16, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, repeat_time,
                                   1, 1, 1, 2, 2, 2)
            # add 0.5 to make sure result of vconv precisely
            self.tik_instance.vadds(16, ub_functmp2_f32, ub_functmp2_f32, 0.5, repeat_time, 1, 1, 2, 2)
            self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp2_f32, repeat_time, 1, 1, 1, 2)
            self.tik_instance.vconv(16, "floor", ub_ret, ub_functmp_f16, repeat_time, 1, 1, 2, 1)
            self.tik_instance.vadd(16, ub_ret, ub_ret, ub_functmp_i32, repeat_time, 1, 1, 1, 2, 2, 2)

    def _calc_roi_width_height(self):
        with self.tik_instance.for_range(0, self.n_roi) as i:
            self.rois_start_w[i].set_as(self.roi_ub[i, 1])
            self.rois_start_h[i].set_as(self.roi_ub[i, 2])
            self.rois_end_w[i].set_as(self.roi_ub[i, 3])
            self.rois_end_h[i].set_as(self.roi_ub[i, 4])

        # compare roi with 0, pick larger num up
        with self.tik_instance.new_stmt_scope():
            zero_tensor = self.tik_instance.Tensor("float16", (16,), name="zero_tensor", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(16, zero_tensor, 0, 1, 1, 1)
            with self.tik_instance.for_range(0, self.align_roi // 16) as i:
                self.tik_instance.vmax(16, self.rois_start_w[i * 16], self.rois_start_w[i * 16], zero_tensor,
                                       1, 1, 1, 1, 1, 1, 1)
                self.tik_instance.vmax(16, self.rois_start_h[i * 16], self.rois_start_h[i * 16], zero_tensor,
                                       1, 1, 1, 1, 1, 1, 1)

        # roi_end_h - roi_start_h
        # roi_end_w - roi_start_w
        with self.tik_instance.for_range(0, self.align_roi // 16) as i:
            one_tensor = self.tik_instance.Tensor("float16", (16,), name="one_tensor", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(16, one_tensor, 1.0, 1, 1, 1)
            self.tik_instance.vsub(16, self.rois_height[i * 16], self.rois_end_h[i * 16],
                                   self.rois_start_h[i * 16],
                                   1, 1, 1, 1, 1, 1, 1)
            self.tik_instance.vsub(16, self.rois_width[i * 16], self.rois_end_w[i * 16], self.rois_start_w[i * 16],
                                   1, 1, 1, 1, 1, 1, 1)

            # self.rois_height + 1
            # self.rois_width + 1
            self.tik_instance.vadds(16, self.rois_height[i * 16], self.rois_height[i * 16], 1.0, 1, 1, 1, 1, 1)
            self.tik_instance.vadds(16, self.rois_width[i * 16], self.rois_width[i * 16], 1.0, 1, 1, 1, 1, 1)

            # calc maximum (roi_height, 1)
            # calc maximum (roi_width, 1)
            self.tik_instance.vmax(16, self.rois_height[i * 16], self.rois_height[i * 16], one_tensor,
                                   1, 1, 1, 1, 1, 1, 1)
            self.tik_instance.vmax(16, self.rois_width[i * 16], self.rois_width[i * 16], one_tensor,
                                   1, 1, 1, 1, 1, 1, 1)

            # self.bin_size_h equals to self.rois_height * pooled_h_reciprocal
            # self.bin_size_w equals to self.rois_width * pooled_w_reciprocal
            self.tik_instance.vmuls(16, self.bin_size_w[i * 16], self.rois_width[i * 16], self.pooled_w_reciprocal,
                                    1, 1, 1, 1, 1)
            self.tik_instance.vmuls(16, self.bin_size_h[i * 16], self.rois_height[i * 16],
                                    self.pooled_h_reciprocal,
                                    1, 1, 1, 1, 1)

    def _calc_h_w_cbuf_center(self, hcenter_cbuf, wcenter_cbuf):
        with self.tik_instance.new_stmt_scope():  # use scope to release one_scalar
            hcenter = self.tik_instance.Tensor("float16", [self.align_roi, self.shape_align],
                                               scope=tik.scope_ubuf, name="hcenter")  # 38KB
            wcenter = self.tik_instance.Tensor("float16", [self.align_roi, self.shape_align],
                                               scope=tik.scope_ubuf, name="wcenter")  # 38KB
            y_tensor = self.tik_instance.Tensor("float16", [self.shape_align, ], scope=tik.scope_ubuf,
                                                name="y_tensor")
            x_tensor = self.tik_instance.Tensor("float16", [self.shape_align, ], scope=tik.scope_ubuf,
                                                name="x_tensor")
            temp_scalar = self.tik_instance.Scalar("float16", "temp_scalar")

            for i in range(0, self.pooled_h):
                for j in range(0, self.pooled_w):
                    temp = float(i) + 0.5
                    temp_scalar.set_as(temp)
                    index = (i * self.pooled_h) + j
                    y_tensor[index] = temp_scalar
                    temp = float(j) + 0.5
                    temp_scalar.set_as(temp)
                    x_tensor[index] = temp_scalar

            with self.tik_instance.for_range(0, self.align_roi) as i:
                temp_scalar.set_as(self.bin_size_h[i])
                self.tik_instance.vmuls(16, hcenter[i, 0], y_tensor[0], temp_scalar,
                                        self.shape_align // 16, 1, 1, 1, 1)
                temp_scalar.set_as(self.bin_size_w[i])
                self.tik_instance.vmuls(16, wcenter[i, 0], x_tensor[0], temp_scalar,
                                        self.shape_align // 16, 1, 1, 1, 1)

            self.tik_instance.data_move(hcenter_cbuf, hcenter, 0, 1, self.align_roi * self.shape_align // 16,
                                        1, 1)
            self.tik_instance.data_move(wcenter_cbuf, wcenter, 0, 1, self.align_roi * self.shape_align // 16,
                                        1, 1)

    def _calc_h_w_center(self, hcenter_cbuf, wcenter_cbuf):
        hcenter_fp32 = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                scope=tik.scope_ubuf, name="hcenter_fp32")
        wcenter_fp32 = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                scope=tik.scope_ubuf, name="wcenter_fp32")
        temp_scalar = self.tik_instance.Scalar("float32", "temp_scalar")
        rois_start_h_fp32 = self.tik_instance.Tensor("float32", [self.align_roi, ], scope=tik.scope_ubuf,
                                                     name="rois_start_h_fp32")
        rois_start_w_fp32 = self.tik_instance.Tensor("float32", [self.align_roi, ], scope=tik.scope_ubuf,
                                                     name="rois_start_w_fp32")
        hcenter = self.tik_instance.Tensor("float16", [self.align_roi, self.shape_align],
                                           scope=tik.scope_ubuf, name="hcenter")
        wcenter = self.tik_instance.Tensor("float16", [self.align_roi, self.shape_align],
                                           scope=tik.scope_ubuf, name="wcenter")
        zero_tensor = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                               name="zero_tensor")
        height_tensor = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                                 name="height_tensor")
        width_tensor = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                                name="width_tensor")
        self.tik_instance.vector_dup(16, zero_tensor, 0, self.shape_align // 16, 1, 2)
        self.tik_instance.vector_dup(16, height_tensor, self.h_feature_map - 1, self.shape_align // 16, 1, 2)
        self.tik_instance.vector_dup(16, width_tensor, self.w_feature_map - 1, self.shape_align // 16, 1, 2)
        self.tik_instance.vconv(16, "none", rois_start_h_fp32, self.rois_start_h, self.align_roi // 16, 1, 1, 2, 1)
        self.tik_instance.vconv(16, "none", rois_start_w_fp32, self.rois_start_w, self.align_roi // 16, 1, 1, 2, 1)
        self.tik_instance.data_move(hcenter, hcenter_cbuf, 0, 1, self.align_roi * self.shape_align // 16, 1, 1)
        self.tik_instance.data_move(wcenter, wcenter_cbuf, 0, 1, self.align_roi * self.shape_align // 16, 1, 1)
        with self.tik_instance.for_range(0, self.align_roi) as i:
            self.tik_instance.vconv(16, "none", hcenter_fp32[i, 0], hcenter[i, 0], self.shape_align // 16, 1, 1, 2, 1)
            self.tik_instance.vconv(16, "none", wcenter_fp32[i, 0], wcenter[i, 0], self.shape_align // 16, 1, 1, 2, 1)
        with self.tik_instance.for_range(0, self.align_roi) as i:
            # add
            temp_scalar.set_as(rois_start_h_fp32[i])
            self.tik_instance.vadds(16, hcenter_fp32[i, 0], hcenter_fp32[i, 0], temp_scalar,
                                    self.shape_align // 16, 1, 1, 2, 2)
            temp_scalar.set_as(rois_start_w_fp32[i])
            self.tik_instance.vadds(16, wcenter_fp32[i, 0], wcenter_fp32[i, 0], temp_scalar,
                                    self.shape_align // 16, 1, 1, 2, 2)

        with self.tik_instance.for_range(0, self.align_roi) as i:
            self.tik_instance.vmax(16, hcenter_fp32[i, 0], hcenter_fp32[i, 0], zero_tensor[0],
                                   self.shape_align // 16, 1, 1, 1, 2, 2, 2)
            self.tik_instance.vmin(16, hcenter_fp32[i, 0], hcenter_fp32[i, 0], height_tensor[0],
                                   self.shape_align // 16, 1, 1, 1, 2, 2, 2)
            self.tik_instance.vmax(16, wcenter_fp32[i, 0], wcenter_fp32[i, 0], zero_tensor[0],
                                   self.shape_align // 16, 1, 1, 1, 2, 2, 2)
            self.tik_instance.vmin(16, wcenter_fp32[i, 0], wcenter_fp32[i, 0], width_tensor[0],
                                   self.shape_align // 16, 1, 1, 1, 2, 2, 2)
        self.tik_instance.data_move(self.hcenter_fp32_cbuf, hcenter_fp32, 0, 1, self.align_roi *
                                    self.shape_align // 8, 1, 1)
        self.tik_instance.data_move(self.wcenter_fp32_cbuf, wcenter_fp32, 0, 1, self.align_roi *
                                    self.shape_align // 8, 1, 1)

    def _h_w_center_processor(self):
        with self.tik_instance.new_stmt_scope():
            hcenter_cbuf = self.tik_instance.Tensor("float16", [self.align_roi, self.shape_align],
                                                    scope=tik.scope_cbuf, name="hcenter_cbuf")
            wcenter_cbuf = self.tik_instance.Tensor("float16", [self.align_roi, self.shape_align],
                                                    scope=tik.scope_cbuf, name="wcenter_cbuf")

            self._calc_h_w_cbuf_center(hcenter_cbuf, wcenter_cbuf)

            # hcenter equals to min(max(hcenter + roi_start_h, 0), (h_feature_map - 1))
            # wcenter equals to min(max(wcenter + roi_start_w, 0), (w_feature_map - 1))
            with self.tik_instance.new_stmt_scope():
                self._calc_h_w_center(hcenter_cbuf, wcenter_cbuf)

    def _calc_h_w_start(self):
        """
        hstart equals to hcenter & hstart = int(hstart)
        """
        with self.tik_instance.new_stmt_scope():
            hstart = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align],
                                              scope=tik.scope_ubuf, name="hstart")
            hcenter_fp32 = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                    scope=tik.scope_ubuf, name="hcenter_fp32")
            self.tik_instance.data_move(hcenter_fp32, self.hcenter_fp32_cbuf, 0, 1,
                                        self.align_roi * self.shape_align // 8, 1, 1)
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.floor_f32toi32(hstart[i, 0], hcenter_fp32[i, 0], self.shape_align)
            self.tik_instance.data_move(self.hstart_cbuf, hstart, 0, 1, self.align_roi * self.shape_align // 8,
                                        1, 1)

        # wstart equals to wcenter & wstart = int(wstart)
        with self.tik_instance.new_stmt_scope():
            wstart = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align],
                                              scope=tik.scope_ubuf, name="wstart")
            wcenter_fp32 = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                    scope=tik.scope_ubuf, name="wcenter_fp32")
            self.tik_instance.data_move(wcenter_fp32, self.wcenter_fp32_cbuf, 0, 1,
                                        self.align_roi * self.shape_align // 8, 1, 1)
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.floor_f32toi32(wstart[i, 0], wcenter_fp32[i, 0], self.shape_align)
            self.tik_instance.data_move(self.wstart_cbuf, wstart, 0, 1, self.align_roi * self.shape_align // 8,
                                        1, 1)

    def _calc_h_w_end(self, one_tensor):
        """
        hend equals to min(max(hstart + 1), 0), 57 - 1) ---> hend equals to hstart + 1
        ---> wend equals to wstart + 1
        """
        with self.tik_instance.new_stmt_scope():
            hstart = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align],
                                              scope=tik.scope_ubuf, name="hstart")
            self.tik_instance.data_move(hstart, self.hstart_cbuf, 0, 1, self.align_roi * self.shape_align // 8,
                                        1, 1)
            hend = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align],
                                            scope=tik.scope_ubuf,
                                            name="hend")
            height_tensor = self.tik_instance.Tensor("int32", [self.shape_align, ], scope=tik.scope_ubuf,
                                                     name="height_tensor")
            self.tik_instance.vector_dup(16, height_tensor, self.h_feature_map - 1,
                                         self.shape_align // 16, 1, 2)

            # on platform 310, vadds not suport int32.use vadd here
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.vadd(16, hend[i, 0], hstart[i, 0], one_tensor, self.shape_align // 16,
                                       1, 1, 1, 2, 2, 2)
                self.tik_instance.vmin(16, hend[i, 0], hend[i, 0], height_tensor, self.shape_align // 16,
                                       1, 1, 1, 2, 2, 2)
            self.tik_instance.data_move(self.hend_cbuf, hend, 0, 1, self.align_roi * self.shape_align // 8,
                                        1, 1)

        with self.tik_instance.new_stmt_scope():
            wstart = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align],
                                              scope=tik.scope_ubuf, name="wstart")
            self.tik_instance.data_move(wstart, self.wstart_cbuf, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)
            wend = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align],
                                            scope=tik.scope_ubuf, name="wend")
            width_tensor = self.tik_instance.Tensor("int32", [self.shape_align, ], scope=tik.scope_ubuf,
                                                    name="width_tensor")
            self.tik_instance.vector_dup(16, width_tensor, self.w_feature_map - 1, self.shape_align // 16, 1, 2)

            # on platform 310, vadds not suport int32.use vadd here
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.vadd(16, wend[i, 0], wstart[i, 0], one_tensor, self.shape_align // 16,
                                       1, 1, 1, 2, 2, 2)
                self.tik_instance.vmin(16, wend[i, 0], wend[i, 0], width_tensor, self.shape_align // 16,
                                       1, 1, 1, 2, 2, 2)
            self.tik_instance.data_move(self.wend_cbuf, wend, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)

    def _calc_h_w_start_end(self):
        with self.tik_instance.new_stmt_scope():  # use scope to release one_scalar
            one_scalar = self.tik_instance.Scalar("int32", "one_scalar")
            one_scalar.set_as(1)
            one_tensor = self.tik_instance.Tensor("int32", [self.shape_align, ], scope=tik.scope_ubuf,
                                                  name="one_tensor")
            self.tik_instance.vector_dup(16, one_tensor, one_scalar, self.shape_align // 16, 1, 2)

            self._calc_h_w_start()
            self._calc_h_w_end(one_tensor)

    def _calc_fx_0(self):
        # fX0 equals to wcenter - wstart
        with self.tik_instance.new_stmt_scope():
            # get wcenter
            wcenter = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                               scope=tik.scope_ubuf, name="wcenter")
            self.tik_instance.data_move(wcenter, self.wcenter_fp32_cbuf, 0, 1,
                                        self.align_roi * self.shape_align // 8, 1, 1)

            wstart = self.tik_instance.Tensor("int32", [self.shape_align, ], scope=tik.scope_ubuf,
                                              name="wstart")
            wstart_fp16 = self.tik_instance.Tensor("float16", [self.shape_align, ], scope=tik.scope_ubuf,
                                                   name="wstart_fp16")
            wstart_fp32 = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                                   name="wstart_fp32")
            # most size of cbuf is used here !!!!
            fx_a = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                            name="fx_a")

            # turn self.wstart_int to wstart_float
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.data_move(wstart, self.wstart_cbuf[i, 0], 0, 1, self.shape_align // 8, 1, 1)
                self.tik_instance.vconv(16, 'none', wstart_fp16, wstart, self.shape_align // 16, 1, 1, 1,
                                        2, deqscale=1.0)
                self.tik_instance.vconv(16, 'none', wstart_fp32, wstart_fp16, self.shape_align // 16, 1,
                                        1, 2, 1, deqscale=None)
                self.tik_instance.vsub(16, fx_a, wcenter[i, 0], wstart_fp32, self.shape_align // 16, 1, 1,
                                       1, 2, 2, 2)
                self.tik_instance.data_move(self.fx_a_cbuf[i, 0], fx_a, 0, 1, self.shape_align // 8, 1, 1)

    def _calc_fx_1(self):
        with self.tik_instance.new_stmt_scope():
            # get wcenter
            wcenter = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                               scope=tik.scope_ubuf, name="wcenter")
            self.tik_instance.data_move(wcenter, self.wcenter_fp32_cbuf, 0, 1,
                                        self.align_roi * self.shape_align // 8, 1, 1)

            wend = self.tik_instance.Tensor("int32", [self.shape_align, ], scope=tik.scope_ubuf,
                                            name="wend")
            wend_fp16 = self.tik_instance.Tensor("float16", [self.shape_align, ], scope=tik.scope_ubuf,
                                                 name="wend_fp16")
            wend_fp32 = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                                 name="wend_fp32")
            fx_b = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                            name="fx_b")

            # turn self.wend_int to wend_float
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.data_move(wend, self.wend_cbuf[i, 0], 0, 1, self.shape_align // 8, 1, 1)
                self.tik_instance.vconv(16, 'none', wend_fp16, wend, self.shape_align // 16, 1, 1, 1, 2,
                                        deqscale=1.0)
                self.tik_instance.vconv(16, 'none', wend_fp32, wend_fp16, self.shape_align // 16, 1, 1,
                                        2, 1, deqscale=None)
                self.tik_instance.vsub(16, fx_b, wend_fp32, wcenter[i, 0], self.shape_align // 16, 1, 1,
                                       1, 2, 2, 2)
                self.tik_instance.data_move(self.fx_b_cbuf[i, 0], fx_b, 0, 1, self.shape_align // 8, 1, 1)

    def _calc_fy_0(self):
        with self.tik_instance.new_stmt_scope():
            # get hcenter
            hcenter = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                               scope=tik.scope_ubuf, name="hcenter")
            self.tik_instance.data_move(hcenter, self.hcenter_fp32_cbuf, 0, 1,
                                        self.align_roi * self.shape_align // 8, 1, 1)

            hstart = self.tik_instance.Tensor("int32", [self.shape_align, ], scope=tik.scope_ubuf,
                                              name="hstart")
            hstart_fp16 = self.tik_instance.Tensor("float16", [self.shape_align, ], scope=tik.scope_ubuf,
                                                   name="hstart_fp16")
            hstart_fp32 = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                                   name="hstart_fp32")
            fy_a = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                            name="fy_a")

            # turn self.hstart_int to hstart_float
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.data_move(hstart, self.hstart_cbuf[i, 0], 0, 1, self.shape_align // 8, 1, 1)
                self.tik_instance.vconv(16, 'none', hstart_fp16, hstart, self.shape_align // 16, 1,
                                        1, 1, 2, deqscale=1.0)
                self.tik_instance.vconv(16, 'none', hstart_fp32, hstart_fp16, self.shape_align // 16,
                                        1, 1, 2, 1, deqscale=None)
                self.tik_instance.vsub(16, fy_a, hcenter[i, 0], hstart_fp32, self.shape_align // 16,
                                       1, 1, 1, 2, 2, 2)
                self.tik_instance.data_move(self.fy_a_cbuf[i, 0], fy_a, 0, 1, self.shape_align // 8, 1, 1)

    def _calc_fy_1(self):
        with self.tik_instance.new_stmt_scope():
            # get hcenter
            hcenter = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                               scope=tik.scope_ubuf, name="hcenter")
            self.tik_instance.data_move(hcenter, self.hcenter_fp32_cbuf, 0, 1,
                                        self.align_roi * self.shape_align // 8, 1, 1)
            hend = self.tik_instance.Tensor("int32", [self.shape_align, ], scope=tik.scope_ubuf,
                                            name="hend")
            hend_fp16 = self.tik_instance.Tensor("float16", [self.shape_align, ], scope=tik.scope_ubuf,
                                                 name="hend_fp16")
            hend_fp32 = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                                 name="hend_fp32")
            fy_b = self.tik_instance.Tensor("float32", [self.shape_align, ], scope=tik.scope_ubuf,
                                            name="fy_b")

            # turn self.hend_int to hend_float
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.data_move(hend, self.hend_cbuf[i, 0], 0, 1, self.shape_align // 8, 1, 1)
                self.tik_instance.vconv(16, 'none', hend_fp16, hend, self.shape_align // 16, 1, 1, 1, 2,
                                        deqscale=1.0)
                self.tik_instance.vconv(16, 'none', hend_fp32, hend_fp16, self.shape_align // 16, 1, 1,
                                        2, 1, deqscale=None)
                self.tik_instance.vsub(16, fy_b, hend_fp32, hcenter[i, 0], self.shape_align // 16, 1, 1,
                                       1, 2, 2, 2)
                self.tik_instance.data_move(self.fy_b_cbuf[i, 0], fy_b, 0, 1, self.shape_align // 8, 1, 1)

    def _calc_factor_a(self):
        # factorA equals to fY1 * fx1 ---> self.factor_a_cbuf = fy_b * fx_b
        with self.tik_instance.new_stmt_scope():
            fy_b = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_ubuf,
                                            name="fy_b")
            fx_b = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_ubuf,
                                            name="fx_b")
            factor_a = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                scope=tik.scope_ubuf, name="facotr_a")
            # most size of ubuf is used here !!!!
            self.tik_instance.data_move(fy_b, self.fy_b_cbuf, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)
            self.tik_instance.data_move(fx_b, self.fx_b_cbuf, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.vmul(16, factor_a[i, 0], fy_b[i, 0], fx_b[i, 0], self.shape_align // 16,
                                       1, 1, 1, 2, 2, 2)
            self.tik_instance.data_move(self.factor_a_cbuf, factor_a, 0, 1, self.align_roi * self.shape_align // 8,
                                        1, 1)

    def _calc_factor_b(self):
        # factorB equals to fY1 * fx0 ---> self.factor_b_cbuf = fy_b * fx_a
        with self.tik_instance.new_stmt_scope():
            fy_b = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_ubuf,
                                            name="fy_b")
            fx_a = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_ubuf,
                                            name="fx_a")
            factor_b = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                scope=tik.scope_ubuf, name="facotr_b")
            self.tik_instance.data_move(fy_b, self.fy_b_cbuf, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)
            self.tik_instance.data_move(fx_a, self.fx_a_cbuf, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.vmul(16, factor_b[i, 0], fy_b[i, 0], fx_a[i, 0], self.shape_align // 16,
                                       1, 1, 1, 2, 2, 2)
            self.tik_instance.data_move(self.factor_b_cbuf, factor_b, 0, 1, self.align_roi * self.shape_align // 8,
                                        1, 1)

    def _calc_factor_c(self):
        # factorC equals to fY0 * fx1 ---> self.factor_c_cbuf = fy_a * fx_b
        with self.tik_instance.new_stmt_scope():
            fy_a = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_ubuf,
                                            name="fy_a")
            fx_b = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_ubuf,
                                            name="fx_b")
            factor_c = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                scope=tik.scope_ubuf, name="facotr_c")
            self.tik_instance.data_move(fy_a, self.fy_a_cbuf, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)
            self.tik_instance.data_move(fx_b, self.fx_b_cbuf, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.vmul(16, factor_c[i, 0], fy_a[i, 0], fx_b[i, 0], self.shape_align // 16,
                                       1, 1, 1, 2, 2, 2)
            self.tik_instance.data_move(self.factor_c_cbuf, factor_c, 0, 1, self.align_roi * self.shape_align // 8,
                                        1, 1)

    def _calc_factor_d(self):
        # factorD equals to fY0 * fx0 ---> self.factor_d_cbuf = fy_a * fx_a
        with self.tik_instance.new_stmt_scope():
            fy_a = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_ubuf,
                                            name="fy_a")
            fx_a = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_ubuf,
                                            name="fx_a")
            factor_d = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                scope=tik.scope_ubuf, name="facotr_d")
            self.tik_instance.data_move(fy_a, self.fy_a_cbuf, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)
            self.tik_instance.data_move(fx_a, self.fx_a_cbuf, 0, 1, self.align_roi * self.shape_align // 8, 1, 1)
            with self.tik_instance.for_range(0, self.align_roi) as i:
                self.tik_instance.vmul(16, factor_d[i, 0], fy_a[i, 0], fx_a[i, 0], self.shape_align // 16,
                                       1, 1, 1, 2, 2, 2)
            self.tik_instance.data_move(self.factor_d_cbuf, factor_d, 0, 1, self.align_roi * self.shape_align // 8,
                                        1, 1)

    def _clear_ub_to_zero(self):
        repeat = self.chn_size // 16
        if repeat > 255:
            repeat_res = repeat - 255
            start = 255 * 16
            self.tik_instance.vmuls(16, self.hstart_wstart_fp32_ub, self.hstart_wstart_fp32_ub, 0,
                                    255, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hstart_wend_fp32_ub, self.hstart_wend_fp32_ub, 0,
                                    255, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hend_wstart_fp32_ub, self.hend_wstart_fp32_ub, 0,
                                    255, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hend_wend_fp32_ub, self.hend_wend_fp32_ub, 0,
                                    255, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hstart_wstart_fp32_ub[start], self.hstart_wstart_fp32_ub[start], 0,
                                    repeat_res, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hstart_wend_fp32_ub[start], self.hstart_wend_fp32_ub[start], 0,
                                    repeat_res, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hend_wstart_fp32_ub[start], self.hend_wstart_fp32_ub[start], 0,
                                    repeat_res, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hend_wend_fp32_ub[start], self.hend_wend_fp32_ub[start], 0,
                                    repeat_res, 1, 1, 2, 2)
        else:
            self.tik_instance.vmuls(16, self.hstart_wstart_fp32_ub, self.hstart_wstart_fp32_ub, 0,
                                    self.chn_size // 16, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hstart_wend_fp32_ub, self.hstart_wend_fp32_ub, 0,
                                    self.chn_size // 16, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hend_wstart_fp32_ub, self.hend_wstart_fp32_ub, 0,
                                    self.chn_size // 16, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, self.hend_wend_fp32_ub, self.hend_wend_fp32_ub, 0,
                                    self.chn_size // 16, 1, 1, 2, 2)

    def _data_move_single_roi(self, loop_a):
        self.tik_instance.data_move(self.hstart_int, self.hstart_cbuf[loop_a, 0], 0, 1, self.shape_align // 8, 1, 1)
        self.tik_instance.data_move(self.wstart_int, self.wstart_cbuf[loop_a, 0], 0, 1, self.shape_align // 8, 1, 1)
        self.tik_instance.data_move(self.hend_int, self.hend_cbuf[loop_a, 0], 0, 1, self.shape_align // 8, 1, 1)
        self.tik_instance.data_move(self.wend_int, self.wend_cbuf[loop_a, 0], 0, 1, self.shape_align // 8, 1, 1)

        self.tik_instance.data_move(self.factor_a_float32, self.factor_a_cbuf[loop_a, 0], 0, 1,
                                    self.shape_align // 8, 1, 1)
        self.tik_instance.data_move(self.factor_b_float32, self.factor_b_cbuf[loop_a, 0], 0, 1,
                                    self.shape_align // 8, 1, 1)
        self.tik_instance.data_move(self.factor_c_float32, self.factor_c_cbuf[loop_a, 0], 0, 1,
                                    self.shape_align // 8, 1, 1)
        self.tik_instance.data_move(self.factor_d_float32, self.factor_d_cbuf[loop_a, 0], 0, 1,
                                    self.shape_align // 8, 1, 1)

    def _vconv_align16(self, dst_ub, src_ub, length):
        repeat = length // 16
        if repeat <= 0:
            return

        if repeat > 255:
            start = 255 * 16
            repeat_res = repeat - 255
            self.tik_instance.vconv(16, 'none', dst_ub, src_ub, 255, 1, 1, 2, 1, deqscale=None)
            self.tik_instance.vconv(16, 'none', dst_ub[start], src_ub[start], repeat_res, 1, 1, 2, 1, deqscale=None)
        else:
            self.tik_instance.vconv(16, 'none', dst_ub, src_ub, repeat, 1, 1, 2, 1, deqscale=None)

    def _roi_align_preprocessor(self, index_here):
        self.hstart_index.set_as(self.hstart_int[index_here])
        with self.tik_instance.if_scope(self.hstart_index >= self.h_feature_map):
            self.hstart_index.set_as(self.h_feature_map - 1)

        self.hend_index.set_as(self.hend_int[index_here])
        with self.tik_instance.if_scope(self.hend_index >= self.h_feature_map):
            self.hend_index.set_as(self.h_feature_map - 1)

        self.wstart_index.set_as(self.wstart_int[index_here])
        with self.tik_instance.if_scope(self.wstart_index >= self.w_feature_map):
            self.wstart_index.set_as(self.w_feature_map - 1)

        self.wend_index.set_as(self.wend_int[index_here])
        with self.tik_instance.if_scope(self.wend_index >= self.w_feature_map):
            self.wend_index.set_as(self.w_feature_map - 1)
        # fFactor_x == 0  if true
        with self.tik_instance.if_scope(tik.all(self.wstart_index == self.w_feature_map,
                                                self.wend_index == self.w_feature_map)):
            self._clear_ub_to_zero()
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(tik.all(self.hstart_index == self.h_feature_map,
                                                    self.hend_index == self.h_feature_map)):
                self._clear_ub_to_zero()
            with self.tik_instance.else_scope():
                gap = self.h_feature_map * self.w_feature_map - 1
                self.tik_instance.data_move(self.hstart_wstart_ub,
                                            self.feature_map_gm[0, 0, self.hstart_index, self.wstart_index, 0],
                                            0, self.c1_feature_map, 1, gap, 0)
                self.tik_instance.data_move(self.hstart_wend_ub,
                                            self.feature_map_gm[0, 0, self.hstart_index, self.wend_index, 0], 0,
                                            self.c1_feature_map, 1, gap, 0)
                self.tik_instance.data_move(self.hend_wstart_ub,
                                            self.feature_map_gm[0, 0, self.hend_index, self.wstart_index, 0], 0,
                                            self.c1_feature_map, 1, gap, 0)
                self.tik_instance.data_move(self.hend_wend_ub,
                                            self.feature_map_gm[0, 0, self.hend_index, self.wend_index, 0], 0,
                                            self.c1_feature_map, 1, gap, 0)

                self._vconv_align16(self.hstart_wstart_fp32_ub, self.hstart_wstart_ub, self.chn_size)
                self._vconv_align16(self.hstart_wend_fp32_ub, self.hstart_wend_ub, self.chn_size)
                self._vconv_align16(self.hend_wstart_fp32_ub, self.hend_wstart_ub, self.chn_size)
                self._vconv_align16(self.hend_wend_fp32_ub, self.hend_wend_ub, self.chn_size)

    def _roi_align_vmuls_align16(self, src_ub, temp_scalar, length):
        repeat = length // 16
        if repeat <= 0:
            return
        if repeat > 255:
            start = 255 * 16
            repeat_res = repeat - 255
            self.tik_instance.vmuls(16, src_ub, src_ub, temp_scalar, 255, 1, 1, 2, 2)
            self.tik_instance.vmuls(16, src_ub[start], src_ub[start], temp_scalar, repeat_res, 1, 1, 2, 2)
        else:
            self.tik_instance.vmuls(16, src_ub, src_ub, temp_scalar, repeat, 1, 1, 2, 2)

    def _roi_align_vadd_align16(self, dst_ub, in_a_ub, in_b_ub, length):
        repeat = length // 16
        if repeat <= 0:
            return

        if repeat > 255:
            start = 255 * 16
            repeat_res = repeat - 255
            self.tik_instance.vadd(16, dst_ub, in_a_ub, in_b_ub, 255, 1, 1, 1, 2, 2, 2)
            self.tik_instance.vadd(16, dst_ub[start], in_a_ub[start], in_b_ub[start], repeat_res, 1, 1, 1, 2, 2, 2)
        else:
            self.tik_instance.vadd(16, dst_ub, in_a_ub, in_b_ub, repeat, 1, 1, 1, 2, 2, 2)

    def _roi_align_processor(self):
        self.chn_size = self.c1_feature_map * 16
        self.hend_wend_ub = self.tik_instance.Tensor("float16", (self.chn_size,), tik.scope_ubuf, "hend_wend_ub")
        self.hend_wstart_ub = self.tik_instance.Tensor("float16", (self.chn_size,), tik.scope_ubuf, "hend_wstart_ub")
        self.hstart_wend_ub = self.tik_instance.Tensor("float16", (self.chn_size,), tik.scope_ubuf, "hstart_wend_ub")
        self.hstart_wstart_ub = self.tik_instance.Tensor("float16", (self.chn_size,), tik.scope_ubuf,
                                                         "hstart_wstart_ub")
        self.hend_wend_fp32_ub = self.tik_instance.Tensor("float32", (self.chn_size,), tik.scope_ubuf,
                                                          "hend_wend_fp32_ub")
        self.hend_wstart_fp32_ub = self.tik_instance.Tensor("float32", (self.chn_size,), tik.scope_ubuf,
                                                            "hend_wstart_fp32_ub")
        self.hstart_wend_fp32_ub = self.tik_instance.Tensor("float32", (self.chn_size,), tik.scope_ubuf,
                                                            "hstart_wend_fp32_ub")
        self.hstart_wstart_fp32_ub = self.tik_instance.Tensor("float32", (self.chn_size,), tik.scope_ubuf,
                                                              "hstart_wstart_fp32_ub")
        temp_sum = self.tik_instance.Tensor("float32", (self.chn_size,), name="temp_sum_a", scope=tik.scope_ubuf)
        temp_fp16_sum = self.tik_instance.Tensor("float16", (self.chn_size,), name="temp_fp16_sum",
                                                 scope=tik.scope_ubuf)

        self.hstart_index = self.tik_instance.Scalar("int32")
        self.wstart_index = self.tik_instance.Scalar("int32")
        self.hend_index = self.tik_instance.Scalar("int32")
        self.wend_index = self.tik_instance.Scalar("int32")
        temp_scalar = self.tik_instance.Scalar("float32", "temp_scalar")

        with self.tik_instance.for_range(0, self.n_roi) as loop_a:
            self._data_move_single_roi(loop_a)
            with self.tik_instance.for_range(0, self.pooled_h) as ind_y:
                with self.tik_instance.for_range(0, self.pooled_w) as ind_x:
                    index_here = ind_y * self.pooled_h + ind_x
                    self._roi_align_preprocessor(index_here)
                    temp_scalar.set_as(self.factor_a_float32[index_here])
                    self._roi_align_vmuls_align16(self.hstart_wstart_fp32_ub, temp_scalar, self.chn_size)
                    temp_scalar.set_as(self.factor_b_float32[index_here])
                    self._roi_align_vmuls_align16(self.hstart_wend_fp32_ub, temp_scalar, self.chn_size)
                    temp_scalar.set_as(self.factor_c_float32[index_here])
                    self._roi_align_vmuls_align16(self.hend_wstart_fp32_ub, temp_scalar, self.chn_size)
                    temp_scalar.set_as(self.factor_d_float32[index_here])
                    self._roi_align_vmuls_align16(self.hend_wend_fp32_ub, temp_scalar, self.chn_size)

                    self._roi_align_vadd_align16(temp_sum, self.hstart_wstart_fp32_ub, self.hstart_wend_fp32_ub,
                                                 self.chn_size)
                    self._roi_align_vadd_align16(temp_sum, temp_sum, self.hend_wstart_fp32_ub, self.chn_size)
                    self._roi_align_vadd_align16(temp_sum, temp_sum, self.hend_wend_fp32_ub, self.chn_size)
                    self.tik_instance.vconv(16, 'none', temp_fp16_sum, temp_sum, self.chn_size // 16,
                                            1, 1, 1, 2, deqscale=None)
                    self.tik_instance.data_move(self.roi_align_gm[loop_a, 0, ind_y, ind_x, 0], temp_fp16_sum, 0,
                                                self.chn_size // 16, 1, 0, self.pooled_h * self.pooled_w - 1)

    def _prepare_calc_roi_width_height_buf(self):
        self.rois_height = self.tik_instance.Tensor("float16", (self.align_roi,), name="rois_height",
                                                    scope=tik.scope_ubuf)
        self.rois_width = self.tik_instance.Tensor("float16", (self.align_roi,), name="rois_width",
                                                   scope=tik.scope_ubuf)
        self.rois_start_w = self.tik_instance.Tensor("float16", (self.align_roi,), name="rois_start_w",
                                                     scope=tik.scope_ubuf)
        self.rois_start_h = self.tik_instance.Tensor("float16", (self.align_roi,), name="rois_start_h",
                                                     scope=tik.scope_ubuf)
        self.rois_end_w = self.tik_instance.Tensor("float16", (self.align_roi,), name="rois_end_w",
                                                   scope=tik.scope_ubuf)
        self.rois_end_h = self.tik_instance.Tensor("float16", (self.align_roi,), name="rois_end_h",
                                                   scope=tik.scope_ubuf)
        self.bin_size_h = self.tik_instance.Tensor("float16", (self.align_roi,), name="bin_size_h",
                                                   scope=tik.scope_ubuf)
        self.bin_size_w = self.tik_instance.Tensor("float16", (self.align_roi,), name="bin_size_w",
                                                   scope=tik.scope_ubuf)

    def _prepare_calc_h_w_center_buf(self):
        self.hstart_cbuf = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align], scope=tik.scope_cbuf,
                                                    name="hstart_cbuf")
        self.wstart_cbuf = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align], scope=tik.scope_cbuf,
                                                    name="wstart_cbuf")
        self.hend_cbuf = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align], scope=tik.scope_cbuf,
                                                  name="hend_cbuf")
        self.wend_cbuf = self.tik_instance.Tensor("int32", [self.align_roi, self.shape_align], scope=tik.scope_cbuf,
                                                  name="wend_cbuf")
        self.fx_a_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_cbuf,
                                                  name="fx_a_cbuf")
        self.fx_b_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_cbuf,
                                                  name="fx_b_cbuf")
        self.fy_a_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_cbuf,
                                                  name="fy_a_cbuf")
        self.fy_b_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align], scope=tik.scope_cbuf,
                                                  name="fy_b_cbuf")

    def _prepare_calc_factor_buf(self):
        self.factor_a_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                      scope=tik.scope_cbuf, name="factor_a")
        self.factor_b_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                      scope=tik.scope_cbuf, name="factor_b")
        self.factor_c_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                      scope=tik.scope_cbuf, name="factor_c")
        self.factor_d_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                      scope=tik.scope_cbuf, name="factor_d")

    def _prepare_roi_align_processor_buf(self):
        self.hstart_int = self.tik_instance.Tensor("int32", (self.shape_align,), name="hstart_int",
                                                   scope=tik.scope_ubuf)
        self.wstart_int = self.tik_instance.Tensor("int32", (self.shape_align,), name="wstart_int",
                                                   scope=tik.scope_ubuf)
        self.hend_int = self.tik_instance.Tensor("int32", (self.shape_align,), name="hend_int", scope=tik.scope_ubuf)
        self.wend_int = self.tik_instance.Tensor("int32", (self.shape_align,), name="wend_int", scope=tik.scope_ubuf)

        self.factor_a_float32 = self.tik_instance.Tensor("float32", (self.shape_align,), name="factor_a_float32",
                                                         scope=tik.scope_ubuf)
        self.factor_b_float32 = self.tik_instance.Tensor("float32", (self.shape_align,), name="factor_b_float32",
                                                         scope=tik.scope_ubuf)
        self.factor_c_float32 = self.tik_instance.Tensor("float32", (self.shape_align,), name="factor_c_float32",
                                                         scope=tik.scope_ubuf)
        self.factor_d_float32 = self.tik_instance.Tensor("float32", (self.shape_align,), name="factor_d_float32",
                                                         scope=tik.scope_ubuf)

    def roi_align(self):
        """
        main process
        :return: None
        """
        self.roi_ub = self.tik_instance.Tensor("float16", (self.n_roi, 16), name="roi_ub", scope=tik.scope_ubuf)
        # rois[][] * self.spatial_scale
        with self.tik_instance.new_stmt_scope():
            spatial_scale_scalar = self.tik_instance.Scalar("float16", "spatial_scale_scalar")
            spatial_scale_scalar.set_as(self.spatial_scale)
            self.tik_instance.data_move(self.roi_ub, self.roi_gm, 0, 1, self.n_roi * 16 // 16, 1, 1)
            # repeat time should be less than 255, maybe there is a better way, but algorithm will be complicated
            with self.tik_instance.for_range(0, self.n_roi) as i:
                self.tik_instance.vmuls(16, self.roi_ub[i, 0], self.roi_ub[i, 0], spatial_scale_scalar, 1, 1, 1, 1, 1)

        # calculate rois_height & rois_width & hcenter & wcenter
        self._prepare_calc_roi_width_height_buf()
        self._calc_roi_width_height()

        # calculate hcenter & wcenter
        # hcenter equals to (y + 0.5) * self.bin_size_h
        # wcenter equals to (x + 0.5) * self.bin_size_w
        # [self.align_roi, ]  * [64, ] --> [self.align_roi, 64]
        self._prepare_calc_h_w_center_buf()
        with self.tik_instance.new_stmt_scope():
            self.hcenter_fp32_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                              scope=tik.scope_cbuf, name="hcenter_fp32_cbuf")
            self.wcenter_fp32_cbuf = self.tik_instance.Tensor("float32", [self.align_roi, self.shape_align],
                                                              scope=tik.scope_cbuf, name="wcenter_fp32_cbuf")
            self._h_w_center_processor()

            # hstart equals to min(max(hcenter, 0), (h_feature_map - 1)), not necessary
            # wstart equals to min(max(wcenter, 0), (w_feature_map - 1)), not necessary
            # hstart equals to int(hstart)
            # wstart equals to int(wstart)
            self._calc_h_w_start_end()

            with self.tik_instance.new_stmt_scope():
                # fX0 equals to wcenter - wstart
                self._calc_fx_0()

                # fX1 equals to wend - wcenter
                self._calc_fx_1()

                # fY0 equals to hcenter - hstart
                self._calc_fy_0()

                # fY1 equals to hend - hcenter
                self._calc_fy_1()

        self._prepare_calc_factor_buf()
        self._calc_factor_a()
        self._calc_factor_b()
        self._calc_factor_c()
        self._calc_factor_d()

        # output[i][j][y][x] equals to feature_map[0][j][hstart][wstart] mutilpy fFactorA
        #                  adds to feature_map[0][j][hstart][wend] mutilpy fFactorB
        #                  adds to feature_map[0][j][hend][wstart] mutilpy fFactorC
        #                  adds to feature_map[0][j][hend][wend] mutilpy fFactorD
        self._prepare_roi_align_processor_buf()
        self._roi_align_processor()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.feature_map_gm, self.roi_gm],
                                   outputs=[self.roi_align_gm], enable_l2=True)

        return self.tik_instance

    def tik_output_debug(self):
        roi_shape = [self.n_roi, self.c1_roi, self.h_roi, self.w_roi, 16]
        feature_map_shape = [self.n_feature_map, self.c1_feature_map, self.h_feature_map, self.w_feature_map, 16]
        roi_data = np.random.uniform(0, 500, roi_shape).astype(np.float16)
        feature_map_data = np.random.uniform(0, 500, feature_map_shape).astype(np.float16)
        feed_dict = {"feature_map_gm": feature_map_data, "roi_gm": roi_data}
        out, = self.tik_instance.tikdb.start_debug(feed_dict, interactive=True)
        np.savetxt("tik_debug.txt", out.reshape(304 * 496, 7 * 7))

    def tik_output_profiling(self):
        roi_shape = [self.n_roi, self.c1_roi, self.h_roi, self.w_roi, 16]
        feature_map_shape = [self.n_feature_map, self.c1_feature_map, self.h_feature_map, self.w_feature_map, 16]
        roi_data = np.random.uniform(0, 500, roi_shape).astype(np.float16)
        feature_map_data = np.random.uniform(0, 500, feature_map_shape).astype(np.float16)
        feed_dict = {"feature_map_gm": feature_map_data, "roi_gm": roi_data}
        self.tik_instance.StartProfiling(feed_dict=feed_dict, simulatorlog_path='./', generate_html=True)

# pylint: disable=unused-argument
def r_oi_align_tik(feature_map_dict, rois_dict, roi_align_dict, pooled_h, pooled_w, spatial_scale,
                   kernel_name='roi_align'):
    """
    :param feature_map_dict 5HD
    :param rois_dict  NX5, ND
    :param roi_align_dict output dict,5HD
    :param pooled_h: pooled height
    :param pooled_w: pooled width
    :param spatial_scale:
    :param kernel_name: roi_align
    :return: tik_instance
    """
    roi_shape = rois_dict.get("shape")
    feature_map_shape = feature_map_dict.get("shape")
    pooled_w_reciprocal = 1.0 / float(pooled_w)
    pooled_h_reciprocal = 1.0 / float(pooled_h)
    obj = ROIAlignTIK(feature_map_shape, roi_shape, pooled_h, spatial_scale, pooled_w_reciprocal,
                      pooled_h_reciprocal, kernel_name)
    return obj.roi_align()
