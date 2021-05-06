# -*- coding:utf-8 -*-
from te import tik
import math

# max allowd roi number
ROINUM = 128

# align with 64 fp16
ALIGN64 = 64

# align with 16 fp16
ALIGN16 = 16

# align with 8 int32/fp32
ALIGN8 = 8

# align with 128 int32/fp32
ALIGN128 = 128

# mask with 128
MASK128 = 128

# Max allow l1 buffer size
MAX_ALLOW_L1BUF_SIZE = 131072

MAX_CHANNELS = 384
MAX_HEIGHT = 8
MAX_WIDTH = 8


# (a + align - 1) // align
def _ceil_div_offline(value, factor):
    return (value + factor - 1) // factor


# ((a + align - 1) // align) * align
def _ceil_div_mul(value, factor):
    return ((value + factor - 1) // factor) * factor


# floor div by value
def _floor_div_mul(value, factor):
    return (value // factor) * factor


# get max value
def _get_max_value(val_1, val_2):
    max_val = val_1 if val_1 >= val_2 else val_2
    return max_val


class PSROIAlign(object):
    def __init__(self, feature_map_shape, roi_shape, output_map_shape, spatial_scale, output_dim, group_size,
                 sample_height, sample_width, kernel_name):
        """
        Introduction
        ------------
            Intialize PSROIAlign parameter
        Parameters
        ----------
            @feature_map_shape, list or tuple, input feature map of ps roi align
            @roi_shape, list or tuple, input of ps roi align
            @output_map_shape, list or tuple, output of ps roi align
            @spatial_scale, attr, spatial scale, float
            @output_dim, attr, output dim, int32
            @group_size, attr, group size, int32
            @self.sample_height, attr, output shape height, int32
            @self.sample_width, attr, output shape width, int32
            @kernel name
        Returns
        -------
        """
        self.tik_inst = tik.Tik(tik.Dprofile("v100", "mini"))
        self.kernel_name = kernel_name
        self.roi_shape = roi_shape
        self.feature_map_shape = feature_map_shape

        # roi shape
        self.n_roi, self.c_roi = roi_shape[0], roi_shape[1]
        self.n_roi_align = 0

        if self.c_roi != 5:
            raise RuntimeError("invalid roi shape, channel should be equal to 5!")

        # feature_map shape nc1hwc0
        self.n_feature_map, self.c1_feature_map, self.h_feature_map, self.w_feature_map, self.c0_feature_map = \
            feature_map_shape[0], feature_map_shape[1], feature_map_shape[2], feature_map_shape[3], \
            feature_map_shape[4]

        self.input_c, self.input_w, self.input_h = \
            feature_map_shape[1] * feature_map_shape[4], feature_map_shape[2], feature_map_shape[3]

        self.input_plus_four = 4 * feature_map_shape[1] * feature_map_shape[4]

        if group_size <= 0 or group_size > self.h_feature_map or group_size > self.w_feature_map:
            raise RuntimeError("ground_size used to set pooled_h and pooled_w, "
                               "make sure larger than 0 and less than feature_map's height and width")

        # p_sroi_align shape
        self.n_p_sroi_align = self.n_roi
        self.c1_p_sroi_align = (output_dim + ALIGN16 - 1) // ALIGN16
        self.h_p_sroi_align = group_size
        self.w_p_sroi_align = group_size
        self.c0_p_sroi_align = self.c0_feature_map
        self.c_p_sroi_align = output_dim
        self.output_align_c = self.c1_p_sroi_align * self.c0_p_sroi_align

        self.output_wh_1 = group_size * group_size - 1
        self.input_wh_1 = self.h_feature_map * self.w_feature_map - 1
        self.h_scalar, self.l_scalar = 0, 0
        self.width_1, self.height_1 = self.input_w - 1, self.input_h - 1

        self.pooled_h, self.pooled_w, self.pooled_wh = group_size, group_size, group_size * group_size

        self.group_size_reciprocal = 1.0 / group_size
        self.sample_height_reciprocal_plus = 1.0 / (sample_height + 1)
        self.sample_width_reciprocal_plus = 1.0 / (sample_width + 1)
        self.sample_height_reciprocal = 1.0 / sample_height
        self.sample_width_reciprocal = 1.0 / sample_width

        if (self.n_roi * self.pooled_h * self.pooled_w) >= MAX_CHANNELS * MAX_WIDTH * MAX_HEIGHT:
            raise RuntimeError("the size of roi data is too large, since avaiable ub and L1 buffer size is limited")

        self.spatial_scale, self.sample_height, self.sample_width = spatial_scale, sample_height, sample_width
        self.sample_wh_rec = 1.0 / (sample_height * sample_width)

        self.sample_algn_w, self.sample_algn_h = \
            _ceil_div_mul(sample_width, ALIGN8), _ceil_div_mul(sample_height, ALIGN8)

        self.sample_algn_wh = self.sample_algn_w * self.sample_algn_h
        self.algn_pooled_h = _ceil_div_mul(self.pooled_h, ALIGN16)
        self.algn_pooled_w = _ceil_div_mul(self.pooled_w, ALIGN16)
        self._init_parameters()
        self._psroialign_prepare_scalar()

    def _init_parameters(self):
        self.repeat_time = self.input_c // ALIGN128
        self.res_start = self.repeat_time * ALIGN128
        self.res_input = self.input_c - ALIGN128 * self.repeat_time

        self.align_pooled_max = _get_max_value(self.algn_pooled_h, self.algn_pooled_w)
        self.align_sample_max = _get_max_value(self.sample_algn_h, self.sample_algn_w)

        self.group_align = self.algn_pooled_w * self.algn_pooled_h

        # _floor_f32toi32 's data_length align to 64
        if self.group_align < ALIGN64:
            self.group_align = ALIGN64

        self.roi_gm = self.tik_inst.Tensor("float16", (self.n_roi, self.c_roi, self.roi_shape[2], self.roi_shape[3]),
                                           name="roi_gm", scope=tik.scope_gm)
        self.feature_map_gm = self.tik_inst.Tensor("float16", (self.n_feature_map, self.c1_feature_map,
                                                               self.h_feature_map, self.w_feature_map,
                                                               self.c0_feature_map),
                                                   name="feature_map_gm", scope=tik.scope_gm)
        self.output_map_gm = self.tik_inst.Tensor("float16", (self.n_p_sroi_align, self.c1_p_sroi_align,
                                                              self.h_p_sroi_align, self.w_p_sroi_align,
                                                              self.c0_p_sroi_align),
                                                  name="output_map_gm", scope=tik.scope_gm)
        self.total_rois_num = self.roi_shape[0]
        self.aicore_num = 2
        # first core
        self.each_core_rois_num = self.total_rois_num // self.aicore_num
        self.each_core_block_left = self.each_core_rois_num % ROINUM
        self.each_core_block_num = int(math.ceil(float(self.each_core_rois_num) / ROINUM))
        # second core
        self.last_core_rois_num = self.total_rois_num - self.each_core_rois_num * (self.aicore_num - 1)
        self.last_core_block_left = self.last_core_rois_num % ROINUM
        self.last_core_block_num = int(math.ceil(float(self.last_core_rois_num) / ROINUM))

        if ((self.n_feature_map * self.c1_feature_map * self.h_feature_map * self.w_feature_map *
             self.c0_feature_map * 2) > MAX_ALLOW_L1BUF_SIZE):
            raise RuntimeError("shape not supprot yet!")

    # high precise floor algorithm, convert float32 to int32
    def _floor_f32toi32(self, ub_ret, ub_in, data_len, align):
        with self.tik_inst.new_stmt_scope():
            repeat_time = data_len // align
            rep_stride = align // 8
            ub_functmp_f16 = self.tik_inst.Tensor("float16", (_ceil_div_mul(data_len, ALIGN16),),
                                                  name="ub_functmp_f16", scope=tik.scope_ubuf)
            ub_functmp_f32 = self.tik_inst.Tensor("float32", (data_len,), name="ub_functmp_f32",
                                                  scope=tik.scope_ubuf)
            ub_functmp2_f32 = self.tik_inst.Tensor("float32", (data_len,), name="ub_functmp2_f32",
                                                   scope=tik.scope_ubuf)
            ub_functmp_i32 = self.tik_inst.Tensor("int32", (data_len,), name="ub_functmp_i32",
                                                  scope=tik.scope_ubuf)
            ub_16_f32_val0 = self.tik_inst.Tensor("float32", (data_len,), name="ub_16_f32_val0",
                                                  scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(align, ub_16_f32_val0, 0.0, repeat_time, 1, align // 8)

            self.tik_inst.vadds(align, ub_functmp2_f32, ub_in, 0.5, repeat_time, 1, 1,
                                rep_stride, rep_stride)
            self.tik_inst.vconv(align, "none", ub_functmp_f16, ub_functmp2_f32, repeat_time,
                                1, 1, 1, rep_stride)
            self.tik_inst.vconv(align, "floor", ub_functmp_i32, ub_functmp_f16, repeat_time,
                                1, 1, rep_stride, 1)
            self.tik_inst.vconv(align, "none", ub_functmp_f16, ub_functmp_i32, repeat_time,
                                1, 1, 1, rep_stride, 1.0)
            self.tik_inst.vconv(align, "none", ub_functmp2_f32, ub_functmp_f16, repeat_time,
                                1, 1, rep_stride, 1)
            self.tik_inst.vsub(align, ub_functmp2_f32, ub_in, ub_functmp2_f32, repeat_time,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
            # out equals to -1 when in lesser than 0, out equals to 0 when in bigger than 0.
            self.tik_inst.vmin(align, ub_functmp2_f32, ub_functmp2_f32, ub_16_f32_val0, repeat_time,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
            self.tik_inst.vrec(align, ub_functmp_f32, ub_functmp2_f32, repeat_time,
                               1, 1, rep_stride, rep_stride)
            self.tik_inst.vabs(align, ub_functmp_f32, ub_functmp_f32, repeat_time,
                               1, 1, rep_stride, rep_stride)
            self.tik_inst.vmul(align, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, repeat_time,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
            self.tik_inst.vrec(align, ub_functmp_f32, ub_functmp2_f32, repeat_time,
                               1, 1, rep_stride, rep_stride)
            self.tik_inst.vabs(align, ub_functmp_f32, ub_functmp_f32, repeat_time,
                               1, 1, rep_stride, rep_stride)
            self.tik_inst.vmul(align, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, repeat_time,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
            # add 0.5 to make sure result of vconv precisely
            self.tik_inst.vadds(align, ub_functmp2_f32, ub_functmp2_f32, 0.5, repeat_time,
                                1, 1, rep_stride, rep_stride)
            self.tik_inst.vconv(align, "none", ub_functmp_f16, ub_functmp2_f32, repeat_time,
                                1, 1, 1, rep_stride)
            self.tik_inst.vconv(align, "floor", ub_ret, ub_functmp_f16, repeat_time,
                                1, 1, rep_stride, 1)
            self.tik_inst.vadd(align, ub_ret, ub_ret, ub_functmp_i32, repeat_time,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)

    # compare tensor by value
    def _compare_with(self, source_ub, value):
        source_n = source_ub.shape[0]
        small_scalar = self.tik_inst.Scalar("float16", "zero")
        small_scalar.set_as(value)
        iter_num = source_n // ALIGN16
        iter_dual = _floor_div_mul(iter_num, 2)

        # align to 32
        with self.tik_inst.for_range(0, iter_dual, thread_num=2) as i:
            small_tensor = self.tik_inst.Tensor("float16", (source_n,), name="zero_tensor", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(ALIGN16, small_tensor, small_scalar, 1, 1, 1)
            cmp_mask = self.tik_inst.vcmp_gt(ALIGN16, source_ub[ALIGN16 * i], small_tensor[ALIGN16 * i], 1, 1)
            self.tik_inst.vsel(ALIGN16, 0, source_ub[ALIGN16 * i], cmp_mask, source_ub[ALIGN16 * i],
                               small_tensor[ALIGN16 * i], 1, 1, 1, 1)

        with self.tik_inst.for_range(iter_dual, iter_num) as i:
            small_tensor = self.tik_inst.Tensor("float16", (source_n,), name="zero_tensor", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(ALIGN16, small_tensor, small_scalar, 1, 1, 1)
            cmp_mask = self.tik_inst.vcmp_gt(ALIGN16, source_ub[ALIGN16 * i], small_tensor[ALIGN16 * i], 1, 1)
            self.tik_inst.vsel(ALIGN16, 0, source_ub[ALIGN16 * i], cmp_mask, source_ub[ALIGN16 * i],
                               small_tensor[ALIGN16 * i], 1, 1, 1, 1)

    def _transpose_roi(self, trans_ub, roi_ub_fp16):
        with self.tik_inst.for_range(0, self.n_roi_align // ALIGN16) as k:
            # [n_roi_align, 16]
            src_list = [roi_ub_fp16[16 * k + i, 0] for i in range(0, 16)]

            # [16, n_roi_align]
            dst_list = [trans_ub[j, 16 * k] for j in range(0, 16)]
            self.tik_inst.vnchwconv(True, True, dst_list, src_list, 1, 0, 0)

    def _transpose_pooled_w_h(self, pw_tensor_ub, ph_tensor_ub):
        with self.tik_inst.for_range(0, self.algn_pooled_h // ALIGN16) as k:
            # [algn_pooled_h, algn_pooled_w]
            src_list = [ph_tensor_ub[16 * k + i, 0] for i in range(0, 16)]
            # [algn_pooled_w, algn_pooled_h]
            dst_list = [pw_tensor_ub[i, 16 * k] for i in range(0, 16)]
            self.tik_inst.vnchwconv(True, True, dst_list, src_list, 1, 0, 0)

    def _calc_sample_w_h(self):
        with self.tik_inst.new_stmt_scope():
            self.tik_inst.data_move(self.rois_start_w, self.roi_ub[1, 0], 0, 1, self.n_roi_align // 8, 0, 0)
            self.tik_inst.data_move(self.rois_start_h, self.roi_ub[2, 0], 0, 1, self.n_roi_align // 8, 0, 0)
            self.tik_inst.data_move(self.rois_end_w, self.roi_ub[3, 0], 0, 1, self.n_roi_align // 8, 0, 0)
            self.tik_inst.data_move(self.rois_end_h, self.roi_ub[4, 0], 0, 1, self.n_roi_align // 8, 0, 0)

            '''
            roi_end_w equals to (roi_data[indx_roi, 3] + 1) * spatial_scale
            roi_end_h equals to (roi_data[indx_roi, 4] + 1) * spatial_scale
            '''
            self.tik_inst.vadds(ALIGN8, self.rois_end_w, self.rois_end_w, self.spatial_scale_sclr,
                                self.n_roi_align // ALIGN8, 1, 1, 1, 1)
            self.tik_inst.vadds(ALIGN8, self.rois_end_h, self.rois_end_h, self.spatial_scale_sclr,
                                self.n_roi_align // ALIGN8, 1, 1, 1, 1)

            '''
            calculate roi_height and roi_width
            '''
            rep_stride = ALIGN16 // 8
            with self.tik_inst.for_range(0, self.n_roi_align // ALIGN16) as i:
                '''
                roi_height equals to max((roi_end_h - roi_self.start_h), 0.1)
                roi_width equals to max((roi_end_w - roi_self.start_w), 0.1)
                '''
                self.tik_inst.vsub(ALIGN16, self.rois_height[i * ALIGN16], self.rois_end_h[i * ALIGN16],
                                   self.rois_start_h[i * ALIGN16], 1, 1, 1, 1, rep_stride, rep_stride,
                                   rep_stride)
                self.tik_inst.vsub(ALIGN16, self.rois_width[i * ALIGN16], self.rois_end_w[i * ALIGN16],
                                   self.rois_start_w[i * ALIGN16], 1, 1, 1, 1, rep_stride, rep_stride,
                                   rep_stride)
                rois_height_fp16 = self.tik_inst.Tensor("float16", (ALIGN16,), name="rois_height_fp16",
                                                        scope=tik.scope_ubuf)
                rois_width_fp16 = self.tik_inst.Tensor("float16", (ALIGN16,), name="rois_width_fp16",
                                                       scope=tik.scope_ubuf)
                self.tik_inst.vconv(ALIGN16, 'none', rois_height_fp16, self.rois_height[i * ALIGN16],
                                    1, 1, 1, 1, rep_stride)
                self.tik_inst.vconv(ALIGN16, 'none', rois_width_fp16, self.rois_width[i * ALIGN16],
                                    1, 1, 1, 1, rep_stride)
                self._compare_with(rois_height_fp16, 0.1)
                self._compare_with(rois_width_fp16, 0.1)
                self.tik_inst.vconv(ALIGN16, 'none', self.rois_height[i * ALIGN16], rois_height_fp16,
                                    1, 1, 1, rep_stride, 1)
                self.tik_inst.vconv(ALIGN16, 'none', self.rois_width[i * ALIGN16], rois_width_fp16,
                                    1, 1, 1, rep_stride, 1)
                '''
                bin_size_h equals to roi_height / group_size
                bin_size_w equals to roi_width / group_size
                self.sample_h equals to bin_size_h / (self.sample_height + 1)
                self.sample_w equals to bin_size_w / (self.sample_width + 1)
                '''
                self.tik_inst.vmuls(ALIGN16, self.bin_size_w[i * ALIGN16], self.rois_width[i * ALIGN16],
                                    self.group_size_reciprocal, 1, 1, 1, rep_stride, rep_stride)
                self.tik_inst.vmuls(ALIGN16, self.bin_size_h[i * ALIGN16], self.rois_height[i * ALIGN16],
                                    self.group_size_reciprocal, 1, 1, 1, rep_stride, rep_stride)
                self.tik_inst.vmuls(ALIGN16, self.sample_w[i * ALIGN16], self.bin_size_w[i * ALIGN16],
                                    self.sample_width_reciprocal_plus, 1, 1, 1, rep_stride, rep_stride)
                self.tik_inst.vmuls(ALIGN16, self.sample_h[i * ALIGN16], self.bin_size_h[i * ALIGN16],
                                    self.sample_height_reciprocal_plus, 1, 1, 1, rep_stride, rep_stride)

    def _calc_start_w_h(self, pw_tensor, ph_tensor):
        """
        ## calculate hstart & wstart
        hstart equals to ph * bin_size_h
        wstart equals to pw * bin_size_w
        hend equals to (ph + 1) * bin_size_h
        wend equals to (pw + 1) * bin_size_w
        """
        rep_stride = ALIGN16 // 8
        with self.tik_inst.for_range(0, self.n_roi_align) as i:
            self.temp_sclr.set_as(self.bin_size_h[i])
            self.tik_inst.vmuls(ALIGN16, self.hstart[i, 0], ph_tensor[0], self.temp_sclr,
                                self.group_align // ALIGN16, 1, 1, rep_stride, rep_stride)
            self.temp_sclr.set_as(self.bin_size_w[i])
            self.tik_inst.vmuls(ALIGN16, self.wstart[i, 0], pw_tensor[0], self.temp_sclr,
                                self.group_align // ALIGN16, 1, 1, rep_stride, rep_stride)

        self.tik_inst.vadds(ALIGN16, ph_tensor, ph_tensor, 1.0, self.group_align // ALIGN16,
                            1, 1, rep_stride, rep_stride)
        self.tik_inst.vadds(ALIGN16, pw_tensor, pw_tensor, 1.0, self.group_align // ALIGN16,
                            1, 1, rep_stride, rep_stride)

        with self.tik_inst.for_range(0, self.n_roi_align) as i:
            self.temp_sclr.set_as(self.bin_size_h[i])
            self.tik_inst.vmuls(ALIGN16, self.hend[i, 0], ph_tensor[0], self.temp_sclr, self.group_align // ALIGN16,
                                1, 1, rep_stride, rep_stride)
            self.temp_sclr.set_as(self.bin_size_w[i])
            self.tik_inst.vmuls(ALIGN16, self.wend[i, 0], pw_tensor[0], self.temp_sclr, self.group_align // ALIGN16,
                                1, 1, rep_stride, rep_stride)

    def _calc_end_w_h(self):
        """
        hstart equals to min(max(hstart adds to roi_self.start_h, 0), input_height minus to 1)
        hend equals to min(max(hend adds to roi_self.start_h, 0), input_height minus to 1)
        wstart equals to min(max(wstart adds to roi_self.start_w, 0), input_width minus to 1)
        wend v min(max(wend adds to roi_self.start_w, 0), input_width minus to 1)
        """
        rep_stride = ALIGN16 // 8
        with self.tik_inst.for_range(0, self.n_roi_align) as i:
            self.temp_sclr.set_as(self.rois_start_h[i])
            self.tik_inst.vadds(ALIGN16, self.hstart[i, 0], self.hstart[i, 0], self.temp_sclr,
                                self.group_align // ALIGN16, 1, 1, rep_stride, rep_stride)
            self.tik_inst.vadds(ALIGN16, self.hend[i, 0], self.hend[i, 0], self.temp_sclr,
                                self.group_align // ALIGN16, 1, 1, rep_stride, rep_stride)
            self.temp_sclr.set_as(self.rois_start_w[i])
            self.tik_inst.vadds(ALIGN16, self.wstart[i, 0], self.wstart[i, 0], self.temp_sclr,
                                self.group_align // ALIGN16, 1, 1, rep_stride, rep_stride)
            self.tik_inst.vadds(ALIGN16, self.wend[i, 0], self.wend[i, 0], self.temp_sclr,
                                self.group_align // ALIGN16, 1, 1, rep_stride, rep_stride)
        self.tik_inst.data_move(self.hstart_cbuf, self.hstart, 0, 1, self.n_roi_align * self.group_align // 8, 1, 1)
        self.tik_inst.data_move(self.wstart_cbuf, self.wstart, 0, 1, self.n_roi_align * self.group_align // 8, 1, 1)
        self.tik_inst.data_move(self.hend_cbuf, self.hend, 0, 1, self.n_roi_align * self.group_align // 8, 1, 1)
        self.tik_inst.data_move(self.wend_cbuf, self.wend, 0, 1, self.n_roi_align * self.group_align // 8, 1, 1)

    def _calc_start_end_w_h(self):
        with self.tik_inst.new_stmt_scope():
            self.hstart = self.tik_inst.Tensor("float32", [self.n_roi_align, self.group_align],
                                               scope=tik.scope_ubuf, name="hstart")
            self.wstart = self.tik_inst.Tensor("float32", [self.n_roi_align, self.group_align],
                                               scope=tik.scope_ubuf, name="wstart")
            self.hend = self.tik_inst.Tensor("float32", [self.n_roi_align, self.group_align],
                                             scope=tik.scope_ubuf, name="hend")
            self.wend = self.tik_inst.Tensor("float32", [self.n_roi_align, self.group_align],
                                             scope=tik.scope_ubuf, name="wend")
            ph_tensor_ub = self.tik_inst.Tensor("float16", [self.algn_pooled_h, self.algn_pooled_w],
                                                scope=tik.scope_ubuf, name="ph_tensor_ub")
            pw_tensor_ub = self.tik_inst.Tensor("float16", [self.algn_pooled_w, self.algn_pooled_h],
                                                scope=tik.scope_ubuf, name="pw_tensor_ub")
            ph_tensor = self.tik_inst.Tensor("float32", [self.algn_pooled_h, self.algn_pooled_w],
                                             scope=tik.scope_ubuf, name="y_tensor")
            pw_tensor = self.tik_inst.Tensor("float32", [self.algn_pooled_w, self.algn_pooled_h],
                                             scope=tik.scope_ubuf, name="x_tensor")

            self.tik_inst.vmuls(ALIGN16, ph_tensor_ub[0, 0], ph_tensor_ub[0, 0], 0.,
                                self.algn_pooled_w * self.algn_pooled_h // ALIGN16, 1, 1, 1, 1)
            self.tik_inst.vmuls(ALIGN16, pw_tensor_ub[0, 0], pw_tensor_ub[0, 0], 0.,
                                self.algn_pooled_w * self.algn_pooled_h // ALIGN16, 1, 1, 1, 1)
            '''
            prepare ph and pw
            '''
            for i in range(0, self.algn_pooled_h):
                self.tik_inst.vector_dup(ALIGN16, ph_tensor_ub[i, 0], i, 1, 1, self.algn_pooled_w // ALIGN16)

            # transpose
            self._transpose_pooled_w_h(pw_tensor_ub, ph_tensor_ub)

            with self.tik_inst.for_range(1, self.algn_pooled_w // ALIGN16) as k:
                self.tik_inst.data_move(pw_tensor_ub[ALIGN16 * k, 0], pw_tensor_ub[0, 0],
                                        0, 1, self.algn_pooled_h, 0, 0)

            self.tik_inst.vconv(ALIGN16, 'none', ph_tensor, ph_tensor_ub,
                                self.group_align // ALIGN16, 1, 1, ALIGN16 // 8, 1)
            self.tik_inst.vconv(ALIGN16, 'none', pw_tensor, pw_tensor_ub,
                                self.group_align // ALIGN16, 1, 1, ALIGN16 // 8, 1)

            self._calc_start_w_h(pw_tensor, ph_tensor)
            self._calc_end_w_h()

    def _psroialign_prepare(self, block_num, block_offset, block_left):
        """
        Introduction
        ------------
            prepare PSROIAlign parameter
        Parameters
        ----------
            @block_num, IN: block number for muti_core, int32
            @block_offset, IN: block offset for muti_core, int32
            @block_left, IN: block left for muti_core, int32
            @rois_height, IN: rois height ub tensor, float, 1-D
            @rois_width, IN: rois width ub tensor, float, 1-D
            @self.sample_h, IN: sample height, float
            @self.sample_w, IN: sample width, float
            @self.hstart_cbuf, IN/OUT: L1 buffer tensor, float, 3-D
            @self.wstart_cbuf, IN/OUT: L1 buffer tensor, float, 3-D
            @self.hend_cbuf, IN/OUT: L1 buffer tensor, float, 3-D
            @self.wend_cbuf, IN/OUT: L1 buffer tensor, float, 3-D
        Returns
        -------
        """
        self.temp_sclr = self.tik_inst.Scalar("float32", "temp_sclr")
        with self.tik_inst.new_stmt_scope():
            self.rois_height = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="rois_height",
                                                    scope=tik.scope_ubuf)
            self.rois_width = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="rois_width",
                                                   scope=tik.scope_ubuf)
            # (n_roi_align * 16 * 2) // 1024 kB
            self.roi_ub = self.tik_inst.Tensor("float32", (ALIGN16, self.n_roi_align), name="roi_ub",
                                               scope=tik.scope_ubuf)
            self.spatial_scale_sclr = self.tik_inst.Scalar("float32", "spatial_scale_scalar")
            self.spatial_scale_sclr.set_as(self.spatial_scale)
            '''
            rois[][] * self.spatial_scale
            '''
            with self.tik_inst.new_stmt_scope():
                roi_ub_fp16 = self.tik_inst.Tensor("float16", (self.n_roi_align, ALIGN16), name="roi_ub_fp16",
                                                   scope=tik.scope_ubuf)
                trans_ub = self.tik_inst.Tensor("float16", (ALIGN16, self.n_roi_align), name="trans_ub",
                                                scope=tik.scope_ubuf)
                with self.tik_inst.for_range(0, block_left) as i:
                    self.tik_inst.data_move(roi_ub_fp16[i, 0], self.roi_gm[i + block_offset, 0, 0, 0], 0, 1, 1, 0, 0)

                # transpose
                self._transpose_roi(trans_ub, roi_ub_fp16)
                self.tik_inst.vconv(ALIGN16, 'none', self.roi_ub, trans_ub, self.n_roi_align, 1, 1, ALIGN16 // 8, 1)

                # roi_self.start_w/h, roi_end_w/h equals to roi_data[indx_roi, :4] * spatial_scale
                self.tik_inst.vmuls(ALIGN16, self.roi_ub, self.roi_ub, self.spatial_scale_sclr, self.n_roi_align,
                                    1, 1, ALIGN16 // 8, ALIGN16 // 8)

            '''
            calculate roi_self.start_w / roi_self.start_h / roi_end_w / roi_end_h
            '''
            self.bin_size_h = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="bin_size_h",
                                                   scope=tik.scope_ubuf)
            self.bin_size_w = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="bin_size_w",
                                                   scope=tik.scope_ubuf)
            self.rois_start_w = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="rois_start_w",
                                                     scope=tik.scope_ubuf)
            self.rois_start_h = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="rois_start_h",
                                                     scope=tik.scope_ubuf)
            self.rois_end_w = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="rois_end_w",
                                                   scope=tik.scope_ubuf)
            self.rois_end_h = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="rois_end_h",
                                                   scope=tik.scope_ubuf)
            self._calc_sample_w_h()
            self._calc_start_end_w_h()

    def _calc_bilinear_w(self, cur_h_ub, cur_w_ub):
        """
        l_dh equals to h - h_low,
        l_dw equals to w - w_low,
        h_dh equals to 1 - l_dh,
        h_dw equals to 1 - l_dw
        """
        rep_stride = ALIGN8 // 8
        # l_dh
        self.tik_inst.vconv(ALIGN8, 'none', self.tmp_fp16_ub, self.h_low_ub, self.sample_algn_h // ALIGN8,
                            1, 1, 1, rep_stride, deqscale=1.0)
        self.tik_inst.vconv(ALIGN8, 'none', self.tmp_fp32_ub, self.tmp_fp16_ub, self.sample_algn_h // ALIGN8,
                            1, 1, rep_stride, 1)
        self.tik_inst.vsub(ALIGN8, self.l_dh_ub, cur_h_ub, self.tmp_fp32_ub, self.sample_algn_h // ALIGN8,
                           1, 1, 1, rep_stride, rep_stride, rep_stride)
        # l_dw
        self.tik_inst.vconv(ALIGN8, 'none', self.tmp_fp16_ub, self.w_low_ub, self.sample_algn_w // ALIGN8,
                            1, 1, 1, rep_stride, deqscale=1.0)
        self.tik_inst.vconv(ALIGN8, 'none', self.tmp_fp32_ub, self.tmp_fp16_ub, self.sample_algn_w // ALIGN8,
                            1, 1, rep_stride, 1)
        self.tik_inst.vsub(ALIGN8, self.l_dw_ub, cur_w_ub, self.tmp_fp32_ub, self.sample_algn_w // ALIGN8,
                           1, 1, 1, rep_stride, rep_stride, rep_stride)
        # h_dh
        self.tik_inst.vconv(ALIGN8, 'none', self.tmp_fp16_ub, self.l_dh_ub, self.sample_algn_h // ALIGN8,
                            1, 1, 1, rep_stride)
        self.tik_inst.vconv(ALIGN8, 'none', self.tmp_fp32_ub, self.tmp_fp16_ub, self.sample_algn_h // ALIGN8,
                            1, 1, rep_stride, 1)
        self.tik_inst.vsub(ALIGN8, self.h_dh_ub, self.one_fp32_ub, self.tmp_fp32_ub, self.sample_algn_h // ALIGN8,
                           1, 1, 1, rep_stride, rep_stride, rep_stride)
        # h_dw
        self.tik_inst.vconv(ALIGN8, 'none', self.tmp_fp16_ub, self.l_dw_ub, self.sample_algn_w // ALIGN8,
                            1, 1, 1, rep_stride)
        self.tik_inst.vconv(ALIGN8, 'none', self.tmp_fp32_ub, self.tmp_fp16_ub, self.sample_algn_w // ALIGN8,
                            1, 1, rep_stride, 1)
        self.tik_inst.vsub(ALIGN8, self.h_dw_ub, self.one_fp32_ub, self.tmp_fp32_ub, self.sample_algn_w // ALIGN8,
                           1, 1, 1, rep_stride, rep_stride, rep_stride)

        """
        w1 equals to h_dh * h_dw,
        w2 equals to h_dh * l_dw,
        w3 equals to l_dh * h_dw,
        w4 equals to l_dh * l_dw
        """
        with self.tik_inst.for_range(0, self.sample_height) as iter_i:
            self.h_scalar.set_as(self.h_dh_ub[iter_i])
            self.l_scalar.set_as(self.l_dh_ub[iter_i])
            self.tik_inst.vmuls(ALIGN8, self.w1_ub[iter_i, 0], self.h_dw_ub[0], self.h_scalar,
                                self.sample_algn_w // ALIGN8,
                                1, 1, rep_stride, rep_stride)
            self.tik_inst.vmuls(ALIGN8, self.w2_ub[iter_i, 0], self.l_dw_ub[0], self.h_scalar,
                                self.sample_algn_w // ALIGN8,
                                1, 1, rep_stride, rep_stride)
            self.tik_inst.vmuls(ALIGN8, self.w3_ub[iter_i, 0], self.h_dw_ub[0], self.l_scalar,
                                self.sample_algn_w // ALIGN8,
                                1, 1, rep_stride, rep_stride)
            self.tik_inst.vmuls(ALIGN8, self.w4_ub[iter_i, 0], self.l_dw_ub[0], self.l_scalar,
                                self.sample_algn_w // ALIGN8,
                                1, 1, rep_stride, rep_stride)

    def _prepare_bilinear_interpolate(self, cur_h_ub, cur_w_ub):
        """
        Introduction
        ------------
            prepare bilinear interpolate parameter
        Parameters
        ----------
            @cur_h_ub, IN: cur_h ub tensor, float, 1-D, self.sample_height
            @cur_w_ub, IN: cur_w ub tensor, float, 1-D, self.sample_height
            @self.w1_ub, IN/OUT: w1 ub tensor, float, 2-D, [self.sample_height, sample_algn_w]
            @self.w2_ub, IN/OUT: w2 ub tensor, float, 2-D, [self.sample_height, sample_algn_w]
            @self.w3_ub, IN/OUT: w3 ub tensor, float, 2-D, [self.sample_height, sample_algn_w]
            @self.w4_ub, IN/OUT: w3 ub tensor, float, 2-D, [self.sample_height, sample_algn_w]
            @self.h_low_ub, IN/OUT: height low ub tensor, float, 2-D, [self.sample_height, sample_algn_w]
            @self.w_low_ub, IN/OUT: width  low ub tensor, float, 2-D, [self.sample_height, sample_algn_w]
            @self.h_high_ub, IN/OUT: height high ub tensor, float, 2-D, [self.sample_height, sample_algn_w]
            @self.w_high_ub, IN/OUT: height low ub tensor, float, 2-D, [self.sample_height, sample_algn_w]
        Returns
        -------
        """
        rep_stride = ALIGN8 // 8
        with self.tik_inst.new_stmt_scope():
            self.h_scalar = self.tik_inst.Scalar("float32")
            self.l_scalar = self.tik_inst.Scalar("float32")
            self.l_dh_ub = self.tik_inst.Tensor("float32", (self.sample_algn_h,), scope=tik.scope_ubuf,
                                                name="l_dh_ub")
            self.l_dw_ub = self.tik_inst.Tensor("float32", (self.sample_algn_w,), scope=tik.scope_ubuf,
                                                name="l_dw_ub")
            self.h_dh_ub = self.tik_inst.Tensor("float32", (self.sample_algn_h,), scope=tik.scope_ubuf,
                                                name="h_dh_ub")
            self.h_dw_ub = self.tik_inst.Tensor("float32", (self.sample_algn_w,), scope=tik.scope_ubuf,
                                                name="h_dw_ub")
            self.tmp_fp16_ub = self.tik_inst.Tensor("float16", (_ceil_div_mul(self.align_sample_max, ALIGN8),),
                                                    scope=tik.scope_ubuf, name="tmp_fp16_ub")
            self.tmp_fp32_ub = self.tik_inst.Tensor("float32", (self.align_sample_max,), scope=tik.scope_ubuf,
                                                    name="tmp_fp32_ub")
            self.tmp_int_ub = self.tik_inst.Tensor("int32", (self.align_sample_max,), scope=tik.scope_ubuf,
                                                   name="tmp_int_ub")
            self.one_fp32_ub = self.tik_inst.Tensor("float32", (self.align_sample_max,), scope=tik.scope_ubuf,
                                                    name="one_fp32_ub")
            self.one_tensor = self.tik_inst.Tensor("int32", (self.align_sample_max,), scope=tik.scope_ubuf,
                                                   name="one_tensor")

            # broadcast to 1 tensor
            self.tik_inst.vector_dup(ALIGN8, self.one_tensor, 1, self.align_sample_max // ALIGN8, 1, ALIGN8 // 8)
            self.tik_inst.vector_dup(ALIGN8, self.one_fp32_ub, 1.0, self.align_sample_max // ALIGN8, 1, ALIGN8 // 8)
            '''
            if (h lesser than 0): h equals to 0
            if (w lesser than 0): w equals to 0
            '''
            self.tik_inst.vector_dup(ALIGN8, self.tmp_fp32_ub, 0.0, self.align_sample_max // ALIGN8, 1, ALIGN8 // 8)
            self.tik_inst.vmax(ALIGN8, cur_h_ub, cur_h_ub, self.tmp_fp32_ub, self.sample_algn_h // ALIGN8,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
            self.tik_inst.vmax(ALIGN8, cur_w_ub, cur_w_ub, self.tmp_fp32_ub, self.sample_algn_w // ALIGN8,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)

            '''
            h_low equals to int(h), w_low equals to int(w)
            '''
            self._floor_f32toi32(self.h_low_ub, cur_h_ub, self.sample_algn_h, ALIGN8)
            self._floor_f32toi32(self.w_low_ub, cur_w_ub, self.sample_algn_w, ALIGN8)

            '''
            if(w_low bigger than width - 1):
                w_low equals to width -1
                w_high equals to width - 1
                w equals to w_low
            else:
                w_high equals to w_low + 1
            '''
            self.tik_inst.vector_dup(ALIGN8, self.tmp_int_ub, self.width_1, self.align_sample_max // ALIGN8,
                                     1, rep_stride)
            self.tik_inst.vadd(ALIGN8, self.w_high_ub, self.w_low_ub, self.one_tensor, self.sample_algn_w // ALIGN8,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
            self.tik_inst.vmin(ALIGN8, self.w_low_ub, self.w_low_ub, self.tmp_int_ub, self.sample_algn_w // ALIGN8,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
            self.tik_inst.vmin(ALIGN8, self.w_high_ub, self.w_high_ub, self.tmp_int_ub, self.sample_algn_w // ALIGN8,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)

            '''
            if(h_low bigger than height - 1):
                h_high equals to height - 1
                h_low equals to height - 1
                h equals to h_low
            else:
                h_high equals to h_low + 1
            '''
            self.tik_inst.vector_dup(ALIGN8, self.tmp_int_ub, self.height_1, self.align_sample_max // ALIGN8,
                                     1, rep_stride)
            self.tik_inst.vadd(ALIGN8, self.h_high_ub, self.h_low_ub, self.one_tensor, self.sample_algn_h // ALIGN8,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
            self.tik_inst.vmin(ALIGN8, self.h_low_ub, self.h_low_ub, self.tmp_int_ub, self.sample_algn_h // ALIGN8,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)
            self.tik_inst.vmin(ALIGN8, self.h_high_ub, self.h_high_ub, self.tmp_int_ub, self.sample_algn_h // ALIGN8,
                               1, 1, 1, rep_stride, rep_stride, rep_stride)

            self._calc_bilinear_w(cur_h_ub, cur_w_ub)

    # bilinear_interpolate with float16
    def _bilinear_interpolate(self, input_map, w_1, w_2, w_3, w_4, h_low, w_low, h_high, w_high):
        if self.input_c > MASK128:
            self.tik_inst.vmuls(MASK128, self.u1_ub[0], input_map[h_low, w_low, 0], w_1, self.repeat_time,
                                1, 1, 8, 8)
            self.tik_inst.vmuls(MASK128, self.u1_ub[self.input_c], input_map[h_low, w_high, 0], w_2, self.repeat_time,
                                1, 1, 8, 8)
            self.tik_inst.vmuls(MASK128, self.u1_ub[2 * self.input_c], input_map[h_high, w_low, 0], w_3,
                                self.repeat_time, 1, 1, 8, 8)
            self.tik_inst.vmuls(MASK128, self.u1_ub[3 * self.input_c], input_map[h_high, w_high, 0], w_4,
                                self.repeat_time, 1, 1, 8, 8)
            if self.res_input > 0:
                self.tik_inst.vmuls(self.res_input, self.u1_ub[self.res_start],
                                    input_map[h_low, w_low, self.res_start], w_1,
                                    1, 1, 1, self.res_input // ALIGN16, self.res_input // ALIGN16)
                self.tik_inst.vmuls(self.res_input, self.u1_ub[self.input_c + self.res_start],
                                    input_map[h_low, w_high, self.res_start], w_2,
                                    1, 1, 1, self.res_input // ALIGN16, self.res_input // ALIGN16)
                self.tik_inst.vmuls(self.res_input, self.u1_ub[2 * self.input_c + self.res_start],
                                    input_map[h_high, w_low, self.res_start], w_3,
                                    1, 1, 1, self.res_input // ALIGN16, self.res_input // ALIGN16)
                self.tik_inst.vmuls(self.res_input, self.u1_ub[3 * self.input_c + self.res_start],
                                    input_map[h_high, w_high, self.res_start], w_4, 1,
                                    1, 1, self.res_input // ALIGN16, self.res_input // ALIGN16)
        else:
            self.tik_inst.vmuls(self.input_c, self.u1_ub[0], input_map[h_low, w_low, 0], w_1, 1,
                                1, 1, self.input_c // ALIGN16, self.input_c // ALIGN16)
            self.tik_inst.vmuls(self.input_c, self.u1_ub[self.input_c], input_map[h_low, w_high, 0], w_2, 1,
                                1, 1, self.input_c // ALIGN16, self.input_c // ALIGN16)
            self.tik_inst.vmuls(self.input_c, self.u1_ub[2 * self.input_c], input_map[h_high, w_low, 0], w_3, 1,
                                1, 1, self.input_c // ALIGN16, self.input_c // ALIGN16)
            self.tik_inst.vmuls(self.input_c, self.u1_ub[3 * self.input_c], input_map[h_high, w_high, 0], w_4, 1,
                                1, 1, self.input_c // ALIGN16, self.input_c // ALIGN16)

        if self.input_plus_four > MASK128:
            repeat_time = self.input_plus_four // MASK128
            res_start = repeat_time * MASK128
            res_input = self.input_plus_four - MASK128 * repeat_time
            self.tik_inst.vadd(MASK128, self.u2_ub, self.u2_ub, self.u1_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            if res_input > 0:
                self.tik_inst.vadd(res_input, self.u2_ub[res_start], self.u2_ub[res_start], self.u1_ub[res_start], 1,
                                   1, 1, 1, res_input // ALIGN16, res_input // ALIGN16, res_input // ALIGN16)
        else:
            self.tik_inst.vadd(self.input_plus_four, self.u2_ub, self.u2_ub, self.u1_ub, 1, 1, 1, 1,
                               self.input_plus_four // ALIGN16, self.input_plus_four // ALIGN16,
                               self.input_plus_four // ALIGN16)

    def _clear_zero_ub(self, temp_ub):
        if self.input_c > MASK128:
            self.tik_inst.vector_dup(MASK128, temp_ub, 0.0, self.repeat_time, 1, 8)
            if self.res_input > 0:
                self.tik_inst.vector_dup(self.res_input, temp_ub[self.res_start], 0.0, 1, 1,
                                         self.res_input // ALIGN16)
        else:
            self.tik_inst.vector_dup(self.input_c, temp_ub, 0.0, 1, 1, self.input_c // ALIGN16)

    def _tensor_mul_value(self, temp_ub):
        if self.input_c > MASK128:
            self.tik_inst.vmuls(MASK128, temp_ub, temp_ub, self.sample_wh_rec,
                                self.repeat_time, 1, 1, 8, 8)
            if self.res_input > 0:
                self.tik_inst.vmuls(self.res_input, temp_ub[self.res_start],
                                    temp_ub[self.res_start], self.sample_wh_rec, 1, 1, 1,
                                    self.res_input // ALIGN16, self.res_input // ALIGN16)
        else:
            self.tik_inst.vmuls(self.input_c, temp_ub, temp_ub, self.sample_wh_rec,
                                1, 1, 1, self.input_c // ALIGN16, self.input_c // ALIGN16)

    def _tensor_interplate(self):
        with self.tik_inst.for_range(0, self.sample_height) as self.sample_h:
            self.h_low_sclar.set_as(self.h_low_ub[self.sample_h])
            self.h_high_sclar.set_as(self.h_high_ub[self.sample_h])
            with self.tik_inst.for_range(0, self.sample_width) as self.sample_w:
                self.w_low_sclar.set_as(self.w_low_ub[self.sample_w])
                self.w_high_sclar.set_as(self.w_high_ub[self.sample_w])
                self.w1_sclar.set_as(self.w1_fp16_ub[self.sample_h, self.sample_w])
                self.w2_sclar.set_as(self.w2_fp16_ub[self.sample_h, self.sample_w])
                self.w3_sclar.set_as(self.w3_fp16_ub[self.sample_h, self.sample_w])
                self.w4_sclar.set_as(self.w4_fp16_ub[self.sample_h, self.sample_w])
                self._bilinear_interpolate(self.input_ub, self.w1_sclar, self.w2_sclar, self.w3_sclar, self.w4_sclar,
                                           self.h_low_sclar, self.w_low_sclar, self.h_high_sclar, self.w_high_sclar)

    def _tensor_add(self):
        if self.input_c > MASK128:
            self.tik_inst.vadd(MASK128, self.acc_fp16_ub, self.acc_fp16_ub, self.u2_ub[0],
                               self.repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(MASK128, self.acc_fp16_ub, self.acc_fp16_ub, self.u2_ub[self.input_c],
                               self.repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(MASK128, self.acc_fp16_ub, self.acc_fp16_ub, self.u2_ub[2 * self.input_c],
                               self.repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(MASK128, self.acc_fp16_ub, self.acc_fp16_ub, self.u2_ub[3 * self.input_c],
                               self.repeat_time, 1, 1, 1, 8, 8, 8)

            if self.res_input > 0:
                self.tik_inst.vadd(self.res_input, self.acc_fp16_ub[self.res_start],
                                   self.acc_fp16_ub[self.res_start], self.u2_ub[self.res_start],
                                   1, 1, 1, 1, self.res_input // ALIGN16,
                                   self.res_input // ALIGN16, self.res_input // ALIGN16)
                self.tik_inst.vadd(self.res_input, self.acc_fp16_ub[self.res_start],
                                   self.acc_fp16_ub[self.res_start],
                                   self.u2_ub[self.input_c + self.res_start],
                                   1, 1, 1, 1, self.res_input // ALIGN16,
                                   self.res_input // ALIGN16, self.res_input // ALIGN16)
                self.tik_inst.vadd(self.res_input, self.acc_fp16_ub[self.res_start],
                                   self.acc_fp16_ub[self.res_start],
                                   self.u2_ub[2 * self.input_c + self.res_start],
                                   1, 1, 1, 1, self.res_input // ALIGN16,
                                   self.res_input // ALIGN16, self.res_input // ALIGN16)
                self.tik_inst.vadd(self.res_input, self.acc_fp16_ub[self.res_start],
                                   self.acc_fp16_ub[self.res_start],
                                   self.u2_ub[3 * self.input_c + self.res_start],
                                   1, 1, 1, 1, self.res_input // ALIGN16,
                                   self.res_input // ALIGN16, self.res_input // ALIGN16)
        else:
            self.tik_inst.vadd(self.input_c, self.acc_fp16_ub, self.acc_fp16_ub, self.u2_ub[0],
                               1, 1, 1, self.input_c // ALIGN16, self.input_c // ALIGN16,
                               self.input_c // ALIGN16)
            self.tik_inst.vadd(self.input_c, self.acc_fp16_ub, self.acc_fp16_ub, self.u2_ub[self.input_c],
                               1, 1, 1, self.input_c // ALIGN16, self.input_c // ALIGN16,
                               self.input_c // ALIGN16)
            self.tik_inst.vadd(self.input_c, self.acc_fp16_ub, self.acc_fp16_ub, self.u2_ub[2 * self.input_c],
                               1, 1, 1, self.input_c // ALIGN16, self.input_c // ALIGN16,
                               self.input_c // ALIGN16)
            self.tik_inst.vadd(self.input_c, self.acc_fp16_ub, self.acc_fp16_ub, self.u2_ub[3 * self.input_c],
                               1, 1, 1, self.input_c // ALIGN16, self.input_c // ALIGN16,
                               self.input_c // ALIGN16)

    def _psroialign_prepare_scalar(self):
        self.h_scalar = self.tik_inst.Scalar("float32")
        self.l_scalar = self.tik_inst.Scalar("float32")
        self.h_low_sclar = self.tik_inst.Scalar("int32")
        self.w_low_sclar = self.tik_inst.Scalar("int32")
        self.h_high_sclar = self.tik_inst.Scalar("int32")
        self.w_high_sclar = self.tik_inst.Scalar("int32")
        self.w1_sclar = self.tik_inst.Scalar("float16")
        self.w2_sclar = self.tik_inst.Scalar("float16")
        self.w3_sclar = self.tik_inst.Scalar("float16")
        self.w4_sclar = self.tik_inst.Scalar("float16")
        self.start_h = self.tik_inst.Scalar("float32")
        self.start_w = self.tik_inst.Scalar("float32")
        self.tmp_sclar = self.tik_inst.Scalar("float32")

    def _psroialign_prepare_buf(self):
        self.hstart_ub = self.tik_inst.Tensor("float32", (self.algn_pooled_h, self.algn_pooled_w),
                                              scope=tik.scope_ubuf, name="hstart_ub")
        self.wstart_ub = self.tik_inst.Tensor("float32", (self.algn_pooled_h, self.algn_pooled_w),
                                              scope=tik.scope_ubuf, name="wstart_ub")
        self.hend_ub = self.tik_inst.Tensor("float32", (self.algn_pooled_h, self.algn_pooled_w),
                                            scope=tik.scope_ubuf, name="hend_ub")
        self.wend_ub = self.tik_inst.Tensor("float32", (self.algn_pooled_h, self.algn_pooled_w),
                                            scope=tik.scope_ubuf, name="wend_ub")
        self.hind_const_ub = self.tik_inst.Tensor("float32", (self.sample_algn_h,), scope=tik.scope_ubuf,
                                                  name="hind_const_ub")
        self.wind_const_ub = self.tik_inst.Tensor("float32", (self.sample_algn_w,), scope=tik.scope_ubuf,
                                                  name="wind_const_ub")
        self.hind_ub = self.tik_inst.Tensor("float32", (self.sample_algn_h,), scope=tik.scope_ubuf, name="hind_ub")
        self.wind_ub = self.tik_inst.Tensor("float32", (self.sample_algn_w,), scope=tik.scope_ubuf, name="wind_ub")

        self.hind_tmp_ub = self.tik_inst.Tensor("float32", (self.sample_algn_h,), scope=tik.scope_ubuf,
                                                name="hind_tmp_ub")
        self.wind_tmp_ub = self.tik_inst.Tensor("float32", (self.sample_algn_w,), scope=tik.scope_ubuf,
                                                name="wind_tmp_ub")

        self.w1_ub = self.tik_inst.Tensor("float32", (self.sample_algn_h, self.sample_algn_w),
                                          scope=tik.scope_ubuf, name="w1_ub")
        self.w2_ub = self.tik_inst.Tensor("float32", (self.sample_algn_h, self.sample_algn_w),
                                          scope=tik.scope_ubuf, name="w2_ub")
        self.w3_ub = self.tik_inst.Tensor("float32", (self.sample_algn_h, self.sample_algn_w),
                                          scope=tik.scope_ubuf, name="w3_ub")
        self.w4_ub = self.tik_inst.Tensor("float32", (self.sample_algn_h, self.sample_algn_w),
                                          scope=tik.scope_ubuf, name="w4_ub")
        self.w1_fp16_ub = self.tik_inst.Tensor("float16", (self.sample_algn_h, self.sample_algn_w),
                                               scope=tik.scope_ubuf, name="w1_fp16_ub")
        self.w2_fp16_ub = self.tik_inst.Tensor("float16", (self.sample_algn_h, self.sample_algn_w),
                                               scope=tik.scope_ubuf, name="w2_fp16_ub")
        self.w3_fp16_ub = self.tik_inst.Tensor("float16", (self.sample_algn_h, self.sample_algn_w),
                                               scope=tik.scope_ubuf, name="w3_fp16_ub")
        self.w4_fp16_ub = self.tik_inst.Tensor("float16", (self.sample_algn_h, self.sample_algn_w),
                                               scope=tik.scope_ubuf, name="w4_fp16_ub")
        self.h_low_ub = self.tik_inst.Tensor("int32", (self.algn_pooled_h,), scope=tik.scope_ubuf, name="h_low_ub")
        self.w_low_ub = self.tik_inst.Tensor("int32", (self.algn_pooled_w,), scope=tik.scope_ubuf, name="w_low_ub")
        self.h_high_ub = self.tik_inst.Tensor("int32", (self.algn_pooled_h,), scope=tik.scope_ubuf, name="h_high_ub")
        self.w_high_ub = self.tik_inst.Tensor("int32", (self.algn_pooled_w,), scope=tik.scope_ubuf, name="w_high_ub")
        self.acc_fp16_ub = self.tik_inst.Tensor("float16", (self.input_c,), scope=tik.scope_ubuf, name="acc_fp16_ub")
        self.aligned_ub = self.tik_inst.Tensor("float16", (self.input_c,), scope=tik.scope_ubuf, name="aligned_ub")
        self.output_ub = self.tik_inst.Tensor("float16",
                                              (self.c1_p_sroi_align, self.h_p_sroi_align, self.w_p_sroi_align,
                                               self.c0_p_sroi_align), name="output_ub", scope=tik.scope_ubuf)
        self.input_ub = self.tik_inst.Tensor("float16", (self.h_feature_map, self.w_feature_map, self.input_c),
                                             name="input_ub", scope=tik.scope_ubuf)
        self.u1_ub = self.tik_inst.Tensor("float16", (self.input_plus_four,), scope=tik.scope_ubuf, name="u1_fp16_ub")
        self.u2_ub = self.tik_inst.Tensor("float16", (self.input_plus_four,), scope=tik.scope_ubuf, name="u2_fp16_ub")

    def _psroialign_main_processor(self, block_left, block_offset, iter_num):
        with self.tik_inst.for_range(0, block_left, thread_num=2) as iter_n:
            batch_id = iter_n + block_offset
            self.tik_inst.data_move(self.hstart_ub, self.hstart_cbuf[iter_n, 0, 0], 0, 1, self.group_align // ALIGN8,
                                    1, 1)
            self.tik_inst.data_move(self.wstart_ub, self.wstart_cbuf[iter_n, 0, 0], 0, 1, self.group_align // ALIGN8,
                                    1, 1)
            self.tik_inst.data_move(self.hend_ub, self.hend_cbuf[iter_n, 0, 0], 0, 1, self.group_align // ALIGN8, 1, 1)
            self.tik_inst.data_move(self.wend_ub, self.wend_cbuf[iter_n, 0, 0], 0, 1, self.group_align // ALIGN8, 1, 1)

            self.tmp_sclar.set_as(self.sample_h[iter_n])
            self.tik_inst.vmuls(ALIGN8, self.hind_ub, self.hind_const_ub, self.tmp_sclar, 1,
                                self.sample_algn_h // ALIGN8,
                                1, ALIGN8 // 8, ALIGN8 // 8)
            self.tmp_sclar.set_as(self.sample_w[iter_n])
            self.tik_inst.vmuls(ALIGN8, self.wind_ub, self.wind_const_ub, self.tmp_sclar, 1,
                                self.sample_algn_w // ALIGN8,
                                1, ALIGN8 // 8, ALIGN8 // 8)
            with self.tik_inst.for_range(0, self.pooled_h) as pool_h:
                with self.tik_inst.for_range(0, self.pooled_w) as pool_w:
                    self.start_h.set_as(self.hstart_ub[pool_h, pool_w])
                    self.start_w.set_as(self.wstart_ub[pool_h, pool_w])
                    '''
                    cur_h euqals to hstart + i * self.sample_h, cur_w equals to wstart + j * self.sample_w
                    '''
                    self.tik_inst.vadds(ALIGN8, self.hind_tmp_ub, self.hind_ub, self.start_h,
                                        self.sample_algn_h // ALIGN8,
                                        1, 1, ALIGN8 // 8, ALIGN8 // 8)
                    self.tik_inst.vadds(ALIGN8, self.wind_tmp_ub, self.wind_ub, self.start_w,
                                        self.sample_algn_w // ALIGN8,
                                        1, 1, ALIGN8 // 8, ALIGN8 // 8)
                    '''
                    calc w1, w2, w3, w4, refer bilinear_interpolate interface
                    '''
                    self._prepare_bilinear_interpolate(self.hind_tmp_ub, self.wind_tmp_ub)
                    '''
                    bilinear interpolate
                    '''
                    self.tik_inst.vconv(ALIGN64, "none", self.w1_fp16_ub, self.w1_ub, self.sample_algn_wh // ALIGN64,
                                        1, 1, ALIGN64 // 16, ALIGN64 // 8)
                    self.tik_inst.vconv(ALIGN64, "none", self.w2_fp16_ub, self.w2_ub, self.sample_algn_wh // ALIGN64,
                                        1, 1, ALIGN64 // 16, ALIGN64 // 8)
                    self.tik_inst.vconv(ALIGN64, "none", self.w3_fp16_ub, self.w3_ub, self.sample_algn_wh // ALIGN64,
                                        1, 1, ALIGN64 // 16, ALIGN64 // 8)
                    self.tik_inst.vconv(ALIGN64, "none", self.w4_fp16_ub, self.w4_ub, self.sample_algn_wh // ALIGN64,
                                        1, 1, ALIGN64 // 16, ALIGN64 // 8)
                    self.tik_inst.vector_dup(ALIGN64, self.u2_ub, 0.0, self.input_plus_four // ALIGN64, 1, 4)

                    self._clear_zero_ub(self.acc_fp16_ub)

                    self._tensor_interplate()

                    self._tensor_add()

                    # acc_ub euqals to acc_ub / (self.sample_height * self.sample_width)
                    self._tensor_mul_value(self.acc_fp16_ub)

                    # c_of_feature_map equals to c * group_size * group_size + pool_h * group_size + pool_w
                    with self.tik_inst.for_range(0, self.c_p_sroi_align) as iter_c:
                        c_of_feature_map = iter_c * self.pooled_wh + pool_h * self.pooled_w + pool_w
                        self.aligned_ub[iter_c] = self.acc_fp16_ub[c_of_feature_map]

                    # convert src(ub) to dst(gm)
                    self.tik_inst.data_move(self.output_ub[0, pool_h, pool_w, 0], self.aligned_ub, 0, 1,
                                            self.output_align_c // ALIGN16, 0, self.output_wh_1)

            self.tik_inst.data_move(self.output_map_gm[batch_id, 0, 0, 0, 0], self.output_ub, 0, 1, iter_num, 0, 0)

    def _psroialign_data_move(self):
        self.repeat_num = self.h_feature_map * self.w_feature_map
        with self.tik_inst.for_range(0, self.n_feature_map) as ind_i:
            with self.tik_inst.for_range(0, self.c1_feature_map) as ind_j:
                self.tik_inst.data_move(self.input_map_cbuf[ind_i, ind_j, 0, 0, 0],
                                        self.feature_map_gm[ind_i, ind_j, 0, 0, 0], 0, 1, self.repeat_num, 0, 0)

    # main PSROIAlign process function with single core
    def _psroialign_compute_each_core(self, block_num, block_offset, block_left):
        self.n_roi_align = _ceil_div_mul(block_left, ALIGN16)

        self.sample_h = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="sample_h", scope=tik.scope_ubuf)
        self.sample_w = self.tik_inst.Tensor("float32", (self.n_roi_align,), name="sample_w", scope=tik.scope_ubuf)
        self.hstart_cbuf = self.tik_inst.Tensor("float32", [self.n_roi_align, self.algn_pooled_h, self.algn_pooled_w],
                                                scope=tik.scope_cbuf, name="hstart_cbuf")
        self.wstart_cbuf = self.tik_inst.Tensor("float32", [self.n_roi_align, self.algn_pooled_h, self.algn_pooled_w],
                                                scope=tik.scope_cbuf, name="wstart_cbuf")
        self.hend_cbuf = self.tik_inst.Tensor("float32", [self.n_roi_align, self.algn_pooled_h, self.algn_pooled_w],
                                              scope=tik.scope_cbuf, name="hend_cbuf")
        self.wend_cbuf = self.tik_inst.Tensor("float32", [self.n_roi_align, self.algn_pooled_h, self.algn_pooled_w],
                                              scope=tik.scope_cbuf, name="wend_cbuf")
        self.input_map_cbuf = self.tik_inst.Tensor("float16", (self.n_feature_map, self.c1_feature_map,
                                                               self.h_feature_map, self.w_feature_map,
                                                               self.c0_feature_map), scope=tik.scope_cbuf,
                                                   name="input_map_cbuf")
        self._psroialign_data_move()

        # calc parameter
        self._psroialign_prepare(block_num, block_offset, block_left)

        self._psroialign_prepare_buf()

        self.tik_inst.vmuls(ALIGN8, self.hind_const_ub, self.hind_const_ub, 0.0, self.sample_algn_h // ALIGN8,
                            1, 1, ALIGN8 // 8, ALIGN8 // 8)
        self.tik_inst.vmuls(ALIGN8, self.wind_const_ub, self.wind_const_ub, 0.0, self.sample_algn_w // ALIGN8,
                            1, 1, ALIGN8 // 8, ALIGN8 // 8)

        for num_h in range(0, self.sample_algn_h):
            temp = float(num_h)
            self.tmp_sclar.set_as(temp)
            self.hind_const_ub[num_h] = self.tmp_sclar

        for num_w in range(0, self.sample_algn_w):
            temp = float(num_w)
            self.tmp_sclar.set_as(temp)
            self.wind_const_ub[num_w] = self.tmp_sclar

        self.tik_inst.vadds(ALIGN8, self.hind_const_ub, self.hind_const_ub, 1.0, self.sample_algn_h // ALIGN8,
                            1, 1, ALIGN8 // 8, ALIGN8 // 8)
        self.tik_inst.vadds(ALIGN8, self.wind_const_ub, self.wind_const_ub, 1.0, self.sample_algn_w // ALIGN8,
                            1, 1, ALIGN8 // 8, ALIGN8 // 8)
        iter_num = self.c1_p_sroi_align * self.h_p_sroi_align * self.w_p_sroi_align

        with self.tik_inst.for_range(0, self.h_feature_map) as iter_i:
            with self.tik_inst.for_range(0, self.w_feature_map) as iter_j:
                self.tik_inst.data_move(self.input_ub[iter_i, iter_j, 0], self.input_map_cbuf[0, 0, iter_i, iter_j, 0],
                                        0, self.input_c // ALIGN16, 1, self.input_wh_1, 0)

        self._psroialign_main_processor(block_left, block_offset, iter_num)

    def compute(self):
        last_core_index = self.aicore_num - 1
        with self.tik_inst.for_range(0, self.aicore_num, block_num=self.aicore_num) as iter_i:
            block_offset = self.tik_inst.Scalar("int32", init_value=0)
            block_offset.set_as(iter_i * self.each_core_rois_num)
            with self.tik_inst.if_scope(iter_i != last_core_index):
                self._psroialign_compute_each_core(self.each_core_block_num, block_offset, self.each_core_block_left)
            with self.tik_inst.else_scope():
                self._psroialign_compute_each_core(self.last_core_block_num, block_offset, self.last_core_block_left)

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.feature_map_gm, self.roi_gm],
                               outputs=[self.output_map_gm], enable_l2=True)
        return self.tik_inst


def p_sroi_align(feature_map_dict, roi_dict, output_map_dict, spatial_scale, output_dim, group_size, sample_num,
                 kernel_name="PSROIAlign"):
    feature_map_shape = feature_map_dict.get("shape")
    roi_shape = roi_dict.get("shape")
    output_map_shape = output_map_dict.get("shape")
    obj = PSROIAlign(feature_map_shape, roi_shape, output_map_shape, spatial_scale, output_dim, group_size,
                     sample_num, sample_num, kernel_name)
    obj.compute()
