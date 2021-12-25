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
roi_align
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max uint16
    PARAMS_SIZE = 2 ** 31 - 1
    TILING_ARG_NUM = 256
    # C0 size
    C0_SIZE = 16
    # batch size
    BATCH_SIZE = 128
    # data type of int64
    INT64 = "int64"
    # one block size takes up 32b
    BLOCK_SIZE = 32
    TYPE_LEN_DICT = {"float16": 2, "float32": 4, "int64": 8}
    TILING_MODE_1 = 1
    TILING_MODE_2 = 2

    # 16K size
    UB_30K_SIZE = 150 * 1024
    ZERO = 0.0


def ceil_value(value, dtype):
    """
    if not divide exactly then plus 1
    """
    value *= Constant.TYPE_LEN_DICT.get(dtype)

    return (value + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE


def align_value(value, factor):
    """
    Alignment based on factor.
    """
    return (value + factor - 1) // factor * factor


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class RoiAlign():
    """
    define roi_align object
    """

    def __init__(self):
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.core_num = profile.get_aicore_num()
        self.ub_size = profile.get_unified_buffer_size()
        self.l1_size = profile.get_l1_buffer_size()
        self.feature_map_to_ub_verify = 0
        self.feature_map_to_l1_verify = 0
        self.w_number_ub = 0
        self.w_number_l1 = 0
        self.tiling_mode = 0
        self.real_core_num = 0
        self.tiling_dtype = Constant.INT64
        self.rois_n = 0
        self.rois_row_length = 0
        self.c1_num = 0
        self.feature_map = None
        self.output = None
        self.rois = None
        self.rois_n_gm = None
        self.tiling_gm = None
        self.pooled_width = 0
        self.pooled_height = 0
        self.x_height = 0
        self.x_width = 0
        self.spatial_scale = 0.0
        self.sample_num = 0
        self.kernel_name = None
        self.dtype = "float32"
        self.exist_rois_n = False
        self.available_c1_num = None
        self.roi_end_mode = 0

    def _calc_vector_params(self):
        if self.dtype == "float32":
            dtype_num = 1
            mask = 64
            repeat_times = 2
        else:
            dtype_num = 2
            mask = 128
            repeat_times = 1

        return dtype_num, mask, repeat_times

    def _get_tiling_args(self, tiling_ub):
        self.tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
        self.tiling_mode.set_as(tiling_ub[0])
        self.real_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="real_core_num")
        self.real_core_num.set_as(tiling_ub[1])
        self.rois_n = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="rois_n")
        self.rois_n.set_as(tiling_ub[2])
        self.rois_row_length = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="rois_row_length")
        self.rois_row_length.set_as(tiling_ub[3])
        self.c1_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="c1_num")
        self.c1_num.set_as(tiling_ub[4])
        self.x_height = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="x_height")
        self.x_height.set_as(tiling_ub[5])
        self.x_width = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="x_width")
        self.x_width.set_as(tiling_ub[6])

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _load_a_w_to_cache(self, cache_table, cache, feature_map,
                           index, y_low, n_bust, point):
        """
        load a width of feature map to l1
        """
        tik_instance = self.tik_instance
        stride = (self.x_height * self.x_width - self.x_width) * n_bust
        c_iter_c1 = self.c1_num
        fm_h = self.x_height
        fm_w = self.x_width

        with tik_instance.if_scope(stride > 65535):
            with tik_instance.for_range(0, c_iter_c1) as c_iter_i:
                feature_map_offset = self._get_feature_map_offset(index, c_iter_i, y_low, 0)
                cache_offset = self._get_cache_offset(point, c_iter_i, 0)
                tik_instance.data_move(
                    cache[cache_offset],
                    feature_map[feature_map_offset],
                    0, 1, fm_w * n_bust, 1, 1)
        with tik_instance.else_scope():
            feature_map_offset = self._get_feature_map_offset(index, 0, y_low, 0)
            cache_offset = self._get_cache_offset(point, 0, 0)
            tik_instance.data_move(cache[cache_offset],
                                   feature_map[feature_map_offset],
                                   0, c_iter_c1, fm_w * n_bust,
                                   (fm_h * fm_w - fm_w) * n_bust,
                                   0)
        # ylow:
        cache_table[point, 0].set_as(index)
        cache_table[point, 1].set_as(y_low)

    # 'pylint: disable=too-many-arguments
    def _load_from_l1_cache(self, fm_grid, cache_l1, point,
                            current_cb, c_block, x_low, x_high, c_valid, n_bust):
        """
        load feature map from l1 cache
        """
        tik_instance = self.tik_instance
        cache_offset = self._get_cache_offset(point, current_cb * c_block, x_low)
        tik_instance.data_move(fm_grid[0, point, 0, 0],
                               cache_l1[cache_offset], 0,
                               c_valid, n_bust, (self.x_width - 1) * n_bust,
                               (4 - 1) * n_bust)
        cache_offset = self._get_cache_offset(point, current_cb * c_block, x_high)
        tik_instance.data_move(fm_grid[0, point, 1, 0],
                               cache_l1[cache_offset], 0,
                               c_valid, n_bust, (self.x_width - 1) * n_bust,
                               (4 - 1) * n_bust)

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _load_feature_map_to_ub(self, fm_grid,
                                c_block, c_valid,
                                feature_map, index, current_cb,
                                y_low, x_low, x_high,
                                y_high, n_bust, cache_flag):
        """
        load feature map from ddr to ub
        """
        tik_instance = self.tik_instance
        stride = (self.x_height * self.x_width - 1) * n_bust
        stride_s = tik_instance.Scalar(dtype="int32", init_value=stride)

        with tik_instance.if_scope(cache_flag == 1):
            index.set_as(0)

        with tik_instance.if_scope(stride <= 65535):
            feature_map_offset = self._get_feature_map_offset(index, current_cb * c_block, y_low, x_low)
            tik_instance.data_move(
                fm_grid[0, 0, 0, 0],
                feature_map[feature_map_offset], 0,
                c_valid, n_bust, stride_s, (4 - 1) * n_bust)

            feature_map_offset = self._get_feature_map_offset(index, current_cb * c_block, y_low, x_high)
            tik_instance.data_move(
                fm_grid[0, 0, 1, 0],
                feature_map[feature_map_offset], 0,
                c_valid, n_bust, stride_s, (4 - 1) * n_bust)

            feature_map_offset = self._get_feature_map_offset(index, current_cb * c_block, y_high, x_high)
            tik_instance.data_move(
                fm_grid[0, 1, 1, 0],
                feature_map[feature_map_offset], 0,
                c_valid, n_bust, stride_s, (4 - 1) * n_bust)

            feature_map_offset = self._get_feature_map_offset(index, current_cb * c_block, y_high, x_low)
            tik_instance.data_move(
                fm_grid[0, 1, 0, 0],
                feature_map[feature_map_offset], 0,
                c_valid, n_bust, stride_s, (4 - 1) * n_bust)

        with tik_instance.else_scope():
            with tik_instance.for_range(0, c_valid) as c_iter_i:
                feature_map_offset = self._get_feature_map_offset(index, current_cb * c_block + c_iter_i,
                                                                  y_low, x_low)
                tik_instance.data_move(fm_grid[c_iter_i, 0, 0, 0], feature_map[feature_map_offset], 0,
                                       1, n_bust, 1, 1)

                feature_map_offset = self._get_feature_map_offset(index, current_cb * c_block + c_iter_i,
                                                                  y_low, x_high)
                tik_instance.data_move(fm_grid[c_iter_i, 0, 1, 0], feature_map[feature_map_offset], 0,
                                       1, n_bust, 1, 1)

                feature_map_offset = self._get_feature_map_offset(index, current_cb * c_block + c_iter_i,
                                                                  y_high, x_high)
                tik_instance.data_move(fm_grid[c_iter_i, 1, 1, 0], feature_map[feature_map_offset], 0,
                                       1, n_bust, 1, 1)

                feature_map_offset = self._get_feature_map_offset(index, current_cb * c_block + c_iter_i,
                                                                  y_high, x_low)
                tik_instance.data_move(fm_grid[c_iter_i, 1, 0, 0], feature_map[feature_map_offset], 0,
                                       1, n_bust, 1, 1)

    # 'pylint: disable=too-many-arguments
    def _load_ret_to_gm(self, n_bust, core_bias, roi_128_number, curr_roi, grid_num_h, val):
        tik_instance = self.tik_instance
        with tik_instance.if_scope(
                (self.pooled_width * self.pooled_height - self.pooled_width) * n_bust <= 65535):
            output_offset = self._get_output_offset(core_bias + 128 * roi_128_number + curr_roi,
                                                    0, grid_num_h // self.sample_num, 0)
            tik_instance.data_move(
                self.output[output_offset], val[0], 0,
                self.c1_num,
                self.pooled_width * n_bust, 0,
                (self.pooled_width * self.pooled_height - self.pooled_width) * n_bust)

        with tik_instance.else_scope():
            with tik_instance.for_range(0, self.c1_num) as c_iter_i:
                output_offset = self._get_output_offset(core_bias + \
                                                        128 * roi_128_number + curr_roi,
                                                        c_iter_i, grid_num_h // self.sample_num, 0)
                tik_instance.data_move(
                    self.output[output_offset],
                    val[c_iter_i * self.pooled_width * Constant.C0_SIZE], 0, 1, self.pooled_width * n_bust, 0, 0)

    # 'pylint: disable=invalid-name
    def _get_feature_map_offset(self, n, c1, h, w):
        """calc x_diff offset
        """
        n_offset = n * self.c1_num * self.x_height * self.x_width * Constant.C0_SIZE
        c1_offset = c1 * self.x_height * self.x_width * Constant.C0_SIZE
        h_offset = h * self.x_width * Constant.C0_SIZE
        w_offset = w * Constant.C0_SIZE

        return n_offset + c1_offset + h_offset + w_offset

    # 'pylint: disable=invalid-name
    def _get_cache_offset(self, n, c1, w):
        """calc x_diff offset
        """
        n_offset = n * self.c1_num * self.x_width * Constant.C0_SIZE
        c1_offset = c1 * self.x_width * Constant.C0_SIZE
        w_offset = w * Constant.C0_SIZE

        return n_offset + c1_offset + w_offset

    # 'pylint: disable=invalid-name
    def _get_output_offset(self, n, c1, h, w):
        """calc output offset
        """
        n_offset = n * self.c1_num * self.pooled_height * self.pooled_width * Constant.C0_SIZE
        c1_offset = c1 * self.pooled_height * self.pooled_width * Constant.C0_SIZE
        h_offset = h * self.pooled_width * Constant.C0_SIZE
        w_offset = w * Constant.C0_SIZE

        return n_offset + c1_offset + h_offset + w_offset

    def _calc_buffer_verify(self):
        tik_instance = self.tik_instance
        if self.dtype == "float32":
            n_bust = 2
        else:
            n_bust = 1
        ub_size_available = self.ub_size - Constant.UB_30K_SIZE
        self.feature_map_to_ub_verify = ub_size_available // \
                                        (self.c1_num * self.x_height * self.x_width * Constant.C0_SIZE * n_bust * 2)
        self.feature_map_to_l1_verify = self.l1_size // \
                                        (self.c1_num * self.x_height * self.x_width * Constant.C0_SIZE * n_bust * 2)

        with tik_instance.if_scope(self.feature_map_to_ub_verify == 0 and self.feature_map_to_l1_verify == 0):
            self.w_number_ub = ub_size_available // \
                               (self.c1_num * self.x_width *
                                Constant.C0_SIZE * n_bust * 2)
        with tik_instance.if_scope(self.feature_map_to_ub_verify == 0 and \
                                   self.feature_map_to_l1_verify == 0 and
                                   self.w_number_ub == 0):
            with tik_instance.if_scope((self.x_width - 1) * n_bust < 65535):
                self.w_number_l1 = self.l1_size // (self.c1_num * self.x_width * Constant.C0_SIZE * n_bust * 2)

    # 'pylint: disable=too-many-arguments
    def _get_input(self, grid_h, grid_w, proposals_ub_y0,
                   proposals_ub_x0, curr_roi):
        """
        define scalar
        """
        tik_instance = self.tik_instance
        grid_h_roi = tik_instance.Scalar(dtype=self.dtype, name="grid_h_roi")
        grid_h_roi.set_as(grid_h[curr_roi])

        grid_w_roi = tik_instance.Scalar(dtype=self.dtype, name="grid_w_roi")
        grid_w_roi.set_as(grid_w[curr_roi])

        rois_start_h = tik_instance.Scalar(dtype=self.dtype, name="rois_start_h")
        rois_start_h.set_as(proposals_ub_y0[curr_roi])
        rois_start_w = tik_instance.Scalar(dtype=self.dtype, name="rois_start_w")
        rois_start_w.set_as(proposals_ub_x0[curr_roi])

        return grid_w_roi, grid_h_roi, rois_start_w, rois_start_h

    def _tf_n52n8(self, rois_ub, rois_n5, block_num):
        """
        transform ROIS form N5 to N8
        """
        tik_instance = self.tik_instance
        with tik_instance.for_range(0, block_num) as rois_num:
            rois_ub[rois_num, 0].set_as(rois_n5[rois_num, 0])
            rois_ub[rois_num, 1].set_as(rois_n5[rois_num, 1])
            rois_ub[rois_num, 2].set_as(rois_n5[rois_num, 2])
            rois_ub[rois_num, 3].set_as(rois_n5[rois_num, 3])
            rois_ub[rois_num, 4].set_as(rois_n5[rois_num, 4])

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _compute_w1234(self, y_hi_w, y_lo_w, x_hi_w, x_lo_w, grid_num_h, grid_num_w,
                       fm_grid, c_valid, n_bust, val):
        """
        get weight 1, 2, 3 and 4
        """
        tik_instance = self.tik_instance
        dtype = self.dtype
        # assume fm_c1 <= 64
        w1_lt = tik_instance.Tensor(
            self.dtype, [64, 16], name="w1_lt", scope=tbe_platform.scope_ubuf)
        w2_rt = tik_instance.Tensor(
            self.dtype, [64, 16], name="w2_rt", scope=tbe_platform.scope_ubuf)
        w3_lb = tik_instance.Tensor(
            self.dtype, [64, 16], name="w3_lb", scope=tbe_platform.scope_ubuf)
        w4_rb = tik_instance.Tensor(
            self.dtype, [64, 16], name="w4_rb", scope=tbe_platform.scope_ubuf)

        h_y = tik_instance.Scalar(self.dtype, init_value=y_hi_w[grid_num_h])
        l_y = tik_instance.Scalar(self.dtype, init_value=y_lo_w[grid_num_h])
        h_x = tik_instance.Scalar(self.dtype, init_value=x_hi_w[grid_num_w])
        l_x = tik_instance.Scalar(self.dtype, init_value=x_lo_w[grid_num_w])

        hy_tensor = tik_instance.Scalar(dtype=dtype, init_value=h_y, name="hy_tensor")
        ly_tensor = tik_instance.Scalar(dtype=dtype, init_value=l_y, name="ly_tensor")
        hx_tensor = tik_instance.Scalar(dtype=dtype, init_value=h_x, name="hx_tensor")
        lx_tensor = tik_instance.Scalar(dtype=dtype, init_value=l_x, name="lx_tensor")

        w_1 = tik_instance.Scalar(dtype=dtype, init_value=hy_tensor * hx_tensor)
        w_2 = tik_instance.Scalar(dtype=dtype, init_value=hy_tensor * lx_tensor)
        w_3 = tik_instance.Scalar(dtype=dtype, init_value=hx_tensor * ly_tensor)
        w_4 = tik_instance.Scalar(dtype=dtype, init_value=ly_tensor * lx_tensor)

        tik_instance.vec_muls(16, w1_lt[0, 0], fm_grid[0, 0, 0, 0], w_1, \
                              c_valid, n_bust, 4 * n_bust)
        tik_instance.vec_muls(16, w2_rt[0, 0], fm_grid[0, 0, 1, 0], w_2, \
                              c_valid, n_bust, 4 * n_bust)
        tik_instance.vec_muls(16, w3_lb[0, 0], fm_grid[0, 1, 0, 0], w_3, \
                              c_valid, n_bust, 4 * n_bust)
        tik_instance.vec_muls(16, w4_rb[0, 0], fm_grid[0, 1, 1, 0], w_4, \
                              c_valid, n_bust, 4 * n_bust)

        tik_instance.vec_add(16, w1_lt[0, 0], w1_lt[0, 0], w2_rt[0, 0], \
                             c_valid, n_bust, n_bust, n_bust)
        tik_instance.vec_add(16, w1_lt[0, 0], w1_lt[0, 0], w3_lb[0, 0], \
                             c_valid, n_bust, n_bust, n_bust)
        tik_instance.vec_add(16, w1_lt[0, 0], w1_lt[0, 0], w4_rb[0, 0], \
                             c_valid, n_bust, n_bust, n_bust)

        with tik_instance.for_range(0, self.c1_num) as c_iter_i:
            val_offset = c_iter_i * self.pooled_width * Constant.C0_SIZE + \
                         grid_num_w // self.sample_num * Constant.C0_SIZE
            tik_instance.vec_add(
                16,
                val[val_offset],
                val[val_offset],
                w1_lt[c_iter_i, 0], 1, n_bust, n_bust, n_bust)

    def _get_average(self, val, c_valid, n_bust):
        """
        get average
        """
        tik_instance = self.tik_instance

        wh_tmp = tik_instance.Scalar(dtype=self.dtype, name="wh_tmp")
        wh_tmp.set_as(self.sample_num * self.sample_num)

        tik_instance.vec_muls(16, val, val, 1.0 / wh_tmp, c_valid * self.pooled_width, \
                              n_bust, n_bust)

    # 'pylint: disable=too-many-locals,too-many-arguments,too-many-statements
    def _get_grid_weight(self, grid_w_roi, grid_h_roi, rois_start_w, rois_start_h):
        """
        get grid size and coordinate in feature
        """
        tik_instance = self.tik_instance
        x_lo_w = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="x_lo_w", scope=tbe_platform.scope_ubuf)
        x_hi_w = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="x_hi_w", scope=tbe_platform.scope_ubuf)
        y_lo_w = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="y_lo_w", scope=tbe_platform.scope_ubuf)
        y_hi_w = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="_lo_w", scope=tbe_platform.scope_ubuf)
        x_lo = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="x_lo", scope=tbe_platform.scope_ubuf)
        x_hi = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="x_hi", scope=tbe_platform.scope_ubuf)
        y_lo = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="y_lo", scope=tbe_platform.scope_ubuf)
        y_hi = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="y_hi", scope=tbe_platform.scope_ubuf)

        raw_x = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="raw_x", scope=tbe_platform.scope_ubuf)
        raw_y = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="raw_y", scope=tbe_platform.scope_ubuf)
        x_output = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="x_output", scope=tbe_platform.scope_ubuf)
        y_output = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="y_output", scope=tbe_platform.scope_ubuf)

        const_value_0_127 = tik_instance.Tensor(
            self.dtype, (Constant.BATCH_SIZE,), name="const_value_0_127", scope=tbe_platform.scope_ubuf)
        if self.dtype == "float32":
            dtype_num = 1
        else:
            dtype_num = 2

        with tik_instance.for_range(0, 128) as i:
            const_value_0_127[i] = i

        grid_w_vector = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="grid_w_vector", scope=tbe_platform.scope_ubuf)
        grid_h_vector = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="grid_h_vector", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_muls(64 * dtype_num, grid_w_vector, const_value_0_127,
                              grid_w_roi, 2 // dtype_num, 8, 8)
        tik_instance.vec_muls(64 * dtype_num, grid_h_vector, const_value_0_127,
                              grid_h_roi, 2 // dtype_num, 8, 8)

        half_grid = 0.5 * grid_w_roi + rois_start_w
        tik_instance.vec_adds(64 * dtype_num, raw_x, grid_w_vector,
                              half_grid, 2 // dtype_num, 8, 8)
        half_grid = 0.5 * grid_h_roi + rois_start_h
        tik_instance.vec_adds(64 * dtype_num, raw_y, grid_h_vector,
                              half_grid, 2 // dtype_num, 8, 8)

        const_zero = tik_instance.Tensor(
            self.dtype, [64 * dtype_num], name="const_zero", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_dup(64 * dtype_num, const_zero, 0, 1, 0)

        tik_instance.vec_max(64 * dtype_num, x_output, raw_x, const_zero,
                             2 // dtype_num, 8, 8, 0)
        tik_instance.vec_max(64 * dtype_num, y_output, raw_y, const_zero,
                             2 // dtype_num, 8, 8, 0)

        tik_instance.vec_conv(64, "floor", x_lo, x_output, 2,
                              8, 8 // dtype_num)
        tik_instance.vec_conv(64, "floor", y_lo, y_output, 2,
                              8, 8 // dtype_num)

        const_one = tik_instance.Tensor(
            "int32", [64], name="const_one", scope=tbe_platform.scope_ubuf)
        tik_instance.vec_dup(64, const_one, 1, 1, 0)
        tik_instance.vec_add(64, x_hi, x_lo, const_one, 2, 8, 8, 0)
        tik_instance.vec_add(64, y_hi, y_lo, const_one, 2, 8, 8, 0)

        const_value_fp32 = tik_instance.Tensor(
            self.dtype, [64 * dtype_num], name="const_value_fp32", scope=tbe_platform.scope_ubuf)
        const_value_int32 = tik_instance.Tensor(
            "int32", [64], name="const_value_int32", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_dup(64 * dtype_num, const_value_fp32, self.x_width - 1, 1, 0)
        tik_instance.vec_dup(64, const_value_int32, self.x_width - 1, 1, 0)
        tik_instance.vec_min(64, x_lo, x_lo, const_value_int32, 2, 8, 8, 0)
        tik_instance.vec_min(64, x_hi, x_hi, const_value_int32, 2, 8, 8, 0)
        tik_instance.vec_min(64 * dtype_num, x_output, x_output, const_value_fp32,
                             2 // dtype_num, 8, 8, 0)

        tik_instance.vec_dup(64, const_value_int32, self.x_height - 1, 1, 0)
        tik_instance.vec_dup(64 * dtype_num, const_value_fp32, self.x_height - 1, 1, 0)
        tik_instance.vec_min(64, y_lo, y_lo, const_value_int32, 2, 8, 8, 0)
        tik_instance.vec_min(64, y_hi, y_hi, const_value_int32, 2, 8, 8, 0)
        tik_instance.vec_min(64 * dtype_num, y_output, y_output, const_value_fp32,
                             2 // dtype_num, 8, 8, 0)

        tmp_fp32 = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="tmp_fp32", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_conv(64, "", tmp_fp32, x_lo, 2, 8, 8)

        tik_instance.vec_sub(64 * dtype_num, x_lo_w, x_output, tmp_fp32,
                             2 // dtype_num, 8, 8, 8)

        tik_instance.vec_conv(64, "", tmp_fp32, y_lo, 2, 8, 8)
        tik_instance.vec_sub(64 * dtype_num, y_lo_w, y_output, tmp_fp32,
                             2 // dtype_num, 8, 8, 8)

        tik_instance.vec_dup(64 * dtype_num, const_value_fp32, 1.0, 1, 0)
        tik_instance.vec_sub(64 * dtype_num, x_hi_w, const_value_fp32, x_lo_w,
                             2 // dtype_num, 8, 0, 8)
        tik_instance.vec_sub(64 * dtype_num, y_hi_w, const_value_fp32, y_lo_w,
                             2 // dtype_num, 8, 0, 8)

        return x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi, raw_x, raw_y

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _bilinear_interpolate_all_in_ub(self, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi,
                                        raw_x, raw_y, n_bust, core_bias,
                                        index, curr_roi, roi_128_number, cache_index):
        """
        _bilinear_interpolate
        """
        tik_instance = self.tik_instance
        ub_size_available = self.ub_size - Constant.UB_30K_SIZE
        available_ub_num = ub_size_available // 2 // n_bust
        feature_map_offset = self._get_feature_map_offset(index, 0, 0, 0)

        feature_map_ub = tik_instance.Tensor(
            self.dtype, [available_ub_num, ],
            name="feature_map_ub",
            scope=tbe_platform.scope_ubuf)
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move(feature_map_ub, self.feature_map[feature_map_offset],
                                   0, 1, self.c1_num * self.x_height * self.x_width * n_bust, 0, 0)
            cache_index.set_as(index)

        # actual size [fm_c1, pw_int, 16], limit of fm_c1 * pw_int is 255
        val = tik_instance.Tensor(
            self.dtype, [255 * 16], name="val", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

        roi_y_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_y_floor", scope=tbe_platform.scope_ubuf)
        roi_x_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_x_floor", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_y[0], 2, 8, 4)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_x[0], 2, 8, 4)

        with tik_instance.for_range(0, self.pooled_height * self.sample_num) as grid_num_h:
            verify = tik_instance.Scalar(dtype="int32", init_value=0)
            y_tmp = tik_instance.Scalar(dtype="int32")
            y_tmp.set_as(roi_y_floor[grid_num_h])
            with tik_instance.if_scope(y_tmp < -1):
                verify.set_as(1)
            with tik_instance.if_scope(y_tmp >= self.x_height):
                verify.set_as(1)

            with tik_instance.if_scope(verify == 0):
                y_low = tik_instance.Scalar(
                    dtype="int32", init_value=y_lo[grid_num_h])
                y_high = tik_instance.Scalar(
                    dtype="int32", init_value=y_hi[grid_num_h])

                with tik_instance.for_range(0, self.pooled_width * self.sample_num) as grid_num_w:
                    x_tmp = tik_instance.Scalar(dtype="int32")
                    x_tmp.set_as(roi_x_floor[grid_num_w])
                    with tik_instance.if_scope(x_tmp < -1):
                        verify.set_as(1)
                    with tik_instance.if_scope(x_tmp >= self.x_width):
                        verify.set_as(1)

                    with tik_instance.if_scope(verify == 0):
                        x_low = tik_instance.Scalar(
                            dtype="int32", init_value=x_lo[grid_num_w])
                        x_high = tik_instance.Scalar(
                            dtype="int32", init_value=x_hi[grid_num_w])
                        fm_grid = tik_instance.Tensor(
                            self.dtype, (64, 2, 2, 16),
                            name="fm_grid",
                            scope=tbe_platform.scope_ubuf)

                        self._load_feature_map_to_ub(fm_grid,
                                                     self.c1_num, self.c1_num, feature_map_ub, index, 0,
                                                     y_low, x_low, x_high, y_high, n_bust, 1)

                        self._compute_w1234(y_hi_w, y_lo_w, x_hi_w, x_lo_w, grid_num_h, grid_num_w,
                                            fm_grid, self.c1_num, n_bust, val)

            with tik_instance.if_scope((grid_num_h + 1) % self.sample_num == 0):
                self._get_average(val, self.c1_num, n_bust)

                self._load_ret_to_gm(n_bust, core_bias, roi_128_number, curr_roi, grid_num_h, val)

                tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _bilinear_interpolate_all_in_l1(self, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi,
                                        raw_x, raw_y, n_bust, core_bias,
                                        index, curr_roi, roi_128_number, cache_index):
        """
        _bilinear_interpolate
        """
        tik_instance = self.tik_instance
        available_l1_num = self.l1_size // 2 // n_bust
        feature_map_offset = self._get_feature_map_offset(index, 0, 0, 0)

        cache_fm = tik_instance.Tensor(
            self.dtype, [available_l1_num, ],
            name="cache_fm",
            scope=tbe_platform.scope_cbuf)
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move(cache_fm, self.feature_map[feature_map_offset],
                                   0, 1, self.c1_num * self.x_height * self.x_width * n_bust, 0, 0)
            cache_index.set_as(index)

        # actual size [fm_c1, pw_int, 16], limit of fm_c1 * pw_int is 255
        val = tik_instance.Tensor(
            self.dtype, [255 * 16], name="val", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

        roi_y_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_y_floor", scope=tbe_platform.scope_ubuf)
        roi_x_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_x_floor", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_y[0], 2, 8, 4)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_x[0], 2, 8, 4)

        with tik_instance.for_range(0, self.pooled_height * self.sample_num) as grid_num_h:
            verify = tik_instance.Scalar(dtype="int32", init_value=0)
            y_tmp = tik_instance.Scalar(dtype="int32")
            y_tmp.set_as(roi_y_floor[grid_num_h])
            with tik_instance.if_scope(y_tmp < -1):
                verify.set_as(1)
            with tik_instance.if_scope(y_tmp >= self.x_height):
                verify.set_as(1)

            with tik_instance.if_scope(verify == 0):
                y_low = tik_instance.Scalar(
                    dtype="int32", init_value=y_lo[grid_num_h])
                y_high = tik_instance.Scalar(
                    dtype="int32", init_value=y_hi[grid_num_h])

                with tik_instance.for_range(0, self.pooled_width * self.sample_num) as grid_num_w:
                    x_tmp = tik_instance.Scalar(dtype="int32")
                    x_tmp.set_as(roi_x_floor[grid_num_w])
                    with tik_instance.if_scope(x_tmp < -1):
                        verify.set_as(1)
                    with tik_instance.if_scope(x_tmp >= self.x_width):
                        verify.set_as(1)

                    with tik_instance.if_scope(verify == 0):
                        x_low = tik_instance.Scalar(
                            dtype="int32", init_value=x_lo[grid_num_w])
                        x_high = tik_instance.Scalar(
                            dtype="int32", init_value=x_hi[grid_num_w])
                        fm_grid = tik_instance.Tensor(
                            self.dtype, (64, 2, 2, 16),
                            name="fm_grid",
                            scope=tbe_platform.scope_ubuf)

                        self._load_feature_map_to_ub(fm_grid,
                                                     self.c1_num, self.c1_num, cache_fm, index, 0,
                                                     y_low, x_low, x_high, y_high, n_bust, 1)

                        self._compute_w1234(y_hi_w, y_lo_w, x_hi_w, x_lo_w, grid_num_h, grid_num_w,
                                            fm_grid, self.c1_num, n_bust, val)

            with tik_instance.if_scope((grid_num_h + 1) % self.sample_num == 0):
                self._get_average(val, self.c1_num, n_bust)

                self._load_ret_to_gm(n_bust, core_bias, roi_128_number, curr_roi, grid_num_h, val)

                tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

    # 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
    def _bilinear_interpolate_w_in_ub(self, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi,
                                      raw_x, raw_y, n_bust, core_bias,
                                      index, curr_roi, roi_128_number, cache_index):
        """
        _bilinear_interpolate
        """
        tik_instance = self.tik_instance
        ub_size_available = self.ub_size - Constant.UB_30K_SIZE
        available_ub_num = ub_size_available // 2 // n_bust

        cache_ub = tik_instance.Tensor( \
            self.dtype, [available_ub_num], \
            name="cache_ub", \
            scope=tbe_platform.scope_ubuf)
        cache_table = tik_instance.Tensor( \
            "int32", [2, 2], name="cache_table", scope=tbe_platform.scope_ubuf)
        with tik_instance.for_range(0, 2) as j:
            one = tik_instance.Scalar(dtype="int32", init_value=-1)
            cache_table[j, 0].set_as(one)

        # actual size [fm_c1, pw_int, 16], limit of fm_c1 * pw_int is 255
        val = tik_instance.Tensor(
            self.dtype, [255 * 16], name="val", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

        roi_y_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_y_floor", scope=tbe_platform.scope_ubuf)
        roi_x_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_x_floor", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_y[0], 2, 8, 4)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_x[0], 2, 8, 4)

        with tik_instance.for_range(0, self.pooled_height * self.sample_num) as grid_num_h:
            verify = tik_instance.Scalar(dtype="int32", init_value=0)
            y_tmp = tik_instance.Scalar(dtype="int32")
            y_tmp.set_as(roi_y_floor[grid_num_h])
            with tik_instance.if_scope(y_tmp < -1):
                verify.set_as(1)
            with tik_instance.if_scope(y_tmp >= self.x_height):
                verify.set_as(1)

            with tik_instance.if_scope(verify == 0):
                y_low = tik_instance.Scalar(
                    dtype="int32", init_value=y_lo[grid_num_h])
                y_high = tik_instance.Scalar(
                    dtype="int32", init_value=y_hi[grid_num_h])
                self._load_a_w_to_cache(cache_table, cache_ub, self.feature_map,
                                        index, y_low, n_bust, 0)
                self._load_a_w_to_cache(cache_table, cache_ub, self.feature_map,
                                        index, y_high, n_bust, 1)

                with tik_instance.for_range(0, self.pooled_width * self.sample_num) as grid_num_w:
                    x_tmp = tik_instance.Scalar(dtype="int32")
                    x_tmp.set_as(roi_x_floor[grid_num_w])
                    with tik_instance.if_scope(x_tmp < -1):
                        verify.set_as(1)
                    with tik_instance.if_scope(x_tmp >= self.x_width):
                        verify.set_as(1)

                    with tik_instance.if_scope(verify == 0):
                        x_low = tik_instance.Scalar(
                            dtype="int32", init_value=x_lo[grid_num_w])
                        x_high = tik_instance.Scalar(
                            dtype="int32", init_value=x_hi[grid_num_w])
                        fm_grid = tik_instance.Tensor(
                            self.dtype, (64, 2, 2, 16),
                            name="fm_grid",
                            scope=tbe_platform.scope_ubuf)

                        self._load_from_l1_cache(fm_grid, cache_ub,
                                                 0, 0, self.c1_num, x_low, x_high, self.c1_num, n_bust)
                        self._load_from_l1_cache(fm_grid, cache_ub,
                                                 1, 0, self.c1_num, x_low, x_high, self.c1_num, n_bust)

                        self._compute_w1234(y_hi_w, y_lo_w, x_hi_w, x_lo_w, grid_num_h, grid_num_w,
                                            fm_grid, self.c1_num, n_bust, val)

            with tik_instance.if_scope((grid_num_h + 1) % self.sample_num == 0):
                self._get_average(val, self.c1_num, n_bust)

                self._load_ret_to_gm(n_bust, core_bias, roi_128_number, curr_roi, grid_num_h, val)

                tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

    # 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
    def _bilinear_interpolate_w_in_l1(self, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi,
                                      raw_x, raw_y, n_bust, core_bias,
                                      index, curr_roi, roi_128_number, cache_index):
        """
        _bilinear_interpolate
        """
        tik_instance = self.tik_instance
        available_l1_num = self.l1_size // 2 // n_bust
        cache_l1 = tik_instance.Tensor(
            self.dtype, [available_l1_num],
            name="cache_l1",
            scope=tbe_platform.scope_cbuf)
        cache_table = tik_instance.Tensor(
            "int32", [10, 2], name="cache_table", scope=tbe_platform.scope_ubuf)

        # actual size [fm_c1, pw_int, 16], limit of fm_c1 * pw_int is 255
        val = tik_instance.Tensor(
            self.dtype, [255 * 16], name="val", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

        roi_y_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_y_floor", scope=tbe_platform.scope_ubuf)
        roi_x_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_x_floor", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_y[0], 2, 8, 4)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_x[0], 2, 8, 4)

        with tik_instance.for_range(0, self.pooled_height * self.sample_num) as grid_num_h:
            verify = tik_instance.Scalar(dtype="int32", init_value=0)
            y_tmp = tik_instance.Scalar(dtype="int32")
            y_tmp.set_as(roi_y_floor[grid_num_h])
            with tik_instance.if_scope(y_tmp < -1):
                verify.set_as(1)
            with tik_instance.if_scope(y_tmp >= self.x_height):
                verify.set_as(1)

            with tik_instance.if_scope(verify == 0):
                y_low = tik_instance.Scalar(
                    dtype="int32", init_value=y_lo[grid_num_h])
                y_high = tik_instance.Scalar(
                    dtype="int32", init_value=y_hi[grid_num_h])

                self._load_a_w_to_cache(cache_table, cache_l1, self.feature_map,
                                        index, y_low, n_bust, 0)
                self._load_a_w_to_cache(cache_table, cache_l1, self.feature_map,
                                        index, y_high, n_bust, 1)

                with tik_instance.for_range(0, self.pooled_width * self.sample_num) as grid_num_w:
                    x_tmp = tik_instance.Scalar(dtype="int32")
                    x_tmp.set_as(roi_x_floor[grid_num_w])
                    with tik_instance.if_scope(x_tmp < -1):
                        verify.set_as(1)
                    with tik_instance.if_scope(x_tmp >= self.x_width):
                        verify.set_as(1)

                    with tik_instance.if_scope(verify == 0):
                        x_low = tik_instance.Scalar(
                            dtype="int32", init_value=x_lo[grid_num_w])
                        x_high = tik_instance.Scalar(
                            dtype="int32", init_value=x_hi[grid_num_w])
                        fm_grid = tik_instance.Tensor(
                            self.dtype, (64, 2, 2, 16),
                            name="fm_grid",
                            scope=tbe_platform.scope_ubuf)

                        self._load_from_l1_cache(fm_grid, cache_l1,
                                                 0, 0, self.c1_num, x_low, x_high, self.c1_num, n_bust)
                        self._load_from_l1_cache(fm_grid, cache_l1,
                                                 1, 0, self.c1_num, x_low, x_high, self.c1_num, n_bust)

                        self._compute_w1234(y_hi_w, y_lo_w, x_hi_w, x_lo_w, grid_num_h, grid_num_w,
                                            fm_grid, self.c1_num, n_bust, val)

            with tik_instance.if_scope((grid_num_h + 1) % self.sample_num == 0):
                self._get_average(val, self.c1_num, n_bust)

                self._load_ret_to_gm(n_bust, core_bias, roi_128_number, curr_roi, grid_num_h, val)

                tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

    # 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
    def _bilinear_interpolate_without_cache(self, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi,
                                            raw_x, raw_y, n_bust, core_bias,
                                            index, curr_roi, roi_128_number, cache_index):
        """
        _bilinear_interpolate
        """
        tik_instance = self.tik_instance

        # actual size [fm_c1, pw_int, 16], limit of fm_c1 * pw_int is 255
        val = tik_instance.Tensor(
            self.dtype, [255 * 16], name="val", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

        roi_y_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_y_floor", scope=tbe_platform.scope_ubuf)
        roi_x_floor = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_x_floor", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_y[0], 2, 8, 4)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_x[0], 2, 8, 4)

        with tik_instance.for_range(0, self.pooled_height * self.sample_num) as grid_num_h:
            verify = tik_instance.Scalar(dtype="int32", init_value=0)
            y_tmp = tik_instance.Scalar(dtype="int32")
            y_tmp.set_as(roi_y_floor[grid_num_h])
            with tik_instance.if_scope(y_tmp < -1):
                verify.set_as(1)
            with tik_instance.if_scope(y_tmp >= self.x_height):
                verify.set_as(1)

            with tik_instance.if_scope(verify == 0):
                y_low = tik_instance.Scalar(
                    dtype="int32", init_value=y_lo[grid_num_h])
                y_high = tik_instance.Scalar(
                    dtype="int32", init_value=y_hi[grid_num_h])

                with tik_instance.for_range(0, self.pooled_width * self.sample_num) as grid_num_w:
                    x_tmp = tik_instance.Scalar(dtype="int32")
                    x_tmp.set_as(roi_x_floor[grid_num_w])
                    with tik_instance.if_scope(x_tmp < -1):
                        verify.set_as(1)
                    with tik_instance.if_scope(x_tmp >= self.x_width):
                        verify.set_as(1)

                    with tik_instance.if_scope(verify == 0):
                        x_low = tik_instance.Scalar(
                            dtype="int32", init_value=x_lo[grid_num_w])
                        x_high = tik_instance.Scalar(
                            dtype="int32", init_value=x_hi[grid_num_w])
                        fm_grid = tik_instance.Tensor(
                            self.dtype, (64, 2, 2, 16),
                            name="fm_grid",
                            scope=tbe_platform.scope_ubuf)

                        self._load_feature_map_to_ub(fm_grid,
                                                     self.c1_num, self.c1_num, self.feature_map,
                                                     index, 0, y_low, x_low,
                                                     x_high, y_high, n_bust, 0)

                        self._compute_w1234(y_hi_w, y_lo_w, x_hi_w, x_lo_w, grid_num_h, grid_num_w,
                                            fm_grid, self.c1_num, n_bust, val)

            with tik_instance.if_scope((grid_num_h + 1) % self.sample_num == 0):
                self._get_average(val, self.c1_num, n_bust)

                self._load_ret_to_gm(n_bust, core_bias, roi_128_number, curr_roi, grid_num_h, val)

                tik_instance.vec_dup(16, val, 0.0, self.c1_num * self.pooled_width, n_bust)

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _common_compute(self, proposals_ub_x0, proposals_ub_y0,
                        roi_128_number, rois_valid_in_block,
                        n_bust, roi_int32_fm_index,
                        grid_h, grid_w, core_bias):
        """
        get ret without L1
        """
        tik_instance = self.tik_instance
        cache_index = tik_instance.Scalar(dtype="int32", init_value=-1)
        with tik_instance.for_range(0, rois_valid_in_block) as curr_roi:
            index = tik_instance.Scalar(dtype="int32")
            index.set_as(roi_int32_fm_index[curr_roi])
            w_num = (self.sample_num * self.pooled_width + 127) // Constant.BATCH_SIZE
            h_num = (self.sample_num * self.pooled_height + 127) // Constant.BATCH_SIZE

            flag_para = tik_instance.Scalar(dtype="int32", init_value=0)
            with tik_instance.if_scope(w_num > 1):
                flag_para.set_as(1)
            with tik_instance.if_scope(h_num > 1):
                flag_para.set_as(1)
            with tik_instance.if_scope(self.c1_num * self.pooled_width > 255):
                flag_para.set_as(1)

            with tik_instance.if_scope(flag_para == 0):
                grid_w_roi, grid_h_roi, rois_start_w, rois_start_h = \
                    self._get_input(grid_h, grid_w,
                                    proposals_ub_y0,
                                    proposals_ub_x0,
                                    curr_roi)

                x_lo_w, x_hi_w, y_lo_w, y_hi_w, \
                x_lo, x_hi, y_lo, y_hi, raw_x, raw_y = \
                    self._get_grid_weight(grid_w_roi, grid_h_roi, rois_start_w, rois_start_h)
                if self.l1_size != 0:
                    with tik_instance.if_scope(self.feature_map_to_ub_verify >= 1):
                        with tik_instance.new_stmt_scope():
                            self._bilinear_interpolate_all_in_ub(x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                                                                x_lo, x_hi, y_lo, y_hi, raw_x, raw_y,
                                                                n_bust, core_bias, index,
                                                                curr_roi, roi_128_number, cache_index)
                    with tik_instance.else_scope():
                        with tik_instance.if_scope(self.feature_map_to_l1_verify >= 1):
                            with tik_instance.new_stmt_scope():
                                self._bilinear_interpolate_all_in_l1(x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                                                                    x_lo, x_hi, y_lo, y_hi, raw_x, raw_y,
                                                                    n_bust, core_bias, index,
                                                                    curr_roi, roi_128_number, cache_index)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(self.w_number_ub >= 2):
                                with tik_instance.new_stmt_scope():
                                    self._bilinear_interpolate_w_in_ub(x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                                                                    x_lo, x_hi, y_lo, y_hi, raw_x, raw_y,
                                                                    n_bust, core_bias, index,
                                                                    curr_roi, roi_128_number, cache_index)
                            with tik_instance.else_scope():
                                with tik_instance.if_scope(self.w_number_l1 >= 2):
                                    with tik_instance.new_stmt_scope():
                                        self._bilinear_interpolate_w_in_l1(x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                                                                        x_lo, x_hi, y_lo, y_hi, raw_x, raw_y,
                                                                        n_bust, core_bias, index,
                                                                        curr_roi, roi_128_number, cache_index)
                                with tik_instance.else_scope():
                                    with tik_instance.new_stmt_scope():
                                        self._bilinear_interpolate_without_cache(x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                                                                                x_lo, x_hi, y_lo, y_hi, raw_x, raw_y,
                                                                                n_bust, core_bias, index,
                                                                                curr_roi, roi_128_number, cache_index)
                else:
                    with tik_instance.if_scope(self.feature_map_to_ub_verify >= 1):
                        with tik_instance.new_stmt_scope():
                            self._bilinear_interpolate_all_in_ub(x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                                                                x_lo, x_hi, y_lo, y_hi, raw_x, raw_y,
                                                                n_bust, core_bias, index,
                                                                curr_roi, roi_128_number, cache_index)
                    with tik_instance.else_scope():
                        with tik_instance.if_scope(self.w_number_ub >= 2):
                            with tik_instance.new_stmt_scope():
                                self._bilinear_interpolate_w_in_ub(x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                                                                x_lo, x_hi, y_lo, y_hi, raw_x, raw_y,
                                                                n_bust, core_bias, index,
                                                                curr_roi, roi_128_number, cache_index)
                        with tik_instance.else_scope():
                            with tik_instance.new_stmt_scope():
                                self._bilinear_interpolate_without_cache(x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                                                                        x_lo, x_hi, y_lo, y_hi, raw_x, raw_y,
                                                                        n_bust, core_bias, index,
                                                                        curr_roi, roi_128_number, cache_index)

    def _get_roi_align_perf_scale_for_zero(self, proposals_ub_x0,
                                           proposals_ub_y0, proposals_ub_x1,
                                           proposals_ub_y1):
        """
        get satart point, bin_size and sample number
        """
        tik_instance = self.tik_instance
        dtype_num, mask, repeat_times = self._calc_vector_params()

        roi_h_fp32 = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="roi_h_fp32", scope=tbe_platform.scope_ubuf)
        roi_w_fp32 = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="roi_w_fp32", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_muls(mask, proposals_ub_x0[0],
                              proposals_ub_x0[0],
                              self.spatial_scale, repeat_times, 8, 8)
        tik_instance.vec_muls(mask, proposals_ub_y0[0],
                              proposals_ub_y0[0],
                              self.spatial_scale, repeat_times, 8, 8)

        if self.roi_end_mode == 1:
            tik_instance.vec_adds(mask, proposals_ub_x1[0],
                                  proposals_ub_x1[0], 1,
                                  repeat_times, 8, 8)
            tik_instance.vec_adds(mask, proposals_ub_y1[0],
                                  proposals_ub_y1[0], 1,
                                  repeat_times, 8, 8)

        tik_instance.vec_muls(mask, proposals_ub_x1[0],
                              proposals_ub_x1[0],
                              self.spatial_scale, repeat_times, 8, 8)
        tik_instance.vec_muls(mask, proposals_ub_y1[0],
                              proposals_ub_y1[0],
                              self.spatial_scale, repeat_times, 8, 8)

        tik_instance.vec_sub(mask, roi_h_fp32, proposals_ub_y1[0],
                             proposals_ub_y0[0], repeat_times,
                             8, 8, 8)
        tik_instance.vec_sub(mask, roi_w_fp32, proposals_ub_x1[0],
                             proposals_ub_x0[0], repeat_times,
                             8, 8, 8)

        # Declare roi_bin_size tik_instance.Tensor
        roi_bin_h_fp32_value = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="roi_bin_h_fp32_value", scope=tbe_platform.scope_ubuf)
        roi_bin_w_fp32_value = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="roi_bin_w_fp32_value", scope=tbe_platform.scope_ubuf)
        grid_h = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="grid_h", scope=tbe_platform.scope_ubuf)
        grid_w = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="grid_w", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_muls(64 * dtype_num, roi_bin_h_fp32_value[:],
                              roi_h_fp32[:], 1.0 / self.pooled_height,
                              repeat_times, 8, 8)
        tik_instance.vec_muls(64 * dtype_num, roi_bin_w_fp32_value[:],
                              roi_w_fp32[:], 1.0 / self.pooled_width,
                              repeat_times, 8, 8)
        tik_instance.vec_muls(64 * dtype_num, grid_h[0],
                              roi_bin_h_fp32_value[0], 1.0 / self.sample_num,
                              repeat_times, 8, 8)
        tik_instance.vec_muls(64 * dtype_num, grid_w[0],
                              roi_bin_w_fp32_value[0], 1.0 / self.sample_num,
                              repeat_times, 8, 8)

        return roi_bin_h_fp32_value, roi_bin_w_fp32_value, proposals_ub_x0, proposals_ub_y0, grid_h, grid_w

    # 'pylint: disable=too-many-locals
    def _compute_mode_1(self, core_rois_n, core_bias):
        """
        roi_align_tik
        """
        tik_instance = self.tik_instance

        if self.dtype == "float32":
            n_bust = 2
        else:
            n_bust = 1

        rois_ub = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE, 8], name="rois_ub", scope=tbe_platform.scope_ubuf)
        proposals_ub_x0 = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="proposals_ub_x0", scope=tbe_platform.scope_ubuf)
        proposals_ub_y0 = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="proposals_ub_y0", scope=tbe_platform.scope_ubuf)
        proposals_ub_x1 = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="proposals_ub_x1", scope=tbe_platform.scope_ubuf)
        proposals_ub_y1 = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="proposals_ub_y1", scope=tbe_platform.scope_ubuf)
        roi_float_fm_index = tik_instance.Tensor(
            self.dtype, [Constant.BATCH_SIZE], name="roi_float_fm_index", scope=tbe_platform.scope_ubuf)
        roi_int32_fm_index = tik_instance.Tensor(
            "int32", [Constant.BATCH_SIZE], name="roi_int32_fm_index", scope=tbe_platform.scope_ubuf)

        if self.dtype == "float32":
            tik_instance.vector_dup(64, rois_ub, 0.0, 16, 1, 8)
        else:
            tik_instance.vector_dup(128, rois_ub, 0.0, 8, 1, 8)

        rois_valid = tik_instance.Scalar(dtype="int32", init_value=core_rois_n)
        rois_batch_num = (core_rois_n + 127) // Constant.BATCH_SIZE
        self._calc_buffer_verify()
        with tik_instance.if_scope(rois_valid != 0):
            with tik_instance.for_range(0, rois_batch_num) as roi_128_number:
                rois_valid_in_block = \
                    tik_instance.Scalar(dtype="int32", init_value=Constant.BATCH_SIZE)
                with tik_instance.if_scope(roi_128_number == (rois_batch_num - 1)):
                    rois_valid_in_block.set_as(rois_valid - roi_128_number * Constant.BATCH_SIZE)

                with tik_instance.if_scope(self.rois_row_length == 5):
                    rois_ub_n5 = tik_instance.Tensor(
                        self.dtype, [Constant.BATCH_SIZE, 5], name="rois_ub_n5", scope=tbe_platform.scope_ubuf)
                    burst_num = (rois_valid_in_block * 5 * n_bust + 15) // 16
                    tik_instance.data_move(rois_ub_n5[0, 0],
                                           self.rois[(core_bias + roi_128_number * Constant.BATCH_SIZE) * 5],
                                           0, 1, burst_num, 0, 0)
                    self._tf_n52n8(rois_ub, rois_ub_n5, rois_valid_in_block)
                with tik_instance.else_scope():
                    burst_num = (rois_valid_in_block * 8 * n_bust + 15) // 16
                    tik_instance.data_move(rois_ub[0, 0],
                                           self.rois[(core_bias + roi_128_number * Constant.BATCH_SIZE) * 8],
                                           0, 1, burst_num, 0, 0)

                support_vextract = tbe_platform.api_check_support("tik.vextract", "float32")
                if not support_vextract:
                    with tik_instance.for_range(0, rois_valid_in_block) as j:
                        roi_float_fm_index[j].set_as(rois_ub[j, 0])
                        proposals_ub_x0[j].set_as(rois_ub[j, 1])
                        proposals_ub_y0[j].set_as(rois_ub[j, 2])
                        proposals_ub_x1[j].set_as(rois_ub[j, 3])
                        proposals_ub_y1[j].set_as(rois_ub[j, 4])
                else:
                    tik_instance.vextract(roi_float_fm_index[0], rois_ub[0], 8, 0)
                    tik_instance.vextract(proposals_ub_x0[0], rois_ub[0], 8, 1)
                    tik_instance.vextract(proposals_ub_y0[0], rois_ub[0], 8, 2)
                    tik_instance.vextract(proposals_ub_x1[0], rois_ub[0], 8, 3)
                    tik_instance.vextract(proposals_ub_y1[0], rois_ub[0], 8, 4)

                tik_instance.vec_conv(64, "ceil", roi_int32_fm_index[0],
                                      roi_float_fm_index[0], 2, 8, 4 * n_bust)

                _, _, proposals_ub_x0, proposals_ub_y0, grid_h, grid_w = \
                    self._get_roi_align_perf_scale_for_zero(proposals_ub_x0,
                                                            proposals_ub_y0,
                                                            proposals_ub_x1,
                                                            proposals_ub_y1)

                self._common_compute(proposals_ub_x0, proposals_ub_y0, roi_128_number,
                                     rois_valid_in_block, n_bust,
                                     roi_int32_fm_index, grid_h, grid_w, core_bias)

    def _roi_align_compute_tiling(self):
        """
        define roi_align tiling method
        """
        tik_instance = self.tik_instance

        with tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            # get tiling data
            tiling_ub = tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), name="tiling_ub",
                                            scope=tik.scope_ubuf)
            tik_instance.data_move(tiling_ub, self.tiling_gm, 0,
                                   1, ceil_value(Constant.TILING_ARG_NUM, self.tiling_dtype),
                                   0, 0)
            # get run info
            self._get_tiling_args(tiling_ub)

            with self.tik_instance.if_scope(block_id < self.real_core_num):
                core_rois_n = self.tik_instance.Scalar(init_value=self.rois_n // self.real_core_num,
                                                       dtype="int32", name="core_rois_n")
                core_tail = self.tik_instance.Scalar(init_value=self.rois_n % self.real_core_num,
                                                     dtype="int32", name="core_tail")
                core_bias = self.tik_instance.Scalar(init_value=core_rois_n * block_id,
                                                     dtype="int32", name="core_bias")

                with self.tik_instance.if_scope(core_tail != 0):
                    with self.tik_instance.if_scope(block_id < core_tail):
                        core_rois_n.set_as(core_rois_n + 1)
                        core_bias.set_as(core_rois_n * block_id)
                    with self.tik_instance.else_scope():
                        core_bias.set_as(core_rois_n * block_id + core_tail)

                with tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
                    with tik_instance.new_stmt_scope():
                        self._compute_mode_1(core_rois_n, core_bias)
                with tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_2):
                    with tik_instance.new_stmt_scope():
                        pass

    def roi_align_compute(self):
        """calc one block gradient
        """
        tik_instance = self.tik_instance
        self.feature_map = tik_instance.Tensor(self.dtype, (Constant.PARAMS_SIZE,),
                                               name="feature_map", scope=tbe_platform.scope_gm)
        self.rois = tik_instance.Tensor(self.dtype, (Constant.PARAMS_SIZE,),
                                        name="rois_data", scope=tbe_platform.scope_gm)
        self.output = tik_instance.Tensor(self.dtype, (Constant.PARAMS_SIZE,), name="x_diff",
                                          scope=tbe_platform.scope_gm)
        self.tiling_gm = tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                             name="tiling_gm", scope=tik.scope_gm)

        inputs = [self.feature_map, self.rois]
        if self.exist_rois_n:
            self.rois_n_gm = tik_instance.Tensor(
                "int32", (Constant.PARAMS_SIZE,), name="rois_n_gm", scope=tbe_platform.scope_gm)
            inputs.append(self.rois_n_gm)

        self._roi_align_compute_tiling()

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }
        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num, "ub_size": self.ub_size})

        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=inputs,
                              outputs=[self.output],
                              flowtable=(self.tiling_gm,),
                              enable_l2=True, config=opt_config)


# 'pylint: disable=unused-argument,too-many-arguments
@register_operator("ROIAlign")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def roi_align(feature_map_dict,
              rois_dict,
              roisn_dict,
              output,
              scale,
              pool_h,
              pool_w,
              sample_ratio=2,
              roi_end_mode=1,
              kernel_name="roi_align",
              impl_mode="high_performance"):
    """
    ROIAlign operator
    """
    dtype = feature_map_dict.get("dtype")
    roi_align_obj = RoiAlign()
    roi_align_obj.pooled_height = pool_h
    roi_align_obj.pooled_width = pool_w
    roi_align_obj.sample_num = sample_ratio
    roi_align_obj.spatial_scale = scale
    roi_align_obj.dtype = dtype.lower()
    roi_align_obj.roi_end_mode = roi_end_mode
    roi_align_obj.kernel_name = kernel_name
    if roisn_dict:
        roi_align_obj.exist_rois_n = True

    return roi_align_obj.roi_align_compute()
