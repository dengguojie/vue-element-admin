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
roi_align_grad
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

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
TILING_MODE_3 = 3


# pylint: disable-msg=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=unused-argument
def ceil_value(value, dtype):
    """
    if not divide exactly then plus 1
    """
    value *= TYPE_LEN_DICT.get(dtype)

    return (value + BLOCK_SIZE - 1) // BLOCK_SIZE


def align_value(value, factor):
    """
    Alignment based on factor.
    """
    return (value + factor - 1) // factor * factor


class RoiAlignGrad():
    """
    define roi_align_grad object
    """

    def __init__(self):
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.core_num = profile.get_aicore_num()
        self.ub_size = profile.get_unified_buffer_size()
        self.tiling_mode = 0
        self.real_core_num = 0
        self.tiling_dtype = INT64
        self.rois_n = 0
        self.rois_row_length = 0
        self.c1_num = 0
        self.x_diff = None
        self.y_diff = None
        self.rois = None
        self.tiling_gm = None
        self.pooled_width = 0
        self.pooled_height = 0
        self.x_height = 0
        self.x_width = 0
        self.spatial_scale = 0.0
        self.sample_num = 0
        self.kernel_name = None
        self.available_c1_num = None
        self.exist_rois_n = False
        self.rois_n_gm = None

    def _get_tiling_args(self, tiling_ub):
        """
        get tiling args
        """
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

    def _get_y_diff_offset(self, n, c1, h, w):
        """calc y_diff offset
        """
        n_offset = n * self.c1_num * self.pooled_height * self.pooled_width * 16
        c1_offset = c1 * self.pooled_height * self.pooled_width * 16
        h_offset = h * self.pooled_width * 16
        w_offset = w * 16

        return n_offset + c1_offset + h_offset + w_offset

    def _get_x_diff_offset(self, n, c1, h, w):
        """calc x_diff offset
        """
        n_offset = n * self.c1_num * self.x_height * self.x_width * 16
        c1_offset = c1 * self.x_height * self.x_width * 16
        h_offset = h * self.x_width * 16
        w_offset = w * 16

        return n_offset + c1_offset + h_offset + w_offset

    def _get_temp_ub_offset(self, n, c1, h, w):
        """calc temp_ub offset
        """
        n_offset = n * 16 * 2 * self.x_width * 16
        c1_offset = c1 * 2 * self.x_width * 16
        h_offset = h * self.x_width * 16
        w_offset = w * 16

        return n_offset + c1_offset + h_offset + w_offset

    def _modify_value(self, c1_gap, w_gap):
        tik_instance = self.tik_instance
        with tik_instance.if_scope(c1_gap < 0):
            c1_gap = 0

        with tik_instance.if_scope(w_gap < 0):
            w_gap = 0

        return c1_gap, w_gap

    def _calc_max_c1_num(self):
        """calc_max_c1_num ub resource
        """
        tik_instance = self.tik_instance
        available_res = 160 * 1024
        self.available_c1_num = tik_instance.Scalar("int32", name="available_c1_num")
        self.available_c1_num.set_as(available_res // ((self.pooled_width + 14) * 16 * 4))
        with tik_instance.if_scope(self.available_c1_num > self.c1_num):
            self.available_c1_num = self.c1_num

    def _clear_ub(self, dst_ub):
        """clear ub to zero
        """
        tik_instance = self.tik_instance
        shape = dst_ub.shape
        data_len = 1
        for i in shape:
            data_len = data_len * i
        dst_ub.reshape((data_len,))

        total_repeat_times = data_len // 64
        tail = data_len % 64
        vector_dup_times = total_repeat_times
        offset = 0
        while vector_dup_times > 255:
            tik_instance.vector_dup(64, dst_ub[offset], 0, 255, 1, 8)
            vector_dup_times = vector_dup_times - 255
            offset = offset + 64 * 255

        if vector_dup_times > 0:
            tik_instance.vector_dup(64, dst_ub[offset], 0, vector_dup_times, 1, 8)
            offset = offset + 64 * vector_dup_times

        if tail > 0:
            tik_instance.vector_dup(tail, dst_ub[offset], 0, 1, 1, 8)

        dst_ub.reshape(shape)

    def _mov_data_ddr(self, x_diff, x_diff_ub,
                      image_index, start_c1, c1_num, h_index, w_index, sum_in_ub_flag):
        """mov_data_ddr
        """
        tik_instance = self.tik_instance
        h_num = self.x_height
        w_num = self.x_width

        with tik_instance.if_scope(h_index < (h_num - 1)):
            with tik_instance.if_scope(w_index < (w_num - 1)):
                self._mov_data_ddr_all(x_diff, x_diff_ub, image_index,
                                       start_c1, c1_num, h_index, w_index, sum_in_ub_flag)
            with tik_instance.else_scope():
                self._mov_data_ddr_onerow(x_diff, x_diff_ub, image_index,
                                          start_c1, c1_num, h_index, w_index, sum_in_ub_flag)
        with tik_instance.else_scope():
            with tik_instance.if_scope(w_index < (w_num - 1)):
                self._mov_data_ddr_oneline(x_diff, x_diff_ub, image_index,
                                           start_c1, c1_num, h_index, w_index, sum_in_ub_flag)
            with tik_instance.else_scope():
                self._mov_data_ddr_onepoint(x_diff, x_diff_ub, image_index,
                                            start_c1, c1_num, h_index, w_index, sum_in_ub_flag)

    def _mov_data_ddr_onepoint(self, x_diff, x_diff_ub, image_index,
                               start_c1, c1_num, h_index, w_index, sum_in_ub_flag):
        """_mov_data_ddr_onepoint
        """
        tik_instance = self.tik_instance
        h_num = self.x_height
        w_num = self.x_width

        if sum_in_ub_flag:
            repeat_str = 2 * w_num * 16 // 8
            if repeat_str <= 255:
                x_offset = self._get_temp_ub_offset(image_index, start_c1, h_index, w_index)
                tik_instance.vadd(16, x_diff[x_offset], x_diff[x_offset], x_diff_ub[0, 0, 0],
                                  c1_num, 1, 1, 1, repeat_str, repeat_str, 8)
            else:
                with tik_instance.for_range(0, c1_num) as i:
                    x_offset = self._get_temp_ub_offset(image_index, start_c1 + i, h_index, w_index)
                    tik_instance.vadd(16, x_diff[x_offset], x_diff[x_offset], x_diff_ub[i, 0, 0],
                                      1, 1, 1, 1, 8, 8, 8)
        else:
            c1_gap = (((h_num * w_num) - 1) * 16 * 4) // 32
            w_gap = ((w_num - 2) * 16 * 4) // 32
            # `c1_gap, w_gap = self._modify_value(c1_gap, w_gap)`
            with tik_instance.if_scope(c1_gap <= 65535):
                x_offset = self._get_x_diff_offset(image_index, start_c1, h_index, w_index)
                tik_instance.tensor_mov(x_diff[x_offset],
                                        x_diff_ub[0, 0, 0], '', c1_num, 2, c1_gap, 6)
            with tik_instance.else_scope():
                with tik_instance.if_scope(w_gap <= 65535):
                    with tik_instance.for_range(0, c1_num) as i:
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 0, 0], '', 2, 4, w_gap, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, c1_num) as i:
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 0, 0], '', 1, 2, 0, 0)

    def _mov_data_ddr_oneline(self, x_diff, x_diff_ub, image_index,
                              start_c1, c1_num, h_index, w_index, sum_in_ub_flag):
        """move xdiff to gm
        """
        tik_instance = self.tik_instance
        h_num = self.x_height
        w_num = self.x_width

        if sum_in_ub_flag:
            repeat_str = 2 * w_num * 16 // 8
            with tik_instance.if_scope(repeat_str <= 255):
                x_offset = self._get_temp_ub_offset(image_index, start_c1, h_index, w_index)
                tik_instance.vadd(32, x_diff[x_offset], x_diff[x_offset], x_diff_ub[0, 0, 0],
                                  c1_num, 1, 1, 1, repeat_str, repeat_str, 8)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, c1_num) as i:
                    x_offset = self._get_temp_ub_offset(image_index, start_c1 + i, h_index, w_index)
                    tik_instance.vadd(32, x_diff[x_offset], x_diff[x_offset], x_diff_ub[i, 0, 0],
                                      1, 1, 1, 1, 8, 8, 8)
        else:
            c1_gap = (((h_num * w_num) - 2) * 16 * 4) // 32
            w_gap = ((w_num - 2) * 16 * 4) // 32
            # `c1_gap, w_gap = self._modify_value(c1_gap, w_gap)`
            with tik_instance.if_scope(c1_gap <= 65535):
                x_offset = self._get_x_diff_offset(image_index, start_c1, h_index, w_index)
                tik_instance.tensor_mov(x_diff[x_offset],
                                        x_diff_ub[0, 0, 0], '', c1_num, 4, c1_gap, 4)
            with tik_instance.else_scope():
                with tik_instance.if_scope(w_gap <= 65535):
                    with tik_instance.for_range(0, c1_num) as i:
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 0, 0], '', 2, 4, w_gap, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, c1_num) as i:
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 0, 0], '', 1, 4, 0, 0)

    def _mov_data_ddr_onerow(self, x_diff, x_diff_ub, image_index,
                             start_c1, c1_num, h_index, w_index, sum_in_ub_flag):
        """move xdiff to gm
        """
        tik_instance = self.tik_instance
        h_num = self.x_height
        w_num = self.x_width

        if sum_in_ub_flag:
            repeat_str = 2 * w_num * 16 // 8
            with tik_instance.if_scope(repeat_str <= 255):
                x_offset = self._get_temp_ub_offset(image_index, start_c1, h_index, w_index)
                tik_instance.vadd(16, x_diff[x_offset], x_diff[x_offset], x_diff_ub[0, 0, 0],
                                  c1_num, 1, 1, 1, repeat_str, repeat_str, 8)
                x_offset = self._get_temp_ub_offset(image_index, start_c1, h_index + 1, w_index)
                tik_instance.vadd(16, x_diff[x_offset], x_diff[x_offset], x_diff_ub[0, 2, 0],
                                  c1_num, 1, 1, 1, repeat_str, repeat_str, 8)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, c1_num) as i:
                    x_offset = self._get_temp_ub_offset(image_index, start_c1 + i, h_index, w_index)
                    tik_instance.vadd(16, x_diff[x_offset], x_diff[x_offset], x_diff_ub[i, 0, 0],
                                      1, 1, 1, 1, 8, 8, 8)
                    x_offset = self._get_temp_ub_offset(image_index, start_c1 + i, h_index + 1, w_index)
                    tik_instance.vadd(16, x_diff[x_offset], x_diff[x_offset], x_diff_ub[i, 2, 0],
                                      1, 1, 1, 1, 8, 8, 8)
        else:
            c1_gap = (((h_num * w_num) - 1) * 16 * 4) // 32
            w_gap = ((w_num - 2) * 16 * 4) // 32
            # `c1_gap, w_gap = self._modify_value(c1_gap, w_gap)`
            with tik_instance.if_scope(c1_gap <= 65535):
                x_offset = self._get_x_diff_offset(image_index, start_c1, h_index, w_index)
                tik_instance.tensor_mov(x_diff[x_offset],
                                        x_diff_ub[0, 0, 0], '', c1_num, 2, c1_gap, 6)
                x_offset = self._get_x_diff_offset(image_index, start_c1, h_index + 1, w_index)
                tik_instance.tensor_mov(x_diff[x_offset],
                                        x_diff_ub[0, 2, 0], '', c1_num, 2, c1_gap, 6)

            with tik_instance.else_scope():
                with tik_instance.if_scope(w_gap <= 65535):
                    with tik_instance.for_range(0, c1_num) as i:
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 0, 0], '', 2, 4, w_gap, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, c1_num) as i:
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 0, 0], '', 1, 2, 0, 0)
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index + 1, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 2, 0], '', 1, 2, 0, 0)

    def _mov_data_ddr_all(self, x_diff, x_diff_ub, image_index,
                          start_c1, c1_num, h_index, w_index, sum_in_ub_flag):
        """mov_data_ddr_all
        """
        tik_instance = self.tik_instance
        h_num = self.x_height
        w_num = self.x_width

        if sum_in_ub_flag:
            repeat_str = 2 * w_num * 16 // 8
            with tik_instance.if_scope(repeat_str <= 255):
                x_offset = self._get_temp_ub_offset(image_index, start_c1, h_index, w_index)
                tik_instance.vadd(32, x_diff[x_offset], x_diff[x_offset], x_diff_ub[0, 0, 0],
                                  c1_num, 1, 1, 1, repeat_str, repeat_str, 8)
                x_offset = self._get_temp_ub_offset(image_index, start_c1, h_index + 1, w_index)
                tik_instance.vadd(32, x_diff[x_offset], x_diff[x_offset], x_diff_ub[0, 2, 0],
                                  c1_num, 1, 1, 1, repeat_str, repeat_str, 8)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, c1_num) as i:
                    x_offset = self._get_temp_ub_offset(image_index, start_c1 + i, h_index, w_index)
                    tik_instance.vadd(32, x_diff[x_offset], x_diff[x_offset], x_diff_ub[i, 0, 0],
                                      1, 1, 1, 1, 8, 8, 8)
                    x_offset = self._get_temp_ub_offset(image_index, start_c1 + i, h_index + 1, w_index)
                    tik_instance.vadd(32, x_diff[x_offset], x_diff[x_offset], x_diff_ub[i, 2, 0],
                                      1, 1, 1, 1, 8, 8, 8)
        else:
            c1_gap = (((h_num * w_num) - 2) * 16 * 4) // 32
            w_gap = ((w_num - 2) * 16 * 4) // 32
            # `c1_gap, w_gap = self._modify_value(c1_gap, w_gap)`
            with tik_instance.if_scope(c1_gap <= 65535):
                x_offset = self._get_x_diff_offset(image_index, start_c1, h_index, w_index)
                tik_instance.tensor_mov(x_diff[x_offset],
                                        x_diff_ub[0, 0, 0], '', c1_num, 4, c1_gap, 4)
                x_offset = self._get_x_diff_offset(image_index, start_c1, h_index + 1, w_index)
                tik_instance.tensor_mov(x_diff[x_offset],
                                        x_diff_ub[0, 2, 0], '', c1_num, 4, c1_gap, 4)
            with tik_instance.else_scope():
                with tik_instance.if_scope(w_gap <= 65535):
                    with tik_instance.for_range(0, c1_num) as i:
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 0, 0], '',
                                                2, 4, w_gap, 0)
                with tik_instance.else_scope():
                    with tik_instance.for_range(0, c1_num) as i:
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 0, 0], '', 1, 4, 0, 0)
                        x_offset = self._get_x_diff_offset(image_index, start_c1 + i, h_index + 1, w_index)
                        tik_instance.tensor_mov(x_diff[x_offset],
                                                x_diff_ub[i, 2, 0], '', 1, 4, 0, 0)

    def _calc_w_vec(self, x_lo_w, x_hi_w, y_lo_w, y_hi_w):
        tik_instance = self.tik_instance
        w1_vec = tik_instance.Tensor(
            "float32", (128,), name="w1", scope=tbe_platform.scope_ubuf)
        w2_vec = tik_instance.Tensor(
            "float32", (128,), name="w2", scope=tbe_platform.scope_ubuf)
        w3_vec = tik_instance.Tensor(
            "float32", (128,), name="w3", scope=tbe_platform.scope_ubuf)
        w4_vec = tik_instance.Tensor(
            "float32", (128,), name="w4", scope=tbe_platform.scope_ubuf)

        tik_instance.vmuls(64, w1_vec, x_hi_w, y_hi_w, 2, 1, 1, 8, 8)
        tik_instance.vmuls(64, w2_vec, x_lo_w, y_hi_w, 2, 1, 1, 8, 8)
        tik_instance.vmuls(64, w3_vec, x_hi_w, y_lo_w, 2, 1, 1, 8, 8)
        tik_instance.vmuls(64, w4_vec, x_lo_w, y_lo_w, 2, 1, 1, 8, 8)

        return w1_vec, w2_vec, w3_vec, w4_vec

    def _calc_x_diff_ub(self, w1_vec, w2_vec, w3_vec, w4_vec, row_num_i,
                        available_c1_num, x_ind, x_lo, x_hi, y_hi, y_diff_ub,
                        c1_num, calc_c1_num):
        tik_instance = self.tik_instance
        pool_w = self.pooled_width
        w1_reg = tik_instance.Scalar(dtype="float32")
        w2_reg = tik_instance.Scalar(dtype="float32")
        w3_reg = tik_instance.Scalar(dtype="float32")
        w4_reg = tik_instance.Scalar(dtype="float32")
        x_lo_reg = tik_instance.Scalar(dtype="int32")
        x_hi_reg = tik_instance.Scalar(dtype="int32")
        w4_loc_reg = tik_instance.Scalar(dtype="int32")

        grad_index_reg = tik_instance.Scalar(dtype="int32")
        x_diff_ub = tik_instance.Tensor(
            "float32", [available_c1_num, 4, 16], name="x_diff_ub", scope=tbe_platform.scope_ubuf)
        tmp_result = tik_instance.Tensor(
            "float32", [3 * available_c1_num * 16],
            name="tmp_result",
            scope=tbe_platform.scope_ubuf)

        grad_index_reg.set_as(x_ind[row_num_i])
        w1_reg.set_as(w1_vec[row_num_i])
        w2_reg.set_as(w2_vec[row_num_i])
        w3_reg.set_as(w3_vec[row_num_i])
        w4_reg.set_as(w4_vec[row_num_i])
        x_lo_reg.set_as(x_lo[row_num_i])
        x_hi_reg.set_as(x_hi[row_num_i])
        w2_loc_reg = x_hi_reg
        w3_loc_reg = 2 * y_hi
        w4_loc_reg.set_as(w2_loc_reg + w3_loc_reg)

        self._clear_ub(x_diff_ub)
        tik_instance.vmuls(16, x_diff_ub[0, 0, 0],
                           y_diff_ub[0, 0, grad_index_reg, 0], w1_reg,
                           calc_c1_num, 1, 1, 8, pool_w * 2)
        tik_instance.vmuls(16, tmp_result[0],
                           y_diff_ub[0, 0, grad_index_reg, 0], w2_reg,
                           calc_c1_num, 1, 1, 2, pool_w * 2)
        tik_instance.vmuls(16, tmp_result[c1_num * 16],
                           y_diff_ub[0, 0, grad_index_reg, 0], w3_reg,
                           calc_c1_num, 1, 1, 2, pool_w * 2)
        tik_instance.vmuls(16, tmp_result[c1_num * 2 * 16],
                           y_diff_ub[0, 0, grad_index_reg, 0], w4_reg,
                           calc_c1_num, 1, 1, 2, pool_w * 2)

        tik_instance.vadd(16, x_diff_ub[0, w2_loc_reg, 0],
                          x_diff_ub[0, w2_loc_reg, 0], tmp_result[0],
                          calc_c1_num, 1, 1, 1, 8, 8, 2)
        tik_instance.vadd(16, x_diff_ub[0, w3_loc_reg, 0],
                          x_diff_ub[0, w3_loc_reg, 0], tmp_result[c1_num * 16],
                          calc_c1_num, 1, 1, 1, 8, 8, 2)
        tik_instance.vadd(16, x_diff_ub[0, w4_loc_reg, 0],
                          x_diff_ub[0, w4_loc_reg, 0], tmp_result[c1_num * 2 * 16],
                          calc_c1_num, 1, 1, 1, 8, 8, 2)

        return x_diff_ub, x_lo_reg

    def _roi_align_calc_grad_line_align_mode1(self, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo,
                                              x_hi, y_hi, y_lo, row_num, y_diff_ub, x_ind,
                                              image_index, start_c1, calc_c1_num):
        """calc one line gradient
        """
        tik_instance = self.tik_instance
        w1_vec, w2_vec, w3_vec, w4_vec = self._calc_w_vec(x_lo_w, x_hi_w, y_lo_w, y_hi_w)

        c1_num = self.available_c1_num
        available_c1_num = 16

        with tik_instance.new_stmt_scope():
            if row_num == 1:
                thread_num = 1
            else:
                thread_num = 2
            with tik_instance.for_range(0, row_num, thread_num=thread_num) as i:
                x_diff_ub, x_lo_reg = self._calc_x_diff_ub(w1_vec, w2_vec, w3_vec, w4_vec, i,
                                                           available_c1_num, x_ind, x_lo, x_hi, y_hi, y_diff_ub,
                                                           c1_num, calc_c1_num)

                self._mov_data_ddr(self.x_diff, x_diff_ub, image_index, start_c1,
                                   calc_c1_num, y_lo, x_lo_reg, False)

    def _roi_align_calc_grad_line_align_mode2(self, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo,
                                              x_hi, y_hi, y_lo, row_num, y_diff_ub, x_ind,
                                              image_index, start_c1, calc_c1_num):
        """calc one line gradient
        """
        tik_instance = self.tik_instance
        w1_vec, w2_vec, w3_vec, w4_vec = self._calc_w_vec(x_lo_w, x_hi_w, y_lo_w, y_hi_w)

        c1_num = self.available_c1_num
        available_c1_num = 16
        h_num = self.x_height
        w_num = self.x_width

        with tik_instance.new_stmt_scope():
            tmp_ub = tik_instance.Tensor("float32", [available_c1_num * 2 * 80 * 16],
                                         name="tmp_ub", scope=tbe_platform.scope_ubuf)
            self._clear_ub(tmp_ub)

            if row_num == 1:
                thread_num = 1
            else:
                thread_num = 2
            with tik_instance.for_range(0, row_num, thread_num=thread_num) as i:
                x_diff_ub, x_lo_reg = self._calc_x_diff_ub(w1_vec, w2_vec, w3_vec, w4_vec, i,
                                                           available_c1_num, x_ind, x_lo, x_hi, y_hi, y_diff_ub,
                                                           c1_num, calc_c1_num)
                self._mov_data_ddr(tmp_ub, x_diff_ub, 0, start_c1,
                                   calc_c1_num, 0, x_lo_reg, True)

            with tik_instance.if_scope(y_lo < (h_num - 1)):
                x_diff_offset = self._get_x_diff_offset(image_index, 0, y_lo, 0)
                tik_instance.data_move(self.x_diff[x_diff_offset], tmp_ub,
                                       0, c1_num, w_num * 2 * 16 // 8, 0,
                                       h_num * w_num * 16 // 8 - w_num * 2 * 16 // 8)
            with tik_instance.else_scope():
                x_diff_offset = self._get_x_diff_offset(image_index, 0, y_lo, 0)
                tik_instance.data_move(self.x_diff[x_diff_offset], tmp_ub,
                                       0, c1_num, w_num * 16 // 8, 0, h_num * w_num * 16 // 8 - w_num * 16 // 8)

    def _get_ydiff_line(self, y_diff_ub, n_index, start_c1, calc_c1_num, line_index):
        """get one line ydiff data
        """
        tik_instance = self.tik_instance
        h_num = self.pooled_height
        w_num = self.pooled_width

        c1_gap = ((h_num - 1) * w_num * 16 * 4) // 32
        if c1_gap <= 65535:
            y_diff_offset = self._get_y_diff_offset(n_index, start_c1, line_index, 0)
            tik_instance.tensor_mov(y_diff_ub[0],
                                    self.y_diff[y_diff_offset], '',
                                    calc_c1_num, w_num * 2, 0, c1_gap)
        else:
            with tik_instance.for_range(0, calc_c1_num) as i:
                y_diff_offset = self._get_y_diff_offset(n_index, start_c1 + i, line_index, 0)
                tik_instance.tensor_mov(
                    y_diff_ub[0], self.y_diff[y_diff_offset],
                    '', 1, w_num * 2, 0, 0)

    def _malloc_res(self):
        """malloc ub resource
        """
        tik_instance = self.tik_instance
        c1_shape = self.c1_num
        pool_w = self.pooled_width

        self._calc_max_c1_num()
        c1_size = 16
        y_diff_ub = tik_instance.Tensor(
            "float32", [c1_size, 1, pool_w, 16],
            name="y_diff_ub",
            scope=tbe_platform.scope_ubuf)

        return y_diff_ub

    def _roi_align_calc_grad_block_align(self, line_num, row_num, x_lo_w, x_hi_w,
                                         y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi,
                                         n_index, x_ind, y_ind, image_index, sum_in_ub_flag):
        """calc one block gradient
        """
        tik_instance = self.tik_instance
        y_diff_ub = self._malloc_res()
        y_lo_w_s = tik_instance.Scalar(dtype="float32")
        y_hi_w_s = tik_instance.Scalar(dtype="float32")
        y_ind_s = tik_instance.Scalar(dtype="int32")
        y_hi_s = tik_instance.Scalar(dtype="int32")
        y_lo_s = tik_instance.Scalar(dtype="int32")
        calc_c1_num = self.available_c1_num
        with tik_instance.for_range(0, line_num) as i:
            y_lo_w_s.set_as(y_lo_w[i])
            y_hi_w_s.set_as(y_hi_w[i])
            y_ind_s.set_as(y_ind[i])
            y_lo_s.set_as(y_lo[i])
            y_hi_s.set_as(y_hi[i])

            # move y_diff data from gm to ub
            self._get_ydiff_line(y_diff_ub, n_index, 0, calc_c1_num, y_ind_s)
            if not sum_in_ub_flag:
                self._roi_align_calc_grad_line_align_mode1(x_lo_w, x_hi_w, y_lo_w_s,
                                                           y_hi_w_s, x_lo, x_hi, y_hi_s, y_lo_s,
                                                           row_num, y_diff_ub, x_ind,
                                                           image_index, 0, calc_c1_num)
            else:
                self._roi_align_calc_grad_line_align_mode2(x_lo_w, x_hi_w, y_lo_w_s,
                                                           y_hi_w_s, x_lo, x_hi, y_hi_s, y_lo_s,
                                                           row_num, y_diff_ub, x_ind,
                                                           image_index, 0, calc_c1_num)

    def _roi_align_calc_grid(self, h_ind, w_ind, const_value_0_127, grid_w,
                             grid_h, rois_start_w, rois_start_h):
        """calc one block gradient
        """
        tik_instance = self.tik_instance
        x_lo_w = tik_instance.Tensor(
            "float32", [128], name="x_lo_w", scope=tbe_platform.scope_ubuf)
        x_hi_w = tik_instance.Tensor(
            "float32", [128], name="x_hi_w", scope=tbe_platform.scope_ubuf)
        y_lo_w = tik_instance.Tensor(
            "float32", [128], name="y_lo_w", scope=tbe_platform.scope_ubuf)
        y_hi_w = tik_instance.Tensor(
            "float32", [128], name="y_hi_w", scope=tbe_platform.scope_ubuf)
        x_lo = tik_instance.Tensor(
            "int32", [128], name="x_lo", scope=tbe_platform.scope_ubuf)
        x_hi = tik_instance.Tensor(
            "int32", [128], name="x_hi", scope=tbe_platform.scope_ubuf)
        y_lo = tik_instance.Tensor(
            "int32", [128], name="y_lo", scope=tbe_platform.scope_ubuf)
        y_hi = tik_instance.Tensor(
            "int32", [128], name="y_hi", scope=tbe_platform.scope_ubuf)

        raw_x = tik_instance.Tensor(
            "float32", [128], name="x", scope=tbe_platform.scope_ubuf)
        raw_y = tik_instance.Tensor(
            "float32", [128], name="y", scope=tbe_platform.scope_ubuf)
        x_vec = tik_instance.Tensor(
            "float32", [128], name="x", scope=tbe_platform.scope_ubuf)
        y_vec = tik_instance.Tensor(
            "float32", [128], name="y", scope=tbe_platform.scope_ubuf)
        x_ind = tik_instance.Tensor(
            "int32", [128], name="x_ind", scope=tbe_platform.scope_ubuf)
        y_ind = tik_instance.Tensor(
            "int32", [128], name="y_ind", scope=tbe_platform.scope_ubuf)

        tik_instance.vadds(64, x_vec, const_value_0_127, w_ind * 128, 2, 1, 1, 8, 8)
        tik_instance.vmuls(64, x_vec, x_vec, 1.0 / self.sample_num, 2, 1, 1, 8, 8)
        tik_instance.vadds(64, y_vec, const_value_0_127, h_ind * 128, 2, 1, 1, 8, 8)
        tik_instance.vmuls(64, y_vec, y_vec, 1.0 / self.sample_num, 2, 1, 1, 8, 8)
        tik_instance.vconv(64, "floor", x_ind, x_vec, 2, 1, 1, 8, 8)
        tik_instance.vconv(64, "floor", y_ind, y_vec, 2, 1, 1, 8, 8)

        grid_w_vector = tik_instance.Tensor(
            "float32", [128], name="grid_w_vector", scope=tbe_platform.scope_ubuf)
        grid_h_vector = tik_instance.Tensor(
            "float32", [128], name="grid_h_vector", scope=tbe_platform.scope_ubuf)
        tik_instance.vmuls(64, grid_w_vector, const_value_0_127, grid_w, 2, 1, 1, 8,
                           8)
        tik_instance.vmuls(64, grid_h_vector, const_value_0_127, grid_h, 2, 1, 1, 8,
                           8)

        half_grid = 0.5 * grid_w + rois_start_w
        tik_instance.vadds(64, raw_x, grid_w_vector, half_grid, 2, 1, 1, 8, 8)
        half_grid = 0.5 * grid_h + rois_start_h
        tik_instance.vadds(64, raw_y, grid_h_vector, half_grid, 2, 1, 1, 8, 8)

        const_zero = tik_instance.Tensor(
            "float32", [16], name="const_zero", scope=tbe_platform.scope_ubuf)
        tik_instance.vector_dup(16, const_zero, 0, 1, 0, 0)

        tik_instance.vmax(64, x_vec, raw_x, const_zero, 2, 1, 1, 0, 8, 8, 0)
        tik_instance.vmax(64, y_vec, raw_y, const_zero, 2, 1, 1, 0, 8, 8, 0)

        tik_instance.vconv(64, "floor", x_lo, x_vec, 2, 1, 1, 8, 8)
        tik_instance.vconv(64, "floor", y_lo, y_vec, 2, 1, 1, 8, 8)

        const_one = tik_instance.Tensor(
            "int32", [8], name="const_one", scope=tbe_platform.scope_ubuf)
        tik_instance.vector_dup(8, const_one, 1, 1, 0, 0)
        tik_instance.vadd(64, x_hi, x_lo, const_one, 2, 1, 1, 0, 8, 8, 0)
        tik_instance.vadd(64, y_hi, y_lo, const_one, 2, 1, 1, 0, 8, 8, 0)

        const_value_fp32 = tik_instance.Tensor(
            "float32", [16], name="const_value", scope=tbe_platform.scope_ubuf)
        const_value_int32 = tik_instance.Tensor(
            "int32", [16], name="const_value", scope=tbe_platform.scope_ubuf)
        tik_instance.vector_dup(16, const_value_fp32, self.x_width - 1, 1, 0, 0)
        tik_instance.vector_dup(16, const_value_int32, self.x_width - 1, 1, 0, 0)
        tik_instance.vmin(64, x_lo, x_lo, const_value_int32, 2, 1, 1, 0, 8, 8, 0)
        tik_instance.vmin(64, x_hi, x_hi, const_value_int32, 2, 1, 1, 0, 8, 8, 0)
        tik_instance.vsub(64, x_hi, x_hi, x_lo, 2, 1, 1, 1, 8, 8, 8)
        tik_instance.vmin(64, x_vec, x_vec, const_value_fp32, 2, 1, 1, 0, 8, 8, 0)

        tik_instance.vector_dup(16, const_value_int32, self.x_height - 1, 1, 0, 0)
        tik_instance.vector_dup(16, const_value_fp32, self.x_height - 1, 1, 0, 0)
        tik_instance.vmin(64, y_lo, y_lo, const_value_int32, 2, 1, 1, 0, 8, 8, 0)
        tik_instance.vmin(64, y_hi, y_hi, const_value_int32, 2, 1, 1, 0, 8, 8, 0)
        tik_instance.vsub(64, y_hi, y_hi, y_lo, 2, 1, 1, 1, 8, 8, 8)
        tik_instance.vmin(64, y_vec, y_vec, const_value_fp32, 2, 1, 1, 0, 8, 8, 0)

        tmp_fp32 = tik_instance.Tensor(
            "float32", [128], name="tmp_fp32", scope=tbe_platform.scope_ubuf)
        tik_instance.vconv(64, "", tmp_fp32, x_lo, 2, 1, 1, 8, 8)
        tik_instance.vsub(64, x_lo_w, x_vec, tmp_fp32, 2, 1, 1, 1, 8, 8, 8)
        tik_instance.vconv(64, "", tmp_fp32, y_lo, 2, 1, 1, 8, 8)
        tik_instance.vsub(64, y_lo_w, y_vec, tmp_fp32, 2, 1, 1, 1, 8, 8, 8)

        tik_instance.vector_dup(16, const_value_fp32, 1., 1, 0, 0)
        tik_instance.vsub(64, x_hi_w, const_value_fp32, x_lo_w, 2, 1, 0, 1, 8, 0, 8)
        tik_instance.vsub(64, y_hi_w, const_value_fp32, y_lo_w, 2, 1, 0, 1, 8, 0, 8)

        tik_instance.vector_dup(8, const_value_fp32, -1., 1, 0, 0)
        tik_instance.vector_dup(8, const_value_fp32[8], self.x_width + 0, 1, 0, 0)
        cmp_mask = tik_instance.Tensor(
            "uint16", (32,), name="cmp_mask", scope=tbe_platform.scope_ubuf)

        tik_instance.vcmpv_lt(cmp_mask, raw_x, const_value_fp32, 1, 1, 0, 8, 0)
        tik_instance.vcmpv_gt(cmp_mask[16], raw_x, const_value_fp32[8], 1, 1, 0, 8,
                              0)
        tik_instance.vor(4, cmp_mask, cmp_mask, cmp_mask[16], 1, 1, 1, 1, 8, 8, 8)

        dst_cmp_mask = tik_instance.mov_tensor_to_cmpmask(cmp_mask)
        tik_instance.vsel(64, 0, x_lo_w, dst_cmp_mask, const_zero, x_lo_w, 1, 1, 0,
                          1, 8, 0, 8)
        tik_instance.vsel(64, 0, x_hi_w, dst_cmp_mask, const_zero, x_hi_w, 1, 1, 0,
                          1, 8, 0, 8)

        tik_instance.vcmpv_lt(cmp_mask, raw_x[64], const_value_fp32, 1, 1, 0, 8, 0)
        tik_instance.vcmpv_gt(cmp_mask[16], raw_x[64], const_value_fp32[8], 1, 1, 0,
                              8, 0)
        tik_instance.vor(4, cmp_mask, cmp_mask, cmp_mask[16], 1, 1, 1, 1, 8, 8, 8)

        dst_cmp_mask = tik_instance.mov_tensor_to_cmpmask(cmp_mask)
        tik_instance.vsel(64, 0, x_lo_w[64], dst_cmp_mask, const_zero, x_lo_w[64],
                          1, 1, 0, 1, 8, 0, 8)
        tik_instance.vsel(64, 0, x_hi_w[64], dst_cmp_mask, const_zero, x_hi_w[64],
                          1, 1, 0, 1, 8, 0, 8)

        tik_instance.vmuls(64, x_lo_w, x_lo_w, 1.0 / self.sample_num, 2, 1, 1, 8, 8)
        tik_instance.vmuls(64, x_hi_w, x_hi_w, 1.0 / self.sample_num, 2, 1, 1, 8, 8)

        tik_instance.vector_dup(8, const_value_fp32[8], self.x_height + 0, 1, 0, 0)
        tik_instance.vcmpv_lt(cmp_mask, raw_y, const_value_fp32, 1, 1, 0, 8, 0)
        tik_instance.vcmpv_gt(cmp_mask[16], raw_y, const_value_fp32[8], 1, 1, 0, 8,
                              0)
        tik_instance.vor(4, cmp_mask, cmp_mask, cmp_mask[16], 1, 1, 1, 1, 8, 8, 8)

        dst_cmp_mask = tik_instance.mov_tensor_to_cmpmask(cmp_mask)
        tik_instance.vsel(64, 0, y_lo_w, dst_cmp_mask, const_zero, y_lo_w, 1, 1, 0,
                          1, 8, 0, 8)
        tik_instance.vsel(64, 0, y_hi_w, dst_cmp_mask, const_zero, y_hi_w, 1, 1, 0,
                          1, 8, 0, 8)

        tik_instance.vcmpv_lt(cmp_mask, raw_y[64], const_value_fp32, 1, 1, 0, 8, 0)
        tik_instance.vcmpv_gt(cmp_mask[16], raw_y[64], const_value_fp32[8], 1, 1, 0,
                              8, 0)
        tik_instance.vor(4, cmp_mask, cmp_mask, cmp_mask[16], 1, 1, 1, 1, 8, 8, 8)

        dst_cmp_mask = tik_instance.mov_tensor_to_cmpmask(cmp_mask)
        tik_instance.vsel(64, 0, y_lo_w[64], dst_cmp_mask, const_zero, y_lo_w[64],
                          1, 1, 0, 1, 8, 0, 8)
        tik_instance.vsel(64, 0, y_hi_w[64], dst_cmp_mask, const_zero, y_hi_w[64],
                          1, 1, 0, 1, 8, 0, 8)

        tik_instance.vmuls(64, y_lo_w, y_lo_w, 1.0 / self.sample_num, 2, 1, 1, 8, 8)
        tik_instance.vmuls(64, y_hi_w, y_hi_w, 1.0 / self.sample_num, 2, 1, 1, 8, 8)

        return x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi, x_ind, y_ind

    def _roi_align_calc_scale_batch(self, rois_data_ub, sample_num=None):
        """calc one block gradient
        """
        tik_instance = self.tik_instance
        roi_h_fp32 = tik_instance.Tensor(
            "float32", [BATCH_SIZE], name="roi_h_fp32", scope=tbe_platform.scope_ubuf)
        roi_w_fp32 = tik_instance.Tensor(
            "float32", [BATCH_SIZE], name="roi_w_fp32", scope=tbe_platform.scope_ubuf)

        rois_start_w = rois_data_ub[1, 0]
        rois_start_h = rois_data_ub[2, 0]
        rois_end_w = rois_data_ub[3, 0]
        rois_end_h = rois_data_ub[4, 0]
        tik_instance.vmuls(64, rois_start_w, rois_start_w, self.spatial_scale, 4, 1, 1, 8, 8)
        tik_instance.vadds(64, rois_end_w, rois_end_w, 1, 4, 1, 1, 8, 8)
        tik_instance.vmuls(64, rois_end_w, rois_end_w, self.spatial_scale, 4, 1, 1, 8, 8)

        tik_instance.vsub(64, roi_w_fp32, rois_end_w, rois_start_w,
                          BATCH_SIZE * 2 // 128, 1, 1, 1, 8, 8, 8)
        tik_instance.vsub(64, roi_h_fp32, rois_end_h, rois_start_h,
                          BATCH_SIZE * 2 // 128, 1, 1, 1, 8, 8, 8)

        const_zero = tik_instance.Tensor("float32", [16], name="const_zero", scope=tbe_platform.scope_ubuf)
        tik_instance.vector_dup(16, const_zero, 0, 1, 0, 0)

        # compare roi_width adn roi_height to 1
        tik_instance.vmax(64, roi_w_fp32, roi_w_fp32, const_zero,
                          BATCH_SIZE * 2 // 128, 1, 1, 0, 8, 8, 0)
        tik_instance.vmax(64, roi_h_fp32, roi_h_fp32, const_zero,
                          BATCH_SIZE * 2 // 128, 1, 1, 0, 8, 8, 0)

        # Declare roi_bin_size tik_instance.Tensor
        rois_bin_w = tik_instance.Tensor(
            "float32", [BATCH_SIZE], name="roi_bin_w", scope=tbe_platform.scope_ubuf)
        rois_bin_h = tik_instance.Tensor(
            "float32", [BATCH_SIZE], name="roi_bin_h", scope=tbe_platform.scope_ubuf)
        # bin size
        tik_instance.vmuls(64, rois_bin_w[:], roi_w_fp32[:], 1.0 / self.pooled_width,
                           BATCH_SIZE * 2 // 128, 1, 1, 8, 8)
        tik_instance.vmuls(64, rois_bin_h[:], roi_h_fp32[:], 1.0 / self.pooled_height,
                           BATCH_SIZE * 2 // 128, 1, 1, 8, 8)

        sample_num_w = tik_instance.Tensor(
            "int32", [BATCH_SIZE], name="sample_num_w", scope=tbe_platform.scope_ubuf)
        sample_num_h = tik_instance.Tensor(
            "int32", [BATCH_SIZE], name="sample_num_h", scope=tbe_platform.scope_ubuf)

        if sample_num is not None:
            if sample_num > 0:
                tik_instance.vector_dup(64, sample_num_w, self.sample_num, 2, 1, 8, 0)
                tik_instance.vector_dup(64, sample_num_h, self.sample_num, 2, 1, 8, 0)
            else:
                tik_instance.vconv(64, 'ceil', sample_num_w, rois_bin_w,
                                   BATCH_SIZE * 2 // 128, 1, 1, 8, 8)
                tik_instance.vconv(64, 'ceil', sample_num_h, rois_bin_h,
                                   BATCH_SIZE * 2 // 128, 1, 1, 8, 8)

        rois_start_w = tik_instance.Tensor(
            "float32", [BATCH_SIZE], name="roi_h_fp32", scope=tbe_platform.scope_ubuf)
        rois_start_h = tik_instance.Tensor(
            "float32", [BATCH_SIZE], name="roi_w_fp32", scope=tbe_platform.scope_ubuf)
        rois_index = tik_instance.Tensor(
            "int32", [BATCH_SIZE], name="roi_index", scope=tbe_platform.scope_ubuf)
        tik_instance.vadds(64, rois_start_w, rois_data_ub[1, 0], 0, 2, 1, 1, 8, 8)
        tik_instance.vadds(64, rois_start_h, rois_data_ub[2, 0], 0, 2, 1, 1, 8, 8)
        tik_instance.vconv(64, "floor", rois_index, rois_data_ub[0, 0], 2, 1, 1, 8, 8)

        return rois_bin_w, rois_bin_h, sample_num_w, sample_num_h, rois_start_w, rois_start_h, rois_index

    def _convert_rois_data_to5n(self, rois_data_index, rois_num):
        """calc one block gradient
        """
        tik_instance = self.tik_instance
        rois_data_ub = tik_instance.Tensor(
            "float32", (5, BATCH_SIZE), name="rois_data_ub", scope=tbe_platform.scope_ubuf)

        with tik_instance.if_scope(self.rois_row_length == 5):
            rois_data_tmp = tik_instance.Tensor(
                "float32", (BATCH_SIZE, 5), name="rois_data_tmp", scope=tbe_platform.scope_ubuf)
            tik_instance.tensor_mov(rois_data_tmp,
                                    self.rois[rois_data_index * 5],
                                    '', 1, ceil_value(rois_num * 5, "float32"), 0, 0)
            with tik_instance.for_range(0, rois_num) as i:
                rois_data_ub[0, i].set_as(rois_data_tmp[i, 0])
                rois_data_ub[1, i].set_as(rois_data_tmp[i, 1])
                rois_data_ub[2, i].set_as(rois_data_tmp[i, 2])
                rois_data_ub[3, i].set_as(rois_data_tmp[i, 3])
                rois_data_ub[4, i].set_as(rois_data_tmp[i, 4])
        with tik_instance.else_scope():
            rois_data_tmp = tik_instance.Tensor(
                "float32", (128, 8), name="rois_data_tmp", scope=tbe_platform.scope_ubuf)
            roi_pos = tik_instance.Tensor(
                "float16", [BATCH_SIZE, 8], name="roi_pos", scope=tbe_platform.scope_ubuf)
            roi_pos_new = tik_instance.Tensor(
                "float16", [5, BATCH_SIZE],
                name="roi_pos_new",
                scope=tbe_platform.scope_ubuf)

            tik_instance.tensor_mov(rois_data_tmp,
                                    self.rois[rois_data_index * 8],
                                    '', 1, (4 * rois_num * 8) // 32, 0, 0)

            tik_instance.vconv(64, "", roi_pos[0, 0], rois_data_tmp[0, 0],
                               (BATCH_SIZE * 8) // 64, 1, 1, 4, 8)

            tik_instance.vextract(roi_pos_new[0, 0], roi_pos, 8, 0)
            tik_instance.vextract(roi_pos_new[1, 0], roi_pos, 8, 1)
            tik_instance.vextract(roi_pos_new[2, 0], roi_pos, 8, 2)
            tik_instance.vextract(roi_pos_new[3, 0], roi_pos, 8, 3)
            tik_instance.vextract(roi_pos_new[4, 0], roi_pos, 8, 4)

            tik_instance.vconv(64, "", rois_data_ub[0, 0], roi_pos_new[0, 0],
                               (BATCH_SIZE * 10) // 128, 1, 1, 8, 4)

        return rois_data_ub

    def _compute_mode_1(self, core_bias, core_rois_n, sum_in_ub_flag):
        """
        compute mode 1
        """
        tik_instance = self.tik_instance

        grid_w_s = tik_instance.Scalar(dtype="float32")
        grid_h_s = tik_instance.Scalar(dtype="float32")
        rois_start_w_s = tik_instance.Scalar(dtype="float32")
        rois_start_h_s = tik_instance.Scalar(dtype="float32")

        const_value_0_127 = tik_instance.Tensor(
            "float32", (BATCH_SIZE,), name="const_value_0_127", scope=tbe_platform.scope_ubuf)
        with tik_instance.for_range(0, BATCH_SIZE) as i:
            const_value_0_127[i] = i

        rois_batch_num = (core_rois_n + 127) // BATCH_SIZE

        with tik_instance.for_range(0, rois_batch_num) as i:
            # move rois data from DDR to UB
            rois_data_ub = self._convert_rois_data_to5n(core_bias + BATCH_SIZE * i, core_rois_n)

            # calc spatial_scale
            rois_bin_w, rois_bin_h, sample_num_w, sample_num_h, rois_start_w, rois_start_h, rois_index = \
                self._roi_align_calc_scale_batch(rois_data_ub, sample_num=None)
            tik_instance.vmuls(64, rois_bin_w, rois_bin_w, 1 / float(self.sample_num), 2, 1, 1, 8, 8)
            tik_instance.vmuls(64, rois_bin_h, rois_bin_h, 1 / float(self.sample_num), 2, 1, 1, 8, 8)

            with tik_instance.for_range(0, core_rois_n) as j:
                image_index = tik_instance.Scalar(dtype="int32")
                image_index.set_as(rois_index[j])
                grid_w_s.set_as(rois_bin_w[j])
                grid_h_s.set_as(rois_bin_h[j])
                rois_start_w_s.set_as(rois_start_w[j])
                rois_start_h_s.set_as(rois_start_h[j])
                calc_w_num = (self.sample_num * self.pooled_width + 127) // 128
                calc_h_num = (self.sample_num * self.pooled_height + 127) // 128

                with tik_instance.for_range(0, calc_h_num) as k:
                    start_h_s = rois_start_h_s + grid_h_s * k * 128
                    line_num = self.sample_num * self.pooled_height
                    with tik_instance.for_range(0, calc_w_num) as w_index:
                        row_num = self.sample_num * self.pooled_width
                        start_w_s = rois_start_w_s + grid_w_s * w_index * 128
                        x_lo_w, x_hi_w, y_lo_w, \
                        y_hi_w, x_lo, x_hi, y_lo, y_hi, x_ind, y_ind = \
                            self._roi_align_calc_grid(k, w_index,
                                                      const_value_0_127, grid_w_s,
                                                      grid_h_s, start_w_s, start_h_s)

                        self._roi_align_calc_grad_block_align(line_num, row_num,
                                                              x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                                                              x_lo, x_hi, y_lo, y_hi,
                                                              core_bias + (i * 128) + j, x_ind, y_ind,
                                                              image_index, sum_in_ub_flag)

    def _compute_mode_2(self, core_bias):
        """
        compute mode 2
        """
        pass

    def roi_align_grad_compute_tiling(self):
        """
        define roi_align_grad tiling method
        """
        tik_instance = self.tik_instance
        with tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            # get tiling data
            tiling_ub = tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,), name="tiling_ub",
                                            scope=tik.scope_ubuf)
            tik_instance.data_move(tiling_ub, self.tiling_gm, 0,
                                   1, ceil_value(TILING_ARG_NUM, self.tiling_dtype),
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

                with tik_instance.if_scope(self.tiling_mode == TILING_MODE_1):
                    with tik_instance.new_stmt_scope():
                        self._compute_mode_1(core_bias, core_rois_n, False)
                with tik_instance.if_scope(self.tiling_mode == TILING_MODE_2):
                    with tik_instance.new_stmt_scope():
                        self._compute_mode_1(core_bias, core_rois_n, True)
                with tik_instance.if_scope(self.tiling_mode == TILING_MODE_3):
                    with tik_instance.new_stmt_scope():
                        self._compute_mode_2(core_bias)

    def roi_align_grad_compute(self):
        """calc one block gradient
        """
        tik_instance = self.tik_instance
        self.y_diff = tik_instance.Tensor("float32", (PARAMS_SIZE,), name="y_diff", scope=tbe_platform.scope_gm)
        self.rois = tik_instance.Tensor("float32", (PARAMS_SIZE,), name="rois_data", scope=tbe_platform.scope_gm)
        self.x_diff = tik_instance.Tensor("float32", (PARAMS_SIZE,), name="x_diff",
                                          scope=tbe_platform.scope_gm, is_atomic_add=True)
        self.tiling_gm = tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,),
                                             name="tiling_gm", scope=tik.scope_gm)

        tik_instance.set_atomic_add(1)
        self.roi_align_grad_compute_tiling()
        tik_instance.set_atomic_add(0)

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }
        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num, "ub_size": self.ub_size})

        inputs = [self.y_diff, self.rois]
        if self.exist_rois_n:
            self.rois_n_gm = tik_instance.Tensor(
                "int32", (PARAMS_SIZE,), name="rois_n_gm", scope=tbe_platform.scope_gm)
            inputs.append(self.rois_n_gm)

        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=inputs,
                              outputs=[self.x_diff],
                              flowtable=(self.tiling_gm,),
                              enable_l2=True, config=opt_config)


@register_operator("ROIAlignGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def roi_align_grad(y_diff,
                   rois,
                   rois_n,
                   x_diff,
                   xdiff_shape,
                   pooled_width,
                   pooled_height,
                   spatial_scale,
                   sample_num,
                   roi_end_mode=1,
                   kernel_name="roi_align_grad"):
    """
    calculating roi_align_grad,
    the type of input_data is "float32"

    Parameters
    ----------
    y_diff: dict
        dict with keys(shape and dtype) of y_diff
    rois: dict
        dict with keys(shape and dtype) of rois
    rois_n: dict
        dict with keys(shape and dtype) of rois_n
    x_diff: dict
        dict with keys(shape and dtype) of x_diff
    xdiff_shape: list
        list xdiff_shape
    pooled_width: int
        pooled_width
    pooled_height: int
        pooled_height
    spatial_scale: float
        spatial_scale
    sample_num: int
        sample_num
    roi_end_mode: int
        roi_end_mode
    kernel_name: str
        kernel name

    Returns
    -------
    tik_instance: tik_instance
    """
    input_list = [y_diff, rois]
    for input_data in input_list:
        input_dtype = input_data.get("dtype").lower()
        check_list = ("float32",)
        para_check.check_dtype(input_dtype, check_list, param_name="y_diff")

    roi_align_grad_obj = RoiAlignGrad()
    roi_align_grad_obj.pooled_height = pooled_height
    roi_align_grad_obj.pooled_width = pooled_width
    roi_align_grad_obj.sample_num = sample_num
    roi_align_grad_obj.spatial_scale = spatial_scale
    roi_align_grad_obj.kernel_name = kernel_name

    if rois_n:
        roi_align_grad_obj.exist_rois_n = True

    return roi_align_grad_obj.roi_align_grad_compute()
