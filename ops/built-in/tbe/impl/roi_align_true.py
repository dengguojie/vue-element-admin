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
roi_align_true
"""

import te.platform as tbe_platform
from te import tik


class Constant:
    """
    The class for constant
    """
    UB_30K_SIZE = 150 * 1024


def _tf_n52n8(tik_instance, rois_ub, rois_n5, block_num):
    """
    transform ROIS form N5 to N8
    """
    with tik_instance.for_range(0, block_num) as rois_num:
        rois_ub[rois_num, 0].set_as(rois_n5[rois_num, 0])
        rois_ub[rois_num, 1].set_as(rois_n5[rois_num, 1])
        rois_ub[rois_num, 2].set_as(rois_n5[rois_num, 2])
        rois_ub[rois_num, 3].set_as(rois_n5[rois_num, 3])
        rois_ub[rois_num, 4].set_as(rois_n5[rois_num, 4])


def _tf_n42n8(tik_instance, rois_ub, rois_n4, block_num):
    """
    transform ROIS form N4 to N8
    """
    with tik_instance.for_range(0, block_num) as rois_num:
        rois_ub[rois_num, 1].set_as(rois_n4[rois_num, 0])
        rois_ub[rois_num, 2].set_as(rois_n4[rois_num, 1])
        rois_ub[rois_num, 3].set_as(rois_n4[rois_num, 2])
        rois_ub[rois_num, 4].set_as(rois_n4[rois_num, 3])


def _adds_muls(tik_instance, dtype_num, scale, proposals_ub_x, proposals_ub_y):
    tik_instance.vec_muls(64 * dtype_num, proposals_ub_x[0, 0], proposals_ub_x[0, 0], scale,
                          128 * 2 // 128 // dtype_num, 8, 8)
    tik_instance.vec_adds(64 * dtype_num, proposals_ub_x[0, 0], proposals_ub_x[0, 0], -0.5,
                          128 * 2 // 128 // dtype_num, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, proposals_ub_y[0, 0], proposals_ub_y[0, 0], scale,
                          128 * 2 // 128 // dtype_num, 8, 8)
    tik_instance.vec_adds(64 * dtype_num, proposals_ub_y[0, 0], proposals_ub_y[0, 0], -0.5,
                          128 * 2 // 128 // dtype_num, 8, 8)


def _get_roi_bin(tik_instance, dtype, sample_num, dtype_num, grid_w_int32, grid_h_int32,
                 grid_w_fp32, grid_h_fp32, grid_w_fp16, grid_h_fp16, roi_bin_w_fp32_value, roi_bin_h_fp32_value):
    suppot_vconv = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32c")
    if sample_num <= 0:
        if suppot_vconv is False and dtype == "float32":
            roi_bin_w_fp16_value = tik_instance.Tensor("float16", [128], name="roi_bin_w_fp16_value",
                                                       scope=tbe_platform.scope_ubuf)
            roi_bin_h_fp16_value = tik_instance.Tensor("float16", [128], name="roi_bin_h_fp16_value",
                                                       scope=tbe_platform.scope_ubuf)
            tik_instance.vec_conv(64 * dtype_num, "", roi_bin_w_fp16_value, roi_bin_w_fp32_value, 2, 4, 8)
            tik_instance.vec_conv(64 * dtype_num, "", roi_bin_h_fp16_value, roi_bin_h_fp32_value, 2, 4, 8)

            tik_instance.vec_conv(64, "ceiling", grid_w_int32, roi_bin_w_fp16_value, 2, 8, 4)
            tik_instance.vec_conv(64, "ceiling", grid_h_int32, roi_bin_h_fp16_value, 2, 8, 4)
        else:
            tik_instance.vec_conv(64, "ceiling", grid_w_int32, roi_bin_w_fp32_value, 2, 8, 8 // dtype_num)
            tik_instance.vec_conv(64, "ceiling", grid_h_int32, roi_bin_h_fp32_value, 2, 8, 8 // dtype_num)

        if suppot_vconv is False and dtype == "float32":
            tik_instance.vec_conv(64 * dtype_num, "", grid_w_fp16, grid_w_int32, 2 // dtype_num, 4, 8, 1.0)
            tik_instance.vec_conv(64 * dtype_num, "", grid_h_fp16, grid_h_int32, 2 // dtype_num, 4, 8, 1.0)
            tik_instance.vec_conv(64 * dtype_num, "", grid_w_fp32, grid_w_fp16, 2 // dtype_num, 8, 4)
            tik_instance.vec_conv(64 * dtype_num, "", grid_h_fp32, grid_h_fp16, 2 // dtype_num, 8, 4)
        else:
            if dtype == "float32":
                tik_instance.vec_conv(64, "", grid_w_fp32, grid_w_int32, 2, 8 // dtype_num, 8)
                tik_instance.vec_conv(64, "", grid_h_fp32, grid_h_int32, 2, 8 // dtype_num, 8)
            else:

                tik_instance.vec_conv(64, "", grid_w_fp32, grid_w_int32, 2, 8 // dtype_num, 8, 1.0)
                tik_instance.vec_conv(64, "", grid_h_fp32, grid_h_int32, 2, 8 // dtype_num, 8, 1.0)

    else:
        tik_instance.vec_dup(64, grid_w_int32, sample_num, 2, 8)
        tik_instance.vec_dup(64, grid_h_int32, sample_num, 2, 8)
        tik_instance.vec_dup(64 * dtype_num, grid_w_fp32, sample_num, 2 // dtype_num, 8)
        tik_instance.vec_dup(64 * dtype_num, grid_h_fp32, sample_num, 2 // dtype_num, 8)


def _get_roi_bin_scale(tik_instance, dtype_num, sample_num, dtype, scale, proposals_ub_x0, proposals_ub_y0,
                       proposals_ub_x1, proposals_ub_y1, roi_w_fp32, roi_h_fp32, proposal_num_128, pool_h, pool_w,
                       roi_end_mode):
    _adds_muls(tik_instance, dtype_num, scale, proposals_ub_x0, proposals_ub_y0)
    _adds_muls(tik_instance, dtype_num, scale, proposals_ub_x1, proposals_ub_y1)

    roi_h_1to8 = tik_instance.Tensor(dtype, [128, 1], name="roi_h_1to8", scope=tbe_platform.scope_ubuf)
    roi_w_1to8 = tik_instance.Tensor(dtype, [128, 1], name="roi_w_1to8", scope=tbe_platform.scope_ubuf)

    tik_instance.vec_sub(64 * dtype_num, roi_h_1to8, proposals_ub_y1[0, 0], proposals_ub_y0[0, 0],
                         128 * 2 // 128 // dtype_num, 8, 8, 8)
    tik_instance.vec_sub(64 * dtype_num, roi_w_1to8, proposals_ub_x1[0, 0], proposals_ub_x0[0, 0],
                         128 * 2 // 128 // dtype_num, 8, 8, 8)

    const_mode = tik_instance.Tensor(dtype, [128, 1], name="const_mode", scope=tbe_platform.scope_ubuf)
    tik_instance.vec_dup(64 * dtype_num, const_mode, 1, 2 // dtype_num, 8)

    # compare roi_width adn roi_height to 1-mode (1 or 0)
    if roi_end_mode != 3:
        tik_instance.vec_max(64 * dtype_num, roi_w_1to8, roi_w_1to8, const_mode, 128 * 2 // 128 // dtype_num, 8, 8, 0)
        tik_instance.vec_max(64 * dtype_num, roi_h_1to8, roi_h_1to8, const_mode, 128 * 2 // 128 // dtype_num, 8, 8, 0)

    with tik_instance.for_range(0, roi_w_fp32.shape[0]) as i:
        roi_w_fp32[i].set_as(roi_w_1to8[i, 0])
        roi_h_fp32[i].set_as(roi_h_1to8[i, 0])

    # Declare roi_bin_size tik_instance.Tensor
    roi_bin_h_fp32_value = tik_instance.Tensor(dtype, [128], name="roi_bin_h_fp32_value", scope=tbe_platform.scope_ubuf)
    roi_bin_w_fp32_value = tik_instance.Tensor(dtype, [128], name="roi_bin_w_fp32_value", scope=tbe_platform.scope_ubuf)

    grid_w_fp32 = tik_instance.Tensor(dtype, [proposal_num_128], name="grid_w_fp32", scope=tbe_platform.scope_ubuf)
    grid_h_fp32 = tik_instance.Tensor(dtype, [proposal_num_128], name="grid_h_fp32", scope=tbe_platform.scope_ubuf)

    grid_w_fp16 = tik_instance.Tensor("float16", [proposal_num_128], name="grid_w_fp16", scope=tbe_platform.scope_ubuf)
    grid_h_fp16 = tik_instance.Tensor("float16", [proposal_num_128], name="grid_h_fp16", scope=tbe_platform.scope_ubuf)

    grid_w_int32 = tik_instance.Tensor("int32", [proposal_num_128], name="grid_w_int32", scope=tbe_platform.scope_ubuf)
    grid_h_int32 = tik_instance.Tensor("int32", [proposal_num_128], name="grid_h_int32", scope=tbe_platform.scope_ubuf)
    # bin size
    tik_instance.vec_muls(64 * dtype_num, roi_bin_h_fp32_value[:], roi_h_fp32[:], 1.0 / pool_h,
                          proposal_num_128 * 2 // dtype_num // 128, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, roi_bin_w_fp32_value[:], roi_w_fp32[:], 1.0 / pool_w,
                          proposal_num_128 * 2 // dtype_num // 128, 8, 8)

    _get_roi_bin(tik_instance, dtype, sample_num, dtype_num, grid_w_int32, grid_h_int32,
                 grid_w_fp32, grid_h_fp32, grid_w_fp16, grid_h_fp16, roi_bin_w_fp32_value, roi_bin_h_fp32_value)
    return tik_instance, roi_bin_h_fp32_value, roi_bin_w_fp32_value, proposals_ub_x0, proposals_ub_y0, grid_w_int32, \
           grid_h_int32, grid_w_fp32, grid_h_fp32


def _get_roi_align_perf_scale_for_zero_v200(tik_instance, roi_fp32_fm_index, proposals_ub_x0,
                                            proposals_ub_y0, proposals_ub_x1,
                                            proposals_ub_y1, scale, pool_h, pool_w,
                                            sample_num, dtype, roi_end_mode):
    """
    get satart point, bin_size and sample number
    """
    proposal_num_128 = 128
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2

    roi_h_fp32 = tik_instance.Tensor(
        dtype, [128], name="roi_h_fp32", scope=tbe_platform.scope_ubuf)
    roi_w_fp32 = tik_instance.Tensor(
        dtype, [128], name="roi_w_fp32", scope=tbe_platform.scope_ubuf)

    roi_int32_fm_index = tik_instance.Tensor(
        "int32", [128], name="roi_int32_fm_index", scope=tbe_platform.scope_ubuf)

    tik_instance.vec_conv(64, "ceil", roi_int32_fm_index[0],
                          roi_fp32_fm_index[0], 2, 8, 8 // dtype_num)

    tik_instance, roi_bin_h_fp32_value, roi_bin_w_fp32_value, proposals_ub_x0, proposals_ub_y0, grid_w_int32, \
    grid_h_int32, grid_w_fp32, grid_h_fp32 = _get_roi_bin_scale(tik_instance, dtype_num, sample_num, dtype, scale,
                                                                proposals_ub_x0, proposals_ub_y0, proposals_ub_x1,
                                                                proposals_ub_y1, roi_w_fp32, roi_h_fp32,
                                                                proposal_num_128, pool_h, pool_w, roi_end_mode)

    return tik_instance, roi_bin_h_fp32_value, roi_bin_w_fp32_value, proposals_ub_x0, proposals_ub_y0, \
           grid_w_int32, grid_h_int32, grid_w_fp32, grid_h_fp32, roi_int32_fm_index


# 'pylint: disable=too-many-statements,too-many-locals,too-many-branches
# 'pylint: disable=no-member
def _get_roi_align_perf_scale_for_zero(tik_instance, proposal, proposals_ub_x0,
                                       proposals_ub_y0, proposals_ub_x1,
                                       proposals_ub_y1, scale, pool_h, pool_w,
                                       sample_num, dtype, roi_end_mode):
    """
    get satart point, bin_size and sample number
    """
    proposal_num_128 = 128
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2

    roi_h_fp32 = tik_instance.Tensor(dtype, [128], name="roi_h_fp32", scope=tbe_platform.scope_ubuf)
    roi_w_fp32 = tik_instance.Tensor(dtype, [128], name="roi_w_fp32", scope=tbe_platform.scope_ubuf)

    roi_fp16_pos = tik_instance.Tensor("float16", proposal.shape, name="roi_fp16_pos", scope=tbe_platform.scope_ubuf)
    roi_fp16_fm_index = tik_instance.Tensor("float16", [128], name="roi_fp16_fm_index", scope=tbe_platform.scope_ubuf)
    roi_fp32_fm_index = tik_instance.Tensor(dtype, [128], name="roi_fp32_fm_index", scope=tbe_platform.scope_ubuf)
    roi_int32_fm_index = tik_instance.Tensor("int32", [128], name="roi_int32_fm_index", scope=tbe_platform.scope_ubuf)
    support_vextract = tbe_platform.api_check_support("tik.vextract", "float32")
    if support_vextract is False and dtype == "float32":
        tik_instance.vec_conv(64, "", roi_fp16_pos[0, 0], proposal[0, 0], (128 * 8) // 64, 4, 8)

        tik_instance.vextract(roi_fp16_fm_index[0], roi_fp16_pos, 8, 0)
        tik_instance.vec_conv(64, "ceil", roi_int32_fm_index[0], roi_fp16_fm_index[0], 2, 8, 4)
    else:
        tik_instance.vextract(roi_fp32_fm_index[0], proposal[0, 0], 8, 0)
        tik_instance.vec_conv(64, "ceil", roi_int32_fm_index[0], roi_fp32_fm_index[0], 2, 8, 8 // dtype_num)

    tik_instance, roi_bin_h_fp32_value, roi_bin_w_fp32_value, proposals_ub_x0, proposals_ub_y0, grid_w_int32, \
    grid_h_int32, grid_w_fp32, grid_h_fp32 = _get_roi_bin_scale(tik_instance, dtype_num, sample_num, dtype, scale,
                                                                proposals_ub_x0, proposals_ub_y0, proposals_ub_x1,
                                                                proposals_ub_y1, roi_w_fp32, roi_h_fp32,
                                                                proposal_num_128, pool_h, pool_w, roi_end_mode)

    return tik_instance, roi_bin_h_fp32_value, roi_bin_w_fp32_value, proposals_ub_x0, proposals_ub_y0, \
           grid_w_int32, grid_h_int32, grid_w_fp32, grid_h_fp32, roi_int32_fm_index


def _newton(tik_instance, mask, dst_ub, src1, src2, repeat, dtype):
    """
    for div and usr newton when in mini
    """
    rec_2 = tik_instance.Tensor(dtype, src2.shape, name="rec_1",
                                scope=tbe_platform.scope_ubuf)
    _reciprocal(tik_instance, mask, rec_2, src2, repeat, dtype)
    tik_instance.vec_mul(mask, dst_ub, rec_2, src1, repeat, 8, 8, 8)


def _reciprocal(tik_instance, mask, dest_ub, src1, repeat, dtype):
    """
    get reciprocal when in mini
    """
    rec_1 = tik_instance.Tensor(dtype, src1.shape, name="rec_1",
                                scope=tbe_platform.scope_ubuf)
    rec_2 = tik_instance.Tensor(dtype, src1.shape, name="rec_2",
                                scope=tbe_platform.scope_ubuf)
    tik_instance.vec_rec(mask, rec_1, src1, repeat, 8, 8)
    tik_instance.vec_mul(mask, rec_2, rec_1, src1, repeat, 8, 8, 8)
    tik_instance.vec_muls(mask, rec_2, rec_2, -1, repeat, 8, 8)
    tik_instance.vec_adds(mask, rec_2, rec_2, 2, repeat, 8, 8)
    tik_instance.vec_mul(mask, rec_2, rec_2, rec_1, repeat, 8, 8, 8)
    tik_instance.vec_mul(mask, rec_1, rec_2, src1, repeat, 8, 8, 8)
    tik_instance.vec_muls(mask, rec_1, rec_1, -1, repeat, 8, 8)
    tik_instance.vec_adds(mask, rec_1, rec_1, 2, repeat, 8, 8)
    tik_instance.vec_mul(mask, dest_ub, rec_1, rec_2, repeat, 8, 8, 8)


def _get_input(tik_instance, dtype, grid_h, grid_w, proposals_ub_y0,
               proposals_ub_x0, grid_h_int32, grid_w_int32,
               grid_h_fp32, grid_w_fp32, curr_roi):
    """
    :param tik_instance:
    :param dtype:
    :param grid_h:
    :param grid_w:
    :param proposals_ub_y0:
    :param proposals_ub_x0:
    :param grid_h_int32:
    :param grid_w_int32:
    :param grid_h_fp32:
    :param grid_w_fp32:
    :param curr_roi:
    :return:list
    """
    grid_h_roi = tik_instance.Scalar(dtype=dtype)
    grid_h_roi.set_as(grid_h[curr_roi])

    grid_w_roi = tik_instance.Scalar(dtype=dtype)
    grid_w_roi.set_as(grid_w[curr_roi])

    rois_start_h = tik_instance.Scalar(dtype=dtype)
    rois_start_h.set_as(proposals_ub_y0[curr_roi, 0])
    rois_start_w = tik_instance.Scalar(dtype=dtype)
    rois_start_w.set_as(proposals_ub_x0[curr_roi, 0])

    grid_h_num = tik_instance.Scalar(dtype="int32")
    grid_h_num.set_as(grid_h_int32[curr_roi])
    grid_w_num = tik_instance.Scalar(dtype="int32")
    grid_w_num.set_as(grid_w_int32[curr_roi])

    grid_h_num_f = tik_instance.Scalar(dtype=dtype)
    grid_h_num_f.set_as(grid_h_fp32[curr_roi])
    grid_w_num_f = tik_instance.Scalar(dtype=dtype)
    grid_w_num_f.set_as(grid_w_fp32[curr_roi])

    return grid_w_roi, grid_h_roi, grid_w_num, grid_h_num, rois_start_w, rois_start_h, grid_h_num_f, grid_w_num_f


def _get_grid_weight(tik_instance, grid_w, grid_h, rois_start_w, rois_start_h, height, width, dtype):
    """
    get grid size and coordinate in feature
    """
    x_lo_w = tik_instance.Tensor(dtype, [128], name="x_lo_w", scope=tbe_platform.scope_ubuf)
    x_hi_w = tik_instance.Tensor(dtype, [128], name="x_hi_w", scope=tbe_platform.scope_ubuf)
    y_lo_w = tik_instance.Tensor(dtype, [128], name="y_lo_w", scope=tbe_platform.scope_ubuf)
    y_hi_w = tik_instance.Tensor(dtype, [128], name="_lo_w", scope=tbe_platform.scope_ubuf)
    x_lo = tik_instance.Tensor("int32", [128], name="x_lo", scope=tbe_platform.scope_ubuf)
    x_hi = tik_instance.Tensor("int32", [128], name="x_hi", scope=tbe_platform.scope_ubuf)
    y_lo = tik_instance.Tensor("int32", [128], name="y_lo", scope=tbe_platform.scope_ubuf)
    y_hi = tik_instance.Tensor("int32", [128], name="y_hi", scope=tbe_platform.scope_ubuf)

    raw_x = tik_instance.Tensor(dtype, [128], name="raw_x", scope=tbe_platform.scope_ubuf)
    raw_y = tik_instance.Tensor(dtype, [128], name="raw_y", scope=tbe_platform.scope_ubuf)
    x_output = tik_instance.Tensor(dtype, [128], name="x_output", scope=tbe_platform.scope_ubuf)
    y_output = tik_instance.Tensor(dtype, [128], name="y_output", scope=tbe_platform.scope_ubuf)
    tmp_fp16 = tik_instance.Tensor("float16", [128], name="tmp_fp16", scope=tbe_platform.scope_ubuf)

    const_value_0_127 = tik_instance.Tensor(dtype, (128,), name="const_value_0_127", scope=tbe_platform.scope_ubuf)
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2
    vconv_f322s32f_suppot = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32f")
    if vconv_f322s32f_suppot is False or dtype == "float16":
        const_value_0_127_int = tik_instance.Tensor("int32", (128,), name="const_value_0_127_int",
                                                    scope=tbe_platform.scope_ubuf)
        with tik_instance.for_range(0, 128) as i:
            const_value_0_127_int[i].set_as(i)
        if dtype == "float32":
            const_value_0_127_float = tik_instance.Tensor("float16", (128,), name="const_value_0_127_float",
                                                          scope=tbe_platform.scope_ubuf)
            tik_instance.vec_conv(64, "", const_value_0_127_float, const_value_0_127_int, 2, 4, 8, 1.0)
            tik_instance.vec_conv(64, "", const_value_0_127, const_value_0_127_float, 2, 8, 4)
        else:
            tik_instance.vec_conv(64, "", const_value_0_127, const_value_0_127_int, 2, 4, 8, 1.0)
    else:
        with tik_instance.for_range(0, 128) as i:
            const_value_0_127[i] = i

    grid_w_vector = tik_instance.Tensor(dtype, [128], name="grid_w_vector", scope=tbe_platform.scope_ubuf)
    grid_h_vector = tik_instance.Tensor(dtype, [128], name="grid_h_vector", scope=tbe_platform.scope_ubuf)

    tik_instance.vec_muls(64 * dtype_num, grid_w_vector, const_value_0_127, grid_w, 2 // dtype_num, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, grid_h_vector, const_value_0_127, grid_h, 2 // dtype_num, 8, 8)
    # fp16 scalar floating-point operation is not allowed in aicore fucntion
    # fl32 scalar floating-point operation is not allowed on mini
    if vconv_f322s32f_suppot is False or dtype == "float16":
        point_05 = tik_instance.Scalar(dtype, init_value=0.5)
        point_05_tensor = tik_instance.Tensor(dtype, [1], name="point_05_tensor", scope=tbe_platform.scope_ubuf)
        tik_instance.vec_dup(1, point_05_tensor, 0.5, 1, 0)
        tik_instance.vec_muls(1, point_05_tensor, point_05_tensor, grid_w, 1, 8, 8)
        tik_instance.vec_adds(1, point_05_tensor, point_05_tensor, rois_start_w, 1, 8, 8)
        point_05.set_as(point_05_tensor[0])
        tik_instance.vec_adds(64 * dtype_num, raw_x, grid_w_vector, point_05, 2 // dtype_num, 8, 8)
        tik_instance.vec_dup(1, point_05_tensor, 0.5, 1, 0)

        tik_instance.vec_muls(1, point_05_tensor, point_05_tensor, grid_h, 1, 8, 8)
        tik_instance.vec_adds(1, point_05_tensor, point_05_tensor, rois_start_h, 1, 8, 8)
        point_05.set_as(point_05_tensor[0])
        tik_instance.vec_adds(64 * dtype_num, raw_y, grid_h_vector, point_05, 2 // dtype_num, 8, 8)
    # fp32 besides mini
    else:
        half_grid = 0.5 * grid_w + rois_start_w
        tik_instance.vec_adds(64 * dtype_num, raw_x, grid_w_vector, half_grid, 2 // dtype_num, 8, 8)
        half_grid = 0.5 * grid_h + rois_start_h
        tik_instance.vec_adds(64 * dtype_num, raw_y, grid_h_vector, half_grid, 2 // dtype_num, 8, 8)

    const_zero = tik_instance.Tensor(dtype, [64 * dtype_num], name="const_zero", scope=tbe_platform.scope_ubuf)

    tik_instance.vec_dup(64 * dtype_num, const_zero, 0, 1, 0)

    tik_instance.vec_max(64 * dtype_num, x_output, raw_x, const_zero, 2 // dtype_num, 8, 8, 0)
    tik_instance.vec_max(64 * dtype_num, y_output, raw_y, const_zero, 2 // dtype_num, 8, 8, 0)

    if vconv_f322s32f_suppot is False and dtype == "float32":
        tik_instance.vec_conv(64, "", tmp_fp16, x_output, 2, 4, 8)
        tik_instance.vec_conv(64, "floor", x_lo, tmp_fp16, 2, 8, 4)
        tik_instance.vec_conv(64, "", tmp_fp16, y_output, 2, 4, 8)
        tik_instance.vec_conv(64, "floor", y_lo, tmp_fp16, 2, 8, 4)
    else:
        tik_instance.vec_conv(64, "floor", x_lo, x_output, 2, 8, 8 // dtype_num)
        tik_instance.vec_conv(64, "floor", y_lo, y_output, 2, 8, 8 // dtype_num)

    const_one = tik_instance.Tensor("int32", [64], name="const_one", scope=tbe_platform.scope_ubuf)
    tik_instance.vec_dup(64, const_one, 1, 1, 0)
    tik_instance.vec_add(64, x_hi, x_lo, const_one, 2, 8, 8, 0)
    tik_instance.vec_add(64, y_hi, y_lo, const_one, 2, 8, 8, 0)

    const_value_fp32 = tik_instance.Tensor(dtype, [64 * dtype_num], name="const_value_fp32",
                                           scope=tbe_platform.scope_ubuf)
    const_value_int32 = tik_instance.Tensor("int32", [64], name="const_value_int32", scope=tbe_platform.scope_ubuf)

    tik_instance.vec_dup(64 * dtype_num, const_value_fp32, width - 1, 1, 0)
    tik_instance.vec_dup(64, const_value_int32, width - 1, 1, 0)
    tik_instance.vec_min(64, x_lo, x_lo, const_value_int32, 2, 8, 8, 0)
    tik_instance.vec_min(64, x_hi, x_hi, const_value_int32, 2, 8, 8, 0)
    tik_instance.vec_min(64 * dtype_num, x_output, x_output, const_value_fp32, 2 // dtype_num, 8, 8, 0)

    tik_instance.vec_dup(64, const_value_int32, height - 1, 1, 0)
    tik_instance.vec_dup(64 * dtype_num, const_value_fp32, height - 1, 1, 0)
    tik_instance.vec_min(64, y_lo, y_lo, const_value_int32, 2, 8, 8, 0)
    tik_instance.vec_min(64, y_hi, y_hi, const_value_int32, 2, 8, 8, 0)
    tik_instance.vec_min(64 * dtype_num, y_output, y_output, const_value_fp32, 2 // dtype_num, 8, 8, 0)

    tmp_fp32 = tik_instance.Tensor(dtype, [128], name="tmp_fp32", scope=tbe_platform.scope_ubuf)

    if vconv_f322s32f_suppot is False and dtype == "float32":
        tik_instance.vec_conv(64, "", tmp_fp16, x_lo, 2, 4, 8, 1.0)
        tik_instance.vec_conv(64, "", tmp_fp32, tmp_fp16, 2, 8, 4)
    else:
        # float16 add 1.0 float32 can not add 1.0
        if dtype == "float32":
            tik_instance.vec_conv(64, "", tmp_fp32, x_lo, 2, 8, 8)
        else:
            tik_instance.vec_conv(64, "", tmp_fp32, x_lo, 2, 8 // dtype_num, 8, 1.0)

    tik_instance.vec_sub(64 * dtype_num, x_lo_w, x_output, tmp_fp32, 2 // dtype_num, 8, 8, 8)

    if vconv_f322s32f_suppot is False and dtype == "float32":
        tik_instance.vec_conv(64 * dtype_num, "", tmp_fp16, y_lo, 2, 4, 8, 1.0)
        tik_instance.vec_conv(64 * dtype_num, "", tmp_fp32, tmp_fp16, 2, 8, 4)
    else:
        if dtype == "float32":
            tik_instance.vec_conv(64, "", tmp_fp32, y_lo, 2, 8, 8)
        else:
            tik_instance.vec_conv(64, "", tmp_fp32, y_lo, 2, 8 // dtype_num, 8, 1.0)

    tik_instance.vec_sub(64 * dtype_num, y_lo_w, y_output, tmp_fp32, 2 // dtype_num, 8, 8, 8)

    tik_instance.vec_dup(64 * dtype_num, const_value_fp32, 1.0, 1, 0)
    tik_instance.vec_sub(64 * dtype_num, x_hi_w, const_value_fp32, x_lo_w, 2 // dtype_num, 8, 0, 8)
    tik_instance.vec_sub(64 * dtype_num, y_hi_w, const_value_fp32, y_lo_w, 2 // dtype_num, 8, 0, 8)

    return x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi, raw_x, raw_y


def _init_l1(tik_instance, dtype, w_number, fm_c1, fm_w, fm_c0):
    """
    initialize L1 cache
    """
    cache_l1 = tik_instance.Tensor(
        dtype, [w_number, fm_c1, fm_w, fm_c0],
        name="cache_l1",
        scope=tbe_platform.scope_cbuf)
    cache_table = tik_instance.Tensor(
        "int32", [w_number, 2], name="cache_table", scope=tbe_platform.scope_ubuf)

    return cache_l1, cache_table


def _load_a_w_to_l1(tik_instance, cache_table, cache_l1, feature_map,
                    index, y_low, n_bust, point):
    """
    load a width of feature map to l1
    """
    stride = (feature_map.shape[2] * feature_map.shape[3] - feature_map.shape[3]) * n_bust
    c_iter_c1 = feature_map.shape[1]
    fm_h = feature_map.shape[2]
    fm_w = feature_map.shape[3]

    if stride > 65535:
        with tik_instance.for_range(0, c_iter_c1) as c_iter_i:
            tik_instance.data_move(
                cache_l1[point, c_iter_i, 0, 0],
                feature_map[index, c_iter_i, y_low, 0, 0],
                0, 1, fm_w * n_bust, 1, 1)
    else:
        tik_instance.data_move(cache_l1[point, 0, 0, 0],
                               feature_map[index, 0, y_low, 0, 0],
                               0, c_iter_c1, fm_w * n_bust,
                               (fm_h * fm_w - fm_w) * n_bust,
                               0)
    # ylow:
    cache_table[point, 0].set_as(index)
    cache_table[point, 1].set_as(y_low)


def _load_feature_map_to_ub(tik_instance, fm_grid, feature_shape,
                            c_block, c_valid,
                            feature_map, index, current_cb,
                            y_low, x_low, x_high,
                            y_high, n_bust, cache_flag):
    """
    load feature map from ddr to ub
    """
    stride = (feature_shape[2] * feature_shape[3] - 1) * n_bust
    stride_s = tik_instance.Scalar(dtype="int32", init_value=stride)

    with tik_instance.if_scope(cache_flag == 1):
        index.set_as(0)

    with tik_instance.if_scope(stride <= 65535):
        tik_instance.data_move(
            fm_grid[0, 0, 0, 0],
            feature_map[index, current_cb * c_block, y_low, x_low, 0], 0,
            c_valid, n_bust, stride_s, (4 - 1) * n_bust)
        tik_instance.data_move(
            fm_grid[0, 0, 1, 0],
            feature_map[index, current_cb * c_block, y_low, x_high, 0], 0,
            c_valid, n_bust, stride_s, (4 - 1) * n_bust)
        tik_instance.data_move(
            fm_grid[0, 1, 1, 0],
            feature_map[index, current_cb * c_block, y_high, x_high, 0], 0,
            c_valid, n_bust, stride_s, (4 - 1) * n_bust)
        tik_instance.data_move(
            fm_grid[0, 1, 0, 0],
            feature_map[index, current_cb * c_block, y_high, x_low, 0], 0,
            c_valid, n_bust, stride_s, (4 - 1) * n_bust)

    with tik_instance.else_scope():
        with tik_instance.for_range(0, c_valid) as c_iter_i:
            tik_instance.data_move(fm_grid[c_iter_i, 0, 0, 0], feature_map[
                index, current_cb * c_block + c_iter_i, y_low, x_low, 0], 0,
                                   1, n_bust, 1, 1)
            tik_instance.data_move(fm_grid[c_iter_i, 0, 1, 0], feature_map[
                index, current_cb * c_block + c_iter_i, y_low, x_high, 0], 0,
                                   1, n_bust, 1, 1)
            tik_instance.data_move(fm_grid[c_iter_i, 1, 1, 0], feature_map[
                index, current_cb * c_block + c_iter_i, y_high, x_high, 0], 0,
                                   1, n_bust, 1, 1)
            tik_instance.data_move(fm_grid[c_iter_i, 1, 0, 0], feature_map[
                index, current_cb * c_block + c_iter_i, y_high, x_low, 0], 0,
                                   1, n_bust, 1, 1)


def _load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_l1, point,
                        current_cb, c_block, x_low, x_high, c_valid, n_bust):
    """
    load feature map from l1 cache
    """

    tik_instance.data_move(fm_grid[0, point, 0, 0],
                           cache_l1[point, current_cb * c_block, x_low, 0], 0,
                           c_valid, n_bust, (feature_map.shape[3] - 1) * n_bust,
                           (4 - 1) * n_bust)
    tik_instance.data_move(fm_grid[0, point, 1, 0],
                           cache_l1[point, current_cb * c_block, x_high, 0], 0,
                           c_valid, n_bust, (feature_map.shape[3] - 1) * n_bust,
                           (4 - 1) * n_bust)


def _compute_w1234(tik_instance, h_y, l_y, h_x, l_x, w1_lt, w2_rt, w3_lb, w4_rb,
                   fm_grid, c_valid, n_bust):
    """
    get weight 1, 2, 3 and 4
    """
    if n_bust == 2:
        dtype = "float32"
    else:
        dtype = "float16"

    vconvf_suppot = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32f")

    if dtype == "float32" and vconvf_suppot is True:
        hy_tensor = tik_instance.Scalar(dtype=dtype, init_value=h_y)
        ly_tensor = tik_instance.Scalar(dtype=dtype, init_value=l_y)
        hx_tensor = tik_instance.Scalar(dtype=dtype, init_value=h_x)
        lx_tensor = tik_instance.Scalar(dtype=dtype, init_value=l_x)

        w_1 = tik_instance.Scalar(dtype=dtype, init_value=hy_tensor * hx_tensor)
        w_2 = tik_instance.Scalar(dtype=dtype, init_value=hy_tensor * lx_tensor)
        w_3 = tik_instance.Scalar(dtype=dtype, init_value=hx_tensor * ly_tensor)
        w_4 = tik_instance.Scalar(dtype=dtype, init_value=ly_tensor * lx_tensor)
    else:
        hy_tensor = tik_instance.Tensor(dtype, [1], name="hy_tensor", scope=tbe_platform.scope_ubuf)
        hy_tensor[0].set_as(h_y)
        ly_tensor = tik_instance.Tensor(dtype, [1], name="ly_tensor", scope=tbe_platform.scope_ubuf)
        ly_tensor[0].set_as(l_y)
        hx_tensor = tik_instance.Tensor(dtype, [1], name="hx_tensor", scope=tbe_platform.scope_ubuf)
        hx_tensor[0].set_as(h_x)
        lx_tensor = tik_instance.Tensor(dtype, [1], name="lx_tensor", scope=tbe_platform.scope_ubuf)
        lx_tensor[0].set_as(l_x)

        w1_tensor = tik_instance.Tensor(dtype, [1], name="w1_tensor", scope=tbe_platform.scope_ubuf)
        w2_tensor = tik_instance.Tensor(dtype, [1], name="w2_tensor", scope=tbe_platform.scope_ubuf)
        w3_tensor = tik_instance.Tensor(dtype, [1], name="w3_tensor", scope=tbe_platform.scope_ubuf)
        w4_tensor = tik_instance.Tensor(dtype, [1], name="w4_tensor", scope=tbe_platform.scope_ubuf)

        tik_instance.vec_mul(1, w1_tensor, hy_tensor, hx_tensor, 1, 8, 8, 8)
        tik_instance.vec_mul(1, w2_tensor, hy_tensor, lx_tensor, 1, 8, 8, 8)
        tik_instance.vec_mul(1, w3_tensor, hx_tensor, ly_tensor, 1, 8, 8, 8)
        tik_instance.vec_mul(1, w4_tensor, ly_tensor, lx_tensor, 1, 8, 8, 8)
        w_1 = tik_instance.Scalar(dtype=dtype)
        w_1.set_as(w1_tensor[0])
        w_2 = tik_instance.Scalar(dtype=dtype)
        w_2.set_as(w2_tensor[0])
        w_3 = tik_instance.Scalar(dtype=dtype)
        w_3.set_as(w3_tensor[0])
        w_4 = tik_instance.Scalar(dtype=dtype)
        w_4.set_as(w4_tensor[0])

    tik_instance.vec_muls(16, w1_lt[0, 0], fm_grid[0, 0, 0, 0], w_1, c_valid, n_bust, 4 * n_bust)
    tik_instance.vec_muls(16, w2_rt[0, 0], fm_grid[0, 0, 1, 0], w_2, c_valid, n_bust, 4 * n_bust)
    tik_instance.vec_muls(16, w3_lb[0, 0], fm_grid[0, 1, 0, 0], w_3, c_valid, n_bust, 4 * n_bust)
    tik_instance.vec_muls(16, w4_rb[0, 0], fm_grid[0, 1, 1, 0], w_4, c_valid, n_bust, 4 * n_bust)

    tik_instance.vec_add(16, w1_lt[0, 0], w1_lt[0, 0], w2_rt[0, 0], c_valid, n_bust, n_bust, n_bust)
    tik_instance.vec_add(16, w1_lt[0, 0], w1_lt[0, 0], w3_lb[0, 0], c_valid, n_bust, n_bust, n_bust)
    tik_instance.vec_add(16, w1_lt[0, 0], w1_lt[0, 0], w4_rb[0, 0], c_valid, n_bust, n_bust, n_bust)


def _get_average(tik_instance, grid_curr_h_f32, grid_curr_w_f32, val, c_valid,
                 p_w, n_bust):
    """
    get average
    """
    if n_bust == 2:
        dtype = "float32"
    else:
        dtype = "float16"
    vconvs32_suppot = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32f")
    if vconvs32_suppot is False or dtype == "float16":
        grid_h_num_f_tensor = tik_instance.Tensor(dtype, [1], name="grid_h_num_f_tensor", scope=tbe_platform.scope_ubuf)
        grid_h_num_f_tensor[0].set_as(grid_curr_h_f32)
        grid_w_num_f_tensor = tik_instance.Tensor(dtype, [1], name="grid_w_num_f_tensor", scope=tbe_platform.scope_ubuf)
        grid_w_num_f_tensor[0].set_as(grid_curr_w_f32)
        h_w_tensor = tik_instance.Tensor(dtype, [1], name="h_w_tensor", scope=tbe_platform.scope_ubuf)
        tik_instance.vec_mul(1, h_w_tensor, grid_w_num_f_tensor, grid_h_num_f_tensor, 1, 8, 8, 8)
        h_w = tik_instance.Scalar(dtype=dtype)

        _reciprocal(tik_instance, 1, h_w_tensor, h_w_tensor, 1, dtype)
        h_w.set_as(h_w_tensor[0])

        tik_instance.vec_muls(16, val, val, h_w, c_valid * p_w, n_bust, n_bust)

    else:
        wh_tmp = tik_instance.Scalar(dtype=dtype)
        wh_tmp.set_as(grid_curr_h_f32 * grid_curr_w_f32)

        tik_instance.vec_muls(16, val, val, 1.0 / wh_tmp, c_valid * p_w, n_bust, n_bust)


def _prepare_vbi_xm(tik_instance, h_y, l_y, h_x, l_x, c1_block_num):
    """
    :param tik_instance:
    :param h_y, l_y, h_x, l_x: the weights data
    :param c1_block_num, the num of c1 block
    :return: vbi_weights ,the rearranged weights(xm),
            hx*hy  lx*hy  hx*ly lx*ly
    """
    hy_tensor = tik_instance.Tensor("float16", [1], name="hy_tensor", scope=tbe_platform.scope_ubuf)
    hy_tensor[0].set_as(h_y)
    ly_tensor = tik_instance.Tensor("float16", [1], name="ly_tensor", scope=tbe_platform.scope_ubuf)
    ly_tensor[0].set_as(l_y)
    hx_tensor = tik_instance.Tensor("float16", [1], name="hx_tensor", scope=tbe_platform.scope_ubuf)
    hx_tensor[0].set_as(h_x)
    lx_tensor = tik_instance.Tensor("float16", [1], name="lx_tensor", scope=tbe_platform.scope_ubuf)
    lx_tensor[0].set_as(l_x)

    vbi_weights = tik_instance.Tensor("float16", [c1_block_num, 16], name="vbi_weights",
                                      scope=tbe_platform.scope_ubuf)
    w1_tensor = tik_instance.Tensor("float16", [1], name="w1_tensor", scope=tbe_platform.scope_ubuf)
    w2_tensor = tik_instance.Tensor("float16", [1], name="w2_tensor", scope=tbe_platform.scope_ubuf)
    w3_tensor = tik_instance.Tensor("float16", [1], name="w3_tensor", scope=tbe_platform.scope_ubuf)
    w4_tensor = tik_instance.Tensor("float16", [1], name="w4_tensor", scope=tbe_platform.scope_ubuf)

    tik_instance.vec_mul(1, w1_tensor, hy_tensor, hx_tensor, 1, 8, 8, 8)
    tik_instance.vec_mul(1, w2_tensor, hy_tensor, lx_tensor, 1, 8, 8, 8)
    tik_instance.vec_mul(1, w3_tensor, hx_tensor, ly_tensor, 1, 8, 8, 8)
    tik_instance.vec_mul(1, w4_tensor, ly_tensor, lx_tensor, 1, 8, 8, 8)

    for current_c1 in range(c1_block_num):
        vbi_weights[current_c1 * 16 + 0].set_as(w1_tensor[0])
        vbi_weights[current_c1 * 16 + 1].set_as(w2_tensor[0])
        vbi_weights[current_c1 * 16 + 2].set_as(w3_tensor[0])
        vbi_weights[current_c1 * 16 + 3].set_as(w4_tensor[0])

    return vbi_weights


def _prepare_vbi_xn(tik_instance, c1_block_num):
    """
    :param tik_instance:
    :param c1_block_num, the num of c1 block
    :return:Rearranged address(Xn)
    """
    vbi_addr = tik_instance.Tensor("int32", [c1_block_num, 32], name="vbi_addr", scope=tbe_platform.scope_ubuf)
    for one_block_element_num in range(16):
        one_block_offset = tik_instance.Scalar(dtype="int32", init_value=32 * one_block_element_num)
        vbi_addr[0, one_block_element_num].set_as(one_block_offset)
    one_block_offset_num = tik_instance.Scalar(dtype="int32", init_value=512)
    tik_instance.vec_adds(16, vbi_addr[0, 16], vbi_addr, one_block_offset_num, 1, 8, 8)

    for current_c1 in range(1, c1_block_num):
        c1_offset = tik_instance.Scalar(dtype="int32", init_value=1024 * current_c1)
        tik_instance.vec_adds(32, vbi_addr[current_c1, 0], vbi_addr, c1_offset, 1, 8, 8)

    return vbi_addr


def _bilinear_interpolate(tik_instance, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo,
                          x_hi, y_lo, y_hi, raw_x, raw_y, sample_num_w,
                          sample_num_h, grid_h_num_f,
                          grid_w_num_f, fm_h, fm_w,
                          fm_c1, dtype, n_bust, pw_s,
                          pw_int, ph_int, block_i, block_num,
                          index, curr_roi, feature_map, ret, roi_128_number,
                          w_number, fm_to_l1, cache_fm, cache_index, fm_to_ub,
                          w_number_ub, feature_map_ub):
    """
    _bilinear_interpolate
    """
    if fm_to_ub >= 1:
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move(feature_map_ub, feature_map[index, 0, 0, 0, 0], 0, 1, fm_c1 * fm_h * fm_w * n_bust,
                                   0, 0)
            cache_index.set_as(index)
    elif fm_to_l1 >= 1:
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move(cache_fm, feature_map[index, 0, 0, 0, 0], 0, 1, fm_c1 * fm_h * fm_w * n_bust, 0, 0)
            cache_index.set_as(index)
    elif w_number_ub >= 2:
        cache_ub = tik_instance.Tensor(dtype, [2, fm_c1, fm_w, 16], name="cache_ub", scope=tbe_platform.scope_ubuf)
        cache_table = tik_instance.Tensor("int32", [2, 2], name="cache_table", scope=tbe_platform.scope_ubuf)
        with tik_instance.for_range(0, 2) as j:
            one = tik_instance.Scalar(dtype="int32", init_value=-1)
            cache_table[j, 0].set_as(one)
    elif w_number >= 2:
        cache_l1, cache_table = _init_l1(tik_instance, dtype, 2, fm_c1, fm_w, 16)
        with tik_instance.for_range(0, 2) as j:
            one = tik_instance.Scalar(dtype="int32", init_value=-1)
            cache_table[j, 0].set_as(one)

    val = tik_instance.Tensor(dtype, [fm_c1, pw_int, 16], name="val", scope=tbe_platform.scope_ubuf)

    tik_instance.vec_dup(16, val, 0.0, fm_c1 * pw_s, n_bust)

    roi_y_floor = tik_instance.Tensor("int32", [128], name="roi_y_floor", scope=tbe_platform.scope_ubuf)
    roi_x_floor = tik_instance.Tensor("int32", [128], name="roi_x_floor", scope=tbe_platform.scope_ubuf)

    suppot_vconv = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32c")
    if not suppot_vconv and dtype == "float32":
        raw_fp16 = tik_instance.Tensor("float16", [128], name="raw_fp16", scope=tbe_platform.scope_ubuf)
        tik_instance.vec_conv(64, "", raw_fp16[0], raw_y[0], 2, 4, 8)
        # maybe repeattimes 1
        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_fp16[0], 2, 8, 4)

        tik_instance.vec_conv(64, "", raw_fp16[0], raw_x[0], 2, 4, 8)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_fp16[0], 2, 8, 4)
    else:
        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_y[0], 2, 8, 4)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_x[0], 2, 8, 4)

    with tik_instance.for_range(0, ph_int * sample_num_h) as grid_num_h:
        verify = tik_instance.Scalar(dtype="int32", init_value=0)
        y_tmp = tik_instance.Scalar(dtype="int32")
        y_tmp.set_as(roi_y_floor[grid_num_h])
        with tik_instance.if_scope(y_tmp < -1):
            verify.set_as(1)
        with tik_instance.if_scope(y_tmp >= fm_h):
            verify.set_as(1)

        with tik_instance.if_scope(verify == 0):
            y_low = tik_instance.Scalar(dtype="int32", init_value=y_lo[grid_num_h])
            y_high = tik_instance.Scalar(dtype="int32", init_value=y_hi[grid_num_h])
            if w_number_ub >= 2:
                _load_a_w_to_l1(tik_instance, cache_table, cache_ub, feature_map, index, y_low, n_bust, 0)
                _load_a_w_to_l1(tik_instance, cache_table, cache_ub, feature_map, index, y_high, n_bust, 1)
            elif w_number >= 2:
                _load_a_w_to_l1(tik_instance, cache_table, cache_l1, feature_map, index, y_low, n_bust, 0)
                _load_a_w_to_l1(tik_instance, cache_table, cache_l1, feature_map, index, y_high, n_bust, 1)

            with tik_instance.for_range(0, pw_int * sample_num_w) as grid_num_w:
                x_tmp = tik_instance.Scalar(dtype="int32")
                x_tmp.set_as(roi_x_floor[grid_num_w])
                verify.set_as(0)
                with tik_instance.if_scope(x_tmp < -1):
                    verify.set_as(1)
                with tik_instance.if_scope(x_tmp >= fm_w):
                    verify.set_as(1)

                with tik_instance.if_scope(verify == 0):
                    w1_lt = tik_instance.Tensor(dtype, [fm_c1, 16], name="w1_lt", scope=tbe_platform.scope_ubuf)
                    w2_rt = tik_instance.Tensor(dtype, [fm_c1, 16], name="w2_rt", scope=tbe_platform.scope_ubuf)
                    w3_lb = tik_instance.Tensor(dtype, [fm_c1, 16], name="w3_lb", scope=tbe_platform.scope_ubuf)
                    w4_rb = tik_instance.Tensor(dtype, [fm_c1, 16], name="w4_rb", scope=tbe_platform.scope_ubuf)
                    x_low = tik_instance.Scalar(dtype="int32", init_value=x_lo[grid_num_w])
                    x_high = tik_instance.Scalar(dtype="int32", init_value=x_hi[grid_num_w])
                    feature_shape = [0, 0, fm_h, fm_w]
                    fm_grid = tik_instance.Tensor(dtype, (fm_c1, 2, 2, 16), name="fm_grid",
                                                  scope=tbe_platform.scope_ubuf)

                    if fm_to_ub >= 1:
                        _load_feature_map_to_ub(tik_instance, fm_grid, feature_shape,
                                                fm_c1, fm_c1, feature_map_ub, index, 0,
                                                y_low, x_low, x_high, y_high, n_bust, 1)
                    elif fm_to_l1 >= 1:
                        _load_feature_map_to_ub(tik_instance, fm_grid, feature_shape,
                                                fm_c1, fm_c1, cache_fm, index, 0,
                                                y_low, x_low, x_high, y_high, n_bust, 1)
                    elif w_number_ub >= 2:
                        _load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_ub, 0, 0, fm_c1, x_low, x_high,
                                            fm_c1, n_bust)
                        _load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_ub, 1, 0, fm_c1, x_low, x_high,
                                            fm_c1, n_bust)
                    elif w_number >= 2:
                        _load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_l1, 0, 0, fm_c1, x_low, x_high,
                                            fm_c1, n_bust)
                        _load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_l1, 1, 0, fm_c1, x_low, x_high,
                                            fm_c1, n_bust)
                    else:
                        _load_feature_map_to_ub(tik_instance, fm_grid, feature_shape, fm_c1, fm_c1, feature_map,
                                                index, 0, y_low, x_low, x_high, y_high, n_bust, 0)

                    h_y = tik_instance.Scalar(dtype, init_value=y_hi_w[grid_num_h])
                    l_y = tik_instance.Scalar(dtype, init_value=y_lo_w[grid_num_h])
                    h_x = tik_instance.Scalar(dtype, init_value=x_hi_w[grid_num_w])
                    l_x = tik_instance.Scalar(dtype, init_value=x_lo_w[grid_num_w])

                    cce_product = tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION)

                    if (cce_product in ("Ascend610", "Ascend710", "Hi3796CV300CS", "SD3403")) \
                            and (dtype == "float16") and (fm_c1 >= 8):
                        c1_block_num = (fm_c1 + 7) // 8
                        vbi_weights = _prepare_vbi_xm(tik_instance, h_y, l_y, h_x, l_x, c1_block_num)
                        vbi_addr = _prepare_vbi_xn(tik_instance, c1_block_num)
                        for current_c1 in range(c1_block_num):
                            tik_instance.vbi(128, w1_lt, fm_grid[0:, current_c1:, 0:, 0:], vbi_weights[current_c1:, 0:],
                                             vbi_addr[current_c1:, 0:], 1, 1, 4, 0, 128)
                    _compute_w1234(tik_instance, h_y, l_y, h_x, l_x, w1_lt, w2_rt, w3_lb, w4_rb, fm_grid, fm_c1, n_bust)

                    with tik_instance.for_range(0, fm_c1) as c_iter_i:
                        tik_instance.vec_add(16, val[c_iter_i, grid_num_w // sample_num_w, 0],
                                             val[c_iter_i, grid_num_w // sample_num_w, 0],
                                             w1_lt[c_iter_i, 0], 1, n_bust, n_bust, n_bust)

        with tik_instance.if_scope((grid_num_h + 1) % sample_num_h == 0):
            _get_average(tik_instance, grid_h_num_f, grid_w_num_f, val, fm_c1, pw_s, n_bust)

            with tik_instance.if_scope((pw_int * ph_int - pw_int) * n_bust <= 65535):
                tik_instance.data_move(ret[block_i * block_num + 128 * roi_128_number + curr_roi,
                                       0, grid_num_h // sample_num_h, 0, 0], val[0, 0, 0], 0, fm_c1,
                                       pw_int * n_bust, 0, (pw_int * ph_int - pw_int)*n_bust)

            with tik_instance.else_scope():
                with tik_instance.for_range(0, fm_c1) as c_iter_i:
                    tik_instance.data_move(ret[block_i * block_num + 128 * roi_128_number + curr_roi, c_iter_i,
                                           grid_num_h // sample_num_h, 0, 0], val[c_iter_i, 0, 0], 0, 1,
                                           pw_int * n_bust, 0, 0)

            tik_instance.vec_dup(16, val, 0.0, fm_c1 * pw_s, n_bust)


def _extract_roi_v200(tik_instance, rois, n_bust, block_i, block_num, roi_128_number, proposals_ub_x0, proposals_ub_y0,
                      proposals_ub_x1, proposals_ub_y1, roi_fm_index, dtype):
    rois_ub_n5 = tik_instance.Tensor(dtype, [128 * 5], name="rois_ub_n5", scope=tbe_platform.scope_ubuf)
    tik_instance.data_move(rois_ub_n5[0], rois[block_i * block_num + roi_128_number * 128, 0], 0, 1, 40 * n_bust, 0, 0)

    src1_ub_fp16 = tik_instance.Tensor("float16", (128,), name="src1_ub_fp16", scope=tbe_platform.scope_ubuf)
    src1_ub_uint16 = tik_instance.Tensor("uint16", (8,), name="src1_ub_uint16", scope=tbe_platform.scope_ubuf)
    one_ub = tik_instance.Tensor("float16", (128,), name="one_ub", scope=tbe_platform.scope_ubuf)
    tik_instance.vec_dup(128, src1_ub_fp16, 0, 1, 8)
    tik_instance.vec_dup(128, one_ub, 1, 1, 8)
    tik_instance.vec_dup(8, src1_ub_uint16, 0, 1, 8)
    tik_instance.vec_add([0x0000000000001084, 0x2108421084210842], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(128, proposals_ub_x0[0, 0], rois_ub_n5, src1_ub_uint16, 7, 1, 5, 0, 0, None, "normal")
    tik_instance.vector_dup(128, src1_ub_fp16, 0, 1, 1, 8)
    tik_instance.vec_add([0x1084210842108421, 0x0842000000000000], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(128, proposals_ub_x0[112, 0], rois_ub_n5[128 * 4], src1_ub_uint16, 1, 1, 8, 0, 0, None,
                         "normal")

    tik_instance.vector_dup(128, src1_ub_fp16, 0, 1, 1, 8)
    tik_instance.vec_add([0x0000000000002108, 0x4210842108421084], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(128, proposals_ub_y0[0, 0], rois_ub_n5, src1_ub_uint16, 7, 1, 5, 0, 0, None, "normal")
    tik_instance.vector_dup(128, src1_ub_fp16, 0, 1, 1, 8)
    tik_instance.vec_add([0x2108421084210842, 0x1084000000000000], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(128, proposals_ub_y0[112, 0], rois_ub_n5[128 * 4], src1_ub_uint16, 1, 1, 8, 0, 0, None,
                         "normal")

    tik_instance.vector_dup(128, src1_ub_fp16, 0, 1, 1, 8)
    tik_instance.vec_add([0x0000000000004210, 0x8421084210842108], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(128, proposals_ub_x1[0, 0], rois_ub_n5, src1_ub_uint16, 7, 1, 5, 0, 0, None, "normal")
    tik_instance.vector_dup(128, src1_ub_fp16, 0, 1, 1, 8)
    tik_instance.vec_add([0x4210842108421084, 0x2108000000000000], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(128, proposals_ub_x1[112, 0], rois_ub_n5[128 * 4], src1_ub_uint16, 1, 1, 8, 0, 0, None,
                         "normal")

    tik_instance.vector_dup(128, src1_ub_fp16, 0, 1, 1, 8)
    tik_instance.vec_add([0x0000000000008421, 0x0842108421084210], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(80, proposals_ub_y1[0, 0], rois_ub_n5, src1_ub_uint16, 7, 1, 5, 0, 0, None, "normal")
    tik_instance.vector_dup(128, src1_ub_fp16, 0, 1, 1, 8)
    tik_instance.vec_add([0x8421084210842108, 0x4210000000000000], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(128, proposals_ub_y1[112, 0], rois_ub_n5[128 * 4], src1_ub_uint16, 1, 1, 8, 0, 0, None,
                         "normal")

    tik_instance.vector_dup(128, src1_ub_fp16, 0, 1, 1, 8)
    tik_instance.vec_add([0x0000000000000842, 0x1084210842108421], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(80, roi_fm_index[0], rois_ub_n5, src1_ub_uint16, 7, 1, 5, 0, 0, None, "normal")
    tik_instance.vector_dup(128, src1_ub_fp16, 0, 1, 1, 8)
    tik_instance.vec_add([0x0842108421084210, 0x8421000000000000], src1_ub_fp16, src1_ub_fp16, one_ub, 1, 8, 8, 8)
    tik_instance.vec_cmpv_eq(src1_ub_uint16, src1_ub_fp16, one_ub, 1, 8, 8)
    tik_instance.vreduce(128, roi_fm_index[112], rois_ub_n5[128 * 4], src1_ub_uint16, 1, 1, 8, 0, 0, None, "normal")


def _extract_roi(tik_instance, rois_shape, dtype, rois, block_i, block_num, roi_128_number, n_bust, rois_ub,
                 cce_product, proposals_ub_x0, proposals_ub_y0, proposals_ub_x1, proposals_ub_y1):
    if rois_shape[1] == 5:
        rois_ub_n5 = tik_instance.Tensor(dtype, [128, 5], name="rois_ub_n5", scope=tbe_platform.scope_ubuf)
        tik_instance.data_move(rois_ub_n5[0, 0], rois[block_i * block_num + roi_128_number * 128, 0],
                               0, 1, 40 * n_bust, 0, 0)
        _tf_n52n8(tik_instance, rois_ub, rois_ub_n5, 128)
    elif rois_shape[1] == 4:
        rois_ub_n4 = tik_instance.Tensor(dtype, [128, 4], name="rois_ub_n4", scope=tbe_platform.scope_ubuf)
        tik_instance.data_move(rois_ub_n4[0, 0], rois[block_i * block_num + roi_128_number * 128, 0], 0, 1,
                               32 * n_bust, 0, 0)
        _tf_n42n8(tik_instance, rois_ub, rois_ub_n4, 128)
    else:
        tik_instance.data_move(rois_ub[0, 0], rois[block_i * block_num + roi_128_number * 128, 0],
                               0, 1, 64 * n_bust, 0, 0)

    support_vextract = tbe_platform.api_check_support("tik.vextract", "float32")
    if dtype == "float16":
        if cce_product == tbe_platform.ASCEND_310:
            j_value = tik_instance.Scalar(dtype=dtype)
            with tik_instance.for_range(0, 128) as j:
                j_value.set_as(rois_ub[j, 4])
                proposals_ub_y1[j, 0].set_as(j_value)
        else:
            tik_instance.vextract(proposals_ub_y1[0, 0], rois_ub[0], 8, 4)
        tik_instance.vextract(proposals_ub_x0[0, 0], rois_ub[0], 8, 1)
        tik_instance.vextract(proposals_ub_y0[0, 0], rois_ub[0], 8, 2)
        tik_instance.vextract(proposals_ub_x1[0, 0], rois_ub[0], 8, 3)
    else:
        if support_vextract is False:
            j_value = tik_instance.Scalar(dtype=dtype)
            with tik_instance.for_range(0, 128) as j:
                j_value.set_as(rois_ub[j, 1])
                proposals_ub_x0[j, 0].set_as(j_value)
                j_value.set_as(rois_ub[j, 2])
                proposals_ub_y0[j, 0].set_as(j_value)
                j_value.set_as(rois_ub[j, 3])
                proposals_ub_x1[j, 0].set_as(j_value)
                j_value.set_as(rois_ub[j, 4])
                proposals_ub_y1[j, 0].set_as(j_value)
        else:
            tik_instance.vextract(proposals_ub_x0[0, 0], rois_ub[0], 8, 1)
            tik_instance.vextract(proposals_ub_y0[0, 0], rois_ub[0], 8, 2)
            tik_instance.vextract(proposals_ub_x1[0, 0], rois_ub[0], 8, 3)
            tik_instance.vextract(proposals_ub_y1[0, 0], rois_ub[0], 8, 4)


def roi_align_compute(tik_instance, feature_map, ret, proposals_ub_x0,
                      proposals_ub_y0, pool_h, pool_w, dtype, roi_128_number,
                      rois_valid_in_block,
                      feature_shape, grid_curr_h, grid_curr_w, fm_c1, n_bust,
                      block_i, block_num, roi_int32_fm_index, grid_h_int32,
                      grid_w_int32, grid_h_fp32, grid_w_fp32,
                      roi_bin_h_fp32_value,
                      roi_bin_w_fp32_value, w_number, fm_to_l1, fm_to_ub,
                      w_number_ub):
    """
    get ret without L1
    """
    grid_h = tik_instance.Tensor(
        dtype, [128], name="grid_h", scope=tbe_platform.scope_ubuf)
    grid_w = tik_instance.Tensor(
        dtype, [128], name="grid_w", scope=tbe_platform.scope_ubuf)
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2

    vdiv_suppot = tbe_platform.api_check_support("tik.vdiv", "float32")
    if vdiv_suppot is False:
        _newton(tik_instance, 64 * dtype_num, grid_h, roi_bin_h_fp32_value, grid_h_fp32, 2 // dtype_num, dtype)
        _newton(tik_instance, 64 * dtype_num, grid_w, roi_bin_w_fp32_value, grid_w_fp32, 2 // dtype_num, dtype)
    else:
        tik_instance.vdiv(64 * dtype_num, grid_h, roi_bin_h_fp32_value, grid_h_fp32, 2 // dtype_num, 1,
                          1, 1, 8, 8, 8)
        tik_instance.vdiv(64 * dtype_num, grid_w, roi_bin_w_fp32_value, grid_w_fp32, 2 // dtype_num, 1,
                          1, 1, 8, 8, 8)

    if fm_to_ub >= 1:
        feature_map_ub = tik_instance.Tensor(
            dtype, [1, fm_c1, feature_shape[2], feature_shape[3], 16],
            name="feature_map_ub",
            scope=tbe_platform.scope_ubuf)
    else:
        feature_map_ub = tik_instance.Tensor(
            dtype, [1], name="feature_map_ub", scope=tbe_platform.scope_ubuf)

    if fm_to_l1 >= 1:
        cache_fm = tik_instance.Tensor(
            dtype, [1, fm_c1, feature_shape[2], feature_shape[3], 16],
            name="cache_fm",
            scope=tbe_platform.scope_cbuf)
        cache_index = tik_instance.Scalar(dtype="int32", init_value=-1)
    else:
        cache_fm = tik_instance.Tensor(
            dtype, [1], name="cache_fm", scope=tbe_platform.scope_ubuf)
        cache_index = tik_instance.Scalar(dtype="int32", init_value=-1)

    with tik_instance.for_range(0, rois_valid_in_block) as curr_roi:
        index = tik_instance.Scalar(dtype="int32")

        index.set_as(roi_int32_fm_index[curr_roi])
        grid_curr_h.set_as(grid_h_int32[curr_roi])
        grid_curr_w.set_as(grid_w_int32[curr_roi])

        w_num = tik_instance.Scalar(dtype="int32")
        h_num = tik_instance.Scalar(dtype="int32")
        w_num.set_as((grid_curr_w * pool_w + 127) // 128)
        h_num.set_as((grid_curr_h * pool_h + 127) // 128)
        grid_curr_h_f32 = tik_instance.Scalar(
            dtype=dtype, init_value=grid_h_fp32[curr_roi])
        grid_curr_w_f32 = tik_instance.Scalar(
            dtype=dtype, init_value=grid_w_fp32[curr_roi])

        flag_para = tik_instance.Scalar(dtype="int32", init_value=0)
        with tik_instance.if_scope(w_num > 1):
            flag_para.set_as(1)
        with tik_instance.if_scope(h_num > 1):
            flag_para.set_as(1)
        with tik_instance.if_scope(fm_c1 * pool_w > 255):
            flag_para.set_as(1)
        pool_w_s = tik_instance.Scalar(dtype="int32", init_value=pool_w)

        with tik_instance.if_scope(flag_para == 0):
            if fm_c1 * pool_w <= 255:
                grid_w_roi, grid_h_roi, grid_w_num, \
                grid_h_num, rois_start_w, rois_start_h, \
                grid_h_num_f, grid_w_num_f = \
                    _get_input(tik_instance, dtype, grid_h, grid_w,
                               proposals_ub_y0,
                               proposals_ub_x0,
                               grid_h_int32, grid_w_int32,
                               grid_h_fp32, grid_w_fp32, curr_roi)

                x_lo_w, x_hi_w, y_lo_w, y_hi_w, \
                x_lo, x_hi, y_lo, y_hi, raw_x, raw_y = \
                    _get_grid_weight(tik_instance, grid_w_roi,
                                     grid_h_roi, rois_start_w, rois_start_h,
                                     feature_shape[2], feature_shape[3], dtype)

                _bilinear_interpolate(
                    tik_instance, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo,
                    y_hi, raw_x, raw_y, grid_w_num, grid_h_num, grid_h_num_f,
                    grid_w_num_f, feature_shape[2], feature_shape[3], fm_c1, dtype,
                    n_bust, pool_w_s, pool_w, pool_h, block_i, block_num, index,
                    curr_roi, feature_map, ret, roi_128_number, w_number,
                    fm_to_l1, cache_fm, cache_index, fm_to_ub, w_number_ub,
                    feature_map_ub)

        with tik_instance.else_scope():
            _compute_roi_with_single_point(tik_instance, feature_shape, dtype,
                                           fm_to_l1,
                                           fm_c1, block_i, index,
                                           roi_128_number,
                                           block_num, w_number, pool_h, pool_w,
                                           n_bust,
                                           grid_curr_h, roi_bin_h_fp32_value,
                                           ret, grid_curr_w_f32,
                                           grid_w_fp32, grid_curr_h_f32,
                                           proposals_ub_y0,
                                           grid_h_fp32, roi_bin_w_fp32_value,
                                           proposals_ub_x0, feature_map, curr_roi,
                                           grid_curr_w, cache_fm, cache_index,
                                           fm_to_ub, w_number_ub, feature_map_ub)


def _compute_roi_with_single_point(tik_instance, feature_shape, dtype,
                                   fm_to_l1, fm_c1,
                                   block_i, index, roi_128_number, block_num,
                                   w_number, pool_h, pool_w, n_bust, grid_curr_h,
                                   roi_bin_h_fp32_value, ret, grid_curr_w_f32,
                                   grid_w_fp32, grid_curr_h_f32,
                                   proposals_ub_y0, grid_h_fp32, roi_bin_w_fp32_value,
                                   proposals_ub_x0, feature_map, curr_roi, grid_curr_w,
                                   cache_fm, cache_index, fm_to_ub,
                                   w_number_ub, feature_map_ub):
    """
    compute roi with single point
    """
    fm_h = feature_shape[2]
    fm_w = feature_shape[3]
    if fm_to_ub >= 1:
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move(feature_map_ub, feature_map[index, 0, 0, 0, 0], 0, 1, fm_c1 * fm_h * fm_w * n_bust,
                                   0, 0)
            cache_index.set_as(index)
    elif fm_to_l1 >= 1:
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move(cache_fm, feature_map[index, 0, 0, 0, 0], 0, 1, fm_c1 * fm_h * fm_w * n_bust, 0, 0)
            cache_index.set_as(index)
    elif w_number_ub >= 2:
        cache_ub = tik_instance.Tensor(dtype, [2, fm_c1, fm_w, 16], name="cache_ub", scope=tbe_platform.scope_ubuf)
        cache_table = tik_instance.Tensor("int32", [2, 2], name="cache_table", scope=tbe_platform.scope_ubuf)
        with tik_instance.for_range(0, 2) as j:
            one = tik_instance.Scalar(dtype="int32", init_value=-1)
            cache_table[j, 0].set_as(one)
    elif w_number >= 2:
        cache_l1, cache_table = _init_l1(tik_instance, dtype, 2, fm_c1, fm_w, 16)
        with tik_instance.for_range(0, 2) as j:
            one = tik_instance.Scalar(dtype="int32", init_value=-1)
            cache_table[j, 0].set_as(one)

    thread_num = 2
    if pool_w == 1:
        thread_num = 1
    with tik_instance.for_range(0, pool_h) as p_h:
        with tik_instance.for_range(0, pool_w, thread_num=thread_num) as p_w:
            # less 255
            c_block = 110
            c_number = (fm_c1 + (c_block - 1)) // c_block
            val = tik_instance.Tensor(dtype, [c_block, 16], name="val", scope=tbe_platform.scope_ubuf)
            for current_cb in range(c_number):
                c_valid = c_block
                if current_cb == c_number - 1:
                    c_valid = fm_c1 - c_block * current_cb

                tik_instance.vec_dup(16, val, 0.0, c_block, n_bust)
                with tik_instance.if_scope(c_valid != 0):
                    w1_lt = tik_instance.Tensor(dtype, [c_block, 16], name="w1_lt", scope=tbe_platform.scope_ubuf)
                    w2_rt = tik_instance.Tensor(dtype, [c_block, 16], name="w2_rt", scope=tbe_platform.scope_ubuf)
                    w3_lb = tik_instance.Tensor(dtype, [c_block, 16], name="w3_lb", scope=tbe_platform.scope_ubuf)
                    w4_rb = tik_instance.Tensor(dtype, [c_block, 16], name="w4_rb", scope=tbe_platform.scope_ubuf)
                    with tik_instance.for_range(0, grid_curr_h) as g_h:
                        verify = tik_instance.Scalar(dtype="int32", init_value=0)

                        roi_bin_ph_gh_ly_int32, l_y, roi_bin_ph_gh_hy_int32, h_y, verify = \
                            _get_grid_weight_per_roi(tik_instance, roi_bin_h_fp32_value, proposals_ub_y0, grid_h_fp32,
                                                     p_h, g_h, fm_h, curr_roi, dtype, verify, 1)

                        with tik_instance.if_scope(verify == 0):
                            y_low = tik_instance.Scalar(dtype="int32", init_value=roi_bin_ph_gh_ly_int32[0])
                            y_high = tik_instance.Scalar(dtype="int32", init_value=roi_bin_ph_gh_hy_int32[0])
                            if w_number_ub >= 2:
                                _load_a_w_to_l1(tik_instance, cache_table, cache_ub, feature_map, index, y_low, n_bust,
                                                0)
                                _load_a_w_to_l1(tik_instance, cache_table, cache_ub, feature_map, index, y_high, n_bust,
                                                1)
                            elif w_number >= 2:
                                _load_a_w_to_l1(tik_instance, cache_table, cache_l1, feature_map, index, y_low, n_bust,
                                                0)
                                _load_a_w_to_l1(tik_instance, cache_table, cache_l1, feature_map, index, y_high, n_bust,
                                                1)
                            with tik_instance.for_range(0, grid_curr_w) as g_w:
                                roi_bin_pw_gw_lx_int32, l_x, roi_bin_pw_gw_hx_int32, h_x, verify = \
                                    _get_grid_weight_per_roi(tik_instance, roi_bin_w_fp32_value, proposals_ub_x0,
                                                             grid_w_fp32, p_w, g_w, fm_w, curr_roi, dtype, verify, 0)

                                with tik_instance.if_scope(verify == 0):
                                    x_low = tik_instance.Scalar(dtype="int32", init_value=roi_bin_pw_gw_lx_int32[0])
                                    x_high = tik_instance.Scalar(dtype="int32", init_value=roi_bin_pw_gw_hx_int32[0])
                                    fm_grid = tik_instance.Tensor(dtype, (c_block, 2, 2, 16), name="fm_grid",
                                                                  scope=tbe_platform.scope_ubuf)
                                    if fm_to_ub >= 1:
                                        _load_feature_map_to_ub(tik_instance, fm_grid, feature_shape, c_block, c_valid,
                                                                feature_map_ub, index, 0, y_low, x_low, x_high, y_high,
                                                                n_bust, 1)
                                    elif fm_to_l1 >= 1:
                                        _load_feature_map_to_ub(tik_instance, fm_grid, feature_shape, c_block, c_valid,
                                                                cache_fm, index, 0, y_low, x_low, x_high, y_high,
                                                                n_bust, 1)
                                    elif w_number_ub >= 2:
                                        _load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_ub, 0, current_cb,
                                                            c_block, x_low, x_high, c_valid, n_bust)
                                        _load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_ub, 1, current_cb,
                                                            c_block, x_low, x_high, c_valid, n_bust)
                                    elif w_number >= 2:
                                        _load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_l1, 0, current_cb,
                                                            c_block, x_low, x_high, c_valid, n_bust)
                                        _load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_l1, 1, current_cb,
                                                            c_block, x_low, x_high, c_valid, n_bust)
                                    else:
                                        _load_feature_map_to_ub(tik_instance, fm_grid, feature_shape, c_block, c_valid,
                                                                feature_map, index, current_cb, y_low, x_low, x_high,
                                                                y_high, n_bust, 0)

                                    _compute_w1234(tik_instance, h_y, l_y, h_x, l_x, w1_lt, w2_rt, w3_lb, w4_rb,
                                                   fm_grid, c_valid, n_bust)

                                    tik_instance.vec_add(16, val, val, w1_lt, c_valid, n_bust, n_bust, n_bust)

                    _get_average(tik_instance, grid_curr_h_f32, grid_curr_w_f32, val, c_valid, 1, n_bust)

                    with tik_instance.if_scope((pool_h * pool_w - 1) * n_bust <= 65535):
                        tik_instance.data_move(ret[block_i * block_num + 128 * roi_128_number + curr_roi,
                                               current_cb * c_block, p_h, p_w, 0], val, 0, c_valid, n_bust, 0,
                                               (pool_h * pool_w - 1) * n_bust)
                    with tik_instance.else_scope():
                        with tik_instance.for_range(0, c_valid) as c_iter_i:
                            tik_instance.data_move(ret[block_i * block_num + 128 * roi_128_number + curr_roi,
                                                       current_cb * c_block + c_iter_i, p_h, p_w, 0], val[c_iter_i, 0],
                                                   0, 1, n_bust, 0, 0)


# 'pylint: disable=unused-argument,invalid-name
def _get_grid_weight_per_roi(tik_instance, roi_bin_h_fp32_value,
                             proposals_ub_y0, grid_h_fp32, pool_n, grid_n, fm_h,
                             curr_roi, dtype, verify, w_h):
    """
    get grid size and coordinate in feature
    """
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2
    roi_bin_ph_gh_hy_int32 = tik_instance.Tensor("int32", [1], name="roi_bin_ph_gh_hy_int32",
                                                 scope=tbe_platform.scope_ubuf)
    roi_bin_ph_gh_ly_int32 = tik_instance.Tensor("int32", [1], name="roi_bin_ph_gh_ly_int32",
                                                 scope=tbe_platform.scope_ubuf)
    roi_bin_ph_gh_pos_fp32 = tik_instance.Tensor(dtype, [1], name="roi_bin_ph_gh_pos_fp32",
                                                 scope=tbe_platform.scope_ubuf)
    roi_bin_ph_gh_ly_weight = tik_instance.Tensor(dtype, [1], name="roi_bin_ph_gh_ly_weight",
                                                  scope=tbe_platform.scope_ubuf)
    roi_bin_ph_gh_hy_weight = tik_instance.Tensor(dtype, [1], name="roi_bin_ph_gh_hy_weight",
                                                  scope=tbe_platform.scope_ubuf)
    tmp_float16 = tik_instance.Tensor("float16", [1], name="tmp_float16", scope=tbe_platform.scope_ubuf)

    const_one = tik_instance.Tensor("int32", [1], name="const_one", scope=tbe_platform.scope_ubuf)
    tik_instance.vec_dup(1, const_one, 1, 1, 0)

    tmp_bin_h_fp32 = tik_instance.Tensor(dtype, [1], name="tmp_bin_h_fp32", scope=tbe_platform.scope_ubuf)
    tmp_bin_h_fp32[0].set_as(roi_bin_h_fp32_value[curr_roi])
    vconv_suppot = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32c")

    if vconv_suppot is False or dtype == "float16":
        tmp_int32 = tik_instance.Tensor("int32", [1], name="tmp_int32", scope=tbe_platform.scope_ubuf)
        tmp_float32 = tik_instance.Tensor(dtype, [1], name="tmp_float32", scope=tbe_platform.scope_ubuf)
        tmp_int32[0].set_as(pool_n)

        tik_instance.vec_conv(1, "", tmp_float16, tmp_int32, 1, 4, 8, 1.0)
        if dtype == "float32":
            tik_instance.vec_conv(1, "", tmp_float32, tmp_float16, 1, 8, 4)
            tik_instance.vec_mul(1, roi_bin_ph_gh_pos_fp32, tmp_bin_h_fp32, tmp_float32, 1, 8, 8, 8)
        else:
            tik_instance.vec_mul(1, roi_bin_ph_gh_pos_fp32, tmp_bin_h_fp32, tmp_float16, 1, 8, 8, 8)
    else:
        tik_instance.vec_muls(1, roi_bin_ph_gh_pos_fp32, tmp_bin_h_fp32, pool_n, 1, 8, 8)

    const_value_fp32 = tik_instance.Tensor(dtype, [1], name="const_value_fp32", scope=tbe_platform.scope_ubuf)

    const_value_fp32[0].set_as(proposals_ub_y0[curr_roi, 0])

    # get every bin_start_pose
    tik_instance.vec_add(1, roi_bin_ph_gh_pos_fp32[0], roi_bin_ph_gh_pos_fp32[0], const_value_fp32, 1, 8, 8, 8)
    const_value_fp32[0].set_as(grid_h_fp32[curr_roi])
    support_div = tbe_platform.api_check_support("tik.vdiv", "float32")
    if support_div is False:
        _newton(tik_instance, 1, tmp_bin_h_fp32, tmp_bin_h_fp32, const_value_fp32, 1, dtype)

    else:
        tik_instance.vdiv(1, tmp_bin_h_fp32, tmp_bin_h_fp32, const_value_fp32, 1, 1, 1, 1, 8, 8, 8)

    # i * bin_size_h /sample_num_h;
    if vconv_suppot is False or dtype == "float16":
        tmp_int32 = tik_instance.Tensor("int32", [1], name="tmp_int32", scope=tbe_platform.scope_ubuf)
        tmp_int32[0].set_as(grid_n)
        if dtype == "float32":
            tik_instance.vec_conv(1, "", tmp_float16, tmp_int32, 1, 4, 8, 1.0)
            tik_instance.vec_conv(1, "", const_value_fp32, tmp_float16, 1, 8, 4)
        else:
            tik_instance.vec_conv(1, "", const_value_fp32, tmp_int32, 1, 4, 8, 1.0)
    else:
        tik_instance.vec_dup(1, const_value_fp32, grid_n, 1, 0)

    tik_instance.vec_adds(1, const_value_fp32, const_value_fp32, 0.5, 1, 8, 8)
    tik_instance.vec_mul(1, tmp_bin_h_fp32, tmp_bin_h_fp32, const_value_fp32, 1, 8, 8, 8)
    tik_instance.vec_add(1, roi_bin_ph_gh_pos_fp32[0], tmp_bin_h_fp32, roi_bin_ph_gh_pos_fp32[0], 1, 8, 8, 8)

    roi_y_floor = tik_instance.Tensor("int32", [1], name="roi_y_floor", scope=tbe_platform.scope_ubuf)
    vconvf_suppot = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322s32f")
    if vconvf_suppot is False and dtype == "float32":
        tik_instance.vec_conv(1, "", tmp_float16, roi_bin_ph_gh_pos_fp32, 1, 4, 8)
        tik_instance.vec_conv(1, "floor", roi_y_floor[0], tmp_float16, 1, 8, 4)
    else:
        tik_instance.vec_conv(1, "floor", roi_y_floor[0], roi_bin_ph_gh_pos_fp32, 1, 8, 8 // dtype_num)

    tmp_verify = tik_instance.Scalar(dtype="int32")
    tmp_verify.set_as(roi_y_floor[0])
    verify.set_as(0)
    with tik_instance.if_scope(tmp_verify < -1):
        verify.set_as(1)
    with tik_instance.if_scope(tmp_verify >= fm_h):
        verify.set_as(1)

    tik_instance.vec_dup(1, const_value_fp32, 0, 1, 0)
    tik_instance.vec_max(1, roi_bin_ph_gh_pos_fp32, roi_bin_ph_gh_pos_fp32, const_value_fp32, 1, 8, 8, 0)

    if vconvf_suppot is False and dtype == "float32":
        tik_instance.vec_conv(1, "", tmp_float16, roi_bin_ph_gh_pos_fp32, 1, 4, 8)
        tik_instance.vec_conv(1, "floor", roi_bin_ph_gh_ly_int32[0], tmp_float16, 1, 8, 4)
    else:
        tik_instance.vec_conv(1, "floor", roi_bin_ph_gh_ly_int32[0], roi_bin_ph_gh_pos_fp32, 1, 8, 4)

    tik_instance.vec_add(1, roi_bin_ph_gh_hy_int32[0], roi_bin_ph_gh_ly_int32[0], const_one, 1, 8, 8, 8)

    tik_instance.vec_dup(1, const_one, fm_h - 1, 1, 0)
    tik_instance.vec_dup(1, const_value_fp32, fm_h - 1, 1, 0)
    tik_instance.vec_min(1, roi_bin_ph_gh_ly_int32, roi_bin_ph_gh_ly_int32, const_one, 1, 8, 8, 0)
    tik_instance.vec_min(1, roi_bin_ph_gh_hy_int32, roi_bin_ph_gh_hy_int32, const_one, 1, 8, 8, 0)
    tik_instance.vec_min(1, roi_bin_ph_gh_pos_fp32, roi_bin_ph_gh_pos_fp32, const_value_fp32, 1, 8, 8, 0)

    if vconvf_suppot is False and dtype == "float32":
        tik_instance.vec_conv(1, "", tmp_float16, roi_bin_ph_gh_ly_int32, 1, 4, 8, 1.0)
        # low level
        tik_instance.vec_conv(1, "", tmp_bin_h_fp32[0], tmp_float16[0], 1, 8, 4)
    else:
        # low level
        if dtype == "float32":
            tik_instance.vec_conv(1, "", tmp_bin_h_fp32[0], roi_bin_ph_gh_ly_int32[0], 1, 8 // dtype_num, 8)
        else:
            tik_instance.vec_conv(1, "", tmp_bin_h_fp32[0], roi_bin_ph_gh_ly_int32[0], 1, 8 // dtype_num, 8, 1.0)

    tik_instance.vec_sub(1, roi_bin_ph_gh_ly_weight[0], roi_bin_ph_gh_pos_fp32, tmp_bin_h_fp32, 1, 8, 0, 8)
    tik_instance.vec_dup(1, const_value_fp32, 1, 1, 0)
    tik_instance.vec_sub(1, roi_bin_ph_gh_hy_weight[0], const_value_fp32[0], roi_bin_ph_gh_ly_weight[0], 1, 8, 0, 8)

    low_y = tik_instance.Scalar(dtype=dtype, init_value=roi_bin_ph_gh_ly_weight[0])
    high_y = tik_instance.Scalar(dtype=dtype, init_value=roi_bin_ph_gh_hy_weight[0])
    return roi_bin_ph_gh_ly_int32, low_y, roi_bin_ph_gh_hy_int32, high_y, verify


# 'pylint: disable=unused-argument,invalid-name
def roi_align_true(feature_map_dict, rois_dict, roisn_dict, output, scale, pool_h, pool_w, sample_ratio, roi_end_mode,
                   kernel_name):
    """
    roi_align_true
    :param feature_map_dict:
    :param rois_dict:
    :param roisn_dict:
    :param output:
    :param scale:
    :param pool_h:
    :param pool_w:
    :param sample_ratio:
    :param roi_end_mode:
    :param kernel_name:
    :return:
    """

    tik_instance = tik.Tik(tik.Dprofile(), True)
    rois_shape = rois_dict.get("shape")
    dtype = feature_map_dict.get("dtype")
    feature_shape = feature_map_dict.get("shape")
    cce_product = tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION)
    feature_map = tik_instance.Tensor(dtype, feature_shape, name="feature_map", scope=tbe_platform.scope_gm)
    rois = tik_instance.Tensor(dtype, rois_shape, name="rois", scope=tbe_platform.scope_gm)
    if roisn_dict:
        roisn_shape = roisn_dict.get("shape")
        roisn_dtype = roisn_dict.get("dtype")
        roisn = tik_instance.Tensor(roisn_dtype, roisn_shape, name="roisn", scope=tbe_platform.scope_gm)
        roisn_ub = tik_instance.Tensor(roisn_dtype, [1, 16], name="roisn_ub", scope=tbe_platform.scope_ubuf)
        tik_instance.data_move(roisn_ub, roisn, 0, 1, 1, 0, 0)

    fm_c1 = feature_shape[1]
    fm_c0 = 16
    proposal_num = rois_shape[0]
    ret = tik_instance.Tensor(dtype, [rois_shape[0], fm_c1, pool_h, pool_w, fm_c0], name="ret",
                              scope=tbe_platform.scope_gm)
    grid_curr_h = tik_instance.Scalar(dtype="int32")
    grid_curr_w = tik_instance.Scalar(dtype="int32")
    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    block_num = rois_shape[0] // core_num
    if rois_shape[0] % core_num != 0:
        block_num = block_num + 1
    if block_num == 0:
        block_num = 1
    if dtype == "float32":
        n_bust = 2
    else:
        n_bust = 1
    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    # every block, process 128 rois
    with tik_instance.for_range(0, (proposal_num + (block_num - 1)) // block_num,
                                block_num=(proposal_num + (block_num - 1)) // block_num) as block_i:

        rois_ub = tik_instance.Tensor(dtype, [128, 8], name="rois_ub", scope=tbe_platform.scope_ubuf)
        proposals_ub_x0 = tik_instance.Tensor(dtype, [128, 1], name="proposals_ub_x0", scope=tbe_platform.scope_ubuf)
        proposals_ub_y0 = tik_instance.Tensor(dtype, [128, 1], name="proposals_ub_y0", scope=tbe_platform.scope_ubuf)
        proposals_ub_x1 = tik_instance.Tensor(dtype, [128, 1], name="proposals_ub_x1", scope=tbe_platform.scope_ubuf)
        proposals_ub_y1 = tik_instance.Tensor(dtype, [128, 1], name="proposals_ub_y1", scope=tbe_platform.scope_ubuf)

        if dtype == "float32":
            tik_instance.vector_dup(64, rois_ub, 0.0, 16, 1, 8)
        else:
            tik_instance.vector_dup(128, rois_ub, 0.0, 8, 1, 8)

        rois_valid = tik_instance.Scalar(dtype="int32", init_value=block_num)
        with tik_instance.if_scope(block_i == ((proposal_num + (block_num - 1)) // block_num - 1)):
            rois_valid.set_as(proposal_num - block_i * block_num)
        with tik_instance.if_scope(rois_valid != 0):
            with tik_instance.for_range(0, (rois_valid + (128 - 1))//128) as roi_128_number:
                rois_valid_in_block = tik_instance.Scalar(dtype="int32", init_value=128)
                with tik_instance.if_scope(roi_128_number == ((rois_valid + (128 - 1))//128 - 1)):
                    rois_valid_in_block.set_as(rois_valid - roi_128_number * 128)

                if (cce_product in (tbe_platform.ASCEND_610, tbe_platform.ASCEND_710)) and (dtype == "float16") \
                        and (rois_shape[1] == 5):
                    roi_fm_index = tik_instance.Tensor(dtype, [128], name="roi_fm_index", scope=tbe_platform.scope_ubuf)
                    _extract_roi_v200(tik_instance, rois, n_bust, block_i, block_num, roi_128_number, proposals_ub_x0,
                                      proposals_ub_y0, proposals_ub_x1, proposals_ub_y1, roi_fm_index, dtype)

                    tik_instance, roi_bin_h_fp32_value, roi_bin_w_fp32_value, proposals_ub_x0, proposals_ub_y0, \
                    grid_w_int32, grid_h_int32, grid_w_fp32, grid_h_fp32, roi_int32_fm_index = \
                        _get_roi_align_perf_scale_for_zero_v200(tik_instance, roi_fm_index,
                                                                proposals_ub_x0,
                                                                proposals_ub_y0,
                                                                proposals_ub_x1,
                                                                proposals_ub_y1,
                                                                scale, pool_h, pool_w,
                                                                sample_ratio,
                                                                dtype, roi_end_mode)
                else:
                    _extract_roi(tik_instance, rois_shape, dtype, rois, block_i, block_num, roi_128_number, n_bust,
                                 rois_ub, cce_product, proposals_ub_x0, proposals_ub_y0, proposals_ub_x1,
                                 proposals_ub_y1)

                    tik_instance, roi_bin_h_fp32_value, roi_bin_w_fp32_value, proposals_ub_x0, proposals_ub_y0, \
                    grid_w_int32, grid_h_int32, grid_w_fp32, grid_h_fp32, roi_int32_fm_index = \
                        _get_roi_align_perf_scale_for_zero(tik_instance, rois_ub,
                                                           proposals_ub_x0,
                                                           proposals_ub_y0,
                                                           proposals_ub_x1,
                                                           proposals_ub_y1,
                                                           scale, pool_h, pool_w,
                                                           sample_ratio,
                                                           dtype, roi_end_mode)
                w_number = 0

                w_number_ub = 0
                ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.UB_30K_SIZE
                feature_map_to_ub_verify = ub_size_bytes // (fm_c1 * feature_shape[2] * feature_shape[3] * 16 *
                                                             n_bust * 2)
                feature_map_to_l1_verify = l1_size // (fm_c1 * feature_shape[2] * feature_shape[3] * 16 * n_bust * 2)

                if feature_map_to_ub_verify == 0 and feature_map_to_l1_verify == 0:
                    w_number_ub = ub_size_bytes // (feature_shape[1] * feature_shape[3] * feature_shape[4] * n_bust * 2)
                if feature_map_to_ub_verify == 0 and feature_map_to_l1_verify == 0 and w_number_ub == 0:
                    if (feature_shape[3] - 1) * n_bust < 65535:
                        w_number = l1_size // (feature_shape[1] * feature_shape[3] * feature_shape[4] * n_bust * 2)

                roi_align_compute(
                    tik_instance, feature_map, ret, proposals_ub_x0,
                    proposals_ub_y0, pool_h, pool_w, dtype, roi_128_number,
                    rois_valid_in_block,
                    feature_shape, grid_curr_h, grid_curr_w, fm_c1, n_bust,
                    block_i, block_num, roi_int32_fm_index, grid_h_int32,
                    grid_w_int32, grid_h_fp32, grid_w_fp32,
                    roi_bin_h_fp32_value,
                    roi_bin_w_fp32_value, w_number,
                    feature_map_to_l1_verify, feature_map_to_ub_verify,
                    w_number_ub)

    if roisn_dict:
        tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[feature_map, rois, roisn], outputs=[ret])
    else:
        tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[feature_map, rois], outputs=[ret])
    return tik_instance
