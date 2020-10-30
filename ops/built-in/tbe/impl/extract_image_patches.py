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
extract_image_patches
"""
import functools
import json
import math
import os
import re
import stat

import te.platform as tbe_platform
from te import tvm
from te.lang import cce as tbe
from te.utils import para_check
from te.utils.error_manager import error_manager_vector

BLOCK_SIZE = 16
BLOCK_SIZE_ALIGN = 16
BLOCK_SIZE_FP16 = 16
BLOCK_SIZE_INT8 = 32

DOUBLE_BUFFER = 2
FP16_SIZE = 2
INT8_SIZE = 1
NEED_UB_SPACE_NUM = 2
SIZE_L1 = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
SIZE_UB = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
LOAD3D_REPEAT_TIME_LIMIT = 255
DELTA = 0.000001  # aviod div zero, fp32 precision


def _ub_split_c1(ub_split_c1_shape, tensor, ksize):
    # pylint: disable=invalid-name
    def _ub_split_c1_indices(indices, tensor):
        n, howo, co1, khw, howo0, co0 = indices
        n_index = n
        hw_index = howo
        hw0_index = howo0
        c1_index = co1 * ksize + khw
        c0_index = co0
        return tensor(n_index, hw_index, c1_index, hw0_index, c0_index)

    return tvm.compute(ub_split_c1_shape, lambda *indices: _ub_split_c1_indices(indices, tensor), name='_ub_split_c1')


def _ub_transpose(ub_transpose_shape, tensor):
    # pylint: disable=invalid-name
    def _ub_transpose_indices(indices, tensor):
        n, howo, howo0, khw, co1, co0 = indices
        n_index = n
        hw_index = howo
        c1_index = co1
        khw_index = khw
        hw0_index = howo0
        c0_index = co0

        return tensor(n_index, hw_index, c1_index, khw_index, hw0_index, c0_index)

    return tvm.compute(ub_transpose_shape,
                       lambda *indices: _ub_transpose_indices(indices, tensor),
                       name='_ub_transpose')


def _ub_merge_hw(ub_merge_shape, tensor):
    # pylint: disable=invalid-name
    def _ub_merge_hw_indices(indices, tensor):
        _, _, in_hw0, _, _, _ = tensor.shape
        n, howo, khw, co1, co0 = indices
        n_index = n
        hw_index = howo // in_hw0
        hw0_index = howo % in_hw0
        c1_index = co1
        khw_index = khw
        c0_index = co0
        return tensor(n_index, hw_index, hw0_index, khw_index, c1_index, c0_index)

    return tvm.compute(ub_merge_shape, lambda *indices: _ub_merge_hw_indices(indices, tensor), name='_ub_merge_hw')


def _ub_merge_co(ub_merge_co_shape, tensor):
    # pylint: disable=invalid-name
    def _ub_merge_co_indices(indices, tensor):
        _, _, _, _, in_c0 = tensor.shape
        n, howo, khw, co = indices
        n_index = n
        hw_index = howo
        khw_index = khw
        c1_index = co // in_c0
        c0_index = co % in_c0
        return tensor(n_index, hw_index, khw_index, c1_index, c0_index)

    return tvm.compute(ub_merge_co_shape, lambda *indices: _ub_merge_co_indices(indices, tensor), name='_ub_merge_co')


def _im2col_row_major_v2(feature_map, im2col_vm_shape, kernel_h, kernel_w, padding, stride, dilate, compute_dtype):
    """
    calculate im2col_row_major tensor
    Parameters
    ----------
    feature_map : feature map

    im2col_vm_shape : shape of A_im2col_row_major

    kernel_h: the kernel value in  h

    kernel_w: the kernel value in  w

    padding: the padding shape

    stride: the stride value

    dilate: the dilation value

    compute_dtype: dtype of compute result
    -------
    Returns : A_im2col_row_major tensor
    """

    # pylint: disable=unused-argument,invalid-name
    def _im2col_row_major_indices(indices, feature_map, kernel_h, kernel_w, padding, stride, dilate):
        """
        calculate im2col_row_major tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        feature_map : feature map

        kernel_h: the kernel value in  h

        kernel_w: the kernel value in  w

        padding: the padding shape

        stride: the stride value

        -------
        Returns  im2col_row_major tvm lambda function
        """
        _, _, in_h, in_w, _ = feature_map.shape

        n, hw, c1, kh, kw, c0 = indices
        stride_h, stride_w = stride
        dilate_h, dilate_w = dilate
        padding_top, _, padding_left, padding_right = padding

        kernel_dilate_w = (kernel_w - 1) * dilate[1] + 1

        width_out = (in_w.value + padding_left + padding_right - kernel_dilate_w) // (stride_w) + 1

        n_index = n
        c1_index = c1
        h_index = (hw // width_out) * stride_h + (kh * dilate_h)
        w_index = (hw % width_out) * stride_w + (kw * dilate_w)
        c0_index = c0
        return tvm.select(
            tvm.any(h_index < padding_top, h_index > in_h.value + padding_top - 1, w_index < padding_left,
                    w_index > in_w.value + padding_left - 1), tvm.const(0.0, compute_dtype),
            feature_map(n_index, c1_index, h_index - padding_top, w_index - padding_left, c0_index))

    return tvm.compute(
        im2col_vm_shape,
        lambda *indices: _im2col_row_major_indices(indices, feature_map, kernel_h, kernel_w, padding, stride, dilate),
        name='im2col_row_major',
        tag='im2col_row_major')


def _im2col_fractal_v2(im2col_shape, feature_map, config, compute_dtype):
    """
    calculate im2col_fractal tensor
    Parameters
    ----------
    im2col_shape : shape of A_im2col

    feature_map : feature map

    config: the config of cube

    compute_dtype: dtype of compute result
    -------
    Returns : A_im2col_fractal tensor
    """

    # pylint: disable=invalid-name
    def _im2col_fractal_indices(indices, feature_map):
        """
        calculate im2col_fractal tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        feature_map : feature map

        -------
        Returns : im2col_fractal tvm lambda function
        """
        block_size = config['mac'][1]
        block_size_m = config['mac'][0]
        _, hw, _, kernel_h, kernel_w, c0 = feature_map.shape
        batch_size, i1, j1, i0, j0 = indices
        n_index = batch_size

        hw_index = i1 * block_size_m + i0

        c1_index = (((j1 * block_size + j0) // c0.value) // kernel_w.value) // kernel_h.value

        kh_index = (((j1 * block_size + j0) // c0.value) // kernel_w.value) % kernel_h.value

        kw_index = ((j1 * block_size + j0) // c0.value) % kernel_w.value

        c0_index = (j1 * block_size + j0) % c0.value

        dtype = compute_dtype
        return tvm.select(tvm.any(hw_index < 0, hw_index > hw.value - 1), tvm.const(0.0, dtype),
                          feature_map(n_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    return tvm.compute(im2col_shape,
                       lambda *indices: _im2col_fractal_indices(indices, feature_map),
                       name='im2col_fractal',
                       tag='im2col_fractal')


# pylint: disable=unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("extract_image_patches")
def extract_image_patches_compute(fmap,
                                  c_in_real,
                                  ksizes,
                                  strides,
                                  dilates,
                                  padding,
                                  kernel_name="extract_image_patches"):
    """
    ops compute

    Parameters
    ----------
    fmap : TVM tensor
        the placeholder of input_x
    c_in_real : real c size of input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    kernel_name : str
    kernel name, default value is "extract_image_patches"

    Returns
    -------
    compute results
    """
    # fmap's format is NC1HWC0
    fmap_shape = fmap.shape
    dtype_input = fmap.dtype
    if dtype_input == "int8" or dtype_input == "uint8":
        align_block_size = BLOCK_SIZE_INT8
    else:
        align_block_size = BLOCK_SIZE

    fmap_n = fmap_shape[0].value
    fmap_c1 = fmap_shape[1].value
    fmap_h = fmap_shape[2].value
    fmap_w = fmap_shape[3].value
    fmap_c0 = fmap_shape[4].value
    # out to L1
    fmap_in_l1 = tvm.compute(fmap_shape, lambda *i: fmap[i], name="fmap_in_l1")

    _, kernel_h, kernel_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates

    out_h, padding_h_before, padding_h_after = tbe.te_compute.common.tf_get_windowed_output_size_verbose_v2(
        fmap_h, kernel_h, dilate_h, stride_h, padding)
    out_w, padding_w_before, padding_w_after = tbe.te_compute.common.tf_get_windowed_output_size_verbose_v2(
        fmap_w, kernel_w, dilate_w, stride_w, padding)

    pad = (padding_h_before, padding_h_after, padding_w_before, padding_w_after)
    stride = (stride_h, stride_w)
    dilate = (dilate_h, dilate_w)

    fmap_vm_shape = (fmap_n, out_h * out_w, fmap_c1, kernel_h, kernel_w, fmap_c0)

    fmap_im2col = _im2col_row_major_v2(fmap_in_l1, fmap_vm_shape, kernel_h, kernel_w, pad, stride, dilate, dtype_input)

    howo = ((out_h * out_w + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    fractal_shape = (fmap_n, howo // BLOCK_SIZE, fmap_c1 * kernel_h * kernel_w, BLOCK_SIZE, align_block_size)

    config = {"mac": [16, align_block_size, 16]}

    fmap_fractal = _im2col_fractal_v2(fractal_shape, fmap_im2col, config, dtype_input)

    extract_params = {
        "out_h": out_h,
        "out_w": out_w,
        "fmap_shape": fmap_shape,
        "c_in_real": c_in_real,
    }
    setfmatrix_dict = {
        "conv_kernel_h": kernel_h,
        "conv_kernel_w": kernel_w,
        "conv_padding_top": padding_h_before,
        "conv_padding_bottom": padding_h_after,
        "conv_padding_left": padding_w_before,
        "conv_padding_right": padding_w_after,
        "conv_stride_h": stride_h,
        "conv_stride_w": stride_w,
        "conv_dilation_h": dilate_h,
        "conv_dilation_w": dilate_w,
        "conv_fm_c": fmap_c1 * fmap_c0,
        "conv_fm_h": fmap_h,
        "conv_fm_w": fmap_w,
    }

    ub_split_c1_shape = (fmap_n, howo // BLOCK_SIZE, fmap_c1, kernel_h * kernel_w, BLOCK_SIZE, align_block_size)
    ub_split_c1_res = _ub_split_c1(ub_split_c1_shape, fmap_fractal, kernel_h * kernel_w)
    ub_transpose_shape = (fmap_n, howo // BLOCK_SIZE, BLOCK_SIZE, kernel_h * kernel_w, fmap_c1, align_block_size)
    ub_transpose_res = _ub_transpose(ub_transpose_shape, ub_split_c1_res)

    ub_merge_hw_shape = (fmap_n, howo, kernel_h * kernel_w, fmap_c1, align_block_size)
    ub_merge_hw_res = _ub_merge_hw(ub_merge_hw_shape, ub_transpose_res)
    ub_merge_co_shape = (fmap_n, howo, kernel_h * kernel_w, fmap_c1 * align_block_size)
    ub_merge_co_res = _ub_merge_co(ub_merge_co_shape, ub_merge_hw_res)
    workspace_shape = (fmap_n, out_h * out_w, kernel_h * kernel_w, fmap_c1 * align_block_size)
    workspace_res = tvm.compute(workspace_shape, lambda *i: ub_merge_co_res[i], name="workspace_res")

    ub_res_shape = (fmap_n, out_h * out_w, kernel_h * kernel_w, fmap_c1 * align_block_size)
    ub_res = tvm.compute(ub_res_shape, lambda *i: workspace_res[i], name="ub_res")

    out_shape = (fmap_n, out_h * out_w, kernel_h * kernel_w, c_in_real)
    output_res = tvm.compute(out_shape,
                             lambda *i: ub_res[i],
                             name="res",
                             attrs={
                                 'extract_params': extract_params,
                                 'setfmatrix_dict': setfmatrix_dict
                             })

    return output_res, workspace_res, workspace_shape


def _get_tiling_param_cut_howo_col(used_ub_size, lcm_out_w, khkw, cut_h_col, fmap_w, fmap_c0, type_size, c_in_real):
    """
    get params for tiling
    """
    # cut howo col
    # ((max_v_ub-1)*khkw+1)*lcm_out_w*BLOCK_SIZE_ALIGN+max_v_ub*lcm_out_w*BLOCK_SIZE_ALIGN <= used_ub_size
    max_v_ub = (used_ub_size // BLOCK_SIZE_ALIGN // lcm_out_w + khkw - 1) // (khkw + 1)
    if max_v_ub > LOAD3D_REPEAT_TIME_LIMIT:
        max_v_ub = LOAD3D_REPEAT_TIME_LIMIT
    max_v_l1 = SIZE_L1 // (cut_h_col * fmap_w * fmap_c0 * type_size * DOUBLE_BUFFER)
    if max_v_ub > max_v_l1:
        max_v_ub = max_v_l1
    if max_v_ub > 1:
        while c_in_real % max_v_ub != 0:
            max_v_ub = max_v_ub - 1
    # cut howo col, move_rate
    # move_rate limit according to mte2 bound
    move_rate = 1 / khkw
    return max_v_ub, move_rate


def _get_tiling_param_cut_howo_row(khkw, fmap_w, fmap_c0, dilated_kernel_h, dilated_kernel_w, stride_h, type_size,
                                   avg_split_ub_size, cut_w_row, cut_h_row, c_in_real):
    # cut howo row
    max_v_ub = avg_split_ub_size // BLOCK_SIZE_ALIGN // BLOCK_SIZE // khkw
    max_v_load3d_limit = LOAD3D_REPEAT_TIME_LIMIT // khkw
    if max_v_ub > max_v_load3d_limit:
        max_v_ub = max_v_load3d_limit
    max_v_l1 = SIZE_L1 // (cut_h_row * fmap_w * fmap_c0 * type_size * DOUBLE_BUFFER)
    if max_v_ub > max_v_l1:
        max_v_ub = max_v_l1
    if max_v_ub > 1:
        while c_in_real % max_v_ub != 0:
            max_v_ub = max_v_ub - 1

    # cut howo row, move_rate
    # move_rate useful move rate while mte2 data move
    double_loaded = dilated_kernel_h // 2 - stride_h
    if double_loaded < 0:
        double_loaded = 0
    slide_dis_h = cut_h_row - dilated_kernel_h + 1
    slide_times_h = slide_dis_h // stride_h + 1
    slide_dis_w = cut_w_row - dilated_kernel_w + 1
    move_rate = slide_dis_w / (slide_times_h * fmap_w) * (1 - double_loaded / cut_h_row)
    return max_v_ub, move_rate


def _get_tiling_param_cut_howo_partial_col(out_w, khkw, fmap_w, type_size, avg_split_ub_size, cut_h_row, c_in_real):
    # cut howo col partially
    max_v_ub = avg_split_ub_size // (khkw * c_in_real * BLOCK_SIZE)
    max_v_load3d_limit = LOAD3D_REPEAT_TIME_LIMIT // khkw
    if max_v_ub > max_v_load3d_limit:
        max_v_ub = 0
    max_v_l1 = SIZE_L1 // (cut_h_row * fmap_w * c_in_real * type_size * DOUBLE_BUFFER)
    if max_v_ub > max_v_l1:
        max_v_ub = max_v_l1
    cut_hw_up_w = (max_v_ub * BLOCK_SIZE + out_w - 1) // out_w * out_w

    # cut howo col partially, move_rate
    # move_rate useful move rate while mte2 data move
    move_rate = max_v_ub * BLOCK_SIZE / (cut_hw_up_w + DELTA)
    return max_v_ub, move_rate


def _get_tiling_param_cut_howo_min(fmap_w, fmap_c0, type_size, avg_split_ub_size, cut_h_row):
    # cut howo khkw c, minimum cut
    max_v_ub = avg_split_ub_size // (1 * BLOCK_SIZE_ALIGN * BLOCK_SIZE)
    if max_v_ub > LOAD3D_REPEAT_TIME_LIMIT:
        max_v_ub = LOAD3D_REPEAT_TIME_LIMIT
    max_v_l1 = SIZE_L1 // (cut_h_row * fmap_w * fmap_c0 * type_size * DOUBLE_BUFFER)
    if max_v_ub > max_v_l1:
        max_v_ub = max_v_l1
    return max_v_ub


def _get_tiling_param(setfmatrix_dict, extract_params, used_ub_size, type_size, avg_split_ub_size):

    out_w = extract_params['out_w']
    fmap_shape = extract_params['fmap_shape']
    c_in_real = extract_params["c_in_real"]
    lcm_out_w = extract_params['lcm_out_w']
    cut_h_col = extract_params['cut_h_col']
    cut_w_row = extract_params['cut_w_row']
    cut_h_row = extract_params['cut_h_row']
    dilated_kernel_h = extract_params['dilated_kernel_h']
    dilated_kernel_w = extract_params['dilated_kernel_w']
    fmap_w = fmap_shape[3].value
    fmap_c0 = fmap_shape[4].value
    kernel_h = setfmatrix_dict['conv_kernel_h']
    kernel_w = setfmatrix_dict['conv_kernel_w']
    stride_h = setfmatrix_dict['conv_stride_h']
    khkw = kernel_h * kernel_w

    max_v_cut_col, move_rate_cut_col = _get_tiling_param_cut_howo_col(used_ub_size, lcm_out_w, khkw, cut_h_col, fmap_w,
                                                                      fmap_c0, type_size, c_in_real)

    max_v_cut_row, move_rate_cut_row = _get_tiling_param_cut_howo_row(khkw, fmap_w, fmap_c0, dilated_kernel_h,
                                                                      dilated_kernel_w, stride_h, type_size,
                                                                      avg_split_ub_size, cut_w_row, cut_h_row,
                                                                      c_in_real)

    max_v_cut_col_p, move_rate_cut_col_p = _get_tiling_param_cut_howo_partial_col(out_w, khkw, fmap_w, type_size,
                                                                                  avg_split_ub_size, cut_h_row,
                                                                                  c_in_real)

    max_v_cut_min = _get_tiling_param_cut_howo_min(fmap_w, fmap_c0, type_size, avg_split_ub_size, cut_h_row)
    return max_v_cut_col, max_v_cut_row, max_v_cut_col_p, max_v_cut_min, move_rate_cut_col, move_rate_cut_row,\
         move_rate_cut_col_p


def _extract_image_patches_schedule(res, sch_list):
    """
    :param res: the multi-results in the operator
    :param sch: schedule list
    """
    sch = sch_list[0]
    setfmatrix_map = res.op.attrs['setfmatrix_dict']
    setfmatrix_dict = {}
    for key, value in setfmatrix_map.items():
        if hasattr(value, "value"):
            setfmatrix_dict[key] = value.value
        else:
            setfmatrix_dict[key] = value

    extract_map = res.op.attrs['extract_params']
    extract_params = {}
    for key, value in extract_map.items():
        if hasattr(value, "value"):
            extract_params[key] = value.value
        else:
            extract_params[key] = value

    out_h = extract_params['out_h']
    out_w = extract_params['out_w']
    fmap_shape = extract_params['fmap_shape']
    c_in_real = extract_params["c_in_real"]
    fmap_c1 = fmap_shape[1].value
    fmap_h = fmap_shape[2].value
    fmap_w = fmap_shape[3].value
    fmap_c0 = fmap_shape[4].value
    kernel_h = setfmatrix_dict['conv_kernel_h']
    kernel_w = setfmatrix_dict['conv_kernel_w']
    dilate_h = setfmatrix_dict['conv_dilation_h']
    dilate_w = setfmatrix_dict['conv_dilation_w']
    stride_h = setfmatrix_dict['conv_stride_h']
    stride_w = setfmatrix_dict['conv_stride_w']

    ub_res = res.op.input_tensors[0]
    workspace_res = ub_res.op.input_tensors[0]
    merge_co_ub = workspace_res.op.input_tensors[0]
    merge_hw_ub = merge_co_ub.op.input_tensors[0]
    transpose_ub = merge_hw_ub.op.input_tensors[0]
    split_c1_ub = transpose_ub.op.input_tensors[0]
    fmap_fractal = split_c1_ub.op.input_tensors[0]
    fmap_im2col = fmap_fractal.op.input_tensors[0]
    fmap_in_l1 = fmap_im2col.op.input_tensors[0]

    sch[fmap_in_l1].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_im2col].set_scope(tbe_platform.scope_cbuf)
    sch[fmap_fractal].set_scope(tbe_platform.scope_ubuf)
    sch[split_c1_ub].set_scope(tbe_platform.scope_ubuf)
    sch[transpose_ub].set_scope(tbe_platform.scope_ubuf)
    sch[merge_hw_ub].set_scope(tbe_platform.scope_ubuf)
    sch[merge_co_ub].set_scope(tbe_platform.scope_ubuf)
    sch[workspace_res].set_scope(tbe_platform.scope_gm)
    sch[ub_res].set_scope(tbe_platform.scope_ubuf)

    dtype_input = ub_res.dtype
    if dtype_input == "int8" or dtype_input == "uint8":
        align_block_size = BLOCK_SIZE_INT8
        type_size = INT8_SIZE
    else:
        align_block_size = BLOCK_SIZE
        type_size = FP16_SIZE

    out_hw_up16 = ((out_h * out_w - 1) // BLOCK_SIZE + 1) * BLOCK_SIZE
    dilated_kernel_h = (kernel_h - 1) * dilate_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilate_w + 1
    lcm_out_w = BLOCK_SIZE // math.gcd(out_w, BLOCK_SIZE) * out_w
    cut_h_col = (BLOCK_SIZE // math.gcd(out_w, BLOCK_SIZE) - 1) * stride_h + 1 + dilated_kernel_h // 2
    if cut_h_col > fmap_h:
        cut_h_col = fmap_h
    # cut_h_col while cut_hw = BLOCK_SIZE
    cut_w_row_s = (BLOCK_SIZE - 1) * stride_w + 1
    cut_h_row_s = (((cut_w_row_s - 1) // fmap_w + 1) - 1) * stride_h + 1
    cut_w_row = cut_w_row_s + dilated_kernel_w - 1
    cut_h_row = cut_h_row_s + dilated_kernel_h - 1
    if lcm_out_w > out_hw_up16:
        lcm_out_w = out_hw_up16

    extract_params['lcm_out_w'] = lcm_out_w
    extract_params['cut_h_col'] = cut_h_col
    extract_params['cut_w_row'] = cut_w_row
    extract_params['cut_h_row'] = cut_h_row
    extract_params['dilated_kernel_h'] = dilated_kernel_h
    extract_params['dilated_kernel_w'] = dilated_kernel_w

    sch[ub_res].buffer_align((1, 1), (1, 1), (1, 1), (1, align_block_size))
    sch[fmap_im2col].buffer_align((1, 1), (out_w, out_w), (1, 1), (1, 1), (1, 1), (1, align_block_size))
    sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, BLOCK_SIZE), (1, align_block_size))

    used_ub_size = SIZE_UB // type_size // DOUBLE_BUFFER
    avg_split_ub_size = used_ub_size // NEED_UB_SPACE_NUM
    howo = out_h * out_w
    khkw = kernel_h * kernel_w
    c_out = khkw * fmap_c1 * fmap_c0

    if c_in_real % align_block_size == 0:
        n_factor = 1
        howo_factor = howo
        khkw_factor = khkw
        c_factor = c_in_real
        max_v = fmap_c1
        max_v_cut_col, max_v_cut_row, max_v_cut_col_p, max_v_cut_min, move_rate_cut_col, move_rate_cut_row,\
            move_rate_cut_col_p = _get_tiling_param(setfmatrix_dict, extract_params, used_ub_size, type_size,\
                 avg_split_ub_size)

        move_rate = move_rate_cut_col
        if move_rate < move_rate_cut_row:
            move_rate = move_rate_cut_row
        if move_rate < move_rate_cut_col_p:
            move_rate = move_rate_cut_col_p

        if lcm_out_w * c_out <= avg_split_ub_size and khkw * fmap_c1 <= LOAD3D_REPEAT_TIME_LIMIT:
            max_v = avg_split_ub_size // lcm_out_w // c_out
            if lcm_out_w * max_v < howo:
                # if True cut n howo else only cut n
                howo_factor = lcm_out_w * max_v
        elif move_rate == move_rate_cut_col and max_v_cut_col > 0:
            # cut howo col
            howo_factor = lcm_out_w
            max_v = max_v_cut_col
            khkw_factor = 1
            c_factor = align_block_size * max_v
        elif move_rate == move_rate_cut_row and max_v_cut_row > 0:
            # cut howo row
            howo_factor = BLOCK_SIZE
            khkw_factor = khkw
            max_v = max_v_cut_row
            c_factor = align_block_size * max_v
        elif move_rate == move_rate_cut_col_p and max_v_cut_col_p > 0:
            # cut howo col partially
            howo_factor = BLOCK_SIZE * max_v_cut_col_p
            c_factor = c_in_real
            khkw_factor = khkw
            max_v = fmap_c1
        else:
            # cut howo khkw c
            howo_factor = BLOCK_SIZE
            max_v = max_v_cut_min
            khkw_factor = 1
            c_factor = align_block_size * max_v

        device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        res_n_inner_outer, res_n_inner = sch[res].split(res.op.axis[0], factor=n_factor)
        res_n_outer_outer, res_n_outer = sch[res].split(res_n_inner_outer, nparts=device_core_num)
        res_howo_outer, res_howo_inner = sch[res].split(res.op.axis[1], factor=howo_factor)
        res_khkw_outer, res_khkw_inner = sch[res].split(res.op.axis[2], factor=khkw_factor)
        res_c_inner_outer, res_c_inner = sch[res].split(res.op.axis[3], factor=align_block_size)
        res_c_outer, res_c_outer_inner = sch[res].split(res_c_inner_outer, factor=c_factor // align_block_size)
        sch[res].reorder(res_n_outer_outer, res_n_outer, res_howo_outer, res_khkw_outer, res_c_outer, res_n_inner,
                         res_c_outer_inner, res_howo_inner, res_khkw_inner, res_c_inner)

        if SIZE_L1 >= fmap_h * fmap_w * fmap_c0 * fmap_c1 * type_size * DOUBLE_BUFFER:
            sch[fmap_im2col].compute_at(sch[res], res_n_outer)
            sch[fmap_in_l1].compute_at(sch[res], res_n_outer)
        elif SIZE_L1 >= cut_h_row * fmap_w * fmap_c0 * fmap_c1 * type_size * DOUBLE_BUFFER\
             and move_rate != move_rate_cut_col:
            sch[fmap_im2col].compute_at(sch[res], res_howo_outer)
            sch[fmap_in_l1].compute_at(sch[res], res_howo_outer)
        elif SIZE_L1 >= cut_h_col * fmap_w * fmap_c0 * fmap_c1 * type_size * DOUBLE_BUFFER\
             and move_rate == move_rate_cut_col:
            sch[fmap_im2col].compute_at(sch[res], res_howo_outer)
            sch[fmap_in_l1].compute_at(sch[res], res_howo_outer)
        else:
            sch[fmap_im2col].compute_at(sch[res], res_c_outer)
            sch[fmap_in_l1].compute_at(sch[res], res_c_outer)

        sch[transpose_ub].compute_at(sch[res], res_c_outer)
        sch[fmap_fractal].compute_at(sch[res], res_c_outer)

        sch[workspace_res].compute_inline()
        sch[ub_res].compute_inline()
        sch[merge_co_ub].compute_inline()
        sch[merge_hw_ub].compute_inline()
        sch[split_c1_ub].compute_inline()

        block = tvm.thread_axis("blockIdx.x")
        sch[res].bind(res_n_outer_outer, block)

        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], tbe_platform.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col.op.axis[0], tbe_platform.SET_FMATRIX, setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], tbe_platform.IM2COL)
        sch[transpose_ub].emit_insn(transpose_ub.op.axis[0], tbe_platform.DMA_COPY)
        sch[res].emit_insn(res_n_inner, tbe_platform.DMA_COPY)
    else:
        c1_factor = align_block_size
        res_n_outer, res_n_inner = sch[res].split(res.op.axis[0], factor=1)
        res_c1_outer, res_c1_inner = sch[res].split(res.op.axis[3], factor=c_in_real)
        sch[ub_res].compute_at(sch[res], res_c1_outer)

        workspace_res_n_outer, workspace_res_n_inner = sch[workspace_res].split(workspace_res.op.axis[0], factor=1)
        workspace_res_howo_outer, workspace_res_howo_inner = sch[workspace_res].split(workspace_res.op.axis[1],
                                                                                      factor=lcm_out_w)
        workspace_res_khkw_outer, workspace_res_khkw_inner = sch[workspace_res].split(workspace_res.op.axis[2],
                                                                                      factor=1)

        workspace_res_c1_outer, workspace_res_c1_inner = sch[workspace_res].split(workspace_res.op.axis[3],
                                                                                  factor=c1_factor)
        sch[workspace_res].reorder(workspace_res_n_outer, workspace_res_howo_outer, workspace_res_khkw_outer,
                                   workspace_res_c1_outer, workspace_res_n_inner, workspace_res_howo_inner,
                                   workspace_res_khkw_inner, workspace_res_c1_inner)

        sch[merge_co_ub].compute_at(sch[workspace_res], workspace_res_c1_outer)
        sch[merge_hw_ub].compute_at(sch[workspace_res], workspace_res_c1_outer)
        sch[transpose_ub].compute_at(sch[workspace_res], workspace_res_c1_outer)
        sch[split_c1_ub].compute_at(sch[workspace_res], workspace_res_c1_outer)

        sch[fmap_fractal].compute_at(sch[workspace_res], workspace_res_c1_outer)
        sch[fmap_im2col].compute_at(sch[workspace_res], workspace_res_howo_outer)
        sch[fmap_in_l1].compute_at(sch[workspace_res], workspace_res_howo_outer)

        if c_in_real > align_block_size:
            sch[workspace_res].compute_at(sch[res], res_n_outer)
            block = tvm.thread_axis("blockIdx.x")
            sch[res].bind(res_n_outer, block)

        sch[split_c1_ub].compute_inline()
        sch[transpose_ub].compute_inline()
        sch[merge_co_ub].compute_inline()

        sch[fmap_in_l1].emit_insn(fmap_in_l1.op.axis[0], tbe_platform.DMA_COPY)
        sch[fmap_im2col].emit_insn(fmap_im2col.op.axis[0], tbe_platform.SET_FMATRIX, setfmatrix_dict)
        sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], tbe_platform.IM2COL)
        sch[split_c1_ub].emit_insn(split_c1_ub.op.axis[0], tbe_platform.DMA_COPY)
        sch[transpose_ub].emit_insn(transpose_ub.op.axis[0], tbe_platform.DMA_COPY)
        sch[merge_hw_ub].emit_insn(merge_hw_ub.op.axis[0], tbe_platform.DMA_COPY)
        sch[merge_co_ub].emit_insn(merge_co_ub.op.axis[0], tbe_platform.DMA_COPY)
        sch[workspace_res].emit_insn(workspace_res_c1_inner, tbe_platform.DMA_COPY)
        sch[ub_res].emit_insn(ub_res.op.axis[3], tbe_platform.DMA_COPY)
        sch[res].emit_insn(res_c1_inner, tbe_platform.DMA_COPY, {"no_overlap": 1})

    sch[fmap_in_l1].double_buffer()
    sch[fmap_im2col].double_buffer()
    sch[fmap_fractal].double_buffer()
    sch[transpose_ub].double_buffer()
    sch[ub_res].double_buffer()


# pylint: disable=too-many-arguments,unused-argument,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def extract_image_patches(images, y, ksizes, strides, dilates, padding, kernel_name="extract_image_patches"):
    """
    calculating data

    Parameters
    ----------
    images : dict
        shape and dtype of input, only support float16
    y : dict
        shape and dtype of output, should be same shape and type as input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    kernel_name : str
        kernel name, default value is "extract_image_patches"

    Returns
    -------
    None
    """
    shape_input_4d = images.get("ori_shape")
    dtype_input = images.get("dtype")
    dtype_input = dtype_input.lower()
    if dtype_input == "int8" or dtype_input == "uint8":
        align_block_size = BLOCK_SIZE_INT8
        type_size = INT8_SIZE
    else:
        align_block_size = BLOCK_SIZE
        type_size = FP16_SIZE
    fmap_n, fmap_h, fmap_w, fmap_c = shape_input_4d
    fmap_c1 = (fmap_c + align_block_size - 1) // align_block_size
    fmap_c0 = align_block_size
    shape_input = (fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0)

    _, kernel_h, kernel_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates
    out_h, _, _ = tbe.te_compute.common.tf_get_windowed_output_size_verbose_v2(fmap_h, kernel_h, dilate_h, stride_h,
                                                                               padding)
    out_w, padding_w_before, padding_w_after = tbe.te_compute.common.tf_get_windowed_output_size_verbose_v2(
        fmap_w, kernel_w, dilate_w, stride_w, padding)

    if (out_h <= 0) or (out_w <= 0):
        error_manager_vector.raise_err_specific_reson(kernel_name, "out_h and out_w can not <= 0!")
    # min cut_h
    dilated_kernel_h = (kernel_h - 1) * dilate_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilate_w + 1
    if (fmap_w + padding_w_before + padding_w_after) <= dilated_kernel_w + stride_w:
        error_manager_vector.raise_err_specific_reson(
            kernel_name, "The size of fmap_w(after pad) <= kernel_w(after dilation) + stride_w is forbidden!")

    if (kernel_h % 2 == 0 and dilate_h > fmap_h):
        error_manager_vector.raise_err_specific_reson(kernel_name, "Get all data from padding is forbidden!")
    cut_h_col = (BLOCK_SIZE // math.gcd(out_w, BLOCK_SIZE) - 1) * stride_h + 1 + dilated_kernel_h // 2
    if cut_h_col > fmap_h:
        cut_h_col = fmap_h

    if cut_h_col * fmap_w * fmap_c0 * type_size * DOUBLE_BUFFER > SIZE_L1:
        error_manager_vector.raise_err_specific_reson(
            kernel_name, "Input size is too large load to L1, while cut h, need size: %d" %
            (cut_h_col * fmap_w * fmap_c0 * type_size * DOUBLE_BUFFER))

    data_input = tvm.placeholder(shape_input, name="data", dtype=dtype_input)
    output_res, workspace_res, _ = extract_image_patches_compute(data_input, fmap_c, ksizes, strides, dilates, padding,
                                                                 kernel_name)
    sch = tvm.create_schedule(output_res.op)
    _extract_image_patches_schedule(output_res, [sch])

    def _write_workspace_info(workspace_list, kernel_name):
        def _write_code(wkspace_dict, fname):
            fname = os.path.realpath(fname)
            if fname.startswith(os.getcwd()):
                if os.path.exists(fname):
                    with open(fname, "r") as f:
                        load_dict = json.load(f)
                    load_dict.update(wkspace_dict)
                    with open(fname, "w") as f:
                        json.dump(load_dict, f, sort_keys=True, indent=4, separators=(',', ':'))

        def _shape_to_list(shape):
            """
            translate tvm.shape to list type in python
            """
            tmp = []
            for i in shape:
                tmp.append(i.value)
            return tmp

        def _get_data_width(dtype):
            m = re.search(r'\d+', dtype)
            if m:
                return int(m.group(0)) // 8
            return 0

        num = len(workspace_list)
        if num:
            shape_list = [_shape_to_list(i.shape) for i in workspace_list]
            total_size = [functools.reduce(lambda x, y: x * y, list_i) for list_i in shape_list]

            total_size = [i * _get_data_width(j.dtype) for i, j in zip(total_size, workspace_list)]
            if not os.path.exists("kernel_meta"):
                os.mkdir("kernel_meta")
                os.chmod("kernel_meta", stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
            wkspace_dict = {"workspace": {"num": num, "size": total_size}}
            _write_code(wkspace_dict, "kernel_meta/" + kernel_name + ".json")

    with tbe_platform.build_config:
        tvm.build(sch, [data_input, output_res, workspace_res], "cce", name=kernel_name)
        if fmap_c % align_block_size != 0:
            _write_workspace_info([workspace_res], kernel_name)
