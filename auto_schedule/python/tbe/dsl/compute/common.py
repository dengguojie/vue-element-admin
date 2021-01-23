# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
common
"""
from __future__ import absolute_import
from te import tvm
import te.platform.cce_params as cce_params


_BLOCK_SIZE = cce_params.BLOCK_REDUCE
_BLOCK_INT8_SIZE = cce_params.BLOCK_REDUCE_INT8


def img2col(input_img,
            col_shape,
            filter_h,
            filter_w,
            pad,
            stride,
            tag=None,
            padding_value=0.0):
    """
    img2col
    """

    # pylint: disable=too-many-locals
    def _img2col_compute(input_img, indices, filter_w, pad, stride,
                         padding_value):
        # fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0
        _, _, fmap_h, fmap_w, _ = input_img.shape
        col_n, col_howo, col_c1, col_hw, col_ww, col_c0 = indices
        stride_h, stride_w = stride
        pad_top, _, pad_left, pad_right = pad

        output_w = (fmap_w.value + pad_left + pad_right - filter_w) \
            // stride_w + 1

        img_n_index = col_n
        img_c1_index = col_c1
        img_h_index = (col_howo // output_w) * stride_h + col_hw
        img_w_index = (col_howo % output_w) * stride_w + col_ww
        img_c0_index = col_c0

        return tvm.select(
            tvm.any(img_h_index < pad_top,
                    img_h_index > fmap_h.value + pad_top - 1,
                    img_w_index < pad_left,
                    img_w_index > fmap_w.value + pad_left - 1),
            tvm.const(padding_value, 'float16'),
            input_img(img_n_index, img_c1_index, img_h_index - pad_top,
                      img_w_index - pad_left, img_c0_index))

    if tag is None:
        tag = 'im2col_row_major'
    return tvm.compute(
        col_shape,
        lambda *indices: _img2col_compute(input_img, indices, filter_w, pad,
                                          stride, padding_value),
        name='im2col_row_major',
        tag=tag,
        attrs={
            'kernel_h': filter_h,
            'kernel_w': filter_w,
            'padding': pad,
            'stride': stride
        })



def im2col_fractal(a_im2col_shape, in_a, dst='ca', tag=None):
    """
    im2col_fractal
    """
    last_dim = in_a.shape[-1]

    # pylint: disable=too-many-locals
    def __im2col_fractal_indices(indices, in_a):
        _, h_w, _, kernel_h, kernel_w, _ = in_a.shape
        if dst == 'ca':
            batch_size, i_1, j_1, i_0, j_0 = indices
        else:
            batch_size, i_1, j_1, j_0, i_0 = indices

        n_index = batch_size
        hw_index = i_1 * _BLOCK_SIZE + i_0
        c1_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) // kernel_h.value
        kh_index = (((j_1 * last_dim + j_0) // last_dim) //
                    kernel_w.value) % kernel_h.value
        kw_index = ((j_1 * last_dim + j_0) // last_dim) % kernel_w.value
        c0_index = (j_1 * last_dim + j_0) % last_dim

        return tvm.select(
            tvm.any(hw_index < 0, hw_index > h_w.value - 1),
            tvm.const(0.0, 'float16'),
            in_a(n_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    if tag is None:
        tag = 'im2col_fractal'
    return tvm.compute(
        a_im2col_shape,
        lambda *indices: __im2col_fractal_indices(indices, in_a),
        name='im2col_fractal',
        tag=tag)
