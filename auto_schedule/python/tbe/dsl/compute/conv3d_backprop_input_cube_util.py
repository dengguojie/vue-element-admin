#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
cube util.
"""
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.dsl.compute import cube_util
from tbe import tvm


def _im2col_row_major(a_im2col_vm_shape,
                      tensor_a,
                      kernel_w,
                      cout_g,
                      padding,
                      stride,
                      compute_dtype,
                      var_map,
                      tag='',
                      special_load3d_flag=False,
                      dilation=(1, 1, 1)):
    """
    calculate im2col_row_major tensor
    Parameters
    ----------
    a_im2col_vm_shape : shape of a_im2col_row_major

    tensor_a : feature map

    kernel_w : width of filter

    cout_g : new filter batch for group

    padding: the padding shape

    stride: the stride value

    compute_dtype: dtype of compute result

    var_map: the parameters for dynamic shape

    tag : tag for different compute stage, '' by default

    dilation: the dilation value, (1, 1, 1) by default
    -------
    Returns : a_im2col_row_major tensor

    """
    def __im2col_row_major_indices(indices, tensor_a, padding_var, stride, dilation):
        """
        calculate im2col_row_major tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        tensor_a : feature map

        kernel_w: width of filter

        padding: the padding shape

        stride: the stride value

        dilation: the dilation value
        -------
        Returns : im2col_row_major tensor
        """
        _, _, _, a_height, a_width, c_0 = tensor_a.shape
        g_index, n_index, deep_index, hw_index, c1_index, kh_index, kw_index, c0_index = indices
        stride_h, stride_w = stride
        _, dilate_h, dilate_w = dilation
        info_padding_up, _, info_padding_left, _ = padding_var

        c1_index = g_index * (cout_g // c_0) + c1_index
        h_index = (hw_index // width_out) * stride_h + kh_index * dilate_h
        w_index = (hw_index % width_out) * stride_w + kw_index * dilate_w

        return tvm.select(tvm.any(h_index < info_padding_up,
                                  h_index > a_height + info_padding_up - 1,
                                  w_index < info_padding_left,
                                  w_index > a_width + info_padding_left - 1),
                          tvm.const(0.0, compute_dtype),
                          tensor_a(n_index,
                                   deep_index,
                                   c1_index,
                                   h_index - info_padding_up,
                                   w_index - info_padding_left,
                                   c0_index))

    info_padding_up = tvm.var("info_padding_up")
    info_padding_bottom = tvm.var("info_padding_bottom")
    info_padding_left = tvm.var("info_padding_left")
    info_padding_right = tvm.var("info_padding_right")
    if var_map:
        padding_var = [info_padding_up, info_padding_bottom, info_padding_left, info_padding_right]
        width_out = tvm.var("width_out")
    else:
        padding_var = padding
        width_out = (tensor_a.shape[-2] + padding[-1] + padding[-2] -
                     ((kernel_w - 1)*dilation[-1] + 1)) // stride[-1] + 1

    return tvm.compute(a_im2col_vm_shape,
                       lambda *indices: __im2col_row_major_indices(
                           indices, tensor_a, padding_var, stride, dilation),
                       name='im2col_row_major',
                       tag=tag + 'im2col_row_major',
                       attrs={'padding': padding, "dilation":dilation,
                              'padding_var': padding_var, 'width_out_var': width_out,
                              'special_load3d_flag': special_load3d_flag})


def _im2col_fractal(a_im2col_shape, tensor_a_row_major, tag=''):
    """
    calculate im2col_fractal tensor
    Parameters
    ----------
    a_im2col_shape : shape of a_im2col

    tensor_a_row_major : feature map after row major

    tag : tag for different compute stage, '' by default
    -------
    Returns : a_im2col_fractal tensor
    """
    def __im2col_fractal_indices(indices, tensor_a_row_major):
        """
        calculate im2col_fractal tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        tensor_a_row_major : feature map

        -------
        Returns : im2col_fractal tvm lambda function
        """
        _, _, _, _, _, a_col_m0, _ = a_im2col_shape
        _, _, _, a_row_major_hw, _, kernel_h, kernel_w, _ = tensor_a_row_major.shape
        g_index, n_index, deep_index, m1_index, k1_index, m0_index, k0_index = indices

        hw_index = m1_index * a_col_m0 + m0_index

        c1_index = k1_index // kernel_w // kernel_h
        kh_index = k1_index // kernel_w % kernel_h

        kw_index = k1_index % kernel_w

        c0_index = k0_index

        return tvm.select(tvm.any(hw_index < 0, hw_index >
                                  a_row_major_hw - 1),
                          tvm.const(0.0, tensor_a_row_major.dtype),
                          tensor_a_row_major(g_index,
                                             n_index,
                                             deep_index,
                                             hw_index,
                                             c1_index,
                                             kh_index,
                                             kw_index,
                                             c0_index))

    return tvm.compute(a_im2col_shape,
                       lambda *indices:
                       __im2col_fractal_indices(indices, tensor_a_row_major),
                       name='im2col_fractal',
                       tag=tag + 'im2col_fractal')


class CubeDslPattern:
    """
    cube mmad calculation

    Parameters
    ----------
    None

    Returns
    -------
    cube_pattern_instance : instance of cube mmad pattern
    """

    type_c_map = {}

    def __init__(self):
        pass

    def generate_c(self, tensor_a, tensor_b, c_type, tag=""):
        """
        calculate the mad result tensor

        Parameters
        ----------
        tensor_a : tensor a

        tensor_b : tensor b

        tensor_bias : bias tensor

        Returns
        ----------
        tensor_c : mad result tensor
        """
        def __mad_condition(indices, axis_kd, axis_k1, axis_k0, tensor_a, tensor_b):
            g_index, n_index, deep_index, co1_index, m_index, co0_index = indices
            tensor_c = tvm.select(
                tvm.all(
                    (deep_index - axis_kd + pad_head) >= 0,
                    (deep_index - axis_kd + pad_head) % stride_d == 0,
                    (deep_index - axis_kd
                     + pad_head) // stride_d < tensor_a.shape[2]),
                tensor_a(g_index, n_index,
                         (deep_index - axis_kd + pad_head) // stride_d,
                         m_index // a_m0, axis_k1, m_index % a_m0,
                         axis_k0).astype(type_c) *
                tensor_b(g_index, axis_kd, axis_k1, co1_index, co0_index,
                         axis_k0).astype(type_c),
                tvm.const(0.0, type_c))
            return tensor_c

        def __mad_condition_stride1(indices, axis_kd, axis_k1, axis_k0, tensor_a, tensor_b):
            g_index, n_index, deep_index, co1_index, m_index, co0_index = indices
            tensor_c = tvm.select(
                tvm.all((deep_index - axis_kd + pad_head) >= 0,
                        (deep_index - axis_kd + pad_head) < tensor_a.shape[2]),
                tensor_a(g_index, n_index, (deep_index - axis_kd + pad_head),
                         m_index // a_m0, axis_k1, m_index % a_m0,
                         axis_k0).astype(type_c) *
                tensor_b(g_index, axis_kd, axis_k1, co1_index, co0_index,
                         axis_k0).astype(type_c),
                tvm.const(0.0, type_c))
            return tensor_c

        def __mad_condition_noverlap(indices, axis_k1, axis_k0, tensor_a, tensor_b):
            g_index, n_index, deep_index, co1_index, m_index, co0_index = indices
            tensor_c = tvm.select(
                tvm.all((deep_index + pad_head) % stride_d < kernel_d,
                        (deep_index + pad_head) // stride_d < tensor_a.shape[2]),
                tensor_a(g_index, n_index, (deep_index + pad_head) // stride_d,
                         m_index // a_m0, axis_k1, m_index % a_m0,
                         axis_k0).astype(type_c) *
                tensor_b(g_index, (deep_index + pad_head) % stride_d, axis_k1,
                         co1_index, co0_index, axis_k0).astype(type_c),
                tvm.const(0.0, type_c))
            return tensor_c

        def __conv3d_backprop_input_mad(indices, tensor_a, tensor_b):
            tensor_c = tvm.sum(__mad_condition(indices, axis_kd, axis_k1,
                                               axis_k0, tensor_a, tensor_b),
                               axis=[axis_kd, axis_k1, axis_k0])
            return tensor_c

        def __conv3d_backprop_input_mad_stride1(indices, tensor_a, tensor_b):
            tensor_c = tvm.sum(__mad_condition_stride1(indices,
                                                       axis_kd, axis_k1,
                                                       axis_k0, tensor_a,
                                                       tensor_b),
                               axis=[axis_kd, axis_k1, axis_k0])
            return tensor_c

        def __conv3d_backprop_input_mad_noverlap(indices, tensor_a, tensor_b):
            tensor_c = tvm.sum(__mad_condition_noverlap(indices,
                                                        axis_k1,
                                                        axis_k0, tensor_a,
                                                        tensor_b),
                               axis=[axis_k1, axis_k0])
            return tensor_c

        a_group, a_batch, a_deep, a_m1, a_k1, a_m0, a_k0 = cube_util.shape_to_list(tensor_a.shape)
        axis_k0 = tvm.reduce_axis([0, a_k0], name='axis_k0')
        axis_k1 = tvm.reduce_axis([0, a_k1], name='axis_k1')
        _, b_kd, _, b_n1, b_n0, _ = [i.value for i in tensor_b.shape]
        axis_kd = tvm.reduce_axis([0, b_kd], name='axis_kd')
        pad_head, pad_tail = self._pad_head, self._pad_tail
        stride_d = self._stride_d
        output_depth = self.output_shape[1]
        kernel_d = self._kernel_d
        shape_c = (a_group, a_batch, output_depth, b_n1, a_m1 * a_m0, b_n0)
        type_c = c_type

        if stride_d == kernel_d and (output_depth + pad_head
                                     + pad_tail) == a_deep * stride_d:
            tensor_c = tvm.compute(
                shape_c,
                lambda g_index, n_index, deep_index, co1_index, m_index, co0_index:
                tvm.sum(
                    (tensor_a(g_index, n_index, (deep_index + pad_head) // stride_d,
                              m_index // a_m0, axis_k1,
                              m_index % a_m0, axis_k0) *
                     tensor_b(g_index, (deep_index + pad_head) % stride_d, axis_k1,
                              co1_index, co0_index,
                              axis_k0)).astype(type_c),
                    axis=[axis_k1, axis_k0]),
                name="C",
                tag=tag + "mad")
        elif kernel_d <= stride_d:
            tensor_c = tvm.compute(
                shape_c,
                lambda *indices: __conv3d_backprop_input_mad_noverlap(
                    indices, tensor_a, tensor_b),
                name="C",
                tag=tag + "mad")
        elif stride_d == 1:
            tensor_c = tvm.compute(
                shape_c,
                lambda *indices: __conv3d_backprop_input_mad_stride1(
                    indices, tensor_a, tensor_b),
                name="C",
                tag=tag + "mad")
        else:
            tensor_c = tvm.compute(
                shape_c,
                lambda *indices: __conv3d_backprop_input_mad(
                    indices, tensor_a, tensor_b),
                name="C",
                tag=tag + "mad")
        return tensor_c


class ConvDslPattern(CubeDslPattern):
    """
    ConvDslPattern

    Parameters
    ----------
    kernel_h: height of filter

    kernel_w: width of filter

    stride : list of strides, [strided, strideh, stridew]

    pad: list of padding, [pad_up, pad_down, pad_left, pad_right]

    dilation: list of dilation, [dilationd, dilationh, dilationw]

    Returns
    -------
    conv_pattern_instance : instance of conv pattern
    """

    def __init__(self, kernel_h, kernel_w, stride, pad, dilation):
        super(ConvDslPattern, self).__init__()
        self._kernel_h = kernel_h
        self._kernel_w = kernel_w
        self._stride_d, self._stride_h, self._stride_w = stride
        self._pad_head, self._pad_tail, self._pad_up, self._pad_down, self._pad_left, self._pad_right = pad
        self._dilate_d, self._dilate_h, self._dilate_w = dilation
        self._m0 = 16
        self.flag_load3d_special_case = False

    def _cal_howo(self, height_in, width_in):
        """
        calculate the height and width of convolution output tensor

        Parameters
        ----------
        height_in : height of input tensor

        width_in : width of input tensor

        Returns
        ----------
        height_out : height of output tensor

        width_out : width of output tensor
        """
        new_hw = [height_in, width_in]
        kernel_h, kernel_w = self._kernel_h, self._kernel_w
        new_pad_before = (self._pad_up, self._pad_left)
        new_pad_after = (self._pad_down, self._pad_right)
        stride = [self._stride_h, self._stride_w]
        dilation = [self._dilate_h, self._dilate_w]

        height_out, width_out = [
            ((i + p_before + p_after - ((kernel - 1) * d + 1)) // s + 1)
            for i, p_before, p_after, kernel, s, d in
            zip(new_hw,
                new_pad_before,
                new_pad_after,
                [kernel_h, kernel_w],
                stride,
                dilation)]
        return height_out, width_out

    def _get_special_load3d_flag(self, var_map, h_out, w_out):
        """
        get special_load3d_flag

        Parameters
        ----------
        var_map : the parameters for dynamic shape, {} by default

        h_out : height of output tensor

        w_out : width of output tensor

        Returns
        ----------
        w_out : width of output tensor
        """
        if tbe_platform_info.get_soc_spec("SOC_VERSION") not in ("Hi3796CV300CS", "Ascend310") \
            and not var_map \
            and int(h_out) != 1 \
            and int(w_out) == 1:
            w_out += 1
            self._pad_right += 1
            self.flag_load3d_special_case = True
        return w_out

    def generate_a(self, feature_map, group_dict, var_map={}, tag=""):
        """
        calculate im2col_fractal tensor

        Parameters
        ----------
        feature_map : feature map tensor in the shape of NDC1HWC0

        group_dict : the information needed for group convolution

        var_map : the parameters for dynamic shape, {} by default

        tag : the tag of tensor, None by default

        Returns
        -------
        a_col : a_im2col_fractal tensor
        """
        a_batch, a_deep, a_c1, a_h, a_w, a_c0 = cube_util.shape_to_list(feature_map.shape)
        kernel_h, kernel_w = self._kernel_h, self._kernel_w

        new_pad = [self._pad_up, self._pad_down,
                   self._pad_left, self._pad_right]
        stride = [self._stride_h, self._stride_w]
        dilation = [self._dilate_d, self._dilate_h, self._dilate_w]
        a_group = group_dict["real_g"]
        cout_g = group_dict["cout_g"]
        a_c1 = cout_g // a_c0

        if "dedy_w" not in var_map:
            _, width_out = self._cal_howo(a_h, a_w)
        else:
            width_out = var_map.get("dedx_w")

        if "dedy_h" not in var_map:
            height_out, _ = self._cal_howo(a_h, a_w)
        else:
            height_out = var_map.get("dedx_h")

        width_out = self._get_special_load3d_flag(var_map, height_out, width_out)
        new_pad = [self._pad_up, self._pad_down,
                    self._pad_left, self._pad_right]

        a_im2col_row_major_shape = (a_group,
                                    a_batch,
                                    a_deep,
                                    height_out * width_out,
                                    cout_g // a_c0,
                                    kernel_h,
                                    kernel_w,
                                    a_c0)

        a_im2col_fractal_shape = (a_group,
                                  a_batch,
                                  a_deep,
                                  (height_out * width_out + self._m0 - 1) // self._m0,
                                  a_c1 * kernel_h * kernel_w,
                                  self._m0,
                                  a_c0)

        a_row_major = _im2col_row_major(a_im2col_row_major_shape,
                                        feature_map,
                                        kernel_w,
                                        cout_g,
                                        padding=new_pad,
                                        stride=stride,
                                        compute_dtype=feature_map.dtype,
                                        var_map=var_map,
                                        dilation=dilation,
                                        tag=tag,
                                        special_load3d_flag=self.flag_load3d_special_case)

        a_col = _im2col_fractal(a_im2col_fractal_shape, a_row_major, tag=tag)

        return a_col
