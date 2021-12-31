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
import math
from collections import namedtuple

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util

# broadcast should be 16
BRC_STANDARD_BLOCK_SIZE = 16


class GroupDictKeys:
    """
    The keys of group_dict
    """
    groups = "groups"
    g_extend = "g_extend"
    multiple_extend = "multiple_extend"
    dx_c1_extend = "dx_c1_extend"
    dy_c1_extend = "dy_c1_extend"
    dx_c_ori = "dx_c_ori"
    dy_c_ori = "dy_c_ori"
    filter_batch_ori = "filter_batch_ori"
    filter_c_ori = "filter_c_ori"
    filter_ori_format = "filter_ori_format"


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = [0 for i in range(len(shape))]
    j = 0
    for i in shape:
        if isinstance(i, tvm.expr.IntImm):
            tmp[j] = i.value
        else:
            tmp[j] = i
        j += 1
    return tmp


def check_pad_zero(pads):
    """
    check if pad is [0, x, 0, x]
    """
    for pad in pads[::2]:
        if isinstance(pad, (int, tvm.expr.IntImm)) and pad != 0:
            return False
    return True


def ceil_div(num1, num2):
    """
    ceil div
    """
    return (num1 + num2 - 1) // num2


def raise_cube_util_err(msg):
    """
    In common component: cube_util, [%s] % (msg)
    msg for discribe the error info
    the error info only for cube_util's developers
    """
    args_dict = {"errCode": "E60108", "reason": msg}
    msg = error_manager_util.get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def im2col_row_major(
        a_im2col_vm_shape,
        tensor_a,
        kernel_w,
        padding,
        stride,
        compute_dtype,
        opti_h_flag=False,
        tag="",
        dilation=(1, 1),
        offset_x=0,
        slice_offset=0,
        l0a_dma_flag=False,
        load3d_special_multiply=1):
    """
    calculate im2col_row_major tensor
    Parameters
    ----------
    a_im2col_vm_shape : shape of a_im2col_row_major

    tensor_a : feature map

    kernel_w: width of filter

    padding: the padding shape

    stride: the stride value

    dilation: the dilation value

    compute_dtype: dtype of compute result

    offset_x: offset of x
    -------
    Returns : a_im2col_row_major tensor
    """
    def __im2col_row_major_indices(
            indices,
            tensor_a,
            kernel_w,
            padding,
            stride,
            dilation,
            slice_offset=0):
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
        _, _, a_height, a_width, _ = tensor_a.shape
        n_index, hw_index, c1_index, kh_index, kw_index, c0_index = indices
        stride_h, stride_w = stride
        if opti_h_flag:
            stride_h = 1
        dilate_h, dilate_w = dilation
        padding_up, _, padding_left, padding_right = padding
        width_out = (a_width.value + padding_left + padding_right -
                     ((kernel_w - 1) * dilate_w + 1)) // (stride_w) + 1

        h_index = (hw_index // width_out) * stride_h + kh_index * dilate_h
        w_index = (hw_index % width_out) * stride_w + kw_index * dilate_w
        if not l0a_dma_flag:
            return tvm.select(tvm.any(h_index < padding_up,
                                      h_index > a_height.value + padding_up - 1,
                                      w_index < padding_left,
                                      w_index > a_width.value + padding_left - 1),
                              tvm.const(offset_x, compute_dtype),
                              tensor_a(n_index, c1_index, h_index - padding_up + slice_offset,
                                       w_index - padding_left, c0_index))
        return tensor_a(n_index, c1_index, h_index + slice_offset, w_index, c0_index)

    return tvm.compute(a_im2col_vm_shape,
                       lambda *indices: __im2col_row_major_indices(indices, tensor_a, kernel_w, padding,
                                                                   stride, dilation, slice_offset),
                       name="im2col_row_major",
                       tag=tag + "im2col_row_major",
                       attrs={
                           "padding": padding,
                           "dilation": dilation,
                           "l0a_dma_flag": l0a_dma_flag,
                           "load3d_special_multiply": load3d_special_multiply
                       })


def im2col_fractal(a_im2col_shape, tensor_a_row_major, l0a_dma_flag=False):
    """
    calculate im2col_fractal tensor
    Parameters
    ----------
    a_im2col_shape : shape of a_im2col

    tensor_a_row_major : feature map after row major

    config: the config of cube

    compute_dtype: dtype of compute result
    -------
    Returns : a_im2col_fractal tensor
    """
    def __im2col_fractal_indices(indices, tensor_a_row_major):
        """
        calculate im2col_fractal tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        a : feature map

        -------
        Returns : im2col_fractal tvm lambda function
        """
        _, _, _, a_col_k1, a_col_m0, a_col_k0 = a_im2col_shape
        _, a_row_major_hw, _, kernel_h, kernel_w, _ = tensor_a_row_major.shape
        g_index, n_index, m1_index, k1_index, m0_index, k0_index = indices

        hw_index = m1_index * a_col_m0 + m0_index
        k_axis_index = (g_index * a_col_k1 + k1_index) * a_col_k0 + k0_index

        c1_index = ((k_axis_index // a_col_k0) // kernel_w.value) // kernel_h.value

        kh_index = ((k_axis_index // a_col_k0) // kernel_w.value) % kernel_h.value

        kw_index = (k_axis_index // a_col_k0) % kernel_w.value

        c0_index = k_axis_index % a_col_k0

        # dtype is compute_dtype
        if not l0a_dma_flag:
            return tvm.select(
                tvm.any(hw_index < 0, hw_index > a_row_major_hw.value - 1),
                tvm.const(0.0, tensor_a_row_major.dtype),
                tensor_a_row_major(n_index, hw_index, c1_index, kh_index, kw_index, c0_index))
        return tvm.select(
            tvm.all(hw_index >= 0, hw_index < a_row_major_hw.value),
            tensor_a_row_major(n_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    return tvm.compute(a_im2col_shape,
                       lambda *indices: __im2col_fractal_indices(indices, tensor_a_row_major),
                       name='im2col_fractal',
                       tag='im2col_fractal')


def im2col_fractal_3d(
        a_im2col_shape,
        tensor_a_row_major,
        fmap_c1,
        d_out,
        filter_d,
        stride_d,
        cin1_g,
        cyclebuffer_flag,
        tag=""):
    """
    calculate 3d im2col_fractal tensor
    Parameters
    ----------
    a_im2col_shape : shape of a_im2col

    tensor_a_row_major : feature map after row major

    fmap_c1 : channel c1

    d_out : output d

    filter_d : kernel_d

    strided : stride d

    cyclebuffer_flag : whether to do  cyclebuffer
    -------
    Returns : a_im2col_fractal tensor
    """
    def __im2col_fractal_indices(indices, tensor_a_row_major):
        """
        calculate im2col_fractal tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        a : feature map

        -------
        Returns : im2col_fractal tvm lambda function
        """
        _, _, _, _, a_col_m0, a_col_k0 = a_im2col_shape
        _, a_row_major_hw, _, kernel_h, kernel_w, _ = tensor_a_row_major.shape

        g_index, n_index, m1_index, k1_index, m0_index, k0_index = indices
        hw_index = m1_index * a_col_m0 + m0_index

        # ====================== new ====================
        kdc1_index = k1_index // (kernel_h.value * kernel_w.value)
        kd_index = (kdc1_index // cin1_g + (n_index % d_out * stride_d) * cyclebuffer_flag) % filter_d
        cin_index = kdc1_index % cin1_g
        c1_index = kd_index * fmap_c1 + g_index * cin1_g + cin_index

        kh_index = (((k1_index * a_col_k0 + k0_index) // a_col_k0) // kernel_w.value) % kernel_h.value

        kw_index = ((k1_index * a_col_k0 + k0_index) // a_col_k0) % kernel_w.value

        c0_index = (k1_index * a_col_k0 + k0_index) % a_col_k0

        # dtype is compute_dtype
        return tvm.select(tvm.any(hw_index < 0, hw_index > a_row_major_hw.value - 1),
                          tvm.const(0.0, tensor_a_row_major.dtype),
                          tensor_a_row_major(n_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    return tvm.compute(a_im2col_shape,
                       lambda *indices: __im2col_fractal_indices(indices, tensor_a_row_major),
                       name="im2col_fractal",
                       tag=tag + "im2col_fractal")


def im2col_fractal_v2(shape, img2col_para):
    """
    calculate im2col_fractal tensor without tensor row_major
    Parameters
    ----------
    shape : shape of a_im2col

    img2col_para : tensor of fmap, kernel_h, kernel_w, padding, stride,
    fmap_wo, dilation
    -------
    Returns : a_im2col_fractal tensor
    """

    block_size = 16
    fmap, kernel_h, kernel_w, padding, stride, fmap_wo, dilation, c1_extend = img2col_para

    def __im2col_idx(idx):
        group, batch, col_h, col_w, block_size_h, block_size_w = idx

        virtual_h = col_h * block_size + block_size_h
        virtual_w = col_w * block_size + block_size_w

        back_c1 = virtual_w // block_size // kernel_w // kernel_h
        back_h = (virtual_h // fmap_wo) * stride[0] + (col_w // kernel_w % kernel_h)
        back_w = (virtual_h % fmap_wo) * stride[1] + (col_w % kernel_w)
        if len(fmap.shape) == 5:
            return tvm.select(
                tvm.any(back_h < padding[0], back_h > fmap.shape[2] + padding[0] - 1, back_w < padding[2],
                        back_w > fmap.shape[3] + padding[2] - 1), tvm.const(0, fmap.dtype),
                fmap(batch, back_c1 + group * c1_extend, back_h - padding[0], back_w - padding[2], block_size_w))
        return tvm.select(
            tvm.any(back_h < padding[0], back_h > fmap.shape[2] + padding[0] - 1, back_w < padding[2],
                    back_w > fmap.shape[3] + padding[2] - 1), tvm.const(0, fmap.dtype),
            fmap(group, batch, back_c1, back_h - padding[0], back_w - padding[2], block_size_w))

    return tvm.compute(shape,
                       lambda *idx: __im2col_idx(idx),
                       name="img2col_fractal_v2",
                       tag="im2col_fractal_v2",
                       attrs={
                           "fmap_shape": fmap.shape,
                           "kernel_h": kernel_h,
                           "kernel_w": kernel_w,
                           "padding": padding,
                           "stride": stride,
                           "dilation": dilation
                       })


class CubeDslPattern:
    """
    class of cube mmad calculation

    Parameters
    ----------
    None

    Returns
    -------
    cube_pattern_instance : instance of cube mmad pattern
    """

    type_c_map = {}

    def __init__(self):
        self._tensor_c = None

    @staticmethod
    def get_type_c(type_a, type_b, type_bias=None):
        """
        get the data type of mad result tensor

        Parameters
        ----------
        type_a : data type of tensor a

        type_b : data type of tensor b

        type_bias : data type of bias

        Returns
        ----------
        type_c : data type of tensor c
        """
        def cal_hash(tp_a, tp_b, tp_bias):
            return hash(str(tp_a) + str(tp_b) + str(tp_bias))

        if CubeDslPattern.type_c_map == {}:
            CubeDslPattern.type_c_map[cal_hash("uint8", "uint8", None)] = "int32"
            CubeDslPattern.type_c_map[cal_hash("int8", "int8", None)] = "int32"
            CubeDslPattern.type_c_map[cal_hash("float16", "float16", None)] = "float16"
            CubeDslPattern.type_c_map[cal_hash("float16", "float16", "float32")] = "float32"
            CubeDslPattern.type_c_map[cal_hash("float16", "float16", "float16")] = "float16"

        type_c_key = cal_hash(type_a, type_b, type_bias)
        type_c = CubeDslPattern.type_c_map.get(type_c_key)

        return type_c

    def generate_c(
            self, tensor_a, tensor_b, tensor_bias=None, c_type=None, offset_x=0, impl_mode="",
            bias_table_flag=False):
        """
        calculate the mad result tensor

        Parameters
        ----------
        tensor_a : tensor a

        tensor_b : tensor b

        tensor_bias : bias tensor

        c_type : data type of c

        offset_x : offset_x of a

        Returns
        ----------
        tensor_c : mad result tensor
        """
        a_group, a_batch, a_m1, a_k1, a_m0, a_k0 = shape_to_list(tensor_a.shape)
        axis_k0 = tvm.reduce_axis([0, a_k0], name='axis_k0')
        axis_k1 = tvm.reduce_axis([0, a_k1], name='axis_k1')

        _, _, b_n1, b_n0, _ = shape_to_list(tensor_b.shape)

        shape_c = (a_group, a_batch, b_n1, a_m1 * a_m0, b_n0)
        type_c = (c_type if c_type is not None else CubeDslPattern.get_type_c(
            tensor_a.dtype, tensor_b.dtype))

        offset_x = offset_x if is_support_v200() else 0

        if bias_table_flag and tensor_bias is not None:
            tensor_c = tvm.compute(
                shape_c,
                lambda g_index, n_index, co1_index, m_index, co0_index: tvm.sum(
                    ((tensor_a(g_index, n_index, m_index // a_m0, axis_k1, m_index % a_m0, axis_k0) - offset_x) *
                    tensor_b(g_index, axis_k1, co1_index, co0_index, axis_k0)).astype(type_c) +
                    tensor_bias(g_index * b_n1 * b_n0 + co1_index * b_n0 + co0_index),
                    axis=[axis_k1, axis_k0]),
                name="C",
                tag="mad")
            self._tensor_c = tensor_c
            return tensor_c

        tensor_c = tvm.compute(
            shape_c,
            lambda g_index, n_index, co1_index, m_index, co0_index: tvm.sum(
                ((tensor_a(g_index, n_index, m_index // a_m0, axis_k1, m_index % a_m0, axis_k0) - offset_x) *
                 tensor_b(g_index, axis_k1, co1_index, co0_index, axis_k0)).astype(type_c),
                axis=[axis_k1, axis_k0]),
            name="C",
            tag="mad",
            attrs={
                "impl_mode": impl_mode
            })
        if tensor_bias is not None:
            bias_ub_brc_shape = list(shape_c)
            bias_ub_brc_shape[3] = bias_ub_brc_shape[3] // BRC_STANDARD_BLOCK_SIZE
            co_k = tbe_platform.CUBE_MKN[tensor_bias.dtype]["mac"][2]
            bias_ub_brc = tvm.compute(bias_ub_brc_shape,
                                      lambda *indices: tensor_bias(indices[0] * shape_c[2] * co_k + indices[
                                          2] * co_k + indices[4]),
                                      name="bias_ub_brc")
            bias_l0c = tvm.compute(
                shape_c,
                lambda g, i, j, k, l: bias_ub_brc(g, i, j, k // BRC_STANDARD_BLOCK_SIZE, l),
                name="bias_l0c")
            tensor_c = tvm.compute(shape_c,
                                   lambda *indices: bias_l0c(*indices) + tensor_c(*indices),
                                   name="c_add_bias")
        self._tensor_c = tensor_c
        return tensor_c


class ConvDslPattern(CubeDslPattern):
    """
    class of convolution

    Parameters
    ----------
    kernel_h: height of filter

    kernel_w: width of filter

    stride : list of strides, [strideh, stridew]

    pad: list of padding, [pad_up, pad_down, pad_left, pad_right]

    Returns
    -------
    conv_pattern_instance : instance of conv pattern
    """
    def __init__(
            self, kernel_h, kernel_w, stride, pad, dilations, offset_x=0, l0a_dma_flag=False):
        super().__init__()
        self._kernel_h = kernel_h
        self._kernel_w = kernel_w
        self._stride_h, self._stride_w = stride
        self._pad_up, self._pad_down, self._pad_left, self._pad_right = pad
        self._dilate_h, self._dilate_w = dilations
        self._m0 = 16
        self._offset_x = offset_x
        self.l0a_dma_flag = l0a_dma_flag
        self.load3d_special_multiply = 1

    def cal_howo(self, height_in, width_in):
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

        height_out, width_out = list(((i + p_before + p_after - (kernel - 1) * d - 1) // s + 1)
                                     for i, p_before, p_after, kernel, d, s in zip(
                                         new_hw, new_pad_before, new_pad_after, [kernel_h, kernel_w],
                                         [self._dilate_h, self._dilate_w], stride))

        return height_out, width_out

    def generate_a(
        self,
        feature_map,
        g_after,
        c1_extend,
        var_map=None,
        slice_offset=0,
        valid_shape=()):
        """
        calculate im2col_fractal tensor

        Parameters
        ----------
        feature_map : feature map tensor in the shape of NC1HWC0

        g_after : the group after optimization

        c1_extend : the C1 after group extension

        var_map : dict of vars for dynamic shape

        slice_offset : offset of fmap

        valid_shape: valid shape of fmap

        Returns
        -------
        a_col : a_im2col_fractal tensor
        """
        if not var_map:
            var_map = {}
            
        def _is_load3d_special(var_map, h_out, w_hout):
            if (tbe_platform_info.get_soc_spec("SOC_VERSION") not in ("Hi3796CV300CS", "Ascend310")
                and not tbe_platform_info.get_soc_spec("CUBE_VECTOR_SPLIT")
                and not var_map
                and int(h_out) != 1
                and int(w_hout) == 1):
                return True
            return False

        if var_map:
            a_batch, _, a_h, a_w, a_c0 = shape_to_list(feature_map.shape)[-5:]
        else:
            a_batch, _, a_h, a_w, a_c0 = shape_to_list(feature_map.shape)
        if valid_shape:
            a_batch, _, a_h, a_w, a_c0 = valid_shape
        kernel_h, kernel_w = self._kernel_h, self._kernel_w

        new_pad = [self._pad_up, self._pad_down, self._pad_left, self._pad_right]
        stride = [self._stride_h, self._stride_w]

        if "dedy_h" in var_map:
            height_out = var_map.get("dx_h")
        else:
            height_out, _ = self.cal_howo(a_h, a_w)
        if "dedy_w" in var_map:
            width_out = var_map.get("dx_w")
        else:
            _, width_out = self.cal_howo(a_h, a_w)

        if _is_load3d_special(var_map, height_out, width_out):
            self.load3d_special_multiply = 2
            width_out *= self.load3d_special_multiply
            self._pad_right += 1
            new_pad = [self._pad_up, self._pad_down, self._pad_left, self._pad_right]

        if not var_map:
            a_im2col_row_major_shape = (a_batch, height_out * width_out, g_after * c1_extend, kernel_h, kernel_w, a_c0)
            a_row_major = im2col_row_major(a_im2col_row_major_shape,
                                           feature_map,
                                           kernel_w,
                                           padding=new_pad,
                                           stride=stride,
                                           compute_dtype=feature_map.dtype,
                                           dilation=(self._dilate_h, self._dilate_w),
                                           offset_x=self._offset_x,
                                           slice_offset=slice_offset,
                                           l0a_dma_flag=self.l0a_dma_flag,
                                           load3d_special_multiply=self.load3d_special_multiply)

        howo = (height_out * width_out + self._m0 - 1) // self._m0 * self._m0
        a_im2col_fractal_shape = (g_after, a_batch, howo // self._m0, c1_extend * kernel_h * kernel_w,
                                  self._m0, a_c0)
        if not var_map:
            a_col = im2col_fractal(a_im2col_fractal_shape, a_row_major, self.l0a_dma_flag)
        else:
            img2col_para = (feature_map, kernel_h, kernel_w, new_pad, stride, width_out,
                            (self._dilate_h, self._dilate_w), c1_extend)
            a_col = im2col_fractal_v2(a_im2col_fractal_shape, img2col_para)
        return a_col

    def generate_c(
            self, tensor_a, tensor_b, tensor_bias=None, c_type=None, offset_x=0):
        """
        calculate convolution output tensor

        Parameters
        ----------
        tensor_a : tensor a

        tensor_b : tensor b

        tensor_bias : bias tensor

        c_type : data type of c

        offset_x : offset_x of a

        Returns
        ----------
        tensor_c: convolution output tensor
        """
        tensor_c = super().generate_c(tensor_a, tensor_b, tensor_bias, c_type, offset_x)
        row_major = tensor_a.op.input_tensors[0]
        ho_wo = row_major.shape[1].value
        _, _, c_m, _ = shape_to_list(tensor_c.shape)
        m_0 = self._m0
        m_1 = c_m // m_0
        if not ((m_1 - 1) * m_0) < ho_wo <= c_m:
            raise_cube_util_err("HoWo param error!")
        return tensor_c


def is_support_v200():
    """
    check if Ascend610/Ascend615/Ascend710/Hi3796CV300CS version
    ----------

    Returns
    -------
    True:  Ascend610/Ascend615/Ascend710/Hi3796CV300CS version
    False: Other version
    """
    soc_version = tbe_platform_info.get_soc_spec("SOC_VERSION")
    if soc_version in ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS", "SD3403"):
        return True
    return False


def calc_info_of_iter_vars(stage):
    """
    Calcuate information of IterVar.

    Args: stage: Stage of schedule.

    Returns:
    A list of elements that are combinations of IterVar.var and information.
    For example:

    [[i0.inner, IterVar(min=0, extent=3, parent=Parent(var=i0, min=0, extent=6, factor=2, nparts=-1))],
    [i0.outer, IterVar(min=0, extent=2, parent=Parent(var=i0, min=0, extent=6, factor=2, nparts=-1))],
    [i1, (0, 16)]]
    """

    Parent = namedtuple("Parent", "var, min, extent, factor, nparts")
    IterVar = namedtuple("IterVar", "min, extent, parent")

    def calc_split_rel(rel, info_iter_var):
        val_min = (rel.parent.dom.min.value
                   if rel.parent.dom is not None else info_iter_var[rel.parent.var][0])
        extent = (rel.parent.dom.extent.value
                  if rel.parent.dom is not None else info_iter_var[rel.parent.var][1])
        factor = rel.factor.value if rel.factor is not None else -1
        nparts = rel.nparts.value if rel.nparts is not None else -1
        parent = Parent(rel.parent.var, val_min, extent, factor, nparts)
        if factor >= 0:
            return IterVar(val_min, math.ceil(extent / factor), parent), IterVar(val_min, factor, parent)
        if nparts >= 0:
            return IterVar(val_min, nparts, parent), IterVar(val_min, math.ceil(extent / nparts), parent)
        return rel

    def fetch_info_from_relations(info_iter_var):
        for rel in stage.relations:
            if isinstance(rel, tvm.schedule.Split):
                outer, inner = calc_split_rel(rel, info_iter_var)
                info_iter_var[rel.inner.var] = inner
                info_iter_var[rel.outer.var] = outer
            elif isinstance(rel, tvm.schedule.Fuse):
                dom_inner = (IterVar(rel.inner.dom.min.value, rel.inner.dom.extent.value, None)
                             if rel.inner.dom is not None else IterVar(
                                 info_iter_vars[rel.inner.var][0], info_iter_vars[rel.inner.var][1], None))
                dom_outer = (IterVar(rel.outer.dom.min.value, rel.outer.dom.extent.value, None)
                             if rel.outer.dom is not None else IterVar(
                                 info_iter_vars[rel.outer.var][0], info_iter_vars[rel.outer.var][1], None))
                info_iter_var[rel.fused.var] = (dom_inner.min * dom_outer.min,
                                                dom_inner.extent * dom_outer.extent)

    info_iter_vars = {
        iter_var.var: (iter_var.dom.min.value, iter_var.dom.extent.value)
        for iter_var in stage.all_iter_vars if iter_var.dom is not None
    }
    fetch_info_from_relations(info_iter_vars)
    res = [[item.var, (item.dom.min.value, item.dom.extent.value) if item.dom is not None else info_iter_vars[item.var]
            ] for item in stage.leaf_iter_vars]

    return res


def print_iter_vars(iter_vars):
    """
    Pretty print iter_vars.

    Args: iter_vars: List of iter_var.

    Returns: None.
    """

    for i, item in enumerate(iter_vars):
        print(i * 4 * ' ', item)


def is_mini_version():
    """
    check if mini version
    -------

    Returns
    -------
    True: mini version
    False: Other version
    """
    soc_version = tbe_platform_info.get_soc_spec("SOC_VERSION")
    if soc_version in [tbe_platform_info.ASCEND_310]:
        return True
    return False


def is_cloud_version():
    """
    check if cloud version
    -------

    Returns
    -------
    True: cloud version
    False: Other version
    """
    soc_version = tbe_platform_info.get_soc_spec("SOC_VERSION")
    if soc_version in [tbe_platform_info.ASCEND_910, tbe_platform_info.ASCEND_910H,
                       tbe_platform_info.ASCEND_910M, tbe_platform_info.ASCEND_910P]:
        return True
    return False


def is_ng1_version():
    """
    check if mini_ng1 version
    -------

    Returns
    -------
    True: mini_ng1 version
    False: Other version
    """
    soc_version = tbe_platform_info.get_soc_spec("SOC_VERSION")
    if soc_version in [tbe_platform_info.ASCEND_610, tbe_platform_info.ASCEND_615,
                       tbe_platform_info.ASCEND_710, tbe_platform_info.ASCEND_710P]:
        return True
    return False


def is_lhisi_version():
    """
    check if shisi version
    -------

    Returns
    -------
    True: shisi version
    False: Other version
    """
    soc_version = tbe_platform_info.get_soc_spec("SOC_VERSION")
    if soc_version in [tbe_platform_info.HI3796CV300ES, tbe_platform_info.HI3796CV300CS,
                       tbe_platform_info.ASCEND_SD]:
        return True
    return False


def is_lhisi_cs_version():
    """
    check if 3796CS version
    -------

    Returns
    -------
    True: 3796CS version
    False: Other version
    """
    soc_version = tbe_platform_info.get_soc_spec("SOC_VERSION")
    if soc_version in [tbe_platform_info.HI3796CV300CS, tbe_platform_info.ASCEND_SD]:
        return True
    return False


def is_v200_version():
    """
    check if v200 version
    -------

    Returns
    -------
    True: v200 version
    False: Other version
    """
    return is_ng1_version()


def is_v200_version_new():
    """
    check if v200 new version
    -------

    Returns
    -------
    True: v200 new version
    False: Other version
    """
    return is_ng1_version() or is_lhisi_cs_version()


def is_mini_or_lhisi_version():
    """
    check if mini or lhisi version
    -------

    Returns
    -------
    True: mini or lhisi version
    False: Other version
    """
    mini_or_lhisi_version_flag = is_mini_version() or is_lhisi_version()
    return mini_or_lhisi_version_flag
