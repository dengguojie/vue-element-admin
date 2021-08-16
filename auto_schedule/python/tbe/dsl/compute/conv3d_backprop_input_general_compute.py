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
conv3d backprop input general compute.
"""
from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils.errormgr import error_manager_cube as cube_err
from tbe.dsl.compute import cube_util
from tbe.dsl.compute import conv3d_backprop_input_cube_util as conv3d_dx_utils
from tbe.tvm.intrin import abs as tvm_abs


_PAD_MIN = 0
_PAD_MAX = 255


class DeConvPattern(conv3d_dx_utils.CubeDslPattern):  # pylint: disable=R0902
    """
    convolution back propagation

    Parameters
    ----------
    kernel_sizes : shape of weight, [N, D, H, W, C]

    strides :list of strides,
             [stridebatch, strided, strideh, stridew, stridechannel]

    pad: list of padding, [pad_front, pad_tail, pad_top,
                           pad_bottom, pad_left, pad_right]

    output_shape : shape of dE/dX, [N, D, H, W, C]

    dilations : list of dilations,
                [dilations_d, dilations_h, dilations_w]

    kernel_name : name of kernel

    group_dict : information for group

    var_map : information for dynamic shape

    Returns
    -------
    deconv_pattern_instance : instance of deconv pattern
    """

    def __init__(self,  # pylint: disable=R0913
                 kernel_sizes, strides, pad, output_shape, output_dtype,
                 dilations, kernel_name, group_dict, var_map, dsl_flag):
        super(DeConvPattern, self).__init__()
        _, _, kernel_d, kernel_h, kernel_w = kernel_sizes
        self._kernel_d = kernel_d
        self._kernel_h = kernel_h
        self._kernel_w = kernel_w
        _, self._stride_d, self._stride_h, self._stride_w, _ = strides
        self._pad_head, self._pad_tail, self._pad_up, self._pad_down, self._pad_left, self._pad_right = pad
        self.output_shape = output_shape
        self.output_dtype = output_dtype
        self._kernel_name = kernel_name
        self._group_dict = group_dict
        self._var_map = var_map
        self._real_g = group_dict["real_g"]
        self._cin1_g = group_dict["cin1_g"]
        self._cout_g = group_dict["cout_g"]
        self.m_0, _, _ = tbe_platform.CUBE_MKN["float16"]['mac']
        self.dy_d = 0
        _, self._dilate_d, self._dilate_h, self._dilate_w, _ = dilations
        self.op_tag = "conv3d_backprop_input_"
        self.dsl_flag = dsl_flag
        self._flag_load3d_special_case = False

    def generate_a(self, dy_ddr):  # pylint: disable=R0914
        """
        generate dy_col tensor for mad

        Parameters
        ----------
        dy_ddr : 6D dE/dY tensor in ddr

        Returns
        ----------
        dy_col: dE/dY tensor of fractal shape in L0A
        """
        dy_batch, dy_deep, kernel_cout1, dy_h, dy_w, kernel_cout0 = cube_util.shape_to_list(dy_ddr.shape)
        stride_d, stride_h, stride_w = self._stride_d, self._stride_h, self._stride_w
        shape_dy_filling = (dy_batch, dy_deep, kernel_cout1, dy_h * stride_h,
                            dy_w * stride_w, kernel_cout0)
        cout1_filling = self._real_g * self._cout_g // kernel_cout0
        shape_dy_filling_l1 = (dy_batch, dy_deep, cout1_filling, dy_h * stride_h,
                               dy_w * stride_w, kernel_cout0)

        self.dy_d = dy_deep
        if stride_h == 1 and stride_w == 1:
            dy_filling = tvm.compute(shape_dy_filling_l1,
                                     lambda i, j, k, l, m, n: tvm.select(k < kernel_cout1,
                                     dy_ddr(i, j, k, l, m, n)),
                                     name="dy_l1_s1",
                                     tag=self.op_tag + "dy_l1_s1")
        else:
            dy_zero = tvm.compute(
                shape_dy_filling,
                lambda *indice: tvm.convert(0).astype(dy_ddr.dtype),
                name="dy_zero",
                tag=self.op_tag + "dy_zero")
            if self._var_map:
                dy_filling = tvm.compute(
                    shape_dy_filling,
                    lambda batch_idx, dy_deep_idx, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx:
                    tvm.select(tvm.all(ho_idx % stride_h == 0, wo_idx % stride_w == 0),
                               dy_ddr(batch_idx,
                                      dy_deep_idx,
                                      kernel_cout1_idx,
                                      ho_idx // stride_h,
                                      wo_idx // stride_w,
                                      kernel_cout0_idx)),
                    name="dy_filling",
                    tag=self.op_tag + "dy_filling",
                    attrs={"stride_expand": (self._stride_h, self._stride_w)})
                dy_vn = tvm.compute(
                    shape_dy_filling,
                    lambda *indice: dy_zero(*indice) + dy_filling(*indice),
                    name = "dy_vn",
                    tag = self.op_tag + "dy_vn")
            else:
                dy_filling = tvm.compute(
                    shape_dy_filling,
                    lambda batch_idx, dy_deep_idx, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx:
                    tvm.select(tvm.all(ho_idx % stride_h == 0,
                                       wo_idx % stride_w == 0),
                               dy_ddr(batch_idx,
                                      dy_deep_idx,
                                      kernel_cout1_idx,
                                      ho_idx // stride_h,
                                      wo_idx // stride_w,
                                      kernel_cout0_idx),
                               dy_zero(batch_idx,
                                       dy_deep_idx,
                                       kernel_cout1_idx,
                                       ho_idx,
                                       wo_idx,
                                       kernel_cout0_idx)),
                    name="dy_filling",
                    tag=self.op_tag + "dy_filling",
                    attrs={"stride_expand": (self._stride_h, self._stride_w)})
                dy_vn = dy_filling

        kernel_d, kernel_h, kernel_w = self._kernel_d, self._kernel_h, self._kernel_w
        dilate_d, dilate_h, dilate_w = self._dilate_d, self._dilate_h, self._dilate_w
        new_stride = (1, 1, 1)
        new_hw = (dy_deep * stride_d, dy_h * stride_h, dy_w * stride_w)
        new_pad_before = ((kernel_d - 1) * dilate_d - self._pad_head,
                          (kernel_h - 1) * dilate_h - self._pad_up,
                          (kernel_w - 1) * dilate_w - self._pad_left)
        pad_head_before, pad_up_before, pad_left_before = new_pad_before
        _, dx_d, _, dx_h, dx_w, _ = self.output_shape
        dilate_shape = (dilate_d, dilate_h, dilate_w)
        new_pad_after = tuple((o - 1) * s + (k - 1) * d + 1 - i - pb for i, pb, k, s, o, d in
                              zip(new_hw,
                                  new_pad_before,
                                  (kernel_d, kernel_h, kernel_w),
                                  new_stride,
                                  (dx_d, dx_h, dx_w),
                                  dilate_shape))
        pad_tail_after, pad_down_after, pad_right_after = new_pad_after
        
        # stride > 1 ub->l1 may cut
        if stride_h > 1 or stride_w > 1:
            if self._var_map or (pad_down_after < 0 or pad_right_after < 0 or pad_tail_after < 0):
                shape_down_modify = (pad_down_after - tvm_abs(pad_down_after)) // 2
                shape_right_modify = (pad_right_after - tvm_abs(pad_right_after)) // 2
                shape_dy_filling_cut = [dy_batch, dy_deep,
                                        cout1_filling,
                                        dy_h * stride_h + shape_down_modify,
                                        dy_w * stride_w + shape_right_modify,
                                        kernel_cout0]
                dy_l1_attrs = {}
                if 'dedx_h' in self._var_map:
                    info_dy_h = tvm.var("info_dy_h")
                    info_dy_h_value = dy_h * stride_h + shape_down_modify
                    shape_dy_filling_cut[3] = info_dy_h
                    dy_l1_attrs["info_dy_h"] = info_dy_h
                    dy_l1_attrs["info_dy_h_value"] = info_dy_h_value

                if 'dedx_w' in self._var_map:
                    info_dy_w = tvm.var("info_dy_w")
                    info_dy_w_value = dy_w * stride_w + shape_right_modify
                    shape_dy_filling_cut[4] = info_dy_w
                    dy_l1_attrs["info_dy_w"] = info_dy_w
                    dy_l1_attrs["info_dy_w_value"] = info_dy_w_value

                if not dy_l1_attrs:
                    dy_l1_attrs = None
                    
                # cut dy_filling
                dy_filling_l1 = tvm.compute(
                    shape_dy_filling_cut,
                    lambda batch_idx, dy_deep, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx:
                    tvm.select(kernel_cout1_idx < kernel_cout1,
                               dy_vn[batch_idx,
                                     dy_deep,
                                     kernel_cout1_idx,
                                     ho_idx,
                                     wo_idx,
                                     kernel_cout0_idx]),
                    name="dy_l1",
                    tag=self.op_tag + "dy_l1",
                    attrs=dy_l1_attrs)

                pad_down_after = (pad_down_after + tvm_abs(pad_down_after)) // 2
                pad_right_after = (pad_right_after + tvm_abs(pad_right_after)) // 2
                pad_tail_after = (pad_tail_after + tvm_abs(pad_tail_after)) // 2
            else:
                dy_filling_l1 = tvm.compute(
                    shape_dy_filling_l1,
                    lambda batch_idx, dy_deep, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx:
                    tvm.select(kernel_cout1_idx < kernel_cout1,
                               dy_filling[batch_idx,
                                          dy_deep,
                                          kernel_cout1_idx,
                                          ho_idx,
                                          wo_idx,
                                          kernel_cout0_idx]),
                    name="dy_l1",
                    tag=self.op_tag + "dy_l1")

        new_pad = (pad_head_before,
                   pad_tail_after,
                   pad_up_before,
                   pad_down_after,
                   pad_left_before,
                   pad_right_after)

        if not self._var_map:
            for i in new_pad[2:]:
                if int(i) < _PAD_MIN or int(i) > _PAD_MAX:
                    cube_err.raise_err_one_para(
                        'E62006', 'conv3d_backprop_input',
                        'pad value in reverse process of convolution should be in [0,255]')

        pat_conv = conv3d_dx_utils.ConvDslPattern(kernel_h, kernel_w, new_stride, new_pad, dilate_shape)

        if stride_h > 1 or stride_w > 1:
            dy_col = pat_conv.generate_a(dy_filling_l1, self._group_dict, self._var_map, self.op_tag)
        else:
            dy_col = pat_conv.generate_a(dy_filling, self._group_dict, self._var_map, self.op_tag)
        
        self._flag_load3d_special_case = pat_conv.flag_load3d_special_case

        return dy_col

    def generate_b(self, kernels):
        """
        generate w_col tensor for mad

        Parameters
        ----------
        kernels : weight tensor of fractal shape before transformation in ddr

        Returns
        ----------
        w_col: w tensor of fractal shape after transformation in L0B
        """
        w_k1, kernel_cout1, kernel_cout0, w_k0 = list(i.value for i in kernels.shape)
        kernel_h, kernel_w, kernel_d = self._kernel_h, self._kernel_w, self._kernel_d
        if w_k1 % (kernel_h * kernel_w) != 0:
            cube_err.raise_err_specific('conv3d_backprop_input',
                'the reduce axis of weight could not be divided by'
                ' {}*{} '.format(kernel_h, kernel_w))

        real_g = self._real_g
        kernel_cin1 = w_k1 // (kernel_w * kernel_h * real_g * kernel_d)
        shape_w_l1 = (real_g,
                      kernel_d,
                      kernel_cin1 * kernel_h * kernel_w,
                      kernel_cout1,
                      kernel_cout0,
                      w_k0)
        ckk = kernel_cin1 * kernel_h * kernel_w
        w_l1 = tvm.compute(shape_w_l1,
                           lambda g, d, k1, n1, n0, k0:
                           kernels(g * kernel_d * ckk + d * ckk + k1, n1, n0, k0),
                           name="w_l1",
                           tag=self.op_tag + "w_l1")

        shape_w_l0b = (real_g,
                       kernel_d,
                       kernel_cout1 * kernel_h * kernel_w,
                       kernel_cin1,
                       w_k0,
                       kernel_cout0)
        w_col = tvm.compute(
            shape_w_l0b,
            lambda g_idx, w_d, w_k1_idx, kernel_cin1_idx, w_k0_idx, kernel_cout0_idx:
            w_l1[g_idx, w_d,
                 kernel_cin1_idx * kernel_h * kernel_w +
                 (kernel_h * kernel_w - 1 - w_k1_idx % (kernel_h * kernel_w)),
                 w_k1_idx // (kernel_h * kernel_w),
                 kernel_cout0_idx, w_k0_idx],
            name="w_col",
            tag=self.op_tag + "w_col")
        return w_col

    def generate_c(self, dy_col, w_col, tensor_bias=None):  # pylint: disable=W0221,R0914
        """
        generate dx_ddr

        Parameters
        ----------
        dy_col : dE/dY tensor of fractal shape in L0A

        w_col : w tensor of fractal shape after transformation in L0B

        tensor_bias : bias tensor of shape in UB

        Returns
        ----------
        dx_ddr: dx tensor in ddr
        """

        def _add_bias_in_ub(in_tensor0, in_tensor1):
            c_add_vector = tvm.compute(
                in_tensor0.shape,
                lambda *indice: in_tensor0(*indice) +
                                in_tensor1(indice[2], indice[4]),
                name="bias_add_vector")
            return c_add_vector

        c_dtype = "float32"
        if tbe_platform_info.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            c_dtype = "float16"
        dx_col = super(DeConvPattern, self).generate_c(dy_col,
                                                       w_col,
                                                       c_type=c_dtype,
                                                       tag=self.op_tag)

        # mad dx shape
        dx_group, dx_batch, dx_deep, dx_c1, dx_hw, dx_c0 = cube_util.shape_to_list(dx_col.shape)
        # real dx shape
        _, dx_d, dx_cin1, dx_h, dx_w, dx_cin0 = self.output_shape
        out_shape = (dx_batch, dx_deep, dx_cin1, dx_h * dx_w, dx_cin0)

        # float32->output_dtype
        dx_ub = tvm.compute(
            (dx_batch, dx_deep, dx_cin1, dx_hw, dx_c0),
            lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
            dx_col[dx_cin1_idx // self._cin1_g, dx_batch_idx, dx_deep_idx,
                   dx_cin1_idx - dx_cin1_idx // self._cin1_g * self._cin1_g,
                   dx_hw_idx, dx_cin0_idx].astype(self.output_dtype),
            name="c_ub",
            tag=self.op_tag + "c_ub")

        # sd>kd,add0
        if self._stride_d >= self._kernel_d:
            dx_filing_zero = tvm.compute(
                (dx_batch, dx_deep, dx_cin1, dx_hw, dx_c0),
                lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
                tvm.select(
                    tvm.any((dx_deep_idx + self._pad_head -
                            (dx_deep_idx + self._pad_head) // self._stride_d * self._stride_d -
                            self._kernel_d) >= 0,
                            dx_deep_idx +
                            self._pad_head >= self._stride_d * self.dy_d),
                    tvm.const(0, dtype=self.output_dtype)),
                name="dx_filing_zero",
                tag=self.op_tag + "dx_filing_zero"
            )
        else:
            dx_filing_zero = tvm.compute(
                (dx_batch, dx_deep, dx_cin1, dx_hw, dx_c0),
                lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
                tvm.select(
                    dx_deep_idx + self._pad_head >=
                    self._stride_d * (self.dy_d - 1) + self._kernel_d,
                    tvm.const(0, dtype=self.output_dtype)),
                name="dx_filing_zero",
                tag=self.op_tag + "dx_filing_zero"
            )

        dx_ub_vn = tvm.compute(
            (dx_batch, dx_deep, dx_cin1, dx_hw, dx_c0),
            lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
            dx_ub[dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx] +
            dx_filing_zero[dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx],
            name="c_ub_vn",
            tag=self.op_tag + "c_ub_vn"
        )

        if tensor_bias is not None and tensor_bias.dtype == self.output_dtype:
            dx_ub_vn = _add_bias_in_ub(dx_ub_vn, tensor_bias)

        if self.dsl_flag:
            if self._flag_load3d_special_case:
                lambda_express = lambda batch_d_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:\
                                        dx_ub_vn[batch_d_idx // dx_d, batch_d_idx % dx_d,
                                                 dx_cin1_idx, 2 * dx_hw_idx, dx_cin0_idx]
            else:
                lambda_express = lambda batch_d_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:\
                                        dx_ub_vn[batch_d_idx // dx_d, batch_d_idx % dx_d,
                                                 dx_cin1_idx, dx_hw_idx, dx_cin0_idx]

            dx_ub_out = tvm.compute(
                (dx_batch * dx_deep, dx_cin1, dx_h * dx_w, dx_cin0),
                lambda_express,
                name="c_ub_exact_hw",
                tag="c_ub_exact_hw",
                attrs={"output_shape": self.output_shape,
                       "output_dtype": self.output_dtype,
                       "stride_d": self._stride_d,
                       'depth_pad': (self._pad_head, self._pad_tail),
                       'kernels': (self._kernel_d,
                                   self._kernel_h, self._kernel_w),
                       "kernel_name": self._kernel_name,
                       "group_dict": self._group_dict})

            return dx_ub_out

        if self._flag_load3d_special_case:
            dx_ddr = tvm.compute(
                out_shape,
                lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
                dx_ub_vn[dx_batch_idx, dx_deep_idx,
                         dx_cin1_idx, 2 * dx_hw_idx, dx_cin0_idx],
                name="c_ddr",
                tag="c_ddr",
                attrs={"output_shape": self.output_shape,
                       "output_dtype": self.output_dtype,
                       "stride_d": self._stride_d,
                       'depth_pad': (self._pad_head, self._pad_tail),
                       'kernels': (self._kernel_d,
                                   self._kernel_h, self._kernel_w),
                       "kernel_name": self._kernel_name,
                       "group_dict": self._group_dict})
        else:
            dx_ddr = tvm.compute(
                out_shape,
                lambda dx_batch_idx, dx_deep_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
                dx_ub_vn[dx_batch_idx, dx_deep_idx,
                         dx_cin1_idx, dx_hw_idx, dx_cin0_idx],
                name="c_ddr",
                tag="c_ddr",
                attrs={"output_shape": self.output_shape,
                       "output_dtype": self.output_dtype,
                       "stride_d": self._stride_d,
                       'depth_pad': (self._pad_head, self._pad_tail),
                       'kernels': (self._kernel_d,
                                   self._kernel_h, self._kernel_w),
                       "kernel_name": self._kernel_name,
                       "group_dict": self._group_dict})

        return dx_ddr
