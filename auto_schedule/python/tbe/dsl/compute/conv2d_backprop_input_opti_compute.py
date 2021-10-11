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
conv2d_backprop_input_opti_compute
"""
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.compute import cube_util
from tbe.dsl.compute.util import int_ceil_div
from tbe.tvm import api as tvm


# broadcast should be 16
BRC_STANDARD_BLOCK_SIZE = 16


class DeConvKernelSize1Pattern(cube_util.CubeDslPattern):  # pylint:disable=R0902
    """
    class of convolution back propagation for kernelsize1 pattern

    Parameters
    ----------
    kernel_sizes : shape of weight, [N, C, H, W]

    strides : list of strides, [strideh, stridew]

    pad: list of padding, [pad_up, pad_down, pad_left, pad_right]

    output_shape : shape of dE/dX, [N, C, H, W]

    fusion_para : parameters of l1 fusion

    var_map : dict of vars for dynamic shape

    kernel_name : kernel name of operator

    offset_x : offset_x of x

    group_dict : The params of group convolution.

    Returns
    -------
    deconv_pattern_instance : instance of deconv pattern
    """

    fusion_para = {}
    dedy = None

    def __init__(  # pylint:disable=R0913,W0613
        self,
        kernel_size,
        strides,
        pad,
        output_shape,
        output_dtype,
        fusion_para,
        kernel_name,
        offset_x,
        group_dict,
        var_map,
        pooling_mode
    ):
        super().__init__()
        _, _, kernel_h, kernel_w = kernel_size
        stride_h, stride_w = strides
        if not (kernel_h == 1 and kernel_w == 1):
            dict_args = dict()
            dict_args["errCode"] = "E60000"
            dict_args["param_name"] = "kernel_h] and [kernel_w"
            dict_args["expected_value"] = "1] and [1"
            dict_args["input_value"] = "%d] and [%d" % (kernel_h, kernel_w)
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if not (kernel_h <= stride_h and kernel_w <= stride_w):
            dict_args = dict()
            dict_args["errCode"] = "E60108"
            dict_args["reason"] = "not match stride HW[{}, {}],kernel HW[{}, {}]".format(stride_h,
                                    stride_w, kernel_h, kernel_w)
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        self._m0 = tbe_platform.CUBE_MKN["float16"]["mac"][0]
        self._output_shape = output_shape
        self.output_dtype = output_dtype
        self._stride_h, self._stride_w = strides
        _, _, self._kernel_h, self._kernel_w = kernel_size
        self._kernel_name = kernel_name
        self._img_h, self._img_w = [0, 0]
        self._fusion_para = fusion_para
        self._var_map = var_map
        self._offset_x = offset_x
        self._pad = pad
        self._group_dict = group_dict
        self._g_extend = self._group_dict.get(cube_util.GroupDictKeys.g_extend)
        self._extend = self._group_dict.get(cube_util.GroupDictKeys.multiple_extend)
        self._dx_c1_extend = self._group_dict.get(cube_util.GroupDictKeys.dx_c1_extend)
        self._dy_c1_extend = self._group_dict.get(cube_util.GroupDictKeys.dy_c1_extend)
        self._groups_ori = self._group_dict.get(cube_util.GroupDictKeys.groups)
        self._cube_vector_split_flag = tbe_platform_info.get_soc_spec("CUBE_VECTOR_SPLIT")
        self.pooling_mode = pooling_mode

    def _get_dilate_tensor(  # pylint:disable=R0913,R0914
        self,
        raw_tensor,
        out_shape_h,
        out_shape_w,
        dilate_h,
        dilate_w
    ):
        (shape_n, shape_c1, _, shape_c0) = raw_tensor.shape
        new_h = out_shape_h  # dilate_h * (shape_h - 1) + 1
        new_w = out_shape_w  # dilate_w * (shape_w - 1) + 1
        dilate_shape = (shape_n, shape_c1, new_h * new_w, shape_c0)
        dx_zero = tvm.compute(
            dilate_shape,
            lambda *indice: tvm.convert(self._offset_x).astype(raw_tensor.dtype),
            name=raw_tensor.name + "_dx_zero",
            tag="init_zero"
        )
        if self._var_map:
            # because tvm.select not support dynamic shape
            dilate_tensor = tvm.compute(
                dilate_shape,
                lambda n, c1, hw, c0: raw_tensor[
                    n,
                    c1,
                    ((hw // new_w) // dilate_h) * self._img_w + (hw % new_w // dilate_w),
                    c0
                ] + dx_zero[n, c1, hw, c0],
                name="dx_dilation",
                tag="dx_dilation",
                attrs={
                    "dilate": [dilate_h, dilate_w],
                    "out_hw": [out_shape_h, out_shape_w],
                    "img_w": self._img_w
                }
            )
            dx_vn = tvm.compute(
                dilate_shape,
                lambda *indice: dx_zero(*indice) + dilate_tensor(*indice),
                name=raw_tensor.name + "_dilation",
                tag="conv2d_backprop_input_opti",
                attrs={
                    "dilate": [dilate_h, dilate_w],
                    "out_hw": [out_shape_h, out_shape_w],
                    "img_w": self._img_w
                }
            )
            dilate_tensor_res = dx_vn
        else:
            dilate_tensor = tvm.compute(
                dilate_shape,
                lambda n, c1, hw, c0: tvm.select(
                    tvm.all(
                        (hw // new_w) % dilate_h == 0, (hw % new_w) % dilate_w == 0
                    ),
                    raw_tensor[
                        n,
                        c1,
                        ((hw // new_w) // dilate_h) * self._img_w
                        + (hw % new_w // dilate_w),
                        c0
                    ]
                    + dx_zero[n, c1, hw, c0],
                    dx_zero[n, c1, hw, c0]
                ),
                name=raw_tensor.name + "_dilation",
                tag="conv2d_backprop_input_opti",
                attrs={
                    "dilate": [dilate_h, dilate_w],
                    "out_hw": [out_shape_h, out_shape_w],
                    "img_w": self._img_w
                }
            )
            dilate_tensor_res = dilate_tensor
        return dilate_tensor_res

    def generate_a(self, dedy):  # pylint:disable=R0914
        """
        generate dedy_col_fractal tensor for mad

        Parameters
        ----------
        dedy: 5D dE/dY tensor in ddr

        Returns
        ----------
        dedy_col_fractal: dE/dY tensor of fractal shape in L0A
        """

        DeConvKernelSize1Pattern.fusion_para = self._fusion_para
        DeConvKernelSize1Pattern.dedy = dedy
        batch_dim, co1_dim, ho_dim, wo_dim, co0_dim = cube_util.shape_to_list(dedy.shape)

        self._img_h = ho_dim
        self._img_w = wo_dim
        hw_dim = int_ceil_div(wo_dim * ho_dim, self._m0)
        real_hwdim = ho_dim * wo_dim
        input_mem = self._fusion_para.get("input_memory_type")
        l1_fusion_type = self._fusion_para.get("l1_fusion_type")

        # select_read_from_l1_flag  L1 in select read
        from_l1_flag = bool(input_mem == 1 and l1_fusion_type != -1)
        # ddr in select read
        from_ddr_flag = bool(input_mem == 0 and l1_fusion_type != -1)
        if from_l1_flag:
            pat_conv = cube_util.ConvDslPattern(
                kernel_h=1,
                kernel_w=1,
                stride=[1, 1],
                pad=[0, 0, 0, 0],
                dilations=(1, 1),
                offset_x=0
            )
            dedy_col = pat_conv.generate_a(dedy, self._g_extend, self._dy_c1_extend)
        elif from_ddr_flag:
            shape = (batch_dim, co1_dim, ho_dim, wo_dim, co0_dim)
            dedy_col = tvm.compute(
                shape,
                lambda *indices: dedy(*indices),
                name=dedy.name + "_col"
            )
            pat_conv = cube_util.ConvDslPattern(
                kernel_h=1,
                kernel_w=1,
                stride=[1, 1],
                pad=[0, 0, 0, 0],
                dilations=(1, 1),
                offset_x=0
            )
            dedy_col = pat_conv.generate_a(dedy_col, self._g_extend, self._dy_c1_extend)
        else:
            # dma copy from DDR(NCHW) to L1(n,c1,h*w,c0),C=groups*C_ori
            shape = (batch_dim, co1_dim, real_hwdim, co0_dim)
            dedy_col = tvm.compute(
                shape,
                lambda n, co1, m1, co0: dedy(n, co1, tvm.floordiv(m1, wo_dim), tvm.floormod(m1, wo_dim), co0),
                name=dedy.name + "_col"
            )

            def __im2col_fractal_indices(indices,  # pylint: disable=R0914
                                         dedy_tensor_l1):
                """
                calculate im2col_fractal tvm lambda function
                :param indices:
                :param tensor_a_row_major:
                :return:
                """
                l0_g_index, l0_n_index, l0_m1_index, l0_co1_index, l0_m0_index, l0_co0_index = indices
                l1_hw_index = l0_m1_index * self._m0 + l0_m0_index
                l1_c1_index = l0_g_index * self._dy_c1_extend + l0_co1_index
                return tvm.select(l1_c1_index < co1_dim,
                                  dedy_tensor_l1(
                                      l0_n_index,
                                      l1_c1_index,
                                      l1_hw_index,
                                      l0_co0_index))

            def __im2col_fractal_indices_dynamic(indices,  # pylint: disable=R0914
                                                 dedy_tensor_l1):
                """
                calculate im2col_fractal tvm lambda function
                :param indices:
                :param tensor_a_row_major:
                :return:
                """
                l0_g_index, l0_n_index, l0_m1_index, l0_co1_index, l0_m0_index, l0_co0_index = indices
                l1_hw_index = l0_m1_index * self._m0 + l0_m0_index
                l1_c1_index = l0_g_index * self._dy_c1_extend + l0_co1_index
                return tvm.select(tvm.any(l1_c1_index <= co1_dim - 1),
                                  dedy_tensor_l1(
                                      l0_n_index,
                                      l1_c1_index,
                                      l1_hw_index,
                                      l0_co0_index))

            # from L1(zN:n,c1_ori*grousps,ho*wo,c0) to L0(zZ :G,n,hw//m0,c1,m0,c0)
            cout1_factor = 2 if dedy.dtype == "float32" else 1
            shape_dy_l0 = (self._g_extend,
                           batch_dim,
                           hw_dim,
                           self._dy_c1_extend * cout1_factor,
                           self._m0,
                           co0_dim)
            if self._var_map:
                dedy_col = tvm.compute(
                    shape_dy_l0,
                    lambda *indices: __im2col_fractal_indices_dynamic(indices, dedy_col),
                    name=dedy.name + "_col_fractal")
            else:
                dedy_col = tvm.compute(
                    shape_dy_l0,
                    lambda *indices: __im2col_fractal_indices(indices, dedy_col),
                    name=dedy.name + "_col_fractal")

        return dedy_col

    def generate_b(self, kernels):
        """
        generate b_l0b tensor for mad

        Parameters
        ----------
        kernels : weight tensor of fractal shape before transformation in ddr

        Returns
        ----------
        b_l0b: w tensor of fractal shape after transformation in L0B
        """

        if kernels.dtype == "int8":
            _, _, ci0_dim, k0_dim = list(i.value for i in kernels.shape)
            def _bl1_elem_func(*index):
                return kernels(*index)

            def _bl0_elem_func(indices, b_l1):
                l0b_g_index, l0b_co1_index, l0b_ci1_index, l0b_ci0_index, l0b_co0_index = indices
                l1b_k1_index = l0b_g_index * self._dy_c1_extend + l0b_co1_index
                return b_l1[l1b_k1_index, l0b_ci1_index, l0b_ci0_index, l0b_co0_index]

            b_l1 = tvm.compute(
                kernels.shape, _bl1_elem_func, name=kernels.name + "_B_l1"
            )
            shape_b_l0 = (self._g_extend,
                          self._dy_c1_extend,
                          self._dx_c1_extend,
                          ci0_dim,
                          k0_dim)
            b_l0 = tvm.compute(
                shape_b_l0, lambda *indices:
                _bl0_elem_func(indices, b_l1),
                name=kernels.name + "_B_l0b",
                attrs={"kernel_hw": (self._kernel_h, self._kernel_w)}
            )
        elif kernels.dtype == "float32":
            k1_dim, co1_dim, co0_dim, k0_dim = list(i.value for i in kernels.shape)
            shape = (k1_dim, co1_dim, co0_dim, k0_dim)

            def _bl1_elem_func(*index):
                return kernels(*index)

            b_l1 = tvm.compute(shape, _bl1_elem_func, name=kernels.name + "_B_l1")
            shape_b_l0 = (self._g_extend,
                          self._dy_c1_extend * 2,
                          (self._dx_c1_extend + 1) // 2,
                          k0_dim * 2,
                          co0_dim // 2)

            def __kernel_l0_compute(indices, b_l1):
                l0b_g_index, l0b_co1_index, l0b_ci1_index, l0b_ci0_index, l0b_co0_index = indices
                _, block_k0, block_n0 = tbe_platform.CUBE_MKN["float32"]["mac"]
                return b_l1[(l0b_ci1_index * block_n0 + l0b_ci0_index) // block_k0,
                            (l0b_co1_index * block_k0 + l0b_co0_index) // block_n0,
                            (l0b_co1_index * block_k0 + l0b_co0_index) % block_n0,
                            (l0b_ci1_index * block_n0 + l0b_ci0_index) % block_k0]

            b_l0 = tvm.compute(
                shape_b_l0,
                lambda *indices: __kernel_l0_compute(indices, b_l1),
                name=kernels.name + "_B_l0b"
            )
        else:
            k1_dim, co1_dim, co0_dim, k0_dim = list(i.value for i in kernels.shape)
            shape = (k1_dim, co1_dim, co0_dim, k0_dim)

            def _bl1_elem_func(*index):
                return kernels(*index)

            b_l1 = tvm.compute(shape, _bl1_elem_func, name=kernels.name + "_B_l1")
            shape_b_l0 = (self._g_extend,
                          self._dy_c1_extend,
                          self._dx_c1_extend,
                          k0_dim,
                          co0_dim)

            def __kernel_l0_compute(indices, b_l1):
                """

                :param indices:(G, dy_c1, dx_c1, dx_c0, dy_c0)
                :param b_l1: (G*dx_c1*KH *KW, Co1, Co0, dx_c0)
                :return:
                """
                l0b_g_index, l0b_co1_index, l0b_ci1_index, l0b_ci0_index, l0b_co0_index = indices
                l1b_k1_index = l0b_g_index * self._dx_c1_extend + l0b_ci1_index

                return b_l1[l1b_k1_index, l0b_co1_index, l0b_co0_index, l0b_ci0_index]


            b_l0 = tvm.compute(
                shape_b_l0,
                lambda *indices: __kernel_l0_compute(indices, b_l1),
                name=kernels.name + "_B_l0b"
            )
        return b_l0

    def generate_c(
        self, tensor_a, tensor_b, tensor_bias=None, c_type=None, offset_x=0
    ):  # pylint:disable=R0914,R0913
        """
        generate img_c

        Parameters
        ----------
        tensor_a : dE/dY tensor of fractal shape in L0A

        tensor_b : w tensor of fractal shape after transformation in L0B

        tensor_bias : same as that in Class->CubeDslPattern

        c_type : same as that in Class->CubeDslPattern

        offset_x : same as that in Class->CubeDslPattern

        Returns
        ----------
        img_c: dx tensor in ddr
        """

        def _add_bias_in_ub(in_tensor0, in_tensor1):
            c_add_vector = tvm.compute(
                in_tensor0.shape,
                lambda *indice: in_tensor0(*indice) + in_tensor1(indice[1] * tbe_platform.CUBE_MKN[
                    in_tensor0.dtype]["mac"][2] + indice[3]),
                name="bias_add_vector")
            return c_add_vector

        def _inner_generate_mmad(matrix_a, matrix_b):  # pylint:disable=R0914
            g_extend_dim, n_dim, hw_dim, k1_dim, m0_dim, k0_dim = cube_util.shape_to_list(matrix_a.shape)
            g_extend_dim, bk1_dim, co1_dim, co0_dim, bk0_dim = list(i.value for i in matrix_b.shape)
            if bk1_dim != k1_dim or bk0_dim != k0_dim:
                dict_args = dict()
                dict_args["errCode"] = "E60108"
                dict_args["reason"] = "invaild shape bk1_dim"
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

            shape_c = (g_extend_dim, n_dim, co1_dim, hw_dim * m0_dim, co0_dim)
            k0_axis = tvm.reduce_axis([0, k0_dim], name="k0")
            k1_axis = tvm.reduce_axis([0, k1_dim], name="k1")
            if matrix_a.dtype == "int8" and matrix_b.dtype == "int8":
                c_type = "int32"
                mode = "s8"
            elif tbe_platform.intrinsic_check_support("Intrinsic_mmad", "f162f32"):
                c_type = "float32"
                mode = "f162f32"
            else:
                c_type = "float16"
                mode = "f162f16"

            offset_x = self._offset_x if cube_util.is_support_v200() else 0
            mmad = tvm.compute(
                shape_c,
                lambda g_index, n_index, co1_index, m_index, co0_index: tvm.sum(
                    (
                        (
                            matrix_a[g_index, n_index, m_index // self._m0,
                                     k1_axis, m_index % self._m0, k0_axis]
                            - offset_x
                        )
                        * matrix_b[g_index, k1_axis, co1_index, co0_index, k0_axis]
                    ).astype(c_type),
                    axis=[k1_axis, k0_axis]
                ),
                name="C",
                attrs={"mode": mode}
            )
            return mmad

        def _add_bias_in_l0c(mmad, tensor_bias):
            if (
                tensor_bias is not None
                and tensor_bias.dtype == "int32"
                and self._stride_h == 1
                and self._stride_w == 1
            ):
                shape_c = mmad.shape
                bias_ub_brc_shape = list(shape_c)
                bias_ub_brc_shape[3] = bias_ub_brc_shape[3] // BRC_STANDARD_BLOCK_SIZE
                co_k = tbe_platform.CUBE_MKN[tensor_bias.dtype]["mac"][2]
                bias_ub_brc = tvm.compute(
                    bias_ub_brc_shape,
                    lambda *indices:
                    tensor_bias(
                        indices[0] * shape_c[2] * co_k + indices[2] * co_k + indices[4]
                    ),
                    name="bias_ub_brc"
                )
                bias_l0c = tvm.compute(
                    shape_c,
                    lambda g, i, j, k, l: bias_ub_brc(
                        g, i, j, k // BRC_STANDARD_BLOCK_SIZE, l
                    ),
                    name="bias_l0c"
                )
                mmad = tvm.compute(
                    shape_c,
                    lambda *indices: bias_l0c(*indices) + mmad(*indices),
                    name="c_add_bias"
                )
            return mmad

        res_c = _inner_generate_mmad(tensor_a, tensor_b)
        res_c = _add_bias_in_l0c(res_c, tensor_bias)

        batch_dx_img, c1_dx_img, h_dx_img, w_dx_img, c0_dx_img = self._output_shape
        group_l0c, batch_l0c, co1_l0c, m_l0c, co0_l0c = cube_util.shape_to_list(
            res_c.shape)
        if not (batch_dx_img == batch_l0c and
                c0_dx_img == co0_l0c and
                c1_dx_img <= group_l0c * co1_l0c):
            dict_args = dict()
            dict_args["errCode"] = "E60108"
            dict_args["reason"] = "ouput shape illegal"
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        ub_dx_shape = [batch_dx_img, c1_dx_img, m_l0c, c0_dx_img]
        res_c_dtype = self.output_dtype
        if tensor_a.dtype == "int8" and tensor_b.dtype == "int8":
            res_c_dtype = "int32"

        # from l0c(GNC1MC0) to ub(N[GC1]MC0)
        output_shape = [batch_dx_img, c1_dx_img, h_dx_img * w_dx_img, c0_dx_img]
        if self._cube_vector_split_flag:
            if res_c_dtype == "float32":
                output_shape = [batch_dx_img, c1_dx_img * 2, h_dx_img * w_dx_img, 8]
                output_shape_fp32 = [batch_dx_img, c1_dx_img * 2, h_dx_img, w_dx_img, 8]
                img_c = tvm.compute(
                    output_shape,
                    lambda n, c1, hw, c0: res_c(c1 // self._dx_c1_extend // 2, n,
                                                c1 // 2 % self._dx_c1_extend, hw, 8 * (c1 % 2) + c0).astype(res_c_dtype),
                    tag="conv2d_backprop_input_opti",
                    name=res_c.name + "_img",
                    attrs={
                        "hw_dim": h_dx_img * w_dx_img,
                        "dx_5D_shape": output_shape_fp32,
                        "group_dict": self._group_dict,
                        "kernel_name": self._kernel_name
                    }
                )
            else:
                img_c = tvm.compute(
                    output_shape,
                    lambda n, c1, hw, c0: res_c(c1 // self._dx_c1_extend, n,
                        c1 % self._dx_c1_extend, hw, c0).astype(res_c_dtype),  # pylint: disable=W0108
                    tag="conv2d_backprop_input_opti",
                    name=res_c.name + "_img",
                    attrs={
                        "hw_dim": h_dx_img * w_dx_img,
                        "dx_5D_shape": self._output_shape,
                        "group_dict": self._group_dict,
                        "kernel_name": self._kernel_name
                    }
                )
        else:
            res_cub = tvm.compute(
                ub_dx_shape,
                lambda batch_ub_index, c1_ub_index, m_ub_index, c0_ub_index:
                res_c[c1_ub_index // self._dx_c1_extend, batch_ub_index,
                      c1_ub_index % self._dx_c1_extend, m_ub_index, c0_ub_index]
                    .astype(res_c_dtype),
                name='CUB'
            )

            if self._stride_h > 1 or self._stride_w > 1:
                res_cub = self._get_dilate_tensor(
                    res_cub, h_dx_img, w_dx_img, self._stride_h, self._stride_w
                )

            if (tensor_bias is not None
                    and ((tensor_bias.dtype == "float16" or tensor_bias.dtype == "float32")
                    or (self._stride_h > 1 or self._stride_w > 1))
            ):
                res_cub = _add_bias_in_ub(res_cub, tensor_bias)

            img_c = tvm.compute(
                output_shape,
                lambda n, c1, hw, c0: res_cub(n, c1, hw, c0),  # pylint: disable=W0108
                tag="conv2d_backprop_input_opti",
                name=res_cub.name + "_img",
                attrs={
                    "hw_dim": h_dx_img * w_dx_img,
                    "dx_5D_shape": self._output_shape,
                    "group_dict": self._group_dict,
                    "kernel_name": self._kernel_name
                }
            )
        return img_c
