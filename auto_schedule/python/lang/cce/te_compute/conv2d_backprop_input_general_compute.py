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
conv2d backprop input general compute.
"""
from te.lang.cce.te_compute.cube_util import ConvDslPattern
from te.lang.cce.te_compute.cube_util import CubeDslPattern
from te.lang.cce.te_compute.cube_util import shape_to_list
from te.lang.cce.te_compute.cube_util import GroupDictKeys
from te.platform import cce_conf
from te.platform import cce_params
from te.tvm import api as tvm
from te.tvm.intrin import abs as tvm_abs
from tbe.common.utils.errormgr import error_manager_util



class DeConvPattern(CubeDslPattern):  # pylint: disable=R0902
    """
    class of convolution back propagation

    Parameters
    ----------
    kernel_sizes : shape of weight, [N, C, H, W]

    strides : list of strides, [strideh, stridew]

    pad: list of padding, [pad_up, pad_down, pad_left, pad_right]

    output_shape : shape of dE/dX, [N, C, H, W]

    dilations: list of dilations, [dilate_n, dilate_c, dilate_h, dilate_w]

    offset_x : offset of x

    kernel_name : cce kernel name

    group_dict : The params of group convolution.

    Returns
    -------
    deconv_pattern_instance : instance of deconv pattern
    """

    fusion_para_map = None
    dedy = None

    def __init__(  # pylint: disable=R0913
        self,
        kernel_sizes,
        strides,
        pad,
        output_shape,
        dilations,
        offset_x,
        fusion_para,
        kernel_name,
        group_dict,
        var_map
    ):
        super().__init__()
        _, _, kernel_h, kernel_w = kernel_sizes
        self._kernel_h = kernel_h
        self._kernel_w = kernel_w
        self._stride_h, self._stride_w = strides
        self._pad_up, self._pad_down, self._pad_left, self._pad_right = pad
        self._output_shape = output_shape
        self._kernel_name = kernel_name
        _, _, self._dilate_h, self._dilate_w = dilations
        self.m_0, _, _ = cce_params.CUBE_MKN["float16"]["mac"]
        self._offset_x = offset_x
        self._fusion_para = fusion_para
        self._var_map = var_map
        self._group_dict = group_dict
        self._real_g = self._group_dict.get(GroupDictKeys.g_extend)
        self._cou1_g = self._group_dict.get(GroupDictKeys.dy_c1_extend)
        self._cin1_g = self._group_dict.get(GroupDictKeys.dx_c1_extend)
        self._cube_vector_split_flag = cce_conf.get_soc_spec("CUBE_VECTOR_SPLIT")

    def generate_a(self, dy_ddr):  # pylint: disable=R0914,R0915
        """
        generate dy_col tensor for mad

        Parameters
        ----------
        dy_ddr: 5D dE/dY tensor in ddr

        Returns
        ----------
        dy_col: dE/dY tensor of fractal shape in L0A
        """

        def _check_pad_zero(pad_list):
            """
            if pad less than 0, return True
            """
            for pad in pad_list:
                if pad < 0:
                    return True
            return False

        def _fill_zero(shape_dy_filling):
            dy_zero = tvm.compute(
                shape_dy_filling,
                lambda *indice: tvm.convert(self._offset_x).astype(dy_ddr.dtype),
                name="dy_filling_i",
                tag="init_zero"
            )
            return dy_zero

        def _write_select(dy_h):
            if valid_shape:
                dy_h = valid_shape[2]
            shape_dy_filling = (
                dy_batch,
                kernel_cout1,
                dy_h * stride_h,
                dy_w * stride_w,
                kernel_cout0
            )
            if valid_shape and (stride_h > 1 or stride_w > 1 or input_mem == 0):
                dy_read = tvm.compute(
                    valid_shape,
                    lambda n, c1, h, w, c0: dy_ddr[n, c1, h + slice_offset[2], w, c0],
                    name="dy_selet_read",
                    tag="dy_selet_read"
                )
            else:
                dy_read = dy_ddr
            if stride_h == 1 and stride_w == 1:
                dy_filling = dy_read
            else:
                dy_zero = _fill_zero(shape_dy_filling)
                dy_filling = tvm.compute(
                    shape_dy_filling,
                    lambda batch_idx, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx: tvm.select(
                        tvm.all(ho_idx % stride_h == 0, wo_idx % stride_w == 0),
                        dy_read[
                            batch_idx,
                            kernel_cout1_idx,
                            ho_idx // stride_h,
                            wo_idx // stride_w,
                            kernel_cout0_idx
                        ],
                        dy_zero[
                            batch_idx,
                            kernel_cout1_idx,
                            ho_idx,
                            wo_idx,
                            kernel_cout0_idx
                        ]
                    ),
                    name="dy_filling",
                    tag="stride_filling",
                    attrs={"stride_expand": (self._stride_h, self._stride_w)}
                )
            return dy_filling, shape_dy_filling, dy_h

        def _write_select_dynamic(dy_h):
            shape_dy_filling = (
                dy_batch,
                kernel_cout1,
                dy_h * stride_h,
                dy_w * stride_w,
                kernel_cout0
            )
            if stride_h == 1 and stride_w == 1:
                dy_filling = dy_ddr
            else:
                dy_zero = _fill_zero(shape_dy_filling)
                dy_vn = tvm.compute(
                    shape_to_list(dy_ddr.shape),
                    lambda *indice: dy_zero(*indice) + dy_ddr(*indice),
                    name="dy_vn",
                    tag="dy_vn"
                )

                dy_filling = tvm.compute(
                    shape_dy_filling,
                    lambda batch_idx, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx: tvm.select(
                        tvm.all(ho_idx % stride_h == 0, wo_idx % stride_w == 0),
                        dy_vn[
                            batch_idx,
                            kernel_cout1_idx,
                            ho_idx // stride_h,
                            wo_idx // stride_w,
                            kernel_cout0_idx
                        ]
                    ),
                    name="dy_filling",
                    tag="stride_filling",
                    attrs={"stride_expand": (self._stride_h, self._stride_w)}
                )

            return dy_filling, shape_dy_filling, dy_h

        fusion_para = self._fusion_para
        DeConvPattern.fusion_para_map = fusion_para
        DeConvPattern.dedy = dy_ddr
        l1_fusion_type = fusion_para.get("l1_fusion_type")
        input_mem = fusion_para.get("input_memory_type")
        valid_shape = fusion_para.get("valid_shape")
        slice_offset = fusion_para.get("slice_offset")

        dy_batch, kernel_cout1, dy_h, dy_w, kernel_cout0 = shape_to_list(dy_ddr.shape)
        stride_h, stride_w = self._stride_h, self._stride_w

        kernel_h, kernel_w = self._kernel_h, self._kernel_w
        dilate_h, dilate_w = self._dilate_h, self._dilate_w

        if self._var_map:
            dy_filling, shape_dy_filling, dy_h = _write_select_dynamic(dy_h)
        else:
            dy_filling, shape_dy_filling, dy_h = _write_select(dy_h)

        new_stride = (1, 1)
        new_hw = (dy_h * stride_h, dy_w * stride_w)

        new_pad_before = (
            (kernel_h - 1) * dilate_h - self._pad_up,
            (kernel_w - 1) * dilate_w - self._pad_left
        )
        pad_up_before, pad_left_before = new_pad_before

        _, _, dx_h, dx_w, _ = self._output_shape
        new_pad_after = tuple(
            i - o - pb + (k - 1) * d
            for i, o, pb, k, d in zip(
                (dx_h, dx_w),
                new_hw,
                new_pad_before,
                (kernel_h, kernel_w),
                (dilate_h, dilate_w)
            )
        )
        pad_down_after, pad_right_after = new_pad_after

        pad_list = (pad_up_before, pad_down_after,
                    pad_left_before, pad_right_after)
        # stride > 1 ub->l1 may cut
        if stride_h > 1 or stride_w > 1:
            if self._var_map or _check_pad_zero(pad_list):
                shape_up_modify = (pad_up_before - tvm_abs(pad_up_before)) // 2
                shape_left_modify = (pad_left_before - tvm_abs(pad_left_before)) // 2
                shape_down_modify = (pad_down_after - tvm_abs(pad_down_after)) // 2
                shape_right_modify = (pad_right_after - tvm_abs(pad_right_after)) // 2

                shape_dy_filling_cut = (
                    dy_batch,
                    kernel_cout1,
                    dy_h * stride_h + shape_up_modify + shape_down_modify,
                    dy_w * stride_w + shape_left_modify + shape_right_modify,
                    kernel_cout0
                )

                # cut dy_filling
                dy_filling_l1 = tvm.compute(
                    shape_dy_filling_cut,
                    lambda batch_idx, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx: dy_filling[
                        batch_idx,
                        kernel_cout1_idx,
                        ho_idx - shape_up_modify,
                        wo_idx - shape_left_modify,
                        kernel_cout0_idx
                    ],
                    name="dy_l1_cut",
                    tag="dy_l1_cut"
                )
            else:
                dy_filling_l1 = tvm.compute(
                    shape_dy_filling,
                    lambda batch_idx, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx: dy_filling[
                        batch_idx, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx
                    ],
                    name="dy_l1",
                    tag="dy_l1"
                )
        elif not self._var_map and _check_pad_zero(pad_list):
            shape_up_modify = (pad_up_before - tvm_abs(pad_up_before)) // 2
            shape_left_modify = (pad_left_before - tvm_abs(pad_left_before)) // 2
            shape_down_modify = (pad_down_after - tvm_abs(pad_down_after)) // 2
            shape_right_modify = (pad_right_after - tvm_abs(pad_right_after)) // 2

            shape_dy_filling_cut = (
                dy_batch,
                kernel_cout1,
                dy_h * stride_h + shape_up_modify + shape_down_modify,
                dy_w * stride_w + shape_left_modify + shape_right_modify,
                kernel_cout0
            )

            # cut dy_filling
            dy_filling = tvm.compute(
                shape_dy_filling_cut,
                lambda batch_idx, kernel_cout1_idx, ho_idx, wo_idx, kernel_cout0_idx: dy_filling[
                    batch_idx,
                    kernel_cout1_idx,
                    ho_idx - shape_up_modify,
                    wo_idx - shape_left_modify,
                    kernel_cout0_idx
                ],
                name="dy_l1_modify",
                tag="dy_l1_modify"
            )

        pad_up_before = (pad_up_before + tvm_abs(pad_up_before)) // 2
        pad_left_before = (pad_left_before + tvm_abs(pad_left_before)) // 2
        pad_down_after = (pad_down_after + tvm_abs(pad_down_after)) // 2
        pad_right_after = (pad_right_after + tvm_abs(pad_right_after)) // 2

        new_pad = (pad_up_before, pad_down_after,
                   pad_left_before, pad_right_after)

        pat_conv = ConvDslPattern(
            kernel_h,
            kernel_w,
            new_stride,
            new_pad,
            (dilate_h, dilate_w),
            self._offset_x
        )

        if stride_h > 1 or stride_w > 1:
            dy_col = pat_conv.generate_a(dy_filling_l1, self._real_g,
                                         self._cou1_g, self._var_map)
        else:
            offset = 0
            if l1_fusion_type != -1 and input_mem == 1 and valid_shape:
                offset = slice_offset[2]
            dy_col = pat_conv.generate_a(
                dy_filling,
                self._real_g,
                self._cou1_g,
                self._var_map,
                slice_offset=offset,
                valid_shape=valid_shape)

        return dy_col

    def generate_b(self, kernels):
        """
        generate w_col tensor for mad

        Parameters
        ----------
        kernels: weight tensor of fractal shape before transformation in ddr

        Returns
        ----------
        w_col: w tensor of fractal shape after transformation in L0B
        """

        if kernels.dtype == "int8":
            w_k1, kernel_cin1, kernel_cin0, w_k0 = shape_to_list(kernels.shape)
            shape_w_l0b = (self._real_g,
                           w_k1//self._real_g,
                           kernel_cin1,
                           kernel_cin0,
                           w_k0
                          )

            def _kernel_elem_func(indices, kernels):
                g_idx, w_k1_idx, kernel_cin1_idx, w_k0_idx, kernel_cout0_idx = indices
                fkk_idx = g_idx * shape_w_l0b[1] + w_k1_idx
                return kernels[fkk_idx, kernel_cin1_idx, w_k0_idx, kernel_cout0_idx]

            w_col = tvm.compute(
                shape_w_l0b, lambda *indices:
                _kernel_elem_func(indices, kernels), name="w_col", tag="inverse_trans_dma"
            )
        else:
            w_k1, kernel_cout1, kernel_cout0, w_k0 = shape_to_list(kernels.shape)
            kernel_h, kernel_w = self._kernel_h, self._kernel_w
            if w_k1 % (kernel_h * kernel_w) != 0:
                dict_args = dict()
                dict_args["errCode"] = "E60108"
                dict_args["reason"] = "w_k1 could not be divided by kernel_h*kernel_w"
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

            shape_w_l0b = (self._real_g,
                           self._cou1_g*kernel_h*kernel_w,
                           self._cin1_g,
                           w_k0,
                           kernel_cout0)

            def __kernel_2_l0_compute(indices, kernels):
                g_idx, w_k1_idx, kernel_cin1_idx, w_k0_idx, kernel_cout0_idx = indices

                fkk_index = g_idx * self._cin1_g * kernel_h * kernel_w + (
                            kernel_cin1_idx * kernel_h * kernel_w) + (
                            kernel_h * kernel_w - 1 - w_k1_idx
                            % (kernel_h * kernel_w))
                cout1_index = w_k1_idx // (kernel_h * kernel_w)
                return kernels[fkk_index, cout1_index, kernel_cout0_idx, w_k0_idx]
            w_col = tvm.compute(shape_w_l0b, lambda *indices:
                __kernel_2_l0_compute(indices, kernels),
                name="w_col",
                tag="inverse_trans_dma")

        return w_col

    def generate_c(  # pylint: disable=R0914,R0913
        self, tensor_a, tensor_b, tensor_bias=None, c_type=None, offset_x=0
    ):
        """
        generate dx_ddr

        Parameters
        ----------
        tensor_a : dE/dY tensor of fractal shape in L0A

        tensor_b : w tensor of fractal shape after transformation in L0B

        tensor_bias : same as that in Class->CubeDslPattern

        c_type : same as that in Class->CubeDslPattern

        offset_x : same as that in Class->CubeDslPattern

        Returns
        ----------
        dx_ddr: dx tensor in ddr
        """

        def _add_bias_in_ub(in_tensor0, in_tensor1):
            c_add_vector = tvm.compute(
                in_tensor0.shape,
                lambda *indice: in_tensor0(*indice)
                + in_tensor1(
                    indice[1] * cce_params.CUBE_MKN[in_tensor0.dtype]["mac"][2]
                    + indice[3]
                ),
                name="bias_add_vector"
            )
            return c_add_vector

        dy_col = tensor_a
        w_col = tensor_b

        res_c_type = "float32"
        if not cce_conf.intrinsic_check_support("Intrinsic_mmad", "f162f32"):
            res_c_type = "float16"
        if w_col.dtype == "int8" and dy_col.dtype == "int8":
            res_c_type = "int32"

        if tensor_bias is not None and tensor_bias.dtype == "int32":
            bias = tensor_bias
        else:
            bias = None
        dx_col = super().generate_c(
            dy_col, w_col, c_type=res_c_type, tensor_bias=bias, offset_x=self._offset_x
        )
        # mad dx shape
        dx_g, dx_batch, dx_c1, dx_hw, dx_c0 = shape_to_list(dx_col.shape)

        # real dx shape
        _, dx_cin1, dx_h, dx_w, dx_cin0 = self._output_shape
        out_shape = (dx_batch, dx_cin1, dx_h * dx_w, dx_cin0)

        # float32->float16
        output_type = "float16"
        if w_col.dtype == "int8" and dy_col.dtype == "int8":
            output_type = "int32"

        if self._cube_vector_split_flag:
            dx_ddr = tvm.compute(
                out_shape,
                lambda dx_batch_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx: dx_col[dx_cin1_idx // self._cin1_g,
                    dx_batch_idx, dx_cin1_idx % self._cin1_g, dx_hw_idx, dx_cin0_idx
                ].astype(output_type),
                name="c_ddr",
                tag="conv2d_backprop_input",
                attrs={"output_shape": (dx_batch, dx_cin1, dx_h, dx_w, dx_cin0),
                       "group_dict": self._group_dict,
                       "l0c_shape": (dx_g, dx_batch, dx_c1, dx_hw, dx_c0),
                       "kernel_name": self._kernel_name})
        else:
            dx_ub = tvm.compute(
                (dx_batch, dx_cin1, dx_hw, dx_c0),
                lambda dx_batch_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
                dx_col[dx_cin1_idx // self._cin1_g, dx_batch_idx,
                    dx_cin1_idx % self._cin1_g, dx_hw_idx, dx_cin0_idx]
                .astype(output_type), name="c_ub")

            if tensor_bias is not None and tensor_bias.dtype == "float16":
                dx_ub = _add_bias_in_ub(dx_ub, tensor_bias)

            dx_ddr = tvm.compute(
                out_shape,
                lambda dx_batch_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx: dx_ub[
                    dx_batch_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx
                ],
                name="c_ddr",
                tag="conv2d_backprop_input",
                attrs={"output_shape": (dx_batch, dx_cin1, dx_h, dx_w, dx_cin0),
                       "group_dict": self._group_dict,
                       "l0c_shape": (dx_g, dx_batch, dx_c1, dx_hw, dx_c0),
                       "kernel_name": self._kernel_name})
        return dx_ddr
