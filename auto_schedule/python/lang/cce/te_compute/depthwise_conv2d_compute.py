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
Compute of depthwise conv2d.
"""
from te.tvm import api as tvm
from te.platform import cce_conf
from te.platform import cce_params
from te.lang.cce.te_compute import common
from te.utils import para_check
from te.utils.error_manager import error_manager_util as err_mana
from topi.cce.util import check_load3d_w_out_1_support

BLOCK_SIZE = cce_params.BLOCK_REDUCE
BLOCK_INT8_SIZE = cce_params.BLOCK_REDUCE_INT8
FRACTAL_M = 16
FRACTAL_N = 16
NAME_INDEX = [0]


# pylint: disable=too-few-public-methods
class DepthwiseConv2dParam:
    """
    class of DepthwiseConv2dParam
    """
    def __init__(self):
        pass

    fusion_para = {
        "input_memory_type": 0,
        "output_memory_type": 0,
        "valid_shape": (),
        "slice_offset": (),
        "l1_fusion_type": -1,
        "fmap_l1_addr_flag": -1,
        "fmap_l1_valid_size": -1
    }


def _check_support_v200():
    """
    check if Ascend610/Ascend615/Ascend710/Hi3796CV300CS version
    ----------

    Returns
    -------
    True:  Ascend610/Ascend615/Ascend710/Hi3796CV300CS version
    False: Other version
    """
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    if soc_version in ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS"):
        return True
    return False


def _fusion_fmap_select(fmap, l1_fusion_para):
    """
     check L1 fusion fmap select
    """
    valid_shape = l1_fusion_para.get("valid_shape")
    offset = l1_fusion_para.get("slice_offset")
    input_memory_type = int(l1_fusion_para.get("input_memory_type"))
    data_res = fmap

    if len(fmap.shape) == 6:
        _, _, _, fmap_h, fmap_w, _ = fmap.shape
    else:
        _, _, fmap_h, fmap_w, _ = fmap.shape
    fmp_select_h = fmap_h
    fmp_select_w = fmap_w

    if offset and valid_shape and input_memory_type == 0:
        fmp_select_n, fmp_select_c1, fmp_select_h, fmp_select_w, fmp_select_c0 = valid_shape
        if len(fmap.shape) == 5:
            n_offset, c1_offset, h_offset, w_offset, c0_offset = offset
            data_res = tvm.compute(
                valid_shape,
                lambda n, c1, h, w, c0: fmap(n + n_offset, c1 + c1_offset, h + h_offset, w + w_offset, c0 + c0_offset),
                name="fusion_fmap_select")
        elif len(fmap.shape) == 6:
            n_offset, c1_offset, h_offset, w_offset, c0_offset = offset
            valid_shape = fmp_select_n, fmp_select_c1, 1, fmp_select_h, fmp_select_w, fmp_select_c0
            data_res = tvm.compute(valid_shape,
                                   lambda n, c1, cg, h, w, c0: fmap(n + n_offset, c1 + c1_offset, cg, h + h_offset, w +
                                                                    w_offset, c0 + c0_offset),
                                   name="fusion_fmap_select")
        else:
            raise RuntimeError("feature_map shape length only support 5 or 6")
    if valid_shape and input_memory_type == 1:
        fmp_select_n, fmp_select_c1, fmp_select_h, fmp_select_w, fmp_select_c0 = valid_shape
    return data_res, fmp_select_h, fmp_select_w


def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    return [i.value for i in shape]


def _bias_add(res, bias):
    """
    calculate depthwise res + bias in UB
    Parameters
    ----------
    res: res_shape = (fmap_n, fmap_c1, fractal_n_split, output_h*output_w,\
                             cce_params.BLOCK_REDUCE)

    bias: bias vector = (filter_c1*filter_c0)

    Returns
    -------
    res+bias tensor
    """
    dim_map = {}
    dim_map["out_img_shape"] = _shape_to_list(res.shape)
    fractal_n_split = dim_map["out_img_shape"][2]
    c_add_vector = tvm.compute(dim_map["out_img_shape"],
                               lambda *indice: res(*indice) + bias(
                                   (indice[1] * fractal_n_split + indice[2]) * cce_params.BLOCK_REDUCE + indice[4]),
                               name='bias_add_vector' + "_cc",
                               tag='depthwise_conv2d')
    return c_add_vector


def _img2col(input_img, col_shape, filter_h, filter_w, pad, stride, dilations):
    """
    img2col for depthwise conv2d backprop filter

    Parameters
    ----------
    input_img : tvm tensor
        tensor to img2col.

    col_shape : tuple or list
        shape of output.

    filter_h: int
        height of filter.

    filter_w: int
        width of filter.

    pad: tuple or list
        pad data.

    stride: tuple or list or int
        stride of convolution.

    dilations: tuple or list or int
        dilations of convolution.

    Returns
    -------
        tensor after img2col.
    """
    def _img2col_compute(indices):
        """img2col for depthwise conv2d backprop filter"""
        _, _, _, fmap_h, fmap_w, _ = input_img.shape

        kernel_dilate_w = (filter_w - 1) * dilations[1] + 1
        output_w = (fmap_w.value + pad[2] + pad[3] - kernel_dilate_w) // stride[1] + 1

        ori_h_index = (indices[2] // output_w) * stride[0] + indices[4] * dilations[0]
        img_w_index = (indices[2] % output_w) * stride[1] + indices[5] * dilations[1]
        img_c0_index = indices[6]

        return tvm.select(
            tvm.any(ori_h_index < pad[0], ori_h_index > fmap_h.value + pad[0] - 1, img_w_index < pad[2],
                    img_w_index > fmap_w.value + pad[2] - 1), tvm.const(0.0, input_img.dtype),
            input_img(indices[0], indices[1], indices[3], ori_h_index - pad[0], img_w_index - pad[2], img_c0_index))

    return tvm.compute(col_shape,
                       lambda *indices: _img2col_compute(indices),
                       name='im2col_row_major',
                       tag='im2col_row_major',
                       attrs={
                           'kernel_h': filter_h,
                           'kernel_w': filter_w,
                           'padding': pad,
                           'stride': stride
                       })


def _im2col_fractal(col_shape, img, dout_h, dout_w):
    """
    fractal(in L0B) for depthwise conv2d backprop filter after img2col

    Parameters
    ----------
    col_shape : tuple or list
        shape of output.

    img : tvm tensor
        tensor to img2col.

    dout_h: int
        height of dout.

    dout_w: int
        width of dout.

    Returns
    -------
        tensor in fractal(in L0B).
    """
    def __im2col_fractal_indices(indices, img, dout_h, dout_w):
        """
        fractal(in L0B) for depthwise conv2d backprop filter after img2col
        """
        _, _, _, _, kernel_h, kernel_w, _ = img.shape

        hw_index = indices[2] * BLOCK_SIZE + indices[5]
        howo = dout_h * dout_w

        c1_index = (((indices[3] * BLOCK_SIZE + indices[4]) // BLOCK_SIZE) // kernel_w.value) // kernel_h.value
        kh_index = ((indices[3] * BLOCK_SIZE + indices[4]) // BLOCK_SIZE) // kernel_w.value
        kw_index = ((indices[3] * BLOCK_SIZE + indices[4]) // BLOCK_SIZE) % kernel_w.value
        c0_index = (indices[3] * BLOCK_SIZE + indices[4]) % BLOCK_SIZE

        return tvm.select(tvm.any(hw_index < 0, hw_index > howo - 1), tvm.const(0.0, img.dtype),
                          img(indices[0], indices[1], hw_index, c1_index, kh_index, kw_index, c0_index))

    return tvm.compute(col_shape,
                       lambda *indices: __im2col_fractal_indices(indices, img, dout_h, dout_w),
                       name='im2col_fractal',
                       tag='im2col_fractal')


def _backprop_filter_matmul(mad_shape, left_tensor, right_tensor, dout_hw, dout_n, res_type):
    """
    depthwiese conv2d backprop filter matmul.

    Parameters
    ----------
    mad_shape : tuple or list
        shape of output.

    left_tensor : tvm tensor
        tensor to batch matmul, which in L0A.

    right_tensor: int
        tensor to batch matmul, which in L0B.

    dout_hw: int
        dout_h * dout_w

    dout_n: int
        dout_n

    res_type: str
        dtype of output in batch matmul.

    Returns
    -------
        tensor in fractal(in L0C).
    """
    k_n = tvm.reduce_axis((0, dout_n), name='k_n')
    k_hw = tvm.reduce_axis((0, dout_hw), name='k_hw')
    k1_val = k_hw.var // BLOCK_SIZE
    k0_val = k_hw.var % BLOCK_SIZE
    return tvm.compute(mad_shape,
                       lambda cg, j1, i, j0: tvm.sum((left_tensor[cg, k_n, i // BLOCK_SIZE, k1_val, i % BLOCK_SIZE, k0_val] *
                                                      right_tensor[cg, k_n, k1_val, j1, j0, k0_val]).astype(res_type),
                                                     axis=[k_n, k_hw]),
                       name='mad')


def _check_stride_rule(stride):
    """
    check stride rule
    """
    if stride[0] > 63 or stride[0] < 1:
        raise RuntimeError("invalid stride params, stride_h size must be [1,63].")
    if stride[1] > 63 or stride[1] < 1:
        raise RuntimeError("invalid stride params, stride_w size must be [1,63].")


def _get_mad_info(fmap, depthwise_res_dtype):
    """
    float32 for cloud, float16 for mini, int8 for quant;
    How to tell the two get_version
    """
    if not cce_conf.intrinsic_check_support("Intrinsic_mmad", "f162f32") and fmap.dtype == "float16":
        mad_out_dtype = "float16"
        mad_res_block_size = BLOCK_SIZE
        fractal_n_split = 1
        if depthwise_res_dtype != "float16":
            dict_args = dict()
            dict_args["errCode"] = "E60005"
            dict_args["param_name"] = "depthwise_res_dtype"
            dict_args["op_name"] = "depthwise_conv2d"
            dict_args["expected_dtype_list"] = "float16"
            dict_args["dtype"] = depthwise_res_dtype
            raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
    elif fmap.dtype in ("int8", "uint8"):
        mad_out_dtype = "int32"
        mad_res_block_size = BLOCK_INT8_SIZE
        fractal_n_split = 2
        if depthwise_res_dtype != "int32":
            dict_args = dict()
            dict_args["errCode"] = "E60005"
            dict_args["param_name"] = "depthwise_res_dtype"
            dict_args["op_name"] = "depthwise_conv2d"
            dict_args["expected_dtype_list"] = "int32"
            dict_args["dtype"] = depthwise_res_dtype
            raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
    elif cce_conf.intrinsic_check_support("Intrinsic_mmad", "f162f32") and fmap.dtype == "float16":

        mad_out_dtype = "float32"
        mad_res_block_size = BLOCK_SIZE
        fractal_n_split = 1
        if depthwise_res_dtype != "float16":
            dict_args = dict()
            dict_args["errCode"] = "E60005"
            dict_args["param_name"] = "depthwise_res_dtype"
            dict_args["op_name"] = "depthwise_conv2d"
            dict_args["expected_dtype_list"] = "float16"
            dict_args["dtype"] = depthwise_res_dtype
            raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
    else:
        dict_args = dict()
        dict_args["errCode"] = "E60004"
        dict_args["param_name"] = "depthwise_res_dtype"
        dict_args["op_name"] = "depthwise_conv2d"
        dict_args["expected_dtype_list"] = "int32 or float16 or float32"
        dict_args["dtype"] = depthwise_res_dtype
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
    return mad_out_dtype, mad_res_block_size, fractal_n_split


def depthwise_conv2d_compute(fmap,
                             weight,
                             depthwise_res_dtype,
                             stride,
                             pad,
                             dilation,
                             para_dict,
                             l1_fusion_para,
                             kernel_name="depthwise_conv2d_compute"):
    """
    algorithm: depthwise_conv2d_compute

    calculating  depthwise convolution compute

    the interface will be eliminated soon!

    Parameters
    ----------
    fmap : feature map placehold
        5-D shape of input tensor [N, C1, H, W, C0]

    weight : filter placehold
        5-D shape of filter tensor [C1, H, W, Co, C0]

    depthwise_res_dtype : dtype of depthwise UB result

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    pad : padding added to each dimension of the input

    dilation : the dilation factor for each dimension of input

    para_dict : bias tensor dict

    Returns
    -------
    depthwise_res : result tensor
       forward depthwise result of out
    """

    fmap_shape = [int(i.value) for i in fmap.shape]
    weight_shape = [int(i.value) for i in weight.shape]
    para_check.check_shape_rule(fmap_shape)
    para_check.check_shape_rule(weight_shape)
    _check_stride_rule(stride)

    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    if len(fmap_shape) == 6:
        fmap_n, fmap_c1, _, fmap_h, fmap_w, _ = fmap_shape
    else:
        fmap_n, fmap_c1, fmap_h, fmap_w, _ = fmap_shape
    filter_c1, kernel_h, kernel_w, filter_co, filter_c0 = weight_shape

    valid_mode_pad = [0, 0, 0, 0]
    howo_one_flag = (fmap_h == kernel_h) and (fmap_w == kernel_w) and (list(pad)
                                                                       == list(valid_mode_pad)) and (fmap.dtype
                                                                                                     == "float16")
    if howo_one_flag:
        pad = (0, 0, 0, 15)

    if l1_fusion_para is None:
        l1_fusion_para = {
            "input_memory_type": 0,
            "output_memory_type": 0,
            "valid_shape": (),
            "slice_offset": (),
            "l1_fusion_type": -1,
            "fmap_l1_addr_flag": -1,
            "fmap_l1_valid_size": -1
        }

    effective_filter_h = (kernel_h - 1) * dilation_h + 1
    effective_filter_w = (kernel_w - 1) * dilation_w + 1

    def _check_h_w_out(fmap_h, fmap_w, kernel_h, kernel_w, effective_filter_h,
                       effective_filter_w, pad, stride_h, stride_w):
        pad_top, pad_bottom, pad_left, pad_right = pad
        fmap_h_with_pad = fmap_h + pad_top + pad_bottom
        fmap_w_with_pad = fmap_w + pad_left + pad_right
        output_h = (fmap_h_with_pad - effective_filter_h) // stride_h + 1
        output_w = (fmap_w_with_pad - effective_filter_w) // stride_w + 1

        load2d_pass_flag = (kernel_h, kernel_w) == (1, 1) and (stride_h, stride_w) == (1, 1) and pad == (0, 0, 0, 0)
        # output_w = 1 case only support load2d or (chips in [Ascend310,
        # Hi3796CV300CS] and fmap_w with padding equals
        # kernel_w after dilation)
        hout_equal_1 = fmap_h + pad_top + pad_bottom - effective_filter_h == 0
        wout_equal_1 = fmap_w + pad_left + pad_right - effective_filter_w == 0
        wout_equal_1_pass_flag = wout_equal_1 \
            if check_load3d_w_out_1_support() else load2d_pass_flag
        # Ascend910 supports w_out equals 1 and h_out equals 1
        out_both_equal_1_pass_flag = hout_equal_1 and wout_equal_1

        # can`t support output_h and output_w are less than 1
        if int(output_h) < 1 or int(output_w) < 1:
            dict_args = {
                'errCode': 'E60039',
                'op_name': 'depthwise_conv2d',
                'attr_name': 'output_h',
                'param_name': 'output',
                'comparator': 'not less',
                'expected_value': '1',
                'input_value': str(output_h),
            }
            raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
        
        # if output_w is equal to 1, wout_equal_1 or load2d_pass_flag must be true
        elif int(output_w) == 1:
            if not (wout_equal_1_pass_flag or out_both_equal_1_pass_flag):
                dict_args = {
                    'errCode': 'E60039',
                    'op_name': 'depthwise_conv2d',
                    'attr_name': 'output_w',
                    'param_name': 'output',
                    'comparator': 'greater',
                    'expected_value': '1',
                    'input_value': str(output_w),
                }
            raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

        else:
            pass
    _check_h_w_out(fmap_h, fmap_w, kernel_h, kernel_w, effective_filter_h,
                   effective_filter_w, pad, stride_h, stride_w)

    DepthwiseConv2dParam.fusion_para = l1_fusion_para
    fmap_select, fmap_select_h, fmap_select_w = _fusion_fmap_select(fmap, l1_fusion_para)
    pad_top, pad_bottom, pad_left, pad_right = pad
    full_height = fmap_select_h + pad_top + pad_bottom
    full_width = fmap_select_w + pad_left + pad_right
    output_h = (full_height - effective_filter_h) // stride_h + 1
    output_w = (full_width - effective_filter_w) // stride_w + 1

    offset_x = para_dict.get('offset_x')
    if offset_x is None:
        offset_x = 0
    bias_tensor = para_dict.get('bias_tensor')
    dsl_flag = para_dict.get('dsl_flag')

    mad_out_dtype, mad_res_block_size, fractal_n_split = _get_mad_info(fmap, depthwise_res_dtype)

    fmap_im2col_row_major_shape = (fmap_n, fmap_c1, output_h * output_w, 1, kernel_h, kernel_w, mad_res_block_size)
    feature_col = common.im2col_6d(fmap_select, fmap_im2col_row_major_shape, kernel_h, kernel_w, pad, stride, offset_x,
                                   dilation)

    howo_mad = (output_h * output_w + FRACTAL_M - 1) // FRACTAL_M * FRACTAL_M
    fmap_im2col_fractal_shape = (fmap_n, fmap_c1, howo_mad // FRACTAL_M, 1 * kernel_h * kernel_w, FRACTAL_M,
                                 mad_res_block_size)

    feature_col_pad = common.im2col_fractal_6d(fmap_im2col_fractal_shape, feature_col)

    if mad_out_dtype == "int32":
        filter_reshape = tvm.compute(
            (filter_c1, kernel_h * kernel_w, 2, FRACTAL_N, filter_c0),
            lambda cg, hw, co1, co0, c0: weight(cg, hw // kernel_w, hw % kernel_w, co1 * FRACTAL_N + co0, c0),
            name='filter_reshape')
    elif mad_out_dtype in ("float16", "float32"):
        filter_reshape = tvm.compute((filter_c1, kernel_h * kernel_w, 1, filter_co, filter_c0),
                                     lambda cg, hw, co1, co0, c0: weight(cg, hw // kernel_w, hw % kernel_w, co0, c0),
                                     name='filter_reshape')
    else:
        dict_args = dict()
        dict_args["errCode"] = "E60004"
        dict_args["param_name"] = "depthwise_res_dtype"
        dict_args["op_name"] = "depthwise_conv2d"
        dict_args["expected_dtype_list"] = "int32 or float16 or float32"
        dict_args["dtype"] = mad_out_dtype
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    mad_shape = (fmap_n, fmap_c1, fractal_n_split, howo_mad, FRACTAL_N)
    v200_flag = _check_support_v200()
    mad_res = common.mad(mad_shape, feature_col_pad, filter_reshape, mad_out_dtype, offset_x, v200_flag)

    bias_tensor_flag = (bias_tensor is not None and bias_tensor != {})
    if bias_tensor_flag:
        bias_flag = True
        if dsl_flag is True and mad_out_dtype == "int32":
            bias_ub_brc_shape = list(mad_shape)
            bias_ub_brc_shape[3] = bias_ub_brc_shape[3] // 16
            bias_ub_brc = tvm.compute(bias_ub_brc_shape,
                                      lambda i, j, a, k, l: bias_tensor(j * 2 * 16 + a * 16 + l),
                                      name='bias_ub_brc')
            bias_l0c = tvm.compute(mad_shape,
                                   lambda i1, j1, a1, k1_val, l1: bias_ub_brc(i1, j1, a1, k1_val // 16, l1),
                                   name='bias_l0c')
            mad_res = tvm.compute(mad_shape, lambda *index: bias_l0c(*index) + mad_res(*index), name='c_col_bias')
        depthwise_cast = tvm.compute(mad_res.shape,
                                     lambda *index: mad_res(*index).astype(depthwise_res_dtype),
                                     name='depthwise_cast',
                                     attrs={
                                         'kernel_h': kernel_h,
                                         'kernel_w': kernel_w,
                                         'padding': pad,
                                         'howo_one_flag': howo_one_flag,
                                         'stride': stride,
                                         'dilation': dilation
                                     })
        if dsl_flag is True and mad_out_dtype == "int32":
            depthwise_res_bias = depthwise_cast
        else:
            depthwise_res_bias = _bias_add(depthwise_cast, bias_tensor)
    else:
        bias_flag = False
        depthwise_cast = tvm.compute(mad_res.shape,
                                     lambda *index: mad_res(*index).astype(depthwise_res_dtype),
                                     name='depthwise_cast',
                                     attrs={
                                         'kernel_h': kernel_h,
                                         'kernel_w': kernel_w,
                                         'padding': pad,
                                         'howo_one_flag': howo_one_flag,
                                         'stride': stride,
                                         'dilation': dilation
                                     })
        depthwise_res_bias = depthwise_cast

    if howo_one_flag:
        output_h = 1
        output_w = 1

    res_shape = (fmap_n, fmap_c1, fractal_n_split, output_h * output_w, FRACTAL_N)
    depthwise_res = tvm.compute(
        res_shape,
        lambda *index: depthwise_res_bias(*index).astype(depthwise_res_dtype),
        name='depthwise_res',
        tag='depthwise_conv2d',
        attrs={
            "bias_flag": bias_flag,
            "dsl_flag": dsl_flag,
            "kernel_name": kernel_name,
            "l1_fusion_para": l1_fusion_para
        },
    )
    return depthwise_res


def depthwise_conv2d_backprop_filter_d_compute(fmap,
                                               dout,
                                               kernel_h,
                                               kernel_w,
                                               stride,
                                               pad,
                                               dilations,
                                               w_dtype,
                                               kernel_name="depthwise_conv2d_compute"):
    """
    compute of depthwise conv2d backprop filter
    
    the interface will be eliminated soon!

    Parameters
    ----------
    fmap : tvm tensor
        feature map tensor in tvm.

    dout : tvm tensor
        dout tensor in tvm.

    kernel_h: int
        height of filter.

    kernel_w: int
        width of filter.

    stride: tuple or list or int
        stride of convolution.

    pad: list
        padding added to each dimension of the input.

    w_dtype: str
        the dtype of dfilter.

    Returns
    -------
    depthwise_dfilter_res: tvm tensor
        the tensor of output.
    """
    def _ceil(x_val):
        """
        Return the least multiple of 16 integer number
        which is greater than or equal to x.
        """
        return ((x_val + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    fmap_dtype = fmap.dtype
    dout_dtype = dout.dtype
    para_check.check_dtype_rule(fmap_dtype, ('float16', ))
    para_check.check_dtype_rule(dout_dtype, ('float16', ))
    para_check.check_dtype_rule(w_dtype, ('float32', ))

    fmap_shape = (int(i.value) for i in fmap.shape)
    dout_shape = (int(i.value) for i in dout.shape)

    fmap_n, fmap_cgroup, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    dout_n, _, dout_c1, dout_h, dout_w, dout_c0 = dout_shape

    stride_h, stride_w = stride
    dilation_h, dilation_w = dilations

    pad_top, pad_bottom, pad_left, pad_right = pad
    full_height = fmap_h + pad_top + pad_bottom
    full_width = fmap_w + pad_left + pad_right
    effective_filter_h = (kernel_h - 1) * dilation_h + 1
    effective_filter_w = (kernel_w - 1) * dilation_w + 1
    output_h = (full_height - effective_filter_h) // stride_h + 1
    output_w = (full_width - effective_filter_w) // stride_w + 1

    if output_h != dout_h:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d',
            'attr_name': 'output h',
            'param1_name': 'output_h',
            'param2_name': 'dout_h',
            'param1_value': str(output_h),
            'param2_value': str(dout_h)
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    if output_w != dout_w:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d',
            'attr_name': 'output w',
            'param1_name': 'output_w',
            'param2_name': 'dout_w',
            'param1_value': str(output_w),
            'param2_value': str(dout_w)
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    fmap_trans_shape = (fmap_cgroup, fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0)
    fmap_transpose = tvm.compute(fmap_trans_shape,
                                 lambda cg, n, c1, h, w, c0: fmap(n, cg, c1, h, w, c0),
                                 name='fmap_transpose')

    a_im2col_row_major_shape = (fmap_cgroup, dout_n, dout_h * dout_w, fmap_c1, kernel_h, kernel_w, BLOCK_SIZE)
    feature_col = _img2col(fmap_transpose, a_im2col_row_major_shape, kernel_h, kernel_w, pad, (stride_h, stride_w),
                           (dilation_h, dilation_w))

    dout_hw_pad = _ceil(dout_h * dout_w)

    a_im2col_fractal_shape = (fmap_cgroup, dout_n, dout_hw_pad // BLOCK_SIZE, fmap_c1 * kernel_h * kernel_w, BLOCK_SIZE,
                              BLOCK_SIZE)

    feature_col_pad = _im2col_fractal(a_im2col_fractal_shape, feature_col, dout_h, dout_w)

    dout_trans_shape = (fmap_cgroup, dout_n, dout_c1, dout_h * dout_w, dout_c0)
    dout_transpose = tvm.compute(dout_trans_shape,
                                 lambda cg, n, c1, hw, c0: dout(n, cg, c1, hw // dout_w, hw % dout_w, c0),
                                 name='dout_transpose')

    def _dout_fractal_compute(index, dout_ph):
        """Transform shape in zZ with block pad."""
        cg_val, n_val, c1_val, hw1_val, c0_val, hw0_val = index
        hw_val = hw1_val * BLOCK_SIZE + hw0_val
        return dout_ph(cg_val, n_val, c1_val, hw_val, c0_val)

    dout_fractal_shape = (fmap_cgroup, dout_n, dout_c1, dout_hw_pad // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    dout_fractal = tvm.compute(dout_fractal_shape,
                               lambda *index: _dout_fractal_compute(index, dout_transpose),
                               name='dout_fractal')

    res_dtype = "float32"
    mad_shape = (fmap_cgroup, fmap_c1 * kernel_h * kernel_w, dout_c1 * BLOCK_SIZE, BLOCK_SIZE)
    mad_res = _backprop_filter_matmul(mad_shape, dout_fractal, feature_col_pad, dout_h * dout_w, dout_n, res_dtype)

    depthwise_dfilter = tvm.compute(
        mad_shape,
        lambda cg, hw, co, c0: tvm.select(co == c0, mad_res(cg, hw, co, c0), tvm.const(0.0, mad_res.dtype)),
        name='depthwise_dfilter')

    depthwise_dfilter_res = tvm.compute(mad_shape,
                                        lambda *index: depthwise_dfilter(*index).astype(w_dtype),
                                        name='depthwise_dfilter_res',
                                        attrs={
                                            'kernel_h': kernel_h,
                                            'kernel_w': kernel_w,
                                            'padding': pad,
                                            'stride': (stride_h, stride_w),
                                            'dilations': (dilation_h, dilation_w),
                                            'kernel_name': kernel_name
                                        })

    return depthwise_dfilter_res


def depthwise_conv2d_backprop_input_d_compute(input_shape,
                                              weight,
                                              dout,
                                              weight_sizes,
                                              strides,
                                              pads,
                                              kernel_name="depthwise_conv2d_compute"):
    """
    Computes the gradients of depthwise convolution with respect to the input.

    the interface will be eliminated soon!

    Parameters
    ----------
    input_shape: a list or tuple representing the shape of input,
                6D format [N, C1, 1, H, W, C0]

    weight: a tensor, 5D with shape [C1, Hf*Wf, 1, C0, C0]

    dout: a tensor, 6D format [N, Co1, 1, Ho, Wo, C0]

    weight_sizes: a list or tuple of two ints,
                  the height and width of the weight of the convolution

    strides: a list or tuple of two ints, the stride of the sliding window for
             height and width of the input of the convolution

    pads: padding added to each dimension of the input

    Returns
    -------
    dx_res: compute of the gradients of depthwise convolution
            with respect to the input
    """
    dout_dtype = dout.dtype
    dout_shape = (int(i.value) for i in dout.shape)

    dout_n, dout_cgroup, dout_c1, dout_h, dout_w, dout_c0 = dout_shape
    stride_h, stride_w = strides
    weight_height, weight_width = weight_sizes
    input_h, input_w = input_shape[3], input_shape[4]
    pad_top, _, pad_left, _ = pads

    dilated_padded_h = input_shape[3] + weight_height - 1
    dilated_padded_w = input_shape[4] + weight_width - 1

    dilated_h = dout_h * stride_h - (stride_h - 1)
    dilated_w = dout_w * stride_w - (stride_w - 1)

    dilated_shape = (input_shape[0], input_shape[1], input_shape[2], dilated_h, dilated_w, input_shape[5])

    dilated_pad_top = weight_height - 1 - pad_top
    dilated_pad_bottom = dilated_padded_h - dilated_pad_top - dilated_h
    dilated_pad_left = weight_width - 1 - pad_left
    dilated_pad_right = dilated_padded_w - dilated_pad_left - dilated_w

    dilated_pad = (dilated_pad_top, dilated_pad_bottom, dilated_pad_left, dilated_pad_right)

    dilated_strides = (1, 1)

    dout_dilated = tvm.compute(
        dilated_shape,
        lambda n, cg, c1, h, w, c0: tvm.select(tvm.all(h % strides[0] == 0, w % strides[1] == 0), dout[
            n, cg, c1, h // strides[0], w // strides[1], c0], tvm.const(0, dout.dtype)),
        attrs={'strides': strides},
        name='dout_dilated')

    dout_im2col_row_major_shape = (dout_n, dout_cgroup, input_h * input_w, dout_c1, weight_height, weight_width,
                                   BLOCK_SIZE)
    dout_col = common.im2col_6d(dout_dilated, dout_im2col_row_major_shape, weight_height, weight_width, dilated_pad,
                                dilated_strides)

    hiwi_mad = (input_h * input_w + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

    dout_im2col_fractal_shape = (dout_n, dout_cgroup, hiwi_mad // BLOCK_SIZE, dout_c1 * weight_height * weight_width,
                                 BLOCK_SIZE, BLOCK_SIZE)

    dout_col_pad = common.im2col_fractal_6d(dout_im2col_fractal_shape, dout_col)

    weight_rotated = tvm.compute(
        weight.shape,
        lambda cg, khkw, co1, co0, c0: weight[cg, (weight_height - 1 - khkw // weight_width) * weight_width +
                                              (weight_width - 1 - khkw % weight_width), co1, co0, c0],
        name='weight_rotated')

    if not cce_conf.intrinsic_check_support("Intrinsic_mmad", "f162f32"):
        mad_out_dtype = "float16"
    else:
        mad_out_dtype = "float32"
    mad_shape = (dout_n, dout_cgroup, dout_c1, hiwi_mad, dout_c0)
    mad_res = common.mad(mad_shape, dout_col_pad, weight_rotated, mad_out_dtype)

    dx_cast = tvm.compute(mad_res.shape, lambda *index: mad_res(*index).astype(dout_dtype), name='dx_cast')

    res_shape = (dout_n, dout_cgroup, dout_c1, input_h * input_w, dout_c0)
    dx_res = tvm.compute(res_shape,
                         lambda *index: dx_cast(*index).astype(dout_dtype),
                         name='dx_res',
                         attrs={
                             'weight_height': weight_height,
                             'weight_width': weight_width,
                             'dilated_pad': dilated_pad,
                             'dilated_strides': dilated_strides,
                             'kernel_name': kernel_name
                         })
    return dx_res
