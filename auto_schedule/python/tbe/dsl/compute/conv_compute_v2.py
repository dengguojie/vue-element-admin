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
Compute of Conv2d in v220.
"""
from __future__ import division
from enum import Enum
from tbe import tvm
from tbe.common.platform import CUBE_MKN
from tbe.common.utils.errormgr import error_manager_cube as err_man
import pdb

OP_TAG = "convolution_"

def ceil_div(num_a, num_b):
    """
    Do upper division.
    """
    if num_b == 0:
        err_man.raise_err_specific("conv2d", "num_b == 0")
    return (num_a + num_b - 1) // num_b


class ConvType(Enum):
    """
    Enum class of conv dtype.
    """
    FP16 = 1
    INT8 = 2
    FP32 = 3
    BF16 = 4


def conv_v220_compute(fmap, weight, para_dict, optim_dict, dsl_flag, conv_param):
    """
    Compute of conv2d in v220.
    """
    def al1_compute(fmap):
        """
        Compute of al1.
        """
        if dynamic_flag:
            fmap_l1_shape = (group_opt,
                             batch,
                             ci1_opt,
                             (in_height - 1) // stride_h + 1 if strideh_opti_flag else in_height,
                             in_width,
                             in_c0)
            fmap_l1 = tvm.compute(
                fmap_l1_shape,
                lambda group_idx, n_idx, ci1_idx, hi_idx, wi_idx, ci0_idx:
                fmap(n_idx,
                     ci1_idx + group_idx*ci1_opt,
                     hi_idx*stride_h if strideh_opti_flag else hi_idx,
                     wi_idx,
                     ci0_idx),
                name="fmap_l1",
                tag=OP_TAG + "fmap_l1")

            conv_param.tensor_map["fmap_l1"] = fmap_l1
            return fmap_l1

        #===========================strideh optimization=====================================
        if strideh_opti_flag:
            fmap_l1_h = (in_height - 1) // stride_h + 1
            fmap_l1_shape = batch, in_c1, fmap_l1_h, in_width, in_c0

            fmap_l1 = tvm.compute(
                fmap_l1_shape,
                lambda n_idx, ci1_idx, hi_idx, wi_idx, ci0_idx:
                fmap(n_idx,
                     ci1_idx,
                     hi_idx*stride_h,
                     wi_idx,
                     ci0_idx),
                name="fmap_l1",
                tag=OP_TAG + "fmap_l1")

            conv_param.tensor_map["fmap_l1"] = fmap_l1
            return fmap_l1

        #=========================C0=4====================================================
        if c04_mode == "not_first_layer_c04":
            # [N, C1, H, W, C0] —> [N, C1, H, W, 4]
            fmap_l1_c04_shape = batch, 1, in_height, in_width, 4
            fmap_l1_c04 = tvm.compute(
                fmap_l1_c04_shape,
                lambda n_idx, ci1_idx, hi_idx, wi_idx, ci0_idx:
                fmap(n_idx,
                     ci1_idx,
                     hi_idx,
                     wi_idx,
                     ci0_idx),
                name="fmap_l1_c04",
                tag=OP_TAG + "fmap_l1_c04")

            conv_param.tensor_map["fmap_l1_c04"] = fmap_l1_c04
            return fmap_l1_c04

        return fmap

    def dynamic_l0a_compute(fmap_l1):
        """
        Compute of al0 in dynamic shape.
        """
        def im2col(fmap, im2col_shape, im2col_para):
            """
            Calculate im2col result.
            """
            block_size = 16
            fmap, kernel_h, kernel_w, padding, stride, dilate, out_width = im2col_para

            def _im2col_idx(idx):
                """
                Calculate im2col result main compute.
                """
                group_idx, n_idx, m1_idx, k1_idx, m0_idx, k0_idx = idx

                virtual_h = m1_idx*block_size + m0_idx
                virtual_w = k1_idx*block_size + k0_idx
                dilate_h, dilate_w = dilate

                back_c1 = virtual_w // block_size // kernel_w // kernel_h
                back_h = (virtual_h // out_width)*stride[0] + (k1_idx // kernel_w % kernel_h)*dilate_h
                back_w = (virtual_h % out_width)*stride[1] + (k1_idx % kernel_w)*dilate_w

                return tvm.select(tvm.any(back_h < padding[0],
                                          back_h > fmap.shape[3] + padding[0] - 1,
                                          back_w < padding[2],
                                          back_w > fmap.shape[4] + padding[2] - 1),
                                  tvm.const(0, fmap.dtype),
                                  fmap(group_idx,
                                       n_idx,
                                       back_c1,
                                       back_h - padding[0],
                                       back_w - padding[2],
                                       k0_idx))
            return tvm.compute(im2col_shape,
                               lambda *idx: _im2col_idx(idx),
                               name='im2col_fractal_v2',
                               tag=OP_TAG + 'im2col_fractal_v2',
                               attrs={'fmap_shape': fmap.shape,
                                      'kernel_h': kernel_h,
                                      'kernel_w': kernel_w,
                                      'padding': padding,
                                      'stride': stride})

        im2col_shape = (group_opt,
                        batch,
                        howo_mad // block_m0,
                        ci1_opt*kernel_h*kernel_w,
                        block_m0,
                        block_k0)

        im2col_para = (fmap_l1,
                       kernel_h,
                       kernel_w,
                       padding, # (pad_top, pad_bottom, pad_left, pad_right)
                       (1, stride_w) if strideh_opti_flag else stride,
                       dilate,
                       out_width)

        fmap_im2col = im2col(fmap_l1, im2col_shape, im2col_para)
        return fmap_im2col

    def row_major_compute(fmap_l1):
        """
        Compute of row major tensor.
        """
        def im2col_row_major(fmap_row_major_shape, fmap_l1, kernel_w, padding, stride, dilate, compute_dtype):
            def __im2col_row_major_indices(group, batch, howo, cin_1, k_h, k_w,
                                           cin_0, fmap_l1, kernel_w, padding,
                                           stride, dilate):
                _, _, in_height, in_weight, _ = fmap_l1.shape
                stride_h, stride_w = stride
                dilate_h, dilate_w = dilate
                padding_top, _, padding_left, padding_right = padding
                width_out = (in_weight.value + padding_left + padding_right -
                             ((kernel_w - 1)*dilate_w + 1)) // (stride_w) + 1

                h_index = (howo // width_out)*stride_h + k_h*dilate_h
                w_index = (howo % width_out)*stride_w + k_w*dilate_w
                if conv_param.l0a_dma_flag:
                    return fmap_l1(batch,
                                   cin_1 + group*ci1_opt,
                                   h_index - padding_top,
                                   w_index - padding_left,
                                   cin_0)
                return tvm.select(
                    tvm.any(h_index < padding_top,
                            h_index > in_height.value + padding_top - 1,
                            w_index < padding_left,
                            w_index > in_weight.value + padding_left - 1),
                    tvm.const(offset_x, compute_dtype),
                    fmap_l1(batch,
                            cin_1 + group*ci1_opt,
                            h_index - padding_top,
                            w_index - padding_left,
                            cin_0))

            return tvm.compute(fmap_row_major_shape,
                               lambda group, batch, howo, cin_1, k_h, k_w, cin_0:
                               __im2col_row_major_indices(
                                   group, batch, howo, cin_1, k_h, k_w, cin_0,
                                   fmap_l1, kernel_w, padding, stride, dilate),
                               name='im2col_row_major',
                               tag=OP_TAG + 'im2col_row_major')

        def im2col_row_major_reshape(row_major_reshape_shape, fmap_row_major, compute_dtype):
            """
            Merge the in_c1, kernel_h, kernel_w, in_c0 axes into k axis.
            """
            _, _, out_hw, in_c1, kernel_h, kernel_w, in_c0 = fmap_row_major.shape
            row_major_reshape = tvm.compute(
                row_major_reshape_shape,
                lambda group_idx, n_idx, howo_idx, k_idx:
                tvm.select(
                    tvm.all(k_idx < in_c1*kernel_h*kernel_w*in_c0,
                            howo_idx < out_hw,
                            group_idx*k1_size + k_idx < reduce_value),
                    fmap_row_major(group_idx,
                                   n_idx,
                                   howo_idx,
                                   k_idx // (kernel_h*kernel_w*in_c0),
                                   k_idx // (kernel_w*in_c0) % kernel_h,
                                   k_idx // (in_c0) % (kernel_w),
                                   k_idx % in_c0),
                    tvm.const(0.0, compute_dtype)),
                name="row_major_reshape",
                tag=OP_TAG + 'row_major_reshape')
            return row_major_reshape

        fmap_row_major_shape = (group_opt,
                                batch,
                                out_height*out_width,
                                ci1_opt,
                                kernel_h,
                                kernel_w,
                                row_major_c0)
        if strideh_opti_flag:
            stride = 1, stride_w
        else:
            stride = stride_h, stride_w

        fmap_row_major = im2col_row_major(fmap_row_major_shape, fmap_l1,
                                          kernel_w, padding, stride,
                                          dilate, fmap_dtype)

        row_major_reshape_shape = (group_opt,
                                   batch,
                                   howo_mad,
                                   k_size)
        fmap_row_major_reshape = im2col_row_major_reshape(row_major_reshape_shape, fmap_row_major, fmap_dtype)

        return fmap_row_major, fmap_row_major_reshape

    def l0a_compute(l0a_src):
        """
        Compute of al0.
        """
        def im2col_fractal_compute(fmap_im2col_shape, fmap_row_major_reshape):
            """
            Generate im2col_fractal in L0A.
            """
            res_im2col_fractal = tvm.compute(
                fmap_im2col_shape,
                lambda group_idx, n_idx, m1_idx, k1_idx, m0_idx, k0_idx:
                fmap_row_major_reshape(group_idx,
                                       n_idx,
                                       m1_idx*block_m0 + m0_idx,
                                       k1_idx*block_k0 + k0_idx),
                name="im2col_fractal",
                tag=OP_TAG + 'im2col_fractal')
            return res_im2col_fractal

        fmap_row_major_reshape = l0a_src
        fmap_l0_shape = (group_opt,
                         batch,
                         howo_mad // block_m0,
                         k1_size,
                         block_m0,
                         block_k0)
        fmap_im2col_res = im2col_fractal_compute(fmap_l0_shape, fmap_row_major_reshape)
        return fmap_im2col_res

    def l0c_compute(fmap_im2col, bias_tensor=None):
        # CL0's M is not aligned by block_m0.
        mad_shape = (group_opt, batch, co1_opt, out_height*out_width, block_n0)
        weight_k1, _, _, _ = weight_fracz_shape

        reduce_k1 = weight_k1 // group_opt

        axis_k1 = tvm.reduce_axis((0, reduce_k1), name='cin_1_kh_kw')
        axis_k0 = tvm.reduce_axis((0, block_k0), name='cin_0')

        c_col = tvm.compute(
            mad_shape,
            lambda group_idx, batch_idx, co1_idx, howo_idx, co0_idx:
            tvm.sum(
                tvm.select(
                    tvm.all((group_idx*reduce_k1 + axis_k1)*block_k0 + axis_k0 < reduce_value),
                    ((fmap_im2col[group_idx,
                                  batch_idx,
                                  howo_idx // block_m0,
                                  axis_k1,
                                  howo_idx % block_m0,
                                  axis_k0]) *
                     weight[group_idx*reduce_k1 + axis_k1,
                            co1_idx,
                            co0_idx,
                            axis_k0]).astype(mad_dtype)),
                axis=[axis_k1, axis_k0]),
            name='mad1',
            tag=OP_TAG + "c_col")

        return c_col

    def res_compute(cl0):
        """
        Compute of conv2d res. merge group axis && remove C padding.
        """
        # L0C —> out / L1
        res = tvm.compute(
            res_shape,
            lambda n_idx, co1_idx, howo_idx, co0_idx:
            cl0(0 if group == 1 else co1_idx // co1_opt,
                n_idx,
                co1_idx if group == 1 else co1_idx % co1_opt,
                howo_idx,
                co0_idx).astype(res_dtype),
            name='res_conv2d',
            tag=OP_TAG + "res_conv2d",
            attrs={"conv_shape": conv_param.dim_map["output_conv_res_shape"]})

        if conv_type == ConvType.FP32:
            # split channel + remove pad
            res_fp32 = tvm.compute(
                res_fp32_shape,
                lambda n_idx, co1_idx, howo_idx, co0_idx:
                res(n_idx,
                    co1_idx // 2,
                    howo_idx,
                    8*(co1_idx % 2) + co0_idx),
                name='res_fp32_conv2d',
                tag=OP_TAG + "res_fp32_conv2d")
            return res_fp32

        return res

    #=================parse_parameters===========================
    # common parameters
    fmap_dtype, weight_dtype = fmap.dtype, weight.dtype

    bias_tensor = para_dict["bias_tensor"]

    kernel_h, kernel_w = para_dict["filter_h"], para_dict["filter_w"]
    pad_top, pad_bottom = para_dict["pad_h"]
    pad_left, pad_right = para_dict["pad_w"]
    stride_h, stride_w = para_dict["stride_h"], para_dict["stride_w"]
    dilate_h, dilate_w = para_dict["dilate_h"], para_dict["dilate_w"]
    offset_x = para_dict["offset_x"]

    padding = pad_top, pad_bottom, pad_left, pad_right
    stride = stride_h, stride_w
    dilate = dilate_h, dilate_w

    out_height = conv_param.h_out
    out_width = conv_param.w_out

    # group conv parameters
    group = para_dict["group"]
    ci1_opt, co1_opt, group_opt = para_dict["c1_opt"], para_dict["cout1_opt"], para_dict["group_opt"]

    # shape
    fmap_5hd_shape = para_dict["a_shape"]
    weight_fracz_shape = para_dict["weight_fracz_shape"]
    weight_ori_shape_nchw = para_dict["weight_ori_shape_nchw"]

    # lxfusion parameters
    lxfusion_para = conv_param.fusion_para

    # flag
    dynamic_flag = conv_param.dynamic_flag

    #======================config fractal unit size=======================================
    block_m0, block_k0, block_n0 = CUBE_MKN[fmap.dtype]["mac"]

    #========================config conv type========================================
    conv_type_dict = {
        "float16": ConvType.FP16,
        "bfloat16": ConvType.BF16,
        "float32": ConvType.FP32,
        "int8": ConvType.INT8,
    }
    conv_type = conv_type_dict[fmap_dtype]

    mad_type_dict = {
        ConvType.FP16: "float32",
        ConvType.BF16: "float32",
        ConvType.FP32: "float32",
        ConvType.INT8: "int32",
    }
    mad_dtype = mad_type_dict[conv_type]

    res_dtype_dict = {
        ConvType.FP16: "float16",
        ConvType.BF16: "bfloat16",
        ConvType.FP32: "float32",
        ConvType.INT8: "int32",
    }
    res_dtype = res_dtype_dict[conv_type]

    #================config various optimization flag================================
    # strided_read_flag
    if fmap.op.tag == "strided_read":
        conv_param.strided_read_flag = True

    # aipp_fuse_flag
    if fmap.op.tag == "aipp_res_convolution":
        conv_param.aipp_fuse_flag = True

    # c04_mode
    c04_mode = optim_dict.get("v220_c04_mode", "disabled")
    c04_flag = optim_dict.get("v220_c04_mode") != "disabled"

    # strideh_opti_flag
    strideh_opti_flag = (kernel_h == 1 and stride_h > 1) and padding == (0, 0, 0, 0) and not c04_flag

    if lxfusion_para["l1_fusion_type"] == 1 or lxfusion_para["input_memory_type"][0] == 1:
        # for L1 breadth fusion, fmap must load all at once
        strideh_opti_flag = False

    # input_nd_flag
    input_nd_flag = fmap.op.tag == "NHWC_trans_5HD"

    if input_nd_flag: # to be completed
        strideh_opti_flag = False

    #======================calculate certain shape params for compute to use=================
    batch, in_c1, in_height, in_width, in_c0 = fmap_5hd_shape

    # M
    howo_mad = (out_height*out_width + block_m0 - 1) // block_m0*block_m0

    # K
    row_major_c0 = 4 if c04_flag else in_c0
    k1_size = (in_c1*kernel_h*kernel_w*row_major_c0 + block_k0 - 1) // block_k0
    k_size = (in_c1*kernel_h*kernel_w*in_c0 + block_k0 - 1) // block_k0*block_k0

    reduce_value = in_c1*kernel_h*kernel_w*block_k0

    out_c1 = ceil_div(weight_ori_shape_nchw[0], block_n0)

    res_shape = batch, out_c1, out_height*out_width, block_n0
    if conv_type == ConvType.FP32:
        res_fp32_shape = batch, out_c1*2, out_height*out_width, 8

    #============================check parameters=========================================
    if c04_flag and kernel_h*kernel_w*4 > 65535:
        err_man.raise_err_specific(
            "conv2d",
            "In v220, small channel case, the 4*Hk*Wk must be smaller than " +
            "or equal to 65535. you can try to disable the small channel.")

    #==========================conv compute begin==============================
    fmap_l1 = al1_compute(fmap)

    if dynamic_flag:
        fmap_im2col = dynamic_l0a_compute(fmap_l1)
    else:
        fmap_row_major, fmap_row_major_reshape = row_major_compute(fmap_l1)
        fmap_im2col = l0a_compute(fmap_row_major_reshape)

    cl0 = l0c_compute(fmap_im2col, bias_tensor)
    conv_res = res_compute(cl0)

    #===========================update tensormap=============================
    update_tensormap = {
        "cl0": cl0,
        "filter": weight,
    }
    if dynamic_flag:
        update_tensormap.update({
            "fmap_im2col": fmap_im2col
        })
    else:
        update_tensormap.update({
            "fmap_row_major": fmap_row_major,
            "fmap_row_major_reshape": fmap_row_major_reshape,
            "fmap_im2col": fmap_im2col
        })

    if strideh_opti_flag or dynamic_flag:
        update_tensormap.update({"fmap_l1": fmap_l1})

    if conv_param.strided_read_flag or conv_param.aipp_fuse_flag:
        update_tensormap.update({"fmap": fmap.op.input_tensors[0]})
    else:
        update_tensormap.update({"fmap": fmap})

    conv_param.tensor_map.update(update_tensormap)

    #===========================update dimmap=============================
    filter_matrix_dim = (
        (ci1_opt*in_c0*kernel_h*kernel_w + block_k0 - 1) // block_k0,
        (out_c1*block_n0 + block_n0 - 1) // block_n0,
        block_n0,
        block_k0)

    fmap_single_group_shape = batch, ci1_opt, in_height, in_width, in_c0

    update_dimmap = {
        "img_shape": fmap_single_group_shape,
        "filter_matrix_dim": filter_matrix_dim,
        "fmap_5hd_shape": fmap_5hd_shape,
    }

    conv_param.dim_map.update(update_dimmap)

    #=======================update tiling_query_param========================
    c_shape = [batch, co1_opt, out_height, out_width, block_n0]
    conv_param.tiling_query_param.update({"c_shape": c_shape})

    #======================update flag=======================================
    conv_param.v220_c04_mode = c04_mode
    conv_param.strideh_opti_flag = strideh_opti_flag
    conv_param.input_nd_flag = input_nd_flag

    #==============save tiling_info_dict for conv2d_tiling_case=============
    tiling_query_param = conv_param.tiling_query_param
    conv_param.tiling_info_dict = {
        "op_type": 'conv2d',
        "a_shape": fmap_5hd_shape,
        "placeholder_fmap_5hd_shape": fmap_5hd_shape,
        "b_shape": list(tiling_query_param["shape_w_nc1hwc0"]),
        "c_shape": tiling_query_param["c_shape"],
        "a_dtype": fmap_dtype,
        "b_dtype": weight_dtype,
        "c_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "pad": padding,
        "stride": stride,
        "dilation": dilate,
        "group": 1,
        "bias_flag": tiling_query_param["bias_flag"],
        "fused_coefficient": [0, 0, 0],
        "fused_channel_wise": [0, 0, 0],
        "in_fm_memory_type": [],
        "out_fm_memory_type": [],
        "l1_fusion_type": -1,
        "fusion_type": 0,
        # "reserved_ub": 0,
        "kernel_name": para_dict.get("kernel_name"),
        "dynamic_shape_flag": True
        }

    return conv_res
