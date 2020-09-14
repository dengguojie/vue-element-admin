#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache Licenses for more details at
http://www.apache.org/licenses/LICENSE-2.0

pooling3d compute
"""

import math
from te import tvm
from te.platform.cce_conf import get_soc_spec

POOL3D_TAG = "pooling3d_"
CAFFE_DATA_MODE = 0
TENSORFLOW_DATA_MODE = 1

MIN_VAL_MAP = {"float16": -65504.0, "float32": 3.4e-38, "double": 1.7e-308}
SIZEOF_DTYPE_MAP = {"float16": 2, "float32": 4, "double": 8}
C0_DIMENSION_DATA_SIZE_MAP = {"float16": 32, "float32": 64, "double": 128}

# 'pylint: disable=too-many-locals, too-many-arguments, invalid-name
# 'pylint: disable=unused-argument,too-many-statements, cell-var-from-loop
def pooling3d(tensor_in, window, stride, padding_mode="SAME",
              pads=(0, 0, 0, 0, 0, 0),
              pooling_mode="MAX", dilation=(1, 1, 1), ceil_mode=0):
    """
    :params:
    :tensor_in: input tensor
    :window: input window
    :stride: window move steps in d/h/w dimension
    :padding_mode: can be SAME, VALID
    :pads: padFT, padBK,padT,padB,padL,padR, used for caffe,all zero with tf
    :pooling_mode: can be MAX, (AVG, GAP, GMP -- Not support yet)
    :dilation: params to be reserved, use default value
    :ceil_mode : caffe round_mode params, 0:CEIL(default), 1:FLOOR
    :return: pooling result
    """
    data_mode = TENSORFLOW_DATA_MODE

    def _select(indices):
        i_n, i_c1, i_d = indices[0], indices[1], indices[2]
        i_h, i_w, i_c0 = indices[3], indices[4], indices[5]
        conds = [i_d >= p_ft, i_d < d_p - p_bk,
                 i_h >= p_t, i_h < h_p - p_b,
                 i_w >= p_l, i_w < w_p - p_r]
        t_b = tvm.select(conds[5], tensor_in[i_n, i_d - p_ft,
                                             i_c1, i_h - p_t,
                                             i_w - p_l, i_c0])
        t_b = tvm.select(conds[4], t_b)
        t_b = tvm.select(conds[3], t_b)
        t_b = tvm.select(conds[2], t_b)
        t_b = tvm.select(conds[1], t_b)
        return tvm.select(conds[0], t_b)

    def _pad(cond):
        min_val = tvm.const(MIN_VAL_MAP[tensor_in.dtype],
                            dtype=tensor_in.dtype)
        return tvm.select(cond, min_val)

    def _fake(i):
        return tx_ub_c[i] + tx_ub_ft[i] + tx_ub_bk[i] + \
               tx_ub_t[i] + tx_ub_b[i] + tx_ub_l[i] + tx_ub_r[i]

    # define dict to transfer pooling params
    pooling_params = {}

    # get shape info of feature map in NDC1HWC0 format
    n = tensor_in.shape[0].value
    d = tensor_in.shape[1].value
    c1 = tensor_in.shape[2].value
    h = tensor_in.shape[3].value
    w = tensor_in.shape[4].value
    c0 = tensor_in.shape[5].value

    k_d, k_h, k_w = window[0], window[1], window[2]
    s_d, s_h, s_w = stride[0], stride[1], stride[2]

    if data_mode == TENSORFLOW_DATA_MODE:
        o_d, o_h, o_w, p_ft, p_bk, p_t, p_b, p_l, p_r = \
            _get_tensorflow_out_and_pad(padding_mode, d, h, w,
                                        window, stride, dilation)
        d_p, h_p, w_p = (o_d - 1) * s_d + k_d, \
                        (o_h - 1) * s_h + k_h, \
                        (o_w - 1) * s_w + k_w
    elif data_mode == CAFFE_DATA_MODE:
        pass

    shape_trans = (n, c1, d_p, h_p, w_p, c0)
    tx_ub_c = tvm.compute(shape_trans, lambda *i: _select(i),
                          name="tx_ub_c", tag="tx_ub_c")
    tx_ub_ft = tvm.compute(shape_trans, lambda *i: _pad(i[2] < p_ft),
                           name="tx_ub_ft", tag="tx_ub_ft")
    tx_ub_bk = tvm.compute(shape_trans, lambda *i: _pad(i[2] >= d_p - p_bk),
                           name="tx_ub_bk", tag="tx_ub_bk")
    tx_ub_t = tvm.compute(shape_trans, lambda *i: _pad(i[3] < p_t),
                          name="tx_ub_t", tag="tx_ub_t")
    tx_ub_b = tvm.compute(shape_trans, lambda *i: _pad(i[3] >= h_p - p_b),
                          name="tx_ub_b", tag="tx_ub_b")
    tx_ub_l = tvm.compute(shape_trans, lambda *i: _pad(i[4] < p_l),
                          name="tx_ub_l", tag="tx_ub_l")
    tx_ub_r = tvm.compute(shape_trans, lambda *i: _pad(i[4] >= w_p - p_r),
                          name="tx_ub_r", tag="tx_ub_r")
    tx_ub = tvm.compute(shape_trans, lambda *i: _fake(i),
                        name="tx_ub", tag="tx_ub")

    tx_rw = _reduce_w(n, c1, d_p, h_p, o_w, c0, s_w, k_w, tx_ub)
    tx_rh = _reduce_h(n, c1, d_p, o_h, o_w, c0, s_h, k_h, tx_rw)
    tx_rd = _reduce_d(n, c1, o_d, o_h, o_w, c0, s_d, k_d, tx_rh)

    shape = (n, o_d, c1, o_h, o_w, c0)
    pooling_params = {}
    pooling_params["batch_size"] = n
    pooling_params["c1_value"] = c1
    pooling_params["in_size_d"] = d
    pooling_params["in_size_h"] = h
    pooling_params["in_size_w"] = w
    pooling_params["c0_value"] = c0
    pooling_params["out_size_d"] = o_d
    pooling_params["out_size_h"] = o_h
    pooling_params["out_size_w"] = o_w
    pooling_params["window_d"] = k_d
    pooling_params["window_h"] = k_h
    pooling_params["window_w"] = k_w
    pooling_params["stride_d"] = s_d
    pooling_params["stride_h"] = s_h
    pooling_params["stride_w"] = s_w
    pooling_params["size_of_data"] = SIZEOF_DTYPE_MAP[tensor_in.dtype]
    pooling_params["dtype"] = tensor_in.dtype
    # copy ub to gm
    res = tvm.compute(
        shape,
        lambda *i: tx_rd[i[0], i[2], i[1], i[3], i[4], i[5]],
        name=POOL3D_TAG + "max_res",
        tag=POOL3D_TAG + "max",
        attrs={"pooling_params": pooling_params,
               "template": "max_pool3d_generic"}
    )

    return res


def _reduce_w(n, c1, d_p, h_p, o_w, c0, s_w, k_w, tx_ub_c):
    shape = (n, c1, d_p, h_p, o_w, c0)

    if k_w == 1:
        tx_rw = tvm.compute(shape,
                            lambda *i: tx_ub_c[i[0], i[1], i[2], i[3], i[4] * s_w, i[5]],
                            name="tx_rw1", tag="reduce_max")
        return tx_rw

    tx_rw1 = tvm.compute(
        shape,
        lambda *i: tvm.max(tx_ub_c[i[0], i[1], i[2], i[3], i[4] * s_w, i[5]],
                           tx_ub_c[i[0], i[1], i[2], i[3], i[4] * s_w + 1, i[5]]),
        name="tx_rw1", tag="reduce_max"
    )
    tx_rw = tx_rw1
    for j in range(2, k_w):
        tx_rwx = tvm.compute(
            shape,
            lambda *i: tvm.max(tx_ub_c[i[0], i[1], i[2], i[3], i[4] * s_w + j, i[5]],
                               tx_rw[i[0], i[1], i[2], i[3], i[4], i[5]]),
            name="tx_rw" + str(j),
            tag="reduce_max"
        )
        tx_rw = tx_rwx
    return tx_rw


def _reduce_h(n, c1, d_p, o_h, o_w, c0, s_h, k_h, tx_rw):
    shape = (n, c1, d_p, o_h, o_w, c0)

    if k_h == 1:
        tx_rh = tvm.compute(shape,
                            lambda *i: tx_rw[i[0], i[1], i[2], i[3] * s_h, i[4], i[5]],
                            name="tx_rh1", tag="reduce_max")
        return tx_rh

    tx_rh1 = tvm.compute(
        shape,
        lambda *i: tvm.max(tx_rw[i[0], i[1], i[2], i[3] * s_h, i[4], i[5]],
                           tx_rw[i[0], i[1], i[2], i[3] * s_h + 1, i[4], i[5]]),
        name="tx_rh1",
        tag="reduce_max"
    )
    tx_rh = tx_rh1
    for j in range(2, k_h):
        tx_rhx = tvm.compute(
            shape,
            lambda *i: tvm.max(tx_rw[i[0], i[1], i[2], i[3] * s_h + j, i[4], i[5]],
                               tx_rh[i[0], i[1], i[2], i[3], i[4], i[5]]),
            name="tx_rh" + str(j),
            tag="reduce_max"
        )
        tx_rh = tx_rhx
    return tx_rh


def _reduce_d(n, c1, o_d, o_h, o_w, c0, s_d, k_d, tx_rh):
    shape = (n, c1, o_d, o_h, o_w, c0)

    if k_d == 1:
        tx_rd = tvm.compute(shape,
                            lambda *i: tx_rh[i[0], i[1], i[2] * s_d, i[3], i[4], i[5]],
                            name="tx_rd1", tag="reduce_max")
        return tx_rd

    tx_rd1 = tvm.compute(
        shape,
        lambda *i: tvm.max(tx_rh[i[0], i[1], i[2] * s_d, i[3], i[4], i[5]],
                           tx_rh[i[0], i[1], i[2] * s_d + 1, i[3], i[4], i[5]]),
        name="tx_rd1",
        tag="reduce_max"
    )
    tx_rd = tx_rd1
    for j in range(2, k_d):
        tx_rdx = tvm.compute(
            shape,
            lambda *i: tvm.max(tx_rh[i[0], i[1], i[2] * s_d + j, i[3], i[4], i[5]],
                               tx_rd[i[0], i[1], i[2], i[3], i[4], i[5]]),
            name="tx_rd" + str(j),
            tag="reduce_max"
        )
        tx_rd = tx_rdx
    return tx_rd


def _get_caffe_out_size_and_pad():
    pass


def _get_tensorflow_out_and_pad(padding_mode, in_size_d, in_size_h, in_size_w,
                                window, stride, dilation):
    """
    :param padding_mode: can be SAME, VALID
    :param in_size_d: input tensor
    :param in_size_h: input tensor
    :param in_size_w: input tensor
    :param window: input window d/h/w
    :param stride: stride d/h/w
    :param dilation: dilation d/h/w
    :param pad_top: pad top
    :param pad_bottom: pad bottom
    :param pad_left: pad left
    :param pad_right: pad right
    :return:
    """
    out_size_d, out_size_h, out_size_w = 0, 0, 0
    pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0, 0, 0
    dilation = (1, 1, 1)
    if padding_mode == "SAME":
        # 'caculate output size in SAME mode
        # 'Dout = ceil(Di, Sd)
        # 'Hout = ceil(Hi, Sh)
        # 'Wout = ceil(Wi, Sw)
        out_size_d = (in_size_d + stride[0] - 1) // stride[0]
        out_size_h = (in_size_h + stride[1] - 1) // stride[1]
        out_size_w = (in_size_w + stride[2] - 1) // stride[2]

        # 'Total padding on rows and cols is
        # 'Pd = (D' - 1) * S + (Kd - 1) * Dd + 1 - D
        # 'Ph = (H' - 1) * S + (Kh - 1) * Dh + 1 - H
        # 'Pw = (W' - 1) * S + (Kw - 1) * Dw + 1 - W
        # 'where (D', H', W') are output dimensions, (D, H, W) are input dims.
        # 'S is stride, (Dd, Dh, Dw) are dilations, (Kd, Kh, Kw) are filter dims.
        # get total pad_d, pad_h, pad_w
        pad_d = (out_size_d - 1) * stride[0] + \
                ((window[0] - 1) * dilation[0] + 1) - in_size_d
        pad_h = (out_size_h - 1) * stride[1] + \
                ((window[1] - 1) * dilation[1] + 1) - in_size_h
        pad_w = (out_size_w - 1) * stride[2] + \
                ((window[2] - 1) * dilation[2] + 1) - in_size_w
        if pad_d < 0:
            pad_d = 0
        if pad_h < 0:
            pad_h = 0
        if pad_w < 0:
            pad_w = 0
        pad_front = pad_d // 2
        pad_back = math.ceil(pad_d / 2)
        pad_top = pad_h // 2
        pad_bottom = math.ceil(pad_h / 2)
        pad_left = pad_w // 2
        pad_right = math.ceil(pad_w / 2)

    else:  # 'padding_mode == "VALID":
        # 'caculate output size in VALID mode
        # 'Dout = ceil(Hi - Fd + 1, Sd)
        # 'Hout = ceil(Hi - Fh + 1, Sh)
        # 'Wout = ceil(Wi - Fw + 1, Sw)
        out_size_d = (in_size_d - window[0] + 1 + (stride[0] - 1)) // stride[0]
        out_size_h = (in_size_h - window[1] + 1 + (stride[1] - 1)) // stride[1]
        out_size_w = (in_size_w - window[2] + 1 + (stride[2] - 1)) // stride[2]

    return out_size_d, out_size_h, out_size_w, pad_front, pad_back, pad_top, \
           pad_bottom, pad_left, pad_right


def _check_ub_tiling(window_d, window_h, window_w, pooling_mode, dtype):
    if pooling_mode == "MAX":
        data_size = _get_ub_least_data_size_for_max(window_d, window_h, window_w)
    else:
        raise RuntimeError("Not suport pooling_mode yet.")

    if data_size > get_soc_spec("UB_SIZE"):
        raise RuntimeError("Window size greater than UB.")


def _get_ub_least_data_size_for_max(window_d, window_h, window_w, dtype):
    c0_size = C0_DIMENSION_DATA_SIZE_MAP[dtype]
    fmap_size = window_d * window_h * window_w * c0_size
    reduce_intermediate_data = (window_d * window_h * c0_size) + \
                               (window_d * c0_size)
    res_size = c0_size
    data_size = fmap_size + reduce_intermediate_data + res_size

    return data_size
