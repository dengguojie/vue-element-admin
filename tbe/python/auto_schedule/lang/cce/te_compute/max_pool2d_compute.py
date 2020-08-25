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

pooling2d compute
"""

from te import tvm

POOL2D_TAG = "pooling2d_"


def max_pool2d(t_x, howo, pad, pooling_params):
    """
    Generic vector template
    :param t_x:
    :param howo:
    :param pad:
    :param pooling_params:
    :return:
    """
    x_n, x_c1, _, _, x_c0 = t_x.shape
    pooling_mode = pooling_params["pooling_mode"]
    if pooling_mode in ["GMP"]:
        k_h, k_w = pooling_params["in_size_h"], pooling_params["in_size_w"]
    else:

        k_h, k_w = pooling_params["window_h"], pooling_params["window_w"]
    s_h, s_w = pooling_params["stride_h"], pooling_params["stride_w"]
    dtype = t_x.dtype
    o_h, o_w = howo
    # h, w with pad
    h_p, w_p = (o_h - 1)*s_h + k_h, (o_w - 1)*s_w + k_w
    p_t, p_b, p_l, p_r = pad

    def _select(indices):
        i_n, i_c1, i_c0 = indices[0], indices[1], indices[4]
        i_h, i_w = indices[2], indices[3]
        conds = [i_h >= p_t, i_h < h_p - p_b, i_w >= p_l, i_w < w_p - p_r]
        t_b = tvm.select(conds[3], t_x[i_n, i_c1, i_h - p_t, i_w - p_l, i_c0])
        t_b = tvm.select(conds[2], t_b)
        t_b = tvm.select(conds[1], t_b)
        return tvm.select(conds[0], t_b)

    def _pad(cond):
        return tvm.select(cond, fp16_min)

    def _fake(i):
        return tx_ub_c[i] + tx_ub_t[i] + tx_ub_b[i] + tx_ub_l[i] + tx_ub_r[i]

    # copy gm to ub with padding
    shape = (x_n, x_c1, h_p, w_p, x_c0)
    fp16_min = tvm.const(-65504.0, dtype=dtype)
    tx_ub_c = tvm.compute(shape, lambda *i: _select(i), name="tx_ub_c")
    tx_ub_t = tvm.compute(shape, lambda *i: _pad(i[2] < p_t), name="tx_ub_t")
    tx_ub_b = tvm.compute(shape, lambda *i: _pad(i[2] >= h_p - p_b),
                          name="tx_ub_b")
    tx_ub_l = tvm.compute(shape, lambda *i: _pad(i[3] < p_l), name="tx_ub_l")
    tx_ub_r = tvm.compute(shape, lambda *i: _pad(i[3] >= w_p - p_r),
                          name="tx_ub_r")
    tx_ub = tvm.compute(shape, lambda *i: _fake(i), name="tx_ub")

    # reduce w
    shape = (x_n, x_c1, h_p, o_w, x_c0)
    if k_w > 1:
        tx_rw1 = tvm.compute(
            shape,
            lambda *i: tvm.max(tx_ub[i[0], i[1], i[2], i[3]*s_w, i[4]],
                               tx_ub[i[0], i[1], i[2], i[3]*s_w + 1, i[4]]),
            name="tx_rw1",
            tag="reduce_max"
        )
        tx_rw = tx_rw1
        for j in range(2, k_w):
            tx_rwx = tvm.compute(
                shape,
                lambda *i: tvm.max(tx_ub[i[0], i[1], i[2], i[3]*s_w + j, i[4]],
                                   tx_rw[i[0], i[1], i[2], i[3], i[4]]),
                name="tx_rw" + str(j),
                tag="reduce_max"
            )
            tx_rw = tx_rwx
    elif k_w == 1:
        tx_rw1 = tvm.compute(
            shape,
            lambda *i: tvm.max(tx_ub[i[0], i[1], i[2], i[3] * s_w, i[4]],
                               tx_ub[i[0], i[1], i[2], i[3] * s_w, i[4]]),
            name="tx_rw1",
            tag="reduce_max"
        )
        tx_rw = tx_rw1

    # reduce h
    shape = (x_n, x_c1, o_h, o_w, x_c0)
    if k_h > 1:
        tx_rh1 = tvm.compute(
            shape,
            lambda *i: tvm.max(tx_rw[i[0], i[1], i[2]*s_h, i[3], i[4]],
                               tx_rw[i[0], i[1], i[2]*s_h + 1, i[3], i[4]]),
            name="tx_rh1",
            tag="reduce_max"
        )
        tx_rh = tx_rh1
        for j in range(2, k_h):
            tx_rhx = tvm.compute(
                shape,
                lambda *i: tvm.max(tx_rw[i[0], i[1], i[2]*s_h + j, i[3], i[4]],
                                   tx_rh[i[0], i[1], i[2], i[3], i[4]]),
                name="tx_rh" + str(j),
                tag="reduce_max"
            )
            tx_rh = tx_rhx
    elif k_h == 1:
        tx_rh1 = tvm.compute(
            shape,
            lambda *i: tvm.max(tx_rw[i[0], i[1], i[2] * s_h, i[3], i[4]],
                               tx_rw[i[0], i[1], i[2] * s_h, i[3], i[4]]),
            name="tx_rh1",
            tag="reduce_max"
        )
        tx_rh = tx_rh1

    # copy ub to gm
    shape = (x_n, x_c1, o_h, o_w, x_c0)
    t_y = tvm.compute(
        shape,
        lambda *i: tx_rh[i],
        name="pooling2d_res",
        tag=POOL2D_TAG + "max",
        attrs={"pooling_params": pooling_params,
               "template": "max_pool2d_generic",
               }
    )

    return t_y
