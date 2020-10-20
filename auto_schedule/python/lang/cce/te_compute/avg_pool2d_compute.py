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
pooling2d compute
"""
from te import tvm
POOL2D_TAG = "pooling2d_"


def avg_pool2d(t_x, pooling_params, fusion_params):
    """
    Generic vector template
    :param t_x: input tensor
    :param pooling_params: pooling parameters
    :param fusion_params:
    :return:
    """
    x_n, x_c1, x_h, _, x_c0 = t_x.shape
    k_h, k_w = pooling_params["window_h"], pooling_params["window_w"]
    s_h, s_w = pooling_params["stride_h"], pooling_params["stride_w"]
    dtype = t_x.dtype
    o_h, o_w = pooling_params["out_size_h"], pooling_params["out_size_w"]
    # h, w with pad
    h_p, w_p = (o_h - 1)*s_h + k_h, (o_w - 1)*s_w + k_w
    p_t, p_b = pooling_params["pad_top"], pooling_params["pad_bottom"]
    p_l, p_r = pooling_params["pad_left"], pooling_params["pad_right"]

    def _select(indices):
        i_n, i_c1, i_c0 = indices[0], indices[1], indices[4]
        i_h, i_w = indices[2], indices[3]
        conds = [i_h >= p_t, i_h < h_p - p_b, i_w >= p_l, i_w < w_p - p_r]
        t_b = tvm.select(conds[3], t_x[i_n, i_c1, i_h - p_t, i_w - p_l, i_c0])
        t_b = tvm.select(conds[2], t_b)
        t_b = tvm.select(conds[1], t_b)
        return tvm.select(conds[0], t_b)

    def _pad(cond):
        return tvm.select(cond, fp16_zero)

    def _fake(i):
        return tx_ub_c[i] + tx_ub_t[i] + tx_ub_b[i] + tx_ub_l[i] + tx_ub_r[i]

    # copy gm to ub with padding
    shape = (x_n, x_c1, h_p, w_p, x_c0)
    fp16_zero = tvm.const(0, dtype=dtype)
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
            lambda *i: tvm.sum(tx_ub[i[0], i[1], i[2], i[3], i[4]],
                               tx_ub[i[0], i[1], i[2], i[3] + 1, i[4]]),
            name="tx_rw1",
            tag="reduce_sum"
        )
        tx_rw = tx_rw1
        for j in range(2, k_w):
            tx_rwx = tvm.compute(
                shape,
                lambda *i: tvm.sum(tx_ub[i[0], i[1], i[2], i[3] + j, i[4]],
                                   tx_rw[i[0], i[1], i[2], i[3], i[4]]),
                name="tx_rw" + str(j),
                tag="reduce_sum"
            )
            tx_rw = tx_rwx
    elif k_w == 1:
        tx_rw = tx_ub

    # reduce h
    shape = (x_n, x_c1, o_h, o_w, x_c0)
    if k_h > 1:
        tx_rh1 = tvm.compute(
            shape,
            lambda *i: tvm.sum(tx_rw[i[0], i[1], i[2]*s_h, i[3], i[4]],
                               tx_rw[i[0], i[1], i[2]*s_h + 1, i[3], i[4]]),
            name="tx_rh1",
            tag="reduce_sum"
        )
        tx_rh = tx_rh1
        for j in range(2, k_h):
            tx_rhx = tvm.compute(
                shape,
                lambda *i: tvm.sum(tx_rw[i[0], i[1], i[2]*s_h + j, i[3], i[4]],
                                   tx_rh[i[0], i[1], i[2], i[3], i[4]]),
                name="tx_rh" + str(j),
                tag="reduce_sum"
            )
            tx_rh = tx_rhx
    elif k_h == 1:
        tx_rh1 = tvm.compute(
            shape,
            lambda *i: tvm.sum(tx_rw[i[0], i[1], i[2] * s_h, i[3], i[4]],
                               tx_rw[i[0], i[1], i[2] * s_h, i[3], i[4]]),
            name="tx_rh1",
            tag="reduce_sum"
        )
        tx_rh = tx_rh1

    # coeff area size
    area = tvm.const(1 / (k_h * k_w), dtype=dtype)
    shape = (x_n, x_c1, o_h, o_w, x_c0)
    h_start = (o_h - 1) * s_h - p_t
    h_value = [h_start + k_h, x_h.value + p_t]
    if h_value[0] < h_value[1]:
        h_end = h_value[0]
    else:
        h_end = h_value[1]
    last_h = h_end - h_start
    if last_h == k_h:
        t_avg = tvm.compute(
            shape,
            lambda *i: area * tx_rh(*i),
            name="tx_avg",
            tag="vector_muls")
    else:
        area1 = tvm.const(1 / (last_h * k_w), dtype=dtype)
        t_avg = tvm.compute(shape,
                            lambda n, c1, h, w, c0:
                            tvm.select(h < (o_h - 1), tx_rh[n, c1, h, w, c0] * area,
                                       tx_rh[n, c1, h, w, c0] * area1),
                            name="tx_avg",
                            tag="vector_muls")

    # copy ub to gm
    t_y = tvm.compute(
        shape,
        lambda *i: t_avg[i],
        name="pooling2d_res",
        tag=POOL2D_TAG + "avg",
        attrs={"pooling_params": pooling_params,
               "template": "avg_pool2d_generic",
               "fusion_params": fusion_params,
               }
    )

    return t_y
