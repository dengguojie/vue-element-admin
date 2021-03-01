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
max_pool2d
"""
from tbe import tvm
from tbe.common.utils.errormgr import get_error_message
from tbe.common.testing.testing import is_debug_mode

_POOL2D_TAG = "pooling2d_"
_MAX_KERNEL_SIZE_H_MUL_W = 255  # kernel_h * kernel_w
_MAX_KERNEL_SIZE = 20


def max_pool2d(t_x, pooling_params, fusion_params):
    """
    Generic vector template
    :param t_x: input tensor
    :param pooling_params: pooling parameters
    :param fusion_params:
    :return:
    """
    x_n, x_c1, _, _, x_c0 = t_x.shape
    pooling_mode = pooling_params["pooling_mode"]
    if pooling_mode in ["GMP"]:
        k_h, k_w = pooling_params["in_size_h"], pooling_params["in_size_w"]
    else:
        k_h, k_w = pooling_params["window_h"], pooling_params["window_w"]
    # if window_h or window_w
    is_support_kernel = (k_h * k_w <= _MAX_KERNEL_SIZE_H_MUL_W) or \
                        (k_h <= _MAX_KERNEL_SIZE and k_w <= _MAX_KERNEL_SIZE)
    if not is_support_kernel:
        dict_args = dict()
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "kw [%s] and kh [%s] is too big! " \
                                      "maxpool schedule only support " \
                                      "(k_h * k_w <= _MAX_KERNEL_SIZE_H_MUL_W) " \
                                      "or (k_h <= _MAX_KERNEL_SIZE and k_w <= _MAX_KERNEL_SIZE)" % (k_w, k_h)
        raise RuntimeError(dict_args, get_error_message(dict_args))
    s_h, s_w = pooling_params["stride_h"], pooling_params["stride_w"]
    dtype = t_x.dtype
    o_h, o_w = pooling_params["out_size_h"], pooling_params["out_size_w"]
    p_t, p_b = pooling_params["pad_top"], pooling_params["pad_bottom"]
    p_l, p_r = pooling_params["pad_left"], pooling_params["pad_right"]
    # h, w with pad
    if pooling_params["ceil_mode"] == 0:
        h_p, w_p = (o_h - 1)*s_h + k_h, (o_w - 1)*s_w + k_w
    elif pooling_params["ceil_mode"] == 1:
        h_p, w_p = pooling_params["in_size_h"] + p_t + p_b, \
            pooling_params["in_size_w"] + p_l + p_r

    def _select(indices, false_value):
        i_n, i_c1, i_c0 = indices[0], indices[1], indices[4]
        i_h, i_w = indices[2], indices[3]
        conds = [i_h >= p_t, i_h < h_p - p_b, i_w >= p_l, i_w < w_p - p_r]
        t_b = tvm.select(conds[3], t_x[i_n, i_c1, i_h - p_t, i_w - p_l, i_c0], false_value)
        t_b = tvm.select(conds[2], t_b, false_value)
        t_b = tvm.select(conds[1], t_b, false_value)
        return tvm.select(conds[0], t_b, false_value)

    def _pad(cond):
        return tvm.select(cond, fp16_min)

    def _fake(i):
        return tx_ub_c[i] + tx_ub_t[i] + tx_ub_b[i] + tx_ub_l[i] + tx_ub_r[i]

    shape = (x_n, x_c1, h_p, w_p, x_c0)
    fp16_min = tvm.const(-65504.0, dtype=dtype)
    if is_debug_mode():
        tx_ub = tvm.compute(shape, lambda *i: _select(i, fp16_min), name="tx_ub")
    else:
        # copy gm to ub with padding
        tx_ub_c = tvm.compute(shape, lambda *i: _select(i, None), name="tx_ub_c")
        tx_ub_t = tvm.compute(shape, lambda *i: _pad(i[2] < p_t), name="tx_ub_t")
        tx_ub_b = tvm.compute(shape, lambda *i: _pad(i[2] >= h_p - p_b), name="tx_ub_b")
        tx_ub_l = tvm.compute(shape, lambda *i: _pad(i[3] < p_l), name="tx_ub_l")
        tx_ub_r = tvm.compute(shape, lambda *i: _pad(i[3] >= w_p - p_r), name="tx_ub_r")
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
        tx_rw0 = tvm.compute(
            shape,
            lambda *i: tvm.max(tx_ub[i[0], i[1], i[2], i[3] * s_w, i[4]],
                               tx_ub[i[0], i[1], i[2], i[3] * s_w, i[4]]),
            name="tx_rw0",
            tag="reduce_max"
        )
        tx_rw = tx_rw0

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
        tx_rh0 = tvm.compute(
            shape,
            lambda *i: tvm.max(tx_rw[i[0], i[1], i[2] * s_h, i[3], i[4]],
                               tx_rw[i[0], i[1], i[2] * s_h, i[3], i[4]]),
            name="tx_rh0",
            tag="reduce_max"
        )
        tx_rh = tx_rh0

    # copy ub to gm
    shape = (x_n, x_c1, o_h, o_w, x_c0)
    t_y = tvm.compute(
        shape,
        lambda *i: tx_rh[i],
        name="pooling2d_res",
        tag=_POOL2D_TAG + "max",
        attrs={"pooling_params": pooling_params,
               "template": "max_pool2d_generic",
               "fusion_params": fusion_params,
               }
    )

    return t_y
