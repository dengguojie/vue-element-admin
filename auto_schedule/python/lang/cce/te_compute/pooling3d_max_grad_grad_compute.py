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
pooling3d_max_grad_grad compute
"""
import math
import warnings

from te import tvm

_POOL3D_TAG = "pooling3d_"
_SIZE_OF_FP16 = 2
_BLOCK_SIZE = 16
_DATA_MODE_CEIL = 0
_DATA_MODE_PADDING = 1
_C0_DIMENSION_DATA_SIZE = 32
_MAX_KERNEL_SIZE = 10 * 10 * 10
_MIN_VAL_MAP = {"float16": -65504.0, "float32": 3.4e-38, "double": 1.7e-308}
_SIZEOF__DTYPE_MAP = {"float16": 2, "float32": 4, "double": 8}
_DTYPE_MAP = {"float16": "uint16", "float32": "uint32", "double": "uint64"}


# 'pylint: disable=too-many-arguments
def pooling3d_max_grad_grad(orig_input, orig_output, grad_grad, assist_tensor,
                            ksize, strides, pads=(0, 0, 0, 0, 0, 0),
                            data_format="NDHWC",
                            padding="SAME"):
    warnings.warn("pooling3d_max_grad_grad is expired, please replace it with the func max_pooling3d_grad_grad",
                  DeprecationWarning)
    return max_pooling3d_grad_grad(orig_input, orig_output, grad_grad, assist_tensor,
                                   ksize, strides, pads, data_format, padding)


# 'pylint: disable=too-many-locals, too-many-arguments, invalid-name,
# 'pylint: disable=unused-argument,too-many-statements, cell-var-from-loop
def max_pooling3d_grad_grad(orig_input, orig_output, grad_grad, assist_tensor,
                            ksize, strides, pads=(0, 0, 0, 0, 0, 0),
                            data_format="NDHWC",
                            padding="SAME"):
    """
    orig_input : dict, shape and dtype of input_data,
                 shape is 6 dims, format is NDC1HWC0
    orig_output : dict, result of max_pool3d(orig_input, ksize, ...)
    grad_grad: dict, input grad of grad
    assist_tensor: dict, helper matrix, it's content is 8,7,6,5,4,3,2,1
                if kernel is 2 x 2 x 2
    ksize : list or tuple, the window of max_pool3d,
            only support max_pool3d in D or H or W
    strides : list or tuple, the stride of max_pool3d window,
              only support max_pool3d in D or H or W
    pads : reserved.
    padding : str, the mode of padding, support SAME or VALID
    ceil_mode: reserved
    """

    def _pad_min(cond):
        min_val = tvm.const(_MIN_VAL_MAP[orig_input.dtype],
                            dtype=orig_input.dtype)
        return tvm.select(cond, min_val)

    def _pad_zero(cond):
        fixed_val = tvm.const(0, dtype=orig_input.dtype)
        return tvm.select(cond, fixed_val)

    def _copy_orig_input():
        def __select_orig_in(indices):
            i_n, i_c1, i_d = indices[0], indices[1], indices[2]
            i_h, i_w, i_c0 = indices[3], indices[4], indices[5]
            conds = [i_d >= p_ft, i_d < d_p - p_bk,
                     i_h >= p_t, i_h < h_p - p_b,
                     i_w >= p_l, i_w < w_p - p_r]
            t_b = tvm.select(conds[5], orig_input[i_n, i_d - p_ft, i_c1, i_h - p_t, i_w - p_l, i_c0])
            t_b = tvm.select(conds[4], t_b)
            t_b = tvm.select(conds[3], t_b)
            t_b = tvm.select(conds[2], t_b)
            t_b = tvm.select(conds[1], t_b)
            return tvm.select(conds[0], t_b)

        def __fake_orig_in(i):
            return tx_orig_in_c[i] + tx_orig_in_ft[i] + tx_orig_in_bk[i] + \
                tx_orig_in_t[i] + tx_orig_in_b[i] + tx_orig_in_l[i] + tx_orig_in_r[i]

        tx_orig_in_c = tvm.compute(shape_trans, lambda *i: __select_orig_in(i),
                                   name="tx_orig_in_c", tag="tx_orig_in_c")
        tx_orig_in_ft = tvm.compute(shape_trans, lambda *i: _pad_min(i[2] < p_ft),
                                    name="tx_orig_in_ft", tag="tx_orig_in_ft")
        tx_orig_in_bk = tvm.compute(shape_trans, lambda *i: _pad_min(i[2] >= d_p - p_bk),
                                    name="tx_orig_in_bk", tag="tx_orig_in_bk")
        tx_orig_in_t = tvm.compute(shape_trans, lambda *i: _pad_min(i[3] < p_t),
                                   name="tx_orig_in_t", tag="tx_orig_in_t")
        tx_orig_in_b = tvm.compute(shape_trans, lambda *i: _pad_min(i[3] >= h_p - p_b),
                                   name="tx_orig_in_b", tag="tx_orig_in_b")
        tx_orig_in_l = tvm.compute(shape_trans, lambda *i: _pad_min(i[4] < p_l),
                                   name="tx_orig_in_l", tag="tx_orig_in_l")
        tx_orig_in_r = tvm.compute(shape_trans, lambda *i: _pad_min(i[4] >= w_p - p_r),
                                   name="tx_orig_in_r", tag="tx_orig_in_r")
        tx_orig_in = tvm.compute(shape_trans, lambda *i: __fake_orig_in(i),
                                 name="tx_orig_in", tag="tx_orig_in")
        return tx_orig_in

    def _copy_grad_grad():
        def __select_grad_grad(indices):
            i_n, i_c1, i_d = indices[0], indices[1], indices[2]
            i_h, i_w, i_c0 = indices[3], indices[4], indices[5]
            conds = [i_d >= p_ft, i_d < d_p - p_bk,
                     i_h >= p_t, i_h < h_p - p_b,
                     i_w >= p_l, i_w < w_p - p_r]
            t_b = tvm.select(conds[5], grad_grad[i_n, i_d - p_ft, i_c1, i_h - p_t, i_w - p_l, i_c0])
            t_b = tvm.select(conds[4], t_b)
            t_b = tvm.select(conds[3], t_b)
            t_b = tvm.select(conds[2], t_b)
            t_b = tvm.select(conds[1], t_b)
            return tvm.select(conds[0], t_b)

        def __fake_grad_grad(i):
            return tx_grad_grad_c[i]

        tx_grad_grad_c = tvm.compute(shape_trans, lambda *i: __select_grad_grad(i),
                                     name="tx_grad_grad_c", tag="tx_grad_grad_c")
        tx_grad_grad = tvm.compute(shape_trans, lambda *i: __fake_grad_grad(i),
                                   name="tx_grad_grad", tag="tx_grad_grad")
        return tx_grad_grad

    def _copy_orig_output():
        tx_orig_out = tvm.compute(shape_orig_out, lambda *i: orig_output(i[0], i[2], i[1], i[3], i[4], i[5]),
                                  name="tx_orig_out", tag="tx_orig_out")
        return tx_orig_out

    def _copy_decrease_kernel():
        tx_decrease_kernel = tvm.compute(shape_assist, lambda *i: assist_tensor(i[0], i[2], i[1], i[3], i[4], i[5]),
                                         name="tx_decrease_kernel", tag="tx_decrease_kernel")
        return tx_decrease_kernel

    def _extend_orig_in():
        tx_orig_in_ext = tvm.compute(shape_trans_ext, lambda *i:
                                     tx_orig_in(i[0], i[1],
                                                i[2] // k_d * s_d + i[2] % k_d,
                                                i[3] // k_h * s_h + i[3] % k_h,
                                                i[4] // k_w * s_w + i[4] % k_w,
                                                i[5]),
                                     name="tx_orig_in_ext", tag="tx_orig_in_ext")
        return tx_orig_in_ext

    def _extend_grad_grad():
        tx_grad_grad_ext = tvm.compute(shape_trans_ext, lambda *i:
                                       tx_grad_grad(i[0], i[1],
                                                    i[2] // k_d * s_d + i[2] % k_d,
                                                    i[3] // k_h * s_h + i[3] % k_h,
                                                    i[4] // k_w * s_w + i[4] % k_w,
                                                    i[5]),
                                       name="tx_grad_grad_ext", tag="tx_grad_grad_ext")
        return tx_grad_grad_ext

    def _extend_orig_output():
        def __select_orig_output(i):
            return tx_orig_out(i[0], i[1], i[2]//k_d, i[3]//k_h, i[4]//k_w, i[5])
        tx_orig_out_ext = tvm.compute(shape_trans_ext, lambda *i: __select_orig_output(i),
                                      name="tx_orig_out_ext", tag="tx_orig_out_ext")
        return tx_orig_out_ext

    def _extend_decrease_kernel():
        def __select_decrease_kernel(i):
            return tx_decrease_kernel(0, 0, i[2] % k_d, i[3] % k_h, i[4] % k_w, i[5])
        tx_decrease_kernel_ext = tvm.compute(shape_trans_ext, lambda *i: __select_decrease_kernel(i),
                                             name="tx_decrease_kernel_ext", tag="tx_decrease_kernel_ext")
        return tx_decrease_kernel_ext

    def _build_tx_all_zero():
        fixed_vale = tvm.const(0.0, dtype="float16")
        return tvm.compute(shape_trans_ext, lambda *i: fixed_vale, name="tx_all_zero", tag="tx_all_zero")

    def _compare_mask():
        tx_mask = tvm.compute(shape_trans_ext, lambda *i: tx_orig_out_ext(*i) == tx_orig_in_ext(*i),
                              name="tx_mask", tag="tx_mask")
        return tx_mask

    def _construct_sparse_matrix():
        tx_decrease_sparse_matrix = tvm.compute(shape_trans_ext,
                                                lambda *i: tvm.select(tx_mask(*i),
                                                                      tx_decrease_kernel_ext(*i), tx_all_zero(*i)),
                                                name="tx_decrease_sparse_matrix", tag="tx_decrease_sparse_matrix")
        return tx_decrease_sparse_matrix

    def _reduce_decrease_sparse_matrix():
        def __reduce_w():
            shape = (n, c1, d_p_ext, h_p_ext, o_w, c0)

            if k_w == 1:
                tx_rw = tvm.compute(shape,
                                    lambda *i: tx_decrease_sparse_matrix[i[0], i[1], i[2], i[3], i[4] * k_w, i[5]],
                                    name="tx_rw1", tag="reduce_max")
                return tx_rw

            tx_rw1 = tvm.compute(
                shape,
                lambda *i: tvm.max(tx_decrease_sparse_matrix(i[0], i[1], i[2], i[3], i[4] * k_w, i[5]),
                                   tx_decrease_sparse_matrix(i[0], i[1], i[2], i[3], i[4] * k_w + 1, i[5])),
                name="tx_rw1", tag="reduce_max")
            tx_rw = tx_rw1
            for j in range(2, k_w):
                tx_rwx = tvm.compute(
                    shape,
                    lambda *i: tvm.max(tx_decrease_sparse_matrix[i[0], i[1], i[2], i[3], i[4] * k_w + j, i[5]],
                                       tx_rw[i[0], i[1], i[2], i[3], i[4], i[5]]),
                    name="tx_rw" + str(j), tag="reduce_max")
                tx_rw = tx_rwx
            return tx_rw

        def __reduce_h():
            shape = (n, c1, d_p_ext, o_h, o_w, c0)

            if k_h == 1:
                tx_rh = tvm.compute(shape,
                                    lambda *i: tx_rw[i[0], i[1], i[2], i[3] * k_h, i[4], i[5]],
                                    name="tx_rh1", tag="reduce_max")
                return tx_rh

            tx_rh1 = tvm.compute(
                shape,
                lambda *i: tvm.max(tx_rw[i[0], i[1], i[2], i[3] * k_h, i[4], i[5]],
                                   tx_rw[i[0], i[1], i[2], i[3] * k_h + 1, i[4], i[5]]),
                name="tx_rh1",
                tag="reduce_max"
            )
            tx_rh = tx_rh1
            for j in range(2, k_h):
                tx_rhx = tvm.compute(
                    shape,
                    lambda *i: tvm.max(tx_rw[i[0], i[1], i[2], i[3] * k_h + j, i[4], i[5]],
                                       tx_rh[i[0], i[1], i[2], i[3], i[4], i[5]]),
                    name="tx_rh" + str(j),
                    tag="reduce_max"
                )
                tx_rh = tx_rhx
            return tx_rh

        def __reduce_d():
            shape = (n, c1, o_d, o_h, o_w, c0)

            if k_d == 1:
                tx_rd = tvm.compute(shape,
                                    lambda *i: tx_rh[i[0], i[1], i[2] * k_d, i[3], i[4], i[5]],
                                    name="tx_rd1", tag="reduce_max")
                return tx_rd

            tx_rd1 = tvm.compute(
                shape,
                lambda *i: tvm.max(tx_rh[i[0], i[1], i[2] * k_d, i[3], i[4], i[5]],
                                   tx_rh[i[0], i[1], i[2] * k_d + 1, i[3], i[4], i[5]]),
                name="tx_rd1",
                tag="reduce_max"
            )
            tx_rd = tx_rd1
            for j in range(2, k_d):
                tx_rdx = tvm.compute(
                    shape,
                    lambda *i: tvm.max(tx_rh[i[0], i[1], i[2] * k_d + j, i[3], i[4], i[5]],
                                       tx_rd[i[0], i[1], i[2], i[3], i[4], i[5]]),
                    name="tx_rd" + str(j), tag="reduce_max")
                tx_rd = tx_rdx
            return tx_rd

        tx_rw = __reduce_w()
        tx_rh = __reduce_h()
        tx_rd = __reduce_d()
        return tx_rd

    def _broadcast_max():
        def __select_broadcast_max(i):
            return tx_max_of_sparse_matrix(i[0], i[1], i[2]//k_d, i[3]//k_h, i[4]//k_w, i[5])
        tx_max_broadcasted = tvm.compute(shape_trans_ext,
                                         lambda *i: __select_broadcast_max(i),
                                         name="tx_max_broadcasted", tag="tx_max_broadcasted")
        return tx_max_broadcasted

    def _construct_mask_no_dup():
        tx_mask_no_dup = tvm.compute(shape_trans_ext,
                                     lambda *i: (tx_max_broadcasted(*i) == tx_decrease_sparse_matrix(*i)),
                                     name="tx_mask_no_dup", tag="tx_mask_no_dup")
        return tx_mask_no_dup

    def _fill_grad_by_mask():
        tx_grad_by_mask = tvm.compute(shape_trans_ext,
                                      lambda *i: tvm.select(tx_mask_no_dup(*i), tx_grad_grad_ext(*i), tx_all_zero(*i)),
                                      name="tx_grad_by_mask", tag="tx_grad_by_mask")
        return tx_grad_by_mask

    def _reduce_grad():
        def __reduce_grad_w():
            shape = (n, c1, d_p_ext, h_p_ext, o_w, c0)

            if k_w == 1:
                tx_grad_rw = tvm.compute(shape, lambda *i: tx_grad_by_mask[i[0], i[1], i[2], i[3], i[4] * k_w, i[5]],
                                         name="tx_grad_rw1", tag="reduce_grad_max")
                return tx_grad_rw

            tx_grad_rw1 = tvm.compute(
                shape,
                lambda *i: tvm.sum(tx_grad_by_mask[i[0], i[1], i[2], i[3], i[4] * k_w, i[5]],
                                   tx_grad_by_mask[i[0], i[1], i[2], i[3], i[4] * k_w + 1, i[5]]),
                name="tx_grad_rw1", tag="reduce_grad_max")
            tx_grad_rw = tx_grad_rw1
            for j in range(2, k_w):
                tx_grad_rwx = tvm.compute(
                    shape, lambda *i: tvm.sum(tx_grad_by_mask[i[0], i[1], i[2], i[3], i[4] * k_w + j, i[5]],
                                              tx_grad_rw[i[0], i[1], i[2], i[3], i[4], i[5]]),
                    name="tx_grad_rw" + str(j), tag="reduce_grad_max")
                tx_grad_rw = tx_grad_rwx
            return tx_grad_rw

        def __reduce_grad_h():
            shape = (n, c1, d_p_ext, o_h, o_w, c0)

            if k_h == 1:
                tx_grad_rh = tvm.compute(shape, lambda *i: tx_grad_rw[i[0], i[1], i[2], i[3] * k_h, i[4], i[5]],
                                         name="tx_grad_rh1", tag="reduce_grad_max")
                return tx_grad_rh

            tx_grad_rh1 = tvm.compute(shape,
                                      lambda *i: tvm.sum(tx_grad_rw[i[0], i[1], i[2], i[3] * k_h, i[4], i[5]],
                                                         tx_grad_rw[i[0], i[1], i[2], i[3] * k_h + 1, i[4], i[5]]),
                                      name="tx_grad_rh1", tag="reduce_grad_max")
            tx_grad_rh = tx_grad_rh1
            for j in range(2, k_h):
                tx_grad_rhx = tvm.compute(
                    shape,
                    lambda *i: tvm.sum(tx_grad_rw[i[0], i[1], i[2], i[3] * k_h + j, i[4], i[5]],
                                       tx_grad_rh[i[0], i[1], i[2], i[3], i[4], i[5]]),
                    name="tx_grad_rh" + str(j), tag="reduce_grad_max")
                tx_grad_rh = tx_grad_rhx
            return tx_grad_rh

        def __reduce_grad_d():
            shape = (n, c1, o_d, o_h, o_w, c0)

            if k_d == 1:
                tx_grad_rd = tvm.compute(shape,
                                         lambda *i: tx_grad_rh[i[0], i[1], i[2] * k_d, i[3], i[4], i[5]],
                                         name="tx_grad_rd1", tag="reduce_grad_max")
                return tx_grad_rd

            tx_grad_rd1 = tvm.compute(shape,
                                      lambda *i: tvm.sum(tx_grad_rh[i[0], i[1], i[2] * k_d, i[3], i[4], i[5]],
                                                         tx_grad_rh[i[0], i[1], i[2] * k_d + 1, i[3], i[4], i[5]]),
                                      name="tx_grad_rd1", tag="reduce_grad_max")
            tx_grad_rd = tx_grad_rd1
            for j in range(2, k_d):
                tx_grad_rdx = tvm.compute(shape,
                                          lambda *i: tvm.sum(tx_grad_rh[i[0], i[1], i[2] * k_d + j, i[3], i[4], i[5]],
                                                             tx_grad_rd[i[0], i[1], i[2], i[3], i[4], i[5]]),
                                          name="tx_grad_rd" + str(j), tag="reduce_grad_max")
                tx_grad_rd = tx_grad_rdx
            return tx_grad_rd

        tx_grad_rw = __reduce_grad_w()
        tx_grad_rh = __reduce_grad_h()
        tx_grad_rd = __reduce_grad_d()
        return tx_grad_rd

    def _is_fast_path():
        return k_d == 1 and k_h == 1 and k_w == 1

    def _fast_path_res():
        tx_res = tvm.compute(shape_orig_out, lambda *i: tx_grad_grad(i[0], i[1], i[2] * s_d, i[3] * s_h, i[4] * s_w, i[5]),
                             name="fast_path_res", tag="fast_path_res")
        return tx_res

    def _build_pooling_params():
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
        pooling_params["size_of_data"] = _SIZEOF__DTYPE_MAP[orig_input.dtype]
        pooling_params["align_axis"] = "axis_w"
        if _is_fast_path():
            pooling_params["fast_path"] = "True"
        else:
            pooling_params["fast_path"] = "False"
        return pooling_params

    def _get_dim_param():
        n = orig_input.shape[0].value
        d = orig_input.shape[1].value
        c1 = orig_input.shape[2].value
        h = orig_input.shape[3].value
        w = orig_input.shape[4].value
        c0 = orig_input.shape[5].value
        return n, d, c1, h, w, c0

    def _get_kernel_param():
        return ksize[0], ksize[1], ksize[2]

    def _get_stride_param():
        return strides[0], strides[1], strides[2]

    def _copy_res_out():
        res = tvm.compute(shape_out, lambda *i: tx_res[i[0], i[2], i[1], i[3], i[4], i[5]],
                          name=_POOL3D_TAG + "max_grad_grad_res", tag=_POOL3D_TAG + "max_grad_grad",
                          attrs={"pooling_params": pooling_params, "template": "max_pool3d_grad_grad"})
        return res

    def _detect_align_axis(pooling_params):
        pooling_params["align_axis"] = "axis_w"
        # if all kernel is 1, no need to align, so just form low address to high address.
        if k_d == 1 and k_h == 1 and k_w == 1:
            return
        if d_p > h_p and d_p > w_p:
            pooling_params["align_axis"] = "axis_d"
        if h_p > w_p and h_p > d_p:
            pooling_params["align_axis"] = "axis_h"

    def _calc_padding_ext(pooling_params):
        if pooling_params["align_axis"] == "axis_w":
            return 0, 0, 2000
        if pooling_params["align_axis"] == "axis_h":
            return 0, 2000, 0
        if pooling_params["align_axis"] == "axis_d":
            return 2000, 0, 0
        return 0, 0, 0

    if _check_max_grad_grad_params():
        raise RuntimeError("Failed to check max grad grad params.")

    n, d, c1, h, w, c0 = _get_dim_param()
    k_d, k_h, k_w = _get_kernel_param()
    s_d, s_h, s_w = _get_stride_param()
    o_d, o_h, o_w, p_ft, p_bk, p_t, p_b, p_l, p_r = 
        _get_out_and_pad_with_padding_mode(padding, d, h, w, ksize, strides)
    d_p = o_d * k_d
    h_p = o_h * k_h
    d_p = d + p_ft + p_bk
    h_p = h + p_t + p_b
    w_p = w + p_l + p_r

    # define dict to transfer pooling params
    pooling_params = _build_pooling_params()
    _detect_align_axis(pooling_params)

    p_bk_ext, p_b_ext, p_r_ext = _calc_padding_ext(pooling_params)
    d_p_ext = o_d * k_d + p_bk_ext
    h_p_ext = o_h * k_h + p_b_ext
    w_p_ext = o_w * k_w + p_r_ext

    shape_orig_out = (n, c1, o_d, o_h, o_w, c0)
    shape_assist = (1, 1, k_d, k_h, k_w, c0)
    shape_trans = (n, c1, d_p, h_p, w_p, c0)
    shape_trans_ext = (n, c1, d_p_ext, h_p_ext, w_p_ext, c0)
    shape_out = (n, o_d, c1, o_h, o_w, c0)

    if _is_fast_path():
        tx_grad_grad = _copy_grad_grad()
        tx_res = _fast_path_res()
    else:
        tx_orig_in = _copy_orig_input()
        tx_orig_out = _copy_orig_output()
        tx_grad_grad = _copy_grad_grad()
        tx_decrease_kernel = _copy_decrease_kernel()
        tx_orig_in_ext = _extend_orig_in()
        tx_orig_out_ext = _extend_orig_output()
        tx_grad_grad_ext = _extend_grad_grad()
        tx_decrease_kernel_ext = _extend_decrease_kernel()
        tx_mask = _compare_mask()
        tx_all_zero = _build_tx_all_zero()
        tx_decrease_sparse_matrix = _construct_sparse_matrix()
        tx_max_of_sparse_matrix = _reduce_decrease_sparse_matrix()
        tx_max_broadcasted = _broadcast_max()
        tx_mask_no_dup = _construct_mask_no_dup()
        tx_grad_by_mask = _fill_grad_by_mask()
        tx_res = _reduce_grad()
    return _copy_res_out()


def _check_max_grad_grad_params():
    return False


def _get_out_and_pad_with_padding_mode(padding_mode, in_size_d, in_size_h, in_size_w,
                                       window, stride):
    """
    :param padding_mode: can be SAME, VALID
    :param in_size_d: input tensor
    :param in_size_h: input tensor
    :param in_size_w: input tensor
    :param window: input window d/h/w
    :param stride: stride d/h/w
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
        pad_d = (out_size_d - 1) * stride[0] + ((window[0] - 1) * dilation[0] + 1) - in_size_d
        pad_h = (out_size_h - 1) * stride[1] + ((window[1] - 1) * dilation[1] + 1) - in_size_h
        pad_w = (out_size_w - 1) * stride[2] + ((window[2] - 1) * dilation[2] + 1) - in_size_w

        pad_d = pad_d if pad_d > 0 else 0
        pad_h = pad_h if pad_h > 0 else 0
        pad_w = pad_w if pad_w > 0 else 0

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
