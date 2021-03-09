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
max_pool_v200
"""
from __future__ import division
import math
from tbe import tvm
from te.platform import cce_conf


class MaxPoolParam:
    """
    class of ConvParam
    """

    def __init__(self):
        pass

    tensor_map = {"is_conv_pool_fused": False}

    @staticmethod
    def update_tensormap(map_key, map_value):
        """
        update the tensor map
        """
        MaxPoolParam.tensor_map[map_key] = map_value

    @staticmethod
    def get_tensormap():
        """
        get then tensor map
        """
        return MaxPoolParam.tensor_map


def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        tmp.append(i.value)
    return tmp


def _is_support_v200():
    """
    check if Ascend610/Ascend710/Hi3796CV300CS version
    ----------

    Returns
    -------
    True:  Ascend610/Ascend710/Hi3796CV300CS version
    False: Other version
    """
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    if soc_version in ("Ascend710", "Ascend610", "Hi3796CV300CS", "SD3403"):
        return True
    return False


NAME = "pooling2d_max_"
OP_TAG = "pooling2d_max_"
C0_SIZE = 16


@tvm.target.generic_func
def max_pool_compute(input_data,  # pylint: disable=too-many-arguments
                     ksize, strides, pad_mode="VALID",
                     padding=(0, 0, 0, 0), ceil_mode=0, data_mode=0):
    """
    Performs max pooling on the input.

    Parameters
    ----------
    input_data: tensor
        tensor of input_data.
    dtype: str
        input and output data type.
    ksize: list or tuple
        A list of `ints` that has length 2
        The size of the window for H, W dimension of the input tensor
    strides: list or tuple
        A list of `ints` that has length 4.
        The stride of the sliding window of the input tensor
    pad_mode: str
        A `string` from: "SAME", "VALID"`.The type of padding algorithm to use
    padding: list or tuple
        A `padding to use
    data_mode: int
        A int can be 0 : CAFFE_DATA_MODE, 1: TENSORFLOW_DATA_MODE
    cei_mode : int
        A int caffe round_mode params, 0:CEIL(default), 1:FLOOR

    Returns:
    -------
    res:
        The result of max pooling
    """

    def _compute_max_pooling(input_data, padding, window):
        # compute max_pooling
        input_shape = _shape_to_list(input_data.shape)

        data_pad, row_max_shape, max_pool_res_shape, input_5d_data = \
            _max_pooling_input(input_shape, width_in, padding)
        out_col_max, ret_out_list = \
            _max_pooling(data_pad, row_max_shape, max_pool_res_shape)
        if window[0] == 3:
            trans_line_data = tvm.compute(input_5d_data.shape,
                                          lambda *i: input_5d_data[i],
                                          name=NAME + 'trans_line_data',
                                          tag=OP_TAG + "trans_line_data")
            MaxPoolParam.update_tensormap("trans_line_data", trans_line_data)
        MaxPoolParam.update_tensormap("max_pool_tensors", ret_out_list)

        MaxPoolParam.update_tensormap("pooling_out",
                                      (max_pool_res_shape[2], max_pool_res_shape[3]))
        input_shape[1] = input_shape[3] // C0_SIZE
        input_shape[3] = C0_SIZE

        ub_reshape_shape = (input_shape[0],
                            input_shape[1],
                            max_pool_res_shape[2],
                            max_pool_res_shape[3],
                            input_shape[3])

        ub_reshape = tvm.compute(ub_reshape_shape,
                                 lambda n, c1, h, w, c0:
                                 out_col_max[n, 0,
                                             h,
                                             w,
                                             c1 * C0_SIZE + c0]
                                 + tvm.const(0.0, dtype=out_col_max.dtype),
                                 name=NAME + 'ub_reshape',
                                 tag=OP_TAG + "ub_reshape")
        MaxPoolParam.update_tensormap("ub_reshape", ub_reshape)
        if window[0] == 3:
            trans_vn_node = tvm.compute(ub_reshape.shape,
                                        lambda n, c1, h, w, c0:
                                        ub_reshape[n, c1, h, w, c0] +
                                        trans_line_data[n, 0, 0, 0, c1 * 16 + c0],
                                        name=NAME + 'trans_vn_node',
                                        tag=OP_TAG + "trans_vn_node")
            MaxPoolParam.update_tensormap("trans_vn_node", trans_vn_node)
        res_shape = (input_shape[0],
                     input_shape[1],
                     max_pool_res_shape[2] * max_pool_res_shape[3],
                     input_shape[3])
        if window[0] == 3:
            res = tvm.compute(res_shape,
                              lambda n, c1, m, c0:
                              trans_vn_node[n, c1,
                                            m // max_pool_res_shape[3],
                                            m % max_pool_res_shape[3],
                                            c0],
                              name=NAME + 'max_pool_res',
                              tag=OP_TAG + "max_pool_res")
        else:
            res = tvm.compute(res_shape,
                              lambda n, c1, m, c0:
                              ub_reshape[n, c1,
                                            m // max_pool_res_shape[3],
                                            m % max_pool_res_shape[3],
                                            c0],
                              name=NAME + 'max_pool_res',
                              tag=OP_TAG + "max_pool_res")
        MaxPoolParam.update_tensormap("max_pool_res", res)

        return res

    def _max_pooling_input(input_shape, width_in, padding):
        # get the 5HD input
        input_shape[3] = input_shape[3] * input_shape[1]
        input_shape[1] = 1
        input_5d_shape = (input_shape[0],
                          1,
                          input_shape[2] // width_in, width_in,
                          input_shape[3])
        input_5d_data = tvm.compute(input_5d_shape,
                                    lambda n, c1, h, w, c0:
                                    input_data[n,
                                               c0 // C0_SIZE,
                                               h * width_in + w,
                                               c0 % C0_SIZE] +
                                    tvm.const(0.0, dtype=input_data.dtype),
                                    name=NAME + 'input_5d_data',
                                    tag=OP_TAG + "input_5d_data")
        MaxPoolParam.update_tensormap("input_5d_data", input_5d_data)

        if data_mode == 0:
            h_out, w_out, padding = \
                _get_caffe_out_size_and_pad(ceil_mode,
                                           input_5d_shape,
                                           ksize,
                                           strides,
                                           padding)
        else:
            h_out, w_out, padding = \
                _get_tensorflow_out_size_and_pad(pad_mode,
                                                input_5d_shape,
                                                ksize,
                                                strides,
                                                padding)
        MaxPoolParam.update_tensormap("pooling_padding", padding)
        if padding == [0, 0, 0, 0]:
            data_pad = input_5d_data
            pad_shape = input_5d_shape
        else:
            data_pad, pad_shape = \
                _max_pooling_padding(input_5d_data, padding, dtype)

        row_max_shape = [input_shape[0],
                         input_shape[1],
                         h_out,
                         pad_shape[3],
                         input_shape[3]]
        max_pool_res_shape = [input_shape[0],
                              input_shape[1],
                              h_out,
                              w_out,
                              input_shape[3]]

        return data_pad, row_max_shape, max_pool_res_shape, input_5d_data

    def _max_pooling(data_pad, row_max_shape, max_pool_res_shape):
        # compute row max
        row_max_res, ret_row_max_pool_tensors = \
            _compute_row_optimization(data_pad, row_max_shape, strides, ksize)

        # compute col max and get final res
        col_max_res, ret_col_max_pool_tensors = \
            _compute_col_optimization(row_max_res,
                                     max_pool_res_shape,
                                     strides,
                                     ksize)
        # get list of vector max tensors
        ret_out_list = ret_row_max_pool_tensors + ret_col_max_pool_tensors

        return col_max_res, ret_out_list

    _check_para(ksize, strides)

    dtype = input_data.dtype
    # hard code to get conv_res attrs!!!

    conv_res = input_data.op.input_tensors[0]
    width_in = conv_res.op.attrs["width_out"].value
    MaxPoolParam.update_tensormap("conv_width", width_in)
    res = _compute_max_pooling(input_data, padding, ksize)
    MaxPoolParam.update_tensormap("is_conv_pool_fused", False)
    MaxPoolParam.update_tensormap("window_size", ksize[0])
    return res


def _check_para(window, strides):
    """
    check the window and stride
    """
    window_h, window_w = window
    if (window_h != 3 or window_w != 3) and (window_h != 2 or window_w != 2):
        raise RuntimeError("pooling window size must be [3,3] or [2,2].")

    stride_h, stride_w = strides
    if (stride_h != 2) or (stride_w != 2):
        raise RuntimeError("pooling stride size must be [2, 2].")


def _compute_col_optimization(data, max_pool_res_shape, stride, window):
    """
    cal the max in cols
    """
    if window[1] < 4:
        col_max_tensors, out_col_max = \
            _compute_3_window(data,
                             max_pool_res_shape,
                             False,
                             stride,
                             window)

    return out_col_max, col_max_tensors


def _compute_row_optimization(data, row_max_shape, stride, window):
    """
    cal the max in cols
    """
    row_max_tensors, out_row_max = \
        _compute_3_window(data,
                         row_max_shape,
                         True,
                         stride,
                         window)

    return out_row_max, row_max_tensors


def _max_pooling_padding(input_data, padding, dtype):
    """
    padding 0 at left, right, up, down
    """
    data_shape = _shape_to_list(input_data.shape)

    pad_shape = [data_shape[0],
                 data_shape[1],
                 data_shape[2] + padding[0] + padding[1],
                 data_shape[3] + padding[2] + padding[3],
                 data_shape[4]]
    h_last = data_shape[2] + padding[0]
    w_last = data_shape[3] + padding[2]
    pad_align = pad_shape
    pad_align[3] = (pad_shape[3] + C0_SIZE - 1) // C0_SIZE * C0_SIZE

    def _padding_v100():
        zero_padding = tvm.compute(pad_shape,
                                   lambda *indice:
                                   tvm.convert(0).astype(input_data.dtype),
                                   name="max_pooling_pad_init_zero",
                                   tag="max_pooling_pad_init_zero")
        pad_data = \
            tvm.compute(
                pad_align,
                lambda n, c1, h, w, c0:
                tvm.select(
                    w > padding[2] - 1,
                    tvm.select(w < w_last,
                               tvm.select(h > padding[0] - 1,
                                          tvm.select(h < h_last,
                                                     input_data(n,
                                                                c1,
                                                                h -
                                                                padding[0],
                                                                w -
                                                                padding[2],
                                                                c0),
                                                     zero_padding(n, c1,
                                                                  h, w, c0)
                                                     ),
                                          zero_padding(n, c1, h, w, c0)
                                          ),
                               zero_padding(n, c1, h, w, c0)
                               ),
                    zero_padding(n, c1, h, w, c0)
                ),
                name=NAME + "max_pooling_pad_data",
                tag=OP_TAG + "max_pooling_pad_data")

        MaxPoolParam.update_tensormap("max_pooling_pad_data", pad_data)

        return pad_data

    def _padding_v200():
        zero_value = tvm.const(0, dtype)
        pad_data = \
            tvm.compute(
                pad_align,
                lambda n, c1, h, w, c0:
                tvm.select(
                    w > padding[2] - 1,
                    tvm.select(w < w_last,
                               tvm.select(h > padding[0] - 1,
                                          tvm.select(h < h_last,
                                                     input_data(n,
                                                                c1,
                                                                h -
                                                                padding[0],
                                                                w -
                                                                padding[2],
                                                                c0),
                                                     ),
                                          ),
                               ),
                ),
                name=NAME + "max_pooling_pad_data",
                tag=OP_TAG + "max_pooling_pad_data")

        pad_top = tvm.compute(pad_align,
                              lambda *i:
                              tvm.select(i[2] < padding[0], zero_value),
                              name="max_pooling_pad_top")
        pad_bottom = tvm.compute(pad_align,
                                 lambda *i:
                                 tvm.select(i[2] >= h_last, zero_value),
                                 name="max_pooling_pad_bottom")
        pad_left = tvm.compute(pad_align,
                               lambda *i:
                               tvm.select(i[3] < padding[2], zero_value),
                               name="max_pooling_pad_left")
        pad_right = tvm.compute(pad_align,
                                lambda *i:
                                tvm.select(i[3] >= w_last, zero_value),
                                name="max_pooling_pad_right")
        pad_vn = tvm.compute(pad_align,
                             lambda *i:
                             pad_data[i] + pad_top[i] + pad_bottom[i] +
                             pad_left[i] + pad_right[i],
                             name="max_pooling_pad_vn")

        MaxPoolParam.update_tensormap("max_pooling_pad_data", pad_data)
        MaxPoolParam.update_tensormap("max_pooling_pad_top", pad_top)
        MaxPoolParam.update_tensormap("max_pooling_pad_bottom", pad_bottom)
        MaxPoolParam.update_tensormap("max_pooling_pad_left", pad_left)
        MaxPoolParam.update_tensormap("max_pooling_pad_right", pad_right)
        MaxPoolParam.update_tensormap("max_pooling_pad_vn", pad_vn)

        return pad_vn

    if _is_support_v200():
        pad_res = _padding_v200()
    else:
        pad_res = _padding_v100()

    return pad_res, pad_shape


def _get_caffe_out_size_and_pad(ceil_mode, input_5d_shape,
                               ksize, strides, padding):
    """
    get the Hout, Wout and padding of caffe
    """
    dilation = [1, 1, 1, 1]
    if ceil_mode == 0:
        out_size_h \
            = math.ceil((input_5d_shape[2] + padding[0]
                         + padding[1] - ksize[0]) / strides[0]) + 1
        out_size_w \
            = math.ceil((input_5d_shape[3] + padding[2]
                         + padding[3] - ksize[1]) / strides[1]) + 1
    else:
        out_size_h \
            = math.floor((input_5d_shape[2] + padding[0]
                          + padding[1] - ksize[0]) / strides[0]) + 1
        out_size_w \
            = math.floor((input_5d_shape[3] + padding[2]
                          + padding[3] - ksize[1]) / strides[1]) + 1
    if padding[0] != 0 or padding[2] != 0:
        if (out_size_h - 1)*strides[0] >= input_5d_shape[2] + padding[0]:
            out_size_h -= 1

        if (out_size_w - 1)*strides[1] >= input_5d_shape[3] + padding[2]:
            out_size_w -= 1

        if (out_size_h - 1)*strides[0] >= input_5d_shape[2] + padding[0]:
            raise RuntimeError("CHECK_LT((out_size_h - 1) * stride_h, in_size_h + pad_top)")
        if (out_size_w - 1)*strides[1] >= input_5d_shape[3] + padding[2]:
            raise RuntimeError("CHECK_LT((out_size_w - 1) * stride_w, in_size_w + pad_left)")
    if ceil_mode == 0:
        pad_rows = (out_size_h - 1)*strides[0] \
            + ((ksize[0] - 1)*dilation[0] + 1) - input_5d_shape[2]
        pad_bottom = pad_rows - padding[0]
        pad_cols = (out_size_w - 1)*strides[1] \
            + ((ksize[1] - 1)*dilation[1] + 1) - input_5d_shape[3]
        pad_right = pad_cols - padding[2]
    else:
        pad_bottom = padding[1]
        pad_right = padding[3]

    if pad_bottom < 0:
        pad_bottom = 0
    if pad_right < 0:
        pad_right = 0
    padding = [padding[0], pad_bottom, padding[2], pad_right]

    return out_size_h, out_size_w, padding


def _get_tensorflow_out_size_and_pad(pad_mode, input_5d_shape,
                                    ksize, strides, padding):
    """
    get the Hout, Wout and padding of TF
    """
    dilation = [1, 1, 1, 1]
    if pad_mode == "SAME":
        h_out = (input_5d_shape[2] + strides[0] - 1) // strides[0]
        w_out = (input_5d_shape[3] + strides[1] - 1) // strides[1]

        padding_rows = (h_out - 1) * strides[0] + \
            ((ksize[0] - 1) * dilation[0] + 1) - input_5d_shape[2]
        padding_cols = (w_out - 1) * strides[1] + \
            ((ksize[1] - 1) * dilation[1] + 1) - input_5d_shape[3]

        padding_top = padding_rows // 2
        padding_bottom = padding_rows - padding_top

        padding_left = padding_cols // 2
        padding_right = padding_cols - padding_left

        if padding_top < 0:
            padding_top = 0

        if padding_bottom < 0:
            padding_bottom = 0

        if padding_left < 0:
            padding_left = 0

        if padding_right < 0:
            padding_right = 0

        padding = [padding_top, padding_bottom, padding_left, padding_right]
    else:
        h_out = (input_5d_shape[2] + strides[0] - 1 - ksize[0] + 1) // strides[0]
        w_out = (input_5d_shape[3] + strides[1] - 1 - ksize[1] + 1) // strides[1]
        padding = list(padding)

    return h_out, w_out, padding


def _compute_3_window(data, data_shape, is_row, stride, window):
    """
    cal the max whne window is less 4
    """
    max_tensors = []
    if is_row:
        if window[0] == 3:
            out_max = _compute_tmp_max(stride[0], data_shape,
                                      data, True, "row_temp_max")
            max_tensors.append(out_max)
            out_max = \
                tvm.compute(data_shape,
                            lambda i2, j2, h2, w2, c2:
                            tvm.max(out_max(i2, j2, h2, w2, c2),
                                    data(i2, j2,
                                         stride[0] * h2 + window[0] - 1,
                                         w2, c2)),
                            name=NAME + "row_max",
                            tag=OP_TAG + "row_max")
        else:
            out_max = \
                tvm.compute(data_shape,
                            lambda i, j, h, w, c:
                            tvm.max(data(i, j, h * stride[0], w, c),
                                    data(i, j, h * stride[0] + 1, w, c)),
                            name=NAME + "row_max",
                            tag=OP_TAG + "row_max")
        max_tensors.append(out_max)
    else:
        if window[0] == 3:
            out_max = _compute_tmp_max(stride[1], data_shape,
                                      data, False, "col_temp_max")
            max_tensors.append(out_max)
            out_max = \
                tvm.compute(data_shape,
                            lambda i2, j2, h2, w2, c2:
                            tvm.max(out_max(i2, j2, h2, w2, c2),
                                    data(i2, j2, h2,
                                         stride[1] * w2 + window[1] - 1,
                                         c2)),
                            name=NAME + "col_max",
                            tag=OP_TAG + "col_max")
        else:
            out_max = \
              tvm.compute(data_shape,
                          lambda i, j, h, w, c:
                          tvm.max(data(i, j, h, w * stride[1], c),
                                  data(i, j, h, w * stride[1] + 1, c)),
                          name=NAME + "col_max",
                          tag=OP_TAG + "col_max")
        max_tensors.append(out_max)
    return max_tensors, out_max


def _compute_tmp_max(stride_tmp, cal_shape, data, is_row, tmp_max_name):
    """
    compute tmp data
    """
    if is_row:
        tmp_out_max = \
            tvm.compute(cal_shape,
                        lambda i, j, h, w, c:
                        tvm.max(data(i, j, h * stride_tmp, w, c),
                                data(i, j, h * stride_tmp + 1, w, c)),
                        name=tmp_max_name,
                        tag=OP_TAG + tmp_max_name)
    else:
        tmp_out_max = \
            tvm.compute(cal_shape,
                        lambda i, j, h, w, c:
                        tvm.max(data(i, j, h, w * stride_tmp, c),
                                data(i, j, h, w * stride_tmp + 1, c)),
                        name=tmp_max_name,
                        tag=OP_TAG + tmp_max_name)

    return tmp_out_max
