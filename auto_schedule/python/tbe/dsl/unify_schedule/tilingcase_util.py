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
tiling case utils
"""

from __future__ import absolute_import
import warnings
import math
import copy

N_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3
H_DIM_2D = 0
W_DIM_2D = 1
RANGE_DIM_LEN = 2
FORMAT_HW_DIM = 2
FORMAT_NCHW_DIM = 4
FORMAT_NC1HWC0_DIM = 5
DYNAMIC_FLAG = -1
UNKNOWN_FLAG = -2
UNKNOWN_SHAPE = [-2]
DIM_TO_NAME = {0: "N", 2: "H", 3: "W"}
INPUT_SIZE_DEFAULT_SHAPE = [4]
DX_OP_TYPE = ["deconvolution", "conv2d_transpose", "conv2d_backprop_input"]
_K_MIN_RANGE = 1
_K_MAX_RANGE = 4096
_K_DIM_SIZE = 5

def ceil_div(x_1, x_2):
    """
    ceil divide for inputs
    """

    if x_1 is None:
        return x_1
    if x_2 == 0:
        raise RuntimeError("conv2dbp division by zero")
    return (x_1 + x_2 - 1) // x_2


def _calc_output(x_in, k_size, pads, stride, dilation):
    if not x_in:
        return None
    return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1


def _calc_ceil(x_1, x_2):
    """
    do ceiling division

    Parameters
    ----------
    x_1: int
    x_2: int
    Returns
    -------
    result
    """
    if x_2 == 0:
        raise RuntimeError("division by zero")
    return (x_1 + x_2 - 1) // x_2


def pos_from_format(ele_format):
    """
    get value from ele_format
    """
    pos_n = ele_format.find('N')
    pos_c = ele_format.find('C')
    pos_h = ele_format.find('H')
    pos_w = ele_format.find('W')
    return {"pos_n": pos_n, "pos_c": pos_c, "pos_h": pos_h, "pos_w": pos_w}


def _get_idx_shape_from_format(obj_format, obj_shape):
    """
    get index and shape from ele_format
    """
    idx_n = obj_format.find('N')
    idx_d = obj_format.find('D')
    idx_h = obj_format.find('H')
    idx_w = obj_format.find('W')
    idx_c = obj_format.find('C')
    return [idx_n, idx_d, idx_h, idx_w, idx_c],\
           [obj_shape[idx_n], obj_shape[idx_d], obj_shape[idx_h], obj_shape[idx_w], obj_shape[idx_c]]

class Conv2dBackpropParaProcess():
    """
    class of param check and preprocess for dynamic conv2d_backprop_input
    """

    def __init__(self, paras):
        self.paras = paras
        self.groups = paras.get("groups")
        self.strides = paras.get("strides")
        self.pads = paras.get("pads")
        self.dilations = paras.get("dilations")
        self.valid_paras = {
            "nhw_min": 1,
            "hw_max": 4096,
            "valid_format": {"weights": ("NCHW", "NHWC", "HWCN"),
                             "input": ("NCHW", "NHWC"),
                             "output": ("NCHW", "NHWC")},
            "valid_dtype": ("float16", "int8", "int32")
        }
        self.op_type = "conv2d_backprop_input"
        self.filters = paras.get("filters")
        self.out_backprop = paras.get("out_backprop")
        self.data_format = paras.get("data_format")
        self.dtype = paras.get("filters").get("dtype")

    def get_input_nchw(self, in_shape, in_format, in_range=None):
        """
        get input shape and range of nchw format
        """

        pos = pos_from_format(in_format)
        pos_n = pos.get("pos_n")
        pos_c = pos.get("pos_c")
        pos_h = pos.get("pos_h")
        pos_w = pos.get("pos_w")
        in_shape = [in_shape[pos_n], in_shape[pos_c], in_shape[pos_h], in_shape[pos_w]]
        if in_range:
            if len(in_range) == FORMAT_NCHW_DIM:
                in_range = [in_range[pos_n], in_range[pos_c], in_range[pos_h], in_range[pos_w]]
            # range in NC1HWC0 format sometimes
            elif len(in_range) == FORMAT_NC1HWC0_DIM:
                in_range = [in_range[N_DIM], (in_shape[C_DIM], in_shape[C_DIM]), in_range[H_DIM], in_range[W_DIM]]
            return in_shape, [tuple(r) if r else r for r in in_range]
        return in_shape

    def get_input_range(self, w_shape, dy_range, dx_range=None):
        """
        calculate input range
        """

        def _get_lower_input(y_in, k_size, pads, stride, dilation):
            if not y_in:
                return y_in
            # dilation = 1
            return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1]

        def _get_higher_input(y_in, k_size, pads, stride, dilation):
            if not y_in:
                return y_in
            # dilation = 1
            return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1] + stride - 1

        def _get_output(x_in, k_size, pads, stride, dilation):
            if not x_in:
                return x_in
            return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1

        correct_range_flag = False
        new_dy_range = copy.deepcopy(dy_range)
        if DYNAMIC_FLAG in self.pads:
            dx_h_lower = (dy_range[H_DIM][0] - 1) * self.strides[H_DIM] + 1
            if not dy_range[H_DIM][1]:
                dx_h_upper = dy_range[H_DIM][1]
            else:
                dx_h_upper = dy_range[H_DIM][1] * self.strides[H_DIM]
                if dx_h_upper > self.valid_paras.get("hw_max"):
                    dx_h_upper = min(dx_h_upper, self.valid_paras.get("hw_max"))
                    new_dy_range[H_DIM] = (new_dy_range[H_DIM][0], _get_output(dx_h_upper, w_shape[H_DIM],
                                                                               (self.pads[0], self.pads[1]),
                                                                               self.strides[H_DIM],
                                                                               self.dilations[H_DIM]))
                    correct_range_flag = True
                    warnings.warn("The input calculated based on the upper limit of the output h " +
                                  "range is more than 4096, and the upper limit of the input h range is corrected " +
                                  "as {}".format(dx_h_upper))

            dx_w_lower = (dy_range[W_DIM][0] - 1) * self.strides[W_DIM] + 1
            if not dy_range[W_DIM][1]:
                dx_w_upper = dy_range[W_DIM][1]
            else:
                dx_w_upper = dy_range[W_DIM][1] * self.strides[W_DIM]
                if dx_w_upper > self.valid_paras.get("hw_max"):
                    dx_w_upper = min(dx_w_upper, self.valid_paras.get("hw_max"))
                    new_dy_range[W_DIM] = (new_dy_range[W_DIM][0], _get_output(dx_w_upper, w_shape[W_DIM],
                                                                               (self.pads[2], self.pads[3]),
                                                                               self.strides[W_DIM],
                                                                               self.dilations[W_DIM]))
                    correct_range_flag = True
                    warnings.warn("The input calculated based on the upper limit of the output w " +
                                  "range is more than 4096, and the upper limit of the input w range is corrected " +
                                  "as {}".format(dx_w_upper))
        else:
            dx_h_lower = _get_lower_input(dy_range[H_DIM][0], w_shape[H_DIM],
                                          (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                          self.dilations[H_DIM])
            if dx_h_lower < self.valid_paras.get("nhw_min"):
                dx_h_lower = max(dx_h_lower, self.valid_paras.get("nhw_min"))
                new_dy_range[H_DIM] = (_get_output(dx_h_lower, w_shape[H_DIM],
                                                   (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                                   self.dilations[H_DIM]), new_dy_range[H_DIM][1])
                correct_range_flag = True
                warnings.warn("The input calculated based on the lower limit of the output h " +
                              "range is less than 1, and the lower limit of the input h range is corrected " +
                              "as {}".format(dx_h_lower))
            if not dy_range[H_DIM][1]:
                dx_h_upper = dy_range[H_DIM][1]
            else:
                dx_h_upper = _get_higher_input(dy_range[H_DIM][1], w_shape[H_DIM],
                                               (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                               self.dilations[H_DIM])
                if dx_h_upper > self.valid_paras.get("hw_max"):
                    dx_h_upper = min(dx_h_upper, self.valid_paras.get("hw_max"))
                    new_dy_range[H_DIM] = (new_dy_range[H_DIM][0], _get_output(dx_h_upper, w_shape[H_DIM],
                                                                               (self.pads[0], self.pads[1]),
                                                                               self.strides[H_DIM],
                                                                               self.dilations[H_DIM]))
                    correct_range_flag = True
                    warnings.warn("The input calculated based on the upper limit of the output h " +
                                  "range is more than 4096, and the upper limit of the input h range is corrected " +
                                  "as {}".format(dx_h_upper))

            dx_w_lower = _get_lower_input(dy_range[W_DIM][0], w_shape[W_DIM],
                                          (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                          self.dilations[W_DIM])
            if dx_w_lower < self.valid_paras.get("nhw_min"):
                dx_w_lower = max(dx_w_lower, self.valid_paras.get("nhw_min"))
                new_dy_range[W_DIM] = (_get_output(dx_w_lower, w_shape[W_DIM],
                                                   (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                                   self.dilations[W_DIM]), new_dy_range[W_DIM][1])
                correct_range_flag = True
                warnings.warn("The input calculated based on the lower limit of the output w " +
                              "range is less than 1, and the lower limit of the input w range is corrected " +
                              "as {}".format(dx_w_lower))
            if not dy_range[W_DIM][1]:
                dx_w_upper = dy_range[W_DIM][1]
            else:
                dx_w_upper = _get_higher_input(dy_range[W_DIM][1], w_shape[W_DIM],
                                               (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                               self.dilations[W_DIM])
                if dx_w_upper > self.valid_paras.get("hw_max"):
                    dx_w_upper = min(dx_w_upper, self.valid_paras.get("hw_max"))
                    new_dy_range[W_DIM] = (new_dy_range[W_DIM][0], _get_output(dx_w_upper, w_shape[W_DIM],
                                                                               (self.pads[2], self.pads[3]),
                                                                               self.strides[W_DIM],
                                                                               self.dilations[W_DIM]))
                    correct_range_flag = True
                    warnings.warn("The input calculated based on the upper limit of the output w " +
                                  "range is more than 4096, and the upper limit of the input w range is corrected " +
                                  "as {}".format(dx_w_upper))
        if dx_range:
            return [dx_range[N_DIM], dx_range[C_DIM], (dx_h_lower, dx_h_upper), (dx_w_lower, dx_w_upper)]
        return [dy_range[N_DIM], (w_shape[N_DIM], w_shape[N_DIM]),
                (dx_h_lower, dx_h_upper), (dx_w_lower, dx_w_upper)], correct_range_flag, new_dy_range

    def get_output_range(self, w_shape, in_range, out_range=None):
        """
        calculate output range
        """

        def _get_output(x_in, k_size, pads, stride, dilation):
            if not x_in:
                return x_in
            return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1

        def _get_lower_input(y_in, k_size, pads, stride, dilation):
            if not y_in:
                return y_in
            # dilation = 1
            return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1]

        def _get_higher_input(y_in, k_size, pads, stride, dilation):
            # dilation = 1
            return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1] + stride - 1

        correct_range_flag = False
        new_in_range = copy.deepcopy(in_range)
        if DYNAMIC_FLAG in self.pads:
            out_h_lower = ceil_div(in_range[H_DIM][0], self.strides[H_DIM])
            out_h_upper = ceil_div(in_range[H_DIM][1], self.strides[H_DIM])
            out_w_lower = ceil_div(in_range[W_DIM][0], self.strides[W_DIM])
            out_w_upper = ceil_div(in_range[W_DIM][1], self.strides[W_DIM])
        else:
            out_h_lower = _get_output(in_range[H_DIM][0], w_shape[H_DIM],
                                      (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                      self.dilations[H_DIM])
            out_h_upper = _get_output(in_range[H_DIM][1], w_shape[H_DIM],
                                      (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                      self.dilations[H_DIM])
            out_w_lower = _get_output(in_range[W_DIM][0], w_shape[W_DIM],
                                      (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                      self.dilations[W_DIM])
            out_w_upper = _get_output(in_range[W_DIM][1], w_shape[W_DIM],
                                      (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                      self.dilations[W_DIM])
        if out_h_lower < self.valid_paras.get("nhw_min"):
            out_h_lower = max(out_h_lower, self.valid_paras.get("nhw_min"))
            new_in_range[H_DIM] = (
            _get_lower_input(out_h_lower, w_shape[H_DIM], (self.pads[0], self.pads[1]), self.strides[H_DIM],
                             self.dilations[H_DIM]), new_in_range[H_DIM][1])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input h " +
                          "range is less than 1, and the lower limit of the output h range is corrected " +
                          "as {}".format(out_h_lower))
        if out_h_upper and out_h_upper > self.valid_paras.get("hw_max"):
            out_h_upper = min(out_h_upper, self.valid_paras.get("hw_max"))
            new_in_range[H_DIM] = (
            new_in_range[H_DIM][0], _get_higher_input(out_h_upper, w_shape[H_DIM], (self.pads[0], self.pads[1]),
                                                      self.strides[H_DIM], self.dilations[H_DIM]))
            correct_range_flag = True
            warnings.warn("The output calculated based on the higher limit of the input h " +
                          "range is more than 4096, and the higher limit of the output h range is corrected " +
                          "as {}".format(out_h_upper))
        if out_w_lower < self.valid_paras.get("nhw_min"):
            out_w_lower = max(out_w_lower, self.valid_paras.get("nhw_min"))
            new_in_range[W_DIM] = (
            _get_lower_input(out_w_lower, w_shape[W_DIM], (self.pads[2], self.pads[3]), self.strides[W_DIM],
                             self.dilations[W_DIM]), new_in_range[W_DIM][1])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input w " +
                          "range is less than 1, and the lower limit of the output w range is corrected " +
                          "as {}".format(out_w_lower))
        if out_w_upper and out_w_upper > self.valid_paras.get("hw_max"):
            out_w_upper = min(out_w_upper, self.valid_paras.get("hw_max"))
            new_in_range[W_DIM] = (
            new_in_range[W_DIM][0], _get_higher_input(out_w_upper, w_shape[W_DIM], (self.pads[2], self.pads[3]),
                                                      self.strides[W_DIM], self.dilations[W_DIM]))
            correct_range_flag = True
            warnings.warn("The output calculated based on the higher limit of the input w " +
                          "range is more than 4096, and the higher limit of the output w range is corrected " +
                          "as {}".format(out_w_upper))
        if out_range:
            return [out_range[N_DIM], out_range[C_DIM], (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)]
        return [in_range[N_DIM], (w_shape[N_DIM], w_shape[N_DIM]),
                (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)], correct_range_flag, new_in_range


class Conv3dBackpropParaProcess():
    """
    class of param check and preprocess for dynamic conv3d_backprop_input
    """
    def __init__(self, para_dict, pad_mode):
        self.para_dict = para_dict
        self.pad_mode = pad_mode
        self.strides = para_dict.get("strides")
        self.pads = para_dict.get("pads")
        self.dilations = para_dict.get("dilations")
        self.groups = para_dict.get("groups")
        self.filter = para_dict.get("ori_tensors").get("filter")
        self.out_backprop = para_dict.get("ori_tensors").get("out_backprop")
        self.y = para_dict.get("ori_tensors").get("y")
        self.input_size = para_dict.get("ori_tensors").get("input_size")

    def _set_conv3dx_dim_range(self, idx, attr_param, dx_range, dy_range):
        stride, kernel, pad = attr_param[0], attr_param[1], attr_param[2]
        low, high = dy_range[idx][0], dy_range[idx][1]
        if self.pad_mode == "VAR":
            dx_range[idx][0] = stride * (low - 1) + 1
            dx_range[idx][1] = stride * high
        else:
            dx_range[idx][0] = stride * (low - 1) + kernel - pad
            dx_range[idx][1] = stride * (high - 1) + kernel - pad + stride - 1
        dx_range[idx][0] = max(dx_range[idx][0], _K_MIN_RANGE)
        if high == -1:
            dx_range[idx][1] = high
        else:
            dx_range[idx][1] = min(dx_range[idx][1], _K_MAX_RANGE)

    def get_dx_range(self, dy_range):
        """
        get dx_range according to dy_range
        """
        _, shape_filter_ndhwc = _get_idx_shape_from_format(self.filter["ori_format"],
                                                           self.filter["ori_shape"])
        idx_y_ndhwc, shape_y_ndhwc = _get_idx_shape_from_format(self.y["ori_format"],
                                                                self.y["ori_shape"])
        _, filter_d, filter_h, filter_w, filter_c = shape_filter_ndhwc
        idx_y_n, idx_y_d, idx_y_h, idx_y_w, idx_y_c = idx_y_ndhwc
        _, dx_d, dx_h, dx_w, _ = shape_y_ndhwc
        stride_d, stride_h, stride_w = \
            self.strides[idx_y_d], self.strides[idx_y_h], self.strides[idx_y_w]
        dilations_d, dilations_h, dilations_w = \
            self.dilations[idx_y_d], self.dilations[idx_y_h], self.dilations[idx_y_w]
        pad_front, pad_back, pad_up, pad_down, pad_left, pad_right = \
            self.pads[0], self.pads[1], self.pads[2], self.pads[3], self.pads[4], self.pads[5]
        kdext = (filter_d - 1) * dilations_d + 1
        khext = (filter_h - 1) * dilations_h + 1
        kwext = (filter_w - 1) * dilations_w + 1
        out_backprop_format = self.out_backprop["ori_format"]
        out_backprop_sizes = self.out_backprop["ori_shape"]
        n_dy_pos = out_backprop_format.find('N')
        dy_n = out_backprop_sizes[n_dy_pos]

        dx_range = [1, 1, 1, 1, 1]
        dx_range[idx_y_n] = [dy_n, dy_n]
        dx_range[idx_y_d] = [dx_d, dx_d]
        dx_range[idx_y_h] = [dx_h, dx_h]
        dx_range[idx_y_w] = [dx_w, dx_w]
        dx_range[idx_y_c] = [filter_c * self.groups, filter_c * self.groups]

        if len(dy_range) == _K_DIM_SIZE:
            dx_range[idx_y_n] = dy_range[n_dy_pos]
            if dx_d == -1:
                attr_param_d = [stride_d, kdext, pad_front + pad_back]
                self._set_conv3dx_dim_range(idx_y_d, attr_param_d, dx_range, dy_range)
            if dx_h == -1:
                attr_param_h = [stride_h, khext, pad_up + pad_down]
                self._set_conv3dx_dim_range(idx_y_h, attr_param_h, dx_range, dy_range)
            if dx_w == -1:
                attr_param_w = [stride_w, kwext, pad_left + pad_right]
                self._set_conv3dx_dim_range(idx_y_w, attr_param_w, dx_range, dy_range)
        return dx_range

    def get_dy_range(self, dx_range_ndhwc):
        """
        get dy_range according to dx_range
        """
        dx_range_n, dx_range_d, dx_range_h, dx_range_w, dx_range_c = dx_range_ndhwc
        _, shape_filter_ndhwc = _get_idx_shape_from_format(self.filter["ori_format"],
                                                           self.filter["ori_shape"])
        _, shape_out_backprop_ndhwc = _get_idx_shape_from_format(self.out_backprop["ori_format"],
                                                                 self.out_backprop["ori_shape"])
        _, filter_d, filter_h, filter_w, _ = shape_filter_ndhwc
        if not all(i == 0 for i in self.pads):
            out_d_upper, out_h_upper, out_w_upper = None, None, None
            out_d_lower = _calc_ceil(dx_range_d[0], self.strides[1])
            if dx_range_d[1]:
                out_d_upper = _calc_ceil(dx_range_d[1], self.strides[1])

            out_h_lower = _calc_ceil(dx_range_h[0], self.strides[2])
            if dx_range_h[1]:
                out_h_upper = _calc_ceil(dx_range_h[1], self.strides[2])

            out_w_lower = _calc_ceil(dx_range_w[0], self.strides[3])
            if dx_range_w[1]:
                out_w_upper = _calc_ceil(dx_range_w[1], self.strides[3])
        else:
            out_d_lower = _calc_output(dx_range_d[0],
                                       filter_d,
                                       (self.pads[0], self.pads[1]),
                                       self.strides[1],
                                       self.dilations[1])
            if out_d_lower < 1:
                fmap_range_d_lower = min(filter_d, dx_range_d[1]) if dx_range_d[1] else filter_d
                fmap_range_d = (fmap_range_d_lower, dx_range_d[1])
                out_d_lower = _calc_output(fmap_range_d[0],
                                           filter_d,
                                           (self.pads[0], self.pads[1]),
                                           self.strides[1],
                                           self.dilations[1])
            out_d_upper = _calc_output(dx_range_d[1],
                                       filter_d,
                                       (self.pads[0], self.pads[1]),
                                       self.strides[1],
                                       self.dilations[1])

            out_h_lower = _calc_output(dx_range_h[0],
                                       filter_h,
                                       (self.pads[2], self.pads[3]),
                                       self.strides[2],
                                       self.dilations[2])
            if out_h_lower < 1:
                fmap_range_h_lower = min(filter_h, dx_range_h[1]) if dx_range_h[1] else filter_h
                fmap_range_h = (fmap_range_h_lower, dx_range_h[1])
                out_h_lower = _calc_output(fmap_range_h[0],
                                           filter_h,
                                           (self.pads[2], self.pads[3]),
                                           self.strides[2],
                                           self.dilations[2])
            out_h_upper = _calc_output(dx_range_h[1],
                                       filter_h,
                                       (self.pads[2], self.pads[3]),
                                       self.strides[2],
                                       self.dilations[2])

            out_w_lower = _calc_output(dx_range_w[0],
                                       filter_w,
                                       (self.pads[4], self.pads[5]),
                                       self.strides[3],
                                       self.dilations[3])
            if out_w_lower < 1:
                fmap_range_w_lower = min(filter_w, dx_range_w[1]) if dx_range_w[1] else filter_w
                fmap_range_w = (fmap_range_w_lower, dx_range_w[1])
                out_w_lower = _calc_output(fmap_range_w[0],
                                           filter_w,
                                           (self.pads[4], self.pads[5]),
                                           self.strides[3],
                                           self.dilations[3])
            out_w_upper = _calc_output(dx_range_w[1],
                                       filter_w,
                                       (self.pads[4], self.pads[5]),
                                       self.strides[3],
                                       self.dilations[3])

        dy_range = [(dx_range_n[0], dx_range_n[1]),
                    (out_d_lower, out_d_upper),
                    (out_h_lower, out_h_upper),
                    (out_w_lower, out_w_upper),
                    (shape_out_backprop_ndhwc[-1], shape_out_backprop_ndhwc[-1])
                    ]
        return dy_range
