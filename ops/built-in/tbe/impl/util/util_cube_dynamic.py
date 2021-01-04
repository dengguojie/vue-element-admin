# Copyright 2020 Huawei Technologies Co., Ltd
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
public function for cube dynamic
"""

from __future__ import absolute_import
import math

import te
from te.tvm import api as tvm
import te.platform as tbe_platform
from te.utils import para_check
from te.utils.error_manager import error_manager_cube as err_man
import te.lang.base as tbe_base

N_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3
FORMAT_NCHW_DIM = 4
FORMAT_NC1HWC0_DIM = 5
DYNAMIC_FLAG = -1


def ceil_div(x_1, x_2):
    """
    ceil divide for inputs
    """

    if x_2 == 0:
        err_man.raise_err_specific("conv2d", "division by zero")
    return (x_1 + x_2 - 1) // x_2


def align(x_1, x_2):
    """
    align up for inputs
    """

    return ceil_div(x_1, x_2) * x_2


def lcm(x_1, x_2):
    """
    get the least common multiple
    """

    return (x_1 * x_2) // math.gcd(x_1, x_2)


def pos_from_format(ele_format):
    """
    get value from ele_format
    """
    pos_n = ele_format.find('N')
    pos_c = ele_format.find('C')
    pos_h = ele_format.find('H')
    pos_w = ele_format.find('W')
    return pos_n, pos_c, pos_h, pos_w


def get_input_nchw(in_shape, in_format, in_range=()):
    """
    get input shape and range of nchw format
    """

    pos_n, pos_c, pos_h, pos_w = pos_from_format(in_format)
    in_shape = [in_shape[pos_n], in_shape[pos_c], in_shape[pos_h], in_shape[pos_w]]

    if len(in_range) == FORMAT_NCHW_DIM:
        in_range = [in_range[pos_n], in_range[pos_c], in_range[pos_h], in_range[pos_w]]
    # range in NC1HWC0 format sometimes
    elif len(in_range) == FORMAT_NC1HWC0_DIM:
        in_range = [in_range[N_DIM],
                    (in_shape[C_DIM], in_shape[C_DIM]), in_range[H_DIM], in_range[W_DIM]]

    if in_range:
        return in_shape, [tuple(r) for r in in_range]
    return in_shape


class CubeParaProcess:
    """
    class of param check and preprocess for dynamic cube ops
    """

    def __init__(self, paras):
        self.paras = paras
        self.groups = paras.get("groups")
        self.strides = paras.get("strides")
        self.pads = paras.get("pads")
        self.dilations = paras.get("dilations")
        self.op_type = None
        self.valid_paras = {
            "hw_min": 1,
            "hw_max": 4096,
            "valid_format": {"weights": ("NCHW", "NHWC", "HWCN"),
                             "input": ("NCHW", "NHWC"),
                             "output": ("NCHW", "NHWC")},
            "valid_dtype": ("float16",)
        }

    def check_range_valid(self, shape, dyn_range, name):
        """
        check if the range is valid
        """

        for _, dim in enumerate(zip(shape, dyn_range)):
            if dim[0] == DYNAMIC_FLAG and not dim[1]:
                err_man.raise_err_specific_user(
                    self.op_type, "must specify range when shape is -1")

        h_range, w_range = dyn_range[H_DIM], dyn_range[W_DIM]
        if h_range[0] < self.valid_paras.get("hw_min") or h_range[1] > self.valid_paras.get("hw_max"):
            err_man.raise_err_attr_range_invalid(
                self.op_type, [self.valid_paras.get("hw_min"), self.valid_paras.get("hw_max")], name, h_range[0])
        if w_range[0] < self.valid_paras.get("hw_min") or w_range[1] > self.valid_paras.get("hw_max"):
            err_man.raise_err_attr_range_invalid(
                self.op_type, [self.valid_paras.get("hw_min"), self.valid_paras.get("hw_max")], name, w_range[0])

    def check_para_dim(self, seq, seq_name):
        """
        check if the sequence is four-dimensional
        """

        if len(seq) != 4:
            err_man.raise_err_should_be_4d(self.op_type, seq_name)

    def check_format(self, param_format, param_name):
        """
        check if the format is valid
        """

        expect_formats = self.valid_paras.get("valid_format").get(param_name)
        if param_format not in expect_formats:
            err_man.raise_err_input_format_invalid(
                self.op_type, param_name, expect_formats, param_format)

    def check_input_dict(self, para, para_name, need_range):
        """
        check if the input dict is valid
        """

        if not isinstance(para, dict):
            err_man.raise_err_check_type(self.op_type, para_name, dict, type(para))
        if not para.get("ori_shape"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass ori_shape in {para_name}")
        if not para.get("dtype"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass dtype in {para_name}")
        if not para.get("ori_format"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass ori_format in {para_name}")
        if need_range and not para.get("range"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass range in {para_name}")

    def get_attr_nchw(self, in_format):
        """
        get the input shape of nchw format
        """

        pos_n, pos_c, pos_h, pos_w = pos_from_format(in_format)
        self.dilations = [self.dilations[pos_n], self.dilations[pos_c],
                          self.dilations[pos_h], self.dilations[pos_w]]
        self.strides = [self.strides[pos_n], self.strides[pos_c],
                        self.strides[pos_h], self.strides[pos_w]]

    def round_channel(self, in_shape, w_shape, dtype):
        """
        round up the channel dimension
        """

        if (self.op_type == "conv2d" and in_shape[C_DIM] != w_shape[C_DIM]
                or self.op_type == "conv2d_backprop_input" and in_shape[C_DIM] != w_shape[N_DIM]):
            err_man.raise_err_scene_equal_limitation(self.op_type, "input feature map channel", "filter channel")

        block_size_k, block_size_n = tbe_platform.CUBE_MKN[dtype]['mac'][1:3]
        in_shape[C_DIM] = align(in_shape[C_DIM], block_size_k)
        w_shape[N_DIM] = align(w_shape[N_DIM], block_size_n)
        w_shape[C_DIM] = align(w_shape[C_DIM], block_size_k)

        return in_shape, w_shape

    def set_group_para(self, in_shape, w_shape, w_dtype):
        """
        calculate paras for group
        """

        block_size_k, block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][1:3]
        cin_ori = in_shape[C_DIM] // self.groups
        cout_ori = w_shape[N_DIM] // self.groups
        cin_lcm = lcm(cin_ori, block_size_k) // cin_ori
        cout_lcm = lcm(cout_ori, block_size_n) // cout_ori
        enlarge = min(lcm(cin_lcm, cout_lcm), self.groups)
        c1_opt = math.ceil(cin_ori * enlarge / block_size_k)
        cout1_opt = math.ceil(cout_ori * enlarge / block_size_n)
        group_opt = math.ceil(self.groups / enlarge)

        return {"enlarge": enlarge, "c1_opt": c1_opt, "cout1_opt": cout1_opt, "group_opt": group_opt}


class Conv2dParaProcess(CubeParaProcess):
    """
    class of param check and preprocess for dynamic conv2d
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "conv2d"
        self.inputs = paras.get("inputs")
        self.weights = paras.get("weights")
        self.bias = paras.get("bias")
        self.data_format = paras.get("data_format")
        self.dtype = paras.get("inputs").get("dtype")

    def check_support_valid(self, in_shape):
        """
        check whether dynamic shape is supported for conv2d
        """

        soc_version = tbe_platform.get_soc_spec("SOC_VERSION")
        if soc_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
            err_man.raise_err_specific_user(
                self.op_type, "Hi3796CV300ES and Hi3796CV300CS don't support dynamic shape")
        if self.groups != 1:
            err_man.raise_err_specific_user(
                self.op_type, "group > 1 is not supported yet in dynamic")
        if self.paras.get("offset_w"):
            err_man.raise_err_specific_user(
                self.op_type, "offset_w is not supported in dynamic shape yet.")
        if in_shape[C_DIM] == -1:
            err_man.raise_err_specific_user(
                self.op_type, "dynamic c dimension is not supported yet.")
        if self.dilations[H_DIM] != 1 or self.dilations[W_DIM] != 1:
            err_man.raise_err_specific_user(
                self.op_type, "dilations is not supported in dynamic shape yet.")

    def get_y_range(self, w_shape, in_range):
        """
        calculate output range
        """

        def _get_output(x_in, k_size, pads, stride, dilation):
            return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1

        if DYNAMIC_FLAG in self.pads:
            y_h_lower = ceil_div(in_range[H_DIM][0], self.strides[H_DIM])
            y_h_upper = ceil_div(in_range[H_DIM][1], self.strides[H_DIM])
            y_w_lower = ceil_div(in_range[W_DIM][0], self.strides[W_DIM])
            y_w_upper = ceil_div(in_range[W_DIM][1], self.strides[W_DIM])
        else:
            y_h_lower = _get_output(in_range[H_DIM][0], w_shape[H_DIM],
                                    (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                    self.dilations[H_DIM])
            y_h_upper = _get_output(in_range[H_DIM][1], w_shape[H_DIM],
                                    (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                    self.dilations[H_DIM])
            y_w_lower = _get_output(in_range[W_DIM][0], w_shape[W_DIM],
                                    (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                    self.dilations[W_DIM])
            y_w_upper = _get_output(in_range[W_DIM][1], w_shape[W_DIM],
                                    (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                    self.dilations[W_DIM])
        return [in_range[N_DIM], (w_shape[N_DIM], w_shape[N_DIM]),
                (y_h_lower, y_h_upper), (y_w_lower, y_w_upper)]

    def calc_shape(self, in_shape, w_shape, in_range, y_range):
        """
        calculate shape for mmad
        """

        in_shape, w_shape = self.round_channel(in_shape, w_shape, self.dtype)
        block_size_k, block_size_n = tbe_platform.CUBE_MKN[self.dtype]['mac'][1:3]

        in_shape_nc1hwc0 = [in_shape[N_DIM], in_shape[C_DIM] // block_size_k,
                            in_shape[H_DIM], in_shape[W_DIM], block_size_k]
        if in_shape_nc1hwc0[N_DIM] == DYNAMIC_FLAG:
            in_shape_nc1hwc0[N_DIM] = tbe_base.var("batch_n", in_range[N_DIM])
            tbe_base.add_exclude_bound_var(in_shape_nc1hwc0[N_DIM])
        if in_shape_nc1hwc0[H_DIM] == DYNAMIC_FLAG:
            in_shape_nc1hwc0[H_DIM] = tbe_base.var("fmap_h", in_range[H_DIM])
            tbe_base.add_exclude_bound_var(in_shape_nc1hwc0[H_DIM])
            tbe_base.add_exclude_bound_var(tbe_base.var("ho", y_range[H_DIM]))
        if in_shape_nc1hwc0[W_DIM] == DYNAMIC_FLAG:
            in_shape_nc1hwc0[W_DIM] = tbe_base.var("fmap_w", in_range[W_DIM])
            tbe_base.add_exclude_bound_var(in_shape_nc1hwc0[W_DIM])
            tbe_base.add_exclude_bound_var(tbe_base.var("wo", y_range[W_DIM]))

        if self.paras.get("optim_dict").get("c0_optim_flg"):
            w_shape_frac_z = (ceil_div(4 * w_shape[H_DIM] * w_shape[W_DIM], block_size_k),
                              w_shape[N_DIM] // block_size_n, block_size_n, block_size_k)
        else:
            w_shape_frac_z = (w_shape[C_DIM] * w_shape[H_DIM] * w_shape[W_DIM] // block_size_k,
                              w_shape[N_DIM] // block_size_n, block_size_n, block_size_k)
        return in_shape, w_shape, in_shape_nc1hwc0, w_shape_frac_z

    def calc_pads(self, in_shape_nc1hwc0, w_shape):
        """
        calculate pads
        """

        pads = self.pads
        if DYNAMIC_FLAG in self.pads:
            # if load2d, return [0,0,0,0]
            if w_shape[H_DIM] * w_shape[W_DIM] == 1 and self.strides[H_DIM] * self.strides[W_DIM] == 1:
                pads = [0, 0, 0, 0]
            else:
                filter_h_dilation = (w_shape[H_DIM] - 1) * self.dilations[H_DIM] + 1
                filter_w_dilation = (w_shape[W_DIM] - 1) * self.dilations[W_DIM] + 1
                pad_h = (align(in_shape_nc1hwc0[H_DIM], self.strides[H_DIM]) -
                         self.strides[H_DIM] + filter_h_dilation - in_shape_nc1hwc0[H_DIM])
                pad_h = tvm.max(pad_h, 0)
                pad_up = pad_h // 2
                pad_down = pad_h - pad_up
                pad_w = (align(in_shape_nc1hwc0[W_DIM], self.strides[W_DIM]) -
                         self.strides[W_DIM] + filter_w_dilation - in_shape_nc1hwc0[W_DIM])
                pad_w = tvm.max(pad_w, 0)
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pads = pad_up, pad_down, pad_left, pad_right
                pads = list(map(lambda x: int(x) if (isinstance(x, te.tvm.expr.IntImm)) else x, pads))
        self.pads = pads

    def check_paras(self):
        """
        check original paras
        """

        self.check_input_dict(self.inputs, "inputs", True)
        self.check_input_dict(self.weights, "weights", False)
        in_shape = list(self.inputs.get("ori_shape"))
        in_range = self.inputs.get("range")
        in_format = self.inputs.get("ori_format")
        w_shape = list(self.weights.get("ori_shape"))
        w_dtype = self.weights.get("dtype")
        w_format = self.weights.get("ori_format")

        para_check.check_dtype_rule(self.dtype, self.valid_paras.get("valid_dtype"))
        para_check.check_dtype_rule(w_dtype, self.valid_paras.get("valid_dtype"))
        para_check.check_dtype_rule(self.paras.get("outputs").get("dtype"), self.valid_paras.get("valid_dtype"))
        self.check_para_dim(in_shape, "in_shape")
        self.check_para_dim(w_shape, "weights")
        self.check_para_dim(self.strides, "strides")
        self.check_para_dim(self.dilations, "dilations")
        self.check_para_dim(self.pads, "pads")
        self.check_format(self.data_format, "input")
        self.check_format(w_format, "weights")
        if self.dtype != w_dtype:
            err_man.raise_err_specific_user("conv2d", "in_dtype != w_dtype")
        if in_format != self.data_format:
            err_man.raise_err_specific_user("conv2d", "in_format != data_format")
        para_check.check_kernel_name(self.paras.get("kernel_name"))
        self.check_range_valid(in_shape, in_range, "fmap")

        return {"in_shape": in_shape, "in_range": in_range, "w_shape": w_shape, "w_format": w_format}

    def config_paras(self):
        """
        config paras and placeholders
        """

        ori_paras = self.check_paras()
        self.get_attr_nchw(self.data_format)
        in_shape, in_range = get_input_nchw(ori_paras.get("in_shape"), self.data_format, ori_paras.get("in_range"))
        w_shape = get_input_nchw(ori_paras.get("w_shape"), ori_paras.get("w_format"))

        y_range = self.get_y_range(w_shape, in_range)
        in_shape, w_shape, in_shape_nc1hwc0, w_shape_frac_z = self.calc_shape(
            in_shape, w_shape, in_range, y_range)
        self.calc_pads(in_shape_nc1hwc0, w_shape)
        self.check_support_valid(in_shape)
        group_para = self.set_group_para(in_shape, w_shape, self.dtype)

        input_tensor = tvm.placeholder(in_shape_nc1hwc0, name="Fmap", dtype=self.dtype)
        weight_tensor = tvm.placeholder(w_shape_frac_z, name="Filter", dtype=self.dtype)
        if self.bias:
            bias_tensor = tvm.placeholder((w_shape[N_DIM],), name="bias_tensor", dtype=self.dtype)
        else:
            bias_tensor = None

        return {"input_tensor": input_tensor, "weight_tensor": weight_tensor, "bias_tensor": bias_tensor,
                "w_shape": w_shape, "in_shape_nc1hwc0": in_shape_nc1hwc0, "w_shape_frac_z": w_shape_frac_z,
                "group_para": group_para}
